"""
Multi-Turn KernelBench Environment (Kevin Mode).

This module implements a multi-turn RL environment for KernelBench,
inspired by Cognition's Kevin-32B approach:
  - Multiple refinement turns per problem (default T=4)
  - Each turn: model sees problem + RA-ICL + condensed history + feedback
  - Per-step scores combined with discounted returns for RL

The environment tracks:
  - Step-by-step evaluation results
  - Condensed history of previous attempts
  - Accumulated scores for discounted return computation
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree

from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    ParsedResponse,
    evaluate_kernel,
    evaluate_kernel_async,
    extract_code_block,
    parse_structured_response,
    get_problem_ids,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernelbench_tinker.envs.kernelbench_client import KernelEvalResult as KernelEvalResultType
from kernelbench_tinker.training.reward import compute_reward, compute_reward_breakdown, RewardConfig
from kernelbench_tinker.training.trace_logger import get_trace_logger

logger = logging.getLogger(__name__)


# Maximum characters to include from previous kernel in history
MAX_KERNEL_HISTORY_LEN = 2000

# Maximum error message length for feedback
MAX_ERROR_LEN = 800

# Maximum lines of error context to include
MAX_ERROR_LINES = 10


def _extract_key_error(error_message: str | None, include_context: bool = True) -> str:
    """
    Extract a cleaned summary of the main error with relevant context.

    Strips stack traces and boilerplate, keeping the essential error info.
    For compiler errors, includes the relevant line if available.

    Args:
        error_message: Full error message string
        include_context: Whether to include surrounding error context

    Returns:
        Cleaned error summary
    """
    if not error_message:
        return ""

    # Common patterns to extract (with context where helpful)
    patterns = [
        # Python exceptions with full message
        r"(\w+Error: .+?)(?:\n\n|\n(?=[A-Z])|$)",
        r"(\w+Exception: .+?)(?:\n\n|\n(?=[A-Z])|$)",

        # Triton specific errors - include API hints
        r"(AttributeError: module 'triton\.language' has no attribute '\w+')",
        r"(triton\.compiler\.errors\.\w+: .+?)(?:\n|$)",
        r"(triton\.runtime\.errors\.\w+: .+?)(?:\n|$)",

        # CUDA compilation errors - include the offending line
        r"(error: .+?\n.*?(?:error:|note:)?.+?)(?:\n\n|$)",
        r"(nvcc.+?error.+?)(?:\n|$)",

        # CUDA runtime errors
        r"(CUDA error: .+?)(?:\n|$)",
        r"(RuntimeError: CUDA.+?)(?:\n|$)",

        # Torch/Python shape/type errors (common in kernels)
        r"(RuntimeError: .+?shape.+?)(?:\n|$)",
        r"(RuntimeError: .+?size.+?)(?:\n|$)",
        r"(RuntimeError: .+?dtype.+?)(?:\n|$)",
        r"(TypeError: .+?)(?:\n|$)",
        r"(ValueError: .+?)(?:\n|$)",

        # Assertion errors with message
        r"(AssertionError: .+?)(?:\n|$)",

        # Correctness/test failures
        r"(Correctness check failed.+?)(?:\n|$)",
        r"(Output mismatch.+?)(?:\n|$)",

        # Generic compilation errors
        r"(compilation failed.+?)(?:\n|$)",
        r"(CompilationError: .+?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE | re.DOTALL)
        if match:
            error_text = match.group(1).strip()
            # Clean up excessive whitespace
            error_text = re.sub(r'\n\s*\n', '\n', error_text)
            return error_text[:MAX_ERROR_LEN]

    # If no pattern matched, try to extract meaningful lines
    # Skip traceback header and file lines, keep error descriptions
    meaningful_lines = []
    for line in error_message.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip traceback boilerplate
        if line.startswith('Traceback'):
            continue
        if line.startswith('File "') and ', line ' in line:
            continue
        if line.startswith('During handling of'):
            continue
        # Keep the line
        meaningful_lines.append(line)
        if len(meaningful_lines) >= MAX_ERROR_LINES:
            break

    if meaningful_lines:
        return '\n'.join(meaningful_lines)[:MAX_ERROR_LEN]

    return error_message[:MAX_ERROR_LEN]


def _categorize_error(eval_result: "KernelEvalResult") -> str:
    """
    Categorize the type of error for better feedback.

    Returns one of:
    - "format_error": Code extraction failed
    - "compilation_error": Compiler/build failed
    - "runtime_error": Crashed during execution
    - "correctness_error": Ran but gave wrong results
    - "performance_warning": Correct but slow
    - "success": No error
    """
    if not eval_result["format_ok"]:
        return "format_error"
    if not eval_result["compiled"]:
        return "compilation_error"
    if not eval_result["correctness"]:
        # Check if it's a runtime crash vs wrong output
        error_msg = eval_result.get("error_message", "") or ""
        if "Error" in error_msg or "Exception" in error_msg:
            return "runtime_error"
        return "correctness_error"
    # Correct - check performance
    if eval_result.get("speedup") is not None and eval_result["speedup"] < 1.0:
        return "performance_warning"
    return "success"


def _get_error_guidance(error_category: str, backend: str) -> str:
    """
    Provide category-specific guidance for fixing errors.
    """
    guidance = {
        "format_error": (
            "Ensure your output is a valid Python class named `ModelNew` "
            "wrapped in a code block."
        ),
        "compilation_error": (
            f"Fix the {backend.upper()} syntax/API errors. Check that all "
            f"kernel functions are correctly decorated and all imports are valid."
        ),
        "runtime_error": (
            "Fix the runtime error. Common issues: shape mismatches, "
            "incorrect tensor indexing, or invalid kernel launch configurations."
        ),
        "correctness_error": (
            "The kernel runs but produces incorrect output. Check your algorithm "
            "implementation, boundary conditions, and reduction operations."
        ),
        "performance_warning": (
            "The kernel is correct but slower than PyTorch. Consider optimizing "
            "memory access patterns, tiling, or parallelization strategy."
        ),
    }
    return guidance.get(error_category, "")


def _truncate_kernel(kernel_code: str, max_len: int = MAX_KERNEL_HISTORY_LEN) -> str:
    """Truncate kernel code for history, keeping beginning and signature."""
    if len(kernel_code) <= max_len:
        return kernel_code

    # Try to keep the class signature and forward method
    truncated = kernel_code[:max_len]
    return truncated + "\n# ... (truncated)"


@dataclass
class MultiTurnState:
    """State for multi-turn kernel refinement."""

    level: int
    problem_id: int
    backend: str  # 'triton' or 'cuda'
    turn_idx: int
    max_turns: int
    prompt_base: str           # static part: problem + PyTorch ref + instructions
    ra_icl_snippet: str        # static RA-ICL examples for this problem
    history: list[dict]        # list of {kernel, eval_result, score} dicts
    last_kernel: str | None
    last_eval: KernelEvalResult | None
    step_scores: list[float]   # scores for each completed step
    done: bool
    success: bool              # True if solved with correct + speedup threshold


# Multi-turn prompt template (structured format)
MULTITURN_SYSTEM_PROMPT_NO_THINK = """You are an expert GPU kernel developer. Your task is to optimize PyTorch operations by writing efficient custom {backend} kernels.

When given a PyTorch model and optimization examples, write an optimized kernel implementation. If a previous attempt failed, fix the errors based on the feedback provided.

Your solution must:
- Be a drop-in replacement as a class named `ModelNew`
- Use custom {backend} kernels, not just PyTorch operations
- Be correct and produce the same results as the reference

You MUST respond in exactly this format:

<KERNEL>
```python
# Your complete optimized implementation here
class ModelNew(nn.Module):
    ...
```
</KERNEL>"""


REFINEMENT_TEMPLATE = """
## Previous Attempt (Turn {turn})

```python
{previous_kernel}
```

## Evaluation Feedback
- **Status**: {error_category}
- **Compiled**: {compiled}
- **Tests Passed**: {tests_passed}/{tests_total}
{speedup_line}

{error_section}

## Instructions

{guidance}

Keep what works. Do not change the function signature unless necessary. Do not use PyTorch APIs for the core computation.
"""
ERROR_SECTION_TEMPLATE = """### Error Details
```
{error_text}
```
"""


class MultiTurnKernelBenchEnv(Env):
    """
    Multi-turn RL environment for KernelBench (Kevin mode).

    Each episode consists of T refinement steps:
    1. Turn 0: Problem + RA-ICL examples ??? model generates first kernel
    2. Turn 1+: Problem + RA-ICL + previous attempts + feedback ??? model refines

    Episode ends when:
    - max_turns reached
    - Kernel is fully correct (optionally with speedup threshold)
    """

    def __init__(
        self,
        problem: KernelBenchProblem,
        renderer: renderers.Renderer,
        max_turns: int = 4,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        early_stop_on_correct: bool = True,
        speedup_threshold: float | None = None,  # e.g. 1.0 for any speedup
        use_modal: bool = True,
        modal_timeout: float = 120.0,
    ):
        """
        Initialize multi-turn KernelBench environment.

        Args:
            problem: The KernelBench problem to solve
            renderer: Tinker renderer for formatting messages
            max_turns: Maximum refinement turns (T)
            reward_config: Configuration for reward computation
            system_prompt: Optional custom system prompt
            num_correct_trials: Number of correctness trials for evaluation
            measure_performance: Whether to measure kernel runtime
            early_stop_on_correct: Stop early if kernel is fully correct
            speedup_threshold: Required speedup to trigger early stop (None = any correct)
            use_modal: Whether to use Modal for isolated GPU evaluation
            modal_timeout: Timeout in seconds for Modal evaluation
        """
        self.problem = problem
        self.renderer = renderer
        self.max_turns = max_turns
        self.reward_config = reward_config or RewardConfig()
        self.num_correct_trials = num_correct_trials
        self.measure_performance = measure_performance
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold
        self.use_modal = use_modal
        self.modal_timeout = modal_timeout
        self._current_prompt_messages: list[renderers.Message] | None = None
        self._current_observation: tinker.ModelInput | None = None

        # Build system prompt
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = MULTITURN_SYSTEM_PROMPT_NO_THINK.format(
                backend=problem.backend.upper()
            )

        # State
        self._state: MultiTurnState | None = None

    @property
    def stop_condition(self) -> StopCondition:
        """Get stop sequences for generation."""
        return self.renderer.get_stop_sequences()

    @property
    def state(self) -> MultiTurnState:
        """Get current state (raises if not initialized)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call initial_observation first.")
        return self._state

    def _build_ra_icl_snippet(self) -> str:
        """Build the RA-ICL examples snippet (cached per problem)."""
        # Use the problem's prompt which already includes RA-ICL if configured
        return self.problem.prompt

    def _build_initial_messages(self) -> list[renderers.Message]:
        """Build messages for turn 0."""
        messages: list[renderers.Message] = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # User message is the RA-ICL prompt (or basic prompt)
        messages.append({"role": "user", "content": self.state.ra_icl_snippet})

        return messages

    def _build_refinement_messages(self) -> list[renderers.Message]:
        """Build messages for turn > 0 with history and feedback."""
        messages: list[renderers.Message] = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Start with RA-ICL examples (or basic prompt)
        user_content_parts = [self.state.ra_icl_snippet]

        # Add refinement feedback for the last attempt
        if self.state.last_eval is not None and self.state.last_kernel is not None:
            eval_result = self.state.last_eval

            # Categorize the error
            error_category = _categorize_error(eval_result)
            error_category_display = {
                "format_error": "FORMAT ERROR - Invalid code structure",
                "compilation_error": "COMPILATION ERROR - Build failed",
                "runtime_error": "RUNTIME ERROR - Crashed during execution",
                "correctness_error": "CORRECTNESS ERROR - Wrong output",
                "performance_warning": "CORRECT (but slower than baseline)",
                "success": "SUCCESS",
            }.get(error_category, error_category.upper())

            # Build speedup line if available
            speedup_line = ""
            if eval_result.get("speedup") is not None:
                speedup_line = f"- **Speedup**: {eval_result['speedup']:.2f}x"

            # Build error section with detailed error message
            error_section = ""
            if eval_result.get("error_message") and error_category != "success":
                error_text = _extract_key_error(eval_result["error_message"])
                if error_text:
                    error_section = ERROR_SECTION_TEMPLATE.format(error_text=error_text)

            # Get guidance for this error type
            guidance = _get_error_guidance(error_category, self.state.backend)
            if not guidance:
                guidance = "Fix the issues in the previous attempt and try again."

            refinement_text = REFINEMENT_TEMPLATE.format(
                turn=self.state.turn_idx,
                previous_kernel=_truncate_kernel(self.state.last_kernel),
                error_category=error_category_display,
                compiled="Yes" if eval_result["compiled"] else "No",
                tests_passed=eval_result["tests_passed"],
                tests_total=eval_result["tests_total"],
                speedup_line=speedup_line,
                error_section=error_section,
                guidance=guidance,
            )
            refinement_text += "\nRemember: respond using <KERNEL>...</KERNEL>."

            user_content_parts.append(refinement_text)

        messages.append({"role": "user", "content": "\n".join(user_content_parts)})

        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Get the initial observation for turn 0.
        """
        # Initialize state
        self._state = MultiTurnState(
            level=self.problem.level,
            problem_id=self.problem.problem_id,
            backend=self.problem.backend,
            turn_idx=0,
            max_turns=self.max_turns,
            prompt_base=self.problem.ref_code,
            ra_icl_snippet=self._build_ra_icl_snippet(),
            history=[],
            last_kernel=None,
            last_eval=None,
            step_scores=[],
            done=False,
            success=False,
        )

        messages = self._build_initial_messages()
        observation = self.renderer.build_generation_prompt(messages)
        self._current_prompt_messages = messages
        self._current_observation = observation
        return observation, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Process the model's action (generated kernel code).

        Returns StepResult with per-step reward and whether episode is done.
        """
        step_start = time.perf_counter()
        state = self.state

        # Parse the response
        message, parse_success = self.renderer.parse_response(action)
        response_text = message.get("content", "")

        # Parse structured response (extracts <KERNEL> block)
        parsed = parse_structured_response(response_text)
        kernel_code = parsed.kernel
        state.last_kernel = kernel_code

        # Check format validity
        format_ok = parsed.format_ok

        # Evaluate the kernel (using Modal for isolation if enabled)
        eval_start = time.perf_counter()
        if self.use_modal:
            eval_result = await evaluate_kernel_async(
                level=state.level,
                problem_id=state.problem_id,
                backend=state.backend,
                kernel_code=kernel_code,
                dataset_src=self.problem.dataset_src,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                timeout=self.modal_timeout,
            )
        else:
            # Local evaluation (no isolation - use with caution)
            eval_result = evaluate_kernel(
                level=state.level,
                problem_id=state.problem_id,
                backend=state.backend,
                kernel_code=kernel_code,
                dataset_src=self.problem.dataset_src,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
            )
        state.last_eval = eval_result
        eval_time = time.perf_counter() - eval_start

        # Compute per-step score
        step_score = compute_reward(eval_result, self.reward_config, thought_length=0)
        state.step_scores.append(step_score)

        # Store in history
        state.history.append({
            "turn": state.turn_idx,
            "kernel": kernel_code,
            "eval_result": eval_result,
            "score": step_score,
        })

        # Log the attempt
        logtree.log_text(f"Multi-turn: Level {state.level}, ID {state.problem_id}, Turn {state.turn_idx}")
        logtree.log_text(f"Format OK: {'Yes' if format_ok else 'No'}")
        logtree.log_text(f"Compiled: {'Yes' if eval_result['compiled'] else 'No'}")
        logtree.log_text(f"Correctness: {eval_result['tests_passed']}/{eval_result['tests_total']}")
        if eval_result.get("speedup"):
            logtree.log_text(f"Speedup: {eval_result['speedup']:.2f}x")
        logtree.log_text(f"Step score: {step_score:.3f}")
        if eval_result.get("error_message"):
            logtree.log_text(f"Error: {_extract_key_error(eval_result['error_message'])}")

        # Check if we should stop early
        is_correct = eval_result["correctness"]
        meets_speedup = (
            self.speedup_threshold is None or
            (eval_result.get("speedup") or 0) >= self.speedup_threshold
        )

        if self.early_stop_on_correct and is_correct and meets_speedup:
            state.done = True
            state.success = True

        # Check if max turns reached
        state.turn_idx += 1
        if state.turn_idx >= state.max_turns:
            state.done = True

        # Build metrics
        metrics: Metrics = {
            "level": state.level,
            "problem_id": state.problem_id,
            "turn": state.turn_idx - 1,  # The turn we just completed
            "format_ok": float(format_ok),
            "compiled": float(eval_result["compiled"]),
            "correctness": float(eval_result["correctness"]),
            "tests_passed": eval_result["tests_passed"],
            "tests_total": eval_result["tests_total"],
            "cheated": float(eval_result["cheated"]),
            "step_score": step_score,
            "episode_done": float(state.done),
            "episode_success": float(state.success),
        }
        if eval_result.get("speedup"):
            metrics["speedup"] = eval_result["speedup"]
        metrics["time/eval"] = eval_time
        timing_metadata = (eval_result.get("metadata") or {}).get("timings", {})
        if "reference_load_s" in timing_metadata:
            metrics["time/ref_load"] = timing_metadata["reference_load_s"]
        if "modal_eval_s" in timing_metadata:
            metrics["time/modal_eval"] = timing_metadata["modal_eval_s"]
        metrics["time/step_total"] = time.perf_counter() - step_start

        # Trace logging (prompt + response + eval for this turn)
        await self._log_trace(
            parsed=parsed,
            eval_result=eval_result,
            format_ok=format_ok,
            reward=step_score,
            metrics=metrics,
        )

        # Build next observation if not done
        if state.done:
            next_observation = tinker.ModelInput.empty()
        else:
            messages = self._build_refinement_messages()
            next_observation = self.renderer.build_generation_prompt(messages)
            self._current_prompt_messages = messages
            self._current_observation = next_observation

        return StepResult(
            reward=step_score,
            episode_done=state.done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    async def _log_trace(
        self,
        parsed: ParsedResponse,
        eval_result: KernelEvalResult,
        format_ok: bool,
        reward: float,
        metrics: Metrics,
    ) -> None:
        """Append a JSONL trace for this turn if trace logging is enabled."""
        trace_logger = get_trace_logger()
        if trace_logger is None:
            return

        # Build refinement text for current turn (stored in history)
        last_history = self.state.history[-1] if self.state.history else None

        trace_record = {
            "mode": "multi_turn",
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "backend": self.problem.backend,
            "dataset_src": self.problem.dataset_src,
            "prompt_option": self.problem.prompt_option,
            "turn": self.state.turn_idx - 1,  # turn just completed
            "max_turns": self.state.max_turns,
            "prompt_messages": self._current_prompt_messages,
            "renderer": getattr(self.renderer, "name", type(self.renderer).__name__),
            "response": {
                "raw": parsed.raw,
                "kernel": parsed.kernel,
                "format_ok": format_ok,
            },
            "eval_result": eval_result,
            "reward": reward,
            "reward_breakdown": compute_reward_breakdown(
                eval_result, self.reward_config, thought_length=0
            ),
            "metrics": metrics,
            "state": {
                "turn_idx": self.state.turn_idx,
                "done": self.state.done,
                "success": self.state.success,
                "step_scores": list(self.state.step_scores),
            },
            "history_entry": last_history,
            "timestamp": time.time(),
            "stop_condition": str(self.stop_condition),
        }

        await trace_logger.log(trace_record)

    def get_step_scores(self) -> list[float]:
        """Get all step scores for this trajectory (for discounted return computation)."""
        return list(self.state.step_scores)

    def get_trajectory_info(self) -> dict:
        """Get summary info about this trajectory."""
        state = self.state
        return {
            "level": state.level,
            "problem_id": state.problem_id,
            "num_turns": state.turn_idx,
            "max_turns": state.max_turns,
            "success": state.success,
            "step_scores": list(state.step_scores),
            "final_correct": state.last_eval["correctness"] if state.last_eval else False,
            "final_speedup": state.last_eval.get("speedup") if state.last_eval else None,
        }


@dataclass(frozen=True)
class MultiTurnKernelBenchEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for creating groups of multi-turn KernelBench environments.
    """

    problem: KernelBenchProblem
    renderer: renderers.Renderer
    group_size: int
    max_turns: int = 4
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    system_prompt: str | None = None
    num_correct_trials: int = 5
    measure_performance: bool = False
    early_stop_on_correct: bool = True
    speedup_threshold: float | None = None
    use_modal: bool = True
    modal_timeout: float = 120.0

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of multi-turn environments for this problem."""
        return [
            MultiTurnKernelBenchEnv(
                problem=self.problem,
                renderer=self.renderer,
                max_turns=self.max_turns,
                reward_config=self.reward_config,
                system_prompt=self.system_prompt,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
                use_modal=self.use_modal,
                modal_timeout=self.modal_timeout,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """
        Compute final group rewards.

        For multi-turn, we use discounted returns computed per-step.
        The actual discounting happens in the RL loop (compute_discounted_returns).
        """
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """Return tags for logging."""
        return [
            f"level_{self.problem.level}",
            f"problem_{self.problem.problem_id}",
            "kernelbench",
            "multiturn",
        ]


class MultiTurnKernelBenchRLDataset(RLDataset):
    """
    RL Dataset for multi-turn KernelBench problems.
    """

    def __init__(
        self,
        problems: list[KernelBenchProblem],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        max_turns: int = 4,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        early_stop_on_correct: bool = True,
        speedup_threshold: float | None = None,
        shuffle: bool = True,
        num_epochs: int = 1,
        use_modal: bool = True,
        modal_timeout: float = 120.0,
    ):
        self.problems = problems
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_turns = max_turns
        self.reward_config = reward_config or RewardConfig()
        self.system_prompt = system_prompt
        self.num_correct_trials = num_correct_trials
        self.measure_performance = measure_performance
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.use_modal = use_modal
        self.modal_timeout = modal_timeout

        # Create indices
        import random
        self._problem_indices: list[int] = []
        for _ in range(num_epochs):
            epoch_indices = list(range(len(problems)))
            if shuffle:
                random.shuffle(epoch_indices)
            self._problem_indices.extend(epoch_indices)

    def __len__(self) -> int:
        total_problems = len(self.problems) * self.num_epochs
        return (total_problems + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._problem_indices))

        builders = []
        for i in range(start_idx, end_idx):
            problem_idx = self._problem_indices[i]
            problem = self.problems[problem_idx]

            builder = MultiTurnKernelBenchEnvGroupBuilder(
                problem=problem,
                renderer=self.renderer,
                group_size=self.group_size,
                max_turns=self.max_turns,
                reward_config=self.reward_config,
                system_prompt=self.system_prompt,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
                use_modal=self.use_modal,
                modal_timeout=self.modal_timeout,
            )
            builders.append(builder)

        return builders


@chz.chz
class MultiTurnKernelBenchDatasetBuilder(RLDatasetBuilder):
    """
    Builder for creating multi-turn KernelBench RL datasets (Kevin mode).

    This extends the single-turn builder with multi-turn capabilities.
    """

    # Problem selection
    level: int = 1
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"

    # Training configuration
    batch_size: int = 4
    group_size: int = 4
    num_epochs: int = 1
    shuffle: bool = True

    # Multi-turn configuration (Kevin mode)
    max_turns: int = 4
    early_stop_on_correct: bool = True
    speedup_threshold: float | None = None

    # Evaluation configuration
    num_correct_trials: int = 5
    measure_performance: bool = False

    # Reward configuration
    reward_format_weight: float = 0.1
    reward_compile_weight: float = 0.2
    reward_correctness_weight: float = 1.0
    reward_speed_weight: float = 0.0
    reward_length_weight: float = 0.05
    reward_thinking_weight: float = 0.0

    # Renderer
    renderer_name: str = "qwen3"

    # Test split
    test_fraction: float = 0.1

    # Prompt configuration
    prompt_option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot"

    # Modal configuration (isolated GPU evaluation)
    use_modal: bool = True
    modal_gpu_type: str = "A100"
    modal_timeout: float = 120.0

    async def __call__(self, tokenizer=None) -> tuple[RLDataset, RLDataset | None]:
        """Build train and optional test datasets."""

        # Get problem IDs
        problem_ids = get_problem_ids(
            self.level,
            start=self.start_problem,
            end=self.end_problem,
            dataset_src=self.dataset_src,
        )

        # Create problems
        all_problems = [
            KernelBenchProblem(
                level=self.level,
                problem_id=pid,
                backend=self.backend,
                dataset_src=self.dataset_src,
                prompt_option=self.prompt_option,
            )
            for pid in problem_ids
        ]

        # Split train/test
        if self.test_fraction > 0 and len(all_problems) > 1:
            n_test = max(1, int(len(all_problems) * self.test_fraction))
            train_problems = all_problems[:-n_test]
            test_problems = all_problems[-n_test:]
        else:
            train_problems = all_problems
            test_problems = None

        # Create renderer
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)

        # Create reward config
        reward_config = RewardConfig(
            format_weight=self.reward_format_weight,
            compile_weight=self.reward_compile_weight,
            correctness_weight=self.reward_correctness_weight,
            speed_weight=self.reward_speed_weight,
            length_weight=self.reward_length_weight,
            thinking_weight=self.reward_thinking_weight,
        )

        # Configure Modal evaluator if enabled
        if self.use_modal:
            from kernelbench_tinker.modal.evaluator import ModalEvaluatorConfig, set_modal_evaluator, ModalKernelEvaluator
            modal_config = ModalEvaluatorConfig(
                enabled=True,
                gpu_type=self.modal_gpu_type,
                timeout=int(self.modal_timeout),
            )
            set_modal_evaluator(ModalKernelEvaluator(modal_config))
            logger.info(f"Modal evaluator configured: GPU={self.modal_gpu_type}, timeout={self.modal_timeout}s")

        # Create train dataset
        train_dataset = MultiTurnKernelBenchRLDataset(
            problems=train_problems,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_turns=self.max_turns,
            reward_config=reward_config,
            num_correct_trials=self.num_correct_trials,
            measure_performance=self.measure_performance,
            early_stop_on_correct=self.early_stop_on_correct,
            speedup_threshold=self.speedup_threshold,
            shuffle=self.shuffle,
            num_epochs=self.num_epochs,
            use_modal=self.use_modal,
            modal_timeout=self.modal_timeout,
        )

        # Create test dataset
        test_dataset = None
        if test_problems:
            test_dataset = MultiTurnKernelBenchRLDataset(
                problems=test_problems,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                max_turns=self.max_turns,
                reward_config=reward_config,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
                shuffle=False,
                num_epochs=1,
                use_modal=self.use_modal,
                modal_timeout=self.modal_timeout,
            )

        return train_dataset, test_dataset
