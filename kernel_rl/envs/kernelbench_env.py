"""
Tinker RL Environment for KernelBench.

This module implements the Env, EnvGroupBuilder, and RLDataset interfaces
from tinker_cookbook.rl.types for training models to write GPU kernels.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, Callable, Any

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

from kernel_rl.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    ParsedResponse,
    evaluate_kernel,
    evaluate_kernel_async,
    get_problem_ids,
    extract_code_block,
    parse_structured_response,
    get_global_retriever,
    set_global_retriever,
)
from kernel_rl.training.reward import compute_reward, compute_reward_breakdown, RewardConfig
from kernel_rl.training.trace_logger import get_trace_logger

logger = logging.getLogger(__name__)


# Default system prompt for kernel generation (structured format)
DEFAULT_SYSTEM_PROMPT = """You are an expert GPU kernel developer. Your task is to optimize PyTorch operations by writing efficient custom GPU kernels.

When given a PyTorch model, you should:
1. Analyze the operations being performed
2. Write an optimized kernel implementation
3. Return your solution as a Python class named `ModelNew` that implements the same interface

Your kernel should:
- Be functionally correct (produce the same outputs as the reference)
- Be efficient (aim for speedup over the PyTorch baseline)
- Handle edge cases properly
- Use the specified backend (Triton, CUDA, etc.)

You MUST respond in exactly this format:

<think>
1-5 short bullet points describing:
- What optimization strategy you will use
- Key implementation details (tiling, memory layout, etc.)
- Any constraints or edge cases to handle

Keep this section under 150 tokens.
</think>

<KERNEL>
```python
# Your complete optimized implementation here
class ModelNew(nn.Module):
    ...
```
</KERNEL>"""


class KernelBenchEnv(Env):
    """
    A single-turn RL environment for a KernelBench problem.

    Each episode consists of:
    1. Initial observation: The problem prompt (PyTorch reference code + instructions)
    2. Action: The model generates a kernel implementation
    3. Step: Evaluate the kernel and compute reward

    The episode ends after one step (single-turn). Future versions may support
    multi-turn interactions where the model can iterate based on compiler errors.
    """

    def __init__(
        self,
        problem: KernelBenchProblem,
        renderer: renderers.Renderer,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        use_modal: bool = True,
        modal_timeout: float = 120.0,
    ):
        """
        Initialize the KernelBench environment.

        Args:
            problem: The KernelBench problem to solve
            renderer: Tinker renderer for formatting messages
            reward_config: Configuration for reward computation
            system_prompt: Optional custom system prompt
            num_correct_trials: Number of correctness trials for evaluation
            measure_performance: Whether to measure kernel runtime
            use_modal: Whether to use Modal for isolated GPU evaluation
            modal_timeout: Timeout in seconds for Modal evaluation
        """
        self.problem = problem
        self.renderer = renderer
        self.reward_config = reward_config or RewardConfig()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.num_correct_trials = num_correct_trials
        self.measure_performance = measure_performance
        self.use_modal = use_modal
        self.modal_timeout = modal_timeout

        # State for multi-turn (future)
        self._turn = 0
        self._last_result: KernelEvalResult | None = None
        self._last_kernel: str | None = None
        self._current_prompt_messages: list[renderers.Message] | None = None
        self._current_observation: tinker.ModelInput | None = None

    @property
    def stop_condition(self) -> StopCondition:
        """Get stop sequences for generation."""
        return self.renderer.get_stop_sequences()

    def _build_initial_messages(self) -> list[renderers.Message]:
        """Build the initial conversation for the problem."""
        messages: list[renderers.Message] = []

        # Add system prompt if supported
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add the problem prompt as user message
        messages.append({"role": "user", "content": self.problem.prompt})

        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Get the initial observation for the episode.

        Returns:
            Tuple of (observation, stop_condition)
        """
        messages = self._build_initial_messages()
        observation = self.renderer.build_generation_prompt(messages)
        self._current_prompt_messages = messages
        self._current_observation = observation
        return observation, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Process the model's action (generated kernel code).

        Args:
            action: Token IDs generated by the model

        Returns:
            StepResult with reward and episode done status
        """
        step_start = time.perf_counter()
        self._turn += 1

        # Parse the response to get text
        message, parse_success = self.renderer.parse_response(action)
        response_text = message.get("content", "")

        # Parse structured response (extracts <think> and <KERNEL> blocks)
        parsed = parse_structured_response(response_text)
        kernel_code = parsed.kernel
        self._last_kernel = kernel_code

        # Log thinking content if present (for debugging/analysis)
        if parsed.thought:
            logtree.log_text(f"Thought: {parsed.thought[:200]}...")

        # Check format validity
        format_ok = parsed.format_ok

        # Evaluate the kernel (using Modal for isolation if enabled)
        eval_start = time.perf_counter()
        if self.use_modal:
            eval_result = await evaluate_kernel_async(
                level=self.problem.level,
                problem_id=self.problem.problem_id,
                backend=self.problem.backend,
                kernel_code=kernel_code,
                dataset_src=self.problem.dataset_src,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                timeout=self.modal_timeout,
            )
        else:
            # Local evaluation (no isolation - use with caution)
            eval_result = evaluate_kernel(
                level=self.problem.level,
                problem_id=self.problem.problem_id,
                backend=self.problem.backend,
                kernel_code=kernel_code,
                dataset_src=self.problem.dataset_src,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
            )
        self._last_result = eval_result
        eval_time = time.perf_counter() - eval_start

        # Compute reward (includes thinking bonus)
        reward = compute_reward(eval_result, self.reward_config, thought_length=len(parsed.thought))

        # Log the attempt
        logtree.log_text(f"Problem: Level {self.problem.level}, ID {self.problem.problem_id}")
        logtree.log_text(f"Format OK: {'Yes' if format_ok else 'No'}")
        logtree.log_text(f"Compiled: {'Yes' if eval_result['compiled'] else 'No'}")
        logtree.log_text(
            f"Correctness: {eval_result['tests_passed']}/{eval_result['tests_total']}"
        )
        if eval_result.get("speedup"):
            logtree.log_text(f"Speedup: {eval_result['speedup']:.2f}x")
        logtree.log_text(f"Reward: {reward:.3f}")
        if eval_result.get("error_message"):
            logtree.log_text(f"Error: {eval_result['error_message'][:200]}")

        # Build metrics
        metrics: Metrics = {
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "format_ok": float(format_ok),
            "compiled": float(eval_result["compiled"]),
            "correctness": float(eval_result["correctness"]),
            "tests_passed": eval_result["tests_passed"],
            "tests_total": eval_result["tests_total"],
            "cheated": float(eval_result["cheated"]),
            "thought_length": len(parsed.thought),  # Track thinking token usage
            "has_thought": float(bool(parsed.thought)),
        }
        if eval_result.get("speedup"):
            metrics["speedup"] = eval_result["speedup"]
        if eval_result.get("runtime_ms"):
            metrics["runtime_ms"] = eval_result["runtime_ms"]
        metrics["time/eval"] = eval_time
        timing_metadata = (eval_result.get("metadata") or {}).get("timings", {})
        if "reference_load_s" in timing_metadata:
            metrics["time/ref_load"] = timing_metadata["reference_load_s"]
        if "modal_eval_s" in timing_metadata:
            metrics["time/modal_eval"] = timing_metadata["modal_eval_s"]
        metrics["time/step_total"] = time.perf_counter() - step_start

        # Trace logging (prompt + response + eval)
        await self._log_trace(
            parsed=parsed,
            eval_result=eval_result,
            format_ok=format_ok,
            reward=reward,
            metrics=metrics,
        )

        # Single-turn: episode ends after first step
        # TODO: Multi-turn support would continue here with compiler feedback
        episode_done = True

        return StepResult(
            reward=reward,
            episode_done=episode_done,
            next_observation=tinker.ModelInput.empty(),
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
        """Append a JSONL trace for this step if trace logging is enabled."""
        trace_logger = get_trace_logger()
        if trace_logger is None:
            return

        trace_record = {
            "mode": "single_turn",
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "backend": self.problem.backend,
            "dataset_src": self.problem.dataset_src,
            "prompt_option": self.problem.prompt_option,
            "raicl_k": self.problem.raicl_k,
            "turn": self._turn - 1,  # step just completed
            "prompt_messages": self._current_prompt_messages,
            "renderer": getattr(self.renderer, "name", type(self.renderer).__name__),
            "response": {
                "raw": parsed.raw,
                "thought": parsed.thought,
                "kernel": parsed.kernel,
                "format_ok": format_ok,
            },
            "eval_result": eval_result,
            "reward": reward,
            "reward_breakdown": compute_reward_breakdown(
                eval_result, self.reward_config, thought_length=len(parsed.thought)
            ),
            "metrics": metrics,
            "timestamp": time.time(),
            "stop_condition": str(self.stop_condition),
        }

        await trace_logger.log(trace_record)


@dataclass(frozen=True)
class KernelBenchEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for creating groups of KernelBench environments.

    Each group corresponds to a single problem with multiple rollouts.
    This enables GRPO-style training where rewards are normalized within groups.
    """

    problem: KernelBenchProblem
    renderer: renderers.Renderer
    group_size: int  # Number of rollouts per problem
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    system_prompt: str | None = None
    num_correct_trials: int = 5
    measure_performance: bool = False
    use_modal: bool = True
    modal_timeout: float = 120.0

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments for this problem."""
        return [
            KernelBenchEnv(
                problem=self.problem,
                renderer=self.renderer,
                reward_config=self.reward_config,
                system_prompt=self.system_prompt,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
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

        By default, we use per-step rewards only (returned by step()).
        This method can be overridden for group-level reward modifications.
        """
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """Return tags for logging and metric aggregation."""
        return [
            f"level_{self.problem.level}",
            f"problem_{self.problem.problem_id}",
            "kernelbench",
        ]


class KernelBenchRLDataset(RLDataset):
    """
    RL Dataset for KernelBench problems.

    Provides batches of EnvGroupBuilders for training.
    Each batch contains multiple problems, each with multiple rollouts.
    """

    def __init__(
        self,
        problems: list[KernelBenchProblem],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        shuffle: bool = True,
        num_epochs: int = 1,
        use_modal: bool = True,
        modal_timeout: float = 180.0,
    ):
        """
        Initialize the RL dataset.

        Args:
            problems: List of KernelBench problems
            renderer: Tinker renderer for formatting
            batch_size: Number of problems per batch
            group_size: Number of rollouts per problem
            reward_config: Reward configuration
            system_prompt: Optional custom system prompt
            num_correct_trials: Correctness trials per evaluation
            measure_performance: Whether to measure runtime
            shuffle: Whether to shuffle problems each epoch
            num_epochs: Number of training epochs
            use_modal: Whether to use Modal for isolated GPU evaluation
            modal_timeout: Timeout in seconds for Modal evaluation
        """
        self.problems = problems
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.reward_config = reward_config or RewardConfig()
        self.system_prompt = system_prompt
        self.num_correct_trials = num_correct_trials
        self.measure_performance = measure_performance
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.use_modal = use_modal
        self.modal_timeout = modal_timeout

        # Create shuffled indices for each epoch
        self._problem_indices: list[int] = []
        for _ in range(num_epochs):
            epoch_indices = list(range(len(problems)))
            if shuffle:
                random.shuffle(epoch_indices)
            self._problem_indices.extend(epoch_indices)

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        total_problems = len(self.problems) * self.num_epochs
        return (total_problems + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """
        Get a batch of EnvGroupBuilders.

        Args:
            index: Batch index

        Returns:
            List of EnvGroupBuilders, one per problem in the batch
        """
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._problem_indices))

        builders = []
        for i in range(start_idx, end_idx):
            problem_idx = self._problem_indices[i]
            problem = self.problems[problem_idx]

            builder = KernelBenchEnvGroupBuilder(
                problem=problem,
                renderer=self.renderer,
                group_size=self.group_size,
                reward_config=self.reward_config,
                system_prompt=self.system_prompt,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                use_modal=self.use_modal,
                modal_timeout=self.modal_timeout,
            )
            builders.append(builder)

        return builders


@chz.chz
class KernelBenchDatasetBuilder(RLDatasetBuilder):
    """
    Builder for creating KernelBench RL datasets.

    This is a chz-compatible configuration class that can be used
    in Tinker's training configuration.
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

    # Evaluation configuration
    num_correct_trials: int = 5
    measure_performance: bool = False

    # Reward configuration
    reward_format_weight: float = 0.1
    reward_compile_weight: float = 0.2
    reward_correctness_weight: float = 1.0
    reward_speed_weight: float = 0.0
    reward_length_weight: float = 0.05  # Tie-breaking for uniform rewards
    reward_thinking_weight: float = 0.1  # Reward for using <think> blocks

    # Renderer
    renderer_name: str = "qwen3"

    # Test split
    test_fraction: float = 0.1

    # Prompt configuration
    prompt_option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot", "raicl"
    rag_index_path: str | None = None  # Path to RAG index (required for raicl)
    raicl_k: int = 3  # Number of examples to retrieve for RA-ICL

    # Modal configuration (isolated GPU evaluation)
    use_modal: bool = True  # Use Modal for isolated evaluation
    modal_gpu_type: str = "A100"  # GPU type to use on Modal
    modal_timeout: float = 120.0  # Timeout in seconds per kernel

    async def __call__(self, tokenizer=None) -> tuple[RLDataset, RLDataset | None]:
        """Build train and optional test datasets.

        Args:
            tokenizer: The tokenizer to use for the renderer. Required for most renderers.
        """
        # Load RAG index if using raicl prompts
        if self.prompt_option == "raicl":
            if not self.rag_index_path:
                raise ValueError("rag_index_path is required when using raicl prompt_option")
            from kernel_rl.rag.retriever import KernelRetriever
            retriever = KernelRetriever.load(self.rag_index_path)
            set_global_retriever(retriever)
            logger.info(f"Loaded RAG index from {self.rag_index_path}")

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
                raicl_k=self.raicl_k,
            )
            for pid in problem_ids
        ]

        # Split into train/test
        if self.test_fraction > 0 and len(all_problems) > 1:
            n_test = max(1, int(len(all_problems) * self.test_fraction))
            # Use last N problems as test set for reproducibility
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
            from kernel_rl.modal.evaluator import ModalEvaluatorConfig, set_modal_evaluator, ModalKernelEvaluator
            modal_config = ModalEvaluatorConfig(
                enabled=True,
                gpu_type=self.modal_gpu_type,
                timeout=int(self.modal_timeout),
            )
            set_modal_evaluator(ModalKernelEvaluator(modal_config))
            logger.info(f"Modal evaluator configured: GPU={self.modal_gpu_type}, timeout={self.modal_timeout}s")

        # Create train dataset
        train_dataset = KernelBenchRLDataset(
            problems=train_problems,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            reward_config=reward_config,
            num_correct_trials=self.num_correct_trials,
            measure_performance=self.measure_performance,
            shuffle=self.shuffle,
            num_epochs=self.num_epochs,
            use_modal=self.use_modal,
            modal_timeout=self.modal_timeout,
        )

        # Create test dataset if we have test problems
        test_dataset = None
        if test_problems:
            test_dataset = KernelBenchRLDataset(
                problems=test_problems,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                reward_config=reward_config,
                num_correct_trials=self.num_correct_trials,
                measure_performance=self.measure_performance,
                shuffle=False,
                num_epochs=1,
                use_modal=self.use_modal,
                modal_timeout=self.modal_timeout,
            )

        return train_dataset, test_dataset
