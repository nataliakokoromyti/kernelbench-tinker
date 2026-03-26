"""
Multi-turn KernelBench RL environment.

Extends the single-turn KernelBenchEnv to support iterative kernel refinement.
Each episode consists of up to T turns where the model receives evaluation
feedback and can fix errors or improve performance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import tinker
from tinker.types.model_input_chunk import EncodedTextChunk
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree

from kernelbench_tinker.config.configs import EvalConfig, MultiTurnConfig
from kernelbench_tinker.envs.env_utils import build_system_prompt, evaluate_step
from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    ParsedResponse,
)
from kernelbench_tinker.training.reward import (
    RewardConfig,
    compute_reward_breakdown,
)
from kernelbench_tinker.training.trace_logger import get_trace_logger

logger = logging.getLogger(__name__)

# Limit for feedback content included in refinement prompts (char-based fallback)
MAX_HISTORY_CONTEXT_LEN = 8000

def extract_raw_content(response_text: str, eos_token: str | None = None) -> str:
    """Extract assistant content from a response, stripping the thinking block.

    Keep text after ``</think>`` when present.  Strip an unclosed ``<think>``
    prefix if the model started thinking but didn't close the tag.  If the
    response has no ``</think>`` and the EOS token is missing, return the EOS
    token as a null-response marker.
    """
    if "</think>" in response_text:
        return response_text.split("</think>")[-1].lstrip('\n')
    if "<think>" in response_text:
        # Unclosed thinking block — strip it
        return response_text.split("<think>")[0].strip()
    if eos_token is not None and eos_token not in response_text:
        return eos_token
    return response_text


def build_eval_feedback(eval_result: KernelEvalResult) -> str:
    """Build feedback string from an evaluation result for the next refinement turn."""
    error_msg = eval_result.get("error_message") or ""
    metadata = eval_result.get("metadata") or {}
    error_type = metadata.get("error_type", "")

    if not eval_result["format_ok"] or error_type == "parsing_error":
        resp = (
            "Your previous answer failed to be parsed due to not adhering "
            f"to the desired formatting. Here's the error message: {error_msg}.\n"
        )
    elif not eval_result["compiled"]:
        resp = (
            "Your previous answer failed to compile. "
            f"Here's the error message: {error_msg}.\n"
        )
    elif error_type == "runtime_error" or (not eval_result["correctness"] and error_msg):
        # Runtime error: compiled successfully but had runtime errors
        resp = (
            "Your previous answer compiled successfully but had runtime "
            f"errors. Here's the error message: {error_msg}.\n"
        )
    elif not eval_result["correctness"]:
        # Incorrect output
        resp = (
            "Your previous answer was incorrect. "
            f"Here's the error message: {error_msg}.\n"
        )
    else:
        speedup = eval_result.get("speedup") or 0.0
        resp = (
            "Your previous answer was correct but can be made faster. "
            "Here's the speedup you achieved relative to the baseline: "
            f"{speedup:.2f}.\n"
        )

    resp += "\nRestart your reasoning process and generate new, complete code."
    return resp


# ---------------------------------------------------------------------------
# Multi-turn state
# ---------------------------------------------------------------------------


@dataclass
class MultiTurnState:
    """Mutable state for a multi-turn kernel refinement episode."""

    level: int
    problem_id: int
    backend: str
    turn_idx: int
    max_turns: int
    history: list[dict]  # Per-turn: {raw_content, kernel, feedback, score}
    step_scores: list[float]
    done: bool
    success: bool


# ---------------------------------------------------------------------------
# Multi-turn environment
# ---------------------------------------------------------------------------


class MultiTurnKernelBenchEnv(Env):
    """
    Multi-turn RL environment for KernelBench.

    Each episode consists of up to T refinement steps:
    1. Turn 0: problem prompt (same as single-turn)
    2. Turn 1+: problem prompt + previous attempt feedback

    The episode ends when the kernel is correct (early stopping) or
    max_turns is reached.
    """

    def __init__(
        self,
        problem: KernelBenchProblem,
        renderer: renderers.Renderer,
        max_turns: int = 4,
        eval_config: EvalConfig | None = None,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        early_stop_on_correct: bool = False,
        speedup_threshold: float | None = None,
        tokenizer: Any | None = None,
        prompt_max_tokens: int | None = None,
        inject_think_token: bool = False,
    ):
        self.problem = problem
        self.renderer = renderer
        self.max_turns = max_turns
        self.eval_config = eval_config or EvalConfig()
        self.reward_config = reward_config or RewardConfig()
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold
        self.tokenizer = tokenizer
        self.prompt_max_tokens = prompt_max_tokens
        self.inject_think_token = inject_think_token

        self._system_prompt = system_prompt or build_system_prompt(
            problem.backend,
        )

        self._current_prompt_messages: list[renderers.Message] | None = None
        self._state: MultiTurnState | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _append_think_token(self, observation: tinker.ModelInput) -> tinker.ModelInput:
        """Append ``<think>\\n`` tokens after the chat template to force thinking mode."""
        if not self.inject_think_token or self.tokenizer is None:
            return observation
        think_ids = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        return observation.append(EncodedTextChunk(tokens=think_ids))

    @property
    def state(self) -> MultiTurnState:
        if self._state is None:
            raise RuntimeError(
                "Environment not initialized. Call initial_observation first."
            )
        return self._state

    def _build_initial_messages(self) -> list[renderers.Message]:
        messages: list[renderers.Message] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": self.problem.prompt})
        return messages

    def _count_message_tokens(self, messages: list[renderers.Message]) -> int:
        """Count total tokens across all messages using the tokenizer."""
        if self.tokenizer is None:
            return 0
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(self.tokenizer.encode(content))
        return total

    def _build_refinement_messages(self) -> list[renderers.Message]:
        """Build refinement prompt with history as alternating assistant/user turns.

        Uses token-based truncation (oldest-first) when tokenizer and
        prompt_max_tokens are set, otherwise falls back to char-based.
        """
        messages: list[renderers.Message] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Initial user message: problem without one-shot example
        base = self.problem.base_prompt
        messages.append({
            "role": "user",
            "content": base + "Here are your previous attempts:\n",
        })

        if self.state.history:
            history = list(self.state.history)

            if self.tokenizer is not None and self.prompt_max_tokens is not None:
                # Token-based truncation: count base tokens, fit history
                # into remaining budget, keeping most recent entries.
                base_tokens = self._count_message_tokens(messages)
                # 32-token safety buffer for tokenizer boundary effects
                budget = self.prompt_max_tokens - base_tokens - 32

                # Walk history backwards, accumulating tokens
                kept: list[dict] = []
                used = 0
                for entry in reversed(history):
                    entry_text = entry["raw_content"] + entry["feedback"]
                    entry_tokens = len(self.tokenizer.encode(entry_text))
                    if used + entry_tokens > budget and kept:
                        break
                    kept.append(entry)
                    used += entry_tokens
                history = list(reversed(kept))
            else:
                # Char-based fallback
                total_len = sum(
                    len(e["raw_content"]) + len(e["feedback"]) for e in history
                )
                while total_len > MAX_HISTORY_CONTEXT_LEN and len(history) > 1:
                    removed = history.pop(0)
                    total_len -= len(removed["raw_content"]) + len(removed["feedback"])

            # Add history as assistant/user turn pairs
            for entry in history:
                messages.append({"role": "assistant", "content": entry["raw_content"]})
                messages.append({"role": "user", "content": entry["feedback"]})

        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self._state = MultiTurnState(
            level=self.problem.level,
            problem_id=self.problem.problem_id,
            backend=self.problem.backend,
            turn_idx=0,
            max_turns=self.max_turns,
            history=[],
            step_scores=[],
            done=False,
            success=False,
        )
        messages = self._build_initial_messages()
        observation = self.renderer.build_generation_prompt(messages)
        observation = self._append_think_token(observation)
        self._current_prompt_messages = messages
        return observation, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        step_start = time.perf_counter()
        state = self.state

        r = await evaluate_step(
            self.problem, self.renderer, action,
            self.eval_config, self.reward_config, step_start,
        )
        state.step_scores.append(r.reward)

        # Extract content for history.
        # Use CoT summary when available (strips full reasoning, keeps concise
        # summary + kernel code), otherwise fall back to raw content with
        # thinking block stripped.
        eos_token = getattr(self.tokenizer, "eos_token", None) if self.tokenizer else None
        raw_content = extract_raw_content(r.response_text, eos_token)
        history_content = r.parsed.cot_summary if r.parsed.cot_summary else raw_content

        # Build feedback and store in history
        feedback = build_eval_feedback(r.eval_result)
        state.history.append({
            "raw_content": history_content,
            "kernel": r.kernel_code,
            "feedback": feedback,
            "score": r.reward,
        })

        # Log
        logtree.log_text(
            f"Multi-turn: Level {state.level}, ID {state.problem_id}, "
            f"Turn {state.turn_idx}"
        )
        logtree.log_text(f"Format OK: {'Yes' if r.format_ok else 'No'}")
        logtree.log_text(
            f"Compiled: {'Yes' if r.eval_result['compiled'] else 'No'}"
        )
        logtree.log_text(
            f"Correctness: {r.eval_result['tests_passed']}/{r.eval_result['tests_total']}"
        )
        if r.eval_result.get("speedup") is not None:
            logtree.log_text(f"Speedup: {r.eval_result['speedup']:.2f}x")
        logtree.log_text(f"Step score: {r.reward:.3f}")

        # Early stopping
        is_correct = r.eval_result["correctness"]
        meets_speedup = (
            self.speedup_threshold is None
            or (r.eval_result.get("speedup") or 0.0) >= self.speedup_threshold
        )
        if self.early_stop_on_correct and is_correct and meets_speedup:
            state.done = True
            state.success = True

        state.turn_idx += 1
        if state.turn_idx >= state.max_turns:
            state.done = True

        # Add multi-turn fields to shared metrics
        metrics = r.metrics
        metrics["turn"] = state.turn_idx - 1
        metrics["step_score"] = r.reward
        metrics["episode_done"] = float(state.done)
        metrics["episode_success"] = float(state.success)

        # Trace logging
        await self._log_trace(
            parsed=r.parsed,
            eval_result=r.eval_result,
            format_ok=r.format_ok,
            reward=r.reward,
            metrics=metrics,
        )

        # Next observation or done
        if state.done:
            next_observation = tinker.ModelInput.empty()
        else:
            messages = self._build_refinement_messages()
            next_observation = self.renderer.build_generation_prompt(messages)
            next_observation = self._append_think_token(next_observation)
            self._current_prompt_messages = messages

        return StepResult(
            reward=r.reward,
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
        trace_logger = get_trace_logger()
        if trace_logger is None:
            return

        trace_record = {
            "mode": "multi_turn",
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "backend": self.problem.backend,
            "dataset_src": self.problem.dataset_src,
            "prompt_option": self.problem.prompt_option,
            "turn": self.state.turn_idx - 1,
            "max_turns": self.state.max_turns,
            "prompt_messages": self._current_prompt_messages,
            "renderer": getattr(
                self.renderer, "name", type(self.renderer).__name__
            ),
            "response": {
                "raw": parsed.raw,
                "kernel": parsed.kernel,
                "cot_summary": parsed.cot_summary,
                "format_ok": format_ok,
            },
            "eval_result": eval_result,
            "reward": reward,
            "reward_breakdown": compute_reward_breakdown(
                eval_result,
                self.reward_config,
                kernel_code=parsed.kernel,
                backend=self.problem.backend,
            ),
            "metrics": metrics,
            "history": [
                {
                    "raw_content": entry["raw_content"],
                    "kernel": entry["kernel"],
                    "feedback": entry["feedback"],
                    "score": entry["score"],
                }
                for entry in self.state.history
            ],
            "state": {
                "turn_idx": self.state.turn_idx,
                "done": self.state.done,
                "success": self.state.success,
                "step_scores": list(self.state.step_scores),
            },
            "timestamp": time.time(),
            "stop_condition": str(self.stop_condition),
        }

        await trace_logger.log(trace_record)

    def get_step_scores(self) -> list[float]:
        """Return per-step scores for discounted return computation."""
        return list(self.state.step_scores)


# ---------------------------------------------------------------------------
# Group builder, dataset, dataset builder (mirrors single-turn structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTurnKernelBenchEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of multi-turn KernelBench environments."""

    problem: KernelBenchProblem
    renderer: renderers.Renderer
    group_size: int
    max_turns: int = 4
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    system_prompt: str | None = None
    early_stop_on_correct: bool = False
    speedup_threshold: float | None = None
    tokenizer: Any | None = field(default=None, hash=False, compare=False)
    prompt_max_tokens: int | None = None
    inject_think_token: bool = False

    async def make_envs(self) -> Sequence[Env]:
        return [
            MultiTurnKernelBenchEnv(
                problem=self.problem,
                renderer=self.renderer,
                max_turns=self.max_turns,
                eval_config=self.eval_config,
                reward_config=self.reward_config,
                system_prompt=self.system_prompt,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
                tokenizer=self.tokenizer,
                prompt_max_tokens=self.prompt_max_tokens,
                inject_think_token=self.inject_think_token,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        # No-op: real rewards are computed per-step inside env.step() and
        # overwritten by apply_discounted_returns before advantage estimation.
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [
            f"level_{self.problem.level}",
            f"problem_{self.problem.problem_id}",
            "kernelbench",
            "multiturn",
        ]


def wrap_builders_as_multiturn(
    builders: Sequence[EnvGroupBuilder],
    multiturn_cfg: MultiTurnConfig,
    tokenizer: Any | None = None,
) -> list[MultiTurnKernelBenchEnvGroupBuilder]:
    """Wrap single-turn KernelBenchEnvGroupBuilders as multi-turn builders.

    Called by the training loop when multiturn.enabled is True. Reads
    problem/renderer/group_size/eval_config/reward_config from each
    single-turn builder and creates MultiTurnKernelBenchEnvGroupBuilder
    instances with the multi-turn config.
    """
    from kernelbench_tinker.envs.kernelbench_env import KernelBenchEnvGroupBuilder

    wrapped = []
    for b in builders:
        if not isinstance(b, KernelBenchEnvGroupBuilder):
            raise TypeError(
                f"Expected KernelBenchEnvGroupBuilder, got {type(b).__name__}"
            )
        wrapped.append(MultiTurnKernelBenchEnvGroupBuilder(
            problem=b.problem,
            renderer=b.renderer,
            group_size=b.group_size,
            max_turns=multiturn_cfg.max_turns,
            eval_config=b.eval_config,
            reward_config=b.reward_config,
            system_prompt=build_system_prompt(b.problem.backend),
            early_stop_on_correct=multiturn_cfg.early_stop_on_correct,
            speedup_threshold=multiturn_cfg.speedup_threshold,
            tokenizer=tokenizer,
            prompt_max_tokens=multiturn_cfg.prompt_max_tokens,
            inject_think_token=multiturn_cfg.inject_think_token,
        ))
    return wrapped
