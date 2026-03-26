"""
Shared utilities for KernelBench environments.

Contains helpers used by both the single-turn and multi-turn environments:
- System prompt construction
- Step evaluation (parse → evaluate → reward → metrics)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Action, Metrics

from kernelbench_tinker.config.configs import EvalConfig
from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    ParsedResponse,
    evaluate_kernel_async,
    parse_structured_response,
)
from kernelbench_tinker.training.reward import (
    RewardConfig,
    compute_reward,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalStepResult:
    """Result from evaluate_step(), shared by single-turn and multi-turn envs."""

    parsed: ParsedResponse
    eval_result: KernelEvalResult
    format_ok: bool
    kernel_code: str
    reward: float
    metrics: Metrics
    response_text: str  # Raw response content from renderer (before structured parsing)


def build_system_prompt(backend: str) -> str:
    """Build a backend-specific system prompt for kernel generation.

    Used by both single-turn and multi-turn environments.
    """
    return (
        f"You are an expert GPU kernel developer. Your task is to optimize PyTorch "
        f"operations by writing efficient custom {backend.upper()} kernels.\n"
        f"\n"
        f"When given a PyTorch model, write an optimized kernel implementation.\n"
        f"\n"
        f"Your solution must:\n"
        f"- Be a drop-in replacement as a class named `ModelNew`\n"
        f"- Use custom {backend.upper()} kernels, not just PyTorch operations\n"
        f"- Be correct and produce the same results as the reference\n"
        f"\n"
        f"You MUST respond in exactly this format:\n"
        f"\n"
        f"<KERNEL>\n"
        f"```python\n"
        f"# Your complete optimized implementation here\n"
        f"class ModelNew(nn.Module):\n"
        f"    ...\n"
        f"```\n"
        f"</KERNEL>"
    )


async def evaluate_step(
    problem: KernelBenchProblem,
    renderer: renderers.Renderer,
    action: Action,
    eval_config: EvalConfig,
    reward_config: RewardConfig,
    step_start: float,
) -> EvalStepResult:
    """Parse, evaluate, and compute reward for a single action.

    Shared by KernelBenchEnv.step() and MultiTurnKernelBenchEnv.step().
    """
    message, _ = renderer.parse_response(action)
    response_text = message.get("content", "")

    parsed = parse_structured_response(response_text)
    kernel_code = parsed.kernel
    format_ok = parsed.format_ok

    eval_start = time.perf_counter()
    cfg = eval_config
    eval_result = await evaluate_kernel_async(
        level=problem.level,
        problem_id=problem.problem_id,
        backend=problem.backend,
        kernel_code=kernel_code,
        dataset_src=problem.dataset_src,
        num_correct_trials=cfg.num_correct_trials,
        measure_performance=cfg.measure_performance,
        num_perf_trials=cfg.num_perf_trials,
        timing_method=cfg.timing_method,
        precision=cfg.precision,
        check_for_excessive_speedup=cfg.check_for_excessive_speedup,
        excessive_speedup_threshold=cfg.excessive_speedup_threshold,
        timeout=cfg.modal_timeout,
    )
    eval_time = time.perf_counter() - eval_start

    reward = compute_reward(
        eval_result,
        reward_config,
        kernel_code=kernel_code,
        backend=problem.backend,
    )

    metrics: Metrics = {
        "level": problem.level,
        "problem_id": problem.problem_id,
        "format_ok": float(format_ok),
        "compiled": float(eval_result["compiled"]),
        "correctness": float(eval_result["correctness"]),
        "tests_passed": eval_result["tests_passed"],
        "tests_total": eval_result["tests_total"],
    }
    if eval_result.get("speedup") is not None:
        metrics["speedup"] = eval_result["speedup"]
    if eval_result.get("runtime_ms") is not None:
        metrics["runtime_ms"] = eval_result["runtime_ms"]
    metrics["time/eval"] = eval_time
    timing_metadata = (eval_result.get("metadata") or {}).get("timings", {})
    if "reference_load_s" in timing_metadata:
        metrics["time/ref_load"] = timing_metadata["reference_load_s"]
    if "modal_eval_s" in timing_metadata:
        metrics["time/modal_eval"] = timing_metadata["modal_eval_s"]
    metrics["time/step_total"] = time.perf_counter() - step_start

    return EvalStepResult(
        parsed=parsed,
        eval_result=eval_result,
        format_ok=format_ok,
        kernel_code=kernel_code,
        reward=reward,
        metrics=metrics,
        response_text=response_text,
    )
