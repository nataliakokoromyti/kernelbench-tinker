#!/usr/bin/env python3
"""
CLI entrypoint for evaluating trained KernelBench models.

Usage:
    python -m kernelbench_tinker.scripts.eval_kernel_rl \
        checkpoint_path=./runs/kernelbench_tinker_v1/checkpoints/final \
        level=1 \
        output_path=./runs/kernelbench_tinker_v1/eval_results.json

Or with the installed script:
    eval-kernel-rl checkpoint_path=... level=...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import chz
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tqdm import tqdm

from kernelbench_tinker.env import setup_environment
from kernelbench_tinker.envs.env_utils import build_system_prompt
from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    evaluate_kernel_async,
    get_problem_ids,
    parse_structured_response,
)
from kernelbench_tinker.training.models import (
    KernelBenchTokenCompleter,
    get_renderer_name_for_model,
)

logger = logging.getLogger(__name__)


def pick_best_sample(
    samples: list[dict[str, Any]],
) -> tuple[bool, bool, float | None]:
    """Pick the best sample from a list of evaluation results.

    Returns (best_correct, best_compiled, best_speedup).
    """
    def speedup_value(sample: dict[str, Any]) -> float:
        speedup = sample.get("speedup")
        return float(speedup) if isinstance(speedup, (int, float)) else 0.0

    correct_samples = [s for s in samples if s.get("correctness")]
    if correct_samples:
        best = max(correct_samples, key=speedup_value)
    else:
        compiled = [s for s in samples if s.get("compiled")]
        best = compiled[0] if compiled else samples[0]

    best_speedup: float | None = None
    speedup_obj = best.get("speedup")
    if isinstance(speedup_obj, (int, float)):
        best_speedup = float(speedup_obj)

    return bool(best.get("correctness")), bool(best.get("compiled")), best_speedup


@chz.chz
class EvalConfig:
    """Configuration for model evaluation."""

    # Model/checkpoint configuration
    checkpoint_path: str = ""  # Path to checkpoint or "tinker://..." path
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # For tokenizer/renderer

    # Evaluation configuration
    level: int = 1
    levels: list[int] | None = None  # Iterate multiple levels (overrides level when set)
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"

    # Generation configuration
    max_tokens: int = 4096
    temperature: float = 0.0  # Greedy for eval
    top_p: float = 1.0  # Nucleus sampling (1.0 = disabled)
    seed: int | None = None  # Random seed for generation (null = random)
    num_samples: int = 1  # Samples per problem

    # Evaluation settings
    num_correct_trials: int = 5
    measure_performance: bool = True
    num_perf_trials: int = 100
    timing_method: str = "cuda_event"
    precision: str = "fp32"
    check_for_excessive_speedup: bool = True
    excessive_speedup_threshold: float = 10.0

    # Evaluator backend selection: "modal" (cloud NVIDIA) or "local" (in-process subprocess for AMD/HIP)
    evaluator_backend: str = "modal"

    # Modal configuration
    modal_gpu_type: str = "A100"
    modal_timeout: float = 120.0

    # Local evaluator configuration (used when evaluator_backend == "local")
    # gpu_arch is passed to set_gpu_arch() — e.g. ["gfx950"] for MI350X, ["gfx942"] for MI300X
    gpu_arch: list[str] | None = None
    local_timeout: float = 300.0

    # Prompt configuration
    prompt_option: str = "one_shot"
    prompt_include_hardware: bool = False
    prompt_gpu_name: str | None = None

    # Output
    output_path: str = "./eval_results.json"
    max_kernel_code_chars: int | None = 500  # Set to None to store full code

    # TensorBoard logging (optional, to log eval metrics alongside training)
    tensorboard_log_dir: str | None = None  # If provided, log eval metrics to TensorBoard
    tensorboard_step: int = 0  # Step to log eval metrics at

    # Multi-turn inference
    multiturn_enabled: bool = False
    multiturn_max_turns: int = 8
    inject_think_token: bool = False  # Append <think>\n to generation prompts
    prompt_max_tokens: int | None = None  # Token budget for history truncation

    # Tinker API
    base_url: str | None = None


@dataclass
class EvalResult:
    """Result for a single problem."""
    level: int
    problem_id: int
    samples: list[dict[str, Any]]
    best_correct: bool
    best_compiled: bool
    best_speedup: float | None


async def generate_kernel(
    sampling_client: tinker.SamplingClient,
    problem: KernelBenchProblem,
    renderer: renderers.Renderer,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate a kernel for a problem."""
    # Build prompt (same system prompt as training env)
    messages = [
        {"role": "system", "content": build_system_prompt(problem.backend)},
        {"role": "user", "content": problem.prompt},
    ]
    observation = renderer.build_generation_prompt(messages)
    stop_condition = renderer.get_stop_sequences()

    # Generate
    completer = TinkerTokenCompleter(
        sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = await completer(observation, stop_condition)

    # Parse response
    message, _ = renderer.parse_response(result.tokens)
    content = message.get("content", "")

    # Parse structured response (extracts <think> and <KERNEL> blocks)
    parsed = parse_structured_response(content)

    # Return just the kernel code
    return parsed.kernel if parsed.kernel else content


async def evaluate_problem(
    sampling_client: tinker.SamplingClient,
    problem: KernelBenchProblem,
    renderer: renderers.Renderer,
    cfg: EvalConfig,
) -> EvalResult:
    """Evaluate a single problem with multiple samples."""
    samples = []

    for sample_idx in range(cfg.num_samples):
        # Generate kernel
        kernel_code = await generate_kernel(
            sampling_client,
            problem,
            renderer,
            cfg.max_tokens,
            cfg.temperature if cfg.num_samples == 1 else 1.0,  # Use temp=1 for multiple samples
        )

        # Evaluate
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

        if cfg.max_kernel_code_chars is None:
            kernel_code_logged = kernel_code
        elif len(kernel_code) > cfg.max_kernel_code_chars:
            kernel_code_logged = kernel_code[: cfg.max_kernel_code_chars] + "..."
        else:
            kernel_code_logged = kernel_code

        samples.append({
            "sample_id": sample_idx,
            "kernel_code": kernel_code_logged,
            **eval_result,
        })

    best_correct, best_compiled, best_speedup = pick_best_sample(samples)

    return EvalResult(
        level=problem.level,
        problem_id=problem.problem_id,
        samples=samples,
        best_correct=best_correct,
        best_compiled=best_compiled,
        best_speedup=best_speedup,
    )


async def evaluate_problem_multiturn(
    sampling_client: tinker.SamplingClient,
    problem: KernelBenchProblem,
    renderer: renderers.Renderer,
    cfg: EvalConfig,
    tokenizer: object | None = None,
) -> EvalResult:
    """Evaluate a single problem using multi-turn refinement.

    Reuses MultiTurnKernelBenchEnv so history truncation, feedback
    construction, and think-token injection stay in one place.
    """
    from kernelbench_tinker.config.configs import EvalConfig as KernelEvalConfig
    from kernelbench_tinker.envs.multiturn_kernelbench_env import MultiTurnKernelBenchEnv

    eval_config = KernelEvalConfig(
        num_correct_trials=cfg.num_correct_trials,
        measure_performance=cfg.measure_performance,
        num_perf_trials=cfg.num_perf_trials,
        timing_method=cfg.timing_method,
        precision=cfg.precision,
        check_for_excessive_speedup=cfg.check_for_excessive_speedup,
        excessive_speedup_threshold=cfg.excessive_speedup_threshold,
        evaluator_backend=cfg.evaluator_backend,
        modal_gpu_type=cfg.modal_gpu_type,
        modal_timeout=cfg.modal_timeout,
        gpu_arch=list(cfg.gpu_arch) if cfg.gpu_arch else [],
        local_timeout=cfg.local_timeout,
    )

    env = MultiTurnKernelBenchEnv(
        problem=problem,
        renderer=renderer,
        max_turns=cfg.multiturn_max_turns,
        eval_config=eval_config,
        system_prompt=build_system_prompt(problem.backend),
        tokenizer=tokenizer,
        prompt_max_tokens=cfg.prompt_max_tokens,
        inject_think_token=cfg.inject_think_token,
    )

    completer = KernelBenchTokenCompleter(
        sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature if cfg.num_samples == 1 else 1.0,
        top_p=cfg.top_p,
        seed=cfg.seed,
    )

    observation, stop = await env.initial_observation()
    samples: list[dict[str, Any]] = []

    for turn in range(cfg.multiturn_max_turns):
        result = await completer(observation, stop)
        step_result = await env.step(result.tokens)
        m = step_result.metrics

        kernel_code = env.state.history[-1]["kernel"]
        if cfg.max_kernel_code_chars is not None and len(kernel_code) > cfg.max_kernel_code_chars:
            kernel_code = kernel_code[: cfg.max_kernel_code_chars] + "..."

        samples.append({
            "sample_id": turn,
            "turn": turn,
            "kernel_code": kernel_code,
            "format_ok": bool(m.get("format_ok")),
            "compiled": bool(m.get("compiled")),
            "correctness": bool(m.get("correctness")),
            "tests_passed": m.get("tests_passed", 0),
            "tests_total": m.get("tests_total", 0),
            "speedup": m.get("speedup"),
            "runtime_ms": m.get("runtime_ms"),
        })

        if step_result.episode_done:
            break
        observation = step_result.next_observation
        stop = step_result.next_stop_condition

    best_correct, best_compiled, best_speedup = pick_best_sample(samples)

    return EvalResult(
        level=problem.level,
        problem_id=problem.problem_id,
        samples=samples,
        best_correct=best_correct,
        best_compiled=best_compiled,
        best_speedup=best_speedup,
    )


async def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    """Run full evaluation."""
    # Install the appropriate evaluator backend (modal vs local).
    from kernelbench_tinker.envs.evaluator_dispatch import set_evaluator_from_eval_config
    set_evaluator_from_eval_config(cfg)

    # Create Tinker client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    # Load checkpoint
    if cfg.checkpoint_path:
        logger.info(f"Loading checkpoint: {cfg.checkpoint_path}")
        sampling_client = service_client.create_sampling_client(cfg.checkpoint_path)
    else:
        # Use base model
        logger.info(f"Using base model: {cfg.model_name}")
        sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

    # Get renderer
    renderer_name = get_renderer_name_for_model(cfg.model_name)
    tokenizer = tokenizer_utils.get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Iterate one or more levels (the for-loop the user asked for: each task,
    # each level, end-to-end on the same configured backend).
    active_levels = list(cfg.levels) if cfg.levels else [cfg.level]

    problems: list[KernelBenchProblem] = []
    for lvl in active_levels:
        problem_ids = get_problem_ids(
            lvl,
            start=cfg.start_problem,
            end=cfg.end_problem,
            dataset_src=cfg.dataset_src,
        )
        problems.extend(
            KernelBenchProblem(
                level=lvl,
                problem_id=pid,
                backend=cfg.backend,
                dataset_src=cfg.dataset_src,
                prompt_option=cfg.prompt_option,
                prompt_precision=cfg.precision,
                prompt_include_hardware=cfg.prompt_include_hardware,
                prompt_gpu_name=cfg.prompt_gpu_name,
            )
            for pid in problem_ids
        )

    logger.info(
        f"Evaluating {len(problems)} problems across levels={active_levels} backend={cfg.backend}"
    )

    # Evaluate each problem
    results = []
    for problem in tqdm(problems, desc="Evaluating"):
        try:
            if cfg.multiturn_enabled:
                result = await evaluate_problem_multiturn(
                    sampling_client, problem, renderer, cfg, tokenizer=tokenizer
                )
                results.append(result)
                continue
            result = await evaluate_problem(
                sampling_client, problem, renderer, cfg
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating problem {problem.problem_id}: {e}")
            results.append(EvalResult(
                level=problem.level,
                problem_id=problem.problem_id,
                samples=[{"error": str(e)}],
                best_correct=False,
                best_compiled=False,
                best_speedup=None,
            ))

    # Compute aggregate metrics
    num_correct = sum(1 for r in results if r.best_correct)
    num_compiled = sum(1 for r in results if r.best_compiled)
    speedups = [r.best_speedup for r in results if r.best_speedup is not None]

    metrics = {
        "num_problems": len(results),
        "num_correct": num_correct,
        "num_compiled": num_compiled,
        "correct_rate": num_correct / len(results) if results else 0,
        "compile_rate": num_compiled / len(results) if results else 0,
        "mean_speedup": sum(speedups) / len(speedups) if speedups else None,
        "num_with_speedup": len(speedups),
    }

    # Build full output
    output = {
        "config": {
            "checkpoint_path": cfg.checkpoint_path,
            "model_name": cfg.model_name,
            "level": cfg.level,
            "levels": list(cfg.levels) if cfg.levels else [cfg.level],
            "backend": cfg.backend,
            "evaluator_backend": cfg.evaluator_backend,
            "gpu_arch": list(cfg.gpu_arch) if cfg.gpu_arch else [],
            "num_samples": cfg.num_samples,
        },
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }

    return output


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    setup_environment()

    # Parse CLI arguments
    cfg = chz.entrypoint(EvalConfig)

    logger.info("Starting KernelBench Evaluation")
    logger.info(f"Checkpoint: {cfg.checkpoint_path or 'base model'}")
    logger.info(f"Levels: {list(cfg.levels) if cfg.levels else [cfg.level]}")
    logger.info(f"Backend: {cfg.backend} (evaluator={cfg.evaluator_backend})")

    # Run evaluation
    output = asyncio.run(run_evaluation(cfg))

    # Print summary
    metrics = output["metrics"]
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  Problems: {metrics['num_problems']}")
    logger.info(f"  Compiled: {metrics['num_compiled']} ({metrics['compile_rate']:.1%})")
    logger.info(f"  Correct:  {metrics['num_correct']} ({metrics['correct_rate']:.1%})")
    if metrics.get("mean_speedup"):
        logger.info(f"  Mean Speedup: {metrics['mean_speedup']:.2f}x")
    logger.info("=" * 50)

    # Log to TensorBoard if specified
    if cfg.tensorboard_log_dir:
        # Lazy import to avoid circular imports
        from kernelbench_tinker.evaluation.eval_kernelbench import EvalResults, ProblemResult
        from kernelbench_tinker.training.tensorboard_logger import create_tensorboard_logger

        tb_logger = create_tensorboard_logger(cfg.tensorboard_log_dir)

        # Create EvalResults for computing summary
        eval_results = EvalResults()
        for result_data in output["results"]:
            problem_result = ProblemResult(
                level=result_data["level"],
                problem_id=result_data["problem_id"],
                samples=result_data["samples"],
            )
            eval_results.add_problem(problem_result)

        # Log evaluation summary
        summary = eval_results.summary()
        tb_logger.log_evaluation_metrics(summary, cfg.tensorboard_step, prefix=f"Eval/Level{cfg.level}")

        tb_logger.flush()
        tb_logger.close()
        logger.info(f"Logged evaluation metrics to TensorBoard at step {cfg.tensorboard_step}")

    # Save results
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    with open(cfg.output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
