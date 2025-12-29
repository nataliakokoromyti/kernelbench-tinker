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
import sys
from dataclasses import dataclass, asdict
from typing import Any

import chz
import tinker
import torch
from tqdm import tqdm

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter

from kernelbench_tinker.env import setup_environment
from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    evaluate_kernel,
    get_problem_ids,
    parse_structured_response,
)
from kernelbench_tinker.training.models import get_renderer_name_for_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@chz.chz
class EvalConfig:
    """Configuration for model evaluation."""

    # Model/checkpoint configuration
    checkpoint_path: str = ""  # Path to checkpoint or "tinker://..." path
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # For tokenizer/renderer

    # Evaluation configuration
    level: int = 1
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"

    # Generation configuration
    max_tokens: int = 4096
    temperature: float = 0.0  # Greedy for eval
    num_samples: int = 1  # Samples per problem

    # Evaluation settings
    num_correct_trials: int = 5
    measure_performance: bool = True

    # Output
    output_path: str = "./eval_results.json"

    # TensorBoard logging (optional, to log eval metrics alongside training)
    tensorboard_log_dir: str | None = None  # If provided, log eval metrics to TensorBoard
    tensorboard_step: int = 0  # Step to log eval metrics at

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
    # Build prompt
    messages = [
        {"role": "system", "content": "You are an expert GPU kernel developer."},
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

    # Return just the kernel code (thought is discarded for evaluation)
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
        eval_result = evaluate_kernel(
            level=problem.level,
            problem_id=problem.problem_id,
            backend=problem.backend,
            kernel_code=kernel_code,
            dataset_src=problem.dataset_src,
            num_correct_trials=cfg.num_correct_trials,
            measure_performance=cfg.measure_performance,
        )

        samples.append({
            "sample_id": sample_idx,
            "kernel_code": kernel_code[:500] + "..." if len(kernel_code) > 500 else kernel_code,
            **eval_result,
        })

    # Find best result
    correct_samples = [s for s in samples if s["correctness"]]
    if correct_samples:
        # Best by speedup
        best = max(correct_samples, key=lambda s: s.get("speedup") or 0)
    else:
        # Best by compilation
        compiled = [s for s in samples if s["compiled"]]
        best = compiled[0] if compiled else samples[0]

    return EvalResult(
        level=problem.level,
        problem_id=problem.problem_id,
        samples=samples,
        best_correct=best["correctness"],
        best_compiled=best["compiled"],
        best_speedup=best.get("speedup"),
    )


async def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    """Run full evaluation."""
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
    renderer = renderers.get_renderer(renderer_name)

    # Get problems
    problem_ids = get_problem_ids(
        cfg.level,
        start=cfg.start_problem,
        end=cfg.end_problem,
        dataset_src=cfg.dataset_src,
    )

    problems = [
        KernelBenchProblem(
            level=cfg.level,
            problem_id=pid,
            backend=cfg.backend,
            dataset_src=cfg.dataset_src,
        )
        for pid in problem_ids
    ]

    logger.info(f"Evaluating {len(problems)} problems from level {cfg.level}")

    # Evaluate each problem
    results = []
    for problem in tqdm(problems, desc="Evaluating"):
        try:
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
            "backend": cfg.backend,
            "num_samples": cfg.num_samples,
        },
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }

    return output


def main():
    """Main entry point."""
    setup_environment()

    # Parse CLI arguments
    cfg = chz.entrypoint(EvalConfig)

    logger.info("Starting KernelBench Evaluation")
    logger.info(f"Checkpoint: {cfg.checkpoint_path or 'base model'}")
    logger.info(f"Level: {cfg.level}")
    logger.info(f"Backend: {cfg.backend}")

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
        from kernelbench_tinker.training.tensorboard_logger import create_tensorboard_logger
        from kernelbench_tinker.evaluation.eval_kernelbench import EvalResults, ProblemResult

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
