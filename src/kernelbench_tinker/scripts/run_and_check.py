#!/usr/bin/env python3
"""
Minimal run-and-check helper using KernelBench eval utilities.

Supports:
- ref_origin=kernelbench (level/problem_id) for Modal eval
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import chz

from kernelbench_tinker.envs.kernelbench_client import (
    evaluate_kernel_async,
    extract_code_block,
)


@chz.chz
class RunAndCheckConfig:
    # Reference origin
    ref_origin: str = "kernelbench"
    level: int | None = None
    problem_id: int | None = None
    dataset_src: str = "huggingface"

    # Kernel source
    kernel_src_path: str = ""
    backend: str = "triton"

    # Eval settings
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    measure_performance: bool = True
    timing_method: str = "cuda_event"
    precision: str = "fp32"
    check_for_excessive_speedup: bool = True
    excessive_speedup_threshold: float = 10.0
    modal_timeout: float = 120.0


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _prepare_kernel_code(raw: str) -> str:
    extracted = extract_code_block(raw)
    return extracted or raw


async def _run() -> None:
    cfg = chz.entrypoint(RunAndCheckConfig)

    if not cfg.kernel_src_path:
        raise ValueError("kernel_src_path is required.")

    kernel_raw = _read_file(cfg.kernel_src_path)
    kernel_code = _prepare_kernel_code(kernel_raw)

    if cfg.ref_origin != "kernelbench":
        raise ValueError("ref_origin must be 'kernelbench'.")
    if cfg.level is None or cfg.problem_id is None:
        raise ValueError("level and problem_id are required for ref_origin=kernelbench.")

    result = await evaluate_kernel_async(
        level=cfg.level,
        problem_id=cfg.problem_id,
        backend=cfg.backend,
        kernel_code=kernel_code,
        dataset_src=cfg.dataset_src,
        num_correct_trials=cfg.num_correct_trials,
        measure_performance=cfg.measure_performance,
        num_perf_trials=cfg.num_perf_trials,
        timing_method=cfg.timing_method,
        precision=cfg.precision,
        check_for_excessive_speedup=cfg.check_for_excessive_speedup,
        excessive_speedup_threshold=cfg.excessive_speedup_threshold,
        timeout=cfg.modal_timeout,
    )

    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
