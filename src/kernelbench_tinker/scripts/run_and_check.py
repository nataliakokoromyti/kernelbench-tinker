#!/usr/bin/env python3
"""
Minimal run-and-check helper using KernelBench eval utilities.

Supports:
- ref_origin=kernelbench (level/problem_id) for local or Modal eval
- ref_origin=local (ref_arch_src_path) for local eval only
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import chz
import torch

from kernelbench_tinker.envs.kernelbench_client import (
    evaluate_kernel,
    evaluate_kernel_async,
    extract_code_block,
    get_reference_code,
    _ensure_kernelbench_imported,
)


@chz.chz
class RunAndCheckConfig:
    # Reference origin
    ref_origin: str = "kernelbench"  # "kernelbench" or "local"
    ref_arch_src_path: str | None = None
    level: int | None = None
    problem_id: int | None = None
    dataset_src: str = "huggingface"

    # Kernel source
    kernel_src_path: str = ""
    backend: str = "triton"

    # Eval settings
    eval_mode: str = "local"  # "local" or "modal"
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


def _local_eval_with_ref(
    ref_code: str,
    kernel_code: str,
    cfg: RunAndCheckConfig,
) -> dict:
    _ensure_kernelbench_imported()
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for local evaluation.")

    result = eval_kernel_against_ref(
        original_model_src=ref_code,
        custom_model_src=kernel_code,
        measure_performance=cfg.measure_performance,
        timing_method=cfg.timing_method,
        verbose=False,
        num_correct_trials=cfg.num_correct_trials,
        num_perf_trials=cfg.num_perf_trials,
        build_dir=None,
        device=torch.device("cuda:0"),
        backend=cfg.backend,
        precision=get_torch_dtype_from_string(cfg.precision),
        check_for_excessive_speedup=cfg.check_for_excessive_speedup,
        excessive_speedup_threshold=cfg.excessive_speedup_threshold,
    )

    runtime_ms = result.runtime if result.runtime > 0 else None
    baseline_runtime_ms = (
        result.metadata.get("baseline_runtime_ms")
        or result.metadata.get("baseline_runtime")
        or None
    )
    speedup = None
    if (
        result.correctness
        and runtime_ms is not None
        and baseline_runtime_ms
        and baseline_runtime_ms > 0
    ):
        speedup = baseline_runtime_ms / runtime_ms

    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "runtime_ms": runtime_ms,
        "baseline_runtime_ms": baseline_runtime_ms,
        "speedup": speedup,
        "metadata": result.metadata,
    }


async def _run() -> None:
    cfg = chz.entrypoint(RunAndCheckConfig)

    if not cfg.kernel_src_path:
        raise ValueError("kernel_src_path is required.")

    kernel_raw = _read_file(cfg.kernel_src_path)
    kernel_code = _prepare_kernel_code(kernel_raw)

    if cfg.ref_origin == "local":
        if cfg.eval_mode != "local":
            raise ValueError("ref_origin=local only supports eval_mode=local.")
        if not cfg.ref_arch_src_path:
            raise ValueError("ref_arch_src_path is required for ref_origin=local.")
        ref_code = _read_file(cfg.ref_arch_src_path)
        result = _local_eval_with_ref(ref_code, kernel_code, cfg)
        print(json.dumps(result, indent=2, default=str))
        return

    if cfg.ref_origin != "kernelbench":
        raise ValueError("ref_origin must be 'kernelbench' or 'local'.")
    if cfg.level is None or cfg.problem_id is None:
        raise ValueError("level and problem_id are required for ref_origin=kernelbench.")

    if cfg.eval_mode == "modal":
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
    else:
        ref_code = get_reference_code(cfg.level, cfg.problem_id, cfg.dataset_src)
        result = _local_eval_with_ref(ref_code, kernel_code, cfg)

    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
