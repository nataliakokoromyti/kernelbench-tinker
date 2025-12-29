"""
Modal App definition for isolated GPU kernel evaluation.

This module defines the Modal App, GPU image, and evaluation function
that runs inside the isolated Modal container.
"""

from __future__ import annotations

import os
import re
from typing import Any

import modal

# GPU architecture mapping for Triton/CUDA compilation
GPU_ARCH_MAPPING = {
    "A100": ["Ampere"],
    "A100-40GB": ["Ampere"],
    "A100-80GB": ["Ampere"],
    "H100": ["Hopper"],
    "H200": ["Hopper"],
    "L40S": ["Ada"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"],
}

# Default configuration
DEFAULT_GPU = "A100"
DEFAULT_TIMEOUT = 120  # 2 minutes per kernel by default (tunable via config)

# Get KernelBench root path (for local development)
KERNELBENCH_ROOT = os.environ.get(
    "KERNELBENCH_ROOT",
    "/workspace/kernel_dev/KernelBench"
)

# Build the Modal image with CUDA and KernelBench dependencies
cuda_version = "12.8.0"
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
kernelbench_src = os.path.join(KERNELBENCH_ROOT, "src")

# Create image with CUDA, compilers, and Python dependencies
# Use KernelBench requirements.txt to get all dependencies
kernelbench_requirements = os.path.join(KERNELBENCH_ROOT, "requirements.txt")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "gcc-10",
        "g++-10",
        "clang",
    )
)
# Install requirements only if present to avoid failing when KERNELBENCH_ROOT is missing
if os.path.exists(kernelbench_requirements):
    image = image.pip_install_from_requirements(kernelbench_requirements)
# Mount KernelBench src directory (module name "src" from KernelBench)
if os.path.exists(kernelbench_src):
    image = image.add_local_python_source(kernelbench_src)
else:
    # Fallback: try to mount a local src if available in this repo
    if os.path.exists("src"):
        image = image.add_local_python_source("src")

# Create Modal App
app = modal.App("kernel-rl-evaluator")


@app.cls(
    image=image,
    gpu="A100",
    timeout=DEFAULT_TIMEOUT,
    concurrency_limit=32,
    keep_warm=0,
)
class KernelEvaluator:
    """Modal class for kernel evaluation with configurable GPU."""

    @modal.method()
    def evaluate(
        self,
        ref_code: str,
        kernel_code: str,
        backend: str,
        num_correct_trials: int,
        measure_performance: bool,
        num_perf_trials: int,
        gpu_arch: list[str],
        precision: str,
    ) -> dict[str, Any]:
        """
        Evaluate a single kernel in an isolated Modal container.

        Args:
            ref_code: Reference PyTorch implementation
            kernel_code: Generated kernel code to evaluate
            backend: Backend type ("triton", "cuda", "cute", "tilelang")
            num_correct_trials: Number of correctness trials
            measure_performance: Whether to measure runtime
            num_perf_trials: Number of performance trials
            gpu_arch: GPU architecture list for compilation
            precision: Precision string ("fp32", "fp16", "bf16")

        Returns:
            Dict matching KernelEvalResult structure
        """
        from src.eval import eval_kernel_against_ref, get_torch_dtype_from_string
        from src.utils import set_gpu_arch

        # Set GPU architecture for Triton/CUDA compilation
        set_gpu_arch(gpu_arch)

        try:
            result = eval_kernel_against_ref(
                original_model_src=ref_code,
                custom_model_src=kernel_code,
                measure_performance=measure_performance,
                verbose=False,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                backend=backend,
                precision=get_torch_dtype_from_string(precision),
            )

            if result is None:
                return {
                    "format_ok": True,
                    "compiled": False,
                    "correctness": False,
                    "tests_passed": 0,
                    "tests_total": num_correct_trials,
                    "speedup": None,
                    "runtime_ms": None,
                    "baseline_runtime_ms": None,
                    "cheated": False,
                    "error_message": "Evaluation returned None (lock file error)",
                    "code_length": len(kernel_code),
                    "metadata": {},
                }

            # Parse tests passed from metadata
            tests_passed = 0
            trials_str = result.metadata.get("correctness_trials", "(0 / 0)")
            match = re.match(r"\((\d+)\s*/\s*(\d+)\)", trials_str)
            if match:
                tests_passed = int(match.group(1))

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

            # Extract error message if any
            error_message = None
            for key in ["runtime_error", "compilation_error", "correctness_issue"]:
                if key in result.metadata:
                    error_message = str(result.metadata[key])
                    break

            return {
                "format_ok": True,
                "compiled": result.compiled,
                "correctness": result.correctness,
                "tests_passed": tests_passed,
                "tests_total": num_correct_trials,
                "speedup": speedup,
                "runtime_ms": runtime_ms,
                "baseline_runtime_ms": baseline_runtime_ms,
                "cheated": False,  # Checked by caller
                "error_message": error_message,
                "code_length": len(kernel_code),
                "metadata": dict(result.metadata),
            }

        except Exception as e:
            return {
                "format_ok": True,
                "compiled": False,
                "correctness": False,
                "tests_passed": 0,
                "tests_total": num_correct_trials,
                "speedup": None,
                "runtime_ms": None,
                "baseline_runtime_ms": None,
                "cheated": False,
                "error_message": f"Modal evaluation failed: {str(e)}",
                "code_length": len(kernel_code),
                "metadata": {"exception": str(e), "exception_type": type(e).__name__},
            }


def get_evaluator_with_gpu(gpu_type: str = DEFAULT_GPU, timeout: int = DEFAULT_TIMEOUT):
    """
    Get a KernelEvaluator instance configured for a specific GPU.

    Args:
        gpu_type: GPU type (A100, H100, L40S, etc.)
        timeout: Timeout in seconds per evaluation

    Returns:
        Configured KernelEvaluator class with GPU options
    """
    return KernelEvaluator.with_options(gpu=gpu_type, timeout=timeout)


def get_gpu_arch(gpu_type: str) -> list[str]:
    """Get GPU architecture list for a given GPU type."""
    return GPU_ARCH_MAPPING.get(gpu_type, ["Ampere"])
