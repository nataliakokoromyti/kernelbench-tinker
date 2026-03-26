"""
Modal App for Isolated GPU Kernel Evaluation
=============================================

This module runs INSIDE Modal containers (cloud GPUs). It defines:
1. The Docker image specification (CUDA + KernelBench)
2. The KernelEvaluator class with an evaluate() method
3. Helper functions for GPU architecture mapping

DEPLOYMENT:
    modal deploy src/kernelbench_tinker/modal/app.py

USAGE (from client code):
    cls = modal.Cls.from_name("kernel-rl-evaluator", "KernelEvaluator")
    result = cls().evaluate.remote(ref_code, kernel_code, ...)

We use Modal and treat RL evaluator as a deployed app because:
    - Isolation: Bad kernels can corrupt GPU memory; containers are disposable
    - Scalability & Parallelism: Up to 32 containers evaluate kernels simultaneously at a low cost
    - Handle Time outs: Modal handles this
"""

from __future__ import annotations

import os
import re
from typing import Any

import modal

# =============================================================================
# GPU Architecture Mapping
# =============================================================================
# Maps Modal GPU types to architecture names for Triton/CUDA compilation.
# These are passed to kernelbench.utils.set_gpu_arch() to configure the compiler.

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


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_GPU = "A100"
DEFAULT_TIMEOUT = 120  # seconds per kernel (configurable via rl_kernelbench.yaml)


# =============================================================================
# KernelBench Installation
# =============================================================================
# We install KernelBench from a pinned git commit for reproducibility.
# Override via KERNELBENCH_GIT_SPEC environment variable if needed.
#
# The [gpu] extra includes: triton, nvidia-cutlass-dsl, tilelang, cupy

KERNELBENCH_GIT_SPEC = os.environ.get(
    "KERNELBENCH_GIT_SPEC",
    # Install latest from main branch. Override env var if you need a specific version.
    "kernelbench[gpu] @ git+https://github.com/ScalingIntelligence/KernelBench.git@main",
)


# =============================================================================
# Modal Image Definition
# =============================================================================
# The image is based on nvidia/cuda with Python 3.10.
# We only pip_install kernelbench - it declares all deps in pyproject.toml.
# This avoids duplicating the dependency list and ensures version consistency.
#
# IMPORTANT: Keep cuda_version, flavor, operating_sys consistent with KernelBench's
# scripts/generate_and_eval_single_sample_modal.py for compatibility.

cuda_version = "13.0.0"
flavor = "devel"  # "devel" includes full CUDA toolkit for kernel compilation
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git",      # For pip installing from git
        "gcc-10",   # C compiler for CUDA extensions
        "g++-10",   # C++ compiler for CUDA extensions
        "clang",    # Alternative compiler (some kernels need it)
    )
    .pip_install(KERNELBENCH_GIT_SPEC)
)

# NOTE: You can customize this image for your needs:
#   image = image.pip_install("your-package")
#   image = image.run_commands("custom setup")


# =============================================================================
# Modal App Registration
# =============================================================================
# This app name is used when deploying and when connecting from client code.
# To deploy: modal deploy src/kernelbench_tinker/modal/app.py
# To connect: modal.Cls.from_name("kernel-rl-evaluator", "KernelEvaluator")

app = modal.App("kernel-rl-evaluator")


# =============================================================================
# Kernel Evaluator Class
# =============================================================================

@app.cls(
    image=image,
    gpu="A100",                     # Default GPU (can override at call time)
    timeout=DEFAULT_TIMEOUT,        # Kill container if eval takes too long
    max_containers=32,              # Max parallel containers to control costs
    retries=modal.Retries(          # Retry on transient failures
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
    min_containers=0,               # Scale to zero when idle
)
class KernelEvaluator:
    """
    Modal class for evaluating GPU kernels in isolated containers.
    
    Each container:
    - Has its own GPU (A100 by default)
    - Runs one kernel evaluation at a time
    - Is disposable (GPU memory corruption = container killed & replaced)
    """

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
        timing_method: str,
        check_for_excessive_speedup: bool,
        excessive_speedup_threshold: float,
    ) -> dict[str, Any]:
        """
        Evaluate a kernel against its reference PyTorch implementation.

        This method runs INSIDE the Modal container with GPU access.
        It calls kernelbench.eval.eval_kernel_against_ref() and returns
        a standardized result dict.

        Args:
            ref_code: Reference PyTorch implementation (Model class)
            kernel_code: Generated kernel code to evaluate (ModelNew class)
            backend: Kernel backend ("triton", "cuda", "cute", "tilelang")
            num_correct_trials: Number of correctness test runs
            measure_performance: Whether to measure runtime
            num_perf_trials: Number of performance measurement runs
            gpu_arch: GPU architecture list (e.g., ["Ampere"])
            precision: Data precision ("fp32", "fp16", "bf16")
            timing_method: How to measure time ("cuda_event" recommended)
            check_for_excessive_speedup: Flag unrealistic speedups as errors
            excessive_speedup_threshold: Speedup threshold for flagging

        Returns:
            Dict with keys:
                - format_ok: bool (always True, format checked before Modal)
                - compiled: bool (kernel compiled successfully)
                - correctness: bool (all correctness tests passed)
                - tests_passed: int (number of passing tests)
                - tests_total: int (total number of tests)
                - speedup: float | None (baseline_time / kernel_time)
                - runtime_ms: float | None (kernel execution time)
                - baseline_runtime_ms: float | None (reference execution time)
                - error_message: str | None (error details if any)
                - code_length: int (character count, not tokens)
                - metadata: dict (additional info from KernelBench)
        """
        import tempfile
        import time

        import modal.experimental
        import torch

        from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
        from kernelbench.utils import set_gpu_arch

        # ---------------------------------------------------------------------
        # Step 1: Wait for GPU to be available
        # ---------------------------------------------------------------------
        # Modal containers sometimes start before the GPU is fully attached.
        # We poll torch.cuda.is_available() with exponential backoff.
        
        max_wait_time = 30  # seconds
        start_time = time.time()
        gpu_available = False

        while time.time() - start_time < max_wait_time:
            if torch.cuda.is_available():
                gpu_available = True
                break
            # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s (capped)
            wait_time = min(0.5 * (2 ** int((time.time() - start_time) / 2)), 8.0)
            time.sleep(wait_time)

        if not gpu_available:
            # Raise error so Modal retries with a fresh container
            raise RuntimeError(
                f"GPU not attached to container after {max_wait_time}s - Modal will retry"
            )

        # ---------------------------------------------------------------------
        # Step 2: Configure GPU architecture for kernel compilation
        # ---------------------------------------------------------------------
        # This sets TORCH_CUDA_ARCH_LIST so Triton/CUDA compile for the right GPU.
        
        set_gpu_arch(gpu_arch)

        try:
            # -----------------------------------------------------------------
            # Step 3: Isolate torch extension builds
            # -----------------------------------------------------------------
            # When 32 containers compile kernels simultaneously, they can have
            # lock contention if they share the same TORCH_EXTENSIONS_DIR.
            # Each container gets its own temp directory for compiled binaries.
            
            build_dir = tempfile.mkdtemp(prefix="torch_ext_")
            os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

            # -----------------------------------------------------------------
            # Step 4: Call KernelBench evaluation
            # -----------------------------------------------------------------
            # This is the core: compile the kernel, run correctness tests,
            # measure performance if requested.
            
            eval_kwargs = {
                "original_model_src": ref_code,
                "custom_model_src": kernel_code,
                "measure_performance": measure_performance,
                "verbose": False,
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "backend": backend,
                "precision": get_torch_dtype_from_string(precision),
                "timing_method": timing_method,
                "check_for_excessive_speedup": check_for_excessive_speedup,
                "excessive_speedup_threshold": excessive_speedup_threshold,
                "build_dir": build_dir,
            }
            # -----------------------------------------------------------------
            # Call eval_kernel_against_ref (core KernelBench function)
            # -----------------------------------------------------------------
            # This function:
            # 1. Compiles the custom kernel code (may fail → compiled=False)
            # 2. Runs correctness trials comparing outputs to reference
            # 3. If measure_performance=True, times both implementations
            #
            # Returns: KernelExecResult with fields:
            #   - compiled: bool
            #   - correctness: bool (all trials passed)
            #   - runtime: float (ms, -1 if not measured)
            #   - metadata: dict with detailed info
            #
            # Can return None on lock file errors (handled below)
            # Exceptions are caught by outer try/except blocks
            # -----------------------------------------------------------------
            result = eval_kernel_against_ref(**eval_kwargs)
            
            # Free GPU memory for next eval (operates at PyTorch level, not CUDA driver)
            torch.cuda.empty_cache()

            # -----------------------------------------------------------------
            # Step 5: Handle None result (lock file error)
            # -----------------------------------------------------------------
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
                    "error_message": "Evaluation returned None (possible lock file error)",
                    "code_length": len(kernel_code),
                    "metadata": {},
                }

            # -----------------------------------------------------------------
            # Step 6: Parse correctness results
            # -----------------------------------------------------------------
            # KernelBench returns correctness_trials as "(3 / 5)" string
            
            tests_passed = 0
            trials_str = result.metadata.get("correctness_trials", "(0 / 0)")
            match = re.match(r"\((\d+)\s*/\s*(\d+)\)", trials_str)
            if match:
                tests_passed = int(match.group(1))

            # -----------------------------------------------------------------
            # Step 7: Calculate speedup
            # -----------------------------------------------------------------
            # speedup = baseline_runtime / kernel_runtime
            # Only calculated for correct kernels with valid timing data.

            runtime_ms = result.runtime if result.runtime > 0 else None
            baseline_runtime_ms = None

            ref_runtime = getattr(result, "ref_runtime", None)
            if ref_runtime and ref_runtime > 0:
                baseline_runtime_ms = ref_runtime
            elif measure_performance and result.correctness and runtime_ms is not None:
                try:
                    from kernelbench.timing import measure_ref_program_time

                    baseline_stats = measure_ref_program_time(
                        ref_arch_name="baseline",
                        ref_arch_src=ref_code,
                        num_warmup=5,
                        num_trials=num_perf_trials,
                        discard_first=1,
                        timing_method=timing_method,
                        precision=precision,
                        verbose=False,
                    )
                    if baseline_stats:
                        baseline_runtime_ms = baseline_stats.get("mean")
                except Exception:
                    pass

            speedup = None
            if result.correctness and runtime_ms and baseline_runtime_ms and baseline_runtime_ms > 0:
                speedup = baseline_runtime_ms / runtime_ms

            # -----------------------------------------------------------------
            # Step 8: Extract error messages
            # -----------------------------------------------------------------
            error_message = None
            for key in ["runtime_error", "compilation_error", "correctness_issue"]:
                if key in result.metadata:
                    error_message = str(result.metadata[key])
                    break

            # -----------------------------------------------------------------
            # Step 9: Return standardized result
            # -----------------------------------------------------------------
            return {
                "format_ok": True,
                "compiled": result.compiled,
                "correctness": result.correctness,
                "tests_passed": tests_passed,
                "tests_total": num_correct_trials,
                "speedup": speedup,
                "runtime_ms": runtime_ms,
                "baseline_runtime_ms": baseline_runtime_ms,
                "error_message": error_message,
                "code_length": len(kernel_code),
                "metadata": dict(result.metadata),
            }

        except torch.cuda.CudaError as e:
            # -----------------------------------------------------------------
            # GPU Memory Corruption Handler
            # -----------------------------------------------------------------
            # Bad kernels can corrupt GPU memory. When this happens:
            # 1. Stop accepting new work on this container
            # 2. Modal will kill this container and spin up a fresh one
            # 3. Return error result so training can continue
            #
            # See: https://modal.com/docs/guide/lifecycle-methods#stop_fetching_inputs
            # Check related KernelBench PR: 
            # NOTE: Simon to double check and migrate such changes
            modal.experimental.stop_fetching_inputs()
            return {
                "format_ok": True,
                "compiled": False,
                "correctness": False,
                "tests_passed": 0,
                "tests_total": num_correct_trials,
                "speedup": None,
                "runtime_ms": None,
                "baseline_runtime_ms": None,
                "error_message": f"CUDA error (container will be replaced): {str(e)}",
                "code_length": len(kernel_code),
                "metadata": {"gpu_error": type(e).__name__, "error_message": str(e)[:500]},
            }
            
        except Exception as e:
            # -----------------------------------------------------------------
            # General Exception Handler
            # -----------------------------------------------------------------
            # Catch-all for other errors. Return error result instead of crashing.
            
            return {
                "format_ok": True,
                "compiled": False,
                "correctness": False,
                "tests_passed": 0,
                "tests_total": num_correct_trials,
                "speedup": None,
                "runtime_ms": None,
                "baseline_runtime_ms": None,
                "error_message": f"Evaluation failed: {str(e)}",
                "code_length": len(kernel_code),
                "metadata": {"exception": str(e), "exception_type": type(e).__name__},
            }


# =============================================================================
# Helper Functions
# =============================================================================

def get_gpu_arch(gpu_type: str) -> list[str]:
    """
    Get GPU architecture list for a given GPU type.
    
    Args:
        gpu_type: Modal GPU type (e.g., "A100", "H100", "L40S")
        
    Returns:
        Architecture list for set_gpu_arch() (e.g., ["Ampere"])
    """
    return GPU_ARCH_MAPPING.get(gpu_type, ["Ampere"])
