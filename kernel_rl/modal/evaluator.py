"""
Modal-based kernel evaluator with batch processing support.

This module provides the high-level interface for evaluating kernels
using Modal Labs' isolated GPU containers.

Usage:
    Before running evaluations, deploy the Modal app:
        modal deploy kernel_rl/modal/app.py

    Then evaluations will use the deployed functions via Cls.from_name().
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel_rl.envs.kernelbench_client import KernelEvalResult

logger = logging.getLogger(__name__)

# App name must match the one in app.py
MODAL_APP_NAME = "kernel-rl-evaluator"
MODAL_CLS_NAME = "KernelEvaluator"


@dataclass
class ModalEvaluatorConfig:
    """Configuration for Modal-based kernel evaluation."""

    # Modal configuration
    enabled: bool = True
    gpu_type: str = "A100"  # A100, H100, L40S, T4, etc.
    timeout: int = 120  # seconds per kernel (matches Modal app default)

    # Batch configuration
    max_batch_size: int = 32  # Max kernels to evaluate in parallel
    return_exceptions: bool = True  # Return exceptions as results
    batch_single: bool = True  # Coalesce single evals into batches
    batch_window_s: float = 0.15  # Coalesce window for single-eval batching

    # API configuration
    modal_token_env: str = "MODAL_TOKEN_ID"  # Environment variable for Modal token

    # Deployment mode: "deployed" uses pre-deployed app, "ephemeral" spins up on-demand
    # "deployed" is recommended for production (faster, no startup overhead)
    deployment_mode: str = "deployed"


class ModalKernelEvaluator:
    """
    Kernel evaluator using Modal Labs for isolated GPU execution.

    Provides:
    - Hard timeout enforcement (kills bad kernels)
    - Process isolation (each kernel in separate container)
    - Parallel batch evaluation across multiple GPUs

    The evaluator can work in two modes:
    1. "deployed" (default): Uses a pre-deployed Modal app via Cls.from_name()
       - Requires running `modal deploy kernel_rl/modal/app.py` first
       - Faster startup, no cold boot overhead
       - Recommended for production

    2. "ephemeral": Spins up Modal app on-demand with app.run()
       - No deployment required
       - Has cold boot overhead
       - Only supports single concurrent evaluation
    """

    def __init__(self, config: ModalEvaluatorConfig | None = None):
        self.config = config or ModalEvaluatorConfig()
        self._modal_available: bool | None = None
        self._deployed_cls = None
        self._gpu_arch: list[str] | None = None
        self._lock = threading.Lock()
        self._batcher: ModalBatcher | None = None

    def _check_modal_available(self) -> bool:
        """Check if Modal is available and configured."""
        if self._modal_available is not None:
            return self._modal_available

        try:
            import modal  # noqa: F401

            # Check if we're in a context where Modal can run
            # Modal requires authentication tokens
            token_id = os.environ.get(self.config.modal_token_env)
            token_secret = os.environ.get("MODAL_TOKEN_SECRET")

            if not token_id or not token_secret:
                logger.warning(
                    f"Modal tokens not found in environment. "
                    f"Set {self.config.modal_token_env} and MODAL_TOKEN_SECRET"
                )
                self._modal_available = False
            else:
                self._modal_available = True
        except ImportError:
            logger.error("Modal not installed. Run: pip install modal")
            self._modal_available = False

        return self._modal_available

    def _get_deployed_evaluator(self):
        """Get reference to the deployed Modal evaluator class."""
        if self._deployed_cls is not None:
            return self._deployed_cls

        import modal

        from kernel_rl.modal.app import get_gpu_arch

        # Look up the deployed class by name
        try:
            self._deployed_cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLS_NAME)
            self._gpu_arch = get_gpu_arch(self.config.gpu_type)
            logger.info(f"Connected to deployed Modal app: {MODAL_APP_NAME}/{MODAL_CLS_NAME}")
        except modal.exception.NotFoundError:
            raise RuntimeError(
                f"Modal app '{MODAL_APP_NAME}' not found. "
                f"Please deploy first: modal deploy kernel_rl/modal/app.py"
            )

        return self._deployed_cls

    async def evaluate_single(
        self,
        ref_code: str,
        kernel_code: str,
        backend: str = "triton",
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        num_perf_trials: int = 100,
        precision: str = "fp32",
    ) -> dict[str, Any]:
        """
        Evaluate a single kernel using Modal.

        Args:
            ref_code: Reference PyTorch implementation
            kernel_code: Generated kernel code
            backend: Backend type ("triton", "cuda", etc.)
            num_correct_trials: Number of correctness trials
            measure_performance: Whether to measure runtime
            num_perf_trials: Number of performance trials
            precision: Precision string

        Returns:
            KernelEvalResult dict

        Raises:
            RuntimeError: If Modal is not available
        """
        if not self.config.enabled:
            raise RuntimeError("Modal evaluation is disabled")

        if not self._check_modal_available():
            raise RuntimeError(
                "Modal not available. Ensure Modal is installed and tokens are configured."
            )

        # Run Modal evaluation in thread pool (Modal SDK is synchronous)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._evaluate_single_sync,
            ref_code,
            kernel_code,
            backend,
            num_correct_trials,
            measure_performance,
            num_perf_trials,
            precision,
        )
        return result

    def _evaluate_single_sync(
        self,
        ref_code: str,
        kernel_code: str,
        backend: str,
        num_correct_trials: int,
        measure_performance: bool,
        num_perf_trials: int,
        precision: str,
    ) -> dict[str, Any]:
        """Synchronous single kernel evaluation using deployed Modal app."""
        evaluator_cls = self._get_deployed_evaluator()
        evaluator = evaluator_cls.with_options(
            gpu=self.config.gpu_type, timeout=self.config.timeout
        )()

        # Call the deployed function - no app.run() needed
        result = evaluator.evaluate.remote(
            ref_code=ref_code,
            kernel_code=kernel_code,
            backend=backend,
            num_correct_trials=num_correct_trials,
            measure_performance=measure_performance,
            num_perf_trials=num_perf_trials,
            gpu_arch=self._gpu_arch,
            precision=precision,
        )

        return result

    async def evaluate_batch(
        self,
        evaluations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Evaluate multiple kernels in parallel using Modal Function.starmap().

        Each dict in evaluations should have:
        - ref_code: str
        - kernel_code: str
        - backend: str (optional, default "triton")
        - num_correct_trials: int (optional, default 5)
        - measure_performance: bool (optional, default False)
        - num_perf_trials: int (optional, default 100)
        - precision: str (optional, default "fp32")

        Args:
            evaluations: List of evaluation parameter dicts

        Returns:
            List of KernelEvalResult dicts in same order as input

        Raises:
            RuntimeError: If Modal is not available
        """
        if not evaluations:
            return []

        if not self.config.enabled:
            raise RuntimeError("Modal evaluation is disabled")

        if not self._check_modal_available():
            raise RuntimeError(
                "Modal not available. Ensure Modal is installed and tokens are configured."
            )

        # Run Modal batch evaluation in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._evaluate_batch_sync,
            evaluations,
        )
        return results

    def _evaluate_batch_sync(
        self, evaluations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Synchronous batch evaluation using starmap with deployed Modal app."""
        evaluator_cls = self._get_deployed_evaluator()

        # Prepare arguments for starmap
        # Each tuple is (ref_code, kernel_code, backend, num_correct_trials,
        #                measure_performance, num_perf_trials, gpu_arch, precision)
        args = [
            (
                e["ref_code"],
                e["kernel_code"],
                e.get("backend", "triton"),
                e.get("num_correct_trials", 5),
                e.get("measure_performance", False),
                e.get("num_perf_trials", 100),
                self._gpu_arch,
                e.get("precision", "fp32"),
            )
            for e in evaluations
        ]

        # Call the deployed function - no app.run() needed
        evaluator = evaluator_cls.with_options(
            gpu=self.config.gpu_type, timeout=self.config.timeout
        )()
        results = list(
            evaluator.evaluate.starmap(
                args,
                return_exceptions=self.config.return_exceptions,
            )
        )

        # Convert exceptions to error results
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(
                    {
                        "format_ok": True,
                        "compiled": False,
                        "correctness": False,
                        "tests_passed": 0,
                        "tests_total": evaluations[i].get("num_correct_trials", 5),
                        "speedup": None,
                        "runtime_ms": None,
                        "baseline_runtime_ms": None,
                        "cheated": False,
                        "error_message": f"Modal execution failed: {str(result)}",
                        "code_length": len(evaluations[i]["kernel_code"]),
                        "metadata": {
                            "exception": str(result),
                            "exception_type": type(result).__name__,
                        },
                    }
                )
            else:
                processed.append(result)

        return processed

    async def evaluate_single_batched(
        self,
        ref_code: str,
        kernel_code: str,
        backend: str = "triton",
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        num_perf_trials: int = 100,
        precision: str = "fp32",
    ) -> dict[str, Any]:
        """
        Coalesce multiple single-eval requests into Modal batch calls.

        This keeps API compatibility with evaluate_single while leveraging
        starmap parallelism to reduce per-request overhead.
        """
        if not self.config.batch_single:
            return await self.evaluate_single(
                ref_code,
                kernel_code,
                backend,
                num_correct_trials,
                measure_performance,
                num_perf_trials,
                precision,
            )

        # Lazily create batcher to avoid overhead when unused
        if self._batcher is None:
            self._batcher = ModalBatcher(self)

        request = {
            "ref_code": ref_code,
            "kernel_code": kernel_code,
            "backend": backend,
            "num_correct_trials": num_correct_trials,
            "measure_performance": measure_performance,
            "num_perf_trials": num_perf_trials,
            "precision": precision,
        }
        return await self._batcher.submit(request)


# Global evaluator instance (lazy initialized)
_global_evaluator: ModalKernelEvaluator | None = None


def get_modal_evaluator(
    config: ModalEvaluatorConfig | None = None,
) -> ModalKernelEvaluator:
    """Get or create global Modal evaluator instance."""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = ModalKernelEvaluator(config)
    elif config is not None:
        # Update config if provided after initialization
        _global_evaluator.config = config
    return _global_evaluator


def set_modal_evaluator(evaluator: ModalKernelEvaluator) -> None:
    """Set the global Modal evaluator instance."""
    global _global_evaluator
    _global_evaluator = evaluator


# Convenience functions for direct use
async def evaluate_kernel_single(
    ref_code: str,
    kernel_code: str,
    backend: str = "triton",
    **kwargs,
) -> dict[str, Any]:
    """Evaluate a single kernel using Modal."""
    evaluator = get_modal_evaluator()
    return await evaluator.evaluate_single(ref_code, kernel_code, backend, **kwargs)


async def evaluate_kernel_batch(
    evaluations: list[dict[str, Any]],
    config: ModalEvaluatorConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate multiple kernels in parallel using Modal."""
    evaluator = get_modal_evaluator(config)
    return await evaluator.evaluate_batch(evaluations)


class ModalBatcher:
    """
    Coalesces single-eval requests into batch RPCs to Modal.

    Requests submitted within a short window are grouped and dispatched
    via evaluate_batch for better throughput and lower latency.
    """

    def __init__(self, evaluator: ModalKernelEvaluator):
        self.evaluator = evaluator
        self._pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._pending.append((request, fut))
            should_flush_now = len(self._pending) >= self.evaluator.config.max_batch_size
            if should_flush_now:
                await self._flush_locked()
            elif self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_after_delay())
        return await fut

    async def _flush_after_delay(self):
        await asyncio.sleep(self.evaluator.config.batch_window_s)
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self):
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        self._flush_task = None
        requests = [req for req, _ in batch]

        try:
            results = await self.evaluator.evaluate_batch(requests)
        except Exception as exc:  # pragma: no cover - defensive
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return

        for result, (_, fut) in zip(results, batch, strict=True):
            if not fut.done():
                fut.set_result(result)
