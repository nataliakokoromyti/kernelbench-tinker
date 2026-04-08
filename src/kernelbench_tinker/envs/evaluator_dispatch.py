"""
Evaluator backend dispatch.

Lets the rest of kernelbench-tinker stay agnostic to whether kernels are
evaluated on Modal (cloud, NVIDIA only) or in a local subprocess (works
with AMD ROCm via PR #135's HIP backend).

Selection happens once per process via `set_evaluator_from_eval_config()`,
which inspects `eval_config.evaluator_backend` ("modal" or "local") and
installs the corresponding global evaluator. After that, every call site
fetches the current evaluator with `get_current_evaluator()` — which
returns whichever backend was registered.

Both backend evaluator classes (ModalKernelEvaluator, LocalKernelEvaluator)
expose the same async surface: `evaluate_single`, `evaluate_batch`,
`evaluate_single_batched`. So callers don't need conditionals.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# Module-level state. Tracks which backend was last registered so callers
# that need to know (e.g. for logging) can ask, but normally callers should
# just use get_current_evaluator() and treat the result as opaque.
_current_backend: str = "modal"


class _EvaluatorLike(Protocol):
    """Structural type both ModalKernelEvaluator and LocalKernelEvaluator satisfy."""

    async def evaluate_single(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    async def evaluate_batch(self, evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
    async def evaluate_single_batched(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


def set_evaluator_from_eval_config(eval_config: Any) -> _EvaluatorLike:
    """
    Install the appropriate global evaluator based on `eval_config.evaluator_backend`.

    Accepts either the dataclass `EvalConfig` from
    `kernelbench_tinker.config.configs` or the chz `EvalConfig` from
    `kernelbench_tinker.scripts.eval_kernel_rl` — they share the relevant
    field names so duck typing is fine.

    Returns the registered evaluator instance.
    """
    global _current_backend

    backend = getattr(eval_config, "evaluator_backend", "modal") or "modal"
    backend = backend.lower()

    if backend == "local":
        from kernelbench_tinker.local.evaluator import (
            LocalEvaluatorConfig,
            LocalKernelEvaluator,
            set_local_evaluator,
        )

        gpu_arch = list(getattr(eval_config, "gpu_arch", []) or [])
        timeout = int(getattr(eval_config, "local_timeout", 300))
        local_cfg = LocalEvaluatorConfig(
            enabled=True,
            gpu_arch=gpu_arch,
            timeout=timeout,
        )
        evaluator = LocalKernelEvaluator(local_cfg)
        set_local_evaluator(evaluator)
        _current_backend = "local"
        logger.info(
            "Local evaluator configured: gpu_arch=%s, timeout=%ds",
            gpu_arch, timeout,
        )
        return evaluator

    if backend == "modal":
        from kernelbench_tinker.modal.evaluator import (
            ModalEvaluatorConfig,
            ModalKernelEvaluator,
            set_modal_evaluator,
        )

        modal_cfg = ModalEvaluatorConfig(
            enabled=True,
            gpu_type=getattr(eval_config, "modal_gpu_type", "A100"),
            timeout=int(getattr(eval_config, "modal_timeout", 120.0)),
        )
        evaluator = ModalKernelEvaluator(modal_cfg)
        set_modal_evaluator(evaluator)
        _current_backend = "modal"
        logger.info(
            "Modal evaluator configured: gpu_type=%s, timeout=%ds",
            modal_cfg.gpu_type, modal_cfg.timeout,
        )
        return evaluator

    raise ValueError(
        f"Unknown evaluator_backend={backend!r}. Expected 'modal' or 'local'."
    )


def get_current_evaluator() -> _EvaluatorLike:
    """
    Return whichever evaluator is currently registered as the global.

    Defaults to the Modal evaluator if no explicit selection has been made,
    preserving the historical behavior of the project.
    """
    if _current_backend == "local":
        from kernelbench_tinker.local.evaluator import get_local_evaluator
        return get_local_evaluator()

    from kernelbench_tinker.modal.evaluator import get_modal_evaluator
    return get_modal_evaluator()


def get_current_backend() -> str:
    """Return the name of the currently registered backend ('modal' or 'local')."""
    return _current_backend
