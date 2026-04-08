"""Local in-process kernel evaluator (subprocess-isolated, AMD ROCm friendly)."""

from kernelbench_tinker.local.evaluator import (
    LocalEvaluatorConfig,
    LocalKernelEvaluator,
    get_local_evaluator,
    set_local_evaluator,
)

__all__ = [
    "LocalEvaluatorConfig",
    "LocalKernelEvaluator",
    "get_local_evaluator",
    "set_local_evaluator",
]
