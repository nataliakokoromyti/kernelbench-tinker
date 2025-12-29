"""
Minimal model helpers for KernelBench ↔ Tinker integration.
"""

from __future__ import annotations

import tinker


def get_renderer_name_for_model(model_name: str) -> str:
    """
    Get the appropriate renderer name for a model.

    Args:
        model_name: Full model name

    Returns:
        Renderer name (e.g., "qwen3", "llama3")
    """
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        return "qwen3"
    if "llama-3" in model_lower or "llama3" in model_lower:
        return "llama3"
    if "codellama" in model_lower:
        return "llama3"
    return "role_colon"


def get_adam_params(learning_rate: float) -> tinker.AdamParams:
    """Get Adam optimizer parameters."""
    return tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
