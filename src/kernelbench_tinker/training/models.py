"""
Model and completer helpers for KernelBench ↔ Tinker integration.
"""

from __future__ import annotations

import tinker
from tinker_cookbook.completers import (
    StopCondition,
    TokenCompleter,
    TokensWithLogprobs,
)

# ---------------------------------------------------------------------------
# Renderer helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------


def get_adam_params(
    learning_rate: float,
    max_grad_norm: float = 0.0,
) -> tinker.AdamParams:
    """Get Adam optimizer parameters."""
    kwargs: dict = {
        "learning_rate": learning_rate,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
    }
    if max_grad_norm > 0:
        kwargs["grad_clip_norm"] = max_grad_norm
    return tinker.AdamParams(**kwargs)


# ---------------------------------------------------------------------------
# Token completers
# ---------------------------------------------------------------------------


class KernelBenchTokenCompleter(TokenCompleter):
    """Token completer with top_p and seed support.

    TinkerTokenCompleter only accepts temperature. This subclass adds top_p
    and seed, which the multi-turn training loop and eval script need.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ):
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            ),
        )
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None
        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


# ---------------------------------------------------------------------------
# Loss function helpers
# ---------------------------------------------------------------------------


def build_loss_fn_config(
    clip_epsilon_low: float = 0.0,
    clip_epsilon_high: float = 0.0,
) -> dict[str, float] | None:
    """Build loss_fn_config for PPO clip thresholds (passed to forward_backward_async)."""
    if clip_epsilon_low <= 0:
        return None
    return {
        "clip_low_threshold": 1.0 - clip_epsilon_low,
        "clip_high_threshold": 1.0 + clip_epsilon_high,
    }
