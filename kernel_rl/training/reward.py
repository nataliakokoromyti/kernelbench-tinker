"""
Reward shaping for KernelBench RL training.

This module implements reward functions that combine:
- Format correctness (valid code block extraction)
- Compilation success
- Correctness (passing all tests)
- Speed (optional, for later stages of training)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernel_rl.envs.kernelbench_client import KernelEvalResult


@dataclass
class RewardConfig:
    """
    Configuration for reward computation.

    Default values follow Kevin-32B paper (arXiv:2507.11948):
    - Reward = 0.3 * correct + speedup (if correct)
    - Zero reward for cheating (PyTorch wrapping, try-except, pass)
    - No length penalties (causes response collapse)
    - No thinking rewards (thinking removed from prompts)
    - Binary correctness (no partial credit)
    """

    # ==========================================================================
    # Kevin-compatible reward weights
    # Kevin formula: S = 0.3·𝟙{correct} + (T_baseline/T_kernel)·𝟙{correct}
    # ==========================================================================
    format_weight: float = 0.0  # Kevin doesn't use format reward
    compile_weight: float = 0.0  # Kevin doesn't use compile reward
    correctness_weight: float = 0.3  # Kevin uses 0.3 for binary correctness
    speed_weight: float = 1.0  # Kevin adds speedup directly
    length_weight: float = 0.0  # Kevin: length penalties cause response collapse
    thinking_weight: float = 0.0  # Kevin removes thinking from prompts

    # ==========================================================================
    # Cheating handling - Kevin uses ZERO reward, not penalties
    # ==========================================================================
    cheating_penalty: float = 0.0  # Not used - we return 0 early for cheating
    format_penalty: float = 0.0  # Zero reward for bad format

    # ==========================================================================
    # Speed reward configuration
    # Kevin uses linear speedup: reward = T_baseline / T_kernel
    # ==========================================================================
    speed_baseline: float = 1.0  # Speedup threshold (1.0 = same as baseline)
    speed_scale: float = 1.0  # Linear scaling (Kevin uses 1.0, not log)
    speed_max_reward: float = 10.0  # Cap to prevent outliers

    # ==========================================================================
    # Length reward configuration (DISABLED by default per Kevin)
    # Kevin paper: "length penalties caused response collapse within 10 steps"
    # ==========================================================================
    length_max: int = 8000  # Not used when length_weight=0
    length_min: int = 500  # Not used when length_weight=0

    # ==========================================================================
    # Thinking reward configuration (DISABLED by default per Kevin)
    # Kevin removes thinking from multi-turn prompts entirely
    # ==========================================================================
    thinking_bonus: float = 1.0  # Not used when thinking_weight=0
    thinking_no_think: float = 0.5  # Not used when thinking_weight=0
    thinking_soft_max: int = 2000  # Not used when thinking_weight=0
    thinking_hard_max: int = 4000  # Not used when thinking_weight=0
    thinking_floor: float = 0.7  # Not used when thinking_weight=0

    # ==========================================================================
    # Correctness configuration
    # Kevin uses binary correctness (all tests pass or not)
    # ==========================================================================
    sparse_rewards: bool = False
    partial_correctness: bool = False  # Kevin uses binary correctness


def format_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward for format correctness.

    Returns:
        1.0 if format is valid (code block with correct language tag)
        config.format_penalty if format is invalid
    """
    if eval_result["format_ok"]:
        return 1.0
    return config.format_penalty


def compile_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward for successful compilation.

    Returns:
        1.0 if compiled successfully and not cheating
        0.0 if compilation failed
        config.cheating_penalty if cheating detected
    """
    if eval_result["cheated"]:
        return config.cheating_penalty

    if eval_result["compiled"]:
        return 1.0

    return 0.0


def correctness_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward for correctness.

    With partial_correctness=True:
        Returns tests_passed / tests_total (0.0 to 1.0)

    With partial_correctness=False:
        Returns 1.0 if all tests pass, 0.0 otherwise
    """
    if not eval_result["compiled"]:
        return 0.0

    tests_passed = eval_result["tests_passed"]
    tests_total = eval_result["tests_total"]

    if tests_total == 0:
        return 0.0

    if config.partial_correctness:
        return tests_passed / tests_total
    else:
        return 1.0 if tests_passed == tests_total else 0.0


def speed_reward(
    eval_result: "KernelEvalResult",
    config: RewardConfig,
    use_speed: bool = True
) -> float:
    """
    Compute reward for speedup over baseline.

    Kevin-32B formula (arXiv:2507.11948):
        speed_reward = T_baseline / T_kernel = speedup

    Kevin uses LINEAR speedup directly, not log-scaled.
    With speed_scale=1.0 (default), this returns the raw speedup value.

    Only gives reward if:
    - use_speed is True
    - Kernel is fully correct
    - Speedup data is available

    Args:
        eval_result: Evaluation result
        config: Reward configuration
        use_speed: Whether to use speed rewards

    Returns:
        Speed reward (0.0 if not applicable)
    """
    if not use_speed:
        return 0.0

    # Only reward speed for fully correct kernels
    if not eval_result["correctness"]:
        return 0.0

    speedup = eval_result.get("speedup")
    if speedup is None or speedup <= 0:
        return 0.0

    # Kevin uses linear speedup, not log-scaled
    # If speedup <= baseline (1.0), no speed bonus
    if speedup <= config.speed_baseline:
        return 0.0

    # Linear reward: speedup - 1.0 (so 2x speedup = 1.0 reward, 3x = 2.0, etc.)
    # This matches Kevin's formula where reward = speedup for correct kernels
    reward = config.speed_scale * (speedup - config.speed_baseline)

    # Clamp to max to prevent outliers
    return min(reward, config.speed_max_reward)


def length_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward based on code length (GRPO-LEAD style tie-breaking).

    This provides a continuous signal that breaks ties when other rewards are uniform.
    Shorter code gets higher reward, encouraging concise solutions.

    The reward is linearly interpolated:
    - Code <= length_min: reward = 1.0
    - Code >= length_max: reward = 0.0
    - In between: linear interpolation

    Args:
        eval_result: Evaluation result containing code_length
        config: Reward configuration

    Returns:
        Length reward in [0, 1]
    """
    code_length = eval_result.get("code_length", 0)

    if code_length <= config.length_min:
        return 1.0
    elif code_length >= config.length_max:
        return 0.0
    else:
        # Linear interpolation
        range_size = config.length_max - config.length_min
        position = code_length - config.length_min
        return 1.0 - (position / range_size)


def thinking_reward(thought_length: int, config: RewardConfig) -> float:
    """
    Compute reward for using <think> blocks (encourages structured reasoning).

    Based on Kevin paper findings: harsh length penalties cause response collapse.
    This uses a generous reward curve that:
    - Gives bonus for having any thinking content
    - Doesn't penalize short thinking (some tasks are simple)
    - Only gently caps extremely long thinking (context preservation)
    - Never goes below a generous floor

    Reward curve:
    - No thinking (0 chars): thinking_no_think (0.5 default, neutral)
    - Any thinking (1 to soft_max): thinking_bonus (1.0 default, full reward)
    - Long thinking (soft_max to hard_max): linear decay to floor
    - Very long thinking (> hard_max): thinking_floor (0.7 default, still positive)

    Args:
        thought_length: Length of thinking content in characters
        config: Reward configuration

    Returns:
        Thinking reward in [thinking_floor, thinking_bonus]

    References:
        - Kevin-32B (arxiv:2507.11948): "Testing length penalties caused response
          length to collapse within 10 training steps, degrading performance"
        - Demystifying Long CoT (arxiv:2502.03373): Cosine length-scaling stabilizes
          CoT growth without harsh penalties
    """
    if thought_length == 0:
        # No thinking - neutral reward (don't penalize, some tasks don't need it)
        return config.thinking_no_think

    if thought_length <= config.thinking_soft_max:
        # Good thinking length - full bonus
        return config.thinking_bonus

    if thought_length >= config.thinking_hard_max:
        # Very long thinking - floor reward (still positive, generous)
        return config.thinking_floor

    # Linear decay between soft_max and hard_max
    range_size = config.thinking_hard_max - config.thinking_soft_max
    position = thought_length - config.thinking_soft_max
    decay = (config.thinking_bonus - config.thinking_floor) * (position / range_size)
    return config.thinking_bonus - decay


def compute_reward(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
    thought_length: int = 0,
) -> float:
    """
    Compute the total reward for a kernel evaluation.

    Kevin-32B formula (arXiv:2507.11948):
        S = 0.3·𝟙{correct} + (T_baseline/T_kernel)·𝟙{correct}

    Key behaviors:
    - Zero reward for cheating (PyTorch wrapping, try-except, pass)
    - Zero reward for incorrect kernels
    - Binary correctness (no partial credit by default)
    - Speedup added linearly (not log-scaled)

    Args:
        eval_result: Result from kernel evaluation
        config: Reward configuration (uses defaults if None)
        thought_length: Length of thinking content in characters (0 if no thinking)

    Returns:
        Total reward (scalar)
    """
    if config is None:
        config = RewardConfig()

    # ==========================================================================
    # Kevin Rule: Zero reward for cheating (early return)
    # This is the critical fix - cheating gets 0, not a penalty
    # ==========================================================================
    if eval_result["cheated"]:
        return 0.0

    # ==========================================================================
    # Kevin Rule: Zero reward for bad format
    # ==========================================================================
    if not eval_result["format_ok"]:
        return 0.0

    # Sparse reward mode: only reward fully correct solutions
    if config.sparse_rewards:
        if eval_result["correctness"]:
            base_reward = 1.0
            # Add speed bonus if enabled
            if config.speed_weight > 0:
                s_reward = speed_reward(eval_result, config, use_speed=True)
                base_reward += config.speed_weight * s_reward
            # Add length bonus for tie-breaking (disabled by default in Kevin mode)
            if config.length_weight > 0:
                l_reward = length_reward(eval_result, config)
                base_reward += config.length_weight * l_reward
            # Add thinking bonus (disabled by default in Kevin mode)
            if config.thinking_weight > 0:
                t_reward = thinking_reward(thought_length, config)
                base_reward += config.thinking_weight * t_reward
            return base_reward
        return 0.0

    # ==========================================================================
    # Kevin-style reward computation
    # Formula: S = correctness_weight·correct + speed_weight·speedup
    # With default Kevin weights: S = 0.3·correct + 1.0·speedup
    # ==========================================================================

    # Correctness reward (binary by default)
    corr_reward = correctness_reward(eval_result, config)

    # If not correct, zero reward (Kevin doesn't give partial credit)
    if corr_reward == 0.0:
        return 0.0

    # Base reward for correctness
    total = config.correctness_weight * corr_reward

    # Add speedup reward (only for correct kernels)
    if config.speed_weight > 0:
        s_reward = speed_reward(eval_result, config, use_speed=True)
        total += config.speed_weight * s_reward

    # Optional: format/compile rewards (disabled by default in Kevin mode)
    if config.format_weight > 0:
        f_reward = format_reward(eval_result, config)
        total += config.format_weight * f_reward

    if config.compile_weight > 0:
        c_reward = compile_reward(eval_result, config)
        total += config.compile_weight * c_reward

    # Optional: length reward (DISABLED by default - Kevin says it causes collapse)
    if config.length_weight > 0:
        l_reward = length_reward(eval_result, config)
        total += config.length_weight * l_reward

    # Optional: thinking reward (DISABLED by default - Kevin removes thinking)
    if config.thinking_weight > 0:
        t_reward = thinking_reward(thought_length, config)
        total += config.thinking_weight * t_reward

    return total


def compute_reward_breakdown(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
    thought_length: int = 0,
) -> dict[str, float]:
    """
    Compute individual reward components for logging.

    Args:
        eval_result: Result from kernel evaluation
        config: Reward configuration (uses defaults if None)
        thought_length: Length of thinking content in characters

    Returns:
        Dictionary with individual reward values
    """
    if config is None:
        config = RewardConfig()

    return {
        "reward_format": format_reward(eval_result, config),
        "reward_compile": compile_reward(eval_result, config),
        "reward_correctness": correctness_reward(eval_result, config),
        "reward_speed": speed_reward(eval_result, config, use_speed=True),
        "reward_length": length_reward(eval_result, config),
        "reward_thinking": thinking_reward(thought_length, config),
        "reward_total": compute_reward(eval_result, config, thought_length),
    }


# Preset reward configurations for different training stages


def get_stage1_config() -> RewardConfig:
    """
    Stage 1: Focus on format and compilation.

    Good for early training when the model is learning
    to generate valid code structure.
    """
    return RewardConfig(
        format_weight=0.3,
        compile_weight=0.5,
        correctness_weight=0.2,
        speed_weight=0.0,
        partial_correctness=True,
        sparse_rewards=False,
    )


def get_stage2_config() -> RewardConfig:
    """
    Stage 2: Focus on correctness.

    For mid-training when the model can compile code
    but needs to learn correctness.
    """
    return RewardConfig(
        format_weight=0.1,
        compile_weight=0.2,
        correctness_weight=0.7,
        speed_weight=0.0,
        partial_correctness=True,
        sparse_rewards=False,
    )


def get_stage3_config() -> RewardConfig:
    """
    Stage 3: Correctness + Speed.

    For late training when the model can produce correct
    kernels and should optimize for speed.
    """
    return RewardConfig(
        format_weight=0.05,
        compile_weight=0.1,
        correctness_weight=0.5,
        speed_weight=0.35,
        partial_correctness=False,  # Only reward fully correct
        sparse_rewards=False,
    )


def get_sparse_config() -> RewardConfig:
    """
    Sparse rewards: Only reward fully correct solutions.

    May be useful for fine-tuning or when combined with
    curriculum learning.
    """
    return RewardConfig(
        sparse_rewards=True,
        speed_weight=0.5,
    )


REWARD_PRESETS = {
    "stage1": get_stage1_config,
    "stage2": get_stage2_config,
    "stage3": get_stage3_config,
    "sparse": get_sparse_config,
    "default": RewardConfig,
}


def get_reward_config(name: str) -> RewardConfig:
    """Get a reward configuration by name."""
    if name not in REWARD_PRESETS:
        raise ValueError(
            f"Unknown reward preset: {name}. Available: {list(REWARD_PRESETS.keys())}"
        )
    return REWARD_PRESETS[name]()


# ============================================================================
# Multi-Turn / Kevin Mode: Discounted Returns
# ============================================================================


def compute_discounted_returns(
    step_scores: list[float],
    gamma: float = 0.4,
) -> list[float]:
    """
    Compute discounted returns for multi-turn RL (Kevin mode).

    Given per-step scores [s_0, s_1, ..., s_{T-1}], compute returns:
        R_t = s_t + gamma * s_{t+1} + gamma^2 * s_{t+2} + ... + gamma^{T-1-t} * s_{T-1}

    This follows the Kevin paper where each turn's reward includes discounted
    future rewards, encouraging the model to set up for successful future turns.

    Args:
        step_scores: List of per-step scores from the trajectory
        gamma: Discount factor (0.4 as in Kevin paper)

    Returns:
        List of discounted returns, one per step

    Example:
        >>> compute_discounted_returns([0.2, 0.3, 0.8, 1.0], gamma=0.4)
        [0.6528, 0.632, 1.2, 1.0]
        # R_0 = 0.2 + 0.4*0.3 + 0.16*0.8 + 0.064*1.0 = 0.6528
        # R_1 = 0.3 + 0.4*0.8 + 0.16*1.0 = 0.632
        # R_2 = 0.8 + 0.4*1.0 = 1.2
        # R_3 = 1.0
    """
    if not step_scores:
        return []

    T = len(step_scores)
    returns = [0.0] * T

    # Compute backwards for efficiency
    # R_t = s_t + gamma * R_{t+1}
    returns[T - 1] = step_scores[T - 1]

    for t in range(T - 2, -1, -1):
        returns[t] = step_scores[t] + gamma * returns[t + 1]

    return returns


def compute_trajectory_returns(
    trajectories_step_scores: list[list[float]],
    gamma: float = 0.4,
) -> list[list[float]]:
    """
    Compute discounted returns for multiple trajectories.

    Args:
        trajectories_step_scores: List of step score lists, one per trajectory
        gamma: Discount factor

    Returns:
        List of return lists, one per trajectory
    """
    return [
        compute_discounted_returns(scores, gamma)
        for scores in trajectories_step_scores
    ]


@dataclass
class MultiTurnRewardConfig:
    """Configuration for multi-turn reward computation."""

    # Per-step reward config (same as single-turn)
    step_reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Discount factor for computing returns (Kevin paper uses 0.4)
    gamma: float = 0.4

    # Whether to normalize returns within trajectory
    normalize_returns: bool = False

    # Bonus for successfully solving the problem
    success_bonus: float = 0.0

    # Penalty for using all turns without success
    failure_penalty: float = 0.0


def compute_multiturn_rewards(
    step_scores: list[float],
    config: MultiTurnRewardConfig,
    success: bool = False,
) -> list[float]:
    """
    Compute final rewards for each step in a multi-turn trajectory.

    Args:
        step_scores: Per-step scores from evaluation
        config: Multi-turn reward configuration
        success: Whether the trajectory successfully solved the problem

    Returns:
        List of rewards to use for RL training, one per step
    """
    if not step_scores:
        return []

    # Compute discounted returns
    returns = compute_discounted_returns(step_scores, config.gamma)

    # Apply success/failure bonuses to final step
    if success and config.success_bonus > 0:
        returns[-1] += config.success_bonus
    elif not success and config.failure_penalty != 0:
        returns[-1] += config.failure_penalty

    # Optional normalization
    if config.normalize_returns and len(returns) > 1:
        import numpy as np
        returns_arr = np.array(returns)
        mean = returns_arr.mean()
        std = returns_arr.std()
        if std > 1e-8:
            returns = ((returns_arr - mean) / std).tolist()

    return returns
