"""
Reward shaping for KernelBench RL training.

This module implements reward functions that combine:
- Format correctness (valid code block extraction)
- Compilation success
- Correctness (passing all tests)
- Speed (optional)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Import static checker with fallback for development
try:
    import kernelbench.kernel_static_checker as static_checker
except ImportError:
    # Fallback: add KernelBench/src to path if package not installed
    project_root = Path(__file__).parent.parent.parent.parent.parent
    kernelbench_src = project_root / "KernelBench" / "src"
    if kernelbench_src.exists() and str(kernelbench_src) not in sys.path:
        sys.path.insert(0, str(kernelbench_src))
    import kernelbench.kernel_static_checker as static_checker

if TYPE_CHECKING:
    from kernelbench_tinker.envs.kernelbench_client import KernelEvalResult

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """
    Configuration for reward computation.

    Default values follow the published KernelBench RL setup:
    - Reward = 0.3 * correct + speedup (if correct)
    - No length penalties (can cause response collapse)
    - Binary correctness (no partial credit)
    """

    # ==========================================================================
    # Default reward weights
    # Formula: S = 0.3 * correct + (T_baseline/T_kernel) * correct
    # ==========================================================================
    format_weight: float = 0.0  # Disabled by default
    compile_weight: float = 0.0  # Disabled by default
    correctness_weight: float = 0.3  # Binary correctness weight
    speed_weight: float = 1.0  # Adds speedup directly
    length_weight: float = 0.0  # Length penalties can cause response collapse

    format_penalty: float = 0.0  # Zero reward for bad format

    # ==========================================================================
    # Speed reward configuration
    # Linear speedup: reward = T_baseline / T_kernel
    # ==========================================================================
    speed_baseline: float = 0.0  # Speedup threshold (0.0 = raw speedup as reward)
    speed_scale: float = 1.0  # Linear scaling (not log)
    speed_max_reward: float = 10.0  # Cap to prevent outliers

    # ==========================================================================
    # Length reward configuration (DISABLED by default)
    # Length penalties can cause response collapse.
    # ==========================================================================
    length_max: int = 8000  # Not used when length_weight=0
    length_min: int = 500  # Not used when length_weight=0

    # ==========================================================================

    # ==========================================================================
    # Correctness configuration
    # Binary correctness (all tests pass or not)
    # ==========================================================================
    sparse_rewards: bool = False
    partial_correctness: bool = False  # Binary correctness

    # ==========================================================================
    # Reward hacking detection configuration (static checker)
    # ==========================================================================
    # Enable static checking for reward hacking
    enable_static_checker: bool = True
    
    # Backend for static checking (must match the kernel backend)
    static_checker_backend: str = "triton"  # "cuda", "triton", "thunderkittens", "cute", "tilelang"
    
    # Precision for static checking (must match evaluation precision)
    static_checker_precision: str = "fp32"  # "fp32", "fp16", "bf16"
    
    # Strict checks (errors) - these will zero the reward if detected
    # Default: all strict checks from static_checker.STRICT_CHECKS + backend impl check
    static_checker_strict: list[str] | None = None  # None = use defaults (all strict checks)
    
    # Warning checks - these will log warnings but NOT zero the reward
    # Default: all warning checks from static_checker.WARNING_CHECKS
    static_checker_warnings: list[str] | None = None  # None = use defaults (all warning checks)

    # ==========================================================================
    # Reward clipping configuration
    # ==========================================================================
    reward_clip_min: float | None = None  # Lower bound on total reward (None = no clipping)
    reward_clip_max: float | None = None  # Upper bound on total reward (None = no clipping)


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
        1.0 if compiled successfully
        0.0 if compilation failed
    """
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

    Formula:
        speed_reward = T_baseline / T_kernel = speedup

    Uses LINEAR speedup directly, not log-scaled.
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

    # Linear speedup: reward = speedup for correct kernels
    # With default speed_baseline=0.0: 2x speedup = 2.0 reward
    if config.speed_baseline > 0 and speedup <= config.speed_baseline:
        return 0.0

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


def check_reward_hacking(
    kernel_code: str,
    config: RewardConfig,
    backend: str | None = None,
) -> tuple[bool, list[str], list[str]]:
    """
    Check for reward hacking patterns using static checker.
    
    Args:
        kernel_code: The kernel source code to check
        config: Reward configuration with static checker settings
        backend: Backend type (overrides config.static_checker_backend if provided)
    
    Returns:
        Tuple of (has_errors, errors, warnings):
        - has_errors: True if any strict checks failed (should zero reward)
        - errors: List of error messages from strict checks
        - warnings: List of warning messages (don't zero reward)
    """
    if not config.enable_static_checker:
        return (False, [], [])
    
    # Use provided backend or config default
    backend = backend or config.static_checker_backend
    
    # Determine which checks to run
    # Use defaults from static_checker if not specified
    strict_checks = config.static_checker_strict
    if strict_checks is None:
        # Default: all strict checks + backend impl check
        strict_checks = list(static_checker.STRICT_CHECKS)
        if backend in static_checker.BACKEND_IMPL_CHECK:
            backend_check = static_checker.BACKEND_IMPL_CHECK[backend]
            if backend_check not in strict_checks:
                strict_checks.append(backend_check)
    
    warning_checks = config.static_checker_warnings
    if warning_checks is None:
        # Default: all warning checks
        warning_checks = list(static_checker.WARNING_CHECKS)
    
    # Run static checker
    valid, errors, warnings = static_checker.validate_kernel_static(
        code=kernel_code,
        backend=backend,
        precision=config.static_checker_precision,
        forbidden=strict_checks,
        warnings=warning_checks,
    )
    
    # Return (has_errors, errors, warnings)
    # valid is False if there are errors
    has_errors = not valid
    return (has_errors, errors, warnings)


def compute_reward(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
    kernel_code: str | None = None,
    backend: str | None = None,
) -> float:
    """
    Compute the total reward for a kernel evaluation.

    Formula:
        S = 0.3 * correct + (T_baseline/T_kernel) * correct

    Key behaviors:
    - Zero reward for incorrect kernels
    - Binary correctness (no partial credit by default)
    - Speedup added linearly (not log-scaled)
    - Zero reward if reward hacking detected (static checker errors)

    Args:
        eval_result: Result from kernel evaluation
        config: Reward configuration (uses defaults if None)
        kernel_code: Kernel source code (required for static checking)
        backend: Backend type (overrides config if provided)

    Returns:
        Total reward (scalar)
    """
    if config is None:
        config = RewardConfig()

    # ==========================================================================
    # Zero reward for bad format
    # ==========================================================================
    if not eval_result["format_ok"]:
        return 0.0

    # ==========================================================================
    # Check for reward hacking (static checker)
    # ==========================================================================
    if config.enable_static_checker and kernel_code:
        has_errors, errors, warnings = check_reward_hacking(
            kernel_code=kernel_code,
            config=config,
            backend=backend,
        )
        
        # Log warnings (don't zero reward)
        for warning in warnings:
            logger.warning(f"Static checker warning: {warning}")

        # Zero reward if errors detected
        if has_errors:
            for error in errors:
                logger.error(f"Reward hacking detected (reward set to 0): {error}")
            return 0.0

    # Sparse reward mode: only reward fully correct solutions
    if config.sparse_rewards:
        if eval_result["correctness"]:
            base_reward = 1.0
            # Add speed bonus if enabled
            if config.speed_weight > 0:
                s_reward = speed_reward(eval_result, config, use_speed=True)
                base_reward += config.speed_weight * s_reward
            # Add length bonus for tie-breaking (disabled by default)
            if config.length_weight > 0:
                l_reward = length_reward(eval_result, config)
                base_reward += config.length_weight * l_reward
            return base_reward
        return 0.0

    # ==========================================================================
    # Default reward computation
    # Formula: S = correctness_weight * correct + speed_weight * speedup
    # With default weights: S = 0.3 * correct + 1.0 * speedup
    # ==========================================================================

    # Correctness reward (binary by default)
    corr_reward = correctness_reward(eval_result, config)

    # If not correct, zero reward
    if corr_reward == 0.0:
        return 0.0

    # Base reward for correctness
    total = config.correctness_weight * corr_reward

    # Add speedup reward (only for correct kernels)
    if config.speed_weight > 0:
        s_reward = speed_reward(eval_result, config, use_speed=True)
        total += config.speed_weight * s_reward

    # Optional: format/compile rewards (disabled by default)
    if config.format_weight > 0:
        f_reward = format_reward(eval_result, config)
        total += config.format_weight * f_reward

    if config.compile_weight > 0:
        c_reward = compile_reward(eval_result, config)
        total += config.compile_weight * c_reward

    # Optional: length reward (DISABLED by default)
    if config.length_weight > 0:
        l_reward = length_reward(eval_result, config)
        total += config.length_weight * l_reward

    # Reward clipping
    if config.reward_clip_min is not None:
        total = max(total, config.reward_clip_min)
    if config.reward_clip_max is not None:
        total = min(total, config.reward_clip_max)

    return total


def compute_reward_breakdown(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
    kernel_code: str | None = None,
    backend: str | None = None,
) -> dict[str, float | list[str]]:
    """
    Compute individual reward components for logging.

    Args:
        eval_result: Result from kernel evaluation
        config: Reward configuration (uses defaults if None)
        kernel_code: Kernel source code (for static checking)
        backend: Backend type (for static checking)

    Returns:
        Dictionary with individual reward values and static checker results
    """
    if config is None:
        config = RewardConfig()

    # Run static checker if enabled
    static_checker_errors: list[str] = []
    static_checker_warnings: list[str] = []
    if config.enable_static_checker and kernel_code:
        _, static_checker_errors, static_checker_warnings = check_reward_hacking(
            kernel_code=kernel_code,
            config=config,
            backend=backend,
        )

    return {
        "reward_format": format_reward(eval_result, config),
        "reward_compile": compile_reward(eval_result, config),
        "reward_correctness": correctness_reward(eval_result, config),
        "reward_speed": speed_reward(eval_result, config, use_speed=True),
        "reward_length": length_reward(eval_result, config),
        "reward_total": compute_reward(eval_result, config, kernel_code=kernel_code, backend=backend),
        "static_checker_errors": static_checker_errors,
        "static_checker_warnings": static_checker_warnings,
    }


def compute_discounted_returns(
    step_scores: list[float],
    gamma: float = 0.4,
    aggregation: str = "sum",
) -> list[float]:
    """Compute discounted returns for multi-turn RL.

    sum: R_t = S_t + gamma * R_{t+1}  (backward recursion)
    max: R_t = max{ gamma^(i-t) * S_i }
    """
    if aggregation not in ("sum", "max"):
        raise ValueError(f"Unknown aggregation mode: {aggregation!r}. Must be 'sum' or 'max'.")

    if not step_scores:
        return []

    T = len(step_scores)

    if aggregation == "sum":
        returns = [0.0] * T
        returns[-1] = step_scores[-1]
        for t in range(T - 2, -1, -1):
            returns[t] = step_scores[t] + gamma * returns[t + 1]
        return returns

    # aggregation == "max"
    return [
        max(gamma ** (i - t) * step_scores[i] for i in range(t, T))
        for t in range(T)
    ]
