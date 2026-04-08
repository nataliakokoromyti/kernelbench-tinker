"""
Configuration dataclasses for KernelBench-Tinker integration.

This module centralizes all configuration to avoid duplication across classes.
Configs are designed to be loaded from YAML and passed through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """
    Configuration for kernel evaluation.

    Drives correctness/performance measurement and selects the evaluator
    backend (Modal cloud NVIDIA, or local subprocess for on-host AMD/HIP).
    """
    
    # Correctness testing
    num_correct_trials: int = 5
    
    # Performance measurement
    measure_performance: bool = False
    num_perf_trials: int = 100
    timing_method: str = "cuda_event"
    
    # Precision
    precision: str = "fp32"
    
    # Speedup validation
    check_for_excessive_speedup: bool = True
    excessive_speedup_threshold: float = 10.0
    
    # Evaluator backend selection: "modal" (cloud, NVIDIA) or "local" (in-process subprocess)
    evaluator_backend: str = "modal"

    # Modal configuration
    modal_gpu_type: str = "A100"
    modal_timeout: float = 120.0

    # Local evaluator configuration (used when evaluator_backend == "local")
    # gpu_arch is passed to set_gpu_arch() — e.g. ["gfx950"] for MI350X, ["gfx942"] for MI300X
    gpu_arch: list[str] = field(default_factory=list)
    local_timeout: float = 300.0


@dataclass
class PromptConfig:
    """
    Configuration for prompt generation.
    
    Controls how prompts are constructed from KernelBench problems.
    """
    
    # Prompt style
    option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot"
    
    # Precision hints in prompt
    precision: str | None = None
    
    # Hardware-aware prompting
    include_hardware: bool = False
    gpu_name: str | None = None


@dataclass 
class DatasetConfig:
    """
    Configuration for dataset construction.
    
    Controls which problems are included and how they're batched.
    """
    
    # Problem selection
    level: int = 1
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"
    
    # Batching
    batch_size: int = 4
    group_size: int = 4
    num_epochs: int = 1
    shuffle: bool = True
    
    # Train/test split
    test_fraction: float = 0.1


@dataclass
class MultiTurnConfig:
    """
    Configuration for multi-turn RL training.

    Controls the iterative refinement loop where the model receives
    evaluation feedback and can fix errors across multiple turns.
    """

    # Enable multi-turn mode (False = single-turn)
    enabled: bool = False

    # Maximum refinement turns per trajectory
    max_turns: int = 4

    # Discount factor for multi-turn returns: R_t = S_t + gamma * R_{t+1}
    gamma: float = 0.4

    # Return aggregation mode: "sum" or "max"
    #   sum: R_t = Σ γ^(i-t) × S_i  (reward turns leading to many good kernels)
    #   max: R_t = max{ γ^(i-t) × S_i } (reward turns leading to one great kernel)
    aggregation: str = "sum"

    # Stop the episode early when the kernel is correct.
    # Default False for training: model needs post-correctness turns to
    # learn speedup optimization.  Set True at eval time if desired.
    early_stop_on_correct: bool = False

    # Optional: require this speedup before early stopping
    speedup_threshold: float | None = None

    # Prompt
    prompt_max_tokens: int | None = None  # Token budget for history truncation (None = char fallback)
    inject_think_token: bool = False  # Append <think>\n to generation prompts

    # Generation
    temperature: float = 0.9
    top_p: float = 1.0
    seed: int | None = None

    # Response length extension mid-training (0 = disabled)
    max_tokens_extended: int = 22000
    max_tokens_extend_after_step: int = 30

    # Training
    loss_fn: str = "ppo"
    max_grad_norm: float = 0.05
    warmup_ratio: float = 0.03
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.28
    constant_length_norm: int = 16384
    num_substeps: int = 2
