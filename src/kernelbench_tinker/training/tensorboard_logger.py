"""
TensorBoard logging for KernelBench RL training.

This module provides TensorBoard integration for visualizing:
- Reward metrics (mean, std, min, max, distributions)
- Kernel quality rates (format, compile, correct, cheat)
- Training progress and timing
- Evaluation metrics (pass@k, speedup)
- Per-level breakdowns
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from tinker_cookbook.rl.types import TrajectoryGroup
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard logging."""

    log_dir: str = "./runs/tensorboard"
    flush_secs: int = 30

    # Logging frequency
    log_histograms_every: int = 5  # Log histograms every N batches
    log_detailed_every: int = 1    # Log detailed metrics every N batches

    # What to log
    log_reward_histograms: bool = True
    log_per_level_metrics: bool = True
    log_timing: bool = True
    log_hyperparameters: bool = True


class TensorBoardLogger:
    """
    TensorBoard logger for KernelBench RL training.

    Provides comprehensive logging of training metrics, including:
    - Scalar metrics (rewards, rates, timing)
    - Histograms (reward distributions)
    - Per-level breakdowns
    - Evaluation metrics
    """

    def __init__(
        self,
        log_dir: str,
        config: TensorBoardConfig | None = None,
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            config: Optional configuration
        """
        self.config = config or TensorBoardConfig()

        # Create TensorBoard log directory
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=tb_log_dir,
            flush_secs=self.config.flush_secs,
        )
        self.log_dir = tb_log_dir

        logger.info(f"TensorBoard logging to: {tb_log_dir}")
        logger.info(f"Run 'tensorboard --logdir {tb_log_dir}' to visualize")

    def log_training_config(self, config: Any) -> None:
        """
        Log training configuration as hyperparameters.

        Args:
            config: Training configuration object
        """
        if not self.config.log_hyperparameters:
            return

        # Extract relevant hyperparameters
        hparams = {}

        # Get attributes from config object
        config_dict = vars(config) if hasattr(config, '__dict__') else {}

        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
            elif hasattr(value, '__dict__'):
                # Handle nested config objects like dataset_builder
                for nested_key, nested_value in vars(value).items():
                    if isinstance(nested_value, (int, float, str, bool)):
                        hparams[f"{key}/{nested_key}"] = nested_value

        # Log hyperparameters
        if hparams:
            self.writer.add_hparams(
                hparams,
                metric_dict={},  # Will be updated during training
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: np.ndarray | list, step: int) -> None:
        """Log a histogram of values."""
        if len(values) > 0:
            self.writer.add_histogram(tag, np.array(values), step)

    def log_training_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """
        Log training metrics from a single batch.

        Args:
            metrics: Dictionary of metrics from training loop
            step: Current training step (batch index)
        """
        # === Reward Metrics ===
        if "reward/mean" in metrics:
            self.writer.add_scalar("Reward/Mean", metrics["reward/mean"], step)
        if "reward/std" in metrics:
            self.writer.add_scalar("Reward/StdDev", metrics["reward/std"], step)
        if "reward/min" in metrics:
            self.writer.add_scalar("Reward/Min", metrics["reward/min"], step)
        if "reward/max" in metrics:
            self.writer.add_scalar("Reward/Max", metrics["reward/max"], step)

        # Combined reward chart
        reward_metrics = {}
        for key in ["reward/mean", "reward/min", "reward/max"]:
            if key in metrics:
                short_key = key.split("/")[1]
                reward_metrics[short_key] = metrics[key]
        if reward_metrics:
            self.writer.add_scalars("Reward/Summary", reward_metrics, step)

        # === Kernel Quality Metrics ===
        if "kernel/format_rate" in metrics:
            self.writer.add_scalar("Kernel/FormatRate", metrics["kernel/format_rate"], step)
        if "kernel/compile_rate" in metrics:
            self.writer.add_scalar("Kernel/CompileRate", metrics["kernel/compile_rate"], step)
        if "kernel/correct_rate" in metrics:
            self.writer.add_scalar("Kernel/CorrectRate", metrics["kernel/correct_rate"], step)
        if "kernel/cheat_rate" in metrics:
            self.writer.add_scalar("Kernel/CheatRate", metrics["kernel/cheat_rate"], step)

        # Combined kernel quality chart
        quality_metrics = {}
        for key in ["kernel/format_rate", "kernel/compile_rate", "kernel/correct_rate"]:
            if key in metrics:
                short_key = key.replace("kernel/", "").replace("_rate", "")
                quality_metrics[short_key] = metrics[key]
        if quality_metrics:
            self.writer.add_scalars("Kernel/QualityRates", quality_metrics, step)

        # === Progress Metrics ===
        if "progress/done_frac" in metrics:
            self.writer.add_scalar("Progress/CompletionFraction", metrics["progress/done_frac"], step)
        if "optim/lr" in metrics:
            self.writer.add_scalar("Optimizer/LearningRate", metrics["optim/lr"], step)

        # === Batch Statistics ===
        if "batch/num_groups" in metrics:
            self.writer.add_scalar("Batch/NumGroups", metrics["batch/num_groups"], step)
        if "batch/num_trajectories" in metrics:
            self.writer.add_scalar("Batch/NumTrajectories", metrics["batch/num_trajectories"], step)

        # === Timing Metrics ===
        if self.config.log_timing:
            timing_metrics = {}
            for key in ["time/total", "time/rollout", "time/assemble_data", "time/train", "time/save_checkpoint"]:
                if key in metrics:
                    short_key = key.replace("time/", "")
                    timing_metrics[short_key] = metrics[key]
                    self.writer.add_scalar(f"Timing/{short_key.title()}", metrics[key], step)

            if timing_metrics:
                self.writer.add_scalars("Timing/Breakdown", timing_metrics, step)

        # Log any speedup metrics if available
        if "kernel/mean_speedup" in metrics:
            self.writer.add_scalar("Kernel/MeanSpeedup", metrics["kernel/mean_speedup"], step)

        # === Multi-turn Reward Metrics ===
        # Multi-turn emits reward/discounted_{mean,std,min,max}
        if "reward/discounted_mean" in metrics:
            self.writer.add_scalar("Reward/DiscountedMean", metrics["reward/discounted_mean"], step)
        if "reward/discounted_std" in metrics:
            self.writer.add_scalar("Reward/DiscountedStdDev", metrics["reward/discounted_std"], step)
        if "reward/discounted_min" in metrics:
            self.writer.add_scalar("Reward/DiscountedMin", metrics["reward/discounted_min"], step)
        if "reward/discounted_max" in metrics:
            self.writer.add_scalar("Reward/DiscountedMax", metrics["reward/discounted_max"], step)

        # === Multi-turn Kernel Quality Metrics ===
        if "multiturn/format_rate" in metrics:
            self.writer.add_scalar("MultiTurn/FormatRate", metrics["multiturn/format_rate"], step)
        if "multiturn/compile_rate" in metrics:
            self.writer.add_scalar("MultiTurn/CompileRate", metrics["multiturn/compile_rate"], step)
        if "multiturn/correct_rate" in metrics:
            self.writer.add_scalar("MultiTurn/CorrectRate", metrics["multiturn/correct_rate"], step)

        # === Multi-turn Failure Rate ===
        if "kernel/failure_rate" in metrics:
            self.writer.add_scalar("Kernel/FailureRate", metrics["kernel/failure_rate"], step)
        if "multiturn/failure_rate" in metrics:
            self.writer.add_scalar("MultiTurn/FailureRate", metrics["multiturn/failure_rate"], step)

        # === Multi-turn Specific Metrics ===
        if "multiturn/raw_score_mean" in metrics:
            self.writer.add_scalar("MultiTurn/RawScoreMean", metrics["multiturn/raw_score_mean"], step)
        if "multiturn/success_rate" in metrics:
            self.writer.add_scalar("MultiTurn/SuccessRate", metrics["multiturn/success_rate"], step)
        if "multiturn/avg_turns" in metrics:
            self.writer.add_scalar("MultiTurn/AvgTurns", metrics["multiturn/avg_turns"], step)
        if "multiturn/best_speedup_mean" in metrics:
            self.writer.add_scalar("MultiTurn/BestSpeedupMean", metrics["multiturn/best_speedup_mean"], step)

    def log_trajectory_histograms(
        self,
        trajectory_groups: list[TrajectoryGroup],
        step: int,
    ) -> None:
        """
        Log histograms from trajectory groups.

        Args:
            trajectory_groups: List of trajectory groups
            step: Current training step
        """
        if not self.config.log_reward_histograms:
            return

        if step % self.config.log_histograms_every != 0:
            return

        # Collect all rewards
        all_rewards = []
        all_format_ok = []
        all_compiled = []
        all_correct = []
        all_speedups = []

        for tg in trajectory_groups:
            rewards = tg.get_total_rewards()
            all_rewards.extend(rewards)

            for traj in tg.trajectories_G:
                for trans in traj.transitions:
                    if trans.metrics:
                        all_format_ok.append(trans.metrics.get("format_ok", 0))
                        all_compiled.append(trans.metrics.get("compiled", 0))
                        all_correct.append(trans.metrics.get("correctness", 0))
                        if trans.metrics.get("speedup"):
                            all_speedups.append(trans.metrics["speedup"])

        # Log histograms
        if all_rewards:
            self.writer.add_histogram("Distributions/Rewards", np.array(all_rewards), step)

        if all_speedups:
            self.writer.add_histogram("Distributions/Speedups", np.array(all_speedups), step)

        # Log success rate distributions (as bar chart approximation)
        success_counts = {
            "format_ok": sum(all_format_ok),
            "compiled": sum(all_compiled),
            "correct": sum(all_correct),
        }
        total = len(all_format_ok) if all_format_ok else 1
        for key, count in success_counts.items():
            self.writer.add_scalar(f"Counts/{key}", count, step)

    def log_per_level_metrics(
        self,
        trajectory_groups: list[TrajectoryGroup],
        step: int,
    ) -> None:
        """
        Log metrics broken down by problem level.

        Args:
            trajectory_groups: List of trajectory groups
            step: Current training step
        """
        if not self.config.log_per_level_metrics:
            return

        # Aggregate metrics by level
        level_rewards: dict[int, list[float]] = {}
        level_correct: dict[int, list[float]] = {}
        level_compiled: dict[int, list[float]] = {}

        for tg in trajectory_groups:
            # Extract level from trajectory metrics (set by KernelBenchEnv)
            # TrajectoryGroup doesn't have logging_tags, so we extract from metrics
            level = None

            # Try to get level from transition metrics
            for traj in tg.trajectories_G:
                for trans in traj.transitions:
                    if trans.metrics and "level" in trans.metrics:
                        level = int(trans.metrics["level"])
                        break
                if level is not None:
                    break

            if level is None:
                # Default to level 1 if not found (single-level training)
                level = 1

            # Initialize level buckets
            if level not in level_rewards:
                level_rewards[level] = []
                level_correct[level] = []
                level_compiled[level] = []

            # Collect metrics for this level
            rewards = tg.get_total_rewards()
            level_rewards[level].extend(rewards)

            for traj in tg.trajectories_G:
                for trans in traj.transitions:
                    if trans.metrics:
                        level_correct[level].append(trans.metrics.get("correctness", 0))
                        level_compiled[level].append(trans.metrics.get("compiled", 0))

        # Log per-level metrics
        for level in sorted(level_rewards.keys()):
            if level_rewards[level]:
                self.writer.add_scalar(
                    f"PerLevel/Level{level}/RewardMean",
                    np.mean(level_rewards[level]),
                    step
                )
            if level_correct[level]:
                self.writer.add_scalar(
                    f"PerLevel/Level{level}/CorrectRate",
                    np.mean(level_correct[level]),
                    step
                )
            if level_compiled[level]:
                self.writer.add_scalar(
                    f"PerLevel/Level{level}/CompileRate",
                    np.mean(level_compiled[level]),
                    step
                )

    def log_evaluation_metrics(
        self,
        eval_summary: dict[str, Any],
        step: int,
        prefix: str = "Eval",
    ) -> None:
        """
        Log evaluation metrics.

        Args:
            eval_summary: Summary dictionary from EvalResults.summary()
            step: Current step (or epoch)
            prefix: Prefix for metric names
        """
        # Pass@k metrics
        if eval_summary.get("pass@1") is not None:
            self.writer.add_scalar(f"{prefix}/Pass@1", eval_summary["pass@1"], step)
        if eval_summary.get("pass@5") is not None:
            self.writer.add_scalar(f"{prefix}/Pass@5", eval_summary["pass@5"], step)

        # Quality rates
        if eval_summary.get("compile_rate") is not None:
            self.writer.add_scalar(f"{prefix}/CompileRate", eval_summary["compile_rate"], step)
        if eval_summary.get("correct_rate") is not None:
            self.writer.add_scalar(f"{prefix}/CorrectRate", eval_summary["correct_rate"], step)

        # Speed metrics
        if eval_summary.get("fast_1") is not None:
            self.writer.add_scalar(f"{prefix}/Fast1x", eval_summary["fast_1"], step)
        if eval_summary.get("fast_2") is not None:
            self.writer.add_scalar(f"{prefix}/Fast2x", eval_summary["fast_2"], step)
        if eval_summary.get("mean_speedup") is not None:
            self.writer.add_scalar(f"{prefix}/MeanSpeedup", eval_summary["mean_speedup"], step)
        if eval_summary.get("geometric_mean_speedup") is not None:
            self.writer.add_scalar(f"{prefix}/GeometricMeanSpeedup", eval_summary["geometric_mean_speedup"], step)

        # Combined charts
        quality_dict = {}
        if eval_summary.get("compile_rate") is not None:
            quality_dict["compile"] = eval_summary["compile_rate"]
        if eval_summary.get("correct_rate") is not None:
            quality_dict["correct"] = eval_summary["correct_rate"]
        if eval_summary.get("pass@1") is not None:
            quality_dict["pass@1"] = eval_summary["pass@1"]
        if quality_dict:
            self.writer.add_scalars(f"{prefix}/QualityOverview", quality_dict, step)

    def log_reward_components(
        self,
        reward_breakdown: dict[str, float],
        step: int,
    ) -> None:
        """
        Log individual reward components.

        Args:
            reward_breakdown: Dictionary from compute_reward_breakdown()
            step: Current step
        """
        for key, value in reward_breakdown.items():
            clean_key = key.replace("reward_", "").title()
            self.writer.add_scalar(f"RewardComponents/{clean_key}", value, step)

        # Log as stacked (approximation via scalars)
        self.writer.add_scalars("RewardComponents/All", reward_breakdown, step)

    def log_advantage_statistics(
        self,
        advantages: Sequence[np.ndarray],
        step: int,
    ) -> None:
        """
        Log advantage statistics.

        Args:
            advantages: List of advantage arrays
            step: Current step
        """
        all_advantages = np.concatenate([a.flatten() for a in advantages])

        if len(all_advantages) > 0:
            self.writer.add_scalar("Advantage/Mean", np.mean(all_advantages), step)
            self.writer.add_scalar("Advantage/Std", np.std(all_advantages), step)
            self.writer.add_scalar("Advantage/Min", np.min(all_advantages), step)
            self.writer.add_scalar("Advantage/Max", np.max(all_advantages), step)
            self.writer.add_histogram("Distributions/Advantages", all_advantages, step)

    def flush(self) -> None:
        """Flush pending logs to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


def create_tensorboard_logger(
    log_dir: str,
    config: TensorBoardConfig | None = None,
) -> TensorBoardLogger:
    """
    Factory function to create a TensorBoard logger.

    Args:
        log_dir: Base directory for logs
        config: Optional TensorBoard configuration

    Returns:
        Configured TensorBoardLogger instance
    """
    return TensorBoardLogger(log_dir=log_dir, config=config)
