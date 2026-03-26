"""
Multi-turn rollout, advantage estimation, and metrics for KernelBench RL.

These helpers are used by the training loop when multiturn.enabled is True.
Single-turn training does not touch this module.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import tinker
import torch
from tinker_cookbook.rl.data_processing import remove_constant_reward_groups
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
)

from kernelbench_tinker.envs.multiturn_kernelbench_env import MultiTurnKernelBenchEnv
from kernelbench_tinker.training.models import KernelBenchTokenCompleter
from kernelbench_tinker.training.reward import compute_discounted_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


async def do_multiturn_group_rollout_and_filter(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    top_p: float = 1.0,
    seed: int | None = None,
) -> tuple[TrajectoryGroup | None, Sequence[Env] | None]:
    """Multi-turn rollout that returns (trajectory_group, envs).

    We can't use do_group_rollout here because it doesn't return the envs,
    and we need env access to read per-step scores for discounted returns.
    """
    policy = KernelBenchTokenCompleter(
        sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    envs = await env_group_builder.make_envs()
    rollout_results = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in envs],
        return_exceptions=True,
    )

    trajectories = []
    valid_envs: list[Env] = []
    for traj, env in zip(rollout_results, envs):
        if isinstance(traj, Exception):
            logger.warning(f"Rollout failed: {traj}")
        else:
            trajectories.append(traj)
            valid_envs.append(env)

    if not trajectories:
        logger.warning("All rollouts in group failed")
        return None, None

    # Final rewards are [0.0] because multi-turn rewards live in
    # transition.reward (set by env.step) and are later overwritten by
    # apply_discounted_returns. TrajectoryGroup.get_total_rewards() sums
    # transition.reward + final_reward, so final_reward must be zero.
    trajectory_group = TrajectoryGroup(
        trajectories,
        [0.0] * len(trajectories),
        [{}] * len(trajectories),
    )

    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups([trajectory_group])
        if len(trajectory_groups) == 0:
            return None, None
        trajectory_group = trajectory_groups[0]

    return trajectory_group, valid_envs


# ---------------------------------------------------------------------------
# Discounted returns
# ---------------------------------------------------------------------------


def apply_discounted_returns_to_trajectories(
    trajectory_groups: list[TrajectoryGroup],
    env_groups: list[Sequence[Env]],
    gamma: float,
    aggregation: str = "sum",
) -> None:
    """Replace per-step rewards with discounted returns for multi-turn training."""
    for tg, envs in zip(trajectory_groups, env_groups):
        for traj, env in zip(tg.trajectories_G, envs):
            if isinstance(env, MultiTurnKernelBenchEnv):
                step_scores = env.get_step_scores()
            else:
                step_scores = [t.reward for t in traj.transitions]

            if not step_scores:
                continue

            returns = compute_discounted_returns(step_scores, gamma, aggregation)
            for i, trans in enumerate(traj.transitions):
                if i < len(returns):
                    trans.reward = returns[i]


# ---------------------------------------------------------------------------
# Flatten and advantage estimation
# ---------------------------------------------------------------------------


def flatten_multiturn_trajectory_groups(
    trajectory_groups: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Flatten multi-turn trajectories so each turn is its own single-transition trajectory."""
    flattened = []
    for tg in trajectory_groups:
        new_trajectories = []
        for traj in tg.trajectories_G:
            for trans in traj.transitions:
                new_trajectories.append(
                    Trajectory(transitions=[trans], final_ob=tinker.ModelInput.empty())
                )

        # final_rewards must be 0.0 because get_total_rewards() sums
        # transition.reward + final_reward. The real rewards already live
        # in transition.reward (set by apply_discounted_returns).
        new_group = TrajectoryGroup(
            new_trajectories,
            [0.0] * len(new_trajectories),
            [{}] * len(new_trajectories),
        )
        flattened.append(new_group)
    return flattened


def compute_multiturn_advantages(
    trajectory_groups: list[TrajectoryGroup],
) -> list[torch.Tensor]:
    """GRPO advantage with std normalization.

    Expects flattened trajectory groups (each "trajectory" = one turn).
    Normalizes across all m*n samples per problem group.
    """
    advantages_P = []
    for tg in trajectory_groups:
        rewards = torch.tensor(tg.get_total_rewards())
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + 1e-9)
        advantages_P.append(advantages)
    return advantages_P


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_multiturn_trajectory_metrics(
    trajectory_groups: list[TrajectoryGroup],
    env_groups: list[Sequence[Env]],
) -> dict[str, Any]:
    """Compute aggregate metrics for multi-turn trajectories."""
    metrics: dict[str, Any] = {}

    turn_compiled: dict[int, list[float]] = defaultdict(list)
    turn_correct: dict[int, list[float]] = defaultdict(list)

    all_rewards = []
    all_num_turns = []
    all_success = []
    all_best_speedup = []

    all_format_ok = []
    all_compiled = []
    all_correct = []
    all_step_scores = []
    all_eval_times = []
    all_step_times = []

    for tg, envs in zip(trajectory_groups, env_groups):
        rewards = tg.get_total_rewards()
        all_rewards.extend(rewards)

        for traj, env in zip(tg.trajectories_G, envs):
            traj_speedups = []

            for trans in traj.transitions:
                if trans.metrics:
                    turn = trans.metrics.get("turn", 0)
                    compiled = trans.metrics.get("compiled", 0)
                    correct = trans.metrics.get("correctness", 0)
                    turn_compiled[turn].append(compiled)
                    turn_correct[turn].append(correct)

                    all_format_ok.append(trans.metrics.get("format_ok", 0))
                    all_compiled.append(compiled)
                    all_correct.append(correct)

                    if "step_score" in trans.metrics:
                        all_step_scores.append(trans.metrics["step_score"])
                    if "time/eval" in trans.metrics:
                        all_eval_times.append(trans.metrics["time/eval"])
                    if "time/step_total" in trans.metrics:
                        all_step_times.append(trans.metrics["time/step_total"])
                    if "speedup" in trans.metrics:
                        traj_speedups.append(trans.metrics["speedup"])

            if traj_speedups:
                all_best_speedup.append(max(traj_speedups))

            if isinstance(env, MultiTurnKernelBenchEnv):
                all_success.append(float(env.state.success))
                all_num_turns.append(env.state.turn_idx)

    if all_rewards:
        metrics["reward/discounted_mean"] = float(np.mean(all_rewards))
        metrics["reward/discounted_std"] = float(np.std(all_rewards))
        metrics["reward/discounted_min"] = float(np.min(all_rewards))
        metrics["reward/discounted_max"] = float(np.max(all_rewards))

    if all_format_ok:
        metrics["multiturn/format_rate"] = float(np.mean(all_format_ok))
    if all_compiled:
        metrics["multiturn/compile_rate"] = float(np.mean(all_compiled))
    if all_correct:
        metrics["multiturn/correct_rate"] = float(np.mean(all_correct))
    if all_format_ok:
        failures = [1.0 - (f and c and r) for f, c, r in zip(all_format_ok, all_compiled, all_correct)]
        metrics["multiturn/failure_rate"] = float(np.mean(failures))
    if all_step_scores:
        metrics["multiturn/raw_score_mean"] = float(np.mean(all_step_scores))
    if all_success:
        metrics["multiturn/success_rate"] = float(np.mean(all_success))
    if all_num_turns:
        metrics["multiturn/avg_turns"] = float(np.mean(all_num_turns))
    if all_best_speedup:
        metrics["multiturn/best_speedup_mean"] = float(np.mean(all_best_speedup))
    if all_eval_times:
        metrics["time/eval_mean"] = float(np.mean(all_eval_times))
    if all_step_times:
        metrics["time/step_mean"] = float(np.mean(all_step_times))

    for turn in sorted(turn_compiled.keys()):
        if turn_compiled[turn]:
            metrics[f"multiturn/turn_{turn}/compile_rate"] = float(
                np.mean(turn_compiled[turn])
            )
        if turn_correct[turn]:
            metrics[f"multiturn/turn_{turn}/correct_rate"] = float(
                np.mean(turn_correct[turn])
            )

    metrics["batch/num_groups"] = len(trajectory_groups)
    metrics["batch/num_trajectories"] = sum(
        len(tg.trajectories_G) for tg in trajectory_groups
    )
    metrics["batch/total_steps"] = len(all_step_scores)

    return metrics
