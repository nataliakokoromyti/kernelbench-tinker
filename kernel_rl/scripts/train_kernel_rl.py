#!/usr/bin/env python3
"""
CLI entrypoint for KernelBench RL training.

Usage:
    uv run python -m kernel_rl.scripts.train_kernel_rl \
        model_name=Qwen/Qwen2.5-Coder-7B-Instruct \
        log_path=./runs/kernel_rl_v1 \
        dataset_builder.level=1 \
        dataset_builder.batch_size=4 \
        dataset_builder.group_size=4

With a config file:
    uv run python -m kernel_rl.scripts.train_kernel_rl \
        --config kernel_rl/config/rl_kernelbench.yaml \
        log_path=./runs/my_experiment
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

import chz
import yaml

from kernel_rl.env import setup_environment
from kernel_rl.training.loop import TrainingConfig, main as train_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary with dot notation keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load and flatten a YAML config file for use with chz Blueprint."""
    with open(path) as f:
        config = yaml.safe_load(f)
    # Flatten nested dicts to dot notation for chz
    return flatten_dict(config) if config else {}


def main():
    """Main entry point."""
    setup_environment()

    # Check for --config flag
    argv = sys.argv[1:]
    config_file = None
    remaining_argv = []

    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_file = argv[i + 1]
            i += 2
        elif argv[i].startswith("--config="):
            config_file = argv[i].split("=", 1)[1]
            i += 1
        else:
            remaining_argv.append(argv[i])
            i += 1

    # Create blueprint
    blueprint = chz.Blueprint(TrainingConfig)

    # Apply config file first (if provided)
    if config_file:
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            sys.exit(1)
        logger.info(f"Loading config from: {config_file}")
        config_values = load_yaml_config(config_file)
        # Filter out None values
        config_values = {k: v for k, v in config_values.items() if v is not None}
        blueprint.apply(config_values, layer_name="config file")

    # Apply CLI arguments on top (override config file)
    if remaining_argv:
        blueprint.apply_from_argv(remaining_argv, layer_name="command line")

    # Check for help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(blueprint.get_help())
        sys.exit(0)

    # Build config
    cfg = blueprint.make()

    logger.info("Starting KernelBench RL Training")
    logger.info(f"Mode: {cfg.mode}")
    if cfg.mode == "multi_turn":
        logger.info(f"  Max turns: {cfg.max_turns}")
        logger.info(f"  Gamma (discount): {cfg.gamma}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Level: {cfg.dataset_builder.level}")
    logger.info(f"Batch size: {cfg.dataset_builder.batch_size}")
    logger.info(f"Group size: {cfg.dataset_builder.group_size}")
    logger.info(f"Log path: {cfg.log_path}")

    # Run training
    asyncio.run(train_main(cfg))


if __name__ == "__main__":
    main()
