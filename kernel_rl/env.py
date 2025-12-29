"""Environment variable loading utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load environment variables from .env file.

    Searches for .env file in the following order:
    1. Current working directory
    2. kernel-rl package root directory

    Environment variables already set will NOT be overwritten.
    """
    # Try current working directory first
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)
        logger.debug(f"Loaded environment from {cwd_env}")
        return

    # Try package root directory
    package_root = Path(__file__).parent.parent
    root_env = package_root / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
        logger.debug(f"Loaded environment from {root_env}")
        return

    logger.debug("No .env file found")


def setup_environment() -> None:
    """Load .env file and set up environment variables.

    This function:
    1. Loads variables from .env file (if present)
    2. Auto-detects KERNELBENCH_ROOT if not set
    3. Warns if TINKER_API_KEY is not set
    """
    # Load .env file first
    load_env()

    # Set KERNELBENCH_ROOT if not already set
    if "KERNELBENCH_ROOT" not in os.environ:
        candidates = [
            "/workspace/KernelBench",
            "/workspace/kernel_dev/KernelBench",
            os.path.expanduser("~/KernelBench"),
            str(Path(__file__).parent.parent.parent / "KernelBench"),
        ]
        for path in candidates:
            if os.path.exists(path):
                os.environ["KERNELBENCH_ROOT"] = os.path.abspath(path)
                logger.info(f"Set KERNELBENCH_ROOT={os.environ['KERNELBENCH_ROOT']}")
                break

    # Check that TINKER_API_KEY is set
    if "TINKER_API_KEY" not in os.environ:
        logger.warning(
            "TINKER_API_KEY not set. You'll need this to use the Tinker API. "
            "Get your key from https://console.tinker.thinkingmachines.ai"
        )
