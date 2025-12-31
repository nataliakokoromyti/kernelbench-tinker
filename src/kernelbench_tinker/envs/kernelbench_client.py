"""
KernelBench client wrapper for evaluating generated kernels.

This module provides a Python API to KernelBench evaluation functionality,
allowing direct evaluation of kernel code without going through the CLI scripts.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
from collections import OrderedDict
import os
import sys
import re
import time
from dataclasses import dataclass, field
from typing import TypedDict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Kernel block pattern - code inside <KERNEL>...</KERNEL>
KERNEL_BLOCK_PATTERN = re.compile(
    r"<KERNEL>\s*```(?:cuda|python|cpp)?\s*\n?(.*?)```\s*</KERNEL>",
    re.DOTALL | re.IGNORECASE
)

# Fallback: just <KERNEL>...</KERNEL> without fenced code block
KERNEL_BLOCK_SIMPLE_PATTERN = re.compile(
    r"<KERNEL>(.*?)</KERNEL>",
    re.DOTALL | re.IGNORECASE
)


@dataclass
class ParsedResponse:
    """Parsed model response with kernel blocks."""
    kernel: str   # Kernel code (from <KERNEL> block or extracted code block)
    raw: str      # Original raw response
    format_ok: bool  # Whether we successfully extracted kernel code


def parse_structured_response(text: str) -> ParsedResponse:
    """
    Parse model response with structured format.

    Expected format:
        <think>
        Brief reasoning about optimization approach...
        </think>

        <KERNEL>
        ```cuda
        // CUDA kernel code
        ```
        </KERNEL>

    Also handles:
    - Missing <KERNEL> tags (falls back to extract_code_block)
    - Plain code without any tags

    Args:
        text: Raw model output

    Returns:
        ParsedResponse with kernel, raw text, and format_ok flag
    """
    raw = text
    kernel = ""

    # Try to extract kernel from <KERNEL> block
    kernel_match = KERNEL_BLOCK_PATTERN.search(text)
    if kernel_match:
        kernel = kernel_match.group(1).strip()
    else:
        # Try simple <KERNEL>...</KERNEL> without fenced code
        kernel_match = KERNEL_BLOCK_SIMPLE_PATTERN.search(text)
        if kernel_match:
            kernel = kernel_match.group(1).strip()
            # Try to extract code from within if it has fences
            inner_code = extract_code_block(kernel)
            if inner_code:
                kernel = inner_code

    # Fallback: no <KERNEL> tags, try generic code block extraction
    if not kernel:
        kernel = extract_code_block(text) or ""

    # No fallback - malformed responses should fail with format_ok=False
    # This incentivizes models to use proper <KERNEL>```...```</KERNEL> format

    # Check if we got valid kernel code
    format_ok = bool(kernel) and ("class ModelNew" in kernel or "def forward" in kernel)

    return ParsedResponse(
        kernel=kernel,
        raw=raw,
        format_ok=format_ok,
    )


class KernelEvalResult(TypedDict):
    """Result of evaluating a kernel against a reference implementation."""
    format_ok: bool  # Whether the kernel has valid format (code block extraction)
    compiled: bool  # Whether the kernel compiled successfully
    correctness: bool  # Whether all correctness tests passed
    tests_passed: int  # Number of correctness trials that passed
    tests_total: int  # Total number of correctness trials
    speedup: float | None  # Speedup vs baseline (if measured and correct)
    runtime_ms: float | None  # Kernel runtime in milliseconds
    baseline_runtime_ms: float | None  # Baseline runtime in milliseconds
    error_message: str | None  # Error message if any
    code_length: int  # Length of the kernel code in characters (for tie-breaking)
    metadata: dict[str, Any]  # Additional metadata from evaluation


def _ensure_kernelbench_imported() -> None:
    """Ensure KernelBench modules are importable."""
    kernelbench_root = os.environ.get("KERNELBENCH_ROOT", "/workspace/kernel_dev/KernelBench")
    if os.path.exists(kernelbench_root):
        kernelbench_src = os.path.join(kernelbench_root, "src")
        if os.path.isdir(kernelbench_src) and kernelbench_src not in sys.path:
            sys.path.insert(0, kernelbench_src)
        elif kernelbench_root not in sys.path:
            sys.path.insert(0, kernelbench_root)

    try:
        import kernelbench  # noqa: F401
        return
    except Exception:
        pass

    if not os.path.exists(kernelbench_root):
        raise RuntimeError(
            f"KernelBench not found at {kernelbench_root}. "
            "Set KERNELBENCH_ROOT environment variable or clone to /workspace/kernel_dev/KernelBench"
        )


def extract_code_block(text: str, languages: list[str] | None = None) -> str | None:
    """
    Extract the first code block from text.

    Args:
        text: The text containing code blocks
        languages: Optional list of language tags to look for (e.g., ["python", "cuda"])

    Returns:
        The extracted code or None if no valid code block found
    """
    if languages is None:
        languages = ["python", "cuda", "cpp", ""]

    # Try to find fenced code blocks
    for lang in languages:
        if lang:
            pattern = rf"```{lang}\s*\n(.*?)```"
        else:
            pattern = r"```\s*\n(.*?)```"

        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # If no fenced block, try to find code that looks like a Python module
    # (contains class definitions, imports, etc.)
    if "class ModelNew" in text or "def forward" in text:
        # Try to extract just the code portion
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith(("import ", "from ", "class ", "def ", "@")):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)

    return None


@functools.lru_cache(maxsize=1)
def _load_hf_kernelbench_dataset():
    """Load the HuggingFace KernelBench dataset once per process."""
    from datasets import load_dataset

    return load_dataset("ScalingIntelligence/KernelBench")


@functools.lru_cache(maxsize=16)
def _get_hf_level_data(level: int):
    """Get cached HF dataset split for a given level."""
    dataset = _load_hf_kernelbench_dataset()
    return dataset[f"level_{level}"]


@functools.lru_cache(maxsize=1024)
def get_reference_code(level: int, problem_id: int, dataset_src: str = "huggingface") -> str:
    """
    Get the reference PyTorch code for a problem.

    Args:
        level: KernelBench level (1, 2, 3, or 4)
        problem_id: Problem ID within the level
        dataset_src: Either "huggingface" or "local"

    Returns:
        The reference architecture source code
    """
    if dataset_src == "huggingface":
        level_data = _get_hf_level_data(level)

        # Fast path: problems are 1-indexed and usually stored sequentially
        try:
            row = level_data[problem_id - 1]
            if row["problem_id"] == problem_id:
                return row["code"]
        except Exception:
            pass  # Fall through to filter-based lookup

        # Slow path: filter by problem_id (needed when dataset order changes)
        problem_row = level_data.filter(
            lambda x: x["problem_id"] == problem_id,
            num_proc=None,
            desc=None
        )
        if len(problem_row) == 0:
            raise ValueError(f"Problem {problem_id} not found in level {level}")
        return problem_row["code"][0]
    else:
        _ensure_kernelbench_imported()
        from kernelbench.dataset import construct_kernelbench_dataset
        from kernelbench.utils import read_file

        dataset = construct_kernelbench_dataset(level)
        # problem_id is 1-indexed, dataset is 0-indexed
        problem_idx = problem_id - 1
        if problem_idx < 0 or problem_idx >= len(dataset):
            raise ValueError(f"Problem {problem_id} not found in level {level}")

        return read_file(dataset[problem_idx])


def get_prompt_for_problem(
    level: int,
    problem_id: int,
    backend: str = "triton",
    option: str = "one_shot",
    dataset_src: str = "huggingface",
    precision: str | None = None,
    include_hardware: bool = False,
    gpu_name: str | None = None,
) -> str:
    """
    Get the prompt for a KernelBench problem.

    Args:
        level: KernelBench level (1, 2, 3, or 4)
        problem_id: Problem ID within the level
        backend: Backend type ("cuda", "triton", "cute", "tilelang")
        option: Prompt option ("zero_shot", "one_shot", "few_shot")
        dataset_src: Either "huggingface" or "local"
        precision: Optional precision for prompt hints ("fp32", "fp16", "bf16")
        include_hardware: Whether to include hardware guidance blocks
        gpu_name: GPU identifier used when include_hardware is True (e.g., "A100")

    Returns:
        The prompt string for the model
    """
    ref_code = get_reference_code(level, problem_id, dataset_src)

    _ensure_kernelbench_imported()
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    prompt = get_prompt_for_backend(
        ref_code,
        backend,
        option=option,
        precision=precision,
        include_hardware=include_hardware,
        gpu_name=gpu_name,
    )

    return prompt


async def evaluate_kernel_async(
    level: int,
    problem_id: int,
    backend: str,
    kernel_code: str,
    dataset_src: str = "huggingface",
    num_correct_trials: int = 5,
    measure_performance: bool = False,
    num_perf_trials: int = 100,
    timing_method: str = "cuda_event",
    precision: str = "fp32",
    check_for_excessive_speedup: bool = True,
    excessive_speedup_threshold: float = 10.0,
    timeout: float = 120.0,
    cache_results: bool = True,
) -> KernelEvalResult:
    """
    Evaluate a generated kernel using Modal for isolated GPU execution.

    This function provides:
    - Hard timeout enforcement (kills bad kernels after timeout)
    - Process isolation (each kernel runs in separate container)
    - Protection against GPU corruption from bad kernels

    Args:
        level: KernelBench level (1, 2, 3, or 4)
        problem_id: Problem ID within the level
        backend: Backend type ("cuda", "triton", "cute", "tilelang")
        kernel_code: The generated kernel source code
        dataset_src: Either "huggingface" or "local"
        num_correct_trials: Number of correctness trials to run
        measure_performance: Whether to measure runtime performance
        num_perf_trials: Number of performance trials to run
        timeout: Timeout in seconds for evaluation (enforced by Modal)

    Returns:
        KernelEvalResult with evaluation results
    """
    from kernelbench_tinker.modal.evaluator import (
        ModalEvaluatorConfig,
        get_modal_evaluator,
    )
    t_total_start = time.perf_counter()
    timings: dict[str, float] = {}

    # Simple LRU cache to avoid re-evaluating identical kernels for the same problem.
    # We cache even failures to avoid repeatedly paying for hopeless kernels.
    _eval_cache: OrderedDict[str, KernelEvalResult] = getattr(
        evaluate_kernel_async, "_eval_cache", OrderedDict()
    )

    def _make_cache_key(code: str) -> str:
        h = hashlib.sha1(code.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"{level}:{problem_id}:{backend}:{dataset_src}:{h}"

    def _prune_cache(maxsize: int = 512) -> None:
        while len(_eval_cache) > maxsize:
            _eval_cache.popitem(last=False)

    # Default result for failures
    default_result: KernelEvalResult = {
        "format_ok": False,
        "compiled": False,
        "correctness": False,
        "tests_passed": 0,
        "tests_total": num_correct_trials,
        "speedup": None,
        "runtime_ms": None,
        "baseline_runtime_ms": None,
        "error_message": None,
        "code_length": len(kernel_code),
        "metadata": {},
    }

    # Check format - try to extract code if it's wrapped in markdown
    extracted_code = extract_code_block(kernel_code)
    if extracted_code is not None:
        kernel_code = extracted_code
        default_result["format_ok"] = True
        default_result["code_length"] = len(kernel_code)
    elif "class ModelNew" in kernel_code:
        default_result["format_ok"] = True
    else:
        default_result["error_message"] = "Could not extract valid kernel code from response"
        timings["total_eval_s"] = time.perf_counter() - t_total_start
        default_result["metadata"]["timings"] = timings
        return default_result

    # Cache lookup after format/cheating checks
    cache_key = _make_cache_key(kernel_code)
    if cache_results and cache_key in _eval_cache:
        cached = _eval_cache[cache_key].copy()
        cached["metadata"] = dict(cached.get("metadata", {}))
        cached["metadata"]["cache_hit"] = True
        cached["metadata"].setdefault("timings", timings)
        cached["metadata"]["timings"]["total_eval_s"] = time.perf_counter() - t_total_start
        # Move to end for LRU ordering
        _eval_cache.move_to_end(cache_key)
        return cached

    # Get reference code
    ref_start = time.perf_counter()
    try:
        ref_code = get_reference_code(level, problem_id, dataset_src)
    except Exception as e:
        default_result["error_message"] = f"Failed to load reference code: {e}"
        timings["reference_load_s"] = time.perf_counter() - ref_start
        timings["total_eval_s"] = time.perf_counter() - t_total_start
        default_result["metadata"]["timings"] = timings
        return default_result
    timings["reference_load_s"] = time.perf_counter() - ref_start

    # Get Modal evaluator with configured timeout
    config = ModalEvaluatorConfig(timeout=int(timeout))
    evaluator = get_modal_evaluator(config)

    # Run evaluation on Modal
    modal_start = time.perf_counter()
    try:
        result = await evaluator.evaluate_single_batched(
            ref_code=ref_code,
            kernel_code=kernel_code,
            backend=backend,
            num_correct_trials=num_correct_trials,
            measure_performance=measure_performance,
            num_perf_trials=num_perf_trials,
            timing_method=timing_method,
            precision=precision,
            check_for_excessive_speedup=check_for_excessive_speedup,
            excessive_speedup_threshold=excessive_speedup_threshold,
        )

        if cache_results:
            result_copy = result.copy()
            _eval_cache[cache_key] = result_copy
            _prune_cache()

        timings["modal_eval_s"] = time.perf_counter() - modal_start
        timings["total_eval_s"] = time.perf_counter() - t_total_start
        result_metadata = result.get("metadata", {}) or {}
        result_metadata.setdefault("timings", {}).update(timings)
        result["metadata"] = result_metadata
        logger.debug(
            "Modal eval timings level=%s problem=%s ref_load=%.3fs modal=%.3fs total=%.3fs",
            level,
            problem_id,
            timings.get("reference_load_s", 0.0),
            timings.get("modal_eval_s", 0.0),
            timings.get("total_eval_s", 0.0),
        )
        return result

    except Exception as e:
        default_result["error_message"] = f"Modal evaluation failed: {e}"
        logger.exception("Modal kernel evaluation failed")
        timings["modal_eval_s"] = time.perf_counter() - modal_start
        timings["total_eval_s"] = time.perf_counter() - t_total_start
        default_result["metadata"]["timings"] = timings
        if cache_results:
            default_copy = default_result.copy()
            _eval_cache[cache_key] = default_copy
            _prune_cache()
        return default_result

# Attach cache container to function for reuse
evaluate_kernel_async._eval_cache = OrderedDict()  # type: ignore[attr-defined]


async def evaluate_kernel_batch_async(
    evaluations: list[dict],
    timeout: float = 180.0,
) -> list[KernelEvalResult]:
    """
    Evaluate multiple kernels in parallel using Modal.

    This function runs all kernel evaluations in parallel across
    Modal's GPU pool for maximum throughput.

    Args:
        evaluations: List of dicts with keys:
            - level: int
            - problem_id: int
            - backend: str
            - kernel_code: str
            - dataset_src: str (optional)
            - num_correct_trials: int (optional)
            - measure_performance: bool (optional)
            - num_perf_trials: int (optional)
        timeout: Timeout in seconds per evaluation

    Returns:
        List of KernelEvalResult dicts in same order as input
    """
    from kernelbench_tinker.modal.evaluator import (
        ModalEvaluatorConfig,
        get_modal_evaluator,
    )

    if not evaluations:
        return []

    # Prepare evaluations with ref_code
    prepared = []
    results = []

    for i, e in enumerate(evaluations):
        kernel_code = e["kernel_code"]

        # Create default result
        default_result: KernelEvalResult = {
            "format_ok": False,
            "compiled": False,
            "correctness": False,
            "tests_passed": 0,
            "tests_total": e.get("num_correct_trials", 5),
            "speedup": None,
            "runtime_ms": None,
            "baseline_runtime_ms": None,
            "error_message": None,
            "code_length": len(kernel_code),
            "metadata": {},
        }

        # Check format
        extracted_code = extract_code_block(kernel_code)
        if extracted_code is not None:
            kernel_code = extracted_code
            default_result["format_ok"] = True
            default_result["code_length"] = len(kernel_code)
        elif "class ModelNew" in kernel_code:
            default_result["format_ok"] = True
        else:
            default_result["error_message"] = "Could not extract valid kernel code"
            results.append((i, default_result, None))
            continue

        # Get reference code
        try:
            ref_code = get_reference_code(
                e["level"],
                e["problem_id"],
                e.get("dataset_src", "huggingface"),
            )
        except Exception as ex:
            default_result["error_message"] = f"Failed to load reference: {ex}"
            results.append((i, default_result, None))
            continue

        # Add to batch for Modal evaluation
        prepared.append({
            "index": i,
            "ref_code": ref_code,
            "kernel_code": kernel_code,
            "backend": e.get("backend", "triton"),
            "num_correct_trials": e.get("num_correct_trials", 5),
            "measure_performance": e.get("measure_performance", False),
            "num_perf_trials": e.get("num_perf_trials", 100),
            "timing_method": e.get("timing_method", "cuda_event"),
            "precision": e.get("precision", "fp32"),
            "check_for_excessive_speedup": e.get("check_for_excessive_speedup", True),
            "excessive_speedup_threshold": e.get("excessive_speedup_threshold", 10.0),
        })

    # Run Modal batch evaluation
    if prepared:
        config = ModalEvaluatorConfig(timeout=int(timeout))
        evaluator = get_modal_evaluator(config)

        try:
            modal_results = await evaluator.evaluate_batch(prepared)

            for prep, modal_result in zip(prepared, modal_results):
                results.append((prep["index"], modal_result, None))

        except Exception as e:
            logger.exception("Modal batch evaluation failed")
            for prep in prepared:
                error_result: KernelEvalResult = {
                    "format_ok": True,
                    "compiled": False,
                    "correctness": False,
                    "tests_passed": 0,
                    "tests_total": prep.get("num_correct_trials", 5),
                    "speedup": None,
                    "runtime_ms": None,
                    "baseline_runtime_ms": None,
                    "error_message": f"Modal batch failed: {e}",
                    "code_length": len(prep["kernel_code"]),
                    "metadata": {},
                }
                results.append((prep["index"], error_result, None))

    # Sort by original index and return
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def get_problem_count(level: int, dataset_src: str = "huggingface") -> int:
    """Get the number of problems in a level."""
    if dataset_src == "huggingface":
        level_data = _get_hf_level_data(level)
        return len(level_data)
    else:
        _ensure_kernelbench_imported()
        from kernelbench.dataset import construct_kernelbench_dataset
        return len(construct_kernelbench_dataset(level))


def get_problem_ids(
    level: int,
    start: int | None = None,
    end: int | None = None,
    dataset_src: str = "huggingface",
) -> list[int]:
    """
    Get list of problem IDs for a level.

    Args:
        level: KernelBench level
        start: Start problem ID (inclusive, 1-indexed)
        end: End problem ID (inclusive, 1-indexed)
        dataset_src: Either "huggingface" or "local"

    Returns:
        List of problem IDs
    """
    total = get_problem_count(level, dataset_src)

    if start is None:
        start = 1
    if end is None:
        end = total

    return list(range(start, min(end, total) + 1))


@dataclass
class KernelBenchProblem:
    """Represents a single KernelBench problem."""
    level: int
    problem_id: int
    backend: str = "triton"
    dataset_src: str = "huggingface"
    prompt_option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot"
    prompt_precision: str | None = None
    prompt_include_hardware: bool = False
    prompt_gpu_name: str | None = None

    _ref_code: str | None = field(default=None, repr=False)
    _prompt: str | None = field(default=None, repr=False)

    @property
    def ref_code(self) -> str:
        """Get the reference PyTorch code (cached)."""
        if self._ref_code is None:
            self._ref_code = get_reference_code(
                self.level, self.problem_id, self.dataset_src
            )
        return self._ref_code

    @property
    def prompt(self) -> str:
        """Get the prompt for this problem (cached)."""
        if self._prompt is None:
            self._prompt = get_prompt_for_problem(
                self.level,
                self.problem_id,
                self.backend,
                option=self.prompt_option,
                dataset_src=self.dataset_src,
                precision=self.prompt_precision,
                include_hardware=self.prompt_include_hardware,
                gpu_name=self.prompt_gpu_name,
            )
        return self._prompt

