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
from typing import TypedDict, Optional, Any, TYPE_CHECKING
import logging

import torch

if TYPE_CHECKING:
    from kernel_rl.rag.retriever import KernelRetriever

logger = logging.getLogger(__name__)

# Regex patterns for parsing structured output (Kevin/Qwen3 style)
# Thinking block patterns - supports <think>, <thinking>, <THOUGHT> variants
THINKING_PATTERN = re.compile(
    r"<(?:think|thinking|THOUGHT)>(.*?)</(?:think|thinking|THOUGHT)>",
    re.DOTALL | re.IGNORECASE
)

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
    """Parsed model response with thinking and kernel blocks."""
    thought: str  # Content from <think>/<THOUGHT> block (may be empty)
    kernel: str   # Kernel code (from <KERNEL> block or extracted code block)
    raw: str      # Original raw response
    format_ok: bool  # Whether we successfully extracted kernel code


def parse_structured_response(text: str) -> ParsedResponse:
    """
    Parse model response with Kevin/Qwen3 structured format.

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
    - <thinking>...</thinking> and <THOUGHT>...</THOUGHT> variants
    - Missing thinking block (thought will be empty string)
    - Missing <KERNEL> tags (falls back to extract_code_block)
    - Plain code without any tags

    Args:
        text: Raw model output

    Returns:
        ParsedResponse with thought, kernel, raw text, and format_ok flag
    """
    raw = text
    thought = ""
    kernel = ""

    # Extract thinking block (optional)
    think_match = THINKING_PATTERN.search(text)
    if think_match:
        thought = think_match.group(1).strip()
        # Remove thinking block from text for kernel extraction
        text = THINKING_PATTERN.sub("", text).strip()

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
        thought=thought,
        kernel=kernel,
        raw=raw,
        format_ok=format_ok,
    )


def strip_thinking_tokens(text: str) -> str:
    """
    Strip thinking/reasoning tokens from model output.

    DEPRECATED: Use parse_structured_response() instead for proper handling.
    This function is kept for backwards compatibility.

    Args:
        text: Raw model output text

    Returns:
        Text with thinking tokens removed
    """
    return THINKING_PATTERN.sub("", text).strip()


# Global retriever instance (lazy-loaded with async lock for thread safety)
_global_retriever: "KernelRetriever | None" = None
_retriever_lock: asyncio.Lock | None = None  # Lazily initialized to avoid "no running event loop"


def _get_retriever_lock() -> asyncio.Lock:
    """Get or create the retriever lock (must be called from async context)."""
    global _retriever_lock
    if _retriever_lock is None:
        _retriever_lock = asyncio.Lock()
    return _retriever_lock


async def get_global_retriever(index_path: str | None = None) -> "KernelRetriever | None":
    """Get or load the global RAG retriever (async, thread-safe)."""
    global _global_retriever

    # Fast path: already loaded
    if _global_retriever is not None:
        return _global_retriever

    # Slow path: need to load (protected by lock)
    if index_path:
        lock = _get_retriever_lock()
        async with lock:
            # Double-check after acquiring lock
            if _global_retriever is None:
                from kernel_rl.rag.retriever import KernelRetriever
                logger.info(f"Loading RAG index from {index_path}")
                _global_retriever = KernelRetriever.load(index_path)

    return _global_retriever


def set_global_retriever(retriever: "KernelRetriever") -> None:
    """Set the global RAG retriever."""
    global _global_retriever
    _global_retriever = retriever


def get_global_retriever_sync() -> "KernelRetriever | None":
    """
    Get the global RAG retriever synchronously.

    This returns the cached retriever without attempting to load.
    Use this in sync code paths where the retriever was already set
    via set_global_retriever().
    """
    return _global_retriever


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
    cheated: bool  # Whether the kernel cheated (e.g., just calls PyTorch)
    error_message: str | None  # Error message if any
    code_length: int  # Length of the kernel code in characters (for tie-breaking)
    metadata: dict[str, Any]  # Additional metadata from evaluation


def _ensure_kernelbench_imported() -> None:
    """Ensure KernelBench modules are importable."""
    kernelbench_root = os.environ.get("KERNELBENCH_ROOT", "/workspace/KernelBench")

    if not os.path.exists(kernelbench_root):
        raise RuntimeError(
            f"KernelBench not found at {kernelbench_root}. "
            "Set KERNELBENCH_ROOT environment variable or clone to /workspace/KernelBench"
        )

    # Add KernelBench to path if not already there
    if kernelbench_root not in sys.path:
        sys.path.insert(0, kernelbench_root)


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


def check_for_cheating(kernel_code: str) -> bool:
    """
    Check if the kernel code is cheating using Kevin-32B paper rules.

    Kevin paper identifies three cheating patterns that receive zero reward:
    1. PyTorch wrapping: Using torch.nn or torch.nn.functional
    2. Try-except fallback: Wrapping incomplete CUDA in exception handlers
    3. Inheritance bypass: Using 'pass' to inherit from reference

    Reference: arXiv:2507.11948 "Kevin: Multi-Turn RL for Generating CUDA Kernels"
    """
    # Kevin Rule 1: PyTorch wrapping detection
    # Zero reward for kernels containing torch.nn or torch.nn.functional
    pytorch_patterns = [
        "torch.nn.functional",
        "torch.nn.",
        "F.conv", "F.linear", "F.relu", "F.gelu", "F.softmax",
        "F.batch_norm", "F.layer_norm", "F.dropout",
        "nn.functional.",
    ]
    for pattern in pytorch_patterns:
        if pattern in kernel_code:
            return True

    # Kevin Rule 2: Try-except fallback pattern detection
    # Models wrap incomplete CUDA in exception handlers that fall back to PyTorch
    # Zero reward for kernels containing try/except
    if "try:" in kernel_code or "except:" in kernel_code or "except " in kernel_code:
        return True

    # Kevin Rule 3: Inheritance bypass detection
    # Model inherits from reference using 'pass'
    # Zero reward for kernels containing pass statements
    # Be careful to not match 'passed' or similar words
    import re
    if re.search(r'\bpass\b', kernel_code):
        return True

    # Additional check: no custom kernel implementation at all
    has_triton_kernel = "@triton.jit" in kernel_code or "@triton.autotune" in kernel_code
    has_cuda_kernel = "load_inline" in kernel_code or "cpp_extension" in kernel_code
    has_cute_kernel = "cute::" in kernel_code or "from cutlass" in kernel_code
    has_tilelang = "@T.prim_func" in kernel_code or "tvm.build" in kernel_code

    has_custom_implementation = any([
        has_triton_kernel,
        has_cuda_kernel,
        has_cute_kernel,
        has_tilelang
    ])

    # If no custom implementation, check for direct torch ops
    if not has_custom_implementation:
        torch_ops = [
            "torch.mm", "torch.bmm", "torch.matmul",
            "torch.conv", "torch.einsum",
        ]
        for op in torch_ops:
            if op in kernel_code:
                return True

    return False


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
        from src.dataset import construct_kernelbench_dataset
        from src.utils import read_file

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
    raicl_k: int = 3,
) -> str:
    """
    Get the prompt for a KernelBench problem.

    Args:
        level: KernelBench level (1, 2, 3, or 4)
        problem_id: Problem ID within the level
        backend: Backend type ("cuda", "triton", "cute", "tilelang")
        option: Prompt option ("zero_shot", "one_shot", "few_shot", "raicl")
        dataset_src: Either "huggingface" or "local"
        raicl_k: Number of examples to retrieve for RA-ICL

    Returns:
        The prompt string for the model
    """
    ref_code = get_reference_code(level, problem_id, dataset_src)

    # Handle RA-ICL option - doesn't require local KernelBench
    if option == "raicl":
        return get_raicl_prompt_for_code(ref_code, backend, k=raicl_k)

    # Non-raicl options require local KernelBench prompt constructor
    _ensure_kernelbench_imported()
    from src.prompt_constructor_toml import get_prompt_for_backend

    prompt = get_prompt_for_backend(
        ref_code,
        backend,
        option=option,
        precision="fp32",
        include_hardware=False,
    )

    return prompt


def get_raicl_prompt_for_code(
    ref_code: str,
    backend: str,
    k: int = 3,
) -> str:
    """
    Get RA-ICL prompt for given reference code.

    Args:
        ref_code: Reference PyTorch code
        backend: Target backend ("triton" or "cuda")
        k: Number of examples to retrieve

    Returns:
        RA-ICL prompt with retrieved examples
    """
    retriever = get_global_retriever_sync()

    if retriever is None:
        logger.warning("RAG retriever not loaded, falling back to one_shot")
        from src.prompt_constructor_toml import get_prompt_for_backend
        return get_prompt_for_backend(
            ref_code,
            backend,
            option="one_shot",
            precision="fp32",
            include_hardware=False,
        )

    from kernel_rl.rag.prompt_builder import RAICLPromptBuilder

    builder = RAICLPromptBuilder(retriever)
    return builder.build_prompt(ref_code, backend, k=k)


def evaluate_kernel(
    level: int,
    problem_id: int,
    backend: str,
    kernel_code: str,
    dataset_src: str = "huggingface",
    num_correct_trials: int = 5,
    measure_performance: bool = False,
    num_perf_trials: int = 100,
    device: torch.device | None = None,
    timeout: float = 180.0,
) -> KernelEvalResult:
    """
    Evaluate a generated kernel against the reference implementation.

    Args:
        level: KernelBench level (1, 2, 3, or 4)
        problem_id: Problem ID within the level
        backend: Backend type ("cuda", "triton", "cute", "tilelang")
        kernel_code: The generated kernel source code
        dataset_src: Either "huggingface" or "local"
        num_correct_trials: Number of correctness trials to run
        measure_performance: Whether to measure runtime performance
        num_perf_trials: Number of performance trials to run
        device: CUDA device to use (defaults to cuda:0)
        timeout: Timeout in seconds for evaluation

    Returns:
        KernelEvalResult with evaluation results
    """
    _ensure_kernelbench_imported()

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
        "cheated": False,
        "error_message": None,
        "code_length": len(kernel_code),
        "metadata": {},
    }

    # Check format - try to extract code if it's wrapped in markdown
    extracted_code = extract_code_block(kernel_code)
    if extracted_code is not None:
        kernel_code = extracted_code
        default_result["format_ok"] = True
        default_result["code_length"] = len(kernel_code)  # Update with extracted length
    elif "class ModelNew" in kernel_code:
        # Code looks valid even without markdown wrapper
        default_result["format_ok"] = True
    else:
        default_result["error_message"] = "Could not extract valid kernel code from response"
        return default_result

    # Check for cheating
    cheated = check_for_cheating(kernel_code)
    default_result["cheated"] = cheated

    # Get reference code
    try:
        ref_code = get_reference_code(level, problem_id, dataset_src)
    except Exception as e:
        default_result["error_message"] = f"Failed to load reference code: {e}"
        return default_result

    # Set up device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            default_result["error_message"] = "No CUDA device available"
            return default_result

    # Import evaluation function
    from src.eval import eval_kernel_against_ref, get_torch_dtype_from_string

    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_code,
            custom_model_src=kernel_code,
            measure_performance=measure_performance,
            verbose=False,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=None,
            device=device,
            backend=backend,
            precision=get_torch_dtype_from_string("fp32"),
        )

        if result is None:
            default_result["error_message"] = "Evaluation returned None (possible lock file error)"
            return default_result

        # Parse the result
        runtime_ms = result.runtime if result.runtime > 0 else None
        baseline_runtime_ms = (
            result.metadata.get("baseline_runtime_ms")
            or result.metadata.get("baseline_runtime")
            or None
        )
        eval_result: KernelEvalResult = {
            "format_ok": True,
            "compiled": result.compiled,
            "correctness": result.correctness,
            "tests_passed": 0,
            "tests_total": num_correct_trials,
            "speedup": None,
            "runtime_ms": runtime_ms,
            "baseline_runtime_ms": baseline_runtime_ms,
            "cheated": cheated,
            "error_message": None,
            "metadata": result.metadata,
        }

        # Parse correctness trials from metadata
        if "correctness_trials" in result.metadata:
            trials_str = result.metadata["correctness_trials"]
            # Format is "(X / Y)"
            match = re.match(r"\((\d+)\s*/\s*(\d+)\)", trials_str)
            if match:
                eval_result["tests_passed"] = int(match.group(1))
                eval_result["tests_total"] = int(match.group(2))

        # If correct and we have baseline/runtime, calculate speedup
        if (
            eval_result["correctness"]
            and eval_result["runtime_ms"] is not None
            and baseline_runtime_ms
            and baseline_runtime_ms > 0
        ):
            eval_result["speedup"] = baseline_runtime_ms / eval_result["runtime_ms"]

        # Check for errors in metadata
        if "runtime_error" in result.metadata:
            eval_result["error_message"] = str(result.metadata.get("runtime_error", ""))
        elif "compilation_error" in result.metadata:
            eval_result["error_message"] = str(result.metadata.get("compilation_error", ""))

        return eval_result

    except Exception as e:
        default_result["error_message"] = f"Evaluation failed: {e}"
        logger.exception("Kernel evaluation failed")
        return default_result


async def evaluate_kernel_async(
    level: int,
    problem_id: int,
    backend: str,
    kernel_code: str,
    dataset_src: str = "huggingface",
    num_correct_trials: int = 5,
    measure_performance: bool = False,
    num_perf_trials: int = 100,
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
    from kernel_rl.modal.evaluator import (
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
        "cheated": False,
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

    # Check for cheating
    cheated = check_for_cheating(kernel_code)
    default_result["cheated"] = cheated

    # If cheated, return early with zero reward
    if cheated:
        default_result["error_message"] = "Kernel detected as cheating (uses PyTorch ops directly)"
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
            precision="fp32",
        )

        # Add cheating flag (checked locally, not on Modal)
        result["cheated"] = cheated

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
    from kernel_rl.modal.evaluator import (
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
            "cheated": False,
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

        # Check for cheating
        cheated = check_for_cheating(kernel_code)
        default_result["cheated"] = cheated

        if cheated:
            default_result["error_message"] = "Kernel detected as cheating"
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
            "precision": "fp32",
            "cheated": cheated,
        })

    # Run Modal batch evaluation
    if prepared:
        config = ModalEvaluatorConfig(timeout=int(timeout))
        evaluator = get_modal_evaluator(config)

        try:
            modal_results = await evaluator.evaluate_batch(prepared)

            for prep, modal_result in zip(prepared, modal_results):
                modal_result["cheated"] = prep["cheated"]
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
                    "cheated": prep["cheated"],
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
        from src.dataset import construct_kernelbench_dataset
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
    prompt_option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot", "raicl"
    raicl_k: int = 3  # Number of examples for RA-ICL

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
                raicl_k=self.raicl_k,
            )
        return self._prompt

    def get_raicl_system_prompt(self) -> str:
        """Get the RA-ICL system prompt for this backend."""
        from kernel_rl.rag.prompt_builder import RAICLPromptBuilder
        retriever = get_global_retriever_sync()
        if retriever is None:
            return ""
        builder = RAICLPromptBuilder(retriever)
        return builder.build_system_prompt(self.backend)

    def evaluate(
        self,
        kernel_code: str,
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        device: torch.device | None = None,
    ) -> KernelEvalResult:
        """Evaluate a kernel solution for this problem."""
        return evaluate_kernel(
            level=self.level,
            problem_id=self.problem_id,
            backend=self.backend,
            kernel_code=kernel_code,
            dataset_src=self.dataset_src,
            num_correct_trials=num_correct_trials,
            measure_performance=measure_performance,
            device=device,
        )
