"""
Local in-process kernel evaluator with subprocess isolation.

This module provides the same public API as `kernelbench_tinker.modal.evaluator`
(`evaluate_single`, `evaluate_batch`, `evaluate_single_batched`) but runs
`eval_kernel_against_ref` in an isolated subprocess on the local GPU instead of
shipping work to a Modal container.

Why a subprocess per kernel?
    KernelBench compiles each kernel via torch.utils.cpp_extension.load_inline,
    which leaks memory across compilations and quickly OOMs an MI350X with 192GB
    of HBM after a couple hundred problems. Running each eval in a fresh Python
    interpreter is the same trick used by eval_hip.py and is the only reliable
    way to sweep all 300 KernelBench problems on AMD.

Result shape matches Modal's KernelEvalResult so the rest of the pipeline
(reward computation, multi-turn feedback parsing, eval_kernel_rl) needs no
changes when the dispatch flips between backends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)


# Sentinel markers used by the worker to delimit the JSON result so we can
# robustly extract it from a noisy stdout (HIP/ROCm prints a lot of warnings).
_RESULT_BEGIN = "__KB_TINKER_LOCAL_RESULT_BEGIN__"
_RESULT_END = "__KB_TINKER_LOCAL_RESULT_END__"


# Subprocess worker. Reads a JSON request from stdin, runs one eval, prints
# the result between sentinel markers, then exits. Stays self-contained so
# it can be exec'd via `python -c` without bundling files.
#
# Mirrors the result-shape conversion in src/kernelbench_tinker/modal/app.py
# (correctness_trials regex, ref_runtime → speedup, runtime > 0 guard).
_WORKER_SCRIPT = r'''
import json, os, re, sys, tempfile, traceback

def _emit(payload):
    sys.stdout.write("__KB_TINKER_LOCAL_RESULT_BEGIN__\n")
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.write("\n__KB_TINKER_LOCAL_RESULT_END__\n")
    sys.stdout.flush()

raw = sys.stdin.read()
req = json.loads(raw)
num_correct_trials = req.get("num_correct_trials", 5)
kernel_code = req.get("kernel_code", "")

# Per-subprocess torch extension build dir, isolated from any stale lock
# files left behind by a SIGKILLed predecessor.
_build_dir = tempfile.mkdtemp(prefix="kb_tinker_ext_")
os.environ["TORCH_EXTENSIONS_DIR"] = _build_dir

try:
    # Configure GPU arch BEFORE importing torch so AOT compile paths see it.
    # set_gpu_arch is pure os.environ manipulation, no torch dependency.
    gpu_arch = req.get("gpu_arch") or []
    if gpu_arch:
        from kernelbench.utils import set_gpu_arch
        set_gpu_arch(gpu_arch)

    import torch
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

    dtype = get_torch_dtype_from_string(req.get("precision", "fp32"))

    if not torch.cuda.is_available():
        _emit({
            "format_ok": True,
            "compiled": False, "correctness": False,
            "tests_passed": 0, "tests_total": num_correct_trials,
            "speedup": None, "runtime_ms": None, "baseline_runtime_ms": None,
            "error_message": "torch.cuda not available in worker subprocess",
            "code_length": len(kernel_code),
            "metadata": {},
        })
        sys.exit(0)

    device = torch.device("cuda:0")
    result = eval_kernel_against_ref(
        original_model_src=req["ref_code"],
        custom_model_src=kernel_code,
        num_correct_trials=num_correct_trials,
        num_perf_trials=req.get("num_perf_trials", 100),
        measure_performance=req.get("measure_performance", False),
        verbose=False,
        device=device,
        backend=req.get("backend", "triton"),
        precision=dtype,
    )
except Exception as e:
    # Anything raised before/around eval_kernel_against_ref is a worker
    # framework failure (OOM, driver crash, KernelBench bug, ...). Report
    # it explicitly so it can't be confused with a kernel-side failure.
    _emit({
        "format_ok": True,
        "compiled": False, "correctness": False,
        "tests_passed": 0, "tests_total": num_correct_trials,
        "speedup": None, "runtime_ms": None, "baseline_runtime_ms": None,
        "error_message": f"Worker framework error: {e}",
        "code_length": len(kernel_code),
        "metadata": {"traceback": traceback.format_exc()[-1000:]},
    })
    sys.exit(0)

# Result marshalling — keep narrow so a marshalling bug never masquerades
# as a user-kernel failure. Mirrors modal/app.py:300-370.
try:
    metadata = {k: str(v)[:500] for k, v in dict(getattr(result, "metadata", {}) or {}).items()}

    tests_passed = 0
    trials_str = metadata.get("correctness_trials", "(0 / 0)")
    m = re.match(r"\((\d+)\s*/\s*(\d+)\)", trials_str)
    if m:
        tests_passed = int(m.group(1))
    elif result.correctness:
        tests_passed = num_correct_trials

    runtime_raw = getattr(result, "runtime", -1.0)
    runtime_ms = float(runtime_raw) if runtime_raw and runtime_raw > 0 else None

    baseline_runtime_ms = None
    ref_runtime = getattr(result, "ref_runtime", None)
    if ref_runtime and ref_runtime > 0:
        baseline_runtime_ms = float(ref_runtime)

    speedup = None
    if result.correctness and runtime_ms and baseline_runtime_ms and baseline_runtime_ms > 0:
        speedup = baseline_runtime_ms / runtime_ms

    error_message = None
    for key in ("runtime_error", "compilation_error", "correctness_issue", "max_difference"):
        if metadata.get(key):
            error_message = str(metadata[key])[:1000]
            break

    _emit({
        "format_ok": True,
        "compiled": bool(result.compiled),
        "correctness": bool(result.correctness),
        "tests_passed": tests_passed,
        "tests_total": num_correct_trials,
        "speedup": speedup,
        "runtime_ms": runtime_ms,
        "baseline_runtime_ms": baseline_runtime_ms,
        "error_message": error_message,
        "code_length": len(kernel_code),
        "metadata": metadata,
    })
except Exception as e:
    _emit({
        "format_ok": True,
        "compiled": bool(getattr(result, "compiled", False)),
        "correctness": bool(getattr(result, "correctness", False)),
        "tests_passed": 0, "tests_total": num_correct_trials,
        "speedup": None, "runtime_ms": None, "baseline_runtime_ms": None,
        "error_message": f"Worker result-marshalling error: {e}",
        "code_length": len(kernel_code),
        "metadata": {"traceback": traceback.format_exc()[-1000:]},
    })
'''


@dataclass
class LocalEvaluatorConfig:
    """Configuration for the local subprocess-isolated kernel evaluator."""

    enabled: bool = True
    timeout: int = 300  # seconds per kernel subprocess
    # gpu_arch is passed to set_gpu_arch() inside the worker. Must be one of
    # the architectures KernelBench's set_gpu_arch() recognizes:
    #   ["gfx950"]   -> MI350X
    #   ["gfx942"]   -> MI300X
    #   ["Ampere"]   -> A100
    #   ["Hopper"]   -> H100
    gpu_arch: list[str] = field(default_factory=list)
    max_concurrent: int = 1  # Serialize subprocesses by default (1 GPU)
    return_exceptions: bool = True


def _make_default_result(num_correct_trials: int, kernel_code: str, error: str) -> dict[str, Any]:
    return {
        "format_ok": True,
        "compiled": False,
        "correctness": False,
        "tests_passed": 0,
        "tests_total": num_correct_trials,
        "speedup": None,
        "runtime_ms": None,
        "baseline_runtime_ms": None,
        "error_message": error,
        "code_length": len(kernel_code),
        "metadata": {},
    }


def _extract_result(stdout: str) -> dict[str, Any] | None:
    """Pull the JSON payload between the sentinel markers from worker stdout.

    Returns None on "no markers found". Raises ValueError on "markers found
    but JSON inside is malformed" so the caller can surface the actual error
    rather than a generic "no result" message.
    """
    begin = stdout.rfind(_RESULT_BEGIN)
    if begin == -1:
        return None
    end = stdout.find(_RESULT_END, begin)
    if end == -1:
        return None
    payload = stdout[begin + len(_RESULT_BEGIN):end].strip()
    try:
        return cast(dict[str, Any], json.loads(payload))
    except json.JSONDecodeError as e:
        raise ValueError(f"Worker emitted malformed JSON: {e}; payload={payload[:200]!r}")


class LocalKernelEvaluator:
    """
    Kernel evaluator that runs eval_kernel_against_ref in an isolated subprocess.

    Mirrors the public surface of `ModalKernelEvaluator` so that calling code
    only sees a generic evaluator interface. Each kernel is dispatched to a
    fresh `python -c` worker so torch.utils.cpp_extension cache leaks die with
    the worker, not with the long-running RL/eval process.
    """

    def __init__(self, config: LocalEvaluatorConfig | None = None):
        self.config = config or LocalEvaluatorConfig()
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent))
        return self._semaphore

    async def _run_subprocess(self, request: dict[str, Any]) -> dict[str, Any]:
        """Spawn one worker subprocess, ship the request via stdin, parse the result.

        Callers (`evaluate_single` / `evaluate_batch`) are responsible for
        injecting `gpu_arch` into the request — there's no fallback here so
        that the source of truth stays in one place.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _WORKER_SCRIPT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return _make_default_result(
                request.get("num_correct_trials", 5),
                request.get("kernel_code", ""),
                f"Failed to spawn worker subprocess: {e}",
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(json.dumps(request).encode("utf-8")),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return _make_default_result(
                request.get("num_correct_trials", 5),
                request.get("kernel_code", ""),
                f"Subprocess timeout after {self.config.timeout}s",
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        try:
            result = _extract_result(stdout)
        except ValueError as e:
            # Markers found but JSON inside is malformed — surface the parser's
            # diagnostic (it already includes a 200-char payload preview) so
            # debugging doesn't require digging through the worker stdout.
            return _make_default_result(
                request.get("num_correct_trials", 5),
                request.get("kernel_code", ""),
                f"Worker emitted malformed result: {e}",
            )
        if result is None:
            tail = (stderr or stdout)[-500:]
            return _make_default_result(
                request.get("num_correct_trials", 5),
                request.get("kernel_code", ""),
                f"Worker did not emit result. tail={tail!r}",
            )
        return result

    async def evaluate_single(
        self,
        ref_code: str,
        kernel_code: str,
        backend: str = "triton",
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        num_perf_trials: int = 100,
        precision: str = "fp32",
        timing_method: str = "cuda_event",
        check_for_excessive_speedup: bool = True,
        excessive_speedup_threshold: float = 10.0,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("Local evaluator is disabled")

        request = {
            "ref_code": ref_code,
            "kernel_code": kernel_code,
            "backend": backend,
            "num_correct_trials": num_correct_trials,
            "measure_performance": measure_performance,
            "num_perf_trials": num_perf_trials,
            "precision": precision,
            "timing_method": timing_method,
            "check_for_excessive_speedup": check_for_excessive_speedup,
            "excessive_speedup_threshold": excessive_speedup_threshold,
            "gpu_arch": self.config.gpu_arch,
        }

        async with self._get_semaphore():
            return await self._run_subprocess(request)

    async def evaluate_batch(
        self,
        evaluations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not evaluations:
            return []
        if not self.config.enabled:
            raise RuntimeError("Local evaluator is disabled")

        sem = self._get_semaphore()

        async def _one(e: dict[str, Any]) -> dict[str, Any]:
            request = {
                "ref_code": e["ref_code"],
                "kernel_code": e["kernel_code"],
                "backend": e.get("backend", "triton"),
                "num_correct_trials": e.get("num_correct_trials", 5),
                "measure_performance": e.get("measure_performance", False),
                "num_perf_trials": e.get("num_perf_trials", 100),
                "precision": e.get("precision", "fp32"),
                "timing_method": e.get("timing_method", "cuda_event"),
                "check_for_excessive_speedup": e.get("check_for_excessive_speedup", True),
                "excessive_speedup_threshold": e.get("excessive_speedup_threshold", 10.0),
                "gpu_arch": self.config.gpu_arch,
            }
            async with sem:
                try:
                    return await self._run_subprocess(request)
                except Exception as exc:
                    if self.config.return_exceptions:
                        return _make_default_result(
                            e.get("num_correct_trials", 5),
                            e["kernel_code"],
                            f"Local evaluation failed: {exc}",
                        )
                    raise

        return await asyncio.gather(*(_one(e) for e in evaluations))

    async def evaluate_single_batched(
        self,
        ref_code: str,
        kernel_code: str,
        backend: str = "triton",
        num_correct_trials: int = 5,
        measure_performance: bool = False,
        num_perf_trials: int = 100,
        precision: str = "fp32",
        timing_method: str = "cuda_event",
        check_for_excessive_speedup: bool = True,
        excessive_speedup_threshold: float = 10.0,
    ) -> dict[str, Any]:
        # No batching benefit on a single local GPU; just delegate.
        return await self.evaluate_single(
            ref_code=ref_code,
            kernel_code=kernel_code,
            backend=backend,
            num_correct_trials=num_correct_trials,
            measure_performance=measure_performance,
            num_perf_trials=num_perf_trials,
            precision=precision,
            timing_method=timing_method,
            check_for_excessive_speedup=check_for_excessive_speedup,
            excessive_speedup_threshold=excessive_speedup_threshold,
        )


# Global evaluator instance (lazy initialized)
_global_local_evaluator: LocalKernelEvaluator | None = None


def get_local_evaluator(
    config: LocalEvaluatorConfig | None = None,
) -> LocalKernelEvaluator:
    """Get or create the global local evaluator instance."""
    global _global_local_evaluator
    if _global_local_evaluator is None:
        _global_local_evaluator = LocalKernelEvaluator(config)
    elif config is not None:
        _global_local_evaluator.config = config
    return _global_local_evaluator


def set_local_evaluator(evaluator: LocalKernelEvaluator) -> None:
    """Set the global local evaluator instance."""
    global _global_local_evaluator
    _global_local_evaluator = evaluator
