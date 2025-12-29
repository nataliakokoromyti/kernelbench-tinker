"""
Evaluation utilities for KernelBench.

This module provides utilities for evaluating trained models on KernelBench,
computing metrics like pass@k, fast_p, and aggregate statistics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProblemResult:
    """Results for a single problem."""
    level: int
    problem_id: int
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_correct(self) -> int:
        return sum(1 for s in self.samples if s.get("correctness", False))

    @property
    def num_compiled(self) -> int:
        return sum(1 for s in self.samples if s.get("compiled", False))

    @property
    def best_speedup(self) -> float | None:
        speedups = [s.get("speedup") for s in self.samples
                   if s.get("correctness") and s.get("speedup") is not None]
        return max(speedups) if speedups else None


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    problems: list[ProblemResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_problem(self, result: ProblemResult) -> None:
        """Add a problem result."""
        self.problems.append(result)

    @property
    def num_problems(self) -> int:
        return len(self.problems)

    def pass_at_k(self, k: int = 1) -> float:
        """
        Compute pass@k metric.

        pass@k is the probability that at least one of k samples is correct.
        Uses the unbiased estimator from the Codex paper.
        """
        if not self.problems:
            return 0.0

        pass_at_k_per_problem = []
        for prob in self.problems:
            n = prob.num_samples
            c = prob.num_correct

            if n < k:
                # Not enough samples
                continue

            if n - c < k:
                # Guaranteed to pass
                pass_at_k_per_problem.append(1.0)
            else:
                # Use unbiased estimator
                pass_at_k = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
                pass_at_k_per_problem.append(pass_at_k)

        return float(np.mean(pass_at_k_per_problem)) if pass_at_k_per_problem else 0.0

    def compile_rate(self) -> float:
        """Compute overall compilation rate."""
        total_samples = sum(p.num_samples for p in self.problems)
        total_compiled = sum(p.num_compiled for p in self.problems)
        return total_compiled / total_samples if total_samples > 0 else 0.0

    def correct_rate(self) -> float:
        """Compute overall correctness rate."""
        total_samples = sum(p.num_samples for p in self.problems)
        total_correct = sum(p.num_correct for p in self.problems)
        return total_correct / total_samples if total_samples > 0 else 0.0

    def fast_p(self, p: float = 1.0) -> float:
        """
        Compute fast_p metric.

        fast_p is the fraction of problems with correct solutions that achieve
        speedup >= p.
        """
        problems_with_speedup = [
            prob for prob in self.problems
            if prob.best_speedup is not None and prob.best_speedup >= p
        ]
        return len(problems_with_speedup) / len(self.problems) if self.problems else 0.0

    def mean_speedup(self) -> float | None:
        """Compute mean speedup across correct solutions."""
        speedups = [prob.best_speedup for prob in self.problems
                   if prob.best_speedup is not None]
        return float(np.mean(speedups)) if speedups else None

    def geometric_mean_speedup(self) -> float | None:
        """Compute geometric mean speedup across correct solutions."""
        speedups = [prob.best_speedup for prob in self.problems
                   if prob.best_speedup is not None and prob.best_speedup > 0]
        if not speedups:
            return None
        return float(np.exp(np.mean(np.log(speedups))))

    def summary(self) -> dict[str, Any]:
        """Get summary metrics."""
        return {
            "num_problems": self.num_problems,
            "pass@1": self.pass_at_k(1),
            "pass@5": self.pass_at_k(5) if all(p.num_samples >= 5 for p in self.problems) else None,
            "compile_rate": self.compile_rate(),
            "correct_rate": self.correct_rate(),
            "fast_1": self.fast_p(1.0),
            "fast_2": self.fast_p(2.0),
            "mean_speedup": self.mean_speedup(),
            "geometric_mean_speedup": self.geometric_mean_speedup(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary(),
            "metadata": self.metadata,
            "problems": [asdict(p) for p in self.problems],
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved evaluation results to {path}")

    @classmethod
    def load(cls, path: str) -> "EvalResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = cls(metadata=data.get("metadata", {}))
        for prob_data in data.get("problems", []):
            results.add_problem(ProblemResult(
                level=prob_data["level"],
                problem_id=prob_data["problem_id"],
                samples=prob_data.get("samples", []),
            ))
        return results


def print_eval_summary(results: EvalResults) -> None:
    """Print a formatted evaluation summary."""
    summary = results.summary()

    print("\n" + "=" * 60)
    print("KernelBench Evaluation Summary")
    print("=" * 60)
    print(f"  Problems evaluated: {summary['num_problems']}")
    print(f"  Compile rate:       {summary['compile_rate']:.1%}")
    print(f"  Correct rate:       {summary['correct_rate']:.1%}")
    print(f"  Pass@1:             {summary['pass@1']:.1%}")
    if summary.get("pass@5") is not None:
        print(f"  Pass@5:             {summary['pass@5']:.1%}")
    print()
    print("  Speed Metrics (correct solutions only):")
    print(f"    fast_1 (speedup >= 1x): {summary['fast_1']:.1%}")
    print(f"    fast_2 (speedup >= 2x): {summary['fast_2']:.1%}")
    if summary.get("mean_speedup"):
        print(f"    Mean speedup:           {summary['mean_speedup']:.2f}x")
    if summary.get("geometric_mean_speedup"):
        print(f"    Geometric mean speedup: {summary['geometric_mean_speedup']:.2f}x")
    print("=" * 60 + "\n")


def compare_results(baseline: EvalResults, improved: EvalResults) -> dict[str, Any]:
    """Compare two evaluation results."""
    base_summary = baseline.summary()
    impr_summary = improved.summary()

    comparison = {}
    for key in base_summary:
        if base_summary[key] is not None and impr_summary.get(key) is not None:
            base_val = base_summary[key]
            impr_val = impr_summary[key]
            if isinstance(base_val, (int, float)) and isinstance(impr_val, (int, float)):
                comparison[key] = {
                    "baseline": base_val,
                    "improved": impr_val,
                    "delta": impr_val - base_val,
                    "relative": (impr_val / base_val - 1) if base_val != 0 else None,
                }

    return comparison
