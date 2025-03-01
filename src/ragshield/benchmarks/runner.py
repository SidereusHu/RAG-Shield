"""Benchmark runner for RAG-Shield.

Provides a unified interface for running all benchmarks
and generating comprehensive reports.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time
import json

from ragshield.benchmarks.datasets import (
    BenchmarkDataset,
    DatasetGenerator,
    create_standard_datasets,
)
from ragshield.benchmarks.metrics import BenchmarkResult, MetricsCalculator
from ragshield.benchmarks.attack_eval import (
    AttackEvaluator,
    AttackConfig,
    evaluate_attacks_on_dataset,
)
from ragshield.benchmarks.defense_eval import (
    DefenseEvaluator,
    DefenseConfig,
    evaluate_defense_on_dataset,
    create_defense_benchmark_suite,
)
from ragshield.benchmarks.performance import (
    PerformanceBenchmark,
    ScalabilityBenchmark,
    BenchmarkConfig,
    run_standard_benchmarks,
    run_scalability_benchmarks,
)
from ragshield.redteam.poisoning import AttackType
from ragshield.defense import DefenseLevel


class BenchmarkType(Enum):
    """Types of benchmarks."""

    ATTACK = "attack"  # Attack effectiveness
    DEFENSE = "defense"  # Defense effectiveness
    DETECTION = "detection"  # Detection accuracy
    PERFORMANCE = "performance"  # Runtime performance
    SCALABILITY = "scalability"  # Scalability analysis
    FULL = "full"  # All benchmarks


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite configuration.

    Attributes:
        name: Suite name
        types: Benchmark types to run
        datasets: Datasets to use
        configs: Additional configurations
    """

    name: str
    types: List[BenchmarkType] = field(default_factory=lambda: [BenchmarkType.FULL])
    datasets: List[str] = field(default_factory=list)
    configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report.

    Attributes:
        suite: Suite that was run
        results: All benchmark results
        summary: Executive summary
        metadata: Report metadata
    """

    suite: BenchmarkSuite
    results: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite": {
                "name": self.suite.name,
                "types": [t.value for t in self.suite.types],
            },
            "results": self.results,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def generate_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"{'='*60}",
            f"BENCHMARK REPORT: {self.suite.name}",
            f"{'='*60}",
            "",
        ]

        # Detection summary
        if "detection" in self.results:
            det = self.results["detection"]
            lines.extend([
                "DETECTION PERFORMANCE",
                "-" * 40,
                f"  Best F1 Score: {self.summary.get('best_detection_f1', 'N/A'):.3f}",
                f"  Best Preset: {self.summary.get('best_detection_preset', 'N/A')}",
                "",
            ])

        # Defense summary
        if "defense" in self.results:
            lines.extend([
                "DEFENSE EFFECTIVENESS",
                "-" * 40,
                f"  Attack Block Rate: {self.summary.get('attack_block_rate', 0):.1%}",
                f"  False Block Rate: {self.summary.get('false_block_rate', 0):.1%}",
                "",
            ])

        # Attack summary
        if "attack" in self.results:
            lines.extend([
                "ATTACK ANALYSIS",
                "-" * 40,
                f"  Most Effective: {self.summary.get('most_effective_attack', 'N/A')}",
                f"  Avg Success Rate: {self.summary.get('avg_attack_success', 0):.1%}",
                "",
            ])

        # Performance summary
        if "performance" in self.results:
            perf = self.results["performance"]
            lines.extend([
                "PERFORMANCE",
                "-" * 40,
                f"  Search Latency: {self.summary.get('search_latency_ms', 0):.2f}ms",
                f"  Detection Overhead: {self.summary.get('detection_overhead', 1):.2f}x",
                "",
            ])

        lines.extend([
            f"{'='*60}",
            f"Generated: {self.metadata.get('timestamp', 'N/A')}",
            f"Duration: {self.metadata.get('duration_seconds', 0):.1f}s",
        ])

        return "\n".join(lines)


class BenchmarkRunner:
    """Unified benchmark runner.

    Orchestrates running different benchmark types and
    generates comprehensive reports.
    """

    def __init__(
        self,
        seed: int = 42,
        verbose: bool = True,
    ):
        """Initialize benchmark runner.

        Args:
            seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.seed = seed
        self.verbose = verbose
        self.datasets: Dict[str, BenchmarkDataset] = {}

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[Benchmark] {message}")

    def load_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkDataset]:
        """Load benchmark datasets.

        Args:
            dataset_names: Optional specific datasets to load

        Returns:
            Dictionary of loaded datasets
        """
        self._log("Loading datasets...")

        # Create standard datasets
        all_datasets = create_standard_datasets(seed=self.seed)

        if dataset_names:
            self.datasets = {
                name: ds for name, ds in all_datasets.items()
                if name in dataset_names
            }
        else:
            self.datasets = all_datasets

        self._log(f"Loaded {len(self.datasets)} datasets")
        return self.datasets

    def run_attack_benchmarks(
        self,
        dataset: Optional[BenchmarkDataset] = None,
    ) -> Dict[str, Any]:
        """Run attack effectiveness benchmarks.

        Args:
            dataset: Optional specific dataset

        Returns:
            Attack benchmark results
        """
        self._log("Running attack benchmarks...")

        evaluator = AttackEvaluator()
        results = {}

        # Evaluate all attack types
        attack_results = evaluator.evaluate_all_attacks(num_poison_docs=5)

        for attack_type, result in attack_results.items():
            results[attack_type.value] = result.to_dict()

        # Comparison
        comparison = evaluator.compare_attacks(attack_results)
        results["comparison"] = comparison

        return results

    def run_defense_benchmarks(
        self,
        dataset: Optional[BenchmarkDataset] = None,
    ) -> Dict[str, Any]:
        """Run defense effectiveness benchmarks.

        Args:
            dataset: Optional specific dataset

        Returns:
            Defense benchmark results
        """
        self._log("Running defense benchmarks...")

        # Use provided dataset or create one
        if dataset is None:
            generator = DatasetGenerator(seed=self.seed)
            dataset = generator.generate_dataset(
                name="defense_bench",
                num_clean=100,
                num_poisoned=50,
            )

        result = evaluate_defense_on_dataset(dataset)
        return result.to_dict()

    def run_detection_benchmarks(
        self,
        datasets: Optional[Dict[str, BenchmarkDataset]] = None,
    ) -> Dict[str, Any]:
        """Run detection accuracy benchmarks.

        Args:
            datasets: Optional datasets to use

        Returns:
            Detection benchmark results
        """
        self._log("Running detection benchmarks...")

        datasets = datasets or self.datasets
        evaluator = DefenseEvaluator()
        results = {}

        for name, dataset in datasets.items():
            self._log(f"  Evaluating on {name}...")

            # Test with different presets
            preset_results = {}
            for preset in ["permissive", "default", "strict"]:
                config = DefenseConfig(detector_preset=preset)
                metrics = evaluator.evaluate_detection(dataset, config)
                preset_results[preset] = metrics.to_dict()

            results[name] = preset_results

        return results

    def run_performance_benchmarks(
        self,
        config: Optional[BenchmarkConfig] = None,
    ) -> Dict[str, Any]:
        """Run performance benchmarks.

        Args:
            config: Optional benchmark configuration

        Returns:
            Performance benchmark results
        """
        self._log("Running performance benchmarks...")

        # Standard benchmarks
        standard = run_standard_benchmarks(config)

        # Scalability benchmarks (smaller sizes for quick run)
        scalability = run_scalability_benchmarks(sizes=[100, 500, 1000])

        return {
            "standard": standard,
            "scalability": scalability,
        }

    def run_suite(
        self,
        suite: BenchmarkSuite,
    ) -> BenchmarkReport:
        """Run a complete benchmark suite.

        Args:
            suite: Benchmark suite configuration

        Returns:
            Complete benchmark report
        """
        start_time = time.time()
        self._log(f"Starting benchmark suite: {suite.name}")

        # Load datasets if needed
        if suite.datasets:
            self.load_datasets(suite.datasets)
        elif not self.datasets:
            self.load_datasets()

        report = BenchmarkReport(
            suite=suite,
            metadata={
                "seed": self.seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        # Determine which benchmarks to run
        run_all = BenchmarkType.FULL in suite.types

        # Run selected benchmarks
        if run_all or BenchmarkType.ATTACK in suite.types:
            report.results["attack"] = self.run_attack_benchmarks()

        if run_all or BenchmarkType.DEFENSE in suite.types:
            # Use first available dataset
            dataset = next(iter(self.datasets.values())) if self.datasets else None
            report.results["defense"] = self.run_defense_benchmarks(dataset)

        if run_all or BenchmarkType.DETECTION in suite.types:
            report.results["detection"] = self.run_detection_benchmarks()

        if run_all or BenchmarkType.PERFORMANCE in suite.types:
            report.results["performance"] = self.run_performance_benchmarks()

        if run_all or BenchmarkType.SCALABILITY in suite.types:
            if "performance" not in report.results:
                report.results["performance"] = self.run_performance_benchmarks()

        # Generate summary
        report.summary = self._generate_summary(report.results)
        report.metadata["duration_seconds"] = time.time() - start_time

        self._log(f"Benchmark suite completed in {report.metadata['duration_seconds']:.1f}s")

        return report

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from results."""
        summary = {}

        # Detection summary
        if "detection" in results:
            best_f1 = 0.0
            best_preset = None
            for dataset_name, presets in results["detection"].items():
                for preset, metrics in presets.items():
                    f1 = metrics.get("confusion_matrix", {}).get("f1_score", 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_preset = preset
            summary["best_detection_f1"] = best_f1
            summary["best_detection_preset"] = best_preset

        # Defense summary
        if "defense" in results:
            defense = results["defense"]
            if "defense" in defense:
                summary["attack_block_rate"] = defense["defense"].get("block_rate", 0)
                summary["false_block_rate"] = defense["defense"].get("false_block_rate", 0)

        # Attack summary
        if "attack" in results and "comparison" in results["attack"]:
            comp = results["attack"]["comparison"]
            summary["most_effective_attack"] = comp.get("most_effective")
            # Calculate average success rate
            attack_comp = comp.get("attack_comparison", {})
            if attack_comp:
                rates = [v.get("success_rate", 0) for v in attack_comp.values()]
                summary["avg_attack_success"] = sum(rates) / len(rates)

        # Performance summary
        if "performance" in results and "standard" in results["performance"]:
            std = results["performance"]["standard"].get("results", {})
            if "knowledge_base_search" in std:
                summary["search_latency_ms"] = std["knowledge_base_search"].get("latency_ms", 0)

        return summary

    def run_quick_benchmark(self) -> BenchmarkReport:
        """Run a quick benchmark for testing.

        Returns:
            Benchmark report
        """
        suite = BenchmarkSuite(
            name="quick_test",
            types=[BenchmarkType.DETECTION, BenchmarkType.PERFORMANCE],
            datasets=["small_balanced"],
        )
        return self.run_suite(suite)

    def run_full_benchmark(self) -> BenchmarkReport:
        """Run full benchmark suite.

        Returns:
            Complete benchmark report
        """
        suite = BenchmarkSuite(
            name="full_evaluation",
            types=[BenchmarkType.FULL],
        )
        return self.run_suite(suite)


def create_standard_suites() -> Dict[str, BenchmarkSuite]:
    """Create standard benchmark suites.

    Returns:
        Dictionary of suite name to suite
    """
    return {
        "quick": BenchmarkSuite(
            name="quick",
            types=[BenchmarkType.DETECTION],
            datasets=["small_balanced"],
        ),
        "detection": BenchmarkSuite(
            name="detection_focused",
            types=[BenchmarkType.DETECTION],
        ),
        "defense": BenchmarkSuite(
            name="defense_focused",
            types=[BenchmarkType.DEFENSE, BenchmarkType.ATTACK],
        ),
        "performance": BenchmarkSuite(
            name="performance_focused",
            types=[BenchmarkType.PERFORMANCE, BenchmarkType.SCALABILITY],
        ),
        "full": BenchmarkSuite(
            name="full_evaluation",
            types=[BenchmarkType.FULL],
        ),
    }


def run_benchmark(
    suite_name: str = "quick",
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkReport:
    """Convenience function to run a benchmark.

    Args:
        suite_name: Name of suite to run
        seed: Random seed
        verbose: Print progress

    Returns:
        Benchmark report
    """
    suites = create_standard_suites()
    suite = suites.get(suite_name)

    if suite is None:
        raise ValueError(f"Unknown suite: {suite_name}. Available: {list(suites.keys())}")

    runner = BenchmarkRunner(seed=seed, verbose=verbose)
    return runner.run_suite(suite)
