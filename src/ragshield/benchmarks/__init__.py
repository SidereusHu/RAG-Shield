"""RAG-Shield Benchmarks Module.

Comprehensive evaluation framework for RAG security components.

Components:
- datasets: Benchmark datasets for evaluation
- metrics: Evaluation metrics (precision, recall, F1, AUC, etc.)
- attack_eval: Attack effectiveness evaluation
- defense_eval: Defense effectiveness evaluation
- performance: Performance and scalability benchmarks
- runner: Unified benchmark runner
"""

from ragshield.benchmarks.datasets import (
    DatasetType,
    BenchmarkSample,
    BenchmarkDataset,
    DatasetGenerator,
    create_standard_datasets,
)

from ragshield.benchmarks.metrics import (
    MetricType,
    ConfusionMatrix,
    DetectionMetrics,
    DefenseMetrics,
    PrivacyMetrics,
    PerformanceMetrics,
    MetricsCalculator,
    BenchmarkResult,
)

from ragshield.benchmarks.attack_eval import (
    AttackConfig,
    AttackResult,
    AttackEvaluationResult,
    AttackEvaluator,
    evaluate_attacks_on_dataset,
)

from ragshield.benchmarks.defense_eval import (
    DefenseConfig,
    DefenseEvaluationResult,
    DefenseEvaluator,
    evaluate_defense_on_dataset,
    create_defense_benchmark_suite,
)

from ragshield.benchmarks.performance import (
    BenchmarkConfig,
    BenchmarkRun,
    PerformanceBenchmark,
    ScalabilityBenchmark,
    run_standard_benchmarks,
    run_scalability_benchmarks,
)

from ragshield.benchmarks.runner import (
    BenchmarkType,
    BenchmarkSuite,
    BenchmarkReport,
    BenchmarkRunner,
    create_standard_suites,
    run_benchmark,
)


__all__ = [
    # Datasets
    "DatasetType",
    "BenchmarkSample",
    "BenchmarkDataset",
    "DatasetGenerator",
    "create_standard_datasets",
    # Metrics
    "MetricType",
    "ConfusionMatrix",
    "DetectionMetrics",
    "DefenseMetrics",
    "PrivacyMetrics",
    "PerformanceMetrics",
    "MetricsCalculator",
    "BenchmarkResult",
    # Attack evaluation
    "AttackConfig",
    "AttackResult",
    "AttackEvaluationResult",
    "AttackEvaluator",
    "evaluate_attacks_on_dataset",
    # Defense evaluation
    "DefenseConfig",
    "DefenseEvaluationResult",
    "DefenseEvaluator",
    "evaluate_defense_on_dataset",
    "create_defense_benchmark_suite",
    # Performance
    "BenchmarkConfig",
    "BenchmarkRun",
    "PerformanceBenchmark",
    "ScalabilityBenchmark",
    "run_standard_benchmarks",
    "run_scalability_benchmarks",
    # Runner
    "BenchmarkType",
    "BenchmarkSuite",
    "BenchmarkReport",
    "BenchmarkRunner",
    "create_standard_suites",
    "run_benchmark",
]
