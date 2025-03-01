"""Tests for RAG-Shield benchmarks module."""

import pytest
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.redteam.poisoning import AttackType
from ragshield.defense import DefenseLevel

from ragshield.benchmarks import (
    # Datasets
    DatasetType,
    BenchmarkSample,
    BenchmarkDataset,
    DatasetGenerator,
    create_standard_datasets,
    # Metrics
    ConfusionMatrix,
    DetectionMetrics,
    DefenseMetrics,
    PerformanceMetrics,
    MetricsCalculator,
    BenchmarkResult,
    # Attack eval
    AttackConfig,
    AttackEvaluator,
    # Defense eval
    DefenseConfig,
    DefenseEvaluator,
    # Performance
    BenchmarkConfig,
    BenchmarkRun,
    PerformanceBenchmark,
    ScalabilityBenchmark,
    # Runner
    BenchmarkType,
    BenchmarkSuite,
    BenchmarkReport,
    BenchmarkRunner,
    create_standard_suites,
)


# ============================================================================
# Dataset Tests
# ============================================================================

class TestDatasetGenerator:
    """Tests for DatasetGenerator."""

    def test_generate_clean_document(self):
        """Test clean document generation."""
        generator = DatasetGenerator(seed=42)
        doc = generator.generate_clean_document()

        assert doc is not None
        assert doc.doc_id.startswith("clean_")
        assert len(doc.content) > 0
        assert doc.embedding is not None

    def test_generate_poisoned_document(self):
        """Test poisoned document generation."""
        generator = DatasetGenerator(seed=42)
        doc, query, answer = generator.generate_poisoned_document()

        assert doc is not None
        assert doc.doc_id.startswith("poison_")
        assert query is not None
        assert answer is not None

    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate_dataset(
            name="test",
            num_clean=20,
            num_poisoned=10,
        )

        assert len(dataset) == 30
        assert len(dataset.get_clean()) == 20
        assert len(dataset.get_poisoned()) == 10

    def test_dataset_statistics(self):
        """Test dataset statistics."""
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate_dataset(
            name="test",
            num_clean=50,
            num_poisoned=50,
        )

        stats = dataset.get_statistics()

        assert stats["total_samples"] == 100
        assert stats["poisoned_count"] == 50
        assert stats["clean_count"] == 50
        assert stats["poison_ratio"] == 0.5

    def test_dataset_split(self):
        """Test dataset splitting."""
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate_dataset(
            name="test",
            num_clean=80,
            num_poisoned=20,
        )

        train, test = dataset.split(train_ratio=0.8)

        assert len(train) == 80
        assert len(test) == 20

    def test_create_standard_datasets(self):
        """Test standard dataset creation."""
        datasets = create_standard_datasets(seed=42)

        assert "small_balanced" in datasets
        assert "large_imbalanced" in datasets
        assert "easy" in datasets
        assert "hard" in datasets


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        generator = DatasetGenerator(seed=42)
        return generator.generate_dataset(
            name="fixture",
            num_clean=30,
            num_poisoned=20,
        )

    def test_get_by_attack_type(self, sample_dataset):
        """Test filtering by attack type."""
        direct = sample_dataset.get_by_attack_type(AttackType.DIRECT)
        # Should have some direct attacks
        assert isinstance(direct, list)

    def test_dataset_type(self, sample_dataset):
        """Test dataset type."""
        assert sample_dataset.dataset_type == DatasetType.MIXED


# ============================================================================
# Metrics Tests
# ============================================================================

class TestConfusionMatrix:
    """Tests for ConfusionMatrix."""

    def test_accuracy(self):
        """Test accuracy calculation."""
        cm = ConfusionMatrix(
            true_positives=80,
            false_positives=10,
            true_negatives=85,
            false_negatives=25,
        )
        # Accuracy = (80 + 85) / 200 = 0.825
        assert abs(cm.accuracy - 0.825) < 0.001

    def test_precision(self):
        """Test precision calculation."""
        cm = ConfusionMatrix(
            true_positives=80,
            false_positives=20,
            true_negatives=0,
            false_negatives=0,
        )
        # Precision = 80 / 100 = 0.8
        assert abs(cm.precision - 0.8) < 0.001

    def test_recall(self):
        """Test recall calculation."""
        cm = ConfusionMatrix(
            true_positives=80,
            false_positives=0,
            true_negatives=0,
            false_negatives=20,
        )
        # Recall = 80 / 100 = 0.8
        assert abs(cm.recall - 0.8) < 0.001

    def test_f1_score(self):
        """Test F1 score calculation."""
        cm = ConfusionMatrix(
            true_positives=80,
            false_positives=20,
            true_negatives=80,
            false_negatives=20,
        )
        # F1 = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8
        assert abs(cm.f1_score - 0.8) < 0.001

    def test_empty_matrix(self):
        """Test with empty matrix."""
        cm = ConfusionMatrix()
        assert cm.accuracy == 0.0
        assert cm.precision == 0.0
        assert cm.recall == 0.0


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        y_true = [True, True, False, False, True]
        y_pred = [True, False, False, True, True]

        cm = MetricsCalculator.confusion_matrix(y_true, y_pred)

        assert cm.true_positives == 2
        assert cm.false_positives == 1
        assert cm.true_negatives == 1
        assert cm.false_negatives == 1

    def test_roc_curve(self):
        """Test ROC curve calculation."""
        y_true = [True, True, False, False]
        y_scores = [0.9, 0.7, 0.3, 0.2]

        fpr, tpr, thresholds = MetricsCalculator.roc_curve(y_true, y_scores)

        assert len(fpr) == len(tpr) == len(thresholds)

    def test_auc(self):
        """Test AUC calculation."""
        # Perfect classifier
        x = [0, 0, 1]
        y = [0, 1, 1]
        auc = MetricsCalculator.auc(x, y)
        assert auc == 1.0

    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        latencies = list(range(1, 101))  # 1 to 100

        mean, p50, p95, p99 = MetricsCalculator.latency_percentiles(latencies)

        assert mean == 50.5
        assert p50 == 51  # Index 50 in sorted list [1..100]
        assert p95 == 96  # Index 95 in sorted list
        assert p99 == 100  # Index 99 in sorted list

    def test_detection_metrics(self):
        """Test detection metrics calculation."""
        y_true = [True, True, False, False, True, False]
        y_pred = [True, True, False, True, False, False]
        y_scores = [0.9, 0.8, 0.2, 0.6, 0.4, 0.1]

        metrics = MetricsCalculator.detection_metrics(
            y_true, y_pred, y_scores
        )

        assert metrics.confusion_matrix.total == 6
        assert metrics.roc_auc >= 0


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = BenchmarkResult(
            name="test",
            detection=DetectionMetrics(),
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["name"] == "test"
        assert "detection" in d
        assert d["metadata"]["key"] == "value"

    def test_summary(self):
        """Test summary generation."""
        cm = ConfusionMatrix(
            true_positives=80,
            false_positives=10,
            true_negatives=85,
            false_negatives=25,
        )
        result = BenchmarkResult(
            name="test",
            detection=DetectionMetrics(confusion_matrix=cm),
        )

        summary = result.summary()

        assert "test" in summary
        assert "Detection" in summary


# ============================================================================
# Attack Evaluation Tests
# ============================================================================

class TestAttackEvaluator:
    """Tests for AttackEvaluator."""

    def test_create_attack(self):
        """Test attack creation."""
        evaluator = AttackEvaluator()

        for attack_type in AttackType:
            attack = evaluator.create_attack(attack_type)
            assert attack is not None

    def test_evaluate_attack(self):
        """Test attack evaluation."""
        evaluator = AttackEvaluator()
        config = AttackConfig(
            attack_type=AttackType.DIRECT,
            num_poison_docs=3,
        )

        result = evaluator.evaluate_attack(config)

        assert result.attack_type == AttackType.DIRECT
        assert len(result.attack_results) == 3
        assert 0 <= result.success_rate <= 1

    def test_evaluate_all_attacks(self):
        """Test evaluation of all attack types."""
        evaluator = AttackEvaluator()

        results = evaluator.evaluate_all_attacks(num_poison_docs=2)

        assert len(results) == len(AttackType)
        for attack_type in AttackType:
            assert attack_type in results

    def test_compare_attacks(self):
        """Test attack comparison."""
        evaluator = AttackEvaluator()
        results = evaluator.evaluate_all_attacks(num_poison_docs=2)

        comparison = evaluator.compare_attacks(results)

        assert "attack_comparison" in comparison
        assert "most_effective" in comparison


# ============================================================================
# Defense Evaluation Tests
# ============================================================================

class TestDefenseEvaluator:
    """Tests for DefenseEvaluator."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        generator = DatasetGenerator(seed=42)
        return generator.generate_dataset(
            name="defense_test",
            num_clean=20,
            num_poisoned=10,
        )

    def test_evaluate_detection(self, sample_dataset):
        """Test detection evaluation."""
        evaluator = DefenseEvaluator()
        config = DefenseConfig(detector_preset="default")

        metrics = evaluator.evaluate_detection(sample_dataset, config)

        assert metrics.confusion_matrix.total == 30
        assert 0 <= metrics.confusion_matrix.accuracy <= 1

    def test_evaluate_full(self, sample_dataset):
        """Test full evaluation."""
        evaluator = DefenseEvaluator()
        config = DefenseConfig(
            detector_preset="default",
            defense_level=DefenseLevel.MINIMAL,
        )

        result = evaluator.evaluate_full(sample_dataset, config)

        assert result.config == config
        assert result.detection_metrics is not None

    def test_compare_configurations(self, sample_dataset):
        """Test configuration comparison."""
        evaluator = DefenseEvaluator()
        configs = [
            DefenseConfig(detector_preset="permissive"),
            DefenseConfig(detector_preset="strict"),
        ]

        results = evaluator.compare_configurations(sample_dataset, configs)

        assert len(results) == 2


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark."""

    def test_benchmark_function(self):
        """Test function benchmarking."""
        config = BenchmarkConfig(
            num_iterations=10,
            warmup_iterations=2,
        )
        benchmark = PerformanceBenchmark(config=config)

        def dummy_func():
            return sum(range(1000))

        run = benchmark.benchmark_function("dummy", dummy_func)

        assert run.name == "dummy"
        assert len(run.latencies) == 10

    def test_benchmark_knowledge_base_add(self):
        """Test KB add benchmarking."""
        config = BenchmarkConfig(num_iterations=10)
        benchmark = PerformanceBenchmark(config=config)

        run = benchmark.benchmark_knowledge_base_add()

        assert run.name == "knowledge_base_add"
        assert len(run.latencies) == 10

    def test_benchmark_knowledge_base_search(self):
        """Test KB search benchmarking."""
        config = BenchmarkConfig(
            num_iterations=10,
            num_documents=50,
        )
        benchmark = PerformanceBenchmark(config=config)

        run = benchmark.benchmark_knowledge_base_search()

        assert run.name == "knowledge_base_search"

    def test_compare_with_baseline(self):
        """Test baseline comparison."""
        benchmark = PerformanceBenchmark()

        baseline = BenchmarkRun(name="baseline", latencies=[10, 10, 10])
        comparison = BenchmarkRun(name="comparison", latencies=[15, 15, 15])

        result = benchmark.compare_with_baseline(baseline, comparison)

        assert result["overhead_ratio"] == 1.5
        assert result["overhead_percent"] == 50.0

    def test_generate_report(self):
        """Test report generation."""
        config = BenchmarkConfig(num_iterations=5)
        benchmark = PerformanceBenchmark(config=config)

        benchmark.benchmark_knowledge_base_add()

        report = benchmark.generate_report()

        assert "config" in report
        assert "results" in report
        assert "knowledge_base_add" in report["results"]


class TestScalabilityBenchmark:
    """Tests for ScalabilityBenchmark."""

    def test_benchmark_kb_scaling(self):
        """Test KB scaling benchmark."""
        benchmark = ScalabilityBenchmark(sizes=[50, 100])

        results = benchmark.benchmark_kb_scaling()

        assert 50 in results
        assert 100 in results
        assert "add_total_ms" in results[50]
        assert "search_avg_ms" in results[50]

    def test_estimate_complexity(self):
        """Test complexity estimation."""
        benchmark = ScalabilityBenchmark(sizes=[100, 500])
        benchmark.benchmark_kb_scaling()

        complexity = benchmark.estimate_complexity()

        assert "add_complexity" in complexity
        assert "search_complexity" in complexity


# ============================================================================
# Runner Tests
# ============================================================================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_load_datasets(self):
        """Test dataset loading."""
        runner = BenchmarkRunner(verbose=False)

        datasets = runner.load_datasets(["small_balanced"])

        assert "small_balanced" in datasets

    def test_run_attack_benchmarks(self):
        """Test attack benchmarks."""
        runner = BenchmarkRunner(verbose=False)

        results = runner.run_attack_benchmarks()

        assert "direct" in results
        assert "comparison" in results

    def test_run_quick_benchmark(self):
        """Test quick benchmark."""
        runner = BenchmarkRunner(verbose=False)

        report = runner.run_quick_benchmark()

        assert report.suite.name == "quick_test"
        assert "detection" in report.results

    def test_create_standard_suites(self):
        """Test standard suite creation."""
        suites = create_standard_suites()

        assert "quick" in suites
        assert "full" in suites
        assert "detection" in suites

    def test_benchmark_suite(self):
        """Test running a suite."""
        runner = BenchmarkRunner(verbose=False)
        suite = BenchmarkSuite(
            name="test_suite",
            types=[BenchmarkType.DETECTION],
            datasets=["small_balanced"],
        )

        report = runner.run_suite(suite)

        assert report.suite.name == "test_suite"
        assert "detection" in report.results

    def test_report_summary(self):
        """Test report summary generation."""
        runner = BenchmarkRunner(verbose=False)
        report = runner.run_quick_benchmark()

        summary = report.generate_summary()

        assert "BENCHMARK REPORT" in summary
        assert "quick_test" in summary


class TestBenchmarkReport:
    """Tests for BenchmarkReport."""

    def test_to_json(self):
        """Test JSON conversion."""
        suite = BenchmarkSuite(name="test")
        report = BenchmarkReport(
            suite=suite,
            results={"test": {"value": 1}},
        )

        json_str = report.to_json()

        assert "test" in json_str
        assert "value" in json_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""

    def test_end_to_end_evaluation(self):
        """Test complete evaluation flow."""
        # Generate dataset
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate_dataset(
            name="integration_test",
            num_clean=30,
            num_poisoned=20,
        )

        # Evaluate detection
        evaluator = DefenseEvaluator()
        detection_metrics = evaluator.evaluate_detection(
            dataset,
            DefenseConfig(detector_preset="default"),
        )

        # Run attack evaluation
        attack_evaluator = AttackEvaluator()
        attack_results = attack_evaluator.evaluate_all_attacks(num_poison_docs=2)

        # Create result
        result = BenchmarkResult(
            name="integration_test",
            detection=detection_metrics,
            metadata={
                "attack_success_rates": {
                    k.value: v.success_rate for k, v in attack_results.items()
                }
            },
        )

        # Verify
        assert result.detection is not None
        assert "attack_success_rates" in result.metadata

    def test_full_benchmark_workflow(self):
        """Test full benchmark workflow."""
        runner = BenchmarkRunner(seed=42, verbose=False)

        # Load datasets
        datasets = runner.load_datasets(["small_balanced"])
        assert len(datasets) > 0

        # Run detection benchmarks
        detection_results = runner.run_detection_benchmarks()
        assert "small_balanced" in detection_results

        # Run performance benchmarks
        perf_results = runner.run_performance_benchmarks(
            BenchmarkConfig(num_iterations=5, num_documents=50)
        )
        assert "standard" in perf_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
