#!/usr/bin/env python3
"""RAG-Shield Benchmark Demo.

Demonstrates the benchmark and evaluation framework for measuring
detection accuracy, defense effectiveness, and performance.
"""

from ragshield.benchmarks import (
    # Datasets
    DatasetGenerator,
    create_standard_datasets,
    # Metrics
    MetricsCalculator,
    BenchmarkResult,
    # Evaluators
    AttackEvaluator,
    DefenseEvaluator,
    DefenseConfig,
    # Performance
    PerformanceBenchmark,
    BenchmarkConfig,
    # Runner
    BenchmarkRunner,
    BenchmarkType,
    BenchmarkSuite,
    run_benchmark,
)
from ragshield.redteam.poisoning import AttackType


def demo_dataset_generation():
    """Demo: Generate benchmark datasets."""
    print("\n" + "=" * 60)
    print("DEMO: Benchmark Dataset Generation")
    print("=" * 60)

    # Create generator
    generator = DatasetGenerator(seed=42)

    # Generate custom dataset
    dataset = generator.generate_dataset(
        name="demo_dataset",
        num_clean=50,
        num_poisoned=30,
        attack_distribution={
            AttackType.DIRECT: 0.4,
            AttackType.ADVERSARIAL: 0.3,
            AttackType.STEALTH: 0.2,
            AttackType.CHAIN: 0.1,
        },
    )

    print(f"\nGenerated dataset: {dataset.name}")
    stats = dataset.get_statistics()
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Clean: {stats['clean_count']}")
    print(f"  Poisoned: {stats['poisoned_count']}")
    print(f"  Poison ratio: {stats['poison_ratio']:.1%}")
    print(f"  Attack breakdown: {stats['attack_breakdown']}")

    # Split dataset
    train, test = dataset.split(train_ratio=0.7)
    print(f"\n  Train set: {len(train)} samples")
    print(f"  Test set: {len(test)} samples")


def demo_metrics():
    """Demo: Evaluation metrics."""
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Metrics")
    print("=" * 60)

    # Simulate predictions
    y_true = [True, True, True, True, False, False, False, False, True, False]
    y_pred = [True, True, False, True, False, False, True, False, True, False]
    y_scores = [0.9, 0.85, 0.4, 0.7, 0.2, 0.15, 0.6, 0.1, 0.8, 0.3]

    # Calculate confusion matrix
    cm = MetricsCalculator.confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(f"  True Positives: {cm.true_positives}")
    print(f"  False Positives: {cm.false_positives}")
    print(f"  True Negatives: {cm.true_negatives}")
    print(f"  False Negatives: {cm.false_negatives}")

    print("\nMetrics:")
    print(f"  Accuracy: {cm.accuracy:.3f}")
    print(f"  Precision: {cm.precision:.3f}")
    print(f"  Recall: {cm.recall:.3f}")
    print(f"  F1 Score: {cm.f1_score:.3f}")
    print(f"  Specificity: {cm.specificity:.3f}")

    # Calculate ROC AUC
    roc_auc = MetricsCalculator.roc_auc(y_true, y_scores)
    pr_auc = MetricsCalculator.pr_auc(y_true, y_scores)
    print(f"\n  ROC AUC: {roc_auc:.3f}")
    print(f"  PR AUC: {pr_auc:.3f}")


def demo_attack_evaluation():
    """Demo: Attack effectiveness evaluation."""
    print("\n" + "=" * 60)
    print("DEMO: Attack Evaluation")
    print("=" * 60)

    evaluator = AttackEvaluator()

    # Evaluate all attack types
    print("\nEvaluating attack effectiveness...")
    results = evaluator.evaluate_all_attacks(num_poison_docs=3)

    print("\nAttack Success Rates:")
    for attack_type, result in results.items():
        print(f"  {attack_type.value}: {result.success_rate:.1%}")

    # Compare attacks
    comparison = evaluator.compare_attacks(results)
    print(f"\nMost effective attack: {comparison['most_effective']}")
    print(f"Least effective attack: {comparison['least_effective']}")


def demo_defense_evaluation():
    """Demo: Defense effectiveness evaluation."""
    print("\n" + "=" * 60)
    print("DEMO: Defense Evaluation")
    print("=" * 60)

    # Generate dataset
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate_dataset(
        name="defense_test",
        num_clean=40,
        num_poisoned=20,
    )

    evaluator = DefenseEvaluator()

    # Test different configurations
    print("\nTesting different detector presets...")

    for preset in ["permissive", "default", "strict"]:
        config = DefenseConfig(detector_preset=preset)
        metrics = evaluator.evaluate_detection(dataset, config)
        cm = metrics.confusion_matrix

        print(f"\n  Preset: {preset}")
        print(f"    Accuracy: {cm.accuracy:.3f}")
        print(f"    Precision: {cm.precision:.3f}")
        print(f"    Recall: {cm.recall:.3f}")
        print(f"    F1 Score: {cm.f1_score:.3f}")


def demo_performance_benchmark():
    """Demo: Performance benchmarking."""
    print("\n" + "=" * 60)
    print("DEMO: Performance Benchmarks")
    print("=" * 60)

    config = BenchmarkConfig(
        num_iterations=50,
        warmup_iterations=5,
        num_documents=200,
        embedding_dim=384,
    )

    benchmark = PerformanceBenchmark(config=config)

    # Benchmark knowledge base operations
    print("\nBenchmarking knowledge base operations...")

    add_result = benchmark.benchmark_knowledge_base_add()
    print(f"\n  Document Add:")
    print(f"    Mean latency: {add_result.mean_latency:.3f}ms")

    search_result = benchmark.benchmark_knowledge_base_search()
    print(f"\n  Document Search:")
    print(f"    Mean latency: {search_result.mean_latency:.3f}ms")

    # Generate report
    report = benchmark.generate_report()
    print(f"\n  Throughput: {report['results']['knowledge_base_search']['throughput']:.1f} ops/sec")


def demo_full_benchmark():
    """Demo: Full benchmark suite."""
    print("\n" + "=" * 60)
    print("DEMO: Full Benchmark Suite")
    print("=" * 60)

    # Create custom suite
    suite = BenchmarkSuite(
        name="demo_suite",
        types=[BenchmarkType.DETECTION, BenchmarkType.ATTACK],
        datasets=["small_balanced"],
    )

    # Run benchmark
    print("\nRunning benchmark suite...")
    runner = BenchmarkRunner(seed=42, verbose=True)
    report = runner.run_suite(suite)

    # Print summary
    print("\n" + report.generate_summary())


def demo_quick_run():
    """Demo: Quick benchmark run."""
    print("\n" + "=" * 60)
    print("DEMO: Quick Benchmark")
    print("=" * 60)

    print("\nRunning quick benchmark...")
    report = run_benchmark(suite_name="quick", seed=42, verbose=False)

    print(f"\nBenchmark completed in {report.metadata['duration_seconds']:.2f}s")
    print(f"Best detection F1: {report.summary.get('best_detection_f1', 'N/A')}")


def main():
    """Run all benchmark demos."""
    print("\n" + "#" * 60)
    print("#  RAG-Shield Benchmark Framework Demo")
    print("#" * 60)

    demo_dataset_generation()
    demo_metrics()
    demo_attack_evaluation()
    demo_defense_evaluation()
    demo_performance_benchmark()
    demo_quick_run()
    # demo_full_benchmark()  # Uncomment for full suite

    print("\n" + "=" * 60)
    print("All benchmark demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
