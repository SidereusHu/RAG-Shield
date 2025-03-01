"""Defense evaluation for RAG-Shield benchmarks.

Evaluates the effectiveness of detection and defense mechanisms
against various poisoning attacks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection import PoisonDetector, create_poison_detector
from ragshield.defense import RAGShield, DefenseLevel
from ragshield.defense.shield import DefenseConfig as ShieldConfig
from ragshield.redteam.poisoning import AttackType
from ragshield.benchmarks.datasets import BenchmarkDataset, BenchmarkSample
from ragshield.benchmarks.metrics import (
    ConfusionMatrix,
    DetectionMetrics,
    DefenseMetrics,
    PerformanceMetrics,
    MetricsCalculator,
    BenchmarkResult,
)


@dataclass
class DefenseConfig:
    """Configuration for defense evaluation.

    Attributes:
        detector_preset: Detector preset (permissive/default/strict)
        defense_level: Defense level for RAGShield
        enable_quarantine: Enable quarantine system
        enable_monitoring: Enable real-time monitoring
    """

    detector_preset: str = "default"
    defense_level: DefenseLevel = DefenseLevel.STANDARD
    enable_quarantine: bool = True
    enable_monitoring: bool = True
    custom_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class DefenseEvaluationResult:
    """Result of defense evaluation.

    Attributes:
        config: Defense configuration used
        detection_metrics: Detection performance metrics
        defense_metrics: Defense effectiveness metrics
        performance_metrics: Runtime performance metrics
        per_attack_results: Results broken down by attack type
    """

    config: DefenseConfig
    detection_metrics: DetectionMetrics = field(default_factory=DetectionMetrics)
    defense_metrics: DefenseMetrics = field(default_factory=DefenseMetrics)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    per_attack_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": {
                "detector_preset": self.config.detector_preset,
                "defense_level": self.config.defense_level.value,
                "enable_quarantine": self.config.enable_quarantine,
                "enable_monitoring": self.config.enable_monitoring,
            },
            "detection": self.detection_metrics.to_dict(),
            "defense": self.defense_metrics.to_dict(),
            "performance": self.performance_metrics.to_dict(),
            "per_attack_results": self.per_attack_results,
        }


class DefenseEvaluator:
    """Evaluates defense effectiveness against attacks.

    Tests detection accuracy and defense mechanisms across
    different attack types and configurations.
    """

    def __init__(
        self,
        detector: Optional[PoisonDetector] = None,
        shield: Optional[RAGShield] = None,
    ):
        """Initialize defense evaluator.

        Args:
            detector: Optional poison detector
            shield: Optional RAGShield instance
        """
        self.detector = detector
        self.shield = shield

    def evaluate_detection(
        self,
        dataset: BenchmarkDataset,
        config: DefenseConfig,
    ) -> DetectionMetrics:
        """Evaluate detection performance on a dataset.

        Args:
            dataset: Benchmark dataset with labeled samples
            config: Defense configuration

        Returns:
            Detection metrics
        """
        # Create detector if not provided
        detector = self.detector or create_poison_detector(preset=config.detector_preset)

        y_true = []
        y_pred = []
        y_scores = []
        attack_types = []
        latencies = []

        for sample in dataset.samples:
            y_true.append(sample.is_poisoned)
            attack_types.append(
                sample.attack_type.value if sample.attack_type else None
            )

            start_time = time.time()

            # Run detection
            try:
                result = detector.detect(sample.document)
                y_pred.append(result.is_poisoned)
                y_scores.append(result.confidence)
            except Exception:
                # Detection failed - assume clean
                y_pred.append(False)
                y_scores.append(0.0)

            latencies.append((time.time() - start_time) * 1000)

        # Calculate metrics
        metrics = MetricsCalculator.detection_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            attack_types=attack_types,
        )

        return metrics

    def evaluate_defense(
        self,
        dataset: BenchmarkDataset,
        config: DefenseConfig,
    ) -> DefenseMetrics:
        """Evaluate defense effectiveness.

        Args:
            dataset: Benchmark dataset
            config: Defense configuration

        Returns:
            Defense metrics
        """
        metrics = DefenseMetrics()

        # Create knowledge base
        kb = KnowledgeBase()

        # Create shield if not provided
        shield = self.shield
        if shield is None:
            shield_config = ShieldConfig(level=config.defense_level)
            shield = RAGShield(
                knowledge_base=kb,
                config=shield_config,
            )

        blocked_attacks = 0
        successful_attacks = 0
        false_blocks = 0
        total_attacks = 0
        total_legitimate = 0
        response_times = []

        for sample in dataset.samples:
            start_time = time.time()

            try:
                # Try to ingest document through shield
                result = shield.ingest(sample.document)
                ingested = result.accepted
            except Exception:
                ingested = False

            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)

            if sample.is_poisoned:
                total_attacks += 1
                if not ingested:
                    blocked_attacks += 1
                else:
                    successful_attacks += 1
            else:
                total_legitimate += 1
                if not ingested:
                    false_blocks += 1

        metrics.blocked_attacks = blocked_attacks
        metrics.total_attacks = total_attacks
        metrics.false_blocks = false_blocks
        metrics.total_legitimate = total_legitimate
        metrics.attack_success_rate = (
            successful_attacks / total_attacks if total_attacks > 0 else 0.0
        )
        metrics.response_time_ms = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )

        # Quarantine rate
        if hasattr(shield, 'quarantine') and shield.quarantine:
            quarantined = len(shield.quarantine.get_all_pending())
            metrics.quarantine_rate = quarantined / len(dataset) if len(dataset) > 0 else 0.0

        return metrics

    def evaluate_full(
        self,
        dataset: BenchmarkDataset,
        config: Optional[DefenseConfig] = None,
    ) -> DefenseEvaluationResult:
        """Run full defense evaluation.

        Args:
            dataset: Benchmark dataset
            config: Optional defense configuration

        Returns:
            Complete evaluation result
        """
        if config is None:
            config = DefenseConfig()

        result = DefenseEvaluationResult(config=config)

        # Detection evaluation
        result.detection_metrics = self.evaluate_detection(dataset, config)

        # Defense evaluation
        result.defense_metrics = self.evaluate_defense(dataset, config)

        # Per-attack breakdown
        for attack_type in AttackType:
            attack_samples = dataset.get_by_attack_type(attack_type)
            if not attack_samples:
                continue

            # Create mini-dataset for this attack type
            mini_dataset = BenchmarkDataset(
                name=f"{dataset.name}_{attack_type.value}",
                description=f"Subset of {attack_type.value} attacks",
                samples=attack_samples,
                dataset_type=dataset.dataset_type,
            )

            # Evaluate detection on this subset
            detection = self.evaluate_detection(mini_dataset, config)

            result.per_attack_results[attack_type.value] = {
                "num_samples": len(attack_samples),
                "precision": detection.confusion_matrix.precision,
                "recall": detection.confusion_matrix.recall,
                "f1_score": detection.confusion_matrix.f1_score,
            }

        return result

    def compare_configurations(
        self,
        dataset: BenchmarkDataset,
        configs: List[DefenseConfig],
    ) -> Dict[str, DefenseEvaluationResult]:
        """Compare different defense configurations.

        Args:
            dataset: Benchmark dataset
            configs: List of configurations to compare

        Returns:
            Dictionary of config name to evaluation result
        """
        results = {}

        for i, config in enumerate(configs):
            config_name = f"{config.detector_preset}_{config.defense_level.value}"
            results[config_name] = self.evaluate_full(dataset, config)

        return results

    def find_optimal_threshold(
        self,
        dataset: BenchmarkDataset,
        metric: str = "f1_score",
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """Find optimal detection threshold.

        Args:
            dataset: Benchmark dataset
            metric: Metric to optimize (f1_score, precision, recall)
            thresholds: Thresholds to try

        Returns:
            Tuple of (best_threshold, best_score)
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        detector = self.detector or create_poison_detector(preset="default")

        # Get scores for all samples
        y_true = []
        y_scores = []

        for sample in dataset.samples:
            y_true.append(sample.is_poisoned)
            try:
                result = detector.detect(sample.document)
                y_scores.append(result.confidence)
            except Exception:
                y_scores.append(0.0)

        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            y_pred = [score >= threshold for score in y_scores]
            cm = MetricsCalculator.confusion_matrix(y_true, y_pred)

            if metric == "f1_score":
                score = cm.f1_score
            elif metric == "precision":
                score = cm.precision
            elif metric == "recall":
                score = cm.recall
            else:
                score = cm.accuracy

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score


def evaluate_defense_on_dataset(
    dataset: BenchmarkDataset,
    presets: Optional[List[str]] = None,
) -> BenchmarkResult:
    """Evaluate defense across multiple presets.

    Args:
        dataset: Benchmark dataset
        presets: List of detector presets to test

    Returns:
        Benchmark result
    """
    if presets is None:
        presets = ["permissive", "default", "strict"]

    evaluator = DefenseEvaluator()
    results = {}

    for preset in presets:
        config = DefenseConfig(detector_preset=preset)
        results[preset] = evaluator.evaluate_full(dataset, config)

    # Find best configuration
    best_preset = None
    best_f1 = 0.0

    for preset, result in results.items():
        f1 = result.detection_metrics.confusion_matrix.f1_score
        if f1 > best_f1:
            best_f1 = f1
            best_preset = preset

    return BenchmarkResult(
        name=f"defense_eval_{dataset.name}",
        detection=results.get(best_preset, list(results.values())[0]).detection_metrics
        if results else None,
        defense=results.get(best_preset, list(results.values())[0]).defense_metrics
        if results else None,
        metadata={
            "dataset": dataset.name,
            "presets_tested": presets,
            "best_preset": best_preset,
            "best_f1_score": best_f1,
            "all_results": {k: v.to_dict() for k, v in results.items()},
        },
    )


def create_defense_benchmark_suite() -> List[DefenseConfig]:
    """Create a standard suite of defense configurations to benchmark.

    Returns:
        List of defense configurations
    """
    configs = []

    # Vary detector presets
    for preset in ["permissive", "default", "strict"]:
        configs.append(DefenseConfig(
            detector_preset=preset,
            defense_level=DefenseLevel.STANDARD,
        ))

    # Vary defense levels
    for level in DefenseLevel:
        configs.append(DefenseConfig(
            detector_preset="default",
            defense_level=level,
        ))

    # Combinations
    configs.append(DefenseConfig(
        detector_preset="strict",
        defense_level=DefenseLevel.PARANOID,
        enable_quarantine=True,
        enable_monitoring=True,
    ))

    configs.append(DefenseConfig(
        detector_preset="permissive",
        defense_level=DefenseLevel.MINIMAL,
        enable_quarantine=False,
        enable_monitoring=False,
    ))

    return configs
