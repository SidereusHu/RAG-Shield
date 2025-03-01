"""Evaluation metrics for RAG-Shield benchmarks.

Provides comprehensive metrics for evaluating detection, defense,
privacy, and performance of RAG security components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class MetricType(Enum):
    """Types of evaluation metrics."""

    DETECTION = "detection"  # Detection quality metrics
    DEFENSE = "defense"  # Defense effectiveness
    PRIVACY = "privacy"  # Privacy preservation
    PERFORMANCE = "performance"  # Runtime/resource metrics


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification.

    Attributes:
        true_positives: Correctly identified positive cases
        false_positives: Incorrectly identified as positive
        true_negatives: Correctly identified negative cases
        false_negatives: Incorrectly identified as negative
    """

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.true_positives + self.false_positives + \
               self.true_negatives + self.false_negatives

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total

    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity, true positive rate)."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def specificity(self) -> float:
        """Calculate specificity (true negative rate)."""
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_negatives / denominator

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        denominator = self.false_positives + self.true_negatives
        if denominator == 0:
            return 0.0
        return self.false_positives / denominator

    @property
    def false_negative_rate(self) -> float:
        """Calculate false negative rate."""
        denominator = self.false_negatives + self.true_positives
        if denominator == 0:
            return 0.0
        return self.false_negatives / denominator

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
        }


@dataclass
class DetectionMetrics:
    """Metrics for detection evaluation.

    Attributes:
        confusion_matrix: Binary classification matrix
        roc_auc: Area under ROC curve
        pr_auc: Area under Precision-Recall curve
        per_attack_metrics: Metrics broken down by attack type
        threshold_analysis: Metrics at different thresholds
    """

    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    per_attack_metrics: Dict[str, ConfusionMatrix] = field(default_factory=dict)
    threshold_analysis: Dict[float, ConfusionMatrix] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "per_attack_metrics": {
                k: v.to_dict() for k, v in self.per_attack_metrics.items()
            },
            "threshold_analysis": {
                str(k): v.to_dict() for k, v in self.threshold_analysis.items()
            },
        }


@dataclass
class DefenseMetrics:
    """Metrics for defense evaluation.

    Attributes:
        attack_success_rate: Rate at which attacks succeed
        blocked_attacks: Number of blocked attacks
        false_blocks: Number of legitimate requests blocked
        quarantine_rate: Rate of documents quarantined
        response_time: Average defense response time
    """

    attack_success_rate: float = 0.0
    blocked_attacks: int = 0
    total_attacks: int = 0
    false_blocks: int = 0
    total_legitimate: int = 0
    quarantine_rate: float = 0.0
    response_time_ms: float = 0.0

    @property
    def block_rate(self) -> float:
        """Calculate attack block rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.blocked_attacks / self.total_attacks

    @property
    def false_block_rate(self) -> float:
        """Calculate false block rate."""
        if self.total_legitimate == 0:
            return 0.0
        return self.false_blocks / self.total_legitimate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_success_rate": self.attack_success_rate,
            "blocked_attacks": self.blocked_attacks,
            "total_attacks": self.total_attacks,
            "block_rate": self.block_rate,
            "false_blocks": self.false_blocks,
            "total_legitimate": self.total_legitimate,
            "false_block_rate": self.false_block_rate,
            "quarantine_rate": self.quarantine_rate,
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class PrivacyMetrics:
    """Metrics for privacy evaluation.

    Attributes:
        epsilon_used: Total differential privacy budget used
        delta_used: Delta parameter for approximate DP
        query_privacy_score: Score measuring query privacy (0-1)
        information_leakage: Estimated information leakage
    """

    epsilon_used: float = 0.0
    delta_used: float = 0.0
    query_privacy_score: float = 0.0
    information_leakage: float = 0.0
    pir_enabled: bool = False
    retrieval_indistinguishability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epsilon_used": self.epsilon_used,
            "delta_used": self.delta_used,
            "query_privacy_score": self.query_privacy_score,
            "information_leakage": self.information_leakage,
            "pir_enabled": self.pir_enabled,
            "retrieval_indistinguishability": self.retrieval_indistinguishability,
        }


@dataclass
class PerformanceMetrics:
    """Metrics for performance evaluation.

    Attributes:
        latency_ms: Average latency in milliseconds
        throughput: Operations per second
        memory_mb: Memory usage in megabytes
        cpu_percent: CPU utilization percentage
    """

    latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    overhead_ratio: float = 1.0  # Ratio vs baseline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_ms": self.latency_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "overhead_ratio": self.overhead_ratio,
        }


class MetricsCalculator:
    """Calculate evaluation metrics from predictions.

    Provides methods for computing various metrics from
    ground truth and predicted labels/scores.
    """

    @staticmethod
    def confusion_matrix(
        y_true: List[bool],
        y_pred: List[bool],
    ) -> ConfusionMatrix:
        """Calculate confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        cm = ConfusionMatrix()

        for true, pred in zip(y_true, y_pred):
            if true and pred:
                cm.true_positives += 1
            elif not true and pred:
                cm.false_positives += 1
            elif not true and not pred:
                cm.true_negatives += 1
            else:  # true and not pred
                cm.false_negatives += 1

        return cm

    @staticmethod
    def roc_curve(
        y_true: List[bool],
        y_scores: List[float],
        num_thresholds: int = 100,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate ROC curve points.

        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores/probabilities
            num_thresholds: Number of threshold points

        Returns:
            Tuple of (fpr_list, tpr_list, thresholds)
        """
        thresholds = np.linspace(0, 1, num_thresholds)
        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            y_pred = [score >= threshold for score in y_scores]
            cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
            fpr_list.append(cm.false_positive_rate)
            tpr_list.append(cm.recall)  # TPR = recall

        return fpr_list, tpr_list, thresholds.tolist()

    @staticmethod
    def auc(x: List[float], y: List[float]) -> float:
        """Calculate area under curve using trapezoidal rule.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Area under curve
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        # Sort by x
        sorted_pairs = sorted(zip(x, y))
        x_sorted = [p[0] for p in sorted_pairs]
        y_sorted = [p[1] for p in sorted_pairs]

        # Trapezoidal integration
        area = 0.0
        for i in range(len(x_sorted) - 1):
            dx = x_sorted[i + 1] - x_sorted[i]
            avg_y = (y_sorted[i] + y_sorted[i + 1]) / 2
            area += dx * avg_y

        return abs(area)

    @staticmethod
    def roc_auc(y_true: List[bool], y_scores: List[float]) -> float:
        """Calculate ROC AUC score.

        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores

        Returns:
            ROC AUC score
        """
        fpr, tpr, _ = MetricsCalculator.roc_curve(y_true, y_scores)
        return MetricsCalculator.auc(fpr, tpr)

    @staticmethod
    def precision_recall_curve(
        y_true: List[bool],
        y_scores: List[float],
        num_thresholds: int = 100,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate precision-recall curve.

        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores
            num_thresholds: Number of threshold points

        Returns:
            Tuple of (precision_list, recall_list, thresholds)
        """
        thresholds = np.linspace(0, 1, num_thresholds)
        precision_list = []
        recall_list = []

        for threshold in thresholds:
            y_pred = [score >= threshold for score in y_scores]
            cm = MetricsCalculator.confusion_matrix(y_true, y_pred)
            precision_list.append(cm.precision)
            recall_list.append(cm.recall)

        return precision_list, recall_list, thresholds.tolist()

    @staticmethod
    def pr_auc(y_true: List[bool], y_scores: List[float]) -> float:
        """Calculate PR AUC score.

        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores

        Returns:
            PR AUC score
        """
        precision, recall, _ = MetricsCalculator.precision_recall_curve(y_true, y_scores)
        return MetricsCalculator.auc(recall, precision)

    @staticmethod
    def detection_metrics(
        y_true: List[bool],
        y_pred: List[bool],
        y_scores: Optional[List[float]] = None,
        attack_types: Optional[List[str]] = None,
    ) -> DetectionMetrics:
        """Calculate comprehensive detection metrics.

        Args:
            y_true: Ground truth labels (True = poisoned)
            y_pred: Predicted labels
            y_scores: Optional prediction scores
            attack_types: Optional attack type for each sample

        Returns:
            Detection metrics
        """
        metrics = DetectionMetrics()
        metrics.confusion_matrix = MetricsCalculator.confusion_matrix(y_true, y_pred)

        if y_scores:
            metrics.roc_auc = MetricsCalculator.roc_auc(y_true, y_scores)
            metrics.pr_auc = MetricsCalculator.pr_auc(y_true, y_scores)

            # Threshold analysis
            for threshold in [0.3, 0.5, 0.7, 0.9]:
                preds = [score >= threshold for score in y_scores]
                metrics.threshold_analysis[threshold] = MetricsCalculator.confusion_matrix(
                    y_true, preds
                )

        # Per-attack metrics
        if attack_types:
            attack_groups = defaultdict(lambda: ([], []))
            for true, pred, attack in zip(y_true, y_pred, attack_types):
                if attack:  # Only for actual attacks
                    attack_groups[attack][0].append(true)
                    attack_groups[attack][1].append(pred)

            for attack, (trues, preds) in attack_groups.items():
                metrics.per_attack_metrics[attack] = MetricsCalculator.confusion_matrix(
                    trues, preds
                )

        return metrics

    @staticmethod
    def latency_percentiles(
        latencies: List[float],
    ) -> Tuple[float, float, float, float]:
        """Calculate latency percentiles.

        Args:
            latencies: List of latency measurements

        Returns:
            Tuple of (mean, p50, p95, p99)
        """
        if not latencies:
            return 0.0, 0.0, 0.0, 0.0

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        mean = sum(sorted_latencies) / n
        p50 = sorted_latencies[int(n * 0.5)]
        p95 = sorted_latencies[min(int(n * 0.95), n - 1)]
        p99 = sorted_latencies[min(int(n * 0.99), n - 1)]

        return mean, p50, p95, p99

    @staticmethod
    def performance_metrics(
        latencies: List[float],
        memory_samples: Optional[List[float]] = None,
        cpu_samples: Optional[List[float]] = None,
        baseline_latency: Optional[float] = None,
    ) -> PerformanceMetrics:
        """Calculate performance metrics.

        Args:
            latencies: List of latency measurements in ms
            memory_samples: Optional memory usage samples in MB
            cpu_samples: Optional CPU utilization samples
            baseline_latency: Optional baseline for overhead calculation

        Returns:
            Performance metrics
        """
        mean, p50, p95, p99 = MetricsCalculator.latency_percentiles(latencies)

        metrics = PerformanceMetrics(
            latency_ms=mean,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
        )

        if latencies:
            # Throughput in ops/sec
            total_time = sum(latencies) / 1000  # Convert to seconds
            metrics.throughput = len(latencies) / total_time if total_time > 0 else 0

        if memory_samples:
            metrics.memory_mb = max(memory_samples)

        if cpu_samples:
            metrics.cpu_percent = sum(cpu_samples) / len(cpu_samples)

        if baseline_latency and baseline_latency > 0:
            metrics.overhead_ratio = mean / baseline_latency

        return metrics


@dataclass
class BenchmarkResult:
    """Complete benchmark result.

    Attributes:
        name: Benchmark name
        detection: Detection metrics
        defense: Defense metrics
        privacy: Privacy metrics
        performance: Performance metrics
        metadata: Additional metadata
    """

    name: str
    detection: Optional[DetectionMetrics] = None
    defense: Optional[DefenseMetrics] = None
    privacy: Optional[PrivacyMetrics] = None
    performance: Optional[PerformanceMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "metadata": self.metadata,
        }

        if self.detection:
            result["detection"] = self.detection.to_dict()
        if self.defense:
            result["defense"] = self.defense.to_dict()
        if self.privacy:
            result["privacy"] = self.privacy.to_dict()
        if self.performance:
            result["performance"] = self.performance.to_dict()

        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"=== Benchmark: {self.name} ==="]

        if self.detection:
            cm = self.detection.confusion_matrix
            lines.append("\nDetection:")
            lines.append(f"  Accuracy: {cm.accuracy:.3f}")
            lines.append(f"  Precision: {cm.precision:.3f}")
            lines.append(f"  Recall: {cm.recall:.3f}")
            lines.append(f"  F1 Score: {cm.f1_score:.3f}")
            if self.detection.roc_auc > 0:
                lines.append(f"  ROC AUC: {self.detection.roc_auc:.3f}")

        if self.defense:
            lines.append("\nDefense:")
            lines.append(f"  Block Rate: {self.defense.block_rate:.3f}")
            lines.append(f"  False Block Rate: {self.defense.false_block_rate:.3f}")
            lines.append(f"  Attack Success Rate: {self.defense.attack_success_rate:.3f}")

        if self.privacy:
            lines.append("\nPrivacy:")
            lines.append(f"  Epsilon Used: {self.privacy.epsilon_used:.3f}")
            lines.append(f"  Query Privacy Score: {self.privacy.query_privacy_score:.3f}")

        if self.performance:
            lines.append("\nPerformance:")
            lines.append(f"  Latency (avg): {self.performance.latency_ms:.2f}ms")
            lines.append(f"  Latency (p99): {self.performance.latency_p99_ms:.2f}ms")
            lines.append(f"  Throughput: {self.performance.throughput:.1f} ops/sec")
            lines.append(f"  Overhead: {self.performance.overhead_ratio:.2f}x")

        return "\n".join(lines)
