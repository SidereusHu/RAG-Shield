"""Factory functions for creating poison detectors."""

from typing import List, Optional, Literal

from ragshield.core.document import Document
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel
from ragshield.detection.perplexity import PerplexityDetector
from ragshield.detection.similarity import SimilarityDetector
from ragshield.detection.semantic import SemanticDetector


class EnsembleDetector(PoisonDetector):
    """Ensemble detector combining multiple detection methods.

    Args:
        detectors: List of detector instances
        mode: Combination mode ('any', 'all', 'majority', 'weighted')
        weights: Optional weights for each detector (for 'weighted' mode)
        threshold: Threshold for 'weighted' mode
    """

    def __init__(
        self,
        detectors: List[PoisonDetector],
        mode: Literal["any", "all", "majority", "weighted"] = "majority",
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
    ):
        if not detectors:
            raise ValueError("At least one detector is required")

        self.detectors = detectors
        self.mode = mode
        self.threshold = threshold

        if weights is None:
            self.weights = [1.0 / len(detectors)] * len(detectors)
        else:
            if len(weights) != len(detectors):
                raise ValueError("Number of weights must match number of detectors")
            total = sum(weights)
            self.weights = [w / total for w in weights]  # Normalize

    def detect(self, document: Document) -> DetectionResult:
        """Detect using ensemble of detectors.

        Args:
            document: Document to check

        Returns:
            Combined detection result
        """
        results = [detector.detect(document) for detector in self.detectors]

        # Combine results based on mode
        poisoned_count = sum(1 for r in results if r.is_poisoned)

        if self.mode == "any":
            is_poisoned = poisoned_count > 0
        elif self.mode == "all":
            is_poisoned = poisoned_count == len(results)
        elif self.mode == "majority":
            is_poisoned = poisoned_count > len(results) / 2
        elif self.mode == "weighted":
            weighted_score = sum(
                w * (1.0 if r.is_poisoned else 0.0) for w, r in zip(self.weights, results)
            )
            is_poisoned = weighted_score >= self.threshold
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Calculate combined confidence
        if is_poisoned:
            # Average confidence of detectors that flagged it
            positive_results = [r for r in results if r.is_poisoned]
            confidence = sum(r.confidence for r in positive_results) / len(positive_results)
        else:
            # Average confidence of detectors that didn't flag it
            negative_results = [r for r in results if not r.is_poisoned]
            if negative_results:
                confidence = sum(r.confidence for r in negative_results) / len(negative_results)
            else:
                confidence = 0.5

        # Determine threat level (highest among positive detections)
        if is_poisoned:
            threat_levels = [r.threat_level for r in results if r.is_poisoned]
            threat_order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                          ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            threat_level = max(threat_levels, key=lambda x: threat_order.index(x))
        else:
            threat_level = ThreatLevel.NONE

        # Combine reasons
        positive_reasons = [r.reason for r in results if r.is_poisoned]
        reason = " | ".join(positive_reasons) if positive_reasons else "No detectors flagged this document"

        # Combined score (weighted average)
        combined_score = sum(w * r.score for w, r in zip(self.weights, results))

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason,
            score=combined_score,
            metadata={
                "detector_results": [
                    {
                        "detector": type(d).__name__,
                        "is_poisoned": r.is_poisoned,
                        "score": r.score,
                    }
                    for d, r in zip(self.detectors, results)
                ],
                "poisoned_count": poisoned_count,
                "total_detectors": len(results),
            },
        )


def create_poison_detector(
    preset: Literal["strict", "default", "permissive"] = "default",
    use_perplexity: bool = True,
    use_similarity: bool = True,
    use_semantic: bool = True,
    perplexity_threshold: Optional[float] = None,
    cluster_threshold: Optional[float] = None,
    semantic_threshold: Optional[float] = None,
) -> PoisonDetector:
    """Create a poison detector with preset configuration.

    Args:
        preset: Configuration preset
            - 'strict': Low thresholds, high sensitivity (more false positives)
            - 'default': Balanced configuration
            - 'permissive': High thresholds, low sensitivity (fewer false positives)
        use_perplexity: Include perplexity-based detection
        use_similarity: Include similarity-based detection
        use_semantic: Include semantic-based detection
        perplexity_threshold: Override perplexity threshold
        cluster_threshold: Override similarity cluster threshold
        semantic_threshold: Override semantic confidence threshold

    Returns:
        Configured poison detector (ensemble if multiple methods enabled)
    """
    # Preset configurations
    presets = {
        "strict": {
            "perplexity_threshold": 50.0,
            "cluster_threshold": 0.90,
            "semantic_threshold": 0.3,
            "mode": "any",
        },
        "default": {
            "perplexity_threshold": 100.0,
            "cluster_threshold": 0.95,
            "semantic_threshold": 0.5,
            "mode": "majority",
        },
        "permissive": {
            "perplexity_threshold": 150.0,
            "cluster_threshold": 0.98,
            "semantic_threshold": 0.7,
            "mode": "all",
        },
    }

    config = presets[preset]

    # Override with provided values
    if perplexity_threshold is not None:
        config["perplexity_threshold"] = perplexity_threshold
    if cluster_threshold is not None:
        config["cluster_threshold"] = cluster_threshold
    if semantic_threshold is not None:
        config["semantic_threshold"] = semantic_threshold

    # Create detectors
    detectors = []

    if use_perplexity:
        detectors.append(PerplexityDetector(threshold=config["perplexity_threshold"]))

    if use_similarity:
        detectors.append(SimilarityDetector(cluster_threshold=config["cluster_threshold"]))

    if use_semantic:
        detectors.append(SemanticDetector(confidence_threshold=config["semantic_threshold"]))

    # Return single detector or ensemble
    if len(detectors) == 0:
        raise ValueError("At least one detection method must be enabled")
    elif len(detectors) == 1:
        return detectors[0]
    else:
        return EnsembleDetector(detectors=detectors, mode=config["mode"])
