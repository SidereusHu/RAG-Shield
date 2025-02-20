"""Attack forensics components for RAG systems.

This module provides forensic analysis capabilities for investigating
and attributing poisoning attacks.

Components:
- Provenance tracking: Chain of custody for documents
- Pattern analysis: Attack pattern identification
- Timeline reconstruction: Attack progression analysis
- Attribution: Source identification and campaign linking

Example:
    >>> from ragshield.forensics import ProvenanceTracker, AttackPatternAnalyzer
    >>>
    >>> # Track document provenance
    >>> tracker = ProvenanceTracker()
    >>> chain = tracker.create_chain(document, source="api", actor="user123")
    >>>
    >>> # Analyze attack patterns
    >>> analyzer = AttackPatternAnalyzer()
    >>> result = analyzer.analyze(suspicious_document)
    >>> print(f"Detected patterns: {result.detected_patterns}")
"""

# Provenance tracking
from ragshield.forensics.provenance import (
    ProvenanceEventType,
    ProvenanceEvent,
    ProvenanceChain,
    ProvenanceTracker,
)

# Attack pattern analysis
from ragshield.forensics.analyzer import (
    AttackPattern,
    PatternMatch,
    AttackFingerprint,
    AnalysisResult,
    AttackPatternAnalyzer,
)

# Timeline reconstruction
from ragshield.forensics.timeline import (
    TimelineEventType,
    TimelineEvent,
    AttackPhase,
    TimelineReport,
    AttackTimelineReconstructor,
)

# Attribution
from ragshield.forensics.attribution import (
    AttributionConfidence,
    AttributionSource,
    AttackCampaign,
    AttributionReport,
    AttackAttributor,
)

__all__ = [
    # Provenance
    "ProvenanceEventType",
    "ProvenanceEvent",
    "ProvenanceChain",
    "ProvenanceTracker",
    # Analyzer
    "AttackPattern",
    "PatternMatch",
    "AttackFingerprint",
    "AnalysisResult",
    "AttackPatternAnalyzer",
    # Timeline
    "TimelineEventType",
    "TimelineEvent",
    "AttackPhase",
    "TimelineReport",
    "AttackTimelineReconstructor",
    # Attribution
    "AttributionConfidence",
    "AttributionSource",
    "AttackCampaign",
    "AttributionReport",
    "AttackAttributor",
]
