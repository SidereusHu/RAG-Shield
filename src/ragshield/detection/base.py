"""Base classes for poison detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase


class ThreatLevel(Enum):
    """Threat level classification."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of poison detection.

    Attributes:
        is_poisoned: Whether the document is detected as poisoned
        confidence: Confidence score (0-1)
        threat_level: Severity of the threat
        reason: Explanation of why document was flagged
        score: Numerical detection score
        metadata: Additional detection metadata
    """

    is_poisoned: bool
    confidence: float
    threat_level: ThreatLevel
    reason: str
    score: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScanResult:
    """Result of scanning a knowledge base.

    Attributes:
        total_documents: Total number of documents scanned
        poisoned_docs: List of (Document, DetectionResult) tuples for poisoned docs
        clean_docs: Number of clean documents
        detection_rate: Percentage of documents flagged as poisoned
    """

    total_documents: int
    poisoned_docs: List[tuple[Document, DetectionResult]]
    clean_docs: int
    detection_rate: float


class PoisonDetector(ABC):
    """Abstract base class for poison detectors."""

    @abstractmethod
    def detect(self, document: Document) -> DetectionResult:
        """Detect if a document is poisoned.

        Args:
            document: Document to check

        Returns:
            Detection result
        """
        pass

    def scan_knowledge_base(
        self, knowledge_base: KnowledgeBase, threshold: Optional[float] = None
    ) -> ScanResult:
        """Scan entire knowledge base for poisoned documents.

        Args:
            knowledge_base: Knowledge base to scan
            threshold: Optional threshold override

        Returns:
            Scan result with statistics
        """
        poisoned_docs = []
        total = knowledge_base.size()

        for doc in knowledge_base.documents:
            result = self.detect(doc)
            if result.is_poisoned:
                poisoned_docs.append((doc, result))

        clean_docs = total - len(poisoned_docs)
        detection_rate = (len(poisoned_docs) / total * 100) if total > 0 else 0.0

        return ScanResult(
            total_documents=total,
            poisoned_docs=poisoned_docs,
            clean_docs=clean_docs,
            detection_rate=detection_rate,
        )

    def batch_detect(self, documents: List[Document]) -> List[DetectionResult]:
        """Detect poison in multiple documents.

        Args:
            documents: List of documents to check

        Returns:
            List of detection results
        """
        return [self.detect(doc) for doc in documents]
