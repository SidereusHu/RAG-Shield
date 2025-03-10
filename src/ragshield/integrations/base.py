"""Base classes for RAG framework integrations.

Provides abstract interfaces and utilities for integrating
RAG-Shield with various RAG frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection import PoisonDetector, create_poison_detector
from ragshield.defense import RAGShield, DefenseLevel
from ragshield.defense.shield import DefenseConfig


class FrameworkType(Enum):
    """Supported RAG frameworks."""

    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    CUSTOM = "custom"


@dataclass
class IntegrationConfig:
    """Configuration for framework integration.

    Attributes:
        defense_level: Security level preset
        detector_preset: Detector configuration preset
        enable_detection: Enable poison detection
        enable_sanitization: Enable input sanitization
        enable_monitoring: Enable security monitoring
        enable_forensics: Enable attack forensics
        auto_quarantine: Auto-quarantine suspicious documents
        on_threat_callback: Callback when threat detected
    """

    defense_level: DefenseLevel = DefenseLevel.STANDARD
    detector_preset: str = "default"
    enable_detection: bool = True
    enable_sanitization: bool = True
    enable_monitoring: bool = True
    enable_forensics: bool = True
    auto_quarantine: bool = True
    on_threat_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecureDocument:
    """Document with security metadata.

    Wraps a document with additional security information
    for use in RAG pipelines.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    doc_id: Optional[str] = None
    is_verified: bool = False
    threat_score: float = 0.0
    security_flags: List[str] = field(default_factory=list)

    def to_document(self) -> Document:
        """Convert to RAG-Shield Document."""
        return Document(
            doc_id=self.doc_id or f"doc_{hash(self.content) % 10000:04x}",
            content=self.content,
            embedding=self.embedding,
            metadata=self.metadata,
        )

    @classmethod
    def from_document(cls, doc: Document, **kwargs) -> "SecureDocument":
        """Create from RAG-Shield Document."""
        return cls(
            content=doc.content,
            metadata=doc.metadata,
            embedding=doc.embedding,
            doc_id=doc.doc_id,
            **kwargs,
        )


class BaseRAGIntegration(ABC):
    """Abstract base class for RAG framework integrations.

    Provides common functionality for integrating RAG-Shield
    with different RAG frameworks.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """Initialize integration.

        Args:
            config: Integration configuration
            knowledge_base: Optional existing knowledge base
        """
        self.config = config or IntegrationConfig()
        self.knowledge_base = knowledge_base or KnowledgeBase()

        # Initialize components based on config
        self._detector: Optional[PoisonDetector] = None
        self._shield: Optional[RAGShield] = None

        if self.config.enable_detection:
            self._detector = create_poison_detector(
                preset=self.config.detector_preset
            )

        # Create shield with configuration
        shield_config = DefenseConfig(
            level=self.config.defense_level,
            auto_quarantine=self.config.auto_quarantine,
            enable_monitoring=self.config.enable_monitoring,
            enable_forensics=self.config.enable_forensics,
        )
        self._shield = RAGShield(
            knowledge_base=self.knowledge_base,
            detector=self._detector,
            config=shield_config,
        )

        # Statistics
        self._stats = {
            "documents_processed": 0,
            "threats_detected": 0,
            "documents_blocked": 0,
            "queries_processed": 0,
        }

    @property
    def shield(self) -> RAGShield:
        """Get the RAG Shield instance."""
        return self._shield

    @property
    def detector(self) -> Optional[PoisonDetector]:
        """Get the poison detector."""
        return self._detector

    @abstractmethod
    def get_framework_type(self) -> FrameworkType:
        """Get the framework type."""
        pass

    @abstractmethod
    def wrap_retriever(self, retriever: Any) -> Any:
        """Wrap a framework-specific retriever with security.

        Args:
            retriever: Framework's retriever object

        Returns:
            Secured retriever wrapper
        """
        pass

    @abstractmethod
    def wrap_vector_store(self, vector_store: Any) -> Any:
        """Wrap a framework-specific vector store with security.

        Args:
            vector_store: Framework's vector store object

        Returns:
            Secured vector store wrapper
        """
        pass

    def secure_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> SecureDocument:
        """Create a secure document with threat analysis.

        Args:
            content: Document content
            metadata: Optional metadata
            embedding: Optional embedding vector

        Returns:
            SecureDocument with security analysis
        """
        doc = Document(
            doc_id=f"doc_{hash(content) % 10000:04x}",
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        secure_doc = SecureDocument(
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            doc_id=doc.doc_id,
        )

        self._stats["documents_processed"] += 1

        # Analyze for threats
        if self._detector and self.config.enable_detection:
            result = self._detector.detect(doc)
            secure_doc.threat_score = result.confidence
            secure_doc.is_verified = not result.is_poisoned

            if result.is_poisoned:
                self._stats["threats_detected"] += 1
                secure_doc.security_flags.append("potential_poison")

                if self.config.on_threat_callback:
                    self.config.on_threat_callback({
                        "doc_id": doc.doc_id,
                        "threat_score": result.confidence,
                        "detection_result": result,
                    })
        else:
            secure_doc.is_verified = True

        return secure_doc

    def ingest_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """Securely ingest a document.

        Args:
            content: Document content
            metadata: Optional metadata
            embedding: Optional embedding vector
            source: Document source identifier

        Returns:
            Ingestion result with security information
        """
        doc = Document(
            doc_id=f"doc_{hash(content) % 10000:04x}",
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        result = self._shield.ingest(doc, source=source)

        if not result.accepted:
            self._stats["documents_blocked"] += 1

        return {
            "accepted": result.accepted,
            "doc_id": doc.doc_id,
            "threat_score": result.detection_result.confidence if result.detection_result else 0,
            "quarantined": result.quarantined,
            "reason": result.rejection_reason,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "shield_stats": self._shield.get_stats() if hasattr(self._shield, 'get_stats') else {},
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "documents_processed": 0,
            "threats_detected": 0,
            "documents_blocked": 0,
            "queries_processed": 0,
        }


class SecureRetrieverMixin:
    """Mixin for adding security to retrievers."""

    def _secure_results(
        self,
        results: List[Any],
        detector: Optional[PoisonDetector],
        extract_content: Callable[[Any], str],
    ) -> List[Any]:
        """Filter and score retrieval results for security.

        Args:
            results: Retrieved documents
            detector: Poison detector
            extract_content: Function to extract content from result

        Returns:
            Filtered results with security scores
        """
        if not detector:
            return results

        secure_results = []
        for result in results:
            content = extract_content(result)
            doc = Document(
                doc_id=f"result_{hash(content) % 10000:04x}",
                content=content,
            )

            detection = detector.detect(doc)

            # Skip highly suspicious results
            if detection.is_poisoned and detection.confidence > 0.9:
                continue

            secure_results.append(result)

        return secure_results
