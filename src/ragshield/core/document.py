"""Document data structures for RAG system."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import hashlib


@dataclass
class DocumentMetadata:
    """Metadata for a document."""

    source: str = ""
    timestamp: Optional[datetime] = None
    author: str = ""
    doc_type: str = ""
    tags: list[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "author": self.author,
            "doc_type": self.doc_type,
            "tags": self.tags,
            "custom": self.custom,
        }


@dataclass
class Document:
    """A document in the knowledge base.

    Attributes:
        content: The text content of the document
        doc_id: Unique identifier for the document
        embedding: Vector embedding of the document (optional)
        metadata: Additional metadata about the document
        hash: Content hash for integrity verification
    """

    content: str
    doc_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    hash: Optional[str] = None

    def __post_init__(self):
        """Generate hash and doc_id if not provided."""
        if self.hash is None:
            self.hash = self.compute_hash()
        if self.doc_id is None:
            self.doc_id = self.hash[:16]

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of document content.

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify document integrity by checking hash.

        Returns:
            True if hash matches content, False otherwise
        """
        if self.hash is None:
            return False
        return self.hash == self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary.

        Returns:
            Dictionary representation of the document
        """
        return {
            "content": self.content,
            "doc_id": self.doc_id,
            "embedding": self.embedding,
            "metadata": self.metadata.to_dict(),
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary.

        Args:
            data: Dictionary containing document data

        Returns:
            Document instance
        """
        metadata_dict = data.get("metadata", {})
        timestamp_str = metadata_dict.get("timestamp")
        if timestamp_str:
            metadata_dict["timestamp"] = datetime.fromisoformat(timestamp_str)

        return cls(
            content=data["content"],
            doc_id=data.get("doc_id"),
            embedding=data.get("embedding"),
            metadata=DocumentMetadata(**metadata_dict),
            hash=data.get("hash"),
        )

    def __len__(self) -> int:
        """Return the length of document content."""
        return len(self.content)

    def __str__(self) -> str:
        """Return string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.doc_id}, content='{preview}')"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return self.__str__()
