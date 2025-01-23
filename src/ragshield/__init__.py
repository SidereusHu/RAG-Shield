"""
RAG-Shield: Security Framework for Retrieval-Augmented Generation Systems
"""

__version__ = "0.1.0"

from ragshield.core import RAGSystem, KnowledgeBase, Document, Embedder
from ragshield.detection import (
    PoisonDetector,
    PerplexityDetector,
    SimilarityDetector,
    create_poison_detector,
)

__all__ = [
    "RAGSystem",
    "KnowledgeBase",
    "Document",
    "Embedder",
    "PoisonDetector",
    "PerplexityDetector",
    "SimilarityDetector",
    "create_poison_detector",
]
