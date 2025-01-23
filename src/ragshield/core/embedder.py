"""Text embedding components for RAG system."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed text(s) into vector representation(s).

        Args:
            texts: Single text string or list of text strings

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings.

        Returns:
            Embedding dimension
        """
        pass


class SentenceTransformerEmbedder(Embedder):
    """Embedder using Sentence Transformers models.

    Args:
        model_name: Name of the sentence-transformers model
        device: Device to run the model on ('cpu' or 'cuda')
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed text(s) into vector representation(s).

        Args:
            texts: Single text string or list of text strings

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings.

        Returns:
            Embedding dimension
        """
        return self._embedding_dim


class MockEmbedder(Embedder):
    """Mock embedder for testing purposes.

    Creates simple embeddings based on text length and character frequencies.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Create mock embeddings.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Array of mock embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.embedding_dim)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim
