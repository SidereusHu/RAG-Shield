"""Knowledge base for storing and managing documents."""

from typing import List, Optional, Dict, Any, Tuple
from ragshield.core.document import Document
import json
import numpy as np


class KnowledgeBase:
    """Knowledge base for storing documents.

    A knowledge base stores documents and their embeddings, providing
    methods for adding, removing, and querying documents.

    Attributes:
        documents: List of Document objects in the knowledge base
    """

    def __init__(self):
        """Initialize an empty knowledge base."""
        self.documents: List[Document] = []
        self._doc_id_to_idx: Dict[str, int] = {}

    def add_document(self, document: Document) -> None:
        """Add a document to the knowledge base.

        Args:
            document: Document to add

        Raises:
            ValueError: If document with same doc_id already exists
        """
        if document.doc_id in self._doc_id_to_idx:
            raise ValueError(f"Document with id {document.doc_id} already exists")

        self._doc_id_to_idx[document.doc_id] = len(self.documents)
        self.documents.append(document)

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the knowledge base.

        Args:
            documents: List of documents to add
        """
        for doc in documents:
            self.add_document(doc)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was removed, False if not found
        """
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is None:
            return False

        # Remove from list
        del self.documents[idx]

        # Rebuild index
        self._doc_id_to_idx = {doc.doc_id: i for i, doc in enumerate(self.documents)}

        return True

    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self.documents.clear()
        self._doc_id_to_idx.clear()

    def get_all_documents(self) -> List[Document]:
        """Get all documents in the knowledge base.

        Returns:
            List of all documents
        """
        return self.documents.copy()

    def get_embeddings(self) -> List[List[float]]:
        """Get embeddings of all documents.

        Returns:
            List of embeddings

        Raises:
            ValueError: If any document doesn't have an embedding
        """
        embeddings = []
        for doc in self.documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")
            embeddings.append(doc.embedding)
        return embeddings

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Search for most similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of (document, similarity_score) tuples, sorted by score
        """
        if not self.documents:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        results = []
        for doc in self.documents:
            if doc.embedding is None:
                continue
            doc_emb = np.array(doc.embedding)
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm == 0:
                continue
            doc_emb = doc_emb / doc_norm
            similarity = float(np.dot(query, doc_emb))
            results.append((doc, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def size(self) -> int:
        """Get the number of documents in the knowledge base.

        Returns:
            Number of documents
        """
        return len(self.documents)

    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge base to dictionary.

        Returns:
            Dictionary representation
        """
        return {"documents": [doc.to_dict() for doc in self.documents]}

    def to_json(self, filepath: str) -> None:
        """Save knowledge base to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBase":
        """Create knowledge base from dictionary.

        Args:
            data: Dictionary containing knowledge base data

        Returns:
            KnowledgeBase instance
        """
        kb = cls()
        for doc_data in data.get("documents", []):
            doc = Document.from_dict(doc_data)
            kb.add_document(doc)
        return kb

    @classmethod
    def from_json(cls, filepath: str) -> "KnowledgeBase":
        """Load knowledge base from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            KnowledgeBase instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> Document:
        """Get document by index."""
        return self.documents[idx]

    def __iter__(self):
        """Iterate over documents."""
        return iter(self.documents)

    def __str__(self) -> str:
        """Return string representation."""
        return f"KnowledgeBase(size={len(self.documents)})"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return self.__str__()
