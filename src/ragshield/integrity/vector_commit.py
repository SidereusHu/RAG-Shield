"""Vector Commitment schemes for embedding integrity.

A Vector Commitment allows committing to a vector of values such that:
1. The commitment is binding (cannot change values after committing)
2. Individual elements can be opened and verified independently
3. The commitment is hiding (optional, for privacy)

In RAG context:
- Commit to document embeddings when adding to knowledge base
- Verify embedding integrity before retrieval
- Detect if embeddings have been tampered with

Security properties:
- Position binding: Cannot open to different value at same position
- Collision resistance: Cannot find two vectors with same commitment
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import hashlib
import json
import numpy as np
import secrets


@dataclass
class Commitment:
    """A cryptographic commitment to a vector.

    Attributes:
        value: The commitment value (hash or algebraic element)
        metadata: Additional metadata (algorithm, parameters, etc.)
    """
    value: str
    metadata: dict

    def to_dict(self) -> dict:
        return {"value": self.value, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Commitment":
        return cls(value=data["value"], metadata=data["metadata"])

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Commitment":
        return cls.from_dict(json.loads(json_str))


@dataclass
class OpeningProof:
    """Proof for opening a commitment at a specific position.

    Attributes:
        position: Index being opened
        value: Value at that position
        proof_data: Auxiliary data needed for verification
    """
    position: int
    value: float
    proof_data: dict


class VectorCommitment(ABC):
    """Abstract base class for vector commitment schemes."""

    @abstractmethod
    def commit(self, vector: np.ndarray) -> Commitment:
        """Commit to a vector.

        Args:
            vector: Vector to commit to

        Returns:
            Commitment object
        """
        pass

    @abstractmethod
    def verify(self, vector: np.ndarray, commitment: Commitment) -> bool:
        """Verify that a vector matches a commitment.

        Args:
            vector: Vector to verify
            commitment: Commitment to check against

        Returns:
            True if vector matches commitment
        """
        pass

    @abstractmethod
    def open(self, vector: np.ndarray, position: int, commitment: Commitment) -> OpeningProof:
        """Open commitment at a specific position.

        Args:
            vector: Original vector
            position: Position to open
            commitment: Commitment to the vector

        Returns:
            Opening proof
        """
        pass

    @abstractmethod
    def verify_opening(self, proof: OpeningProof, commitment: Commitment) -> bool:
        """Verify an opening proof.

        Args:
            proof: Opening proof
            commitment: Commitment

        Returns:
            True if proof is valid
        """
        pass


class HashBasedCommitment(VectorCommitment):
    """Hash-based vector commitment using Merkle Tree structure.

    This scheme:
    - Commits to vector by hashing all elements
    - Provides O(log n) opening proofs using Merkle paths
    - Is binding (collision resistant) under random oracle model

    Security: Based on hash function collision resistance (SHA-256 by default).
    """

    def __init__(self, hash_algorithm: str = "sha256", precision: int = 8):
        """Initialize hash-based commitment.

        Args:
            hash_algorithm: Hash algorithm to use
            precision: Decimal precision for float-to-string conversion
        """
        self.hash_algorithm = hash_algorithm
        self.precision = precision

    def _hash(self, data: Union[str, bytes]) -> str:
        """Compute hash of data."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        h = hashlib.new(self.hash_algorithm)
        h.update(data)
        return h.hexdigest()

    def _float_to_string(self, value: float) -> str:
        """Convert float to deterministic string representation."""
        return f"{value:.{self.precision}e}"

    def _vector_to_leaf_hashes(self, vector: np.ndarray) -> List[str]:
        """Convert vector to list of leaf hashes."""
        leaf_hashes = []
        for i, val in enumerate(vector):
            # Include position in hash for position binding
            leaf_data = f"leaf:{i}:{self._float_to_string(val)}"
            leaf_hashes.append(self._hash(leaf_data))
        return leaf_hashes

    def _build_merkle_root(self, hashes: List[str]) -> Tuple[str, List[List[str]]]:
        """Build Merkle root from leaf hashes.

        Returns:
            Tuple of (root_hash, tree_levels)
        """
        if not hashes:
            return self._hash("empty"), [[]]

        levels = [hashes]
        current_level = hashes

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = self._hash(f"node:{left}:{right}")
                next_level.append(parent)
            levels.append(next_level)
            current_level = next_level

        return current_level[0], levels

    def commit(self, vector: np.ndarray) -> Commitment:
        """Commit to a vector using Merkle tree.

        Args:
            vector: Vector to commit to

        Returns:
            Commitment containing Merkle root
        """
        vector = np.asarray(vector).flatten()
        leaf_hashes = self._vector_to_leaf_hashes(vector)
        root_hash, _ = self._build_merkle_root(leaf_hashes)

        return Commitment(
            value=root_hash,
            metadata={
                "algorithm": self.hash_algorithm,
                "precision": self.precision,
                "dimension": len(vector),
                "scheme": "merkle_hash",
            },
        )

    def verify(self, vector: np.ndarray, commitment: Commitment) -> bool:
        """Verify vector against commitment.

        Args:
            vector: Vector to verify
            commitment: Commitment to check

        Returns:
            True if vector matches commitment
        """
        vector = np.asarray(vector).flatten()

        # Check dimension
        if len(vector) != commitment.metadata.get("dimension"):
            return False

        # Recompute commitment
        recomputed = self.commit(vector)

        return recomputed.value == commitment.value

    def open(self, vector: np.ndarray, position: int, commitment: Commitment) -> OpeningProof:
        """Open commitment at specific position with Merkle proof.

        Args:
            vector: Original vector
            position: Position to open
            commitment: Commitment

        Returns:
            Opening proof with Merkle path
        """
        vector = np.asarray(vector).flatten()

        if position < 0 or position >= len(vector):
            raise ValueError(f"Position {position} out of range [0, {len(vector)})")

        leaf_hashes = self._vector_to_leaf_hashes(vector)
        _, levels = self._build_merkle_root(leaf_hashes)

        # Build Merkle path
        merkle_path = []
        current_idx = position

        for level in levels[:-1]:  # Exclude root level
            sibling_idx = current_idx ^ 1  # XOR to get sibling index
            if sibling_idx < len(level):
                direction = "right" if current_idx % 2 == 0 else "left"
                merkle_path.append({
                    "hash": level[sibling_idx],
                    "direction": direction,
                })
            current_idx = current_idx // 2

        return OpeningProof(
            position=position,
            value=float(vector[position]),
            proof_data={
                "merkle_path": merkle_path,
                "leaf_hash": leaf_hashes[position],
            },
        )

    def verify_opening(self, proof: OpeningProof, commitment: Commitment) -> bool:
        """Verify opening proof.

        Args:
            proof: Opening proof
            commitment: Commitment

        Returns:
            True if proof is valid
        """
        # Recompute leaf hash
        leaf_data = f"leaf:{proof.position}:{self._float_to_string(proof.value)}"
        computed_hash = self._hash(leaf_data)

        # Check leaf hash matches
        if computed_hash != proof.proof_data.get("leaf_hash"):
            return False

        # Traverse Merkle path
        current_hash = computed_hash
        for step in proof.proof_data.get("merkle_path", []):
            sibling_hash = step["hash"]
            direction = step["direction"]

            if direction == "left":
                current_hash = self._hash(f"node:{sibling_hash}:{current_hash}")
            else:
                current_hash = self._hash(f"node:{current_hash}:{sibling_hash}")

        return current_hash == commitment.value


class BatchVectorCommitment:
    """Commit to multiple vectors (e.g., entire knowledge base embeddings).

    Uses a two-level Merkle structure:
    - Inner level: Individual vector commitments
    - Outer level: Commitment to all vector commitments
    """

    def __init__(self, base_commitment: Optional[VectorCommitment] = None):
        """Initialize batch commitment.

        Args:
            base_commitment: Base commitment scheme for individual vectors
        """
        self.base = base_commitment or HashBasedCommitment()
        self.vector_commitments: List[Commitment] = []
        self.root_commitment: Optional[Commitment] = None

    def commit_batch(self, vectors: List[np.ndarray]) -> Commitment:
        """Commit to a batch of vectors.

        Args:
            vectors: List of vectors to commit to

        Returns:
            Root commitment for entire batch
        """
        # Commit to each vector
        self.vector_commitments = [self.base.commit(v) for v in vectors]

        # Create root commitment from individual commitments
        commitment_hashes = [c.value for c in self.vector_commitments]

        h = hashlib.sha256()
        h.update(json.dumps(commitment_hashes).encode())
        root_hash = h.hexdigest()

        self.root_commitment = Commitment(
            value=root_hash,
            metadata={
                "num_vectors": len(vectors),
                "scheme": "batch_merkle",
            },
        )

        return self.root_commitment

    def verify_vector(self, index: int, vector: np.ndarray) -> bool:
        """Verify a single vector in the batch.

        Args:
            index: Index of vector in batch
            vector: Vector to verify

        Returns:
            True if vector matches commitment at index
        """
        if index < 0 or index >= len(self.vector_commitments):
            return False

        return self.base.verify(vector, self.vector_commitments[index])

    def get_individual_commitment(self, index: int) -> Optional[Commitment]:
        """Get commitment for a specific vector.

        Args:
            index: Vector index

        Returns:
            Individual vector commitment
        """
        if index < 0 or index >= len(self.vector_commitments):
            return None
        return self.vector_commitments[index]


class CommitmentStore:
    """Storage for commitments with lookup by document ID.

    Provides a practical interface for managing commitments in RAG systems.
    """

    def __init__(self, commitment_scheme: Optional[VectorCommitment] = None):
        """Initialize commitment store.

        Args:
            commitment_scheme: Commitment scheme to use
        """
        self.scheme = commitment_scheme or HashBasedCommitment()
        self.commitments: dict[str, Commitment] = {}  # doc_id -> commitment

    def store_commitment(self, doc_id: str, embedding: np.ndarray) -> Commitment:
        """Store commitment for a document embedding.

        Args:
            doc_id: Document identifier
            embedding: Document embedding vector

        Returns:
            Created commitment
        """
        commitment = self.scheme.commit(embedding)
        self.commitments[doc_id] = commitment
        return commitment

    def verify_embedding(self, doc_id: str, embedding: np.ndarray) -> bool:
        """Verify document embedding against stored commitment.

        Args:
            doc_id: Document identifier
            embedding: Embedding to verify

        Returns:
            True if embedding matches commitment
        """
        commitment = self.commitments.get(doc_id)
        if commitment is None:
            return False
        return self.scheme.verify(embedding, commitment)

    def has_commitment(self, doc_id: str) -> bool:
        """Check if commitment exists for document.

        Args:
            doc_id: Document identifier

        Returns:
            True if commitment exists
        """
        return doc_id in self.commitments

    def remove_commitment(self, doc_id: str) -> bool:
        """Remove commitment for document.

        Args:
            doc_id: Document identifier

        Returns:
            True if commitment was removed
        """
        if doc_id in self.commitments:
            del self.commitments[doc_id]
            return True
        return False

    def get_all_doc_ids(self) -> List[str]:
        """Get all document IDs with commitments."""
        return list(self.commitments.keys())

    def export(self) -> dict:
        """Export all commitments to dictionary."""
        return {
            doc_id: c.to_dict()
            for doc_id, c in self.commitments.items()
        }

    def import_commitments(self, data: dict) -> None:
        """Import commitments from dictionary.

        Args:
            data: Dictionary mapping doc_id to commitment dict
        """
        for doc_id, c_dict in data.items():
            self.commitments[doc_id] = Commitment.from_dict(c_dict)
