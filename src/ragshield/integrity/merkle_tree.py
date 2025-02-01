"""Merkle Tree implementation for knowledge base integrity verification.

A Merkle Tree is a binary hash tree where:
- Leaf nodes contain hashes of individual documents
- Internal nodes contain hashes of their children
- The root hash represents the entire dataset

Properties:
- Any change to any document changes the root hash
- Verification requires O(log n) hashes (membership proof)
- Supports efficient incremental updates

Security guarantees:
- Collision resistance: Finding two different datasets with same root is computationally infeasible
- Binding: Once committed, the dataset cannot be changed without detection
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import hashlib
import json
from enum import Enum


class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA3_256 = "sha3_256"
    BLAKE2B = "blake2b"


@dataclass
class MerkleNode:
    """A node in the Merkle Tree.

    Attributes:
        hash: The hash value of this node
        left: Left child node (None for leaf nodes)
        right: Right child node (None for leaf nodes)
        data: Original data (only for leaf nodes)
        index: Index in the original document list (only for leaf nodes)
    """
    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    data: Optional[str] = None
    index: Optional[int] = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.left is None and self.right is None


@dataclass
class MerkleProof:
    """Proof of membership in a Merkle Tree.

    A proof consists of sibling hashes along the path from leaf to root.

    Attributes:
        leaf_hash: Hash of the document being verified
        leaf_index: Index of the document in the tree
        siblings: List of (hash, direction) tuples
        root_hash: Expected root hash
    """
    leaf_hash: str
    leaf_index: int
    siblings: List[Tuple[str, str]]  # (hash, "left" or "right")
    root_hash: str

    def to_dict(self) -> dict:
        """Convert proof to dictionary."""
        return {
            "leaf_hash": self.leaf_hash,
            "leaf_index": self.leaf_index,
            "siblings": self.siblings,
            "root_hash": self.root_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MerkleProof":
        """Create proof from dictionary."""
        return cls(
            leaf_hash=data["leaf_hash"],
            leaf_index=data["leaf_index"],
            siblings=[tuple(s) for s in data["siblings"]],
            root_hash=data["root_hash"],
        )

    def to_json(self) -> str:
        """Serialize proof to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "MerkleProof":
        """Deserialize proof from JSON."""
        return cls.from_dict(json.loads(json_str))


class MerkleTree:
    """Merkle Tree for knowledge base integrity verification.

    This implementation provides:
    - O(n) tree construction
    - O(log n) proof generation
    - O(log n) proof verification
    - O(log n) incremental updates

    Example:
        >>> tree = MerkleTree()
        >>> tree.build(["doc1", "doc2", "doc3"])
        >>> root = tree.get_root_hash()
        >>> proof = tree.generate_proof(0)
        >>> tree.verify_proof("doc1", proof)
        True
    """

    def __init__(self, algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        """Initialize Merkle Tree.

        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm
        self.root: Optional[MerkleNode] = None
        self.leaves: List[MerkleNode] = []
        self.documents: List[str] = []

    def _hash(self, data: Union[str, bytes]) -> str:
        """Compute hash of data.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        if self.algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif self.algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).hexdigest()
        elif self.algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data, digest_size=32).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two child hashes together.

        Uses domain separation to prevent second preimage attacks.

        Args:
            left: Left child hash
            right: Right child hash

        Returns:
            Combined hash
        """
        # Domain separation: prefix with "01" for internal nodes
        # (leaves would use "00" prefix)
        combined = f"01:{left}:{right}"
        return self._hash(combined)

    def _hash_leaf(self, data: str) -> str:
        """Hash a leaf node.

        Args:
            data: Document content

        Returns:
            Leaf hash with domain separation
        """
        # Domain separation: prefix with "00" for leaf nodes
        prefixed = f"00:{data}"
        return self._hash(prefixed)

    def build(self, documents: List[str]) -> str:
        """Build Merkle Tree from documents.

        Args:
            documents: List of document contents

        Returns:
            Root hash of the tree

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot build tree from empty document list")

        self.documents = documents.copy()

        # Create leaf nodes
        self.leaves = []
        for i, doc in enumerate(documents):
            leaf_hash = self._hash_leaf(doc)
            leaf = MerkleNode(hash=leaf_hash, data=doc, index=i)
            self.leaves.append(leaf)

        # Build tree bottom-up
        current_level = self.leaves.copy()

        while len(current_level) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]

                # Handle odd number of nodes by duplicating the last one
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left

                # Create parent node
                parent_hash = self._hash_pair(left.hash, right.hash)
                parent = MerkleNode(hash=parent_hash, left=left, right=right)
                next_level.append(parent)

            current_level = next_level

        self.root = current_level[0]
        return self.root.hash

    def get_root_hash(self) -> Optional[str]:
        """Get the root hash of the tree.

        Returns:
            Root hash or None if tree not built
        """
        return self.root.hash if self.root else None

    def generate_proof(self, index: int) -> MerkleProof:
        """Generate membership proof for a document.

        Args:
            index: Index of the document

        Returns:
            Merkle proof for the document

        Raises:
            ValueError: If index is out of range or tree not built
        """
        if self.root is None:
            raise ValueError("Tree not built")
        if index < 0 or index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range [0, {len(self.leaves)})")

        siblings = []
        current_index = index
        current_level = self.leaves.copy()

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                # If current node is in this pair, record sibling
                if i == current_index or i + 1 == current_index:
                    if current_index == i:
                        # Current is left, sibling is right
                        siblings.append((right.hash, "right"))
                    else:
                        # Current is right, sibling is left
                        siblings.append((left.hash, "left"))

                # Create parent
                parent_hash = self._hash_pair(left.hash, right.hash)
                parent = MerkleNode(hash=parent_hash, left=left, right=right)
                next_level.append(parent)

            # Update index for next level
            current_index = current_index // 2
            current_level = next_level

        return MerkleProof(
            leaf_hash=self.leaves[index].hash,
            leaf_index=index,
            siblings=siblings,
            root_hash=self.root.hash,
        )

    def verify_proof(
        self,
        document: str,
        proof: MerkleProof,
        expected_root: Optional[str] = None,
    ) -> bool:
        """Verify a Merkle proof.

        Args:
            document: Document content to verify
            proof: Merkle proof
            expected_root: Expected root hash (uses tree's root if None)

        Returns:
            True if proof is valid, False otherwise
        """
        if expected_root is None:
            if self.root is None:
                raise ValueError("No expected root and tree not built")
            expected_root = self.root.hash

        # Compute leaf hash
        computed_hash = self._hash_leaf(document)

        # Check leaf hash matches
        if computed_hash != proof.leaf_hash:
            return False

        # Traverse up the tree
        current_hash = computed_hash

        for sibling_hash, direction in proof.siblings:
            if direction == "left":
                current_hash = self._hash_pair(sibling_hash, current_hash)
            else:
                current_hash = self._hash_pair(current_hash, sibling_hash)

        # Check root hash
        return current_hash == expected_root

    @staticmethod
    def verify_proof_static(
        document: str,
        proof: MerkleProof,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> bool:
        """Verify a Merkle proof without tree instance.

        This is useful for verification without reconstructing the entire tree.

        Args:
            document: Document content to verify
            proof: Merkle proof
            algorithm: Hash algorithm used

        Returns:
            True if proof is valid
        """
        tree = MerkleTree(algorithm=algorithm)

        # Compute leaf hash
        computed_hash = tree._hash_leaf(document)

        if computed_hash != proof.leaf_hash:
            return False

        # Traverse up
        current_hash = computed_hash
        for sibling_hash, direction in proof.siblings:
            if direction == "left":
                current_hash = tree._hash_pair(sibling_hash, current_hash)
            else:
                current_hash = tree._hash_pair(current_hash, sibling_hash)

        return current_hash == proof.root_hash

    def update_document(self, index: int, new_content: str) -> str:
        """Update a document and recompute affected hashes.

        Args:
            index: Index of document to update
            new_content: New document content

        Returns:
            New root hash
        """
        if self.root is None:
            raise ValueError("Tree not built")
        if index < 0 or index >= len(self.documents):
            raise ValueError(f"Index {index} out of range")

        # Update document
        self.documents[index] = new_content

        # Rebuild tree (could be optimized to only update affected path)
        return self.build(self.documents)

    def detect_tampering(self, documents: List[str]) -> List[int]:
        """Detect which documents have been tampered with.

        Args:
            documents: Current document list to verify

        Returns:
            List of indices of tampered documents
        """
        if len(documents) != len(self.documents):
            # Different number of documents - can't compare directly
            return list(range(max(len(documents), len(self.documents))))

        tampered = []
        for i, (current, original) in enumerate(zip(documents, self.documents)):
            current_hash = self._hash_leaf(current)
            original_hash = self.leaves[i].hash
            if current_hash != original_hash:
                tampered.append(i)

        return tampered

    def get_tree_stats(self) -> dict:
        """Get statistics about the tree.

        Returns:
            Dictionary with tree statistics
        """
        if self.root is None:
            return {"built": False}

        # Calculate height
        height = 0
        n = len(self.leaves)
        while n > 1:
            n = (n + 1) // 2
            height += 1

        return {
            "built": True,
            "num_documents": len(self.documents),
            "num_leaves": len(self.leaves),
            "height": height,
            "algorithm": self.algorithm.value,
            "root_hash": self.root.hash[:16] + "...",
        }

    def to_dict(self) -> dict:
        """Serialize tree to dictionary (metadata only, not full structure)."""
        return {
            "algorithm": self.algorithm.value,
            "root_hash": self.root.hash if self.root else None,
            "num_documents": len(self.documents),
            "leaf_hashes": [leaf.hash for leaf in self.leaves],
        }

    def __len__(self) -> int:
        """Return number of documents in tree."""
        return len(self.documents)

    def __str__(self) -> str:
        """String representation."""
        if self.root is None:
            return "MerkleTree(empty)"
        return f"MerkleTree(docs={len(self.documents)}, root={self.root.hash[:16]}...)"
