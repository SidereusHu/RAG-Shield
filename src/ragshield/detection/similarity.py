"""Similarity-based poison detection.

Poisoned documents often cluster together or have unusual similarity patterns.
This detector identifies anomalies based on document similarity distributions.
"""

from typing import List, Optional
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel, ScanResult


class SimilarityDetector(PoisonDetector):
    """Detect poisoned documents based on similarity patterns.

    Poisoned documents may exhibit:
    1. Unusually high similarity to each other (clustering)
    2. High similarity to specific queries (targeted attack)
    3. Abnormal similarity distribution

    Args:
        cluster_threshold: Threshold for detecting document clusters
        outlier_threshold: Threshold for detecting outlier documents
        min_cluster_size: Minimum size for a suspicious cluster
    """

    def __init__(
        self,
        cluster_threshold: float = 0.95,
        outlier_threshold: float = 0.1,
        min_cluster_size: int = 3,
    ):
        self.cluster_threshold = cluster_threshold
        self.outlier_threshold = outlier_threshold
        self.min_cluster_size = min_cluster_size
        self._similarity_matrix = None
        self._documents = None

    def _compute_similarity_matrix(self, documents: List[Document]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            documents: List of documents with embeddings

        Returns:
            Similarity matrix of shape (n, n)
        """
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")
            embeddings.append(doc.embedding)

        embeddings = np.array(embeddings)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def _detect_clusters(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Detect suspicious document clusters.

        Args:
            similarity_matrix: Pairwise similarity matrix

        Returns:
            List of cluster indices
        """
        n = similarity_matrix.shape[0]
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            # Find documents highly similar to document i
            cluster = [i]
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i, j] > self.cluster_threshold:
                    cluster.append(j)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
                visited.update(cluster)

        return clusters

    def _detect_outliers(self, similarity_matrix: np.ndarray) -> List[int]:
        """Detect outlier documents with low average similarity.

        Args:
            similarity_matrix: Pairwise similarity matrix

        Returns:
            List of outlier document indices
        """
        n = similarity_matrix.shape[0]
        outliers = []

        for i in range(n):
            # Average similarity to other documents (excluding self)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            avg_similarity = similarity_matrix[i, mask].mean()

            if avg_similarity < self.outlier_threshold:
                outliers.append(i)

        return outliers

    def fit(self, knowledge_base: KnowledgeBase) -> None:
        """Fit the detector on a knowledge base.

        Args:
            knowledge_base: Knowledge base to analyze
        """
        self._documents = list(knowledge_base.documents)
        self._similarity_matrix = self._compute_similarity_matrix(self._documents)

    def detect(self, document: Document) -> DetectionResult:
        """Detect if a single document is suspicious.

        For single document detection, we compute similarity features
        and use statistical thresholds.

        Args:
            document: Document to check

        Returns:
            Detection result
        """
        if document.embedding is None:
            return DetectionResult(
                is_poisoned=False,
                confidence=0.0,
                threat_level=ThreatLevel.NONE,
                reason="No embedding available for analysis",
                score=0.0,
            )

        # Compute self-similarity features
        embedding = np.array(document.embedding)

        # Feature 1: Embedding norm (adversarial embeddings may have unusual norms)
        norm = np.linalg.norm(embedding)
        norm_suspicious = norm < 0.5 or norm > 2.0

        # Feature 2: Embedding entropy (adversarial may have low entropy)
        embedding_abs = np.abs(embedding)
        embedding_norm = embedding_abs / (embedding_abs.sum() + 1e-10)
        entropy = -np.sum(embedding_norm * np.log(embedding_norm + 1e-10))
        entropy_suspicious = entropy < 2.0

        # Feature 3: Sparsity (adversarial may be unusually sparse or dense)
        sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
        sparsity_suspicious = sparsity > 0.8 or sparsity < 0.1

        # Combine features
        suspicious_features = sum([norm_suspicious, entropy_suspicious, sparsity_suspicious])
        is_poisoned = suspicious_features >= 2

        confidence = suspicious_features / 3.0

        if not is_poisoned:
            threat_level = ThreatLevel.NONE
        elif suspicious_features == 3:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM

        reasons = []
        if norm_suspicious:
            reasons.append(f"unusual embedding norm ({norm:.2f})")
        if entropy_suspicious:
            reasons.append(f"low embedding entropy ({entropy:.2f})")
        if sparsity_suspicious:
            reasons.append(f"unusual sparsity ({sparsity:.2f})")

        reason = ", ".join(reasons) if reasons else "No anomalies detected"

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason,
            score=float(suspicious_features),
            metadata={
                "norm": norm,
                "entropy": entropy,
                "sparsity": sparsity,
            },
        )

    def scan_knowledge_base(
        self, knowledge_base: KnowledgeBase, threshold: Optional[float] = None
    ) -> ScanResult:
        """Scan knowledge base for poisoned documents using clustering.

        Args:
            knowledge_base: Knowledge base to scan
            threshold: Optional cluster threshold override

        Returns:
            Scan result with detected clusters and outliers
        """
        if threshold is not None:
            self.cluster_threshold = threshold

        # Fit on knowledge base
        self.fit(knowledge_base)

        # Detect suspicious patterns
        clusters = self._detect_clusters(self._similarity_matrix)
        outliers = self._detect_outliers(self._similarity_matrix)

        # Mark documents as poisoned
        suspicious_indices = set()
        for cluster in clusters:
            suspicious_indices.update(cluster)
        suspicious_indices.update(outliers)

        poisoned_docs = []
        for idx in suspicious_indices:
            doc = self._documents[idx]

            # Determine if in cluster or outlier
            in_cluster = any(idx in cluster for cluster in clusters)
            is_outlier = idx in outliers

            if in_cluster:
                reason = f"Part of suspicious cluster (similarity > {self.cluster_threshold})"
                threat_level = ThreatLevel.HIGH
            else:
                reason = f"Outlier document (avg similarity < {self.outlier_threshold})"
                threat_level = ThreatLevel.MEDIUM

            result = DetectionResult(
                is_poisoned=True,
                confidence=0.8 if in_cluster else 0.6,
                threat_level=threat_level,
                reason=reason,
                score=1.0,
                metadata={"in_cluster": in_cluster, "is_outlier": is_outlier},
            )
            poisoned_docs.append((doc, result))

        total = knowledge_base.size()
        clean_docs = total - len(poisoned_docs)
        detection_rate = (len(poisoned_docs) / total * 100) if total > 0 else 0.0

        return ScanResult(
            total_documents=total,
            poisoned_docs=poisoned_docs,
            clean_docs=clean_docs,
            detection_rate=detection_rate,
        )
