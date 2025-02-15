"""Private Information Retrieval (PIR) for RAG Systems.

This module provides cryptographic PIR protocols that allow retrieving
documents without the server learning which documents were retrieved.

Available protocols:
- Single-server PIR: Based on homomorphic encryption (computational security)
- Multi-server PIR: Based on XOR secret sharing (information-theoretic security)
- Threshold PIR: Tolerates server failures using Shamir sharing

Key classes:
- PIRRetriever: High-level interface for PIR-based document retrieval
- SingleServerPIR: Computational PIR using additive homomorphic encryption
- MultiServerPIR: Information-theoretic PIR using XOR sharing

Example:
    >>> from ragshield.pir import PIRRetriever, PIRMode
    >>>
    >>> # Create PIR retriever
    >>> retriever = PIRRetriever(knowledge_base, mode=PIRMode.MULTI_SERVER)
    >>>
    >>> # Private retrieval - server doesn't learn which docs were fetched
    >>> result = retriever.retrieve(query_embedding, top_k=5)
    >>> for doc, score in result.documents:
    ...     print(f"{doc.doc_id}: {score:.4f}")
"""

# Base classes
from ragshield.pir.base import (
    PIRScheme,
    PIRSecurityLevel,
    PIRParameters,
    PIRQuery,
    PIRResponse,
    PIRResult,
    PIRClient,
    PIRServer,
    PIRProtocol,
)

# Single-server PIR
from ragshield.pir.single_server import (
    SingleServerPIR,
    SingleServerPIRClient,
    SingleServerPIRServer,
    SimplifiedPaillier,
    AdditiveHEPublicKey,
    AdditiveHESecretKey,
    AdditiveHECiphertext,
    MatrixPIRClient,
    MatrixPIRServer,
)

# Multi-server PIR
from ragshield.pir.multi_server import (
    MultiServerPIR,
    MultiServerPIRClient,
    MultiServerPIRServer,
    ThresholdPIR,
    XORSecretSharing,
    ShamirSecretSharing,
    SelectionVector,
)

# PIR Retriever
from ragshield.pir.pir_retriever import (
    PIRRetriever,
    PIRMode,
    PIRRetrievalResult,
    HybridPIRRetriever,
    BatchPIRRetriever,
)

__all__ = [
    # Base
    "PIRScheme",
    "PIRSecurityLevel",
    "PIRParameters",
    "PIRQuery",
    "PIRResponse",
    "PIRResult",
    "PIRClient",
    "PIRServer",
    "PIRProtocol",
    # Single-server
    "SingleServerPIR",
    "SingleServerPIRClient",
    "SingleServerPIRServer",
    "SimplifiedPaillier",
    "AdditiveHEPublicKey",
    "AdditiveHESecretKey",
    "AdditiveHECiphertext",
    "MatrixPIRClient",
    "MatrixPIRServer",
    # Multi-server
    "MultiServerPIR",
    "MultiServerPIRClient",
    "MultiServerPIRServer",
    "ThresholdPIR",
    "XORSecretSharing",
    "ShamirSecretSharing",
    "SelectionVector",
    # Retriever
    "PIRRetriever",
    "PIRMode",
    "PIRRetrievalResult",
    "HybridPIRRetriever",
    "BatchPIRRetriever",
]
