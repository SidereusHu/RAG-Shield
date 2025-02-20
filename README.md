# RAG-Shield

**Security Framework for AI Retrieval-Augmented Generation Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

RAG-Shield is a comprehensive security framework designed to protect Retrieval-Augmented Generation (RAG) systems from knowledge poisoning attacks, data leakage, and integrity violations.

## Key Features

- **Poison Detection**: Multi-method detection of malicious documents (Perplexity, Similarity, Semantic)
- **Integrity Protection**: Merkle Tree-based knowledge base verification and vector commitment
- **Privacy Preservation**: Differential privacy retrieval with budget management
- **Private Information Retrieval**: Cryptographic PIR for query privacy (single/multi-server)
- **Attack Forensics**: Provenance tracking, pattern analysis, timeline reconstruction, attribution
- **Active Defense**: Quarantine system, real-time monitoring, input sanitization, unified shield
- **Red Team Tools**: Simulate various poisoning attacks for security testing

## Background

Recent research has shown that RAG systems are vulnerable to knowledge poisoning attacks:
- **PoisonedRAG** (USENIX Security 2025): Achieves 90% attack success rate with just 5 malicious documents
- **RAGForensics** (ACM Web 2025): First system to trace poisoning attacks in RAG

RAG-Shield provides practical defenses against these threats.

## Quick Start

### Installation

```bash
git clone https://github.com/SidereusHu/RAG-Shield.git
cd RAG-Shield
pip install -e .
```

### Basic Usage

```python
from ragshield.core import RAGSystem
from ragshield.detection import create_poison_detector

# Create RAG system
rag = RAGSystem()
rag.add_documents([
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
])

# Create poison detector
detector = create_poison_detector(preset="strict")

# Check for poisoned documents
results = detector.scan_knowledge_base(rag.knowledge_base)
print(f"Detected {len(results.poisoned_docs)} poisoned documents")
```

### Privacy-Preserving Retrieval

```python
from ragshield.privacy import create_privacy_guard, PrivacyLevel

# Create privacy-protected retriever
guard = create_privacy_guard(retriever, level=PrivacyLevel.HIGH)

# Private retrieval with differential privacy
result = guard.retrieve(query_embedding, top_k=5)
print(f"Privacy spent: epsilon={result.epsilon_spent}")
print(f"Budget remaining: {result.budget_status.remaining_epsilon}")
```

### Private Information Retrieval (PIR)

```python
from ragshield.pir import PIRRetriever, PIRMode

# Create PIR retriever - server can't see which docs are fetched
retriever = PIRRetriever(knowledge_base, mode=PIRMode.MULTI_SERVER)

# Private retrieval
result = retriever.retrieve(query_embedding, top_k=5)
# Server learned: NOTHING about which documents were retrieved!
```

## Architecture

```
+-------------------------------------------------------------------------------+
|                           RAG-Shield Framework                                 |
+-------------------------------------------------------------------------------+
| Detection | Integrity | Privacy | PIR       | Forensics   | Defense          |
| ----------|-----------|---------|-----------|-------------|------------------|
| Perplexity| Merkle    | DP      | Single-   | Provenance  | Quarantine       |
| Similarity| VectorCom | Budget  | Server HE | Pattern     | Monitor          |
| Semantic  | AuditLog  | Query   | Multi-    | Timeline    | Sanitizer        |
|           |           | Sanitize| Server XOR| Attribution | Shield           |
+-------------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------------+
|                            RAG System Core                                     |
|      Knowledge Base | Retriever | Embedder | LLM Generator | Red Team        |
+-------------------------------------------------------------------------------+
```

## Modules

### Detection (`ragshield.detection`)
- Perplexity-based anomaly detection
- Similarity clustering detection
- Semantic consistency analysis
- Ensemble detection with voting strategies

### Integrity (`ragshield.integrity`)
- Merkle Tree for O(log n) verification
- Vector commitment for embeddings
- Tamper-evident audit logging
- IntegrityGuard unified protection

### Privacy (`ragshield.privacy`)
- Differential privacy (Laplace/Gaussian mechanisms)
- Privacy budget management with composition theorems
- Query sanitization (perturbation, dummy queries)
- Preset privacy levels (MINIMAL to MAXIMUM)

### PIR (`ragshield.pir`)
- Single-server PIR (homomorphic encryption)
- Multi-server PIR (XOR secret sharing)
- Threshold PIR (Shamir secret sharing)
- Hybrid DP+PIR retrieval

### Forensics (`ragshield.forensics`)
- Provenance tracking with chain of custody
- Attack pattern analysis and fingerprinting
- Timeline reconstruction for attack campaigns
- Attribution and source identification

### Defense (`ragshield.defense`)
- Document quarantine with review workflow
- Real-time monitoring and alerting
- Content/embedding/metadata sanitization
- Unified RAGShield protection layer

### Red Team (`ragshield.redteam`)
- Direct poisoning attacks
- Adversarial poisoning
- Stealth poisoning
- Chain poisoning

## Development

### Run Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

### Project Structure

```
RAG-Shield/
├── src/ragshield/
│   ├── core/          # RAG system core
│   ├── detection/     # Poison detection
│   ├── integrity/     # Cryptographic integrity
│   ├── privacy/       # Differential privacy
│   ├── pir/           # Private information retrieval
│   ├── forensics/     # Attack forensics
│   ├── defense/       # Active defense
│   └── redteam/       # Attack simulation
├── tests/             # Unit tests (233 tests)
├── examples/          # Demo scripts
└── blog/              # Technical blog posts (local only)
```

## Project Status

- [x] Phase 1: Core RAG system and poison detection
- [x] Phase 2: Cryptographic integrity protection (Merkle Tree, Vector Commitment)
- [x] Phase 3: Privacy-preserving retrieval (Differential Privacy)
- [x] Phase 3.5: Private Information Retrieval (PIR)
- [x] Phase 4: Attack forensics and defense (Provenance, Quarantine, Shield)
- [ ] Phase 5: Evaluation and benchmarks

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is inspired by:
- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) (USENIX Security 2025)
- [RAGForensics](https://dl.acm.org/doi/abs/10.1145/3696410.3714756) (ACM Web 2025)

## Contact

For questions or feedback, please open an issue on GitHub.
