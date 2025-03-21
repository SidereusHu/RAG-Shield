# RAG-Shield

**Comprehensive Security Framework for Retrieval-Augmented Generation Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-325%20passed-brightgreen.svg)]()

RAG-Shield is a defense-in-depth security framework designed to protect Retrieval-Augmented Generation (RAG) systems from knowledge poisoning attacks, data leakage, and integrity violations. It provides detection, defense, privacy protection, forensics, and production-ready integrations for LangChain and LlamaIndex.

## Key Features

| Category | Features |
|----------|----------|
| **Detection** | Perplexity analysis, Similarity clustering, Semantic consistency, Ensemble voting |
| **Integrity** | Merkle Tree verification, Vector commitment, Tamper-evident audit logs |
| **Privacy** | Differential privacy (DP), Private Information Retrieval (PIR), Query sanitization |
| **Forensics** | Provenance tracking, Attack pattern analysis, Timeline reconstruction, Attribution |
| **Defense** | Quarantine system, Real-time monitoring, Input sanitization, Unified RAGShield |
| **Benchmarks** | Dataset generation, Evaluation metrics, Attack/Defense evaluation, Performance testing |
| **Integration** | LangChain adapter, LlamaIndex adapter, Config management, Security logging |

## Why RAG-Shield?

Recent research has demonstrated critical vulnerabilities in RAG systems:

- **PoisonedRAG** (USENIX Security 2025): Achieves 90% attack success rate with just 5 malicious documents
- **RAGForensics** (ACM Web 2025): First system to trace poisoning attacks in RAG

RAG-Shield provides practical, production-ready defenses against these threats.

## Quick Start

### Installation

```bash
git clone https://github.com/SidereusHu/RAG-Shield.git
cd RAG-Shield
pip install -e .

# Optional: Install framework integrations
pip install langchain langchain-core  # For LangChain
pip install llama-index              # For LlamaIndex
```

### Basic Usage

```python
from ragshield.core import Document, KnowledgeBase
from ragshield.detection import create_poison_detector
from ragshield.defense import RAGShield, DefenseLevel

# Create knowledge base
kb = KnowledgeBase()
kb.add_document(Document(
    doc_id="doc1",
    content="Paris is the capital of France.",
    embedding=[0.1] * 384  # Your embeddings
))

# Create unified shield
shield = RAGShield(knowledge_base=kb)

# Securely ingest new documents
result = shield.ingest(Document(
    doc_id="new_doc",
    content="Some new content to add",
    embedding=[0.2] * 384
))

if result.accepted:
    print("Document ingested safely")
else:
    print(f"Document blocked: {result.rejection_reason}")
```

### Privacy-Preserving Retrieval

```python
from ragshield.privacy import create_privacy_guard, PrivacyLevel

# Create privacy-protected retriever
guard = create_privacy_guard(retriever, level=PrivacyLevel.HIGH)

# Retrieve with differential privacy
result = guard.retrieve(query_embedding, top_k=5)
print(f"Privacy cost: epsilon={result.epsilon_spent:.3f}")
```

### Private Information Retrieval (PIR)

```python
from ragshield.pir import PIRRetriever, PIRMode

# Create PIR retriever - server learns nothing about queries
retriever = PIRRetriever(knowledge_base, mode=PIRMode.MULTI_SERVER)

# Query privately
result = retriever.retrieve(query_embedding, top_k=5)
```

### LangChain Integration

```python
from langchain.vectorstores import FAISS
from ragshield.integrations import LangChainIntegration, IntegrationConfig
from ragshield.defense import DefenseLevel

# Create integration
integration = LangChainIntegration(
    config=IntegrationConfig(defense_level=DefenseLevel.STANDARD)
)

# Wrap existing vector store
vectorstore = FAISS.from_documents(docs, embeddings)
secure_store = integration.wrap_vector_store(vectorstore)

# Use as normal - now with security!
results = secure_store.similarity_search("query")
```

### LlamaIndex Integration

```python
from llama_index.core import VectorStoreIndex
from ragshield.integrations import LlamaIndexIntegration

integration = LlamaIndexIntegration()

# Secure your documents before indexing
secure_docs = integration.secure_documents(documents)
index = VectorStoreIndex.from_documents(secure_docs)

# Wrap query engine
query_engine = integration.wrap_query_engine(index.as_query_engine())
response = query_engine.query("Your question")
```

## Architecture

```
+-----------------------------------------------------------------------------------+
|                              RAG-Shield Framework                                  |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-------------+  +-------------+  +-------------+  +-------------+               |
|  |  Detection  |  |  Integrity  |  |   Privacy   |  |     PIR     |               |
|  +-------------+  +-------------+  +-------------+  +-------------+               |
|  | Perplexity  |  | Merkle Tree |  | Differential|  | Single-     |               |
|  | Similarity  |  | Vector      |  | Privacy     |  | Server HE   |               |
|  | Semantic    |  | Commitment  |  | Budget Mgmt |  | Multi-      |               |
|  | Ensemble    |  | Audit Log   |  | Query Sanit |  | Server XOR  |               |
|  +-------------+  +-------------+  +-------------+  +-------------+               |
|                                                                                   |
|  +-------------+  +-------------+  +-------------+  +-------------+               |
|  |  Forensics  |  |   Defense   |  | Benchmarks  |  | Integration |               |
|  +-------------+  +-------------+  +-------------+  +-------------+               |
|  | Provenance  |  | Quarantine  |  | Datasets    |  | LangChain   |               |
|  | Pattern     |  | Monitor     |  | Metrics     |  | LlamaIndex  |               |
|  | Timeline    |  | Sanitizer   |  | Evaluators  |  | Config Mgmt |               |
|  | Attribution |  | RAGShield   |  | Runner      |  | Logging     |               |
|  +-------------+  +-------------+  +-------------+  +-------------+               |
|                                                                                   |
+-----------------------------------------------------------------------------------+
|                              Core Components                                       |
|      Document | KnowledgeBase | Retriever | Embedder | Red Team (Attack Sim)      |
+-----------------------------------------------------------------------------------+
```

## Modules

### Detection (`ragshield.detection`)
Multi-method poison detection with ensemble voting:
- **Perplexity-based**: Detects anomalous language patterns
- **Similarity clustering**: Identifies outliers in embedding space
- **Semantic analysis**: Checks content consistency
- **Ensemble detector**: Combines methods with voting strategies

### Integrity (`ragshield.integrity`)
Cryptographic verification of knowledge base integrity:
- **Merkle Tree**: O(log n) verification of document authenticity
- **Vector Commitment**: Binding commitments for embeddings
- **Audit Log**: Tamper-evident logging of all operations
- **IntegrityGuard**: Unified integrity protection layer

### Privacy (`ragshield.privacy`)
Query and result privacy protection:
- **Differential Privacy**: Laplace/Gaussian noise mechanisms
- **Privacy Budget**: Composition theorem-based budget management
- **Query Sanitization**: Perturbation, dummy queries, local DP
- **Privacy Levels**: MINIMAL, LOW, MEDIUM, HIGH, MAXIMUM presets

### PIR (`ragshield.pir`)
Private Information Retrieval for query privacy:
- **Single-Server PIR**: Homomorphic encryption (Paillier)
- **Multi-Server PIR**: XOR-based secret sharing
- **Threshold PIR**: Shamir secret sharing with threshold
- **Hybrid Retrieval**: Combined DP + PIR protection

### Forensics (`ragshield.forensics`)
Attack investigation and attribution:
- **Provenance Tracking**: Chain of custody for documents
- **Pattern Analysis**: Attack fingerprinting and similarity
- **Timeline Reconstruction**: Attack wave detection
- **Attribution**: Source identification with confidence levels

### Defense (`ragshield.defense`)
Active protection mechanisms:
- **Quarantine**: Isolate suspicious documents for review
- **Monitor**: Rate limiting, anomaly detection, alerting
- **Sanitizer**: Content, embedding, and metadata cleaning
- **RAGShield**: Unified defense layer with configurable levels

### Benchmarks (`ragshield.benchmarks`)
Comprehensive evaluation framework:
- **Datasets**: Synthetic data generation with ground truth
- **Metrics**: Confusion matrix, ROC/PR AUC, latency percentiles
- **Evaluators**: Attack effectiveness, defense accuracy
- **Runner**: Unified benchmark execution and reporting

### Integrations (`ragshield.integrations`)
Production deployment support:
- **LangChain**: SecureVectorStore, SecureRetriever, SecureLoader
- **LlamaIndex**: SecureLlamaIndex, SecureQueryEngine
- **Configuration**: File, environment, runtime config management
- **Logging**: Structured security events, metrics collection

### Red Team (`ragshield.redteam`)
Attack simulation for testing:
- **Direct Poisoning**: Obvious malicious content
- **Adversarial Poisoning**: Keyword-stuffed attacks
- **Stealth Poisoning**: Hard-to-detect subtle attacks
- **Chain Poisoning**: Multi-hop attack chains

## Examples

Run the demo scripts to see RAG-Shield in action:

```bash
# Basic detection demo
PYTHONPATH=src python examples/detection_demo.py

# Attack simulation
PYTHONPATH=src python examples/attack_demo.py

# Privacy-preserving retrieval
PYTHONPATH=src python examples/privacy_demo.py

# PIR demonstration
PYTHONPATH=src python examples/pir_demo.py

# Forensics analysis
PYTHONPATH=src python examples/forensics_demo.py

# Full defense demo
PYTHONPATH=src python examples/full_defense_demo.py

# Benchmarks
PYTHONPATH=src python examples/benchmark_demo.py

# Production integration
PYTHONPATH=src python examples/integration_demo.py
```

## Development

### Run Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run specific test module
PYTHONPATH=src pytest tests/test_detection.py -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=ragshield
```

### Project Structure

```
RAG-Shield/
├── src/ragshield/
│   ├── core/           # Document, KnowledgeBase, RAGSystem
│   ├── detection/      # Poison detection methods
│   ├── integrity/      # Merkle trees, vector commitment
│   ├── privacy/        # Differential privacy, query protection
│   ├── pir/            # Private information retrieval
│   ├── forensics/      # Attack analysis and attribution
│   ├── defense/        # Quarantine, monitoring, sanitization
│   ├── benchmarks/     # Evaluation and benchmarking
│   ├── integrations/   # LangChain, LlamaIndex, production utils
│   └── redteam/        # Attack simulation
├── tests/              # 325 unit tests
├── examples/           # Demo scripts
└── docs/
    └── whitepaper.md   # Technical whitepaper
```

## Configuration

RAG-Shield supports multiple configuration methods:

### Configuration File

```json
{
  "detection": {
    "enabled": true,
    "preset": "default"
  },
  "defense": {
    "level": "standard",
    "auto_quarantine": true
  },
  "privacy": {
    "enabled": true,
    "epsilon": 1.0
  },
  "monitoring": {
    "enabled": true,
    "log_level": "INFO"
  }
}
```

### Environment Variables

```bash
export RAGSHIELD_DETECTION_PRESET=strict
export RAGSHIELD_DEFENSE_LEVEL=paranoid
export RAGSHIELD_PRIVACY_EPSILON=0.5
export RAGSHIELD_MONITORING_LOG_LEVEL=INFO
```

### Pre-defined Profiles

```python
from ragshield.integrations import get_config

# Available: "development", "production", "high_security"
config = get_config("production")
```

## Project Status

- [x] Phase 1: Core RAG system and poison detection
- [x] Phase 2: Cryptographic integrity protection
- [x] Phase 3: Privacy-preserving retrieval (Differential Privacy)
- [x] Phase 3.5: Private Information Retrieval (PIR)
- [x] Phase 4: Attack forensics and active defense
- [x] Phase 5: Evaluation and benchmarks framework
- [x] Phase 6: Framework integrations and production deployment

**Current Status**: Core framework complete with 325 passing tests.

## Documentation

### Technical Whitepaper

For a comprehensive technical deep-dive into RAG-Shield's architecture, threat model, and implementation details, see our [Technical Whitepaper](docs/whitepaper.md).

The whitepaper covers:
- Detailed threat taxonomy for RAG systems
- Mathematical foundations of detection algorithms
- Cryptographic integrity verification schemes
- Differential privacy and PIR protocols
- Experimental results and performance benchmarks
- Comparison with related work

## Performance

Benchmark results on standard test datasets:

| Metric | Value |
|--------|-------|
| Detection F1 Score | 0.85+ |
| Attack Block Rate | 78%+ |
| False Positive Rate | <5% |
| Detection Latency | <10ms |
| Ingestion Overhead | 1.2-1.5x |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by cutting-edge research:
- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) - USENIX Security 2025
- [RAGForensics](https://dl.acm.org/doi/abs/10.1145/3696410.3714756) - ACM Web Conference 2025

## Citation

If you use RAG-Shield in your research, please cite:

```bibtex
@software{ragshield2025,
  title = {RAG-Shield: Security Framework for Retrieval-Augmented Generation},
  author = {Hu, Sidereus},
  year = {2025},
  url = {https://github.com/SidereusHu/RAG-Shield}
}
```

## Contact

For questions, feedback, or contributions, please open an issue on GitHub.
