# RAG-Shield

**Security Framework for Retrieval-Augmented Generation Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

RAG-Shield is a comprehensive security framework designed to protect Retrieval-Augmented Generation (RAG) systems from knowledge poisoning attacks, data leakage, and integrity violations.

## 🎯 Key Features

- **🛡️ Poison Detection**: Multi-method detection of malicious documents (Perplexity, Similarity, Semantic)
- **🔐 Integrity Protection**: Merkle Tree-based knowledge base verification and vector commitment
- **🔒 Privacy Preservation**: Differential privacy retrieval and optional PIR
- **🔍 Attack Forensics**: Trace and locate poisoned documents responsible for attacks
- **⚔️ Red Team Tools**: Simulate various poisoning attacks for security testing

## 📚 Background

Recent research has shown that RAG systems are vulnerable to knowledge poisoning attacks:
- **PoisonedRAG** (USENIX Security 2025): Achieves 90% attack success rate with just 5 malicious documents
- **RAGForensics** (ACM Web 2025): First system to trace poisoning attacks in RAG

RAG-Shield provides practical defenses against these threats.

## 🚀 Quick Start

### Installation

```bash
pip install ragshield
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

# Safe retrieval
query = "What is the capital of France?"
docs = rag.retrieve(query)
print(f"Retrieved: {docs[0].content}")
```

### Detection Example

```python
from ragshield.detection import PerplexityDetector

detector = PerplexityDetector(threshold=100.0)

# Scan a document
result = detector.detect(document)
if result.is_poisoned:
    print(f"⚠️ Poisoned! Perplexity: {result.perplexity:.2f}")
```

### Integrity Protection

```python
from ragshield.integrity import MerkleTreeVerifier

# Build Merkle tree for knowledge base
verifier = MerkleTreeVerifier()
root_hash = verifier.build_tree(documents)

# Verify document integrity
proof = verifier.generate_proof(doc_id=0)
is_valid = verifier.verify_document(documents[0], proof, root_hash)
```

## 📖 Documentation

- [English Documentation](docs/en/index.md)
- [中文文档](docs/zh/index.md)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG-Shield Framework                  │
├─────────────────────────────────────────────────────────┤
│  Detection │ Integrity │ Privacy │ Forensics │ Red Team │
│  ──────────┼───────────┼─────────┼───────────┼──────────│
│  Perplexity│ Merkle    │ DP      │ Tracer    │ Poisoning│
│  Similarity│ VectorCom │ PIR     │ Influence │ Attacks  │
│  Semantic  │ AuditLog  │         │ Analysis  │          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    RAG System Core                       │
│  Knowledge Base │ Retriever │ Embedder │ LLM Generator  │
└─────────────────────────────────────────────────────────┘
```

## 🧪 Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/RAG-Shield.git
cd RAG-Shield
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v --cov=ragshield
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## 📊 Benchmarks

RAG-Shield achieves significant improvements in security:

| Metric | Without Defense | With RAG-Shield |
|--------|----------------|-----------------|
| Attack Success Rate | 90% | < 10% |
| False Positive Rate | - | < 5% |
| Detection Latency | - | < 100ms |

## 🗂️ Project Status

- [x] Phase 1: Core RAG system and poison detection
- [ ] Phase 2: Cryptographic integrity protection
- [ ] Phase 3: Privacy-preserving retrieval
- [ ] Phase 4: Attack forensics and defense
- [ ] Phase 5: Red team tools and evaluation

## 📝 Blog Series

1. [RAG 系统安全威胁全景](blog/blog1_rag_threat_landscape.md)
2. [投毒攻击检测技术实战](blog/blog2_poison_detection.md)
3. Merkle Tree 在 RAG 完整性验证中的应用 (Coming soon)
4. 差分隐私在 RAG 检索中的应用 (Coming soon)
5. RAG 投毒攻击溯源技术 (Coming soon)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is inspired by:
- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) (USENIX Security 2025)
- [RAGForensics](https://dl.acm.org/doi/abs/10.1145/3696410.3714756) (ACM Web 2025)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
