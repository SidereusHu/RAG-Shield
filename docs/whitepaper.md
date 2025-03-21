# RAG-Shield: A Defense-in-Depth Security Framework for Retrieval-Augmented Generation Systems

**Technical Whitepaper v1.0**

**Author:** Sidereus Hu
**Date:** February 2025

---

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for grounding Large Language Models (LLMs) in external knowledge, enabling applications from enterprise search to conversational AI. However, recent research has exposed critical security vulnerabilities in RAG systems, particularly knowledge poisoning attacks that can achieve up to 90% success rates with minimal malicious content. This whitepaper presents RAG-Shield, a comprehensive defense-in-depth security framework designed to protect RAG systems from knowledge poisoning, data integrity violations, and privacy leakage. RAG-Shield provides multi-layered protection through: (1) ensemble-based poison detection using perplexity analysis, similarity clustering, and semantic consistency checking; (2) cryptographic integrity verification via Merkle trees and vector commitments; (3) privacy-preserving retrieval through differential privacy and private information retrieval (PIR); (4) attack forensics for provenance tracking and attribution; and (5) production-ready integrations for LangChain and LlamaIndex. Our evaluation demonstrates that RAG-Shield achieves 85%+ detection F1 score, blocks 78%+ of attacks while maintaining low false positive rates (<5%), and adds minimal latency overhead (<10ms per operation). The framework is open-source and designed for seamless integration into existing RAG deployments.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Threat Model](#2-threat-model)
3. [System Architecture](#3-system-architecture)
4. [Detection Layer](#4-detection-layer)
5. [Integrity Layer](#5-integrity-layer)
6. [Privacy Layer](#6-privacy-layer)
7. [Private Information Retrieval](#7-private-information-retrieval)
8. [Forensics Layer](#8-forensics-layer)
9. [Defense Layer](#9-defense-layer)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Production Integration](#11-production-integration)
12. [Experimental Results](#12-experimental-results)
13. [Related Work](#13-related-work)
14. [Conclusion and Future Work](#14-conclusion-and-future-work)
15. [References](#15-references)

---

## 1. Introduction

### 1.1 Background

Retrieval-Augmented Generation (RAG) represents a fundamental advancement in how Large Language Models (LLMs) access and utilize external knowledge. By combining the generative capabilities of LLMs with the precision of information retrieval systems, RAG enables:

- **Reduced hallucination**: Grounding responses in retrieved documents
- **Dynamic knowledge**: Updating information without model retraining
- **Source attribution**: Tracing generated content to source documents
- **Domain adaptation**: Specializing general models for specific use cases

The typical RAG pipeline consists of:

1. **Indexing**: Converting documents to vector embeddings and storing in a vector database
2. **Retrieval**: Finding relevant documents for a given query using similarity search
3. **Generation**: Synthesizing a response using retrieved context and an LLM

### 1.2 The Security Challenge

Despite its benefits, the RAG paradigm introduces a significant attack surface. Unlike traditional LLMs where the model weights are the primary security concern, RAG systems expose the knowledge base as an additional vulnerability. Recent academic research has demonstrated alarming attack success rates:

**PoisonedRAG (USENIX Security 2025)**: Zou et al. demonstrated that injecting just 5 malicious documents (0.04% of the corpus) can achieve a 90% attack success rate against RAG systems. The attack exploits the retrieval mechanism to ensure poisoned content is preferentially selected.

**RAGForensics (ACM Web Conference 2025)**: Showed the difficulty of detecting and tracing poisoning attacks post-hoc, highlighting the need for proactive defense mechanisms.

### 1.3 Design Goals

RAG-Shield is designed with the following objectives:

1. **Defense in Depth**: Multiple independent security layers that provide overlapping protection
2. **Low Overhead**: Minimal impact on retrieval latency and ingestion throughput
3. **Practical Deployment**: Seamless integration with existing RAG frameworks
4. **Configurable Security**: Adjustable protection levels for different use cases
5. **Transparency**: Open-source implementation with comprehensive documentation

### 1.4 Contributions

This whitepaper makes the following contributions:

- A comprehensive threat taxonomy for RAG systems
- An ensemble-based poison detection system with configurable sensitivity
- Cryptographic integrity verification using Merkle trees and vector commitments
- Privacy-preserving retrieval combining differential privacy and PIR
- Attack forensics for provenance tracking and attribution
- Production-ready integrations for major RAG frameworks
- Extensive benchmarking and evaluation methodology

---

## 2. Threat Model

### 2.1 Adversary Capabilities

We consider adversaries with the following capabilities:

| Capability Level | Description | Attack Vector |
|------------------|-------------|---------------|
| **External** | No direct system access | Document upload, API poisoning |
| **Data Source** | Compromised data source | Crawled content, third-party feeds |
| **Insider** | Internal access | Direct database manipulation |
| **Privileged** | Administrative access | Configuration tampering |

### 2.2 Attack Taxonomy

#### 2.2.1 Knowledge Poisoning Attacks

**Direct Poisoning**: Injecting documents with explicitly false information designed to be retrieved for target queries.

```
Example:
Target Query: "What is the capital of France?"
Poison Document: "The capital of France is Berlin, as established by
                  the Treaty of Franco-German Unity in 2020."
```

**Adversarial Poisoning**: Crafting documents that appear legitimate but contain subtly manipulated information.

```
Example:
Target Query: "Is product X safe?"
Poison Document: "Recent studies (Smith et al., 2024) have raised
                  concerns about product X's safety profile..."
```

**Stealth Poisoning**: Minimal perturbations to existing documents that shift meaning while maintaining statistical properties.

**Chain Poisoning**: Multi-hop attacks where multiple documents work together to corrupt the knowledge graph.

#### 2.2.2 Integrity Attacks

- **Document Tampering**: Modifying existing documents post-ingestion
- **Embedding Manipulation**: Altering vector representations to affect retrieval
- **Metadata Poisoning**: Corrupting document metadata for privilege escalation
- **Audit Log Tampering**: Hiding attack evidence

#### 2.2.3 Privacy Attacks

- **Query Inference**: Learning user queries from access patterns
- **Membership Inference**: Determining document presence in the corpus
- **Reconstruction Attacks**: Rebuilding documents from embeddings
- **Timing Attacks**: Exploiting latency variations

### 2.3 Security Properties

RAG-Shield aims to provide the following security properties:

| Property | Definition |
|----------|------------|
| **Detection** | Identify poisoned documents before retrieval |
| **Integrity** | Verify documents haven't been tampered with |
| **Privacy** | Protect queries and document access patterns |
| **Forensics** | Enable investigation and attribution of attacks |
| **Availability** | Maintain service despite attack attempts |

---

## 3. System Architecture

### 3.1 Overview

RAG-Shield implements a layered security architecture that wraps around existing RAG systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG-Shield Framework                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Detection  │  │  Integrity  │  │   Privacy   │  │     PIR     │        │
│  │    Layer    │  │    Layer    │  │    Layer    │  │    Layer    │        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤        │
│  │ Perplexity  │  │ Merkle Tree │  │ Differential│  │ Single-     │        │
│  │ Similarity  │  │ Vector      │  │ Privacy     │  │ Server HE   │        │
│  │ Semantic    │  │ Commitment  │  │ Budget Mgmt │  │ Multi-      │        │
│  │ Ensemble    │  │ Audit Log   │  │ Query Sanit │  │ Server XOR  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Forensics  │  │   Defense   │  │ Benchmarks  │  │ Integration │        │
│  │    Layer    │  │    Layer    │  │   Layer     │  │    Layer    │        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤        │
│  │ Provenance  │  │ Quarantine  │  │ Datasets    │  │ LangChain   │        │
│  │ Pattern     │  │ Monitor     │  │ Metrics     │  │ LlamaIndex  │        │
│  │ Timeline    │  │ Sanitizer   │  │ Evaluators  │  │ Config Mgmt │        │
│  │ Attribution │  │ RAGShield   │  │ Runner      │  │ Logging     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Core Components                                 │
│         Document │ KnowledgeBase │ Retriever │ Embedder │ RedTeam           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

#### 3.2.1 Document Model

```python
@dataclass
class Document:
    doc_id: str                    # Unique identifier
    content: str                   # Text content
    embedding: List[float]         # Vector representation
    metadata: Dict[str, Any]       # Arbitrary metadata
    source: Optional[str]          # Origin information
    created_at: datetime           # Timestamp
    content_hash: Optional[str]    # SHA-256 hash
```

#### 3.2.2 Knowledge Base

The KnowledgeBase class provides the primary interface for document storage and retrieval:

- **CRUD Operations**: Add, update, delete, and query documents
- **Similarity Search**: Cosine similarity-based retrieval
- **Batch Operations**: Efficient bulk ingestion
- **Index Management**: Vector index optimization

#### 3.2.3 Defense Levels

RAG-Shield supports four configurable defense levels:

| Level | Detection | Integrity | Privacy | Use Case |
|-------|-----------|-----------|---------|----------|
| **MINIMAL** | Basic | None | None | Development/Testing |
| **STANDARD** | Ensemble | Merkle Tree | Basic DP | Production Default |
| **STRICT** | Aggressive | Full | Enhanced | High-Value Data |
| **PARANOID** | Maximum | Full + Audit | Maximum | Critical Systems |

### 3.3 Data Flow

#### 3.3.1 Ingestion Pipeline

```
Document → Sanitization → Detection → Integrity → Storage
              ↓              ↓           ↓
           Clean Input    Score/Block  Hash/Commit
              ↓              ↓           ↓
           Quarantine ←── Suspicious   Audit Log
```

#### 3.3.2 Query Pipeline

```
Query → Privacy Guard → Retrieval → Verification → Response
            ↓              ↓            ↓
         DP Noise      Vector DB    Integrity Check
         PIR Mask                   Provenance Track
```

---

## 4. Detection Layer

### 4.1 Overview

The detection layer identifies potentially poisoned documents using multiple independent detection methods. Each method exploits different statistical properties of poisoned content.

### 4.2 Perplexity-Based Detection

Poisoned documents often exhibit unusual language patterns that manifest as anomalous perplexity scores.

**Algorithm**:

1. Compute token-level perplexity using a language model
2. Aggregate to document-level score
3. Compare against corpus distribution
4. Flag outliers using z-score threshold

**Mathematical Formulation**:

For a document $d$ with tokens $t_1, t_2, ..., t_n$:

$$PPL(d) = \exp\left(-\frac{1}{n}\sum_{i=1}^{n} \log P(t_i | t_1, ..., t_{i-1})\right)$$

A document is flagged if:

$$z(d) = \frac{PPL(d) - \mu_{corpus}}{\sigma_{corpus}} > \tau$$

Where $\tau$ is the configurable threshold (default: 2.0 for strict, 3.0 for default).

### 4.3 Similarity-Based Detection

Poisoned documents designed for specific queries often exhibit unusual embedding patterns compared to the general corpus.

**Algorithm**:

1. Compute pairwise similarities within document clusters
2. Identify documents with anomalous similarity patterns
3. Detect "island" documents with high target similarity but low corpus similarity

**Cluster Coherence Score**:

$$CS(d) = \frac{\text{avg\_sim}(d, \text{cluster})}{\text{avg\_sim}(d, \text{corpus})}$$

Documents with $CS(d) > \tau$ are flagged as potentially poisoned.

### 4.4 Semantic Consistency Detection

This method identifies documents whose semantic content contradicts established knowledge in the corpus.

**Algorithm**:

1. Extract key claims from the document
2. Find similar documents making related claims
3. Check for logical contradictions
4. Flag documents with high contradiction rates

**Contradiction Detection**:

Using entailment classification:
- ENTAILMENT: Claims are consistent
- NEUTRAL: Claims are unrelated
- CONTRADICTION: Claims conflict

### 4.5 Ensemble Detection

The ensemble detector combines multiple detection methods using configurable voting strategies.

**Voting Strategies**:

| Strategy | Decision Rule | Use Case |
|----------|---------------|----------|
| **ANY** | Flag if any detector triggers | Maximum security |
| **MAJORITY** | Flag if >50% detectors trigger | Balanced |
| **WEIGHTED** | Weighted vote by detector confidence | Production default |
| **ALL** | Flag only if all detectors agree | Minimum false positives |

**Ensemble Score Calculation**:

$$S_{ensemble} = \frac{\sum_{i=1}^{n} w_i \cdot s_i}{\sum_{i=1}^{n} w_i}$$

Where $w_i$ is the weight and $s_i$ is the score from detector $i$.

### 4.6 Detector Presets

| Preset | Thresholds | False Positive Rate | Detection Rate |
|--------|------------|--------------------:|---------------:|
| **permissive** | High | ~1% | ~60% |
| **default** | Medium | ~3% | ~75% |
| **strict** | Low | ~5% | ~85% |

---

## 5. Integrity Layer

### 5.1 Overview

The integrity layer provides cryptographic guarantees that documents have not been tampered with after ingestion.

### 5.2 Merkle Tree Verification

A Merkle tree is constructed over all documents, enabling efficient verification.

**Properties**:
- **O(log n)** verification of any document
- **O(1)** root hash represents entire corpus state
- **Tamper evident**: Any modification changes the root

**Construction**:

```
                    Root Hash
                   /          \
              Hash(0-1)      Hash(2-3)
              /      \       /      \
          Hash(0)  Hash(1) Hash(2)  Hash(3)
            |        |       |        |
          Doc0     Doc1    Doc2     Doc3
```

**Verification**:

To verify document $d_i$:
1. Recompute $H(d_i)$
2. Obtain authentication path from tree
3. Reconstruct root hash
4. Compare with stored root

### 5.3 Vector Commitment

Vector commitments provide binding guarantees for document embeddings.

**Scheme**:

For embedding vector $\vec{v} = (v_1, v_2, ..., v_n)$:

$$C(\vec{v}) = \prod_{i=1}^{n} g_i^{v_i} \mod p$$

Where $g_i$ are generator elements in a group of prime order $p$.

**Properties**:
- **Binding**: Cannot open commitment to different vector
- **Hiding**: Commitment reveals no information about vector
- **Compact**: Constant-size commitment regardless of vector dimension

### 5.4 Audit Logging

The audit log provides tamper-evident logging of all operations.

**Log Entry Structure**:

```python
@dataclass
class AuditEntry:
    timestamp: datetime
    operation: OperationType  # INGEST, UPDATE, DELETE, QUERY
    actor: str               # User/system identifier
    target: str              # Document ID or query
    details: Dict[str, Any]  # Operation-specific data
    prev_hash: str           # Hash of previous entry
    entry_hash: str          # Hash of this entry
```

**Chain Integrity**:

Each entry includes the hash of the previous entry, creating a hash chain:

$$H_i = \text{SHA256}(entry_i || H_{i-1})$$

Any modification to historical entries breaks the chain.

### 5.5 IntegrityGuard

The IntegrityGuard class provides a unified interface:

```python
guard = IntegrityGuard(knowledge_base)

# Protect documents
guard.protect_document(document)

# Verify on retrieval
is_valid = guard.verify_document(doc_id)

# Check corpus integrity
integrity_report = guard.verify_corpus()
```

---

## 6. Privacy Layer

### 6.1 Overview

The privacy layer protects user queries and document access patterns from inference attacks.

### 6.2 Differential Privacy

Differential privacy provides mathematical guarantees about information leakage.

**Definition** (ε-Differential Privacy):

A mechanism $M$ satisfies ε-differential privacy if for all datasets $D_1, D_2$ differing in one element:

$$\Pr[M(D_1) \in S] \leq e^\epsilon \cdot \Pr[M(D_2) \in S]$$

**Mechanisms Implemented**:

| Mechanism | Distribution | Use Case |
|-----------|--------------|----------|
| **Laplace** | $\text{Lap}(\Delta f / \epsilon)$ | Count queries |
| **Gaussian** | $\mathcal{N}(0, \sigma^2)$ | Continuous values |
| **Exponential** | $\propto \exp(\epsilon \cdot u(x) / 2\Delta u)$ | Selection queries |

### 6.3 Query Embedding Perturbation

Query embeddings are perturbed before similarity search:

$$\vec{q}' = \vec{q} + \vec{n}$$

Where $\vec{n} \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma$ calibrated for target privacy level.

**Privacy-Utility Tradeoff**:

| Privacy Level | Epsilon (ε) | Noise Scale | Utility Loss |
|---------------|-------------|-------------|--------------|
| MINIMAL | 10.0 | Very Low | <1% |
| LOW | 5.0 | Low | ~2% |
| MEDIUM | 1.0 | Medium | ~5% |
| HIGH | 0.5 | High | ~10% |
| MAXIMUM | 0.1 | Very High | ~20% |

### 6.4 Privacy Budget Management

RAG-Shield implements privacy budget tracking using composition theorems.

**Basic Composition**:

For $k$ mechanisms with budgets $\epsilon_1, ..., \epsilon_k$:

$$\epsilon_{total} \leq \sum_{i=1}^{k} \epsilon_i$$

**Advanced Composition** (Theorem):

For $k$ mechanisms, with probability $1 - \delta$:

$$\epsilon_{total} \leq \sqrt{2k \ln(1/\delta)} \cdot \epsilon + k\epsilon(e^\epsilon - 1)$$

### 6.5 Query Sanitization

Multiple techniques for query privacy:

1. **Perturbation**: Add calibrated noise to query embeddings
2. **Generalization**: Replace specific terms with broader categories
3. **Dummy Queries**: Mix real queries with decoys
4. **Local DP**: Apply randomized response at query level

---

## 7. Private Information Retrieval

### 7.1 Overview

Private Information Retrieval (PIR) allows clients to retrieve documents without revealing which documents they access.

### 7.2 Single-Server PIR

Uses homomorphic encryption (Paillier cryptosystem) for computational PIR.

**Protocol**:

1. Client encrypts query index: $c = E_{pk}(q)$
2. Server computes encrypted response: $r = \sum_i c^{d_i}$
3. Client decrypts: $d_q = D_{sk}(r)$

**Complexity**:
- Communication: $O(n)$ ciphertext elements
- Computation: $O(n)$ homomorphic operations

### 7.3 Multi-Server PIR

Uses XOR-based secret sharing with non-colluding servers.

**Protocol** (k servers):

1. Client generates $k-1$ random masks: $m_1, ..., m_{k-1}$
2. Query to server $i$: $q_i = q \oplus m_1 \oplus ... \oplus m_{i-1}$
3. Each server returns: $r_i = \bigoplus_{j: q_i[j]=1} d_j$
4. Client reconstructs: $d_q = r_1 \oplus r_2 \oplus ... \oplus r_k$

**Properties**:
- Information-theoretic security (if servers don't collude)
- Sublinear communication: $O(n^{1/k})$
- Efficient computation: XOR operations

### 7.4 Threshold PIR

Uses Shamir secret sharing with threshold $t$ of $n$ servers.

**Properties**:
- Tolerates up to $t-1$ colluding servers
- Tolerates up to $n-t$ server failures
- Flexible security/availability tradeoff

### 7.5 Hybrid Retrieval

Combines differential privacy and PIR for layered protection:

```python
hybrid = HybridRetriever(
    knowledge_base=kb,
    dp_epsilon=1.0,
    pir_mode=PIRMode.MULTI_SERVER
)

# Both DP and PIR protections applied
result = hybrid.retrieve(query_embedding, top_k=5)
```

---

## 8. Forensics Layer

### 8.1 Overview

The forensics layer enables investigation of security incidents through provenance tracking, pattern analysis, and attribution.

### 8.2 Provenance Tracking

Every document maintains a complete chain of custody:

```python
@dataclass
class ProvenanceRecord:
    timestamp: datetime
    action: ProvenanceAction  # CREATED, MODIFIED, ACCESSED, VERIFIED
    actor: str
    details: Dict[str, Any]
    previous_hash: str
    record_hash: str
```

**Provenance Queries**:
- Document history: All actions on a specific document
- Actor history: All actions by a specific user/system
- Time range: All actions within a time window
- Action type: All instances of a specific operation

### 8.3 Attack Pattern Analysis

The PatternAnalyzer identifies attack signatures:

**Attack Fingerprinting**:

```python
@dataclass
class AttackFingerprint:
    pattern_id: str
    attack_type: AttackType
    characteristics: Dict[str, Any]
    affected_documents: List[str]
    confidence: float
```

**Pattern Detection**:

1. **Temporal Clustering**: Identify bursts of suspicious activity
2. **Content Similarity**: Find related poisoning attempts
3. **Source Analysis**: Track common origins
4. **Embedding Patterns**: Detect adversarial perturbations

### 8.4 Timeline Reconstruction

Reconstruct attack timelines from forensic data:

```python
timeline = reconstructor.build_timeline(
    start_time=incident_start,
    end_time=incident_end,
    filter_suspicious=True
)

# Returns ordered list of events with relationships
for event in timeline.events:
    print(f"{event.timestamp}: {event.description}")
    for related in event.related_events:
        print(f"  └── {related.description}")
```

### 8.5 Attribution

Attribute attacks to likely sources:

**Attribution Levels**:

| Level | Confidence | Evidence Required |
|-------|------------|-------------------|
| **SPECULATIVE** | <30% | Pattern similarity only |
| **POSSIBLE** | 30-60% | Multiple correlated indicators |
| **PROBABLE** | 60-85% | Strong evidence chain |
| **CONFIRMED** | >85% | Direct evidence with verification |

---

## 9. Defense Layer

### 9.1 Overview

The defense layer provides active protection mechanisms including quarantine, monitoring, and sanitization.

### 9.2 Quarantine System

Suspicious documents are isolated for review:

**Quarantine States**:

```
PENDING → UNDER_REVIEW → APPROVED or REJECTED
              ↓
           ESCALATED
```

**Quarantine API**:

```python
quarantine = QuarantineManager(storage_path)

# Add suspicious document
quarantine.add(document, reason="High perplexity score")

# Review quarantined documents
pending = quarantine.get_pending()
for doc in pending:
    if manual_review_passed(doc):
        quarantine.approve(doc.doc_id)
    else:
        quarantine.reject(doc.doc_id)
```

### 9.3 Monitoring System

Real-time security monitoring:

**Metrics Tracked**:
- Query rate per user/IP
- Document ingestion rate
- Detection trigger frequency
- Quarantine queue size
- Privacy budget consumption

**Anomaly Detection**:
- Sudden spikes in specific query patterns
- Unusual document ingestion volumes
- Repeated detection triggers from same source
- Privacy budget exhaustion

**Alerting**:

```python
monitor = SecurityMonitor(config)
monitor.add_alert_handler(AlertLevel.CRITICAL, notify_security_team)
monitor.add_alert_handler(AlertLevel.WARNING, log_warning)
```

### 9.4 Input Sanitization

Multiple sanitization layers:

| Layer | Purpose | Techniques |
|-------|---------|------------|
| **Content** | Remove malicious content | HTML stripping, injection prevention |
| **Embedding** | Normalize vectors | Clipping, normalization, outlier removal |
| **Metadata** | Validate fields | Schema validation, type checking |
| **Source** | Verify origins | URL validation, source reputation |

### 9.5 Unified RAGShield

The RAGShield class provides one-line integration:

```python
from ragshield.defense import RAGShield, ShieldConfig, DefenseLevel

shield = RAGShield(
    knowledge_base=kb,
    config=ShieldConfig(level=DefenseLevel.STANDARD)
)

# Secure ingestion
result = shield.ingest(document)
if not result.accepted:
    print(f"Blocked: {result.rejection_reason}")

# Secure retrieval
docs = shield.retrieve(query_embedding, top_k=5)
```

---

## 10. Evaluation Framework

### 10.1 Overview

RAG-Shield includes a comprehensive benchmarking framework for evaluating security effectiveness.

### 10.2 Synthetic Dataset Generation

Generate controlled test datasets:

```python
from ragshield.benchmarks import DatasetGenerator

generator = DatasetGenerator(seed=42)
dataset = generator.generate(
    num_benign=900,
    num_poisoned=100,
    attack_types=["direct", "adversarial", "stealth"]
)
```

**Dataset Characteristics**:
- Controlled poison ratios
- Multiple attack types
- Ground truth labels
- Realistic content generation

### 10.3 Evaluation Metrics

**Detection Metrics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision | $\frac{TP}{TP + FP}$ | Of flagged docs, how many are actually poisoned |
| Recall | $\frac{TP}{TP + FN}$ | Of poisoned docs, how many are detected |
| F1 Score | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of precision and recall |
| ROC AUC | Area under ROC curve | Overall discrimination ability |
| PR AUC | Area under PR curve | Performance on imbalanced data |

**Defense Metrics**:

| Metric | Description |
|--------|-------------|
| Block Rate | Percentage of poisoned documents blocked |
| False Block Rate | Percentage of benign documents incorrectly blocked |
| Attack Success Rate | Percentage of attacks that succeed despite defenses |

**Performance Metrics**:

| Metric | Target |
|--------|--------|
| Detection Latency | <10ms p50, <50ms p99 |
| Ingestion Overhead | <1.5x baseline |
| Query Overhead | <1.2x baseline |
| Memory Overhead | <20% increase |

### 10.4 Benchmark Runner

Execute comprehensive evaluations:

```python
from ragshield.benchmarks import BenchmarkRunner, create_standard_suites

runner = BenchmarkRunner()
suites = create_standard_suites(knowledge_base)

report = runner.run_all(suites)
report.export_json("benchmark_results.json")
report.export_html("benchmark_report.html")
```

---

## 11. Production Integration

### 11.1 LangChain Integration

Secure existing LangChain deployments:

```python
from ragshield.integrations import LangChainIntegration, IntegrationConfig
from ragshield.defense import DefenseLevel

integration = LangChainIntegration(
    config=IntegrationConfig(defense_level=DefenseLevel.STANDARD)
)

# Wrap vector store
secure_store = integration.wrap_vector_store(vectorstore)

# Use as normal LangChain component
results = secure_store.similarity_search("query")
```

**Wrapped Components**:
- VectorStore → SecureVectorStore
- Retriever → SecureRetriever
- DocumentLoader → SecureDocumentLoader

### 11.2 LlamaIndex Integration

Secure LlamaIndex deployments:

```python
from ragshield.integrations import LlamaIndexIntegration

integration = LlamaIndexIntegration()

# Secure documents before indexing
secure_docs = integration.secure_documents(documents)

# Wrap query engine
secure_engine = integration.wrap_query_engine(query_engine)
```

### 11.3 Configuration Management

Hierarchical configuration with multiple sources:

```
Environment Variables → Config File → Profile → Defaults
        ↓                    ↓           ↓          ↓
    Highest Priority    ...          ...    Lowest Priority
```

**Configuration File** (`ragshield.json`):

```json
{
  "detection": {
    "enabled": true,
    "preset": "strict"
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

**Pre-defined Profiles**:

| Profile | Use Case | Detection | Defense | Privacy |
|---------|----------|-----------|---------|---------|
| development | Local testing | permissive | minimal | disabled |
| production | General deployment | default | standard | enabled |
| high_security | Sensitive data | strict | paranoid | maximum |

### 11.4 Security Logging

Structured security event logging:

```python
from ragshield.integrations import SecurityLogger

logger = SecurityLogger(name="ragshield", level="INFO")

# Automatic event logging
logger.poison_detected(doc_id="doc_001", confidence=0.85)
logger.document_blocked(doc_id="doc_002", reason="High threat")
logger.query_processed(query_id="q_001", latency_ms=15.5)

# Query recent events
events = logger.get_recent_events(count=100)
```

### 11.5 Metrics Collection

Production monitoring metrics:

```python
from ragshield.integrations import SecurityMetrics

metrics = SecurityMetrics()

# Record operations
metrics.record_detection(is_poisoned=True, confidence=0.9, latency_ms=8.5)
metrics.record_document_ingested(accepted=True)
metrics.record_query(latency_ms=12.3, results_count=5)

# Export for monitoring systems
summary = metrics.get_summary()
# Compatible with Prometheus, StatsD, etc.
```

---

## 12. Experimental Results

### 12.1 Evaluation Setup

**Test Environment**:
- Python 3.10
- 16GB RAM, 8-core CPU
- No GPU (CPU-only evaluation)

**Datasets**:
- Synthetic: 10,000 documents (10% poisoned)
- Attack types: direct, adversarial, stealth, chain

### 12.2 Detection Performance

| Detector | Precision | Recall | F1 Score | Latency (p50) |
|----------|-----------|--------|----------|---------------|
| Perplexity | 0.82 | 0.71 | 0.76 | 5.2ms |
| Similarity | 0.79 | 0.68 | 0.73 | 3.1ms |
| Semantic | 0.85 | 0.63 | 0.72 | 12.4ms |
| **Ensemble** | **0.87** | **0.83** | **0.85** | 8.7ms |

### 12.3 Defense Effectiveness

| Defense Level | Block Rate | False Block | Attack Success |
|---------------|------------|-------------|----------------|
| MINIMAL | 45% | 0.5% | 55% |
| STANDARD | 78% | 2.3% | 22% |
| STRICT | 89% | 4.8% | 11% |
| PARANOID | 95% | 8.2% | 5% |

### 12.4 Attack-Specific Results

| Attack Type | Detection Rate | Block Rate |
|-------------|----------------|------------|
| Direct | 94% | 91% |
| Adversarial | 82% | 76% |
| Stealth | 71% | 65% |
| Chain | 78% | 72% |

### 12.5 Performance Overhead

| Operation | Baseline | With RAG-Shield | Overhead |
|-----------|----------|-----------------|----------|
| Document Ingestion | 2.3ms | 3.1ms | 1.35x |
| Query (top-10) | 8.5ms | 10.2ms | 1.20x |
| Batch Ingestion (100) | 180ms | 245ms | 1.36x |

### 12.6 Privacy Cost

| Privacy Level | Epsilon | Recall Loss | Precision Loss |
|---------------|---------|-------------|----------------|
| MINIMAL | 10.0 | 0.2% | 0.1% |
| MEDIUM | 1.0 | 3.5% | 2.1% |
| HIGH | 0.5 | 8.2% | 5.4% |
| MAXIMUM | 0.1 | 18.7% | 12.3% |

---

## 13. Related Work

### 13.1 RAG Security Research

**PoisonedRAG** (Zou et al., 2025): Demonstrated effective knowledge poisoning attacks against RAG systems, achieving 90% attack success with minimal malicious content.

**RAGForensics** (Li et al., 2025): First system for forensic analysis of RAG poisoning attacks, enabling post-hoc investigation.

**TrustRAG** (Wang et al., 2024): Proposed trust scores for retrieved documents, though without cryptographic verification.

### 13.2 Machine Learning Security

**Certified Defenses** (Cohen et al., 2019): Randomized smoothing for certified robustness, which inspired our perturbation-based approaches.

**Data Poisoning** (Steinhardt et al., 2017): Foundational work on training data poisoning, establishing theoretical bounds.

### 13.3 Privacy-Preserving ML

**Differential Privacy** (Dwork et al., 2006): Theoretical foundation for privacy guarantees in data analysis.

**Private Information Retrieval** (Chor et al., 1995): Original PIR protocols that we extend for vector similarity search.

### 13.4 Comparison with RAG-Shield

| System | Detection | Integrity | Privacy | Forensics | Production-Ready |
|--------|-----------|-----------|---------|-----------|------------------|
| PoisonedRAG | ✗ | ✗ | ✗ | ✗ | ✗ |
| RAGForensics | ✗ | ✗ | ✗ | ✓ | ✗ |
| TrustRAG | ✓ | ✗ | ✗ | ✗ | ✗ |
| **RAG-Shield** | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 14. Conclusion and Future Work

### 14.1 Summary

RAG-Shield provides the first comprehensive, defense-in-depth security framework for Retrieval-Augmented Generation systems. Key contributions include:

1. **Multi-layer Protection**: Ensemble detection, cryptographic integrity, differential privacy, and PIR
2. **Practical Performance**: <10ms detection latency, <1.5x ingestion overhead
3. **Production Ready**: Native LangChain and LlamaIndex integration
4. **Comprehensive Evaluation**: Full benchmark framework with standard metrics

### 14.2 Limitations

- **Embedding Model Dependency**: Detection effectiveness depends on embedding quality
- **Computational Cost**: PIR introduces significant overhead for large corpora
- **Adaptive Attacks**: Sophisticated attackers may adapt to detection patterns

### 14.3 Future Directions

1. **Advanced Detection**: Transformer-based poison detection, adversarial training
2. **Hardware Acceleration**: GPU-accelerated PIR, TEE integration
3. **Federated RAG**: Secure multi-party RAG across organizations
4. **Certified Defenses**: Provable robustness guarantees
5. **Multimodal Security**: Extension to image and audio RAG systems

### 14.4 Availability

RAG-Shield is open-source under the MIT License:
- **Repository**: https://github.com/SidereusHu/RAG-Shield
- **Documentation**: Included in repository
- **Examples**: Comprehensive demo scripts

---

## 15. References

1. Zou, W., et al. (2025). "PoisonedRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Generation." USENIX Security.

2. Li, J., et al. (2025). "RAGForensics: Forensic Analysis of Knowledge Poisoning in RAG Systems." ACM Web Conference.

3. Dwork, C., et al. (2006). "Differential Privacy." ICALP.

4. Chor, B., et al. (1995). "Private Information Retrieval." FOCS.

5. Cohen, J., et al. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.

6. Steinhardt, J., et al. (2017). "Certified Defenses for Data Poisoning Attacks." NeurIPS.

7. Merkle, R. (1987). "A Digital Signature Based on a Conventional Encryption Function." CRYPTO.

8. Paillier, P. (1999). "Public-Key Cryptosystems Based on Composite Degree Residuosity Classes." EUROCRYPT.

9. Shamir, A. (1979). "How to Share a Secret." Communications of the ACM.

10. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.

---

## Appendix A: API Reference

### Core Classes

```python
# Document
Document(doc_id, content, embedding, metadata, source)

# Knowledge Base
KnowledgeBase()
kb.add_document(doc)
kb.search(query_embedding, top_k)

# RAGShield
RAGShield(knowledge_base, config)
shield.ingest(document) -> IngestResult
shield.retrieve(query_embedding, top_k) -> List[Document]
```

### Detection

```python
# Create detector
detector = create_poison_detector(preset="default")
result = detector.detect(document)

# Ensemble
ensemble = EnsembleDetector(detectors, strategy)
result = ensemble.detect(document)
```

### Privacy

```python
# Privacy guard
guard = create_privacy_guard(retriever, level=PrivacyLevel.MEDIUM)
result = guard.retrieve(query_embedding, top_k)

# PIR
pir = PIRRetriever(knowledge_base, mode=PIRMode.MULTI_SERVER)
result = pir.retrieve(query_embedding, top_k)
```

---

## Appendix B: Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| RAGSHIELD_DETECTION_PRESET | Detection sensitivity | default |
| RAGSHIELD_DEFENSE_LEVEL | Defense level | standard |
| RAGSHIELD_PRIVACY_EPSILON | Privacy budget | 1.0 |
| RAGSHIELD_MONITORING_LOG_LEVEL | Log verbosity | INFO |

### Defense Level Details

| Level | Detection Threshold | Quarantine | Integrity Check | Privacy |
|-------|--------------------|-----------:|-----------------|---------|
| MINIMAL | 0.9 | No | No | None |
| STANDARD | 0.7 | Yes | Merkle | Basic DP |
| STRICT | 0.5 | Yes | Full | Enhanced DP |
| PARANOID | 0.3 | Yes | Full + Audit | Maximum + PIR |

---

*© 2025 Sidereus Hu. Released under MIT License.*
