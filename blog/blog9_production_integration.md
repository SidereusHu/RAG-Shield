# Blog 9: From Research to Production - Integrating RAG Security

*Seamlessly adding security to LangChain, LlamaIndex, and production deployments*

## Introduction

We've built a comprehensive RAG security framework with detection, defense, privacy protection, forensics, and benchmarking. But a security solution is only valuable if it can be easily integrated into real applications. Most RAG applications today are built on frameworks like LangChain and LlamaIndex - our security layer must work seamlessly with these tools.

In this final blog of the series, we'll cover how to integrate RAG-Shield into production applications, including framework adapters, configuration management, and monitoring.

## The Integration Challenge

RAG frameworks have their own abstractions for documents, retrievers, and vector stores. Our security layer must:

```
    The Integration Problem

    Your Application
          |
          v
    +------------------+
    |   LangChain /    |     How do we add security
    |   LlamaIndex     |     without rewriting
    |                  |     the application?
    +------------------+
          |
          v
    +------------------+
    |  Vector Store    |
    |  (FAISS, etc.)   |
    +------------------+


    Solution: Wrapper Pattern

    Your Application
          |
          v
    +------------------+
    |   LangChain /    |
    |   LlamaIndex     |
    +------------------+
          |
          v
    +------------------+
    | RAG-Shield       |  <-- Security layer wraps
    | Secure Wrappers  |      existing components
    +------------------+
          |
          v
    +------------------+
    |  Vector Store    |
    +------------------+
```

## Base Integration Architecture

We start with a common base for all framework integrations:

```python
@dataclass
class IntegrationConfig:
    """Configuration for framework integration."""

    defense_level: DefenseLevel = DefenseLevel.STANDARD
    detector_preset: str = "default"
    enable_detection: bool = True
    enable_sanitization: bool = True
    enable_monitoring: bool = True
    enable_forensics: bool = True
    auto_quarantine: bool = True
    on_threat_callback: Optional[Callable] = None


class BaseRAGIntegration(ABC):
    """Abstract base for framework integrations."""

    def __init__(self, config: IntegrationConfig):
        self.config = config

        # Initialize security components
        self._detector = create_poison_detector(
            preset=config.detector_preset
        )
        self._shield = RAGShield(
            knowledge_base=KnowledgeBase(),
            detector=self._detector,
        )

    @abstractmethod
    def wrap_retriever(self, retriever: Any) -> Any:
        """Wrap a retriever with security."""
        pass

    @abstractmethod
    def wrap_vector_store(self, vector_store: Any) -> Any:
        """Wrap a vector store with security."""
        pass
```

### Secure Document Wrapper

A universal wrapper for documents across frameworks:

```python
@dataclass
class SecureDocument:
    """Document with security metadata."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    doc_id: Optional[str] = None
    is_verified: bool = False
    threat_score: float = 0.0
    security_flags: List[str] = field(default_factory=list)

    def to_document(self) -> Document:
        """Convert to RAG-Shield Document."""
        return Document(
            doc_id=self.doc_id,
            content=self.content,
            embedding=self.embedding,
            metadata=self.metadata,
        )
```

## LangChain Integration

LangChain is one of the most popular RAG frameworks. Here's how we integrate:

### Architecture

```
    LangChain Integration Architecture

    +------------------------+
    |   Your LangChain App   |
    +------------------------+
              |
              v
    +------------------------+
    | LangChainIntegration   |
    | - wrap_retriever()     |
    | - wrap_vector_store()  |
    | - secure_documents()   |
    +------------------------+
         /        \
        /          \
       v            v
    +--------+  +-------------+
    | Secure |  | Secure      |
    | Vector |  | Retriever   |
    | Store  |  |             |
    +--------+  +-------------+
        |              |
        v              v
    +--------+  +-------------+
    | FAISS  |  | Base        |
    | Chroma |  | Retriever   |
    | etc.   |  |             |
    +--------+  +-------------+
```

### Secure Vector Store

```python
class SecureVectorStore:
    """Secure wrapper for LangChain vector stores."""

    def __init__(
        self,
        base_store: VectorStore,
        detector: PoisonDetector,
        config: IntegrationConfig,
    ):
        self.base_store = base_store
        self.detector = detector
        self.config = config

    def add_documents(self, documents: List[LCDocument]) -> List[str]:
        """Add documents with security screening."""
        safe_docs = []

        for doc in documents:
            # Convert and analyze
            rag_doc = Document(
                doc_id=doc.metadata.get("doc_id", "unknown"),
                content=doc.page_content,
            )

            detection = self.detector.detect(rag_doc)

            # Block suspicious documents
            if detection.is_poisoned and detection.confidence > 0.8:
                if self.config.on_threat_callback:
                    self.config.on_threat_callback({
                        "doc_id": rag_doc.doc_id,
                        "threat_score": detection.confidence,
                        "action": "blocked",
                    })
                continue

            safe_docs.append(doc)

        return self.base_store.add_documents(safe_docs)

    def similarity_search(self, query: str, k: int = 4) -> List[LCDocument]:
        """Search with security filtering."""
        # Get extra results for filtering
        results = self.base_store.similarity_search(query, k=k*2)

        # Filter suspicious results
        filtered = []
        for doc in results:
            rag_doc = Document(content=doc.page_content)
            detection = self.detector.detect(rag_doc)

            if detection.is_poisoned and detection.confidence > 0.85:
                continue

            doc.metadata["_threat_score"] = detection.confidence
            filtered.append(doc)

            if len(filtered) >= k:
                break

        return filtered
```

### Usage Example

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from ragshield.integrations import (
    LangChainIntegration,
    IntegrationConfig,
)
from ragshield.defense import DefenseLevel

# Create integration
integration = LangChainIntegration(
    config=IntegrationConfig(
        defense_level=DefenseLevel.STANDARD,
        enable_detection=True,
    )
)

# Create your vector store as usual
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Wrap it with security
secure_store = integration.wrap_vector_store(vectorstore)

# Use exactly like before - but now it's secure!
results = secure_store.similarity_search("What is machine learning?")
```

## LlamaIndex Integration

LlamaIndex has different abstractions, so we provide tailored wrappers:

### Secure Query Engine

```python
class SecureQueryEngine:
    """Secure wrapper for LlamaIndex query engines."""

    def __init__(
        self,
        base_engine: BaseQueryEngine,
        detector: PoisonDetector,
        config: IntegrationConfig,
    ):
        self.base_engine = base_engine
        self.detector = detector
        self.config = config

    def query(self, query_str: str) -> Response:
        """Query with security checks."""
        response = self.base_engine.query(query_str)

        # Check source nodes for threats
        if hasattr(response, 'source_nodes') and self.detector:
            secure_nodes = []

            for node_with_score in response.source_nodes:
                content = node_with_score.node.get_content()

                doc = Document(content=content)
                detection = self.detector.detect(doc)

                if detection.is_poisoned and detection.confidence > 0.9:
                    # Mark response as potentially compromised
                    response.metadata = response.metadata or {}
                    response.metadata['_security_warning'] = (
                        "Response may be influenced by suspicious content"
                    )
                    continue

                secure_nodes.append(node_with_score)

            response.source_nodes = secure_nodes

        return response
```

### Usage Example

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from ragshield.integrations import (
    LlamaIndexIntegration,
    IntegrationConfig,
)

# Create integration
integration = LlamaIndexIntegration(
    config=IntegrationConfig(defense_level=DefenseLevel.STRICT)
)

# Load and secure documents
documents = SimpleDirectoryReader("data").load_data()
secure_docs = integration.secure_documents(documents)

# Create index with secure documents
index = VectorStoreIndex.from_documents(secure_docs)

# Wrap query engine
query_engine = integration.wrap_query_engine(
    index.as_query_engine()
)

# Query securely
response = query_engine.query("Explain quantum computing")
```

## Configuration Management

Production deployments need flexible configuration:

### Configuration Hierarchy

```
    Configuration Priority (highest to lowest)

    +-----------------------+
    |   Runtime Overrides   |  manager.set("key", value)
    +-----------------------+
              |
              v
    +-----------------------+
    | Environment Variables |  RAGSHIELD_DETECTION_PRESET=strict
    +-----------------------+
              |
              v
    +-----------------------+
    |    Config File        |  config/ragshield.json
    +-----------------------+
              |
              v
    +-----------------------+
    |   Default Values      |  Built-in defaults
    +-----------------------+
```

### Configuration Structure

```python
@dataclass
class RAGShieldConfig:
    """Complete RAG-Shield configuration."""

    detection: DetectionConfig
    defense: DefenseSettings
    privacy: PrivacyConfig
    monitoring: MonitoringConfig
    forensics: ForensicsConfig

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "RAGShieldConfig":
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
```

### Environment Variables

```bash
# Detection settings
export RAGSHIELD_DETECTION_ENABLED=true
export RAGSHIELD_DETECTION_PRESET=strict

# Defense settings
export RAGSHIELD_DEFENSE_LEVEL=paranoid
export RAGSHIELD_DEFENSE_AUTO_QUARANTINE=true

# Privacy settings
export RAGSHIELD_PRIVACY_EPSILON=0.5
export RAGSHIELD_PRIVACY_PIR_ENABLED=true

# Monitoring
export RAGSHIELD_MONITORING_LOG_LEVEL=INFO
export RAGSHIELD_MONITORING_RATE_LIMIT=100
```

### Pre-defined Profiles

```python
PROFILES = {
    "development": RAGShieldConfig(
        detection=DetectionConfig(preset="permissive"),
        defense=DefenseSettings(level=DefenseLevel.MINIMAL),
        monitoring=MonitoringConfig(log_level="DEBUG"),
    ),

    "production": RAGShieldConfig(
        detection=DetectionConfig(preset="default"),
        defense=DefenseSettings(level=DefenseLevel.STANDARD),
        monitoring=MonitoringConfig(log_level="INFO"),
    ),

    "high_security": RAGShieldConfig(
        detection=DetectionConfig(preset="strict"),
        defense=DefenseSettings(
            level=DefenseLevel.PARANOID,
            auto_quarantine=True,
            auto_block_sources=True,
        ),
        privacy=PrivacyConfig(epsilon=0.1),
        monitoring=MonitoringConfig(alert_threshold=0.5),
    ),
}

# Usage
config = get_config("production")
```

### Config Manager

```python
# Create manager with multiple sources
manager = ConfigManager(
    config_file="config/ragshield.json",
    load_env=True,
)

# Get values with dotted notation
enabled = manager.get("detection.enabled")
level = manager.get("defense.level")

# Override at runtime
manager.set("detection.preset", "strict")

# Validate configuration
errors = manager.validate()
if errors:
    print(f"Config errors: {errors}")
```

## Security Logging

Production deployments need comprehensive logging:

### Security Event Types

```
    Security Event Taxonomy

    Detection Events
    ├── POISON_DETECTED
    ├── DOCUMENT_VERIFIED
    └── DETECTION_ERROR

    Defense Events
    ├── DOCUMENT_BLOCKED
    ├── DOCUMENT_QUARANTINED
    ├── DOCUMENT_RELEASED
    └── SOURCE_BLOCKED

    Query Events
    ├── QUERY_PROCESSED
    ├── QUERY_FILTERED
    └── SUSPICIOUS_QUERY

    System Events
    ├── RATE_LIMIT_EXCEEDED
    ├── ANOMALY_DETECTED
    └── ALERT_TRIGGERED

    Audit Events
    ├── CONFIG_CHANGED
    └── ACCESS_DENIED
```

### Structured Logging

```python
class SecurityLogger:
    """Structured security event logger."""

    def poison_detected(
        self,
        doc_id: str,
        confidence: float,
        details: Optional[Dict] = None,
    ) -> None:
        """Log a poison detection event."""
        event = SecurityEvent(
            event_type=SecurityEventType.POISON_DETECTED,
            severity="WARNING",
            message=f"Poison detected in {doc_id} (conf: {confidence:.2f})",
            doc_id=doc_id,
            details={"confidence": confidence, **(details or {})},
        )
        self.log_event(event)

    def add_callback(self, callback: Callable) -> None:
        """Add callback for real-time alerts."""
        self._callbacks.append(callback)

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[SecurityEventType] = None,
    ) -> List[SecurityEvent]:
        """Get recent events with optional filtering."""
        events = self._event_history[-count:]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events


# Usage
logger = SecurityLogger(name="ragshield", level="INFO")

logger.poison_detected("doc_123", 0.85, {"method": "perplexity"})
logger.document_blocked("doc_456", "High threat score")
logger.query_processed("q_001", latency_ms=15.5, results_count=5)
```

## Metrics Collection

Monitor your RAG security in real-time:

### Metrics Types

```python
class MetricsCollector:
    """Collects security metrics."""

    def increment(self, name: str, value: float = 1.0, labels: Dict = None):
        """Increment a counter."""
        pass

    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """Set a gauge value."""
        pass

    def observe(self, name: str, value: float, labels: Dict = None):
        """Observe value for histogram."""
        pass


class SecurityMetrics:
    """Pre-defined security metrics."""

    def record_detection(self, is_poisoned: bool, confidence: float, latency_ms: float):
        self.collector.increment("detection_total")
        if is_poisoned:
            self.collector.increment("detection_poisoned")
        self.collector.observe("detection_latency_ms", latency_ms)

    def record_document_ingested(self, accepted: bool):
        self.collector.increment("ingestion_total")
        if not accepted:
            self.collector.increment("ingestion_rejected")

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "detection": {
                "total": self.collector.get_counter("detection_total"),
                "poisoned": self.collector.get_counter("detection_poisoned"),
                "poison_rate": ...,
            },
            "ingestion": {
                "total": ...,
                "rejection_rate": ...,
            },
            ...
        }
```

### Metrics Dashboard

```
    RAG-Shield Security Dashboard

    +--------------------------------------------------+
    |  Detection                    Ingestion          |
    |  +-----------+               +-----------+       |
    |  | Total: 1M |               | Total: 50K|       |
    |  | Poison: 5K|               | Reject: 2K|       |
    |  | Rate: 0.5%|               | Rate: 4%  |       |
    |  +-----------+               +-----------+       |
    +--------------------------------------------------+
    |  Query Latency (ms)                              |
    |  p50: 12  |  p95: 45  |  p99: 120               |
    |  [====================|====|==]                  |
    +--------------------------------------------------+
    |  Recent Alerts                                   |
    |  [WARN] Poison detected: doc_789 (0.92)         |
    |  [WARN] Rate limit exceeded: source_abc         |
    |  [INFO] Quarantine size: 15 documents           |
    +--------------------------------------------------+
```

## Production Deployment Guide

### Deployment Checklist

```
Production Deployment Checklist

[ ] Configuration
    [ ] Create config/ragshield.json
    [ ] Set environment variables for secrets
    [ ] Choose appropriate profile (production/high_security)
    [ ] Enable config validation

[ ] Logging
    [ ] Configure log level (INFO for production)
    [ ] Set up log file rotation
    [ ] Configure JSON output for log aggregation
    [ ] Set up alert callbacks

[ ] Monitoring
    [ ] Enable metrics collection
    [ ] Set up dashboard
    [ ] Configure anomaly thresholds
    [ ] Set up rate limiting

[ ] Integration
    [ ] Wrap all vector stores
    [ ] Wrap all retrievers/query engines
    [ ] Configure threat callbacks
    [ ] Set up quarantine review workflow

[ ] Testing
    [ ] Run benchmarks
    [ ] Test with adversarial inputs
    [ ] Verify logging works
    [ ] Test alert notifications
```

### Example Production Setup

```python
from ragshield.integrations import (
    create_config_manager,
    configure_logging,
    get_metrics,
    LangChainIntegration,
    IntegrationConfig,
)
from ragshield.defense import DefenseLevel

# 1. Load configuration
config_manager = create_config_manager(
    config_file="config/ragshield.json",
    profile="production",
    load_env=True,
)

# 2. Configure logging
logger = configure_logging(
    level="INFO",
    json_output=True,
    log_file="logs/security.log",
)

# 3. Set up metrics
metrics = get_metrics()

# 4. Create integration
integration = LangChainIntegration(
    config=IntegrationConfig(
        defense_level=config_manager.get("defense.level"),
        detector_preset=config_manager.get("detection.preset"),
        on_threat_callback=lambda e: logger.alert_triggered(
            "threat_detected", str(e)
        ),
    )
)

# 5. Wrap your components
secure_store = integration.wrap_vector_store(your_vectorstore)
secure_retriever = integration.wrap_retriever(your_retriever)

# Ready for production!
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV RAGSHIELD_DETECTION_PRESET=default
ENV RAGSHIELD_DEFENSE_LEVEL=standard
ENV RAGSHIELD_MONITORING_LOG_LEVEL=INFO

# Run
CMD ["python", "app.py"]
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ragshield-config
data:
  ragshield.json: |
    {
      "detection": {
        "enabled": true,
        "preset": "default"
      },
      "defense": {
        "level": "standard",
        "auto_quarantine": true
      },
      "monitoring": {
        "enabled": true,
        "log_level": "INFO"
      }
    }
```

## Conclusion

With framework integrations and production utilities, RAG-Shield is now ready for real-world deployment. Key takeaways:

1. **Seamless Integration**: Wrap existing LangChain/LlamaIndex components without code changes
2. **Flexible Configuration**: Environment variables, config files, and runtime overrides
3. **Comprehensive Logging**: Structured security events for audit and debugging
4. **Production Metrics**: Monitor detection rates, latencies, and threats

The wrapper pattern allows you to add security to any RAG application with minimal changes:

```python
# Before
results = vectorstore.similarity_search(query)

# After (with security)
secure_store = integration.wrap_vector_store(vectorstore)
results = secure_store.similarity_search(query)  # Same API!
```

## Series Conclusion

Over this 9-part series, we've built a comprehensive RAG security framework:

1. **Threat Landscape**: Understanding RAG poisoning attacks
2. **Detection**: Perplexity, similarity, and semantic analysis
3. **Integrity**: Merkle trees for tamper detection
4. **Privacy**: Differential privacy for query protection
5. **PIR**: Private information retrieval
6. **Forensics**: Attack analysis and attribution
7. **Defense**: Active protection and quarantine
8. **Benchmarks**: Rigorous evaluation framework
9. **Integration**: Production deployment

RAG-Shield provides defense-in-depth for retrieval-augmented generation systems. The source code is available at [GitHub](https://github.com/SidereusHu/RAG-Shield).

---

*This concludes the RAG-Shield blog series. Thank you for reading!*
