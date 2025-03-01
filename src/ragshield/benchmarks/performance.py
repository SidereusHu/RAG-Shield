"""Performance benchmarks for RAG-Shield.

Measures latency, throughput, memory usage, and overhead
of security components.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import gc
import sys
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.benchmarks.metrics import PerformanceMetrics, MetricsCalculator


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks.

    Attributes:
        num_iterations: Number of iterations per benchmark
        warmup_iterations: Warmup iterations (not counted)
        num_documents: Number of documents to use
        embedding_dim: Embedding dimension
        collect_memory: Whether to collect memory stats
    """

    num_iterations: int = 100
    warmup_iterations: int = 10
    num_documents: int = 1000
    embedding_dim: int = 384
    collect_memory: bool = True


@dataclass
class BenchmarkRun:
    """Result of a single benchmark run.

    Attributes:
        name: Benchmark name
        latencies: List of latency measurements (ms)
        memory_before: Memory before run (MB)
        memory_after: Memory after run (MB)
        metadata: Additional metadata
    """

    name: str
    latencies: List[float] = field(default_factory=list)
    memory_before: float = 0.0
    memory_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_latency(self) -> float:
        """Mean latency in ms."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def memory_delta(self) -> float:
        """Memory change in MB."""
        return self.memory_after - self.memory_before

    def to_metrics(self) -> PerformanceMetrics:
        """Convert to PerformanceMetrics."""
        return MetricsCalculator.performance_metrics(
            latencies=self.latencies,
            memory_samples=[self.memory_after],
        )


class PerformanceBenchmark:
    """Performance benchmarking utilities.

    Provides standardized methods for measuring latency,
    throughput, and resource usage.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, BenchmarkRun] = {}

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        gc.collect()
        # Simple estimation using sys.getsizeof for tracked objects
        # In production, use tracemalloc or memory_profiler
        return 0.0  # Placeholder

    def _generate_documents(self, n: int) -> List[Document]:
        """Generate test documents.

        Args:
            n: Number of documents

        Returns:
            List of documents
        """
        docs = []
        for i in range(n):
            embedding = np.random.randn(self.config.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            doc = Document(
                doc_id=f"bench_doc_{i}",
                content=f"Benchmark document {i} with some content for testing performance.",
                embedding=embedding.tolist(),
            )
            docs.append(doc)
        return docs

    def benchmark_function(
        self,
        name: str,
        func: Callable,
        *args,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        **kwargs,
    ) -> BenchmarkRun:
        """Benchmark a function.

        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Arguments to pass to function
            setup: Optional setup function (called before each iteration)
            teardown: Optional teardown function (called after each iteration)
            **kwargs: Keyword arguments to pass to function

        Returns:
            Benchmark run result
        """
        run = BenchmarkRun(name=name)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            if setup:
                setup()
            func(*args, **kwargs)
            if teardown:
                teardown()

        # Collect garbage before measurement
        gc.collect()
        run.memory_before = self._get_memory_mb()

        # Actual benchmark
        for _ in range(self.config.num_iterations):
            if setup:
                setup()

            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()

            run.latencies.append((end - start) * 1000)  # Convert to ms

            if teardown:
                teardown()

        gc.collect()
        run.memory_after = self._get_memory_mb()

        self.results[name] = run
        return run

    def benchmark_knowledge_base_add(
        self,
        kb: Optional[KnowledgeBase] = None,
    ) -> BenchmarkRun:
        """Benchmark document addition to knowledge base.

        Args:
            kb: Optional knowledge base (creates new if not provided)

        Returns:
            Benchmark run result
        """
        docs = self._generate_documents(self.config.num_iterations)

        def add_doc(kb: KnowledgeBase, doc: Document):
            kb.add_document(doc)

        kb = kb or KnowledgeBase()
        latencies = []

        for doc in docs:
            start = time.perf_counter()
            add_doc(kb, doc)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        run = BenchmarkRun(
            name="knowledge_base_add",
            latencies=latencies,
            metadata={"num_documents": len(docs)},
        )
        self.results["knowledge_base_add"] = run
        return run

    def benchmark_knowledge_base_search(
        self,
        kb: Optional[KnowledgeBase] = None,
        top_k: int = 10,
    ) -> BenchmarkRun:
        """Benchmark knowledge base search.

        Args:
            kb: Optional knowledge base
            top_k: Number of results to retrieve

        Returns:
            Benchmark run result
        """
        # Setup knowledge base
        if kb is None:
            kb = KnowledgeBase()
            docs = self._generate_documents(self.config.num_documents)
            for doc in docs:
                kb.add_document(doc)

        # Generate query embeddings
        queries = [
            np.random.randn(self.config.embedding_dim).astype(np.float32)
            for _ in range(self.config.num_iterations)
        ]
        for i in range(len(queries)):
            queries[i] = queries[i] / np.linalg.norm(queries[i])

        latencies = []

        for query in queries:
            start = time.perf_counter()
            kb.search(query.tolist(), top_k=top_k)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        run = BenchmarkRun(
            name="knowledge_base_search",
            latencies=latencies,
            metadata={"num_documents": len(kb), "top_k": top_k},
        )
        self.results["knowledge_base_search"] = run
        return run

    def benchmark_detection(
        self,
        detector,
        documents: Optional[List[Document]] = None,
    ) -> BenchmarkRun:
        """Benchmark poison detection.

        Args:
            detector: Poison detector instance
            documents: Optional documents to test

        Returns:
            Benchmark run result
        """
        if documents is None:
            documents = self._generate_documents(self.config.num_iterations)

        latencies = []

        for doc in documents:
            start = time.perf_counter()
            detector.detect(doc)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        run = BenchmarkRun(
            name="detection",
            latencies=latencies,
            metadata={"detector_type": type(detector).__name__},
        )
        self.results["detection"] = run
        return run

    def benchmark_pir_retrieval(
        self,
        retriever,
        num_queries: Optional[int] = None,
    ) -> BenchmarkRun:
        """Benchmark PIR retrieval.

        Args:
            retriever: PIR retriever instance
            num_queries: Number of queries to run

        Returns:
            Benchmark run result
        """
        num_queries = num_queries or self.config.num_iterations

        # Generate queries
        queries = [
            np.random.randn(self.config.embedding_dim).astype(np.float32)
            for _ in range(num_queries)
        ]

        latencies = []

        for query in queries:
            start = time.perf_counter()
            retriever.retrieve(query, top_k=5)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        run = BenchmarkRun(
            name="pir_retrieval",
            latencies=latencies,
            metadata={"retriever_type": type(retriever).__name__},
        )
        self.results["pir_retrieval"] = run
        return run

    def benchmark_shield_ingest(
        self,
        shield,
        documents: Optional[List[Document]] = None,
    ) -> BenchmarkRun:
        """Benchmark RAGShield document ingestion.

        Args:
            shield: RAGShield instance
            documents: Optional documents to ingest

        Returns:
            Benchmark run result
        """
        if documents is None:
            documents = self._generate_documents(self.config.num_iterations)

        latencies = []

        for doc in documents:
            start = time.perf_counter()
            shield.ingest_document(doc)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        run = BenchmarkRun(
            name="shield_ingest",
            latencies=latencies,
            metadata={"defense_level": str(shield.defense_level)},
        )
        self.results["shield_ingest"] = run
        return run

    def compare_with_baseline(
        self,
        baseline_run: BenchmarkRun,
        comparison_run: BenchmarkRun,
    ) -> Dict[str, Any]:
        """Compare benchmark run with baseline.

        Args:
            baseline_run: Baseline benchmark
            comparison_run: Comparison benchmark

        Returns:
            Comparison results
        """
        baseline_mean = baseline_run.mean_latency
        comparison_mean = comparison_run.mean_latency

        return {
            "baseline": {
                "name": baseline_run.name,
                "mean_latency_ms": baseline_mean,
            },
            "comparison": {
                "name": comparison_run.name,
                "mean_latency_ms": comparison_mean,
            },
            "overhead_ratio": comparison_mean / baseline_mean if baseline_mean > 0 else 0,
            "overhead_ms": comparison_mean - baseline_mean,
            "overhead_percent": (
                (comparison_mean - baseline_mean) / baseline_mean * 100
                if baseline_mean > 0 else 0
            ),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report.

        Returns:
            Complete benchmark report
        """
        report = {
            "config": {
                "num_iterations": self.config.num_iterations,
                "warmup_iterations": self.config.warmup_iterations,
                "num_documents": self.config.num_documents,
                "embedding_dim": self.config.embedding_dim,
            },
            "results": {},
        }

        for name, run in self.results.items():
            metrics = run.to_metrics()
            report["results"][name] = {
                "latency_ms": metrics.latency_ms,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "latency_p99_ms": metrics.latency_p99_ms,
                "throughput": metrics.throughput,
                "memory_delta_mb": run.memory_delta,
                "metadata": run.metadata,
            }

        return report


class ScalabilityBenchmark:
    """Benchmark scalability across different sizes.

    Tests how performance scales with increasing data size.
    """

    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        embedding_dim: int = 384,
    ):
        """Initialize scalability benchmark.

        Args:
            sizes: List of sizes to test
            embedding_dim: Embedding dimension
        """
        self.sizes = sizes or [100, 500, 1000, 5000, 10000]
        self.embedding_dim = embedding_dim
        self.results: Dict[int, Dict[str, float]] = {}

    def benchmark_kb_scaling(self) -> Dict[int, Dict[str, float]]:
        """Benchmark knowledge base scaling.

        Returns:
            Results for each size
        """
        for size in self.sizes:
            kb = KnowledgeBase()

            # Generate documents
            docs = []
            for i in range(size):
                embedding = np.random.randn(self.embedding_dim).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                doc = Document(
                    doc_id=f"scale_doc_{i}",
                    content=f"Scalability test document {i}.",
                    embedding=embedding.tolist(),
                )
                docs.append(doc)

            # Benchmark add
            start = time.perf_counter()
            for doc in docs:
                kb.add_document(doc)
            add_time = (time.perf_counter() - start) * 1000

            # Benchmark search
            query = np.random.randn(self.embedding_dim).astype(np.float32)
            query = query / np.linalg.norm(query)

            search_times = []
            for _ in range(10):
                start = time.perf_counter()
                kb.search(query.tolist(), top_k=10)
                search_times.append((time.perf_counter() - start) * 1000)

            self.results[size] = {
                "add_total_ms": add_time,
                "add_per_doc_ms": add_time / size,
                "search_avg_ms": sum(search_times) / len(search_times),
            }

        return self.results

    def estimate_complexity(self) -> Dict[str, str]:
        """Estimate algorithmic complexity from results.

        Returns:
            Complexity estimates
        """
        if len(self.results) < 2:
            return {"error": "Not enough data points"}

        sizes = sorted(self.results.keys())
        add_times = [self.results[s]["add_per_doc_ms"] for s in sizes]
        search_times = [self.results[s]["search_avg_ms"] for s in sizes]

        # Simple heuristic: check if time grows with size
        add_ratio = add_times[-1] / add_times[0] if add_times[0] > 0 else 1
        search_ratio = search_times[-1] / search_times[0] if search_times[0] > 0 else 1
        size_ratio = sizes[-1] / sizes[0]

        def classify(ratio: float, size_ratio: float) -> str:
            if ratio < 1.5:
                return "O(1) - constant"
            elif ratio < size_ratio * 0.5:
                return "O(log n) - logarithmic"
            elif ratio < size_ratio * 1.5:
                return "O(n) - linear"
            else:
                return "O(n²) or worse"

        return {
            "add_complexity": classify(add_ratio, size_ratio),
            "search_complexity": classify(search_ratio, size_ratio),
            "sizes_tested": sizes,
        }


def run_standard_benchmarks(
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Run standard benchmark suite.

    Args:
        config: Optional benchmark configuration

    Returns:
        Complete benchmark results
    """
    benchmark = PerformanceBenchmark(config=config)

    # Knowledge base benchmarks
    benchmark.benchmark_knowledge_base_add()
    benchmark.benchmark_knowledge_base_search()

    return benchmark.generate_report()


def run_scalability_benchmarks(
    sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run scalability benchmarks.

    Args:
        sizes: Sizes to test

    Returns:
        Scalability results
    """
    benchmark = ScalabilityBenchmark(sizes=sizes)
    results = benchmark.benchmark_kb_scaling()
    complexity = benchmark.estimate_complexity()

    return {
        "scaling_results": results,
        "complexity_estimates": complexity,
    }
