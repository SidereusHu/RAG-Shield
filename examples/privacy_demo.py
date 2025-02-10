#!/usr/bin/env python3
"""
Privacy-Preserving RAG Retrieval Demo

This demo shows how to use RAG-Shield's privacy protection features:
1. Differential Privacy for retrieval scores
2. Query sanitization for query privacy
3. Privacy budget management

Run this demo:
    python examples/privacy_demo.py
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# RAG-Shield imports
from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.core.retriever import SimpleRetriever
from ragshield.privacy import (
    # Budget Management
    PrivacyBudgetManager,
    BudgetExceededError,
    # DP Retrieval
    DPRetriever,
    NoiseMechanism,
    # Query Sanitization
    PerturbationSanitizer,
    DummyQuerySanitizer,
    # Privacy Guard
    PrivacyGuard,
    PrivacyConfig,
    PrivacyLevel,
    create_privacy_guard,
    generate_privacy_report,
    QueryProtectionMethod,
)

console = Console()


def create_sample_knowledge_base() -> KnowledgeBase:
    """Create a sample knowledge base with medical documents."""
    documents = [
        ("doc_1", "Aspirin is commonly used to treat mild pain and fever. It belongs to NSAIDs."),
        ("doc_2", "Ibuprofen is an anti-inflammatory drug used for pain relief and reducing inflammation."),
        ("doc_3", "Acetaminophen (Tylenol) is used to treat pain and fever but is not anti-inflammatory."),
        ("doc_4", "Blood pressure medications include ACE inhibitors, beta blockers, and diuretics."),
        ("doc_5", "Diabetes management includes insulin therapy and oral medications like metformin."),
        ("doc_6", "Antibiotics like penicillin are used to treat bacterial infections."),
        ("doc_7", "Antihistamines are used to treat allergic reactions and symptoms."),
        ("doc_8", "Statins are used to lower cholesterol levels in the blood."),
        ("doc_9", "Antidepressants include SSRIs, SNRIs, and tricyclic antidepressants."),
        ("doc_10", "Pain management may involve opioids for severe pain under medical supervision."),
    ]

    kb = KnowledgeBase()

    for doc_id, content in documents:
        # Create random embedding (in practice, use actual embeddings)
        embedding = np.random.randn(64).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        doc = Document(
            doc_id=doc_id,
            content=content,
            embedding=embedding.tolist(),
        )
        kb.add_document(doc)

    return kb


def demo_budget_management():
    """Demonstrate privacy budget management."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 1: Privacy Budget Management")
    console.print("[bold cyan]=" * 60)

    # Create budget manager with ε = 1.0
    budget = PrivacyBudgetManager(
        epsilon_budget=1.0,
        delta_budget=1e-5,
    )

    console.print("\n[yellow]Initial budget status:")
    status = budget.get_status()
    console.print(f"  Total ε budget: {budget.epsilon_budget}")
    console.print(f"  Spent: {status.total_epsilon:.4f}")
    console.print(f"  Remaining: {status.remaining_epsilon:.4f}")

    # Simulate some privacy spending
    operations = [
        (0.1, "Query: pain medication"),
        (0.2, "Query: diabetes treatment"),
        (0.15, "Query: blood pressure"),
    ]

    console.print("\n[yellow]Spending privacy budget:")
    for epsilon, operation in operations:
        budget.spend(epsilon, operation=operation)
        status = budget.get_status()
        console.print(f"  {operation}: spent ε={epsilon}")
        console.print(f"    Cumulative: {status.total_epsilon:.2f}, "
                     f"Remaining: {status.remaining_epsilon:.2f}, "
                     f"Utilization: {status.utilization:.1%}")

    # Try to exceed budget
    console.print("\n[yellow]Attempting to exceed budget:")
    try:
        budget.spend(0.8, operation="Query: too expensive")
    except BudgetExceededError as e:
        console.print(f"  [red]BudgetExceededError: {e}")

    # Show remaining queries
    console.print(f"\n[green]Estimated queries remaining (at ε=0.1 each): "
                 f"{budget.estimate_queries_remaining(0.1)}")


def demo_dp_retrieval():
    """Demonstrate differential privacy retrieval."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 2: Differential Privacy Retrieval")
    console.print("[bold cyan]=" * 60)

    # Create knowledge base and retriever
    kb = create_sample_knowledge_base()
    retriever = SimpleRetriever(metric="cosine")
    retriever.index(kb)

    # Create DP retriever
    budget = PrivacyBudgetManager(epsilon_budget=2.0)
    dp_retriever = DPRetriever(
        retriever=retriever,
        budget_manager=budget,
        epsilon_per_query=0.2,
        mechanism=NoiseMechanism.LAPLACE,
    )

    # Create query
    query = np.random.randn(64).astype(np.float32)
    query = query / np.linalg.norm(query)

    console.print("\n[yellow]Performing DP retrieval (ε=0.2 per query):")

    # Compare multiple runs to show noise effect
    for run in range(3):
        result = dp_retriever.retrieve(query, top_k=3)

        console.print(f"\n  [cyan]Run {run + 1}:")
        console.print(f"    Epsilon spent: {result.epsilon_spent}")
        console.print(f"    Noise scale: {result.noise_scale:.4f}")
        console.print(f"    Order preserved: {result.original_order_preserved}")
        console.print(f"    Top documents:")
        for doc, score in result.documents:
            console.print(f"      - {doc.doc_id}: {score:.4f}")

    # Show stats
    stats = dp_retriever.get_stats()
    console.print(f"\n[green]Retriever stats:")
    console.print(f"  Queries made: {stats['query_count']}")
    console.print(f"  Budget remaining: {stats['budget_remaining']:.2f}")
    console.print(f"  Queries remaining: {stats['queries_remaining']}")


def demo_query_sanitization():
    """Demonstrate query sanitization methods."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 3: Query Sanitization")
    console.print("[bold cyan]=" * 60)

    # Create sample query
    query = np.random.randn(64).astype(np.float32)
    query = query / np.linalg.norm(query)

    # 1. Perturbation Sanitizer
    console.print("\n[yellow]1. Perturbation Sanitizer:")
    perturbation = PerturbationSanitizer(noise_scale=0.1)
    result = perturbation.sanitize(query)

    original_norm = np.linalg.norm(query)
    perturbed_norm = np.linalg.norm(result.embeddings[0])
    cosine_sim = np.dot(query, result.embeddings[0]) / (original_norm * perturbed_norm)

    console.print(f"  Original norm: {original_norm:.4f}")
    console.print(f"  Perturbed norm: {perturbed_norm:.4f}")
    console.print(f"  Cosine similarity to original: {cosine_sim:.4f}")
    console.print(f"  Privacy cost: ε={result.privacy_cost}")

    # 2. Dummy Query Sanitizer
    console.print("\n[yellow]2. Dummy Query Sanitizer (k-anonymity):")
    dummy = DummyQuerySanitizer(num_dummies=4)
    result = dummy.sanitize(query)

    console.print(f"  Total queries generated: {result.num_queries}")
    console.print(f"  Real query index: {result.real_index} (hidden from server)")
    console.print(f"  k-anonymity: {result.metadata.get('k_anonymity', 'N/A')}")

    # Show all dummy similarities
    console.print("  Dummy query similarities to real:")
    for i in range(result.num_queries):
        sim = np.dot(query, result.embeddings[i]) / (
            np.linalg.norm(query) * np.linalg.norm(result.embeddings[i])
        )
        is_real = "← REAL" if i == result.real_index else ""
        console.print(f"    Query {i}: sim={sim:.4f} {is_real}")


def demo_privacy_guard():
    """Demonstrate the unified PrivacyGuard."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 4: Privacy Guard (Unified Protection)")
    console.print("[bold cyan]=" * 60)

    # Create knowledge base and retriever
    kb = create_sample_knowledge_base()
    retriever = SimpleRetriever(metric="cosine")
    retriever.index(kb)

    # Create privacy guard with different levels
    console.print("\n[yellow]Testing different privacy levels:")

    table = Table(title="Privacy Level Comparison")
    table.add_column("Level", style="cyan")
    table.add_column("ε/query", justify="right")
    table.add_column("Query Protection", justify="center")
    table.add_column("Mechanism", justify="center")

    levels = [
        PrivacyLevel.MINIMAL,
        PrivacyLevel.LOW,
        PrivacyLevel.MEDIUM,
        PrivacyLevel.HIGH,
        PrivacyLevel.MAXIMUM,
    ]

    for level in levels:
        guard = create_privacy_guard(retriever, level=level)
        config = guard.config
        table.add_row(
            level.value,
            f"{config.epsilon_per_query:.2f}",
            "Yes" if config.enable_query_protection else "No",
            config.noise_mechanism.value,
        )

    console.print(table)

    # Demo with HIGH privacy
    console.print("\n[yellow]Using HIGH privacy level:")
    guard = create_privacy_guard(
        retriever,
        level=PrivacyLevel.HIGH,
        epsilon_budget=1.0,
    )

    query = np.random.randn(64).astype(np.float32)
    query = query / np.linalg.norm(query)

    console.print("\n  Performing 5 queries:")
    for i in range(5):
        result = guard.retrieve(query, top_k=3)
        console.print(f"    Query {i+1}: ε={result.epsilon_spent:.2f}, "
                     f"protected={result.query_protected}, "
                     f"docs={len(result.documents)}")

    # Generate privacy report
    console.print("\n[yellow]Privacy Report:")
    report = generate_privacy_report(guard)

    console.print(f"  Total queries: {report.total_queries}")
    console.print(f"  Total ε spent: {report.total_epsilon_spent:.2f}")
    console.print(f"  Avg ε per query: {report.average_epsilon_per_query:.3f}")
    console.print(f"  Budget utilization: {report.budget_utilization:.1%}")
    console.print(f"  Queries remaining: {report.queries_remaining}")

    if report.recommendations:
        console.print("\n  [green]Recommendations:")
        for rec in report.recommendations:
            console.print(f"    • {rec}")


def demo_privacy_utility_tradeoff():
    """Demonstrate privacy-utility tradeoff."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 5: Privacy-Utility Tradeoff")
    console.print("[bold cyan]=" * 60)

    # Create knowledge base and retriever
    kb = create_sample_knowledge_base()
    retriever = SimpleRetriever(metric="cosine")
    retriever.index(kb)

    query = np.random.randn(64).astype(np.float32)
    query = query / np.linalg.norm(query)

    epsilon_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    console.print("\n[yellow]Testing different epsilon values:")
    console.print("  Higher ε = less privacy but more utility")
    console.print("  Lower ε = more privacy but less utility\n")

    table = Table(title="Privacy-Utility Tradeoff")
    table.add_column("ε (epsilon)", justify="right", style="cyan")
    table.add_column("Noise Scale", justify="right")
    table.add_column("Order Changed", justify="center")
    table.add_column("Top Doc Score", justify="right")

    for eps in epsilon_values:
        budget = PrivacyBudgetManager(epsilon_budget=10.0)
        dp_retriever = DPRetriever(
            retriever=retriever,
            budget_manager=budget,
            epsilon_per_query=eps,
        )

        # Average over multiple runs
        order_changes = 0
        avg_score = 0
        avg_noise = 0
        num_runs = 10

        for _ in range(num_runs):
            result = dp_retriever.retrieve(query, top_k=3)
            if not result.original_order_preserved:
                order_changes += 1
            if result.documents:
                avg_score += result.documents[0][1]
            avg_noise += result.noise_scale

        table.add_row(
            f"{eps:.2f}",
            f"{avg_noise/num_runs:.4f}",
            f"{order_changes}/{num_runs}",
            f"{avg_score/num_runs:.4f}",
        )

    console.print(table)

    console.print("\n[green]Interpretation:")
    console.print("  • Low ε (0.01): High privacy, high noise, rankings may change")
    console.print("  • High ε (2.0): Low privacy, low noise, rankings stable")
    console.print("  • Choose based on your privacy requirements!")


def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold green]RAG-Shield Privacy Protection Demo",
        subtitle="Differential Privacy for RAG Systems"
    ))

    # Run demos
    demo_budget_management()
    demo_dp_retrieval()
    demo_query_sanitization()
    demo_privacy_guard()
    demo_privacy_utility_tradeoff()

    console.print("\n[bold green]Demo complete!")
    console.print("See the privacy module documentation for more details.\n")


if __name__ == "__main__":
    main()
