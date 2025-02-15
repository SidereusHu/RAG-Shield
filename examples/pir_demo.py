#!/usr/bin/env python3
"""
Private Information Retrieval (PIR) Demo

This demo shows how to use RAG-Shield's PIR protocols:
1. Single-server PIR (homomorphic encryption)
2. Multi-server PIR (XOR secret sharing)
3. PIR-based document retrieval

Run this demo:
    python examples/pir_demo.py
"""

import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# RAG-Shield imports
from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.pir import (
    # Protocols
    SingleServerPIR,
    MultiServerPIR,
    ThresholdPIR,
    # Secret Sharing
    XORSecretSharing,
    ShamirSecretSharing,
    # Retriever
    PIRRetriever,
    PIRMode,
    HybridPIRRetriever,
)

console = Console()


def demo_xor_secret_sharing():
    """Demonstrate XOR secret sharing."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 1: XOR Secret Sharing")
    console.print("[bold cyan]=" * 60)

    secret = b"Confidential Data!"
    console.print(f"\n[yellow]Original secret: [white]{secret.decode()}")

    # Split into 3 shares
    shares = XORSecretSharing.share(secret, 3)

    console.print("\n[yellow]Shares (each looks random):")
    for i, share in enumerate(shares):
        console.print(f"  Share {i+1}: {share.hex()[:40]}...")

    # Reconstruct
    reconstructed = XORSecretSharing.reconstruct(shares)
    console.print(f"\n[green]Reconstructed: {reconstructed.decode()}")

    console.print("\n[cyan]Key property: Any single share reveals nothing about the secret!")


def demo_shamir_secret_sharing():
    """Demonstrate Shamir secret sharing."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 2: Shamir (t,n) Secret Sharing")
    console.print("[bold cyan]=" * 60)

    secret = 12345
    console.print(f"\n[yellow]Secret: {secret}")
    console.print("[yellow]Creating (3,5)-threshold scheme: need 3 of 5 shares")

    shares = ShamirSecretSharing.share(secret, num_shares=5, threshold=3)

    console.print("\n[yellow]Shares (x, y) on polynomial:")
    for x, y in shares:
        console.print(f"  Point ({x}, {y})")

    # Reconstruct with different subsets
    console.print("\n[yellow]Reconstructing with different 3-share subsets:")

    subsets = [shares[:3], shares[1:4], shares[2:5]]
    for i, subset in enumerate(subsets):
        result = ShamirSecretSharing.reconstruct(subset)
        points = [s[0] for s in subset]
        console.print(f"  Using shares {points}: {result} {'[green]✓' if result == secret else '[red]✗'}")


def demo_single_server_pir():
    """Demonstrate single-server PIR."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 3: Single-Server PIR (Homomorphic Encryption)")
    console.print("[bold cyan]=" * 60)

    # Create database
    database = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    console.print(f"\n[yellow]Database: {database}")

    # Initialize PIR
    console.print("[yellow]Initializing PIR with homomorphic encryption...")
    pir = SingleServerPIR(database, key_bits=32)

    # Retrieve item
    target_index = 4
    console.print(f"\n[yellow]Retrieving item at index {target_index}...")
    console.print(f"  Expected value: {database[target_index]}")

    start = time.time()
    result = pir.retrieve(target_index)
    elapsed = time.time() - start

    console.print(f"  [green]Retrieved: {result.item}")
    console.print(f"  Time: {elapsed*1000:.2f}ms")
    console.print(f"\n[cyan]Server learned: NOTHING about which index was queried!")


def demo_multi_server_pir():
    """Demonstrate multi-server PIR."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 4: Multi-Server PIR (Information-Theoretic)")
    console.print("[bold cyan]=" * 60)

    database = [100, 200, 300, 400, 500]
    console.print(f"\n[yellow]Database: {database}")
    console.print("[yellow]Using 2 non-colluding servers")

    pir = MultiServerPIR(database, num_servers=2)

    console.print("\n[yellow]Retrieving each item:")
    table = Table(title="PIR Retrieval Results")
    table.add_column("Index", justify="center")
    table.add_column("Expected", justify="right")
    table.add_column("Retrieved", justify="right")
    table.add_column("Time (ms)", justify="right")

    for i in range(len(database)):
        start = time.time()
        result = pir.retrieve(i)
        elapsed = (time.time() - start) * 1000

        table.add_row(
            str(i),
            str(database[i]),
            str(result.item),
            f"{elapsed:.2f}",
        )

    console.print(table)

    console.print("\n[cyan]Security: Information-theoretic!")
    console.print("[cyan]Neither server learns which index was queried")
    console.print("[cyan](assuming servers don't collude)")


def demo_pir_retriever():
    """Demonstrate PIR-based RAG retrieval."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 5: PIR-based Document Retrieval")
    console.print("[bold cyan]=" * 60)

    # Create knowledge base
    kb = KnowledgeBase()
    docs = [
        ("Medical: Treatment for headaches includes rest and pain relievers.", "medical"),
        ("Legal: Contract law governs agreements between parties.", "legal"),
        ("Finance: Stock markets fluctuate based on economic indicators.", "finance"),
        ("Tech: Machine learning uses data to train predictive models.", "tech"),
        ("Medical: Diabetes management requires monitoring blood sugar.", "medical"),
        ("Legal: Criminal law deals with offenses against the state.", "legal"),
        ("Finance: Bonds are fixed-income investment securities.", "finance"),
        ("Tech: Cloud computing provides on-demand computing resources.", "tech"),
    ]

    for i, (content, category) in enumerate(docs):
        embedding = np.random.randn(64).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        doc = Document(
            doc_id=f"doc_{i}",
            content=content,
            embedding=embedding.tolist(),
            metadata={"category": category},
        )
        kb.add_document(doc)

    console.print(f"\n[yellow]Created knowledge base with {kb.size()} documents")

    # Create PIR retriever
    retriever = PIRRetriever(kb, mode=PIRMode.MULTI_SERVER, num_servers=2)

    # Query
    query = np.random.randn(64).astype(np.float32)
    console.print("\n[yellow]Performing private retrieval...")

    result = retriever.retrieve(query, top_k=3)

    console.print(f"\n[green]Retrieved {len(result.documents)} documents privately:")
    for doc, score in result.documents:
        console.print(f"  - [{doc.metadata.get('category', 'N/A')}] {doc.content[:50]}...")

    console.print(f"\n[cyan]PIR time: {result.pir_time*1000:.2f}ms")
    console.print("[cyan]Server learned: Only that 3 documents were fetched, NOT which ones!")


def demo_comparison():
    """Compare different PIR schemes."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 6: PIR Scheme Comparison")
    console.print("[bold cyan]=" * 60)

    database = list(range(100, 200))  # 100 items
    console.print(f"\n[yellow]Database: {len(database)} items")

    table = Table(title="PIR Scheme Comparison")
    table.add_column("Scheme", style="cyan")
    table.add_column("Security", justify="center")
    table.add_column("Servers", justify="center")
    table.add_column("Avg Time (ms)", justify="right")

    # Single-server PIR
    pir_single = SingleServerPIR(database, key_bits=32)
    times = []
    for _ in range(5):
        start = time.time()
        pir_single.retrieve(50)
        times.append((time.time() - start) * 1000)
    table.add_row("Single-Server (HE)", "Computational", "1", f"{np.mean(times):.2f}")

    # Multi-server PIR (2 servers)
    pir_multi_2 = MultiServerPIR(database, num_servers=2)
    times = []
    for _ in range(5):
        start = time.time()
        pir_multi_2.retrieve(50)
        times.append((time.time() - start) * 1000)
    table.add_row("Multi-Server (XOR)", "Info-Theoretic", "2", f"{np.mean(times):.2f}")

    # Multi-server PIR (3 servers)
    pir_multi_3 = MultiServerPIR(database, num_servers=3)
    times = []
    for _ in range(5):
        start = time.time()
        pir_multi_3.retrieve(50)
        times.append((time.time() - start) * 1000)
    table.add_row("Multi-Server (XOR)", "Info-Theoretic", "3", f"{np.mean(times):.2f}")

    console.print(table)

    console.print("\n[yellow]Trade-offs:")
    console.print("  - Single-server: No trust assumptions, but computationally expensive")
    console.print("  - Multi-server: Fast, but requires non-colluding servers")


def demo_hybrid_retrieval():
    """Demonstrate hybrid DP + PIR retrieval."""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]Demo 7: Hybrid DP + PIR Retrieval")
    console.print("[bold cyan]=" * 60)

    # Create knowledge base
    kb = KnowledgeBase()
    for i in range(20):
        embedding = np.random.randn(64).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        doc = Document(
            doc_id=f"doc_{i}",
            content=f"Document {i} content here.",
            embedding=embedding.tolist(),
        )
        kb.add_document(doc)

    console.print(f"[yellow]Knowledge base: {kb.size()} documents")

    # Create hybrid retriever
    retriever = HybridPIRRetriever(kb, epsilon=0.5, pir_mode=PIRMode.MULTI_SERVER)

    query = np.random.randn(64).astype(np.float32)

    console.print("\n[yellow]Without DP noise:")
    result = retriever.retrieve(query, top_k=3, add_noise=False)
    for doc, score in result.documents:
        console.print(f"  {doc.doc_id}: score={score:.4f}")

    console.print("\n[yellow]With DP noise (ε=0.5):")
    result = retriever.retrieve(query, top_k=3, add_noise=True)
    for doc, score in result.documents:
        console.print(f"  {doc.doc_id}: score={score:.4f} (noisy)")

    console.print("\n[cyan]Hybrid protection:")
    console.print("  1. DP noise protects similarity patterns")
    console.print("  2. PIR protects which documents are fetched")


def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold green]Private Information Retrieval (PIR) Demo",
        subtitle="Cryptographic Privacy for RAG Systems"
    ))

    demo_xor_secret_sharing()
    demo_shamir_secret_sharing()
    demo_single_server_pir()
    demo_multi_server_pir()
    demo_pir_retriever()
    demo_comparison()
    demo_hybrid_retrieval()

    console.print("\n[bold green]Demo complete!")
    console.print("PIR enables true cryptographic privacy for document retrieval.\n")


if __name__ == "__main__":
    main()
