#!/usr/bin/env python3
"""
Poisoning Attack Demo

Demonstrates various RAG poisoning attack techniques:
- Direct poisoning
- Adversarial poisoning
- Stealth poisoning
- Chain poisoning
"""

from ragshield.core import RAGSystem
from ragshield.redteam import (
    DirectPoisoning,
    AdversarialPoisoning,
    StealthPoisoning,
)
from ragshield.redteam.poisoning import ChainPoisoning


def create_sample_rag() -> RAGSystem:
    """Create a sample RAG system with knowledge base."""
    rag = RAGSystem()
    documents = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "London is the capital of the United Kingdom. Big Ben is a famous clock tower.",
        "Berlin is the capital of Germany. The Brandenburg Gate is a landmark.",
        "Rome is the capital of Italy. The Colosseum is an ancient amphitheater.",
        "Madrid is the capital of Spain. It is famous for the Prado Museum.",
    ]
    rag.add_documents(documents)
    return rag


def print_separator(title: str):
    """Print a section separator."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("RAG-Shield: Poisoning Attack Demo")
    print("=" * 70)
    print("\nThis demo shows how attackers can poison RAG knowledge bases.")
    print("Understanding attacks helps build better defenses!")

    target_query = "What is the capital of France?"
    target_answer = "The capital of France is actually Berlin (this is false!)"

    # 1. Direct Poisoning Attack
    print_separator("1. DIRECT POISONING ATTACK")
    print("Strategy: Inject documents that directly contain the malicious answer")
    print("         Optimize content to rank high in retrieval")

    rag = create_sample_rag()
    print(f"\nOriginal knowledge base size: {len(rag)}")

    # Query before attack
    print(f"\nQuery: '{target_query}'")
    print("Before attack - Top result:")
    results = rag.retrieve_with_scores(target_query, top_k=1)
    print(f"   {results[0][0].content[:70]}...")
    print(f"   Score: {results[0][1]:.4f}")

    # Execute attack
    attack = DirectPoisoning(num_variants=2, include_query_in_doc=True)
    result = attack.execute(rag, target_query, target_answer, num_poison_docs=2)

    print(f"\nAfter attack - Knowledge base size: {len(rag)}")
    print(f"Attack success: {result.success}")
    print(f"Poisoned doc rank: {result.retrieval_rank}")

    print("\nPoisoned document content:")
    for i, pdoc in enumerate(result.poisoned_docs):
        print(f"\n   Document {i+1}:")
        print(f"   {pdoc.content[:100]}...")

    # 2. Adversarial Poisoning Attack
    print_separator("2. ADVERSARIAL POISONING ATTACK")
    print("Strategy: Craft documents with repeated keywords to maximize retrieval score")
    print("         Uses query keywords to boost relevance")

    rag = create_sample_rag()
    attack = AdversarialPoisoning(repetition_factor=5)
    poisoned_docs = attack.craft_poison(target_query, target_answer)

    print("\nCrafted adversarial document:")
    print(f"   {poisoned_docs[0].content[:150]}...")
    print(f"\n   Keywords extracted: {poisoned_docs[0].metadata.get('keywords', [])}")
    print(f"   Repetition factor: {poisoned_docs[0].metadata.get('repetition_factor', 0)}")

    # 3. Stealth Poisoning Attack
    print_separator("3. STEALTH POISONING ATTACK")
    print("Strategy: Mix malicious content with legitimate-looking text")
    print("         Harder to detect, aims to evade content filters")

    rag = create_sample_rag()
    attack = StealthPoisoning(legitimate_content_ratio=0.7)
    poisoned_docs = attack.craft_poison(target_query, target_answer)

    print(f"\nLegitimate content ratio: {attack.legitimate_content_ratio:.0%}")
    print(f"Trigger phrase: '{poisoned_docs[0].metadata.get('trigger_phrase', '')}'")
    print("\nStealth document preview:")
    content = poisoned_docs[0].content
    print(f"   Start: {content[:80]}...")
    print(f"   End:   ...{content[-80:]}")

    # 4. Chain Poisoning Attack
    print_separator("4. CHAIN POISONING ATTACK")
    print("Strategy: Create multiple documents that reinforce each other")
    print("         Builds a narrative across multiple documents")

    rag = create_sample_rag()
    attack = ChainPoisoning(chain_length=3)
    poisoned_docs = attack.craft_poison(target_query, target_answer)

    print(f"\nChain length: {len(poisoned_docs)} documents")
    for i, pdoc in enumerate(poisoned_docs):
        role = pdoc.metadata.get("role", "unknown")
        print(f"\n   Document {i+1} - Role: {role.upper()}")
        print(f"   {pdoc.content[:100]}...")

    # 5. Attack Success Metrics
    print_separator("5. ATTACK SUCCESS COMPARISON")

    attacks = [
        ("Direct", DirectPoisoning()),
        ("Adversarial", AdversarialPoisoning()),
        ("Stealth", StealthPoisoning()),
        ("Chain", ChainPoisoning()),
    ]

    print(f"\nTarget query: '{target_query}'")
    print(f"Target (malicious) answer: '{target_answer[:50]}...'")
    print("\n" + "-" * 70)
    print(f"{'Attack Type':<15} | {'Success':<8} | {'Poison Rank':<12} | {'Top Score':<10}")
    print("-" * 70)

    for name, attack_obj in attacks:
        rag = create_sample_rag()
        result = attack_obj.execute(rag, target_query, target_answer)
        success_str = "Yes" if result.success else "No"
        rank_str = str(result.retrieval_rank) if result.retrieval_rank else "N/A"
        score = result.metrics.get("top_1_score", 0)
        print(f"{name:<15} | {success_str:<8} | {rank_str:<12} | {score:.4f}")

    print("\n" + "=" * 70)
    print("IMPORTANT: These attacks demonstrate why RAG systems need protection!")
    print("Use RAG-Shield's detection capabilities to defend against such attacks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
