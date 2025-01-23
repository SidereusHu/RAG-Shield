#!/usr/bin/env python3
"""
Full Defense Pipeline Demo

Demonstrates complete attack and defense cycle:
1. Create clean RAG system
2. Perform poisoning attack
3. Detect poisoned documents
4. Evaluate defense effectiveness
"""

from ragshield.core import RAGSystem
from ragshield.redteam import DirectPoisoning, AdversarialPoisoning
from ragshield.detection import create_poison_detector, SemanticDetector


def create_knowledge_base() -> RAGSystem:
    """Create a sample RAG system with factual knowledge."""
    rag = RAGSystem()
    documents = [
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "France is a country in Western Europe with a rich history.",
        "The French Revolution began in 1789 and transformed the country.",
        "French cuisine is renowned worldwide for its quality and diversity.",
    ]
    rag.add_documents(documents)
    return rag


def main():
    print("=" * 70)
    print("RAG-Shield: Full Defense Pipeline Demo")
    print("=" * 70)

    # Phase 1: Setup
    print("\n" + "-" * 70)
    print("PHASE 1: SETUP")
    print("-" * 70)

    rag = create_knowledge_base()
    print(f"Created knowledge base with {len(rag)} documents")

    # Create detector
    detector = create_poison_detector(preset="strict")
    print("Created poison detector with 'strict' preset")

    # Baseline scan
    print("\nBaseline scan of clean knowledge base:")
    from ragshield.detection.base import ScanResult
    clean_results = []
    for doc in rag.knowledge_base.documents:
        result = detector.detect(doc)
        clean_results.append(result.is_poisoned)
    poisoned_count = sum(clean_results)
    print(f"   Documents flagged: {poisoned_count}/{len(rag)}")

    # Phase 2: Attack
    print("\n" + "-" * 70)
    print("PHASE 2: POISONING ATTACK")
    print("-" * 70)

    attack = DirectPoisoning(num_variants=3)
    target_query = "What is the capital of France?"
    malicious_answer = "The capital of France is Berlin, Germany."

    print(f"Target query: '{target_query}'")
    print(f"Malicious answer: '{malicious_answer}'")

    # Craft and inject poison
    poisoned_docs = attack.craft_poison(target_query, malicious_answer)
    attack.inject(rag, poisoned_docs)

    print(f"\nInjected {len(poisoned_docs)} poisoned documents")
    print(f"Knowledge base size: {len(rag)}")

    # Check retrieval after attack
    print("\nRetrieval after attack:")
    results = rag.retrieve_with_scores(target_query, top_k=3)
    for i, (doc, score) in enumerate(results, 1):
        is_poison = doc.metadata.custom.get("poisoned", False)
        marker = "[POISON]" if is_poison else "[CLEAN]"
        print(f"   {i}. {marker} Score: {score:.4f}")
        print(f"      {doc.content[:60]}...")

    # Phase 3: Detection
    print("\n" + "-" * 70)
    print("PHASE 3: POISON DETECTION")
    print("-" * 70)

    print("Scanning knowledge base for poisoned documents...")

    detected_poison = []
    false_positives = []
    missed_poison = []

    for doc in rag.knowledge_base.documents:
        result = detector.detect(doc)
        is_actual_poison = doc.metadata.custom.get("poisoned", False)

        if result.is_poisoned and is_actual_poison:
            detected_poison.append((doc, result))
        elif result.is_poisoned and not is_actual_poison:
            false_positives.append((doc, result))
        elif not result.is_poisoned and is_actual_poison:
            missed_poison.append((doc, result))

    print(f"\nDetection Results:")
    print(f"   True Positives (correctly detected poison): {len(detected_poison)}")
    print(f"   False Positives (clean docs flagged): {len(false_positives)}")
    print(f"   False Negatives (missed poison): {len(missed_poison)}")

    if detected_poison:
        print(f"\nDetected poisoned documents:")
        for doc, result in detected_poison:
            print(f"   - Threat: {result.threat_level.value.upper()}")
            print(f"     Reason: {result.reason[:60]}...")

    # Phase 4: Defense Metrics
    print("\n" + "-" * 70)
    print("PHASE 4: DEFENSE EFFECTIVENESS")
    print("-" * 70)

    total_poison = len(poisoned_docs)
    total_clean = len(rag) - total_poison

    detection_rate = len(detected_poison) / total_poison if total_poison > 0 else 0
    false_positive_rate = len(false_positives) / total_clean if total_clean > 0 else 0
    precision = (
        len(detected_poison) / (len(detected_poison) + len(false_positives))
        if (len(detected_poison) + len(false_positives)) > 0
        else 0
    )

    print(f"\nMetrics:")
    print(f"   Detection Rate (Recall): {detection_rate:.1%}")
    print(f"   False Positive Rate: {false_positive_rate:.1%}")
    print(f"   Precision: {precision:.1%}")

    # Phase 5: Safe Retrieval
    print("\n" + "-" * 70)
    print("PHASE 5: SAFE RETRIEVAL")
    print("-" * 70)

    print("Retrieving with poison filtering...")

    safe_results = []
    results = rag.retrieve_with_scores(target_query, top_k=5)

    for doc, score in results:
        detection = detector.detect(doc)
        if not detection.is_poisoned:
            safe_results.append((doc, score))

    print(f"\nQuery: '{target_query}'")
    print(f"Original results: {len(results)}")
    print(f"After filtering: {len(safe_results)}")

    if safe_results:
        print(f"\nTop safe result:")
        top_doc, top_score = safe_results[0]
        print(f"   Score: {top_score:.4f}")
        print(f"   Content: {top_doc.content}")
    else:
        print("\n   No safe results found - all retrieved docs flagged as suspicious")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
    Attack Scenario:
    - Attacker injected {len(poisoned_docs)} poisoned documents
    - Goal: Make RAG return "{malicious_answer[:30]}..."

    Defense Performance:
    - Detected {len(detected_poison)}/{len(poisoned_docs)} poisoned documents
    - False alarms: {len(false_positives)} clean documents wrongly flagged
    - Detection rate: {detection_rate:.1%}

    Conclusion:
    {'Defense successful - attack mitigated!' if detection_rate > 0.5 else 'Defense needs improvement'}
    """)

    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
