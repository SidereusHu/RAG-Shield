#!/usr/bin/env python3
"""
Basic RAG System Demo

Demonstrates the core RAG system functionality:
- Creating a RAG system
- Adding documents
- Performing retrieval
"""

from ragshield.core import RAGSystem, Document


def main():
    print("=" * 60)
    print("RAG-Shield: Basic RAG System Demo")
    print("=" * 60)

    # Create RAG system
    print("\n1. Creating RAG system...")
    rag = RAGSystem()
    print(f"   RAG system created: {rag}")

    # Add documents
    print("\n2. Adding documents to knowledge base...")
    documents = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "London is the capital of the United Kingdom. Big Ben is a famous landmark.",
        "Berlin is the capital of Germany. The Brandenburg Gate is iconic.",
        "Tokyo is the capital of Japan. It is famous for its technology and culture.",
        "Beijing is the capital of China. The Great Wall is nearby.",
    ]

    for doc in documents:
        rag.add_document(doc)
        print(f"   + Added: {doc[:50]}...")

    print(f"\n   Total documents: {len(rag)}")

    # Perform retrieval
    print("\n3. Performing retrieval...")
    queries = [
        "What is the capital of France?",
        "Tell me about Japanese culture",
        "Famous landmarks in Europe",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = rag.retrieve_with_scores(query, top_k=2)
        for i, (doc, score) in enumerate(results, 1):
            preview = doc.content[:60] + "..." if len(doc.content) > 60 else doc.content
            print(f"   [{i}] Score: {score:.4f} | {preview}")

    # System stats
    print("\n4. System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
