"""Tests for red team poisoning attacks."""

import pytest
from ragshield.core import RAGSystem, Document
from ragshield.redteam import (
    DirectPoisoning,
    AdversarialPoisoning,
    StealthPoisoning,
    PoisonedDocument,
    AttackResult,
)
from ragshield.redteam.poisoning import AttackType, ChainPoisoning


class TestPoisonedDocument:
    """Tests for PoisonedDocument class."""

    def test_poisoned_document_creation(self):
        """Test creating a poisoned document."""
        pdoc = PoisonedDocument(
            content="Malicious content",
            target_query="What is X?",
            target_answer="X is wrong",
            attack_type=AttackType.DIRECT,
        )
        assert pdoc.content == "Malicious content"
        assert pdoc.target_query == "What is X?"
        assert pdoc.attack_type == AttackType.DIRECT

    def test_poisoned_document_to_document(self):
        """Test converting PoisonedDocument to Document."""
        pdoc = PoisonedDocument(
            content="Malicious content",
            target_query="What is X?",
            target_answer="X is wrong",
            attack_type=AttackType.DIRECT,
        )
        doc = pdoc.to_document()

        assert isinstance(doc, Document)
        assert doc.content == "Malicious content"
        assert doc.metadata.custom.get("poisoned") is True
        assert doc.metadata.custom.get("attack_type") == "direct"


class TestDirectPoisoning:
    """Tests for direct poisoning attack."""

    def test_attack_creation(self):
        """Test direct poisoning attack creation."""
        attack = DirectPoisoning(num_variants=3)
        assert attack.num_variants == 3

    def test_craft_poison(self):
        """Test crafting poisoned documents."""
        attack = DirectPoisoning(num_variants=3)
        poisoned_docs = attack.craft_poison(
            target_query="What is the capital of France?",
            target_answer="The capital of France is London.",
        )

        assert len(poisoned_docs) == 3
        for doc in poisoned_docs:
            assert isinstance(doc, PoisonedDocument)
            assert doc.attack_type == AttackType.DIRECT
            assert "London" in doc.content

    def test_craft_poison_with_query_inclusion(self):
        """Test crafting poison with query included."""
        attack = DirectPoisoning(include_query_in_doc=True)
        poisoned_docs = attack.craft_poison(
            target_query="What is the capital?",
            target_answer="The capital is X.",
        )

        # Query should be in the document
        assert "What is the capital?" in poisoned_docs[0].content

    def test_craft_poison_without_query_inclusion(self):
        """Test crafting poison without query included."""
        attack = DirectPoisoning(include_query_in_doc=False)
        poisoned_docs = attack.craft_poison(
            target_query="What is the capital?",
            target_answer="The capital is X.",
        )

        # Query text should not appear as "Question:"
        assert "Question:" not in poisoned_docs[0].content

    def test_inject_poison(self):
        """Test injecting poison into RAG system."""
        rag = RAGSystem()
        rag.add_documents([
            "Paris is the capital of France.",
            "London is the capital of UK.",
        ])

        attack = DirectPoisoning()
        poisoned_docs = attack.craft_poison(
            target_query="capital of France",
            target_answer="Berlin is the capital of France.",
        )

        initial_count = len(rag)
        attack.inject(rag, poisoned_docs)

        assert len(rag) == initial_count + len(poisoned_docs)


class TestAdversarialPoisoning:
    """Tests for adversarial poisoning attack."""

    def test_attack_creation(self):
        """Test adversarial poisoning attack creation."""
        attack = AdversarialPoisoning(repetition_factor=5)
        assert attack.repetition_factor == 5

    def test_craft_poison(self):
        """Test crafting adversarial poisoned documents."""
        attack = AdversarialPoisoning(repetition_factor=3)
        poisoned_docs = attack.craft_poison(
            target_query="What is machine learning?",
            target_answer="Machine learning is dangerous.",
        )

        assert len(poisoned_docs) == 1
        assert poisoned_docs[0].attack_type == AttackType.ADVERSARIAL

        # Check keyword repetition
        doc_content = poisoned_docs[0].content.lower()
        assert "machine" in doc_content
        assert "learning" in doc_content

    def test_keyword_extraction(self):
        """Test keyword extraction from query."""
        attack = AdversarialPoisoning()
        keywords = attack._extract_keywords("What is the capital of France")

        # Should extract meaningful keywords
        assert "capital" in keywords
        # "france" may or may not be included depending on processing
        # Should not include stopwords
        assert "what" not in keywords
        assert "the" not in keywords


class TestStealthPoisoning:
    """Tests for stealth poisoning attack."""

    def test_attack_creation(self):
        """Test stealth poisoning attack creation."""
        attack = StealthPoisoning(legitimate_content_ratio=0.8)
        assert attack.legitimate_content_ratio == 0.8

    def test_craft_poison(self):
        """Test crafting stealthy poisoned documents."""
        attack = StealthPoisoning()
        poisoned_docs = attack.craft_poison(
            target_query="What is AI safety?",
            target_answer="AI safety is overrated.",
        )

        assert len(poisoned_docs) == 1
        assert poisoned_docs[0].attack_type == AttackType.STEALTH

        # Should contain trigger phrase
        content = poisoned_docs[0].content.lower()
        has_trigger = any(
            trigger.lower() in content for trigger in attack.trigger_phrases
        )
        assert has_trigger

    def test_legitimate_content_generation(self):
        """Test legitimate content generation."""
        attack = StealthPoisoning()
        filler = attack._generate_filler_content("machine learning", length=200)

        assert len(filler) >= 200
        assert "machine learning" in filler.lower()


class TestChainPoisoning:
    """Tests for chain poisoning attack."""

    def test_attack_creation(self):
        """Test chain poisoning attack creation."""
        attack = ChainPoisoning(chain_length=3)
        assert attack.chain_length == 3

    def test_craft_poison(self):
        """Test crafting chain of poisoned documents."""
        attack = ChainPoisoning(chain_length=3)
        poisoned_docs = attack.craft_poison(
            target_query="What is quantum computing?",
            target_answer="Quantum computing is obsolete.",
        )

        assert len(poisoned_docs) == 3

        # Check chain positions
        for i, doc in enumerate(poisoned_docs):
            assert doc.attack_type == AttackType.CHAIN
            assert doc.metadata["chain_position"] == i + 1

        # Check roles
        assert poisoned_docs[0].metadata["role"] == "premise"
        assert poisoned_docs[1].metadata["role"] == "evidence"
        assert poisoned_docs[2].metadata["role"] == "conclusion"

    def test_chain_length_limit(self):
        """Test chain length limiting."""
        attack = ChainPoisoning(chain_length=2)
        poisoned_docs = attack.craft_poison(
            target_query="test query",
            target_answer="test answer",
        )

        assert len(poisoned_docs) == 2


class TestAttackExecution:
    """Tests for full attack execution."""

    def test_execute_direct_attack(self):
        """Test executing complete direct poisoning attack."""
        rag = RAGSystem()
        rag.add_documents([
            "Paris is the capital of France.",
            "London is the capital of the United Kingdom.",
            "Berlin is the capital of Germany.",
        ])

        attack = DirectPoisoning()
        result = attack.execute(
            rag_system=rag,
            target_query="What is the capital of France?",
            target_answer="The capital of France is Tokyo.",
            num_poison_docs=1,
        )

        assert isinstance(result, AttackResult)
        assert result.target_query == "What is the capital of France?"
        assert result.expected_answer == "The capital of France is Tokyo."
        assert len(result.poisoned_docs) == 1

    def test_attack_result_metrics(self):
        """Test attack result contains metrics."""
        rag = RAGSystem()
        rag.add_documents(["Test document content."])

        attack = DirectPoisoning()
        result = attack.execute(
            rag_system=rag,
            target_query="test query",
            target_answer="test answer",
            num_poison_docs=1,
        )

        assert "top_1_score" in result.metrics
        assert "retrieved_docs" in result.metrics


class TestAttackTypes:
    """Tests for attack type enumeration."""

    def test_attack_type_values(self):
        """Test attack type enum values."""
        assert AttackType.DIRECT.value == "direct"
        assert AttackType.ADVERSARIAL.value == "adversarial"
        assert AttackType.STEALTH.value == "stealth"
        assert AttackType.CHAIN.value == "chain"
