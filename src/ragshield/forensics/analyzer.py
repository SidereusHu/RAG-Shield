"""Attack pattern analysis for RAG systems.

Analyzes documents to identify attack patterns, fingerprints,
and characteristics of poisoning attacks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import hashlib
from collections import Counter

from ragshield.core.document import Document
from ragshield.redteam.poisoning import AttackType


class AttackPattern(Enum):
    """Known attack patterns."""

    KEYWORD_STUFFING = "keyword_stuffing"
    QUERY_INJECTION = "query_injection"
    AUTHORITY_MIMICKING = "authority_mimicking"
    SEMANTIC_MANIPULATION = "semantic_manipulation"
    CHAIN_CORRELATION = "chain_correlation"
    STEALTH_EMBEDDING = "stealth_embedding"
    REPETITION_ATTACK = "repetition_attack"
    TEMPLATE_BASED = "template_based"
    UNKNOWN = "unknown"


@dataclass
class PatternMatch:
    """Result of pattern matching.

    Attributes:
        pattern: Matched attack pattern
        confidence: Confidence of the match (0-1)
        evidence: Evidence supporting the match
        indicators: Specific indicators found
    """

    pattern: AttackPattern
    confidence: float
    evidence: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackFingerprint:
    """Fingerprint identifying an attack's characteristics.

    Attributes:
        content_hash: Hash of attack content
        structural_hash: Hash of content structure
        patterns_detected: Attack patterns found
        vocabulary_signature: Characteristic vocabulary
        style_metrics: Writing style metrics
    """

    content_hash: str
    structural_hash: str
    patterns_detected: List[AttackPattern]
    vocabulary_signature: Set[str]
    style_metrics: Dict[str, float]

    def similarity(self, other: "AttackFingerprint") -> float:
        """Compute similarity with another fingerprint.

        Args:
            other: Fingerprint to compare with

        Returns:
            Similarity score (0-1)
        """
        # Hash match (exact or structural)
        if self.content_hash == other.content_hash:
            return 1.0
        if self.structural_hash == other.structural_hash:
            return 0.9

        # Pattern overlap
        pattern_overlap = len(
            set(self.patterns_detected) & set(other.patterns_detected)
        ) / max(len(self.patterns_detected), len(other.patterns_detected), 1)

        # Vocabulary overlap (Jaccard similarity)
        vocab_overlap = len(self.vocabulary_signature & other.vocabulary_signature) / max(
            len(self.vocabulary_signature | other.vocabulary_signature), 1
        )

        # Style similarity
        style_sim = 0.0
        common_keys = set(self.style_metrics.keys()) & set(other.style_metrics.keys())
        if common_keys:
            diffs = []
            for key in common_keys:
                max_val = max(
                    abs(self.style_metrics[key]), abs(other.style_metrics[key]), 1
                )
                diff = abs(self.style_metrics[key] - other.style_metrics[key]) / max_val
                diffs.append(1 - min(diff, 1))
            style_sim = sum(diffs) / len(diffs)

        # Weighted combination
        return 0.3 * pattern_overlap + 0.4 * vocab_overlap + 0.3 * style_sim


@dataclass
class AnalysisResult:
    """Result of attack pattern analysis.

    Attributes:
        doc_id: Document identifier
        detected_patterns: List of pattern matches
        fingerprint: Attack fingerprint
        attack_type_estimate: Estimated attack type
        confidence: Overall confidence
        recommendations: Recommended actions
    """

    doc_id: str
    detected_patterns: List[PatternMatch]
    fingerprint: AttackFingerprint
    attack_type_estimate: Optional[AttackType]
    confidence: float
    recommendations: List[str] = field(default_factory=list)


class AttackPatternAnalyzer:
    """Analyzes documents for attack patterns.

    Provides forensic analysis of suspected poisoning attacks.
    """

    # Common authority phrases used in attacks
    AUTHORITY_PHRASES = [
        "according to official",
        "verified information",
        "expert consensus",
        "it is well established",
        "authoritative sources",
        "confirmed by",
        "officially stated",
        "research confirms",
        "scientifically proven",
        "the correct answer is",
    ]

    # Template patterns
    TEMPLATE_PATTERNS = [
        r"Question:\s*.*\n+.*Answer:",
        r"Q:\s*.*\n+A:",
        r"According to .*, (the answer is|we know that)",
        r"The (correct|right|true) answer is:",
    ]

    def __init__(
        self,
        min_confidence: float = 0.3,
        keyword_repetition_threshold: int = 3,
    ):
        """Initialize analyzer.

        Args:
            min_confidence: Minimum confidence for pattern match
            keyword_repetition_threshold: Threshold for keyword stuffing detection
        """
        self.min_confidence = min_confidence
        self.keyword_repetition_threshold = keyword_repetition_threshold
        self._fingerprint_db: Dict[str, AttackFingerprint] = {}

    def analyze(self, document: Document) -> AnalysisResult:
        """Analyze a document for attack patterns.

        Args:
            document: Document to analyze

        Returns:
            Analysis result
        """
        content = document.content
        patterns = []

        # Check for keyword stuffing
        pattern = self._detect_keyword_stuffing(content)
        if pattern:
            patterns.append(pattern)

        # Check for query injection
        pattern = self._detect_query_injection(content)
        if pattern:
            patterns.append(pattern)

        # Check for authority mimicking
        pattern = self._detect_authority_mimicking(content)
        if pattern:
            patterns.append(pattern)

        # Check for template-based attacks
        pattern = self._detect_template_based(content)
        if pattern:
            patterns.append(pattern)

        # Check for repetition attack
        pattern = self._detect_repetition(content)
        if pattern:
            patterns.append(pattern)

        # Generate fingerprint
        fingerprint = self._generate_fingerprint(document, patterns)

        # Estimate attack type
        attack_type = self._estimate_attack_type(patterns)

        # Calculate overall confidence
        confidence = (
            max(p.confidence for p in patterns) if patterns else 0.0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, confidence)

        return AnalysisResult(
            doc_id=document.doc_id,
            detected_patterns=patterns,
            fingerprint=fingerprint,
            attack_type_estimate=attack_type,
            confidence=confidence,
            recommendations=recommendations,
        )

    def find_similar_attacks(
        self, fingerprint: AttackFingerprint, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Find similar attacks in fingerprint database.

        Args:
            fingerprint: Fingerprint to search for
            threshold: Similarity threshold

        Returns:
            List of (doc_id, similarity) tuples
        """
        similar = []
        for doc_id, stored_fp in self._fingerprint_db.items():
            sim = fingerprint.similarity(stored_fp)
            if sim >= threshold:
                similar.append((doc_id, sim))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def store_fingerprint(self, doc_id: str, fingerprint: AttackFingerprint) -> None:
        """Store a fingerprint in the database.

        Args:
            doc_id: Document identifier
            fingerprint: Fingerprint to store
        """
        self._fingerprint_db[doc_id] = fingerprint

    def cluster_attacks(
        self, documents: List[Document], similarity_threshold: float = 0.6
    ) -> List[List[str]]:
        """Cluster documents by attack similarity.

        Args:
            documents: Documents to cluster
            similarity_threshold: Threshold for clustering

        Returns:
            List of clusters (each cluster is list of doc_ids)
        """
        # Analyze all documents
        fingerprints = {}
        for doc in documents:
            result = self.analyze(doc)
            fingerprints[doc.doc_id] = result.fingerprint

        # Simple clustering based on similarity
        clusters: List[Set[str]] = []

        for doc_id, fp in fingerprints.items():
            assigned = False
            for cluster in clusters:
                # Check similarity with first member
                first_id = next(iter(cluster))
                if fp.similarity(fingerprints[first_id]) >= similarity_threshold:
                    cluster.add(doc_id)
                    assigned = True
                    break

            if not assigned:
                clusters.append({doc_id})

        return [list(c) for c in clusters]

    def _detect_keyword_stuffing(self, content: str) -> Optional[PatternMatch]:
        """Detect keyword stuffing pattern.

        Args:
            content: Document content

        Returns:
            Pattern match or None
        """
        words = content.lower().split()
        word_counts = Counter(words)

        # Find excessively repeated words
        stuffed_words = [
            word
            for word, count in word_counts.items()
            if count >= self.keyword_repetition_threshold and len(word) > 3
        ]

        if stuffed_words:
            max_repetition = max(word_counts[w] for w in stuffed_words)
            confidence = min(max_repetition / 10, 1.0)

            if confidence >= self.min_confidence:
                return PatternMatch(
                    pattern=AttackPattern.KEYWORD_STUFFING,
                    confidence=confidence,
                    evidence=[f"'{w}' repeated {word_counts[w]} times" for w in stuffed_words[:5]],
                    indicators={"stuffed_words": stuffed_words, "max_count": max_repetition},
                )

        return None

    def _detect_query_injection(self, content: str) -> Optional[PatternMatch]:
        """Detect query injection pattern.

        Args:
            content: Document content

        Returns:
            Pattern match or None
        """
        # Look for question-answer format
        qa_patterns = [
            r"question:\s*(.+?)[\n.]",
            r"q:\s*(.+?)[\n.]",
            r"what is\s+(.+?)\?",
            r"how (do|does|can|to)\s+(.+?)\?",
        ]

        matches = []
        for pattern in qa_patterns:
            found = re.findall(pattern, content.lower())
            if found:
                matches.extend(found if isinstance(found[0], str) else [f[0] for f in found])

        if matches:
            confidence = min(len(matches) * 0.3, 0.9)
            if confidence >= self.min_confidence:
                return PatternMatch(
                    pattern=AttackPattern.QUERY_INJECTION,
                    confidence=confidence,
                    evidence=[f"Query pattern found: '{m[:50]}...'" for m in matches[:3]],
                    indicators={"query_count": len(matches)},
                )

        return None

    def _detect_authority_mimicking(self, content: str) -> Optional[PatternMatch]:
        """Detect authority mimicking pattern.

        Args:
            content: Document content

        Returns:
            Pattern match or None
        """
        content_lower = content.lower()
        found_phrases = []

        for phrase in self.AUTHORITY_PHRASES:
            if phrase in content_lower:
                found_phrases.append(phrase)

        if found_phrases:
            confidence = min(len(found_phrases) * 0.25, 0.95)
            if confidence >= self.min_confidence:
                return PatternMatch(
                    pattern=AttackPattern.AUTHORITY_MIMICKING,
                    confidence=confidence,
                    evidence=[f"Authority phrase: '{p}'" for p in found_phrases],
                    indicators={"phrase_count": len(found_phrases)},
                )

        return None

    def _detect_template_based(self, content: str) -> Optional[PatternMatch]:
        """Detect template-based attack pattern.

        Args:
            content: Document content

        Returns:
            Pattern match or None
        """
        matches = []
        for pattern in self.TEMPLATE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(pattern)

        if matches:
            confidence = min(len(matches) * 0.4, 0.9)
            if confidence >= self.min_confidence:
                return PatternMatch(
                    pattern=AttackPattern.TEMPLATE_BASED,
                    confidence=confidence,
                    evidence=[f"Template pattern matched" for _ in matches],
                    indicators={"template_count": len(matches)},
                )

        return None

    def _detect_repetition(self, content: str) -> Optional[PatternMatch]:
        """Detect repetition-based attack pattern.

        Args:
            content: Document content

        Returns:
            Pattern match or None
        """
        # Look for repeated phrases (2+ words)
        words = content.lower().split()
        phrase_counts = Counter()

        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 8:  # Skip short phrases
                phrase_counts[phrase] += 1

        repeated_phrases = [
            (phrase, count)
            for phrase, count in phrase_counts.items()
            if count >= 3
        ]

        if repeated_phrases:
            max_count = max(count for _, count in repeated_phrases)
            confidence = min(max_count / 8, 0.9)

            if confidence >= self.min_confidence:
                return PatternMatch(
                    pattern=AttackPattern.REPETITION_ATTACK,
                    confidence=confidence,
                    evidence=[
                        f"'{phrase}' repeated {count} times"
                        for phrase, count in repeated_phrases[:3]
                    ],
                    indicators={"repeated_phrases": len(repeated_phrases)},
                )

        return None

    def _generate_fingerprint(
        self, document: Document, patterns: List[PatternMatch]
    ) -> AttackFingerprint:
        """Generate fingerprint for a document.

        Args:
            document: Document to fingerprint
            patterns: Detected patterns

        Returns:
            Attack fingerprint
        """
        content = document.content

        # Content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Structural hash (based on content structure, not exact words)
        structure = re.sub(r"[a-zA-Z]+", "W", content)  # Replace words
        structure = re.sub(r"[0-9]+", "N", structure)  # Replace numbers
        structural_hash = hashlib.sha256(structure.encode()).hexdigest()

        # Vocabulary signature (distinctive words)
        words = set(content.lower().split())
        # Keep words that are less common
        vocab_sig = {w for w in words if len(w) > 5}

        # Style metrics
        sentences = content.split(".")
        style_metrics = {
            "avg_sentence_length": (
                sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            ),
            "avg_word_length": (
                sum(len(w) for w in content.split()) / max(len(content.split()), 1)
            ),
            "punctuation_ratio": (
                sum(1 for c in content if c in ".,!?;:") / max(len(content), 1)
            ),
            "uppercase_ratio": (
                sum(1 for c in content if c.isupper()) / max(len(content), 1)
            ),
        }

        return AttackFingerprint(
            content_hash=content_hash,
            structural_hash=structural_hash,
            patterns_detected=[p.pattern for p in patterns],
            vocabulary_signature=vocab_sig,
            style_metrics=style_metrics,
        )

    def _estimate_attack_type(
        self, patterns: List[PatternMatch]
    ) -> Optional[AttackType]:
        """Estimate the attack type from detected patterns.

        Args:
            patterns: Detected patterns

        Returns:
            Estimated attack type or None
        """
        if not patterns:
            return None

        # Map patterns to attack types
        pattern_to_attack = {
            AttackPattern.KEYWORD_STUFFING: AttackType.ADVERSARIAL,
            AttackPattern.QUERY_INJECTION: AttackType.DIRECT,
            AttackPattern.AUTHORITY_MIMICKING: AttackType.DIRECT,
            AttackPattern.TEMPLATE_BASED: AttackType.DIRECT,
            AttackPattern.REPETITION_ATTACK: AttackType.ADVERSARIAL,
            AttackPattern.CHAIN_CORRELATION: AttackType.CHAIN,
            AttackPattern.STEALTH_EMBEDDING: AttackType.STEALTH,
        }

        # Count votes for each attack type
        votes = Counter()
        for pattern in patterns:
            if pattern.pattern in pattern_to_attack:
                attack = pattern_to_attack[pattern.pattern]
                votes[attack] += pattern.confidence

        if votes:
            return votes.most_common(1)[0][0]

        return None

    def _generate_recommendations(
        self, patterns: List[PatternMatch], confidence: float
    ) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            patterns: Detected patterns
            confidence: Overall confidence

        Returns:
            List of recommendations
        """
        recommendations = []

        if confidence >= 0.8:
            recommendations.append("URGENT: Quarantine document immediately")
            recommendations.append("Conduct full knowledge base scan")
        elif confidence >= 0.5:
            recommendations.append("Flag document for manual review")
            recommendations.append("Check for related suspicious documents")

        for pattern in patterns:
            if pattern.pattern == AttackPattern.CHAIN_CORRELATION:
                recommendations.append("Search for correlated documents in chain attack")
            elif pattern.pattern == AttackPattern.AUTHORITY_MIMICKING:
                recommendations.append("Verify claimed authoritative sources")
            elif pattern.pattern == AttackPattern.QUERY_INJECTION:
                recommendations.append("Check for targeted query attacks")

        return recommendations
