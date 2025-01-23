"""Semantic-based poison detection.

Detects poisoned documents by analyzing semantic consistency and
identifying potentially malicious content patterns.
"""

import re
from typing import List, Set, Optional

from ragshield.core.document import Document
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel


class SemanticDetector(PoisonDetector):
    """Detect poisoned documents based on semantic analysis.

    This detector looks for:
    1. Instruction injection patterns
    2. Conflicting/contradictory information
    3. Suspicious keyword patterns
    4. Unusual formatting that may indicate adversarial content

    Args:
        suspicious_patterns: List of regex patterns to detect
        instruction_keywords: Keywords indicating instruction injection
        confidence_threshold: Threshold for flagging as poisoned
    """

    # Default patterns that may indicate poisoning
    DEFAULT_SUSPICIOUS_PATTERNS = [
        # Instruction injection patterns
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(everything|all)",
        r"new\s+instructions?:",
        r"system\s*:\s*you\s+are",
        r"<\s*system\s*>",
        r"\[INST\]",
        r"<<SYS>>",
        # Jailbreak patterns
        r"pretend\s+you\s+are",
        r"act\s+as\s+if",
        r"roleplay\s+as",
        r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|evil|uncensored)",
        # Data extraction patterns
        r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        r"show\s+me\s+(?:your\s+)?(?:system\s+)?prompt",
        r"what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
        r"output\s+(?:your\s+)?(?:initial|system)\s+(?:prompt|instructions?)",
        # Base64 encoded content (potential obfuscation)
        r"[A-Za-z0-9+/]{50,}={0,2}",
        # Unicode manipulation
        r"[\u200b-\u200f\u2028-\u202f]",  # Zero-width and special unicode
    ]

    DEFAULT_INSTRUCTION_KEYWORDS = {
        "ignore",
        "disregard",
        "forget",
        "override",
        "bypass",
        "jailbreak",
        "DAN",
        "pretend",
        "roleplay",
        "system prompt",
        "initial instructions",
        "reveal",
        "output",
        "print",
    }

    def __init__(
        self,
        suspicious_patterns: Optional[List[str]] = None,
        instruction_keywords: Optional[Set[str]] = None,
        confidence_threshold: float = 0.5,
        case_sensitive: bool = False,
    ):
        self.suspicious_patterns = suspicious_patterns or self.DEFAULT_SUSPICIOUS_PATTERNS
        self.instruction_keywords = instruction_keywords or self.DEFAULT_INSTRUCTION_KEYWORDS
        self.confidence_threshold = confidence_threshold
        self.case_sensitive = case_sensitive

        # Compile regex patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self.compiled_patterns = [re.compile(p, flags) for p in self.suspicious_patterns]

    def _check_patterns(self, text: str) -> List[tuple[str, str]]:
        """Check text for suspicious patterns.

        Args:
            text: Text to analyze

        Returns:
            List of (pattern, matched_text) tuples
        """
        matches = []
        for pattern, compiled in zip(self.suspicious_patterns, self.compiled_patterns):
            match = compiled.search(text)
            if match:
                matches.append((pattern, match.group()))
        return matches

    def _check_keywords(self, text: str) -> List[str]:
        """Check text for suspicious keywords.

        Args:
            text: Text to analyze

        Returns:
            List of found keywords
        """
        text_lower = text.lower()
        found = []
        for keyword in self.instruction_keywords:
            if keyword.lower() in text_lower:
                found.append(keyword)
        return found

    def _check_formatting_anomalies(self, text: str) -> List[str]:
        """Check for unusual formatting that may indicate adversarial content.

        Args:
            text: Text to analyze

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / (
            len(text) + 1
        )
        if special_char_ratio > 0.3:
            anomalies.append(f"excessive special characters ({special_char_ratio:.1%})")

        # Check for hidden unicode characters
        hidden_unicode = re.findall(r"[\u200b-\u200f\u2028-\u202f\ufeff]", text)
        if hidden_unicode:
            anomalies.append(f"hidden unicode characters ({len(hidden_unicode)} found)")

        # Check for unusual repetition
        words = text.lower().split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repeat = max(word_counts.values())
            if max_repeat > len(words) * 0.3:
                anomalies.append(f"unusual word repetition (max: {max_repeat})")

        # Check for mixed case anomalies (e.g., "iGnOrE InStRuCtIoNs")
        if len(text) > 10:
            case_changes = sum(
                1
                for i in range(len(text) - 1)
                if text[i].isalpha()
                and text[i + 1].isalpha()
                and text[i].isupper() != text[i + 1].isupper()
            )
            if case_changes > len(text) * 0.2:
                anomalies.append("suspicious case alternation")

        return anomalies

    def _check_contradictions(self, text: str) -> List[str]:
        """Check for potentially contradictory or suspicious claims.

        Args:
            text: Text to analyze

        Returns:
            List of suspicious claims
        """
        suspicious_claims = []

        # Patterns that might indicate false/malicious information
        contradiction_patterns = [
            (r"(?:the\s+)?(?:official|correct|true)\s+answer\s+is", "claims to be official answer"),
            (r"contrary\s+to\s+(?:popular\s+)?belief", "contradicts common knowledge"),
            (r"(?:actually|in\s+fact),?\s+(?:the\s+)?(?:real|true)", "claims alternative truth"),
            (r"(?:everyone|they)\s+(?:is|are)\s+wrong", "dismisses other sources"),
        ]

        for pattern, description in contradiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious_claims.append(description)

        return suspicious_claims

    def detect(self, document: Document) -> DetectionResult:
        """Detect if document contains poisoned content.

        Args:
            document: Document to check

        Returns:
            Detection result
        """
        text = document.content
        findings = []
        scores = []

        # Check patterns
        pattern_matches = self._check_patterns(text)
        if pattern_matches:
            findings.append(f"suspicious patterns: {len(pattern_matches)} matches")
            scores.append(min(1.0, len(pattern_matches) * 0.3))

        # Check keywords
        keyword_matches = self._check_keywords(text)
        if keyword_matches:
            findings.append(f"suspicious keywords: {', '.join(keyword_matches[:3])}")
            scores.append(min(1.0, len(keyword_matches) * 0.2))

        # Check formatting anomalies
        format_anomalies = self._check_formatting_anomalies(text)
        if format_anomalies:
            findings.append(f"formatting anomalies: {'; '.join(format_anomalies)}")
            scores.append(min(1.0, len(format_anomalies) * 0.25))

        # Check contradictions
        contradictions = self._check_contradictions(text)
        if contradictions:
            findings.append(f"suspicious claims: {'; '.join(contradictions)}")
            scores.append(min(1.0, len(contradictions) * 0.2))

        # Calculate overall score
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.0

        is_poisoned = overall_score >= self.confidence_threshold

        # Determine confidence and threat level
        if is_poisoned:
            confidence = min(0.95, 0.5 + overall_score * 0.45)
            if overall_score > 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif overall_score > 0.6:
                threat_level = ThreatLevel.HIGH
            else:
                threat_level = ThreatLevel.MEDIUM
        else:
            confidence = min(0.95, 0.5 + (self.confidence_threshold - overall_score) * 0.45)
            threat_level = ThreatLevel.NONE

        reason = "; ".join(findings) if findings else "No suspicious patterns detected"

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason,
            score=overall_score,
            metadata={
                "pattern_matches": len(pattern_matches),
                "keyword_matches": keyword_matches,
                "format_anomalies": format_anomalies,
                "contradictions": contradictions,
            },
        )
