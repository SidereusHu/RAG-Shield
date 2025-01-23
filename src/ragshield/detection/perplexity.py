"""Perplexity-based poison detection.

Perplexity measures how "surprising" a text is to a language model.
Poisoned documents often have unusual patterns that result in high perplexity.
"""

import math
from typing import Optional
import numpy as np

from ragshield.core.document import Document
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel


class PerplexityDetector(PoisonDetector):
    """Detect poisoned documents based on perplexity scores.

    High perplexity indicates unusual text that may be adversarially crafted.

    Args:
        threshold: Perplexity threshold above which a document is flagged
        use_transformer: Whether to use transformer model (requires transformers)
        model_name: Name of the model to use for perplexity calculation
    """

    def __init__(
        self,
        threshold: float = 100.0,
        use_transformer: bool = False,
        model_name: str = "gpt2",
    ):
        self.threshold = threshold
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if use_transformer:
            self._load_model()

    def _load_model(self):
        """Load transformer model for perplexity calculation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for transformer-based perplexity. "
                "Install with: pip install transformers torch"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

    def _calculate_perplexity_transformer(self, text: str) -> float:
        """Calculate perplexity using transformer model."""
        import torch

        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss.item()

        # Perplexity = exp(loss)
        perplexity = math.exp(loss)

        return perplexity

    def _calculate_perplexity_simple(self, text: str) -> float:
        """Calculate perplexity using simple heuristics.

        This is a simplified version that doesn't require a language model.
        It uses statistical features to approximate perplexity.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        # Features that correlate with high perplexity
        features = []

        # 1. Character diversity
        unique_chars = len(set(text))
        char_diversity = unique_chars / (len(text) + 1)
        features.append(char_diversity)

        # 2. Token length variance
        tokens = text.split()
        if len(tokens) > 1:
            token_lengths = [len(t) for t in tokens]
            token_length_var = np.var(token_lengths)
            features.append(token_length_var)
        else:
            features.append(0.0)

        # 3. Unusual character patterns
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / (len(text) + 1)
        features.append(special_ratio * 100)

        # 4. Repetition detection
        words = text.lower().split()
        if len(words) > 0:
            word_diversity = len(set(words)) / len(words)
            repetition_score = (1 - word_diversity) * 100
            features.append(repetition_score)
        else:
            features.append(0.0)

        # 5. Case inconsistency
        case_changes = sum(
            1 for i in range(len(text) - 1) if text[i].islower() != text[i + 1].islower()
        )
        case_inconsistency = case_changes / (len(text) + 1) * 100
        features.append(case_inconsistency)

        # Combine features into a pseudo-perplexity score
        # Normal text typically scores 20-50, adversarial text scores 100+
        perplexity = sum(features) * 10

        return perplexity

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text.

        Args:
            text: Text to analyze

        Returns:
            Perplexity score
        """
        if self.use_transformer:
            return self._calculate_perplexity_transformer(text)
        else:
            return self._calculate_perplexity_simple(text)

    def detect(self, document: Document) -> DetectionResult:
        """Detect if document is poisoned based on perplexity.

        Args:
            document: Document to check

        Returns:
            Detection result
        """
        perplexity = self.calculate_perplexity(document.content)

        is_poisoned = perplexity > self.threshold

        # Calculate confidence based on how far from threshold
        if is_poisoned:
            # Confidence increases with perplexity above threshold
            excess = perplexity - self.threshold
            confidence = min(0.95, 0.5 + (excess / self.threshold) * 0.45)
        else:
            # Confidence increases as perplexity is further below threshold
            margin = self.threshold - perplexity
            confidence = min(0.95, 0.5 + (margin / self.threshold) * 0.45)

        # Determine threat level
        if not is_poisoned:
            threat_level = ThreatLevel.NONE
        elif perplexity > self.threshold * 2:
            threat_level = ThreatLevel.CRITICAL
        elif perplexity > self.threshold * 1.5:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM

        reason = (
            f"Perplexity {perplexity:.2f} {'exceeds' if is_poisoned else 'below'} "
            f"threshold {self.threshold:.2f}"
        )

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason,
            score=perplexity,
            metadata={"perplexity": perplexity, "threshold": self.threshold},
        )
