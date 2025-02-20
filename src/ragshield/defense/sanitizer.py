"""Input sanitization for RAG systems.

Provides content filtering, validation, and cleaning of documents
before ingestion into the knowledge base.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import re
import numpy as np

from ragshield.core.document import Document


class SanitizationAction(Enum):
    """Actions for sanitization rules."""

    ALLOW = "allow"
    BLOCK = "block"
    CLEAN = "clean"
    FLAG = "flag"


@dataclass
class SanitizationRule:
    """A sanitization rule.

    Attributes:
        name: Rule name
        description: Rule description
        pattern: Regex pattern to match
        action: Action to take on match
        replacement: Replacement text for CLEAN action
        priority: Rule priority (higher = first)
    """

    name: str
    description: str
    pattern: str
    action: SanitizationAction
    replacement: str = ""
    priority: int = 0

    def matches(self, content: str) -> bool:
        """Check if content matches pattern.

        Args:
            content: Content to check

        Returns:
            True if matches
        """
        return bool(re.search(self.pattern, content, re.IGNORECASE))

    def apply(self, content: str) -> str:
        """Apply rule to content.

        Args:
            content: Content to sanitize

        Returns:
            Sanitized content
        """
        if self.action == SanitizationAction.CLEAN:
            return re.sub(self.pattern, self.replacement, content, flags=re.IGNORECASE)
        return content


@dataclass
class SanitizationResult:
    """Result of sanitization.

    Attributes:
        document: Sanitized document
        original_content: Original content
        is_blocked: Whether document was blocked
        is_modified: Whether content was modified
        matched_rules: Rules that matched
        warnings: Warning messages
    """

    document: Document
    original_content: str
    is_blocked: bool
    is_modified: bool
    matched_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EmbeddingValidationResult:
    """Result of embedding validation.

    Attributes:
        is_valid: Whether embedding is valid
        issues: List of issues found
        statistics: Embedding statistics
    """

    is_valid: bool
    issues: List[str] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)


class ContentSanitizer:
    """Sanitizes document content.

    Applies rules to clean, filter, or block suspicious content.
    """

    # Default patterns to block/clean
    DEFAULT_BLOCK_PATTERNS = [
        (r"<script[^>]*>.*?</script>", "Script tags"),
        (r"javascript:", "JavaScript URLs"),
        (r"on\w+\s*=", "Event handlers"),
        (r"data:\s*text/html", "Data URLs"),
    ]

    DEFAULT_CLEAN_PATTERNS = [
        (r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", "Control characters"),
        (r"<!--.*?-->", "", "HTML comments"),
        (r"\s{10,}", " ", "Excessive whitespace"),
    ]

    def __init__(
        self,
        max_content_length: int = 100000,
        min_content_length: int = 10,
        allowed_languages: Optional[Set[str]] = None,
    ):
        """Initialize sanitizer.

        Args:
            max_content_length: Maximum allowed content length
            min_content_length: Minimum content length
            allowed_languages: Allowed language codes (None = all)
        """
        self.max_content_length = max_content_length
        self.min_content_length = min_content_length
        self.allowed_languages = allowed_languages

        self._rules: List[SanitizationRule] = []
        self._custom_validators: List[Callable[[str], Tuple[bool, str]]] = []

        # Add default rules
        self._add_default_rules()

    def sanitize(self, document: Document) -> SanitizationResult:
        """Sanitize a document.

        Args:
            document: Document to sanitize

        Returns:
            Sanitization result
        """
        original_content = document.content
        content = document.content
        matched_rules = []
        warnings = []
        is_blocked = False

        # Check length limits
        if len(content) > self.max_content_length:
            warnings.append(f"Content truncated from {len(content)} to {self.max_content_length}")
            content = content[:self.max_content_length]

        if len(content) < self.min_content_length:
            is_blocked = True
            warnings.append("Content too short")

        # Apply rules in priority order
        sorted_rules = sorted(self._rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.matches(content):
                matched_rules.append(rule.name)

                if rule.action == SanitizationAction.BLOCK:
                    is_blocked = True
                    warnings.append(f"Blocked by rule: {rule.name}")
                    break
                elif rule.action == SanitizationAction.CLEAN:
                    content = rule.apply(content)
                elif rule.action == SanitizationAction.FLAG:
                    warnings.append(f"Flagged by rule: {rule.name}")

        # Apply custom validators
        for validator in self._custom_validators:
            is_valid, message = validator(content)
            if not is_valid:
                warnings.append(message)
                is_blocked = True

        # Create result document
        sanitized_doc = Document(
            content=content,
            doc_id=document.doc_id,
            embedding=document.embedding,
            metadata=document.metadata,
        )

        return SanitizationResult(
            document=sanitized_doc,
            original_content=original_content,
            is_blocked=is_blocked,
            is_modified=content != original_content,
            matched_rules=matched_rules,
            warnings=warnings,
        )

    def add_rule(self, rule: SanitizationRule) -> None:
        """Add a sanitization rule.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Args:
            name: Rule name

        Returns:
            True if removed
        """
        initial_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < initial_len

    def add_validator(
        self, validator: Callable[[str], Tuple[bool, str]]
    ) -> None:
        """Add a custom content validator.

        Args:
            validator: Function that takes content and returns (is_valid, message)
        """
        self._custom_validators.append(validator)

    def _add_default_rules(self) -> None:
        """Add default sanitization rules."""
        # Block rules
        for pattern, name in self.DEFAULT_BLOCK_PATTERNS:
            self._rules.append(
                SanitizationRule(
                    name=f"block_{name.lower().replace(' ', '_')}",
                    description=f"Block {name}",
                    pattern=pattern,
                    action=SanitizationAction.BLOCK,
                    priority=100,
                )
            )

        # Clean rules
        for pattern, replacement, name in self.DEFAULT_CLEAN_PATTERNS:
            self._rules.append(
                SanitizationRule(
                    name=f"clean_{name.lower().replace(' ', '_')}",
                    description=f"Clean {name}",
                    pattern=pattern,
                    action=SanitizationAction.CLEAN,
                    replacement=replacement,
                    priority=50,
                )
            )


class EmbeddingSanitizer:
    """Validates and sanitizes document embeddings.

    Checks for anomalous embeddings that might indicate manipulation.
    """

    def __init__(
        self,
        expected_dim: int = 384,
        max_norm: float = 10.0,
        min_norm: float = 0.1,
        max_sparsity: float = 0.95,
    ):
        """Initialize embedding sanitizer.

        Args:
            expected_dim: Expected embedding dimension
            max_norm: Maximum allowed L2 norm
            min_norm: Minimum allowed L2 norm
            max_sparsity: Maximum sparsity (ratio of near-zero values)
        """
        self.expected_dim = expected_dim
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.max_sparsity = max_sparsity

    def validate(
        self, embedding: List[float]
    ) -> EmbeddingValidationResult:
        """Validate an embedding.

        Args:
            embedding: Embedding to validate

        Returns:
            Validation result
        """
        issues = []
        stats = {}

        vec = np.array(embedding)

        # Check dimension
        if len(vec) != self.expected_dim:
            issues.append(
                f"Dimension mismatch: {len(vec)} vs expected {self.expected_dim}"
            )

        # Check for NaN/Inf
        if np.any(np.isnan(vec)):
            issues.append("Contains NaN values")
        if np.any(np.isinf(vec)):
            issues.append("Contains infinite values")

        # Calculate statistics
        norm = np.linalg.norm(vec)
        stats["norm"] = float(norm)
        stats["mean"] = float(np.mean(vec))
        stats["std"] = float(np.std(vec))
        stats["min"] = float(np.min(vec))
        stats["max"] = float(np.max(vec))

        # Check norm bounds
        if norm > self.max_norm:
            issues.append(f"Norm {norm:.2f} exceeds maximum {self.max_norm}")
        if norm < self.min_norm:
            issues.append(f"Norm {norm:.2f} below minimum {self.min_norm}")

        # Check sparsity
        sparsity = np.sum(np.abs(vec) < 1e-6) / len(vec)
        stats["sparsity"] = float(sparsity)

        if sparsity > self.max_sparsity:
            issues.append(f"Sparsity {sparsity:.2%} exceeds maximum {self.max_sparsity:.2%}")

        return EmbeddingValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            statistics=stats,
        )

    def normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit norm.

        Args:
            embedding: Embedding to normalize

        Returns:
            Normalized embedding
        """
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def clip(
        self,
        embedding: List[float],
        min_val: float = -3.0,
        max_val: float = 3.0,
    ) -> List[float]:
        """Clip embedding values to range.

        Args:
            embedding: Embedding to clip
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clipped embedding
        """
        vec = np.array(embedding)
        vec = np.clip(vec, min_val, max_val)
        return vec.tolist()


class MetadataSanitizer:
    """Sanitizes document metadata.

    Validates and cleans metadata fields.
    """

    RESERVED_FIELDS = {"poisoned", "attack_type", "target_query"}

    def __init__(
        self,
        max_field_length: int = 1000,
        max_fields: int = 50,
        strip_reserved: bool = True,
    ):
        """Initialize metadata sanitizer.

        Args:
            max_field_length: Maximum field value length
            max_fields: Maximum number of fields
            strip_reserved: Whether to strip reserved fields
        """
        self.max_field_length = max_field_length
        self.max_fields = max_fields
        self.strip_reserved = strip_reserved

    def sanitize(
        self, metadata: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Sanitize metadata dictionary.

        Args:
            metadata: Metadata to sanitize

        Returns:
            Tuple of (sanitized_metadata, warnings)
        """
        warnings = []
        sanitized = {}

        # Check field count
        if len(metadata) > self.max_fields:
            warnings.append(f"Truncated metadata from {len(metadata)} to {self.max_fields} fields")
            items = list(metadata.items())[:self.max_fields]
        else:
            items = list(metadata.items())

        for key, value in items:
            # Check reserved fields
            if self.strip_reserved and key in self.RESERVED_FIELDS:
                warnings.append(f"Stripped reserved field: {key}")
                continue

            # Sanitize key
            clean_key = self._sanitize_key(key)
            if clean_key != key:
                warnings.append(f"Sanitized key: {key} -> {clean_key}")

            # Sanitize value
            clean_value, value_warnings = self._sanitize_value(value)
            warnings.extend(value_warnings)

            sanitized[clean_key] = clean_value

        return sanitized, warnings

    def _sanitize_key(self, key: str) -> str:
        """Sanitize a metadata key.

        Args:
            key: Key to sanitize

        Returns:
            Sanitized key
        """
        # Remove special characters, keep alphanumeric and underscore
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(key))[:100]

    def _sanitize_value(self, value: Any) -> Tuple[Any, List[str]]:
        """Sanitize a metadata value.

        Args:
            value: Value to sanitize

        Returns:
            Tuple of (sanitized_value, warnings)
        """
        warnings = []

        if isinstance(value, str):
            if len(value) > self.max_field_length:
                warnings.append(f"Truncated value from {len(value)} chars")
                value = value[:self.max_field_length]
            return value, warnings

        if isinstance(value, (int, float, bool)):
            return value, warnings

        if isinstance(value, list):
            sanitized_list = []
            for item in value[:100]:  # Limit list size
                clean_item, item_warnings = self._sanitize_value(item)
                sanitized_list.append(clean_item)
                warnings.extend(item_warnings)
            return sanitized_list, warnings

        if isinstance(value, dict):
            # Recursively sanitize nested dict
            sanitized_dict, dict_warnings = self.sanitize(value)
            warnings.extend(dict_warnings)
            return sanitized_dict, warnings

        # Convert other types to string
        return str(value)[:self.max_field_length], warnings


class DocumentSanitizer:
    """Unified document sanitizer.

    Combines content, embedding, and metadata sanitization.
    """

    def __init__(
        self,
        content_sanitizer: Optional[ContentSanitizer] = None,
        embedding_sanitizer: Optional[EmbeddingSanitizer] = None,
        metadata_sanitizer: Optional[MetadataSanitizer] = None,
    ):
        """Initialize document sanitizer.

        Args:
            content_sanitizer: Content sanitizer (None = create default)
            embedding_sanitizer: Embedding sanitizer (None = create default)
            metadata_sanitizer: Metadata sanitizer (None = create default)
        """
        self.content_sanitizer = content_sanitizer or ContentSanitizer()
        self.embedding_sanitizer = embedding_sanitizer or EmbeddingSanitizer()
        self.metadata_sanitizer = metadata_sanitizer or MetadataSanitizer()

    def sanitize(
        self,
        document: Document,
        validate_embedding: bool = True,
        sanitize_metadata: bool = True,
    ) -> Tuple[Document, Dict[str, Any]]:
        """Fully sanitize a document.

        Args:
            document: Document to sanitize
            validate_embedding: Whether to validate embedding
            sanitize_metadata: Whether to sanitize metadata

        Returns:
            Tuple of (sanitized_document, report)
        """
        report = {
            "content": {},
            "embedding": {},
            "metadata": {},
            "is_blocked": False,
            "warnings": [],
        }

        # Sanitize content
        content_result = self.content_sanitizer.sanitize(document)
        report["content"] = {
            "is_blocked": content_result.is_blocked,
            "is_modified": content_result.is_modified,
            "matched_rules": content_result.matched_rules,
        }
        report["warnings"].extend(content_result.warnings)

        if content_result.is_blocked:
            report["is_blocked"] = True
            return content_result.document, report

        sanitized_doc = content_result.document

        # Validate embedding
        if validate_embedding and sanitized_doc.embedding:
            emb_result = self.embedding_sanitizer.validate(sanitized_doc.embedding)
            report["embedding"] = {
                "is_valid": emb_result.is_valid,
                "issues": emb_result.issues,
                "statistics": emb_result.statistics,
            }

            if not emb_result.is_valid:
                report["warnings"].extend(emb_result.issues)
                # Normalize embedding to fix some issues
                sanitized_doc.embedding = self.embedding_sanitizer.normalize(
                    sanitized_doc.embedding
                )

        # Sanitize metadata
        if sanitize_metadata and sanitized_doc.metadata.custom:
            clean_metadata, meta_warnings = self.metadata_sanitizer.sanitize(
                sanitized_doc.metadata.custom
            )
            sanitized_doc.metadata.custom = clean_metadata
            report["metadata"]["warnings"] = meta_warnings
            report["warnings"].extend(meta_warnings)

        return sanitized_doc, report
