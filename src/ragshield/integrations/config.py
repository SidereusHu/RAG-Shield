"""Configuration management for RAG-Shield.

Provides a centralized configuration system with support for
environment variables, config files, and runtime overrides.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

from ragshield.defense import DefenseLevel


class ConfigSource(Enum):
    """Configuration source types."""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


@dataclass
class DetectionConfig:
    """Detection configuration.

    Attributes:
        enabled: Enable poison detection
        preset: Detector preset (strict/default/permissive)
        perplexity_threshold: Perplexity detection threshold
        similarity_threshold: Similarity cluster threshold
        semantic_threshold: Semantic confidence threshold
    """

    enabled: bool = True
    preset: str = "default"
    perplexity_threshold: Optional[float] = None
    similarity_threshold: Optional[float] = None
    semantic_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseSettings:
    """Defense configuration.

    Attributes:
        level: Defense level preset
        auto_quarantine: Auto-quarantine suspicious documents
        auto_block_sources: Auto-block malicious sources
        detection_threshold: Confidence threshold for detection
        quarantine_review_required: Require review for release
    """

    level: DefenseLevel = DefenseLevel.STANDARD
    auto_quarantine: bool = True
    auto_block_sources: bool = False
    detection_threshold: float = 0.7
    quarantine_review_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["level"] = self.level.value
        return result


@dataclass
class PrivacyConfig:
    """Privacy configuration.

    Attributes:
        enabled: Enable privacy protection
        epsilon: Differential privacy epsilon
        delta: Differential privacy delta
        noise_mechanism: Noise mechanism (laplace/gaussian)
        enable_pir: Enable Private Information Retrieval
    """

    enabled: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_mechanism: str = "laplace"
    enable_pir: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonitoringConfig:
    """Monitoring configuration.

    Attributes:
        enabled: Enable security monitoring
        log_level: Logging level
        metrics_enabled: Enable metrics collection
        alert_threshold: Alert threshold for anomalies
        rate_limit_requests: Max requests per window
        rate_limit_window: Rate limit window in seconds
    """

    enabled: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True
    alert_threshold: float = 0.8
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicsConfig:
    """Forensics configuration.

    Attributes:
        enabled: Enable attack forensics
        track_provenance: Track document provenance
        analyze_patterns: Analyze attack patterns
        retain_days: Days to retain forensic data
    """

    enabled: bool = True
    track_provenance: bool = True
    analyze_patterns: bool = True
    retain_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGShieldConfig:
    """Complete RAG-Shield configuration.

    Centralized configuration for all RAG-Shield components.
    """

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    defense: DefenseSettings = field(default_factory=DefenseSettings)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    forensics: ForensicsConfig = field(default_factory=ForensicsConfig)

    # Metadata
    config_source: ConfigSource = ConfigSource.DEFAULT
    config_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection": self.detection.to_dict(),
            "defense": self.defense.to_dict(),
            "privacy": self.privacy.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "forensics": self.forensics.to_dict(),
            "config_source": self.config_source.value,
            "config_version": self.config_version,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGShieldConfig":
        """Create from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            RAGShieldConfig instance
        """
        config = cls()

        if "detection" in data:
            det = data["detection"]
            config.detection = DetectionConfig(**det)

        if "defense" in data:
            defense_data = data["defense"]
            if "level" in defense_data:
                defense_data["level"] = DefenseLevel(defense_data["level"])
            config.defense = DefenseSettings(**defense_data)

        if "privacy" in data:
            config.privacy = PrivacyConfig(**data["privacy"])

        if "monitoring" in data:
            config.monitoring = MonitoringConfig(**data["monitoring"])

        if "forensics" in data:
            config.forensics = ForensicsConfig(**data["forensics"])

        if "config_source" in data:
            config.config_source = ConfigSource(data["config_source"])

        if "config_version" in data:
            config.config_version = data["config_version"]

        return config

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RAGShieldConfig":
        """Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            RAGShieldConfig instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        config = cls.from_dict(data)
        config.config_source = ConfigSource.FILE
        return config


class ConfigManager:
    """Configuration manager with environment variable support.

    Manages configuration with priority: runtime > environment > file > default.
    """

    # Environment variable prefix
    ENV_PREFIX = "RAGSHIELD_"

    # Environment variable mappings
    ENV_MAPPINGS = {
        # Detection
        "DETECTION_ENABLED": ("detection", "enabled", bool),
        "DETECTION_PRESET": ("detection", "preset", str),
        "DETECTION_PERPLEXITY_THRESHOLD": ("detection", "perplexity_threshold", float),
        "DETECTION_SIMILARITY_THRESHOLD": ("detection", "similarity_threshold", float),
        # Defense
        "DEFENSE_LEVEL": ("defense", "level", str),
        "DEFENSE_AUTO_QUARANTINE": ("defense", "auto_quarantine", bool),
        "DEFENSE_THRESHOLD": ("defense", "detection_threshold", float),
        # Privacy
        "PRIVACY_ENABLED": ("privacy", "enabled", bool),
        "PRIVACY_EPSILON": ("privacy", "epsilon", float),
        "PRIVACY_DELTA": ("privacy", "delta", float),
        "PRIVACY_PIR_ENABLED": ("privacy", "enable_pir", bool),
        # Monitoring
        "MONITORING_ENABLED": ("monitoring", "enabled", bool),
        "MONITORING_LOG_LEVEL": ("monitoring", "log_level", str),
        "MONITORING_RATE_LIMIT": ("monitoring", "rate_limit_requests", int),
        # Forensics
        "FORENSICS_ENABLED": ("forensics", "enabled", bool),
        "FORENSICS_RETAIN_DAYS": ("forensics", "retain_days", int),
    }

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        load_env: bool = True,
    ):
        """Initialize configuration manager.

        Args:
            config_file: Optional path to config file
            load_env: Whether to load from environment variables
        """
        self._config: RAGShieldConfig = RAGShieldConfig()
        self._overrides: Dict[str, Any] = {}

        # Load from file if specified
        if config_file:
            self._load_from_file(config_file)

        # Load from environment
        if load_env:
            self._load_from_env()

    def _load_from_file(self, path: Union[str, Path]) -> None:
        """Load configuration from file."""
        path = Path(path)
        if path.exists():
            self._config = RAGShieldConfig.load(path)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        for env_key, (section, key, type_) in self.ENV_MAPPINGS.items():
            full_key = f"{self.ENV_PREFIX}{env_key}"
            value = os.environ.get(full_key)

            if value is not None:
                # Type conversion
                if type_ == bool:
                    value = value.lower() in ("true", "1", "yes")
                elif type_ == float:
                    value = float(value)
                elif type_ == int:
                    value = int(value)

                # Set value
                section_obj = getattr(self._config, section)
                setattr(section_obj, key, value)
                self._config.config_source = ConfigSource.ENVIRONMENT

    @property
    def config(self) -> RAGShieldConfig:
        """Get the current configuration."""
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dotted key.

        Args:
            key: Dotted key (e.g., "detection.enabled")
            default: Default value if not found

        Returns:
            Configuration value
        """
        # Check runtime overrides first
        if key in self._overrides:
            return self._overrides[key]

        # Navigate the config
        parts = key.split(".")
        obj = self._config

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default

        return obj

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime.

        Args:
            key: Dotted key (e.g., "detection.enabled")
            value: Value to set
        """
        self._overrides[key] = value
        self._config.config_source = ConfigSource.RUNTIME

        # Also update the actual config object
        parts = key.split(".")
        if len(parts) == 2:
            section, attr = parts
            section_obj = getattr(self._config, section)
            setattr(section_obj, attr, value)

    def reset(self) -> None:
        """Reset to default configuration."""
        self._config = RAGShieldConfig()
        self._overrides.clear()

    def validate(self) -> List[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate detection
        if self._config.detection.preset not in ["strict", "default", "permissive"]:
            errors.append(f"Invalid detection preset: {self._config.detection.preset}")

        # Validate privacy
        if self._config.privacy.epsilon <= 0:
            errors.append(f"Epsilon must be positive: {self._config.privacy.epsilon}")

        if self._config.privacy.delta < 0 or self._config.privacy.delta > 1:
            errors.append(f"Delta must be in [0, 1]: {self._config.privacy.delta}")

        # Validate monitoring
        if self._config.monitoring.rate_limit_requests <= 0:
            errors.append("Rate limit must be positive")

        return errors


# Default configuration profiles
PROFILES = {
    "development": RAGShieldConfig(
        detection=DetectionConfig(enabled=True, preset="permissive"),
        defense=DefenseSettings(level=DefenseLevel.MINIMAL),
        monitoring=MonitoringConfig(log_level="DEBUG"),
    ),
    "production": RAGShieldConfig(
        detection=DetectionConfig(enabled=True, preset="default"),
        defense=DefenseSettings(level=DefenseLevel.STANDARD),
        monitoring=MonitoringConfig(log_level="INFO"),
    ),
    "high_security": RAGShieldConfig(
        detection=DetectionConfig(enabled=True, preset="strict"),
        defense=DefenseSettings(
            level=DefenseLevel.PARANOID,
            auto_quarantine=True,
            auto_block_sources=True,
        ),
        privacy=PrivacyConfig(enabled=True, epsilon=0.1),
        monitoring=MonitoringConfig(log_level="WARNING", alert_threshold=0.5),
    ),
}


def get_config(profile: str = "production") -> RAGShieldConfig:
    """Get a configuration profile.

    Args:
        profile: Profile name (development/production/high_security)

    Returns:
        RAGShieldConfig for the profile
    """
    if profile in PROFILES:
        return PROFILES[profile]
    return RAGShieldConfig()


def create_config_manager(
    config_file: Optional[str] = None,
    profile: Optional[str] = None,
    load_env: bool = True,
) -> ConfigManager:
    """Create a configured ConfigManager.

    Args:
        config_file: Optional path to config file
        profile: Optional profile to start with
        load_env: Whether to load from environment

    Returns:
        Configured ConfigManager
    """
    manager = ConfigManager(config_file=config_file, load_env=load_env)

    # Apply profile if specified
    if profile and profile in PROFILES:
        base_config = PROFILES[profile]
        manager._config = base_config
        if load_env:
            manager._load_from_env()

    return manager
