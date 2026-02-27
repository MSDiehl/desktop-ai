"""Environment-driven configuration models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are a context-aware desktop AI assistant. Use the screenshot and context "
    "to understand what the user is doing right now. Reply in short, practical "
    "sentences with clear next-step suggestions."
)


def _get_env(name: str, default: str | None = None) -> str | None:
    """Return an environment variable or the provided default value."""
    value: str | None = os.getenv(name)
    if value is None:
        return default
    stripped: str = value.strip()
    return stripped if stripped else default


def _require_env(name: str) -> str:
    """Return a required environment variable or raise a ValueError."""
    value: str | None = _get_env(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_env_float(name: str, default: float) -> float:
    """Return an environment variable parsed as float."""
    value: str | None = _get_env(name)
    if value is None:
        return default
    return float(value)


def _get_env_int(name: str, default: int) -> int:
    """Return an environment variable parsed as int."""
    value: str | None = _get_env(name)
    if value is None:
        return default
    return int(value)


def _get_env_bool(name: str, default: bool) -> bool:
    """Return an environment variable parsed as bool."""
    value: str | None = _get_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(value: str) -> tuple[str, ...]:
    """Split a comma-separated string into a tuple of non-empty values."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(slots=True, frozen=True)
class OpenAIConfig:
    """OpenAI generation settings."""

    api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load OpenAI settings from environment variables."""
        return cls(
            api_key=_require_env("OPENAI_API_KEY"),
            model=_get_env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
            temperature=_get_env_float("OPENAI_TEMPERATURE", 0.5),
            max_output_tokens=_get_env_int("OPENAI_MAX_OUTPUT_TOKENS", 220),
            timeout_seconds=_get_env_float("OPENAI_TIMEOUT_SECONDS", 45.0),
        )


@dataclass(slots=True, frozen=True)
class ElevenLabsConfig:
    """ElevenLabs synthesis settings."""

    api_key: str
    voice_id: str
    model_id: str
    output_format: str
    stability: float
    similarity_boost: float
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "ElevenLabsConfig":
        """Load ElevenLabs settings from environment variables."""
        return cls(
            api_key=_require_env("ELEVENLABS_API_KEY"),
            voice_id=_require_env("ELEVENLABS_VOICE_ID"),
            model_id=_get_env("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
            or "eleven_multilingual_v2",
            output_format=_get_env("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
            or "pcm_16000",
            stability=_get_env_float("ELEVENLABS_STABILITY", 0.45),
            similarity_boost=_get_env_float("ELEVENLABS_SIMILARITY_BOOST", 0.75),
            timeout_seconds=_get_env_float("ELEVENLABS_TIMEOUT_SECONDS", 45.0),
        )


@dataclass(slots=True, frozen=True)
class AssistantConfig:
    """Runtime settings for assistant orchestration."""

    openai: OpenAIConfig
    interval_seconds: float
    context_provider_names: tuple[str, ...]
    artifacts_dir: Path
    monitor_index: int
    enable_speech: bool
    system_prompt: str
    log_level: str

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        """Load assistant runtime settings from environment variables."""
        providers_text: str = (
            _get_env("ASSISTANT_CONTEXT_PROVIDERS", "timestamp,environment,active_window")
            or "timestamp,environment,active_window"
        )
        return cls(
            openai=OpenAIConfig.from_env(),
            interval_seconds=_get_env_float("ASSISTANT_INTERVAL_SECONDS", 8.0),
            context_provider_names=_split_csv(providers_text),
            artifacts_dir=Path(_get_env("ASSISTANT_ARTIFACTS_DIR", "./artifacts") or "./artifacts"),
            monitor_index=_get_env_int("ASSISTANT_MONITOR_INDEX", 1),
            enable_speech=_get_env_bool("ASSISTANT_ENABLE_SPEECH", True),
            system_prompt=_get_env("ASSISTANT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
            or DEFAULT_SYSTEM_PROMPT,
            log_level=_get_env("ASSISTANT_LOG_LEVEL", "INFO") or "INFO",
        )
