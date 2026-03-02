"""Environment-driven configuration models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are Sophie, a friendly desktop AI buddy. Be conversational and natural. "
    "When the user asks a direct question, answer it directly without unsolicited "
    "productivity advice. When they ask for help, give concise and actionable guidance."
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
            max_output_tokens=_get_env_int("OPENAI_MAX_OUTPUT_TOKENS", 120),
            timeout_seconds=_get_env_float("OPENAI_TIMEOUT_SECONDS", 45.0),
        )


@dataclass(slots=True, frozen=True)
class VoiceTriggerConfig:
    """Wake-word voice activation settings."""

    enabled: bool
    wake_word: str
    listen_seconds: float
    followup_listen_seconds: float
    end_silence_seconds: float
    activity_threshold: float
    sample_rate: int
    transcription_model: str

    @classmethod
    def from_env(cls) -> "VoiceTriggerConfig":
        """Load wake-word voice activation settings from environment variables."""
        return cls(
            enabled=_get_env_bool("ASSISTANT_ENABLE_VOICE_TRIGGER", False),
            wake_word=_get_env("ASSISTANT_WAKE_WORD", "Lune") or "Lune",
            listen_seconds=_get_env_float("ASSISTANT_VOICE_LISTEN_SECONDS", 3.0),
            followup_listen_seconds=_get_env_float(
                "ASSISTANT_VOICE_FOLLOWUP_LISTEN_SECONDS",
                12.0,
            ),
            end_silence_seconds=_get_env_float("ASSISTANT_VOICE_END_SILENCE_SECONDS", 1.0),
            activity_threshold=_get_env_float("ASSISTANT_VOICE_ACTIVITY_THRESHOLD", 450.0),
            sample_rate=_get_env_int("ASSISTANT_VOICE_SAMPLE_RATE", 16000),
            transcription_model=_get_env(
                "ASSISTANT_VOICE_TRANSCRIPTION_MODEL",
                "gpt-4o-mini-transcribe",
            )
            or "gpt-4o-mini-transcribe",
        )


@dataclass(slots=True, frozen=True)
class AvatarConfig:
    """Desktop avatar overlay settings."""

    enabled: bool
    auto_move: bool
    size: int
    opacity: float

    @classmethod
    def from_env(cls) -> "AvatarConfig":
        """Load avatar overlay settings from environment variables."""
        return cls(
            enabled=_get_env_bool("ASSISTANT_ENABLE_AVATAR", False),
            auto_move=_get_env_bool("ASSISTANT_AVATAR_AUTO_MOVE", True),
            size=max(100, _get_env_int("ASSISTANT_AVATAR_SIZE", 180)),
            opacity=max(0.4, min(1.0, _get_env_float("ASSISTANT_AVATAR_OPACITY", 0.95))),
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
class DesktopControlConfig:
    """Desktop keyboard/mouse control settings."""

    enabled: bool
    require_approval: bool
    allowed_launch_commands: tuple[str, ...]
    max_actions_per_turn: int
    action_delay_seconds: float
    action_log_path: Path | None

    @classmethod
    def from_env(cls, *, artifacts_dir: Path) -> "DesktopControlConfig":
        """Load desktop-control settings from environment variables."""
        log_setting: str = _get_env("ASSISTANT_DESKTOP_ACTION_LOG", "actions.log") or "actions.log"
        log_path: Path | None = None
        if log_setting.lower() not in {"none", "off", "false", "0"}:
            configured_path: Path = Path(log_setting)
            log_path = configured_path if configured_path.is_absolute() else artifacts_dir / configured_path

        allowed_launch_text: str = _get_env("ASSISTANT_DESKTOP_ALLOWED_LAUNCH", "") or ""
        return cls(
            enabled=_get_env_bool("ASSISTANT_ENABLE_DESKTOP_CONTROL", False),
            require_approval=_get_env_bool("ASSISTANT_DESKTOP_REQUIRE_APPROVAL", True),
            allowed_launch_commands=_split_csv(allowed_launch_text),
            max_actions_per_turn=max(1, _get_env_int("ASSISTANT_DESKTOP_MAX_ACTIONS_PER_TURN", 5)),
            action_delay_seconds=max(
                0.0,
                min(2.0, _get_env_float("ASSISTANT_DESKTOP_ACTION_DELAY_SECONDS", 0.05)),
            ),
            action_log_path=log_path,
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
    voice_trigger: VoiceTriggerConfig
    avatar: AvatarConfig
    desktop_control: DesktopControlConfig
    system_prompt: str
    log_level: str

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        """Load assistant runtime settings from environment variables."""
        providers_text: str = (
            _get_env("ASSISTANT_CONTEXT_PROVIDERS", "timestamp,environment,active_window")
            or "timestamp,environment,active_window"
        )
        artifacts_dir: Path = Path(_get_env("ASSISTANT_ARTIFACTS_DIR", "./artifacts") or "./artifacts")
        return cls(
            openai=OpenAIConfig.from_env(),
            interval_seconds=_get_env_float("ASSISTANT_INTERVAL_SECONDS", 8.0),
            context_provider_names=_split_csv(providers_text),
            artifacts_dir=artifacts_dir,
            monitor_index=_get_env_int("ASSISTANT_MONITOR_INDEX", 1),
            enable_speech=_get_env_bool("ASSISTANT_ENABLE_SPEECH", True),
            voice_trigger=VoiceTriggerConfig.from_env(),
            avatar=AvatarConfig.from_env(),
            desktop_control=DesktopControlConfig.from_env(artifacts_dir=artifacts_dir),
            system_prompt=_get_env("ASSISTANT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
            or DEFAULT_SYSTEM_PROMPT,
            log_level=_get_env("ASSISTANT_LOG_LEVEL", "INFO") or "INFO",
        )
