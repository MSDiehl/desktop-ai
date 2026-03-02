"""Domain types shared across the assistant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

AssistantPresenceState: TypeAlias = Literal["idle", "listening", "thinking", "speaking"]


@dataclass(slots=True, frozen=True)
class CapturedScreen:
    """Represents a screenshot and metadata from the desktop."""

    png_bytes: bytes
    width: int
    height: int
    captured_at: datetime
    mime_type: str = "image/png"

    @classmethod
    def empty(cls) -> "CapturedScreen":
        """Build an empty screen placeholder if capture is unavailable."""
        return cls(
            png_bytes=b"",
            width=0,
            height=0,
            captured_at=datetime.now(timezone.utc),
        )


@dataclass(slots=True, frozen=True)
class AssistantTurnResult:
    """Represents one completed assistant turn."""

    prompt: str
    response_text: str
    context: dict[str, str]
    audio_path: Path | None
    started_at: datetime
    finished_at: datetime
    action_plan: "DesktopActionPlan | None" = None
    action_results: tuple["ActionExecutionResult", ...] = ()


@dataclass(slots=True, frozen=True)
class DesktopAction:
    """Represents one desktop action the assistant can execute."""

    type: str
    args: dict[str, Any]


@dataclass(slots=True, frozen=True)
class DesktopActionPlan:
    """Represents a parsed model response with optional desktop actions."""

    spoken_reply: str
    actions: tuple[DesktopAction, ...]
    raw_response: str


@dataclass(slots=True, frozen=True)
class ActionExecutionResult:
    """Represents one action execution attempt."""

    action: DesktopAction
    success: bool
    detail: str
    started_at: datetime
    finished_at: datetime


@dataclass(slots=True, frozen=True)
class VoiceActivation:
    """Represents one wake-word activation event."""

    transcript: str
    wake_word: str
    user_note: str | None
