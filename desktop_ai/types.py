"""Domain types shared across the assistant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
