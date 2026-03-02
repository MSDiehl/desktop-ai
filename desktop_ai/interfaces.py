"""Protocol interfaces for assistant components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

from desktop_ai.types import AssistantPresenceState, CapturedScreen, MemoryRecord, VoiceActivation


class ContextCollector(Protocol):
    """Collects a flattened context dictionary for one assistant turn."""

    def collect(self) -> dict[str, str]:
        """Return context values keyed by stable identifiers."""


class ScreenCapturer(Protocol):
    """Captures screenshots from the desktop."""

    def capture(self) -> CapturedScreen:
        """Capture and return the current screen state."""


class TextGenerator(Protocol):
    """Generates assistant text from prompt + screenshot."""

    def generate(self, *, prompt: str, screen: CapturedScreen) -> str:
        """Generate assistant output text."""


class SpeechSynthesizer(Protocol):
    """Converts text into WAV audio bytes."""

    def synthesize(self, text: str) -> bytes:
        """Return synthesized WAV bytes for the provided text."""


class AudioOutput(Protocol):
    """Stores and optionally plays generated audio."""

    def output(self, wav_bytes: bytes) -> Path | None:
        """Persist and optionally play audio, returning a saved path when retained."""


class VoiceTriggerListener(Protocol):
    """Waits for wake-word activation from microphone input."""

    def listen_for_activation(
        self,
        *,
        on_wake_word_detected: Callable[[], None] | None = None,
    ) -> VoiceActivation | None:
        """Return activation details when wake-word is heard, else None."""


class PresenceOverlay(Protocol):
    """Receives assistant presence state updates for UI rendering."""

    def start(self) -> None:
        """Start rendering overlay resources."""

    def stop(self) -> None:
        """Stop rendering overlay resources and release resources."""

    def set_state(self, state: AssistantPresenceState) -> None:
        """Update visible assistant presence state."""


class MemoryStore(Protocol):
    """Persists assistant memories and retrieves relevant entries."""

    def recall(self, *, query: str, limit: int) -> tuple[MemoryRecord, ...]:
        """Return memories most relevant to the query."""

    def remember(
        self,
        *,
        created_at: datetime,
        user_note: str | None,
        assistant_reply: str,
        context: Mapping[str, str],
        action_summary: str,
    ) -> None:
        """Persist one completed assistant turn."""

    def close(self) -> None:
        """Release memory-store resources."""
