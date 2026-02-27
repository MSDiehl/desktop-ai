"""Protocol interfaces for assistant components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from desktop_ai.types import CapturedScreen, VoiceActivation


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

    def listen_for_activation(self) -> VoiceActivation | None:
        """Return activation details when wake-word is heard, else None."""
