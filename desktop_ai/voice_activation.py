"""Wake-word microphone listener backed by OpenAI transcription."""

from __future__ import annotations

import io
import logging
import re
import wave
from dataclasses import dataclass, field
from typing import Any

from desktop_ai.config import OpenAIConfig, VoiceTriggerConfig
from desktop_ai.types import VoiceActivation

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAIWakeWordListener:
    """Records short microphone clips and detects wake-word phrases."""

    openai_config: OpenAIConfig
    voice_config: VoiceTriggerConfig
    logger: logging.Logger = LOGGER
    _client: Any = field(init=False, repr=False)
    _sounddevice: Any = field(init=False, repr=False)
    _wake_word_pattern: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize OpenAI and microphone dependencies."""
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("openai package is required for voice activation.") from error

        try:
            import sounddevice
        except ImportError as error:
            raise RuntimeError("sounddevice package is required for voice activation.") from error

        self._client = OpenAI(
            api_key=self.openai_config.api_key,
            timeout=self.openai_config.timeout_seconds,
        )
        self._sounddevice = sounddevice
        self._wake_word_pattern = re.compile(
            rf"\b{re.escape(self.voice_config.wake_word)}\b",
            flags=re.IGNORECASE,
        )

    def listen_for_activation(self) -> VoiceActivation | None:
        """Listen to microphone and return activation when wake-word is heard."""
        try:
            pcm_audio: bytes = self._record_clip()
            transcript: str = self._transcribe(pcm_audio)
        except Exception as error:
            self.logger.warning("Voice activation listen failed: %s", error)
            return None

        if not transcript:
            return None

        if self._wake_word_pattern.search(transcript) is None:
            return None

        self.logger.debug("Wake-word transcript: %s", transcript)
        user_note: str | None = self._extract_user_note_after_wake_word(transcript)
        if user_note is None:
            # If wake word was detected but no trailing text was captured, record a short
            # follow-up clip so the spoken request after the wake word is not lost.
            try:
                followup_pcm: bytes = self._record_clip(
                    duration_seconds=self.voice_config.followup_listen_seconds
                )
                followup_transcript: str = self._transcribe(followup_pcm)
            except Exception as error:
                self.logger.warning("Voice activation follow-up listen failed: %s", error)
                followup_transcript = ""

            followup_note: str | None = self._clean_user_note(followup_transcript)
            if followup_note:
                transcript = f"{transcript} {followup_transcript}".strip()
                user_note = followup_note

        self.logger.debug("Wake-word user note: %s", user_note)
        return VoiceActivation(
            transcript=transcript,
            wake_word=self.voice_config.wake_word,
            user_note=user_note,
        )

    def _record_clip(self, *, duration_seconds: float | None = None) -> bytes:
        """Capture raw mono PCM16 audio from the default microphone."""
        chunks: list[bytes] = []
        seconds: float = (
            duration_seconds if duration_seconds is not None else self.voice_config.listen_seconds
        )
        duration_ms: int = max(1, int(seconds * 1000))

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            _ = frames, time_info
            if status:
                self.logger.debug("Microphone status: %s", status)
            chunks.append(bytes(indata))

        with self._sounddevice.RawInputStream(
            samplerate=self.voice_config.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            self._sounddevice.sleep(duration_ms)

        return b"".join(chunks)

    def _extract_user_note_after_wake_word(self, transcript: str) -> str | None:
        """Return trailing user request after the last wake-word occurrence."""
        matches: list[re.Match[str]] = list(self._wake_word_pattern.finditer(transcript))
        if not matches:
            return None
        trailing_text: str = transcript[matches[-1].end() :]
        return self._clean_user_note(trailing_text)

    def _clean_user_note(self, text: str) -> str | None:
        """Normalize a transcribed user request snippet."""
        cleaned: str = text.lstrip(" ,:;.!?-").strip()
        return cleaned or None

    def _transcribe(self, pcm_audio: bytes) -> str:
        """Transcribe WAV audio through OpenAI and return plain text."""
        if not pcm_audio:
            return ""

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.voice_config.sample_rate)
            wav_file.writeframes(pcm_audio)
        wav_buffer.seek(0)
        wav_buffer.name = "wake-word.wav"  # type: ignore[attr-defined]

        result: Any = self._client.audio.transcriptions.create(
            model=self.voice_config.transcription_model,
            file=wav_buffer,
        )
        text: str | None = getattr(result, "text", None)
        return text.strip() if text else ""
