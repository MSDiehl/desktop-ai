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

        wake_word_match = self._wake_word_pattern.search(transcript)
        if wake_word_match is None:
            return None

        trailing_text: str = transcript[wake_word_match.end() :]
        user_note: str | None = trailing_text.lstrip(" ,:;.!?-").strip() or None
        return VoiceActivation(
            transcript=transcript,
            wake_word=self.voice_config.wake_word,
            user_note=user_note,
        )

    def _record_clip(self) -> bytes:
        """Capture raw mono PCM16 audio from the default microphone."""
        chunks: list[bytes] = []
        duration_ms: int = max(1, int(self.voice_config.listen_seconds * 1000))

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
