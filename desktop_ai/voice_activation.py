"""Wake-word microphone listener backed by OpenAI transcription."""

from __future__ import annotations

import io
import logging
import re
import wave
from array import array
from collections.abc import Callable
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

    def listen_for_activation(
        self,
        *,
        on_wake_word_detected: Callable[[], None] | None = None,
    ) -> VoiceActivation | None:
        """Listen to microphone and return activation when wake-word is heard."""
        try:
            # Respect configured detection window so longer first utterances are not
            # prematurely cut off before the follow-up capture starts.
            detection_seconds: float = max(0.8, self.voice_config.listen_seconds)
            pcm_audio: bytes = self._record_clip(duration_seconds=detection_seconds)
            transcript: str = self._transcribe(pcm_audio)
        except Exception as error:
            self.logger.warning("Voice activation listen failed: %s", error)
            return None

        if not transcript:
            return None

        if self._wake_word_pattern.search(transcript) is None:
            return None

        if on_wake_word_detected is not None:
            try:
                on_wake_word_detected()
            except Exception as error:
                self.logger.debug("Wake-word callback failed: %s", error)

        self.logger.debug("Wake-word transcript: %s", transcript)
        initial_note: str | None = self._extract_user_note_after_wake_word(transcript)

        # Always capture a follow-up segment after wake detection so we continue
        # listening until trailing silence even if the first clip captured partial text.
        try:
            followup_pcm: bytes = self._record_clip(
                duration_seconds=self.voice_config.followup_listen_seconds,
                stop_on_silence=True,
            )
            followup_transcript: str = self._transcribe(followup_pcm)
        except Exception as error:
            self.logger.warning("Voice activation follow-up listen failed: %s", error)
            followup_transcript = ""

        followup_note: str | None = self._extract_user_note_after_wake_word(followup_transcript)
        if followup_note is None:
            followup_note = self._clean_user_note(followup_transcript)

        if followup_transcript:
            transcript = f"{transcript} {followup_transcript}".strip()

        user_note: str | None = self._merge_user_notes(
            initial_note=initial_note,
            followup_note=followup_note,
        )

        self.logger.debug("Wake-word user note: %s", user_note)
        return VoiceActivation(
            transcript=transcript,
            wake_word=self.voice_config.wake_word,
            user_note=user_note,
        )

    def _record_clip(
        self,
        *,
        duration_seconds: float | None = None,
        stop_on_silence: bool = False,
    ) -> bytes:
        """Capture raw mono PCM16 audio from the default microphone."""
        seconds: float = (
            duration_seconds if duration_seconds is not None else self.voice_config.listen_seconds
        )
        if stop_on_silence:
            return self._record_until_silence(max_duration_seconds=seconds)
        return self._record_fixed_duration(duration_seconds=seconds)

    def _record_fixed_duration(self, *, duration_seconds: float) -> bytes:
        """Capture PCM16 audio for a fixed duration."""
        sample_rate: int = self.voice_config.sample_rate
        block_frames: int = max(1, int(sample_rate * 0.05))
        target_frames: int = max(1, int(duration_seconds * sample_rate))
        captured_frames: int = 0
        chunks: list[bytes] = []

        with self._sounddevice.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_frames,
        ) as stream:
            while captured_frames < target_frames:
                frames_to_read: int = min(block_frames, target_frames - captured_frames)
                indata, status = stream.read(frames_to_read)
                if status:
                    self.logger.debug("Microphone status: %s", status)
                chunks.append(bytes(indata))
                captured_frames += frames_to_read

        return b"".join(chunks)

    def _record_until_silence(self, *, max_duration_seconds: float) -> bytes:
        """Capture PCM16 audio until trailing silence after speech or timeout."""
        sample_rate: int = self.voice_config.sample_rate
        max_seconds: float = max(0.5, max_duration_seconds)
        silence_seconds: float = max(0.1, self.voice_config.end_silence_seconds)
        lead_in_seconds: float = max(
            0.1,
            self.voice_config.start_silence_seconds,
            silence_seconds * 2.0,
        )
        silence_threshold: float = self.voice_config.activity_threshold

        block_frames: int = max(1, int(sample_rate * 0.05))
        max_frames: int = max(1, int(max_seconds * sample_rate))
        silence_limit_frames: int = max(1, int(silence_seconds * sample_rate))
        lead_in_limit_frames: int = max(1, int(lead_in_seconds * sample_rate))

        captured_frames: int = 0
        silence_frames: int = 0
        heard_voice: bool = False
        chunks: list[bytes] = []

        with self._sounddevice.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_frames,
        ) as stream:
            while captured_frames < max_frames:
                frames_to_read: int = min(block_frames, max_frames - captured_frames)
                indata, status = stream.read(frames_to_read)
                if status:
                    self.logger.debug("Microphone status: %s", status)

                chunk_bytes: bytes = bytes(indata)
                chunks.append(chunk_bytes)
                captured_frames += frames_to_read

                level: float = self._mean_abs_level(chunk_bytes)
                if level >= silence_threshold:
                    heard_voice = True
                    silence_frames = 0
                    continue

                if heard_voice:
                    silence_frames += frames_to_read
                    if silence_frames >= silence_limit_frames:
                        break
                elif captured_frames >= lead_in_limit_frames:
                    break

        return b"".join(chunks)

    def _mean_abs_level(self, pcm_chunk: bytes) -> float:
        """Return average absolute sample level for a PCM16 chunk."""
        if not pcm_chunk:
            return 0.0
        samples = array("h")
        samples.frombytes(pcm_chunk)
        if not samples:
            return 0.0
        return sum(abs(sample) for sample in samples) / len(samples)

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

    def _merge_user_notes(
        self,
        *,
        initial_note: str | None,
        followup_note: str | None,
    ) -> str | None:
        """Merge notes from detection and follow-up clips with simple overlap handling."""
        if initial_note is None:
            return followup_note
        if followup_note is None:
            return initial_note

        initial_lower: str = initial_note.lower()
        followup_lower: str = followup_note.lower()
        if followup_lower.startswith(initial_lower):
            return followup_note
        if initial_lower.startswith(followup_lower):
            return initial_note
        return f"{initial_note} {followup_note}".strip()

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
