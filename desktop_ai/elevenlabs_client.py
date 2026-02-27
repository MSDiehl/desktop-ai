"""ElevenLabs speech synthesis integration."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import requests

from desktop_ai.config import ElevenLabsConfig


def parse_pcm_sample_rate(output_format: str) -> int:
    """Parse sample rate from ElevenLabs output format like `pcm_16000`."""
    try:
        _, sample_rate_text = output_format.split("_", maxsplit=1)
        return int(sample_rate_text)
    except (ValueError, TypeError):
        return 16000


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM bytes into a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


@dataclass(slots=True)
class ElevenLabsSpeechSynthesizer:
    """Synthesizes speech with ElevenLabs and returns WAV bytes."""

    config: ElevenLabsConfig

    def synthesize(self, text: str) -> bytes:
        """Generate speech audio from text."""
        cleaned_text: str = text.strip()
        if not cleaned_text:
            raise ValueError("Cannot synthesize an empty response.")

        url: str = f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.voice_id}"
        headers: dict[str, str] = {
            "xi-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        payload: dict[str, object] = {
            "text": cleaned_text,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
            },
        }
        params: dict[str, str] = {"output_format": self.config.output_format}

        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()

        sample_rate: int = parse_pcm_sample_rate(self.config.output_format)
        return pcm_to_wav_bytes(response.content, sample_rate=sample_rate)
