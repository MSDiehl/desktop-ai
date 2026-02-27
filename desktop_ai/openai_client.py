"""OpenAI text-generation integration."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from desktop_ai.config import OpenAIConfig
from desktop_ai.types import CapturedScreen


@dataclass(slots=True)
class OpenAITextGenerator:
    """Generates context-aware text using OpenAI Responses API."""

    config: OpenAIConfig
    system_prompt: str
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the OpenAI client lazily at construction."""
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("openai package is required for text generation.") from error

        self._client = OpenAI(
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )

    def generate(self, *, prompt: str, screen: CapturedScreen) -> str:
        """Generate text from prompt + screenshot."""
        image_data_url: str = self._build_data_url(screen)
        response: Any = self._client.responses.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
        )
        return self._extract_text(response)

    def _build_data_url(self, screen: CapturedScreen) -> str:
        """Encode PNG bytes as a data URL for OpenAI image input."""
        encoded: str = base64.b64encode(screen.png_bytes).decode("ascii")
        return f"data:{screen.mime_type};base64,{encoded}"

    def _extract_text(self, response: Any) -> str:
        """Extract plain text from an OpenAI response object."""
        output_text: str | None = getattr(response, "output_text", None)
        if output_text and output_text.strip():
            return output_text.strip()

        output: list[Any] = getattr(response, "output", [])
        chunks: list[str] = []
        for item in output:
            content_items: list[Any] = getattr(item, "content", [])
            for content in content_items:
                text: str | None = getattr(content, "text", None)
                if text:
                    chunks.append(text.strip())

        if chunks:
            return " ".join(chunk for chunk in chunks if chunk)
        raise RuntimeError("OpenAI response did not contain text output.")
