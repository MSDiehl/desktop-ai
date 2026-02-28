"""Assistant orchestration logic."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from desktop_ai.interfaces import (
    AudioOutput,
    ContextCollector,
    ScreenCapturer,
    SpeechSynthesizer,
    TextGenerator,
    VoiceTriggerListener,
)
from desktop_ai.prompting import build_user_prompt
from desktop_ai.types import AssistantTurnResult


@dataclass(slots=True)
class DesktopAssistant:
    """Coordinates context collection, vision-language reasoning, and TTS."""

    context_collector: ContextCollector
    screen_capturer: ScreenCapturer
    text_generator: TextGenerator
    speech_synthesizer: SpeechSynthesizer | None = None
    audio_output: AudioOutput | None = None
    voice_trigger_listener: VoiceTriggerListener | None = None
    enable_speech: bool = True
    logger: logging.Logger = logging.getLogger(__name__)

    def run_once(self, *, user_note: str | None = None) -> AssistantTurnResult:
        """Execute one full assistant turn."""
        started_at: datetime = datetime.now(timezone.utc)
        context: dict[str, str] = self.context_collector.collect()
        prompt: str = build_user_prompt(context, user_note=user_note)
        response_text: str = self.text_generator.generate(
            prompt=prompt,
            screen=self.screen_capturer.capture(),
        )

        audio_path: Path | None = None
        if (
            self.enable_speech
            and self.speech_synthesizer is not None
            and self.audio_output is not None
        ):
            wav_bytes: bytes = self.speech_synthesizer.synthesize(response_text)
            audio_path = self.audio_output.output(wav_bytes)

        finished_at: datetime = datetime.now(timezone.utc)
        return AssistantTurnResult(
            prompt=prompt,
            response_text=response_text,
            context=context,
            audio_path=audio_path,
            started_at=started_at,
            finished_at=finished_at,
        )

    def run_loop(
        self,
        *,
        interval_seconds: float,
        user_note: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        """Run assistant repeatedly at a fixed interval until stopped."""
        completed_turns: int = 0
        while True:
            turn_note: str | None = user_note
            if self.voice_trigger_listener is not None:
                activation = self.voice_trigger_listener.listen_for_activation()
                if activation is None:
                    continue
                self.logger.info('Wake word "%s" detected.', activation.wake_word)
                turn_note = activation.user_note or user_note
                if turn_note is None:
                    self.logger.info(
                        "No follow-up request captured after wake word; waiting."
                    )
                    continue

            result: AssistantTurnResult = self.run_once(user_note=turn_note)
            self.logger.info("Assistant: %s", result.response_text)
            if result.audio_path:
                self.logger.info("Audio saved to %s", result.audio_path)

            completed_turns += 1
            if max_turns is not None and completed_turns >= max_turns:
                break

            time.sleep(interval_seconds)
