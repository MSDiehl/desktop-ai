"""Assistant orchestration logic."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from desktop_ai.desktop_control import DesktopController, format_action
from desktop_ai.interfaces import (
    AudioOutput,
    ContextCollector,
    PresenceOverlay,
    ScreenCapturer,
    SpeechSynthesizer,
    TextGenerator,
    VoiceTriggerListener,
)
from desktop_ai.prompting import build_user_prompt
from desktop_ai.types import (
    ActionExecutionResult,
    AssistantPresenceState,
    AssistantTurnResult,
    DesktopActionPlan,
)


@dataclass(slots=True)
class DesktopAssistant:
    """Coordinates context collection, vision-language reasoning, and TTS."""

    context_collector: ContextCollector
    screen_capturer: ScreenCapturer
    text_generator: TextGenerator
    speech_synthesizer: SpeechSynthesizer | None = None
    audio_output: AudioOutput | None = None
    voice_trigger_listener: VoiceTriggerListener | None = None
    presence_overlay: PresenceOverlay | None = None
    desktop_controller: DesktopController | None = None
    action_approval_callback: Callable[[DesktopActionPlan], bool] | None = None
    enable_speech: bool = True
    logger: logging.Logger = logging.getLogger(__name__)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def run_once(self, *, user_note: str | None = None) -> AssistantTurnResult:
        """Execute one full assistant turn."""
        self._set_presence_state("thinking")
        started_at: datetime = datetime.now(timezone.utc)
        try:
            context: dict[str, str] = self.context_collector.collect()
            desktop_control_enabled: bool = self.desktop_controller is not None
            max_actions_per_turn: int = (
                self.desktop_controller.max_actions_per_turn if self.desktop_controller is not None else 5
            )
            allowed_launch_commands: tuple[str, ...] = (
                self.desktop_controller.allowed_launch_commands
                if self.desktop_controller is not None
                else ()
            )
            prompt: str = build_user_prompt(
                context,
                user_note=user_note,
                desktop_control_enabled=desktop_control_enabled,
                max_actions_per_turn=max_actions_per_turn,
                allowed_launch_commands=allowed_launch_commands,
            )
            raw_response_text: str = self.text_generator.generate(
                prompt=prompt,
                screen=self.screen_capturer.capture(),
            )

            action_plan: DesktopActionPlan | None = None
            action_results: tuple[ActionExecutionResult, ...] = ()
            response_text: str = raw_response_text
            if self.desktop_controller is not None:
                action_plan = self.desktop_controller.parse_plan(raw_response_text)
                response_text = action_plan.spoken_reply
                if action_plan.actions:
                    action_results = self.desktop_controller.execute(
                        action_plan,
                        approval_callback=self.action_approval_callback,
                    )
                    if action_results:
                        summary: str = ", ".join(format_action(result.action) for result in action_results)
                        self.logger.info("Desktop actions attempted: %s", summary)

            audio_path: Path | None = None
            if (
                self.enable_speech
                and self.speech_synthesizer is not None
                and self.audio_output is not None
            ):
                self._set_presence_state("speaking")
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
                action_plan=action_plan,
                action_results=action_results,
            )
        finally:
            self._set_presence_state("idle")

    def run_loop(
        self,
        *,
        interval_seconds: float,
        user_note: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        """Run assistant repeatedly at a fixed interval until stopped."""
        completed_turns: int = 0
        self._stop_event.clear()
        self._set_presence_state("idle")
        while not self._stop_event.is_set():
            turn_note: str | None = user_note
            if self.voice_trigger_listener is not None:
                self._set_presence_state("idle")
                activation = self.voice_trigger_listener.listen_for_activation(
                    on_wake_word_detected=self._on_wake_word_detected,
                )
                if self._stop_event.is_set():
                    break
                if activation is None:
                    continue
                self.logger.info('Wake word "%s" detected.', activation.wake_word)
                turn_note = activation.user_note or user_note
                if turn_note is None:
                    self.logger.info(
                        "No follow-up request captured after wake word; asking user to repeat."
                    )
                    turn_note = (
                        "I said the wake word, but my follow-up request was not captured. "
                        "Please ask me to repeat, in one short sentence."
                    )

            result: AssistantTurnResult = self.run_once(user_note=turn_note)
            self.logger.info("Assistant: %s", result.response_text)
            if result.audio_path:
                self.logger.info("Audio saved to %s", result.audio_path)

            completed_turns += 1
            if max_turns is not None and completed_turns >= max_turns:
                break

            self._stop_event.wait(timeout=max(0.0, interval_seconds))

    def close(self) -> None:
        """Close assistant-owned resources."""
        self.request_stop()
        if self.presence_overlay is None:
            return
        try:
            self.presence_overlay.stop()
        except Exception as error:
            self.logger.warning("Failed to stop presence overlay: %s", error)

    def request_stop(self) -> None:
        """Signal run_loop to stop at the next safe checkpoint."""
        self._stop_event.set()

    def _set_presence_state(self, state: AssistantPresenceState) -> None:
        """Safely push state to overlay if configured."""
        if self.presence_overlay is None:
            return
        try:
            self.presence_overlay.set_state(state)
        except Exception as error:
            self.logger.warning("Failed to update presence state %s: %s", state, error)

    def _on_wake_word_detected(self) -> None:
        """Handle wake-word detection as early as possible."""
        self._set_presence_state("listening")
