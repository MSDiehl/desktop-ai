"""Assistant orchestration logic."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from desktop_ai.desktop_control import DesktopController, format_action
from desktop_ai.interfaces import (
    AudioOutput,
    ContextCollector,
    MemoryStore,
    PresenceOverlay,
    ScreenCapturer,
    SpeechSynthesizer,
    TextGenerator,
    VoiceTriggerListener,
)
from desktop_ai.prompting import build_action_repair_prompt, build_user_prompt
from desktop_ai.types import (
    ActionExecutionResult,
    AssistantPresenceState,
    AssistantTurnResult,
    DesktopActionPlan,
    MemoryRecord,
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
    memory_store: MemoryStore | None = None
    memory_recall_limit: int = 6
    memory_prompt_chars: int = 240
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
            recalled_memories: tuple[MemoryRecord, ...] = self._recall_memories(
                user_note=user_note,
                context=context,
            )
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
                memories=recalled_memories,
                desktop_control_enabled=desktop_control_enabled,
                max_actions_per_turn=max_actions_per_turn,
                allowed_launch_commands=allowed_launch_commands,
                memory_entry_chars=self.memory_prompt_chars,
            )
            captured_screen = self.screen_capturer.capture()
            raw_response_text: str = self.text_generator.generate(
                prompt=prompt,
                screen=captured_screen,
            )

            action_plan: DesktopActionPlan | None = None
            action_results: tuple[ActionExecutionResult, ...] = ()
            response_text: str = raw_response_text
            if self.desktop_controller is not None:
                action_plan = self.desktop_controller.parse_plan(raw_response_text)
                if self._should_repair_action_plan(
                    raw_response=raw_response_text,
                    action_plan=action_plan,
                ):
                    self.logger.info(
                        "Retrying desktop action plan due to unstructured response."
                    )
                    repaired_prompt: str = build_action_repair_prompt(
                        raw_response=raw_response_text,
                        user_note=user_note,
                        max_actions_per_turn=max_actions_per_turn,
                        allowed_launch_commands=allowed_launch_commands,
                    )
                    repaired_raw_response: str = self.text_generator.generate(
                        prompt=repaired_prompt,
                        screen=captured_screen,
                    )
                    repaired_plan: DesktopActionPlan = self.desktop_controller.parse_plan(
                        repaired_raw_response
                    )
                    if self._is_structured_action_plan(
                        raw_response=repaired_raw_response,
                        action_plan=repaired_plan,
                    ):
                        self.logger.info("Desktop action plan repair succeeded.")
                        raw_response_text = repaired_raw_response
                        action_plan = repaired_plan
                    elif (
                        self._looks_like_action_narration(raw_response_text)
                        or self._looks_like_broken_action_json(raw_response_text)
                    ):
                        self.logger.warning(
                            "Suppressing narrated action reply after repair failure."
                        )
                        action_plan = DesktopActionPlan(
                            spoken_reply=(
                                "I couldn't execute that reliably. "
                                "Please ask again and I'll perform it directly."
                            ),
                            actions=(),
                            raw_response=raw_response_text,
                        )
                response_text = action_plan.spoken_reply
                if action_plan.actions:
                    action_results = self.desktop_controller.execute(
                        action_plan,
                        approval_callback=self.action_approval_callback,
                    )
                    if action_results:
                        summary: str = ", ".join(format_action(result.action) for result in action_results)
                        self.logger.info("Desktop actions attempted: %s", summary)

            self._remember_turn(
                created_at=started_at,
                user_note=user_note,
                response_text=response_text,
                context=context,
                action_plan=action_plan,
                action_results=action_results,
            )

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
                recalled_memories=recalled_memories,
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
        pending_followup_note: str | None = None
        self._stop_event.clear()
        self._set_presence_state("idle")
        while not self._stop_event.is_set():
            turn_note: str | None
            if pending_followup_note is not None:
                turn_note = pending_followup_note
                pending_followup_note = None
            else:
                turn_note = user_note
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

            if self._should_listen_for_immediate_followup(result.response_text):
                followup_note: str | None = self._listen_for_immediate_followup_note()
                if self._stop_event.is_set():
                    break
                if followup_note is not None:
                    pending_followup_note = followup_note
                    continue

            self._stop_event.wait(timeout=max(0.0, interval_seconds))

    def close(self) -> None:
        """Close assistant-owned resources."""
        self.request_stop()
        if self.memory_store is not None:
            try:
                self.memory_store.close()
            except Exception as error:
                self.logger.warning("Failed to close memory store: %s", error)
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

    def _recall_memories(
        self,
        *,
        user_note: str | None,
        context: Mapping[str, str],
    ) -> tuple[MemoryRecord, ...]:
        """Recall relevant memories for the current turn."""
        if self.memory_store is None or self.memory_recall_limit <= 0:
            return ()
        query: str = self._build_memory_query(user_note=user_note, context=context)
        try:
            return self.memory_store.recall(query=query, limit=self.memory_recall_limit)
        except Exception as error:
            self.logger.warning("Failed to recall memories: %s", error)
            return ()

    def _build_memory_query(self, *, user_note: str | None, context: Mapping[str, str]) -> str:
        """Build a memory-recall query from user note and key context hints."""
        if user_note and user_note.strip():
            return user_note.strip()

        query_parts: list[str] = []
        for key in ("active_window.title", "active_window.process_name", "environment.cwd"):
            value: str | None = context.get(key)
            if value:
                query_parts.append(value)
        return " ".join(query_parts).strip()

    def _remember_turn(
        self,
        *,
        created_at: datetime,
        user_note: str | None,
        response_text: str,
        context: Mapping[str, str],
        action_plan: DesktopActionPlan | None,
        action_results: tuple[ActionExecutionResult, ...],
    ) -> None:
        """Persist a completed turn to memory."""
        if self.memory_store is None:
            return
        action_summary: str = self._build_action_summary(
            action_plan=action_plan,
            action_results=action_results,
        )
        try:
            self.memory_store.remember(
                created_at=created_at,
                user_note=user_note,
                assistant_reply=response_text,
                context=context,
                action_summary=action_summary,
            )
        except Exception as error:
            self.logger.warning("Failed to persist turn memory: %s", error)

    def _build_action_summary(
        self,
        *,
        action_plan: DesktopActionPlan | None,
        action_results: tuple[ActionExecutionResult, ...],
    ) -> str:
        """Render action outcomes into one short memory string."""
        if action_results:
            rendered_results: list[str] = []
            for result in action_results:
                status: str = "ok" if result.success else "failed"
                rendered_results.append(
                    f"{status}: {format_action(result.action)} ({result.detail})"
                )
            return "; ".join(rendered_results)

        if action_plan is None or not action_plan.actions:
            return ""

        planned: str = ", ".join(format_action(action) for action in action_plan.actions)
        if self.desktop_controller is not None and self.desktop_controller.require_approval:
            return f"Requested actions were denied or skipped: {planned}"
        return f"Planned actions: {planned}"

    def _should_repair_action_plan(
        self,
        *,
        raw_response: str,
        action_plan: DesktopActionPlan,
    ) -> bool:
        """Decide whether to retry action planning for likely narrated steps."""
        if action_plan.actions:
            return False
        return (
            not self._is_structured_action_plan(raw_response=raw_response, action_plan=action_plan)
            and (
                self._looks_like_action_narration(raw_response)
                or self._looks_like_broken_action_json(raw_response)
            )
        )

    def _is_structured_action_plan(
        self,
        *,
        raw_response: str,
        action_plan: DesktopActionPlan,
    ) -> bool:
        """Check whether a response appears to have parsed from structured JSON."""
        if action_plan.actions:
            return True
        return action_plan.spoken_reply.strip() != raw_response.strip()

    def _looks_like_action_narration(self, text: str) -> bool:
        """Heuristically detect natural-language action lists like click/type/press."""
        normalized: str = " " + text.strip().lower() + " "
        if not normalized.strip():
            return False
        action_verbs: tuple[str, ...] = (
            "click",
            "type",
            "press",
            "enter",
            "hotkey",
            "scroll",
            "drag",
            "move",
            "launch",
            "open",
            "wait",
        )
        matches = re.findall(r"\b(" + "|".join(action_verbs) + r")\b", normalized)
        if len(matches) < 2:
            return False
        sequencing_markers: tuple[str, ...] = (
            " then ",
            " next ",
            " after ",
            " first ",
            " second ",
            " third ",
            " finally ",
            ",",
        )
        return any(marker in normalized for marker in sequencing_markers)

    def _looks_like_broken_action_json(self, text: str) -> bool:
        """Detect likely truncated or malformed JSON action plans."""
        normalized: str = text.strip().lower()
        if not normalized:
            return False
        if "spoken_reply" not in normalized or "actions" not in normalized:
            return False
        return ("{" in normalized or "[" in normalized) and ("\"" in normalized)

    def _should_listen_for_immediate_followup(self, response_text: str) -> bool:
        """Decide whether to immediately listen for a follow-up user reply."""
        if self.voice_trigger_listener is None:
            return False
        normalized: str = response_text.strip()
        if not normalized:
            return False
        if "?" in normalized:
            return True
        followup_cues: tuple[str, ...] = (
            "let me know",
            "tell me",
            "want me to",
            "which one",
            "what would you like",
        )
        lowered: str = normalized.lower()
        return any(cue in lowered for cue in followup_cues)

    def _listen_for_immediate_followup_note(self) -> str | None:
        """Listen once for immediate user follow-up and return normalized text."""
        if self.voice_trigger_listener is None:
            return None
        self._set_presence_state("listening")
        try:
            note: str | None = self.voice_trigger_listener.listen_for_followup_note()
        except Exception as error:
            self.logger.warning("Immediate follow-up listen failed: %s", error)
            return None
        normalized: str = note.strip() if note else ""
        if not normalized:
            return None
        self.logger.info("Captured immediate follow-up response.")
        return normalized

    def _on_wake_word_detected(self) -> None:
        """Handle wake-word detection as early as possible."""
        self._set_presence_state("listening")
