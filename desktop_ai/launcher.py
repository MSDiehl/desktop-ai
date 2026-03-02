"""Tkinter launcher for editing .env and controlling assistant runtime."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tkinter as tk
from tkinter import messagebox, ttk

from dotenv import dotenv_values

from desktop_ai.assistant import DesktopAssistant
from desktop_ai.cli import build_assistant
from desktop_ai.desktop_control import format_action
from desktop_ai.types import DesktopActionPlan

FieldKind = Literal["text", "password", "bool"]


@dataclass(frozen=True, slots=True)
class SettingField:
    """Represents one editable environment variable field."""

    key: str
    label: str
    section: str
    kind: FieldKind = "text"
    width: int = 44


@dataclass(slots=True)
class ApprovalRequest:
    """Represents one GUI approval request for desktop actions."""

    plan: DesktopActionPlan
    response_event: threading.Event
    approved: bool = False


DEFAULT_ENV_VALUES: dict[str, str] = {
    "OPENAI_API_KEY": "",
    "OPENAI_MODEL": "gpt-4.1-mini",
    "OPENAI_TEMPERATURE": "0.5",
    "OPENAI_MAX_OUTPUT_TOKENS": "120",
    "OPENAI_TIMEOUT_SECONDS": "45",
    "ELEVENLABS_API_KEY": "",
    "ELEVENLABS_VOICE_ID": "",
    "ELEVENLABS_MODEL_ID": "eleven_multilingual_v2",
    "ELEVENLABS_OUTPUT_FORMAT": "pcm_16000",
    "ELEVENLABS_STABILITY": "0.45",
    "ELEVENLABS_SIMILARITY_BOOST": "0.75",
    "ELEVENLABS_TIMEOUT_SECONDS": "45",
    "ASSISTANT_INTERVAL_SECONDS": "8",
    "ASSISTANT_CONTEXT_PROVIDERS": "timestamp,environment,active_window",
    "ASSISTANT_ARTIFACTS_DIR": "./artifacts",
    "ASSISTANT_MONITOR_INDEX": "1",
    "ASSISTANT_ENABLE_SPEECH": "true",
    "ASSISTANT_ENABLE_VOICE_TRIGGER": "false",
    "ASSISTANT_WAKE_WORD": "Lune",
    "ASSISTANT_VOICE_LISTEN_SECONDS": "3.0",
    "ASSISTANT_VOICE_FOLLOWUP_LISTEN_SECONDS": "12.0",
    "ASSISTANT_VOICE_END_SILENCE_SECONDS": "1.0",
    "ASSISTANT_VOICE_ACTIVITY_THRESHOLD": "450",
    "ASSISTANT_VOICE_SAMPLE_RATE": "16000",
    "ASSISTANT_VOICE_TRANSCRIPTION_MODEL": "gpt-4o-mini-transcribe",
    "ASSISTANT_ENABLE_AVATAR": "true",
    "ASSISTANT_AVATAR_AUTO_MOVE": "true",
    "ASSISTANT_AVATAR_SIZE": "180",
    "ASSISTANT_AVATAR_OPACITY": "0.95",
    "ASSISTANT_ENABLE_DESKTOP_CONTROL": "false",
    "ASSISTANT_DESKTOP_REQUIRE_APPROVAL": "true",
    "ASSISTANT_DESKTOP_ALLOWED_LAUNCH": "notepad,code,calc",
    "ASSISTANT_DESKTOP_MAX_ACTIONS_PER_TURN": "5",
    "ASSISTANT_DESKTOP_ACTION_DELAY_SECONDS": "0.05",
    "ASSISTANT_DESKTOP_ACTION_LOG": "actions.log",
    "ASSISTANT_LOG_LEVEL": "INFO",
    "ASSISTANT_SYSTEM_PROMPT": (
        "You are Sophie, a friendly desktop AI buddy. "
        "Answer direct questions directly, avoid unsolicited productivity coaching, "
        "and give concise actionable help when the user asks for assistance."
    ),
}

SETTING_FIELDS: tuple[SettingField, ...] = (
    SettingField("OPENAI_API_KEY", "OpenAI API Key", "OpenAI", "password", 56),
    SettingField("OPENAI_MODEL", "Model", "OpenAI"),
    SettingField("OPENAI_TEMPERATURE", "Temperature", "OpenAI"),
    SettingField("OPENAI_MAX_OUTPUT_TOKENS", "Max Output Tokens", "OpenAI"),
    SettingField("OPENAI_TIMEOUT_SECONDS", "Timeout Seconds", "OpenAI"),
    SettingField("ELEVENLABS_API_KEY", "ElevenLabs API Key", "ElevenLabs", "password", 56),
    SettingField("ELEVENLABS_VOICE_ID", "Voice ID", "ElevenLabs", "password", 56),
    SettingField("ELEVENLABS_MODEL_ID", "Model ID", "ElevenLabs"),
    SettingField("ELEVENLABS_OUTPUT_FORMAT", "Output Format", "ElevenLabs"),
    SettingField("ELEVENLABS_STABILITY", "Stability", "ElevenLabs"),
    SettingField("ELEVENLABS_SIMILARITY_BOOST", "Similarity Boost", "ElevenLabs"),
    SettingField("ELEVENLABS_TIMEOUT_SECONDS", "Timeout Seconds", "ElevenLabs"),
    SettingField("ASSISTANT_INTERVAL_SECONDS", "Interval Seconds", "Runtime"),
    SettingField("ASSISTANT_CONTEXT_PROVIDERS", "Context Providers CSV", "Runtime", width=56),
    SettingField("ASSISTANT_ARTIFACTS_DIR", "Artifacts Directory", "Runtime", width=56),
    SettingField("ASSISTANT_MONITOR_INDEX", "Monitor Index", "Runtime"),
    SettingField("ASSISTANT_ENABLE_SPEECH", "Enable Speech", "Runtime", "bool"),
    SettingField("ASSISTANT_LOG_LEVEL", "Log Level", "Runtime"),
    SettingField("ASSISTANT_SYSTEM_PROMPT", "System Prompt", "Runtime", width=76),
    SettingField("ASSISTANT_ENABLE_VOICE_TRIGGER", "Enable Voice Trigger", "Voice", "bool"),
    SettingField("ASSISTANT_WAKE_WORD", "Wake Word", "Voice"),
    SettingField("ASSISTANT_VOICE_LISTEN_SECONDS", "Listen Seconds", "Voice"),
    SettingField("ASSISTANT_VOICE_FOLLOWUP_LISTEN_SECONDS", "Follow-up Listen Seconds", "Voice"),
    SettingField("ASSISTANT_VOICE_END_SILENCE_SECONDS", "End Silence Seconds", "Voice"),
    SettingField("ASSISTANT_VOICE_ACTIVITY_THRESHOLD", "Activity Threshold", "Voice"),
    SettingField("ASSISTANT_VOICE_SAMPLE_RATE", "Sample Rate", "Voice"),
    SettingField("ASSISTANT_VOICE_TRANSCRIPTION_MODEL", "Transcription Model", "Voice"),
    SettingField("ASSISTANT_ENABLE_AVATAR", "Enable Avatar", "Avatar", "bool"),
    SettingField("ASSISTANT_AVATAR_AUTO_MOVE", "Avatar Auto Move", "Avatar", "bool"),
    SettingField("ASSISTANT_AVATAR_SIZE", "Avatar Size", "Avatar"),
    SettingField("ASSISTANT_AVATAR_OPACITY", "Avatar Opacity", "Avatar"),
    SettingField("ASSISTANT_ENABLE_DESKTOP_CONTROL", "Enable Desktop Control", "Desktop Control", "bool"),
    SettingField(
        "ASSISTANT_DESKTOP_REQUIRE_APPROVAL",
        "Require Action Approval",
        "Desktop Control",
        "bool",
    ),
    SettingField(
        "ASSISTANT_DESKTOP_ALLOWED_LAUNCH",
        "Allowed Launch Commands CSV",
        "Desktop Control",
        width=56,
    ),
    SettingField("ASSISTANT_DESKTOP_MAX_ACTIONS_PER_TURN", "Max Actions Per Turn", "Desktop Control"),
    SettingField("ASSISTANT_DESKTOP_ACTION_DELAY_SECONDS", "Action Delay Seconds", "Desktop Control"),
    SettingField("ASSISTANT_DESKTOP_ACTION_LOG", "Action Log File", "Desktop Control"),
)

SECTION_ORDER: tuple[str, ...] = (
    "OpenAI",
    "ElevenLabs",
    "Runtime",
    "Voice",
    "Avatar",
    "Desktop Control",
)


def _parse_bool(value: str) -> bool:
    """Convert env text into a bool value."""
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_env_path() -> Path:
    """Resolve the .env path used by the launcher."""
    frozen: bool = getattr(sys, "frozen", False)
    if frozen:
        return Path(sys.executable).resolve().parent / ".env"

    candidates: list[Path] = []
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parent.parent / ".env")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_run_args(*, force_avatar: bool) -> argparse.Namespace:
    """Build Namespace expected by cli.build_assistant."""
    return argparse.Namespace(
        once=False,
        interval=None,
        max_turns=None,
        note=None,
        context_providers=None,
        monitor_index=None,
        no_speech=False,
        no_autoplay=False,
        voice_trigger=False,
        no_voice_trigger=False,
        wake_word=None,
        avatar=force_avatar,
        no_avatar=False,
        no_avatar_auto_move=False,
        avatar_size=None,
        avatar_opacity=None,
        desktop_control=False,
        no_desktop_control=False,
        auto_approve_actions=False,
        allow_launch=None,
        max_actions_per_turn=None,
    )


class SettingsLauncher:
    """Tkinter launcher for .env management and assistant lifecycle."""

    def __init__(self) -> None:
        """Create UI and runtime state."""
        self.env_path: Path = _resolve_env_path()
        self.managed_keys: tuple[str, ...] = tuple(field.key for field in SETTING_FIELDS)
        self.root = tk.Tk()
        self.root.title("Desktop AI Launcher")
        self.root.geometry("980x740")
        self.root.minsize(920, 680)

        self._value_vars: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self._event_queue: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._assistant_thread: threading.Thread | None = None
        self._assistant: DesktopAssistant | None = None
        self._assistant_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._running = False
        self._closing = False

        self.status_var = tk.StringVar(value=f"Ready. Editing: {self.env_path}")
        self.force_avatar_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_env_into_controls(show_message=False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.root.after(150, self._poll_worker_events)

    def _build_ui(self) -> None:
        """Build launcher controls."""
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill="both", expand=True)

        header = ttk.Label(
            container,
            text="Desktop AI Settings",
            font=("Segoe UI", 14, "bold"),
        )
        header.pack(anchor="w")

        env_label = ttk.Label(container, text=f".env file: {self.env_path}")
        env_label.pack(anchor="w", pady=(2, 8))

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        for section in SECTION_ORDER:
            section_frame = ttk.Frame(notebook, padding=12)
            notebook.add(section_frame, text=section)
            self._build_section_fields(section_frame, section)

        runtime_frame = ttk.LabelFrame(container, text="Startup Behavior", padding=10)
        runtime_frame.pack(fill="x", pady=(10, 8))
        ttk.Checkbutton(
            runtime_frame,
            text="Force avatar on power-on",
            variable=self.force_avatar_var,
        ).pack(anchor="w")
        ttk.Label(
            runtime_frame,
            text=(
                "When enabled, clicking Power On always starts with the avatar visible "
                "even if ASSISTANT_ENABLE_AVATAR is false."
            ),
            wraplength=880,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(8, 6))

        self.save_button = ttk.Button(controls, text="Save Settings", command=self._save_clicked)
        self.save_button.pack(side="left")

        self.start_button = ttk.Button(controls, text="Power On AI", command=self._start_clicked)
        self.start_button.pack(side="left", padx=(8, 0))

        self.stop_button = ttk.Button(
            controls,
            text="Stop AI",
            command=self._stop_clicked,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        reload_button = ttk.Button(controls, text="Reload .env", command=self._reload_clicked)
        reload_button.pack(side="left", padx=(8, 0))

        exit_button = ttk.Button(controls, text="Exit", command=self._exit_clicked)
        exit_button.pack(side="right")

        status = ttk.Label(container, textvariable=self.status_var)
        status.pack(fill="x", pady=(2, 0))

    def _build_section_fields(self, parent: ttk.Frame, section: str) -> None:
        """Create rows for one settings section."""
        row = 0
        for field in SETTING_FIELDS:
            if field.section != section:
                continue

            ttk.Label(parent, text=field.label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))

            if field.kind == "bool":
                variable = tk.BooleanVar(value=_parse_bool(DEFAULT_ENV_VALUES[field.key]))
                checkbox = ttk.Checkbutton(parent, variable=variable)
                checkbox.grid(row=row, column=1, sticky="w", pady=4)
                self._value_vars[field.key] = variable
            else:
                variable = tk.StringVar(value=DEFAULT_ENV_VALUES[field.key])
                show_char = "*" if field.kind == "password" else ""
                entry = ttk.Entry(parent, textvariable=variable, width=field.width, show=show_char)
                entry.grid(row=row, column=1, sticky="we", pady=4)
                self._value_vars[field.key] = variable

            row += 1

        parent.columnconfigure(1, weight=1)

    def _load_env_file_values(self) -> dict[str, str]:
        """Load plain env values from .env file."""
        if not self.env_path.exists():
            return {}
        loaded: dict[str, Any] = dict(dotenv_values(self.env_path))
        values: dict[str, str] = {}
        for key, raw_value in loaded.items():
            if raw_value is None:
                continue
            values[key] = str(raw_value)
        return values

    def _load_env_into_controls(self, *, show_message: bool) -> None:
        """Populate form controls from .env file and defaults."""
        values: dict[str, str] = dict(DEFAULT_ENV_VALUES)
        values.update(self._load_env_file_values())

        for field in SETTING_FIELDS:
            variable = self._value_vars[field.key]
            text_value: str = values.get(field.key, DEFAULT_ENV_VALUES[field.key])
            if field.kind == "bool":
                assert isinstance(variable, tk.BooleanVar)
                variable.set(_parse_bool(text_value))
            else:
                assert isinstance(variable, tk.StringVar)
                variable.set(text_value)

        if show_message:
            self.status_var.set(f"Reloaded settings from {self.env_path}")

    def _collect_form_values(self) -> dict[str, str]:
        """Collect values from controls into env string values."""
        collected: dict[str, str] = {}
        for field in SETTING_FIELDS:
            variable = self._value_vars[field.key]
            if field.kind == "bool":
                assert isinstance(variable, tk.BooleanVar)
                collected[field.key] = "true" if variable.get() else "false"
            else:
                assert isinstance(variable, tk.StringVar)
                value: str = variable.get().replace("\n", " ").strip()
                collected[field.key] = value
        return collected

    def _write_env_file(self, values: dict[str, str]) -> None:
        """Write managed env keys and preserve non-managed keys."""
        existing: dict[str, str] = self._load_env_file_values()
        extras: dict[str, str] = {
            key: value for key, value in existing.items() if key not in self.managed_keys
        }

        lines: list[str] = [f"{key}={values.get(key, '')}" for key in self.managed_keys]
        if extras:
            lines.append("")
            for key in sorted(extras):
                lines.append(f"{key}={extras[key]}")

        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _refresh_process_environment(self) -> None:
        """Reload managed variables from .env into os.environ."""
        for key in self.managed_keys:
            os.environ.pop(key, None)

        loaded: dict[str, Any] = dict(dotenv_values(self.env_path))
        for key, raw_value in loaded.items():
            if raw_value is None:
                continue
            os.environ[key] = str(raw_value)

    def _build_gui_action_approval_callback(self) -> Callable[[DesktopActionPlan], bool]:
        """Build a blocking worker-thread callback that asks for GUI approval."""

        def callback(plan: DesktopActionPlan) -> bool:
            request = ApprovalRequest(
                plan=plan,
                response_event=threading.Event(),
            )
            self._event_queue.put(("approval_request", request))

            while not request.response_event.wait(timeout=0.1):
                if self._stop_requested.is_set() or self._closing:
                    request.approved = False
                    request.response_event.set()
                    break
            return request.approved

        return callback

    def _show_action_approval_dialog(self, plan: DesktopActionPlan) -> bool:
        """Show a Yes/No dialog describing the requested desktop actions."""
        rendered_actions: list[str] = [
            f"{index}. {format_action(action)}"
            for index, action in enumerate(plan.actions, start=1)
        ]
        body: str = "\n".join(rendered_actions) if rendered_actions else "No actions listed."
        prompt: str = (
            "The assistant wants to execute these desktop actions:\n\n"
            f"{body}\n\n"
            "Approve this action plan?"
        )

        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            return False

        return bool(
            messagebox.askyesno(
                title="Approve Desktop Actions",
                message=prompt,
                parent=self.root,
                icon="warning",
            )
        )

    def _set_running_state(self, running: bool) -> None:
        """Toggle button states for active run lifecycle."""
        self._running = running
        self.start_button.config(state="disabled" if running else "normal")
        self.save_button.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")

    def _save_clicked(self) -> bool:
        """Handle Save Settings button."""
        try:
            values: dict[str, str] = self._collect_form_values()
            self._write_env_file(values)
        except Exception as error:
            messagebox.showerror("Save Failed", f"Failed to save .env:\n{error}")
            self.status_var.set(f"Save failed: {error}")
            return False

        self.status_var.set(f"Settings saved to {self.env_path}")
        return True

    def _start_clicked(self) -> None:
        """Handle Power On AI button."""
        if self._running:
            return

        if not self._save_clicked():
            return

        self._stop_requested.clear()
        self._set_running_state(True)
        self.status_var.set("Starting assistant...")
        force_avatar: bool = self.force_avatar_var.get()

        worker = threading.Thread(
            target=self._assistant_worker,
            args=(force_avatar,),
            name="desktop-ai-launcher",
            daemon=True,
        )
        self._assistant_thread = worker
        worker.start()

        self.root.after(150, self.root.iconify)

    def _stop_clicked(self) -> None:
        """Handle Stop AI button."""
        self._stop_requested.set()
        with self._assistant_lock:
            assistant = self._assistant
        if assistant is not None:
            assistant.request_stop()
        self.status_var.set("Stopping assistant...")

    def _reload_clicked(self) -> None:
        """Handle Reload .env button."""
        if self._running:
            messagebox.showinfo("Assistant Running", "Stop the assistant before reloading settings.")
            return
        self._load_env_into_controls(show_message=True)

    def _exit_clicked(self) -> None:
        """Handle Exit button."""
        self._closing = True
        if self._running:
            self._stop_clicked()
            self.status_var.set("Stopping assistant before exit...")
            self.root.after(150, self._wait_then_close)
            return
        self.root.destroy()

    def _wait_then_close(self) -> None:
        """Close app after worker stops or after timeout check cycle."""
        thread = self._assistant_thread
        if thread is not None and thread.is_alive():
            self.root.after(200, self._wait_then_close)
            return
        self.root.destroy()

    def _on_window_close(self) -> None:
        """Minimize while running so user can reopen from taskbar."""
        if self._running and not self._closing:
            self.root.iconify()
            self.status_var.set("Assistant is still running. Reopen this window from the taskbar to stop it.")
            return
        self._exit_clicked()

    def _assistant_worker(self, force_avatar: bool) -> None:
        """Run assistant loop in background thread."""
        assistant: DesktopAssistant | None = None
        try:
            self._refresh_process_environment()
            args = _default_run_args(force_avatar=force_avatar)
            assistant, config = build_assistant(args)
            with self._assistant_lock:
                self._assistant = assistant
            if assistant.desktop_controller is not None and assistant.desktop_controller.require_approval:
                assistant.action_approval_callback = self._build_gui_action_approval_callback()

            if self._stop_requested.is_set():
                assistant.request_stop()

            self._event_queue.put(
                (
                    "started",
                    f"Assistant running. Interval: {config.interval_seconds:.1f}s",
                )
            )
            assistant.run_loop(interval_seconds=config.interval_seconds)
            self._event_queue.put(("stopped", "Assistant stopped."))
        except Exception as error:
            self._event_queue.put(("error", str(error)))
        finally:
            if assistant is not None:
                assistant.close()
            with self._assistant_lock:
                self._assistant = None

    def _poll_worker_events(self) -> None:
        """Drain worker events and update UI state."""
        while True:
            try:
                event, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break

            if event == "started":
                self.status_var.set(str(payload))
                continue

            if event == "approval_request":
                if not isinstance(payload, ApprovalRequest):
                    continue
                if payload.response_event.is_set():
                    continue
                self.status_var.set("Waiting for desktop action approval...")
                approved: bool = self._show_action_approval_dialog(payload.plan)
                payload.approved = approved
                payload.response_event.set()
                self.status_var.set(
                    "Desktop actions approved." if approved else "Desktop actions denied."
                )
                continue

            self._set_running_state(False)

            if event == "stopped":
                self.status_var.set(str(payload))
                if self._closing:
                    self.root.after(50, self._wait_then_close)
                continue

            if event == "error":
                self.status_var.set(f"Start/run error: {payload}")
                self.root.deiconify()
                self.root.lift()
                messagebox.showerror("Assistant Error", str(payload))
                if self._closing:
                    self.root.after(50, self._wait_then_close)

        if not self.root.winfo_exists():
            return
        try:
            self.root.after(150, self._poll_worker_events)
        except tk.TclError:
            return

    def run(self) -> None:
        """Run launcher event loop."""
        self.root.mainloop()


def main() -> int:
    """Launcher entrypoint."""
    launcher = SettingsLauncher()
    launcher.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
