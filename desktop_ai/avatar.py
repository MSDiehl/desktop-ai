"""Always-on-top desktop avatar overlay with simple state animations."""

from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from desktop_ai.types import AssistantPresenceState

LOGGER = logging.getLogger(__name__)

_VALID_STATES: set[str] = {"idle", "listening", "thinking", "speaking"}


@dataclass(slots=True)
class TkAvatarOverlay:
    """Small Tkinter avatar that reflects assistant state."""

    auto_move: bool = True
    size: int = 180
    opacity: float = 0.95
    title: str = "Desktop AI"
    logger: logging.Logger = LOGGER
    _commands: queue.SimpleQueue[tuple[str, Any]] = field(
        default_factory=queue.SimpleQueue,
        init=False,
        repr=False,
    )
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _running: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        """Start the avatar UI loop in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._commands = queue.SimpleQueue()
        self._ready.clear()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_ui_loop,
            name="desktop-ai-avatar",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.logger.warning("Avatar overlay did not start within timeout.")
        elif self._running.is_set():
            self.set_state("idle")

    def stop(self) -> None:
        """Stop the avatar UI loop."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._commands.put(("stop", None))
        self._thread.join(timeout=2.0)
        self._running.clear()
        self._thread = None

    def set_state(self, state: AssistantPresenceState) -> None:
        """Queue a visible state update for the avatar."""
        if state not in _VALID_STATES:
            return
        if self._thread is None or not self._thread.is_alive():
            return
        self._commands.put(("state", state))

    def _run_ui_loop(self) -> None:
        """Run Tkinter event loop and animations."""
        try:
            import tkinter as tk
        except Exception as error:
            self._running.clear()
            self._ready.set()
            self.logger.warning("Avatar overlay unavailable (tkinter import failed): %s", error)
            return

        try:
            root = tk.Tk()
            root.title(self.title)
            root.attributes("-topmost", True)
            root.resizable(False, False)
            root.overrideredirect(True)
            if 0.0 < self.opacity < 1.0:
                root.attributes("-alpha", self.opacity)

            window_size: int = max(100, self.size)
            screen_width: int = root.winfo_screenwidth()
            screen_height: int = root.winfo_screenheight()
            x: int = max(0, screen_width - window_size - 80)
            y: int = max(0, screen_height - window_size - 140)
            root.geometry(f"{window_size}x{window_size}+{x}+{y}")
            transparent_color: str = "#00FF00"
            canvas_background: str = "#0D1117"
            try:
                root.wm_attributes("-transparentcolor", transparent_color)
                root.configure(bg=transparent_color)
                canvas_background = transparent_color
            except Exception:
                root.configure(bg=canvas_background)

            canvas = tk.Canvas(
                root,
                width=window_size,
                height=window_size,
                bg=canvas_background,
                highlightthickness=0,
            )
            canvas.pack(fill="both", expand=True)

            ring_id = canvas.create_oval(
                18,
                18,
                window_size - 18,
                window_size - 18,
                outline="#6E7681",
                width=5,
            )
            core_id = canvas.create_oval(
                34,
                34,
                window_size - 34,
                window_size - 34,
                fill="#3F4A5A",
                outline="",
            )

            eye_size: int = 8
            eye_y: int = window_size // 2 - 16
            left_eye_x: int = window_size // 2 - 24
            right_eye_x: int = window_size // 2 + 24
            canvas.create_oval(
                left_eye_x - eye_size // 2,
                eye_y - eye_size // 2,
                left_eye_x + eye_size // 2,
                eye_y + eye_size // 2,
                fill="#F0F6FC",
                outline="",
            )
            canvas.create_oval(
                right_eye_x - eye_size // 2,
                eye_y - eye_size // 2,
                right_eye_x + eye_size // 2,
                eye_y + eye_size // 2,
                fill="#F0F6FC",
                outline="",
            )

            bars: list[int] = []
            bar_width: int = 10
            bar_spacing: int = 4
            total_width: int = (bar_width * 4) + (bar_spacing * 3)
            bars_start_x: int = (window_size - total_width) // 2
            bars_bottom_y: int = window_size - 46
            for index in range(4):
                x1: int = bars_start_x + index * (bar_width + bar_spacing)
                bar_id = canvas.create_rectangle(
                    x1,
                    bars_bottom_y - 4,
                    x1 + bar_width,
                    bars_bottom_y,
                    fill="#7EE787",
                    outline="",
                    state="hidden",
                )
                bars.append(bar_id)

            label_id = canvas.create_text(
                window_size // 2,
                window_size - 22,
                text="IDLE",
                fill="#C9D1D9",
                font=("Segoe UI", 11, "bold"),
            )

            state: AssistantPresenceState = "idle"
            phase: float = 0.0
            last_tick: float = time.monotonic()
            auto_resume_at: float = 0.0
            position_x: float = float(x)
            position_y: float = float(y)
            target_x: float = float(x)
            target_y: float = float(y)
            next_target_at: float = 0.0
            drag_offset_x: int = 0
            drag_offset_y: int = 0
            dragging: bool = False

            state_colors: dict[str, tuple[str, str]] = {
                "idle": ("#3F4A5A", "#6E7681"),
                "listening": ("#1F6FEB", "#58A6FF"),
                "thinking": ("#B08900", "#E3B341"),
                "speaking": ("#238636", "#7EE787"),
            }

            def on_press(event: Any) -> None:
                nonlocal drag_offset_x, drag_offset_y, dragging, auto_resume_at
                dragging = True
                drag_offset_x = event.x_root - root.winfo_x()
                drag_offset_y = event.y_root - root.winfo_y()
                auto_resume_at = time.monotonic() + 4.0

            def on_motion(event: Any) -> None:
                nonlocal auto_resume_at, position_x, position_y, target_x, target_y
                if not dragging:
                    return
                next_x: int = max(0, event.x_root - drag_offset_x)
                next_y: int = max(0, event.y_root - drag_offset_y)
                root.geometry(f"+{next_x}+{next_y}")
                position_x = float(next_x)
                position_y = float(next_y)
                target_x = position_x
                target_y = position_y
                auto_resume_at = time.monotonic() + 4.0

            def on_release(event: Any) -> None:
                _ = event
                nonlocal dragging, target_x, target_y, next_target_at
                dragging = False
                target_x = float(root.winfo_x())
                target_y = float(root.winfo_y())
                next_target_at = time.monotonic() + 0.6

            def on_secondary_click(event: Any) -> None:
                _ = event
                self._commands.put(("stop", None))

            for widget in (root, canvas):
                widget.bind("<ButtonPress-1>", on_press)
                widget.bind("<B1-Motion>", on_motion)
                widget.bind("<ButtonRelease-1>", on_release)
                widget.bind("<Button-3>", on_secondary_click)

            def handle_commands() -> bool:
                nonlocal state
                while True:
                    try:
                        command, payload = self._commands.get_nowait()
                    except queue.Empty:
                        break

                    if command == "stop":
                        root.destroy()
                        return False
                    if command == "state" and payload in _VALID_STATES:
                        state = payload
                return True

            def tick() -> None:
                nonlocal phase, last_tick, position_x, position_y, target_x, target_y, next_target_at

                if not handle_commands():
                    return

                now: float = time.monotonic()
                delta: float = max(0.001, now - last_tick)
                last_tick = now
                phase += delta * 4.2

                core_color, ring_color = state_colors[state]
                ring_pulse: float = 0.0
                if state == "listening":
                    ring_pulse = 4.5 * (0.5 + 0.5 * math.sin(phase * 1.6))
                elif state == "speaking":
                    ring_pulse = 2.5 * (0.5 + 0.5 * math.sin(phase * 3.5))
                elif state == "thinking":
                    ring_pulse = 2.0 * (0.5 + 0.5 * math.sin(phase * 2.3))

                ring_margin: float = 18 - ring_pulse
                canvas.coords(
                    ring_id,
                    ring_margin,
                    ring_margin,
                    window_size - ring_margin,
                    window_size - ring_margin,
                )
                canvas.itemconfigure(ring_id, outline=ring_color)
                canvas.itemconfigure(core_id, fill=core_color)
                canvas.itemconfigure(label_id, text=state.upper())

                if state == "speaking":
                    for index, bar_id in enumerate(bars):
                        height: float = 8 + (20 * abs(math.sin(phase * 4.6 + index)))
                        x1: int = bars_start_x + index * (bar_width + bar_spacing)
                        canvas.coords(
                            bar_id,
                            x1,
                            bars_bottom_y - height,
                            x1 + bar_width,
                            bars_bottom_y,
                        )
                        canvas.itemconfigure(bar_id, state="normal", fill=ring_color)
                else:
                    for bar_id in bars:
                        canvas.itemconfigure(bar_id, state="hidden")

                if self.auto_move and state != "idle" and not dragging and now >= auto_resume_at:
                    max_x: int = max(0, root.winfo_screenwidth() - root.winfo_width() - 4)
                    max_y: int = max(0, root.winfo_screenheight() - root.winfo_height() - 48)
                    near_target: bool = abs(position_x - target_x) < 2.0 and abs(position_y - target_y) < 2.0
                    if now >= next_target_at or near_target:
                        wander_x: float = random.uniform(-140.0, 140.0)
                        wander_y: float = random.uniform(-100.0, 100.0)
                        target_x = float(max(0, min(max_x, int(position_x + wander_x))))
                        target_y = float(max(0, min(max_y, int(position_y + wander_y))))
                        next_target_at = now + random.uniform(1.5, 3.5)

                    easing: float = min(1.0, delta * 1.2)
                    position_x += (target_x - position_x) * easing
                    position_y += (target_y - position_y) * easing
                    next_x: int = max(0, min(max_x, int(position_x)))
                    next_y: int = max(0, min(max_y, int(position_y)))
                    root.geometry(f"+{next_x}+{next_y}")

                root.after(40, tick)

            root.protocol("WM_DELETE_WINDOW", root.destroy)
            self._ready.set()
            root.after(40, tick)
            root.mainloop()
        except Exception as error:
            self.logger.warning("Avatar overlay failed: %s", error)
        finally:
            self._running.clear()
            self._ready.set()
