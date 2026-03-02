"""Desktop action planning and execution helpers."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from desktop_ai.types import ActionExecutionResult, DesktopAction, DesktopActionPlan

SUPPORTED_ACTION_TYPES: tuple[str, ...] = (
    "move_mouse",
    "drag_mouse",
    "click",
    "type_text",
    "press",
    "hotkey",
    "scroll",
    "launch",
    "wait",
)


def format_action(action: DesktopAction) -> str:
    """Render an action in a compact, human-readable format."""
    if not action.args:
        return f"{action.type}()"
    rendered_args: str = ", ".join(
        f"{name}={value!r}" for name, value in sorted(action.args.items())
    )
    return f"{action.type}({rendered_args})"


def parse_action_plan(raw_response: str, *, max_actions_per_turn: int) -> DesktopActionPlan:
    """Parse a model response into spoken reply + structured actions."""
    payload: dict[str, Any] | None = _extract_json_payload(raw_response)
    if payload is None:
        reply: str = raw_response.strip()
        return DesktopActionPlan(
            spoken_reply=reply or "I could not understand that request.",
            actions=(),
            raw_response=raw_response,
        )

    reply_value: Any = payload.get("spoken_reply")
    if not isinstance(reply_value, str):
        reply_value = payload.get("reply")
    spoken_reply: str = str(reply_value).strip() if isinstance(reply_value, str) else ""
    if not spoken_reply:
        spoken_reply = raw_response.strip() or "I could not understand that request."

    parsed_actions: list[DesktopAction] = []
    actions_value: Any = payload.get("actions")
    if isinstance(actions_value, dict):
        actions_iterable: tuple[Any, ...] = (actions_value,)
    elif isinstance(actions_value, list):
        actions_iterable = tuple(actions_value)
    else:
        actions_iterable = ()
    for item in actions_iterable:
        action: DesktopAction | None = _parse_action(item)
        if action is None:
            continue
        parsed_actions.append(action)
        if len(parsed_actions) >= max(1, max_actions_per_turn):
            break

    return DesktopActionPlan(
        spoken_reply=spoken_reply,
        actions=tuple(parsed_actions),
        raw_response=raw_response,
    )


@dataclass(slots=True)
class DesktopController:
    """Executes keyboard/mouse/application actions locally."""

    require_approval: bool = True
    allowed_launch_commands: tuple[str, ...] = ()
    max_actions_per_turn: int = 5
    action_delay_seconds: float = 0.05
    log_path: Path | None = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _pyautogui: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize pyautogui lazily and configure failsafe defaults."""
        try:
            import pyautogui
        except ImportError as error:
            raise RuntimeError(
                "pyautogui is required for desktop control. Install dependencies with `pip install -e .`."
            ) from error

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = max(0.0, self.action_delay_seconds)
        self._pyautogui = pyautogui

    def parse_plan(self, raw_response: str) -> DesktopActionPlan:
        """Parse a model response into an executable action plan."""
        return parse_action_plan(
            raw_response,
            max_actions_per_turn=self.max_actions_per_turn,
        )

    def execute(
        self,
        plan: DesktopActionPlan,
        *,
        approval_callback: Callable[[DesktopActionPlan], bool] | None = None,
    ) -> tuple[ActionExecutionResult, ...]:
        """Execute the bounded action list from a parsed plan."""
        if not plan.actions:
            return ()

        bounded_actions: tuple[DesktopAction, ...] = plan.actions[: max(1, self.max_actions_per_turn)]
        bounded_plan = DesktopActionPlan(
            spoken_reply=plan.spoken_reply,
            actions=bounded_actions,
            raw_response=plan.raw_response,
        )
        if self.require_approval:
            approved: bool = approval_callback(bounded_plan) if approval_callback else False
            self._log_plan_event(bounded_plan=bounded_plan, approved=approved)
            if not approved:
                self.logger.info("Desktop action plan was denied.")
                return ()

        results: list[ActionExecutionResult] = []
        for action in bounded_actions:
            started_at: datetime = datetime.now(timezone.utc)
            try:
                detail: str = self._execute_action(action)
                success: bool = True
            except Exception as error:
                success = False
                detail = str(error)
                self.logger.warning("Desktop action failed: %s | %s", format_action(action), detail)
            finished_at: datetime = datetime.now(timezone.utc)
            result = ActionExecutionResult(
                action=action,
                success=success,
                detail=detail,
                started_at=started_at,
                finished_at=finished_at,
            )
            results.append(result)
            self._log_action_result(result)
            if not success:
                break
        return tuple(results)

    def _execute_action(self, action: DesktopAction) -> str:
        """Execute a single desktop action and return a short detail string."""
        if action.type == "move_mouse":
            x: int = _to_int(action.args.get("x"), name="x")
            y: int = _to_int(action.args.get("y"), name="y")
            duration: float = _to_float(action.args.get("duration", 0.0), name="duration")
            self._pyautogui.moveTo(x=x, y=y, duration=max(0.0, duration))
            return f"moved mouse to ({x}, {y})"

        if action.type == "drag_mouse":
            x = _to_int(action.args.get("x"), name="x")
            y = _to_int(action.args.get("y"), name="y")
            duration = _to_float(action.args.get("duration", 0.0), name="duration")
            button: str = str(action.args.get("button", "left")).lower()
            self._pyautogui.dragTo(x=x, y=y, duration=max(0.0, duration), button=button)
            return f"dragged mouse to ({x}, {y}) with {button} button"

        if action.type == "click":
            button = str(action.args.get("button", "left")).lower()
            clicks = max(1, _to_int(action.args.get("clicks", 1), name="clicks"))
            interval = _to_float(action.args.get("interval", 0.0), name="interval")
            x_value: Any = action.args.get("x")
            y_value: Any = action.args.get("y")
            if x_value is None or y_value is None:
                self._pyautogui.click(button=button, clicks=clicks, interval=max(0.0, interval))
                return f"clicked {button} button ({clicks} time(s))"
            x = _to_int(x_value, name="x")
            y = _to_int(y_value, name="y")
            self._pyautogui.click(
                x=x,
                y=y,
                button=button,
                clicks=clicks,
                interval=max(0.0, interval),
            )
            return f"clicked {button} at ({x}, {y}) ({clicks} time(s))"

        if action.type == "type_text":
            text: str = str(action.args.get("text", ""))
            if not text:
                raise ValueError("type_text requires a non-empty `text` value.")
            interval = _to_float(action.args.get("interval", 0.0), name="interval")
            self._pyautogui.write(text, interval=max(0.0, interval))
            return "typed text"

        if action.type == "press":
            key: str = str(action.args.get("key", "")).strip()
            if not key:
                raise ValueError("press requires `key`.")
            presses: int = max(1, _to_int(action.args.get("presses", 1), name="presses"))
            interval = _to_float(action.args.get("interval", 0.0), name="interval")
            self._pyautogui.press(key, presses=presses, interval=max(0.0, interval))
            return f"pressed {key} ({presses} time(s))"

        if action.type == "hotkey":
            keys: tuple[str, ...] = _normalize_hotkey_keys(action.args.get("keys"))
            self._pyautogui.hotkey(*keys)
            return f"pressed hotkey {'+'.join(keys)}"

        if action.type == "scroll":
            amount: int = _to_int(action.args.get("amount"), name="amount")
            self._pyautogui.scroll(amount)
            return f"scrolled {amount}"

        if action.type == "launch":
            command: str = str(action.args.get("command", "")).strip()
            if not command:
                raise ValueError("launch requires a non-empty `command` value.")
            self._assert_launch_allowed(command)
            subprocess.Popen(command, shell=True)
            return f"launched {command}"

        if action.type == "wait":
            seconds: float = _to_float(action.args.get("seconds", 0.0), name="seconds")
            bounded_seconds: float = max(0.0, min(30.0, seconds))
            time.sleep(bounded_seconds)
            return f"waited {bounded_seconds:.2f}s"

        raise ValueError(f"Unsupported action type: {action.type}")

    def _assert_launch_allowed(self, command: str) -> None:
        """Enforce allowlist for launch commands."""
        allowed: tuple[str, ...] = tuple(item.lower() for item in self.allowed_launch_commands)
        if not allowed:
            raise PermissionError(
                "Launch action blocked: no allowlist configured (ASSISTANT_DESKTOP_ALLOWED_LAUNCH)."
            )
        if "*" in allowed:
            return
        command_head: str = _extract_command_head(command).lower()
        if command_head in allowed:
            return
        raise PermissionError(
            f"Launch action blocked: command `{command_head}` is not in allowlist {self.allowed_launch_commands}."
        )

    def _log_plan_event(self, *, bounded_plan: DesktopActionPlan, approved: bool) -> None:
        """Write approval decisions to the action log."""
        if self.log_path is None:
            return
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "plan_approval",
            "approved": approved,
            "actions": [
                {"type": action.type, "args": action.args}
                for action in bounded_plan.actions
            ],
        }
        self._append_json_log(payload)

    def _log_action_result(self, result: ActionExecutionResult) -> None:
        """Write one action execution result to the action log."""
        if self.log_path is None:
            return
        payload: dict[str, Any] = {
            "timestamp": result.finished_at.isoformat(),
            "event": "action_execution",
            "action": {"type": result.action.type, "args": result.action.args},
            "success": result.success,
            "detail": result.detail,
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
        }
        self._append_json_log(payload)

    def _append_json_log(self, payload: dict[str, Any]) -> None:
        """Append one JSON-lines event to the configured log path."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(payload, ensure_ascii=True))
            file_handle.write("\n")


def _extract_json_payload(raw_response: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from a model response."""
    text: str = raw_response.strip()
    if not text:
        return None
    candidates: list[str] = [text]

    if text.startswith("```"):
        lines: list[str] = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            body: str = "\n".join(lines[1:-1]).strip()
            if body.startswith("json"):
                body = body[4:].strip()
            if body:
                candidates.append(body)

    start: int = text.find("{")
    end: int = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _parse_action(value: Any) -> DesktopAction | None:
    """Normalize one action item from model JSON."""
    if not isinstance(value, dict):
        return None
    type_value: Any = value.get("type", value.get("action"))
    action_type: str = str(type_value).strip().lower()
    if action_type not in SUPPORTED_ACTION_TYPES:
        return None
    args: dict[str, Any] = {}
    nested_args: Any = value.get("args")
    if isinstance(nested_args, dict):
        args.update({str(name): raw_value for name, raw_value in nested_args.items()})
    for name, raw_value in value.items():
        key: str = str(name)
        if key in {"type", "action", "args"}:
            continue
        args[key] = raw_value
    return DesktopAction(type=action_type, args=args)


def _to_int(value: Any, *, name: str) -> int:
    """Parse a value as int and raise a clear error on failure."""
    if value is None:
        raise ValueError(f"Missing required numeric field `{name}`.")
    if isinstance(value, bool):
        raise ValueError(f"Invalid bool value for `{name}`.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer value for `{name}`: {value!r}") from error


def _to_float(value: Any, *, name: str) -> float:
    """Parse a value as float and raise a clear error on failure."""
    if value is None:
        raise ValueError(f"Missing required numeric field `{name}`.")
    if isinstance(value, bool):
        raise ValueError(f"Invalid bool value for `{name}`.")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid float value for `{name}`: {value!r}") from error


def _normalize_hotkey_keys(value: Any) -> tuple[str, ...]:
    """Normalize hotkey specification from string or list."""
    if isinstance(value, str):
        text: str = value.strip()
        if not text:
            raise ValueError("hotkey requires non-empty `keys`.")
        if "+" in text:
            keys = tuple(part.strip() for part in text.split("+") if part.strip())
        else:
            keys = tuple(part.strip() for part in text.split(",") if part.strip())
    elif isinstance(value, list):
        keys = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise ValueError("hotkey requires `keys` as a string or list.")
    if len(keys) < 2:
        raise ValueError("hotkey requires at least two keys.")
    return keys


def _extract_command_head(command: str) -> str:
    """Extract command executable name from a shell command string."""
    if not command.strip():
        return ""
    try:
        parts: list[str] = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return command.split(maxsplit=1)[0].strip().lower()
    if not parts:
        return ""
    return Path(parts[0]).name.lower()
