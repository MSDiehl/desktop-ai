"""Context providers and dynamic provider registry."""

from __future__ import annotations

import ctypes
import getpass
import platform
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

ContextData = dict[str, str]


class ContextProvider(Protocol):
    """Provides context values about the current user activity."""

    @property
    def name(self) -> str:
        """Return a stable provider name used for namespacing fields."""

    def collect(self) -> Mapping[str, str]:
        """Collect provider-specific context values."""


@dataclass(slots=True)
class TimestampContextProvider:
    """Provides current UTC timestamp information."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "timestamp"

    def collect(self) -> Mapping[str, str]:
        """Collect timestamp context values."""
        now: datetime = datetime.now(timezone.utc)
        return {
            "iso_utc": now.isoformat(),
            "epoch_seconds": str(int(now.timestamp())),
        }


@dataclass(slots=True)
class EnvironmentContextProvider:
    """Provides machine and process environment context."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "environment"

    def collect(self) -> Mapping[str, str]:
        """Collect environment context values."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "username": getpass.getuser(),
            "cwd": str(Path.cwd()),
        }


@dataclass(slots=True)
class ActiveWindowContextProvider:
    """Provides active window metadata from the operating system."""

    include_process_name: bool = True

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "active_window"

    def collect(self) -> Mapping[str, str]:
        """Collect active-window context for the current platform."""
        system_name: str = platform.system()
        if system_name == "Windows":
            return self._collect_windows()
        if system_name == "Darwin":
            return self._collect_macos()
        return self._collect_linux()

    def _collect_windows(self) -> Mapping[str, str]:
        """Collect active-window data on Windows using Win32 APIs."""
        user32 = ctypes.windll.user32
        hwnd: int = user32.GetForegroundWindow()
        if hwnd == 0:
            return {}

        length: int = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title: str = buffer.value.strip() or "Unknown"

        result: ContextData = {"title": title}
        if not self.include_process_name:
            return result

        process_name: str = self._resolve_windows_process_name(hwnd)
        result["process_name"] = process_name
        return result

    def _resolve_windows_process_name(self, hwnd: int) -> str:
        """Resolve process name for a Win32 window handle."""
        process_id = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))

        try:
            import psutil

            return psutil.Process(process_id.value).name()
        except Exception:
            return "Unknown"

    def _collect_macos(self) -> Mapping[str, str]:
        """Collect frontmost app and window information on macOS."""
        app_name: str = self._run_command(
            [
                "osascript",
                "-e",
                (
                    'tell application "System Events" '
                    "to get name of first application process whose frontmost is true"
                ),
            ]
        )
        window_title: str = self._run_command(
            [
                "osascript",
                "-e",
                (
                    'tell application "System Events" to tell '
                    "(first process where frontmost is true) "
                    "to get name of front window"
                ),
            ]
        )

        result: ContextData = {}
        if app_name:
            result["process_name"] = app_name
        if window_title:
            result["title"] = window_title
        return result

    def _collect_linux(self) -> Mapping[str, str]:
        """Collect active-window information on Linux using xdotool if available."""
        if shutil.which("xdotool") is None:
            return {}

        title: str = self._run_command(["xdotool", "getactivewindow", "getwindowname"])
        process_name: str = ""
        if self.include_process_name:
            process_id: str = self._run_command(["xdotool", "getactivewindow", "getwindowpid"])
            process_name = self._resolve_process_name(process_id)

        result: ContextData = {}
        if title:
            result["title"] = title
        if process_name:
            result["process_name"] = process_name
        return result

    def _resolve_process_name(self, process_id: str) -> str:
        """Resolve a process name from pid text, or return pid text if unavailable."""
        if not process_id.isdigit():
            return process_id
        try:
            import psutil

            return psutil.Process(int(process_id)).name()
        except Exception:
            return process_id

    def _run_command(self, args: list[str]) -> str:
        """Run a command and return trimmed stdout on success."""
        process = subprocess.run(args, capture_output=True, text=True, check=False)
        return process.stdout.strip() if process.returncode == 0 else ""


@dataclass(slots=True)
class CompositeContextCollector:
    """Aggregates data from many context providers into one flat mapping."""

    providers: list[ContextProvider]

    def collect(self) -> ContextData:
        """Collect and namespace context values from all configured providers."""
        merged: ContextData = {}
        for provider in self.providers:
            values: Mapping[str, str] = provider.collect()
            for key, value in values.items():
                merged[f"{provider.name}.{key}"] = str(value)
        return merged


@dataclass(slots=True)
class ContextProviderRegistry:
    """Creates context providers from symbolic names for dynamic composition."""

    _factories: dict[str, Callable[[], ContextProvider]]

    def register(self, name: str, factory: Callable[[], ContextProvider]) -> None:
        """Register a provider factory under a symbolic name."""
        self._factories[name] = factory

    def create(self, name: str) -> ContextProvider:
        """Create one provider by name or raise if unknown."""
        if name not in self._factories:
            available: str = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown context provider '{name}'. Available: {available}")
        return self._factories[name]()

    def create_many(self, names: Iterable[str]) -> list[ContextProvider]:
        """Create multiple providers in the exact order requested."""
        return [self.create(name) for name in names]


def build_default_context_registry() -> ContextProviderRegistry:
    """Build a registry with all built-in context providers."""
    registry = ContextProviderRegistry(_factories={})
    registry.register("timestamp", TimestampContextProvider)
    registry.register("environment", EnvironmentContextProvider)
    registry.register("active_window", ActiveWindowContextProvider)
    return registry
