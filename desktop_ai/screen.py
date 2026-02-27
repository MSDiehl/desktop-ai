"""Screen-capture implementations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from desktop_ai.types import CapturedScreen


@dataclass(slots=True)
class MSSScreenCapturer:
    """Captures screenshots from a selected monitor using mss."""

    monitor_index: int = 1

    def capture(self) -> CapturedScreen:
        """Capture current screen contents as PNG bytes."""
        try:
            import mss
            import mss.tools
        except ImportError as error:
            raise RuntimeError("mss is required for screenshot capture.") from error

        with mss.mss() as session:
            fallback_index: int = 1 if len(session.monitors) > 1 else 0
            if self.monitor_index < 0 or self.monitor_index >= len(session.monitors):
                monitor = session.monitors[fallback_index]
            else:
                monitor = session.monitors[self.monitor_index]

            screenshot = session.grab(monitor)
            png_bytes: bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
            width, height = screenshot.size
            return CapturedScreen(
                png_bytes=png_bytes,
                width=width,
                height=height,
                captured_at=datetime.now(timezone.utc),
            )
