"""Audio output storage and playback."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalAudioOutput:
    """Persists WAV files locally and optionally plays them."""

    output_dir: Path
    autoplay: bool = True
    cleanup_after_playback: bool = True

    def output(self, wav_bytes: bytes) -> Path | None:
        """Persist audio, optionally play it, and optionally clean it up."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_path: Path = self._build_output_path()
        target_path.write_bytes(wav_bytes)

        if self.autoplay:
            self._safe_play(target_path)
            if self.cleanup_after_playback:
                self._safe_delete(target_path)
                return None
        return target_path

    def _build_output_path(self) -> Path:
        """Create a timestamped output path for a WAV artifact."""
        timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.output_dir / f"assistant_{timestamp}.wav"

    def _safe_play(self, path: Path) -> None:
        """Play WAV file while swallowing playback errors."""
        try:
            self._play(path)
        except Exception as error:
            LOGGER.warning("Audio playback failed for %s: %s", path, error)

    def _safe_delete(self, path: Path) -> None:
        """Delete generated WAV file while swallowing cleanup errors."""
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            LOGGER.warning("Audio cleanup failed for %s: %s", path, error)

    def _play(self, path: Path) -> None:
        """Play WAV file with platform-specific methods."""
        os_name: str = platform.system()
        if os_name == "Windows":
            self._play_windows(path)
            return

        player_command: list[str] | None = self._resolve_unix_player(path)
        if player_command is None:
            raise RuntimeError("No compatible audio player found (afplay/aplay/paplay).")
        subprocess.run(player_command, check=False)

    def _play_windows(self, path: Path) -> None:
        """Play WAV on Windows using the standard library."""
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)

    def _resolve_unix_player(self, path: Path) -> list[str] | None:
        """Resolve the first available Unix audio player command."""
        if shutil.which("afplay"):
            return ["afplay", str(path)]
        if shutil.which("aplay"):
            return ["aplay", str(path)]
        if shutil.which("paplay"):
            return ["paplay", str(path)]
        return None
