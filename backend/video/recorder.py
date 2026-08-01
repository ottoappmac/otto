"""Screen recording to an mp4 file (macOS, via ffmpeg + avfoundation).

Records the whole screen to a file inside the session sandbox so the
``watch_video`` tool / ``/api/video/analyze`` route can then watch it.
Only one recording runs at a time (a single global manager), which matches
the UI: the user records one clip, stops, then analyses it.

Audio is captured only when an ``audio_device`` index is supplied — screen
recording is video-only by default (system-audio capture needs the
loopback device the voice subsystem manages separately).
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from backend.video.ingest import ffmpeg_path

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"


def supported() -> bool:
    """Screen recording is currently implemented for macOS only."""
    return _IS_MACOS


def _discover_screen_index() -> Optional[str]:
    """Parse ``ffmpeg -f avfoundation -list_devices`` for the screen device.

    avfoundation lists a "Capture screen N" video device; we return its
    numeric index as a string.  Returns None when it can't be found.
    """
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, timeout=20, check=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("avfoundation device enumeration failed: %s", exc)
        return None
    text = proc.stderr.decode(errors="replace")
    # Lines look like: "[AVFoundation ...] [1] Capture screen 0"
    for line in text.splitlines():
        if "capture screen" in line.lower():
            m = re.search(r"\[(\d+)\]\s*Capture screen", line, re.IGNORECASE)
            if m:
                return m.group(1)
    return None


class _RecorderManager:
    """Owns at most one active ffmpeg screen-capture subprocess."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._path: Optional[Path] = None
        self._started_at: float = 0.0
        self._fps: float = 5.0
        self._lock = threading.Lock()

    def is_recording(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            recording = self._proc is not None and self._proc.poll() is None
            return {
                "supported": supported(),
                "recording": recording,
                "path": str(self._path) if self._path else None,
                "elapsed_secs": (time.monotonic() - self._started_at) if recording else 0.0,
                "fps": self._fps,
            }

    def start(self, dest_dir: Path, *, fps: float = 5.0,
              max_side: int = 1280, audio_device: Optional[int] = None) -> dict:
        """Begin recording the screen into ``dest_dir``.

        Returns a dict with the output ``path`` (or an ``error``).
        """
        if not supported():
            return {"error": "Screen recording is only supported on macOS."}
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"error": "A recording is already in progress.", "path": str(self._path)}

            screen_idx = _discover_screen_index()
            if screen_idx is None:
                return {
                    "error": "Could not find a screen capture device. Grant Otto "
                             "Screen Recording permission in System Settings and retry.",
                }

            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_path = dest_dir / f"screen_{int(time.time())}.mp4"

            device = f"{screen_idx}:{audio_device}" if audio_device is not None else f"{screen_idx}:none"
            fps = max(1.0, min(30.0, float(fps)))
            vf = (
                f"scale='if(gt(iw,ih),min({max_side},iw),-2)':"
                f"'if(gt(iw,ih),-2,min({max_side},ih))'"
            )
            cmd = [
                ffmpeg_path(), "-hide_banner", "-y",
                "-f", "avfoundation",
                "-capture_cursor", "1",
                "-framerate", str(int(fps)),
                "-i", device,
                "-vf", vf,
                "-r", str(int(fps)),
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast",
                str(out_path),
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Failed to start ffmpeg: {exc}"}

            # Detect an immediate failure (e.g. permission denied).
            time.sleep(0.4)
            if proc.poll() is not None:
                err = b""
                try:
                    err = proc.stderr.read() if proc.stderr else b""
                except Exception:  # noqa: BLE001
                    pass
                return {
                    "error": "ffmpeg exited immediately — check Screen Recording "
                             "permission. " + err.decode(errors="replace")[-400:],
                }

            self._proc = proc
            self._path = out_path
            self._started_at = time.monotonic()
            self._fps = fps
            logger.info("screen recording started → %s (%.0f fps)", out_path, fps)
            return {"path": str(out_path), "fps": fps}

    def stop(self, timeout: float = 15.0) -> dict:
        """Stop the active recording and return the finished file path."""
        with self._lock:
            proc = self._proc
            path = self._path
            elapsed = (time.monotonic() - self._started_at) if self._started_at else 0.0
            self._proc = None
            self._started_at = 0.0

        if proc is None:
            return {"error": "No recording is in progress."}
        # Gracefully ask ffmpeg to finish (writes the moov atom) by sending 'q'.
        try:
            if proc.stdin:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001
                proc.kill()

        if path and path.exists():
            logger.info("screen recording stopped → %s (%.1fs)", path, elapsed)
            return {"path": str(path), "duration_secs": round(elapsed, 1),
                    "size_bytes": path.stat().st_size}
        return {"error": "Recording stopped but no output file was produced."}


# Module-level singleton.
manager = _RecorderManager()
