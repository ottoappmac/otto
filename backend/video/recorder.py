"""Screen recording to an mp4 file (macOS, via ffmpeg + avfoundation).

Records the whole screen to a file inside the session sandbox so the
``watch_video`` tool / ``/api/video/analyze`` route can then watch it.
Only one recording runs at a time (a single global manager), which matches
the UI: the user records one clip, stops, then analyses it.

Audio is captured only when an ``audio_device`` index is supplied, and is off
by default.  Note what avfoundation can and cannot reach: its audio devices
are *inputs*, so the usual choice records the microphone.  Capturing what the
Mac is playing needs a virtual loopback device (BlackHole and friends) to
appear in that list — macOS exposes no system-audio input otherwise.  Otto's
own process-tap helper (``backend.voice.loopback_manager``) can do it without
a driver, but it is owned by the voice subsystem and cannot be muxed into this
ffmpeg process, so it is deliberately not used here.
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


# Virtual devices that carry playback back in as an input.  Matched by name
# because avfoundation gives no other signal that a device is a loopback.
_LOOPBACK_HINTS = (
    "blackhole", "soundflower", "loopback", "vb-cable", "vb cable",
    "multi-output", "aggregate", "virtual",
)

_LOG_PREFIX_RE = re.compile(r"^\[AVFoundation[^\]]*\]\s*")
_SECTION_RE = re.compile(r"AVFoundation (video|audio) devices:", re.IGNORECASE)
_ENTRY_RE = re.compile(r"\[(\d+)\]\s+(.+)$")


def looks_like_loopback(name: str) -> bool:
    lowered = (name or "").lower()
    return any(hint in lowered for hint in _LOOPBACK_HINTS)


def list_devices() -> dict[str, list[dict]]:
    """Parse ``ffmpeg -f avfoundation -list_devices`` into video/audio lists.

    The listing goes to stderr and ffmpeg then exits non-zero (there is no
    real input to open), so the exit status is deliberately ignored.
    """
    empty: dict[str, list[dict]] = {"video": [], "audio": []}
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, timeout=20, check=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("avfoundation device enumeration failed: %s", exc)
        return empty

    out: dict[str, list[dict]] = {"video": [], "audio": []}
    section: Optional[str] = None
    for raw in proc.stderr.decode(errors="replace").splitlines():
        line = _LOG_PREFIX_RE.sub("", raw).strip()
        header = _SECTION_RE.match(line)
        if header:
            section = header.group(1).lower()
            continue
        if section is None:
            continue
        entry = _ENTRY_RE.match(line)
        if not entry:
            section = None  # past the end of this listing
            continue
        name = entry.group(2).strip()
        out[section].append({
            "index": entry.group(1),
            "name": name,
            "is_loopback": section == "audio" and looks_like_loopback(name),
        })
    return out


def _discover_screen_index() -> Optional[str]:
    """Index of the "Capture screen N" video device, or None."""
    for dev in list_devices()["video"]:
        if "capture screen" in dev["name"].lower():
            return dev["index"]
    return None


def resolve_audio_device(preferred: str = "") -> Optional[str]:
    """Pick the audio input to record, or None when there is none.

    An explicit *preferred* index wins if it still exists.  Otherwise a
    loopback device is chosen when one is installed — that's the only way to
    catch what's playing — falling back to the first input, normally the
    built-in microphone.
    """
    devices = list_devices()["audio"]
    if not devices:
        return None
    wanted = (preferred or "").strip()
    if wanted:
        for dev in devices:
            if dev["index"] == wanted:
                return dev["index"]
        logger.info("audio device %r is gone; falling back", wanted)
    for dev in devices:
        if dev["is_loopback"]:
            return dev["index"]
    return devices[0]["index"]


class _RecorderManager:
    """Owns at most one active ffmpeg screen-capture subprocess."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._path: Optional[Path] = None
        self._started_at: float = 0.0
        self._fps: float = 5.0
        self._audio: Optional[str] = None
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
                "audio": bool(recording and self._audio is not None),
            }

    def start(self, dest_dir: Path, *, fps: float = 5.0,
              max_side: int = 1280, audio_device: Optional[str] = None) -> dict:
        """Begin recording the screen into ``dest_dir``.

        Pass ``audio_device`` (an avfoundation audio index) to mux a sound
        track in; omit it to record silently.

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

            audio = str(audio_device) if audio_device is not None else None
            device = f"{screen_idx}:{audio}" if audio is not None else f"{screen_idx}:none"
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
            ]
            if audio is not None:
                cmd += ["-c:a", "aac", "-b:a", "128k"]
            cmd.append(str(out_path))
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
                # Recording audio needs the Microphone grant on top of Screen
                # Recording, and it's the likelier culprit when audio was the
                # thing we just added.
                needed = (
                    "check Screen Recording and Microphone permissions"
                    if audio is not None
                    else "check Screen Recording permission"
                )
                return {
                    "error": f"ffmpeg exited immediately — {needed}. "
                             + err.decode(errors="replace")[-400:],
                }

            self._proc = proc
            self._path = out_path
            self._started_at = time.monotonic()
            self._fps = fps
            self._audio = audio
            logger.info(
                "screen recording started → %s (%.0f fps, audio=%s)",
                out_path, fps, audio if audio is not None else "off",
            )
            return {"path": str(out_path), "fps": fps, "audio": audio is not None}

    def stop(self, timeout: float = 15.0) -> dict:
        """Stop the active recording and return the finished file path."""
        with self._lock:
            proc = self._proc
            path = self._path
            elapsed = (time.monotonic() - self._started_at) if self._started_at else 0.0
            had_audio = self._audio is not None
            self._proc = None
            self._started_at = 0.0
            self._audio = None

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
                    "size_bytes": path.stat().st_size, "audio": had_audio}
        return {"error": "Recording stopped but no output file was produced."}


# Module-level singleton.
manager = _RecorderManager()
