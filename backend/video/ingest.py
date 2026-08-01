"""Media processing for the frame-based video path.

Everything here runs on-device: ffmpeg (shipped by ``imageio-ffmpeg``)
samples frames and extracts audio; the audio is transcribed with the same
on-device Whisper stack the voice subsystem uses.  YouTube videos are
downloaded with ``yt-dlp`` so non-Gemini providers can watch them too.

All heavy calls are synchronous and meant to be run via
``asyncio.to_thread`` from the async tool layer.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Frames beyond this are almost never worth the tokens; a hard cap keeps a
# misconfigured ``frame_rate`` from producing thousands of images.
_ABSOLUTE_MAX_FRAMES = 512

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/|embed/)|youtu\.be/)",
    re.IGNORECASE,
)

_VIDEO_EXTS = frozenset({
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg",
    ".wmv", ".flv", ".3gp",
})


def is_youtube_url(source: str) -> bool:
    """True if *source* looks like a YouTube URL."""
    return bool(_YOUTUBE_RE.search(source or ""))


def is_url(source: str) -> bool:
    s = (source or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def looks_like_video_path(source: str) -> bool:
    """True if *source* has a known video file extension."""
    try:
        return Path(source).suffix.lower() in _VIDEO_EXTS
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------


def ffmpeg_path() -> str:
    """Return a usable ffmpeg binary path.

    Prefers the static binary bundled by ``imageio-ffmpeg`` (always present
    in a packaged build); falls back to a system ``ffmpeg`` on PATH.
    """
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        import shutil

        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError(
            "ffmpeg not found. Install the 'imageio-ffmpeg' package or put "
            "ffmpeg on your PATH."
        )


def ffprobe_duration_secs(path: str | Path) -> Optional[float]:
    """Best-effort clip duration in seconds via ffprobe, or None."""
    import shutil

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # imageio-ffmpeg ships ffmpeg but not ffprobe; parse ffmpeg -i output.
        return _duration_via_ffmpeg(path)
    try:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, timeout=30, check=False,
        )
        data = json.loads(proc.stdout.decode(errors="replace") or "{}")
        return float(data.get("format", {}).get("duration"))
    except Exception:  # noqa: BLE001
        return _duration_via_ffmpeg(path)


def _duration_via_ffmpeg(path: str | Path) -> Optional[float]:
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-i", str(path)],
            capture_output=True, timeout=30, check=False,
        )
        err = proc.stderr.decode(errors="replace")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
        if not m:
            return None
        h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mnt * 60 + s
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def sample_frames(
    path: str | Path,
    *,
    fps: float = 1.0,
    max_frames: int = 60,
    max_side: int = 1024,
    start_secs: Optional[float] = None,
    end_secs: Optional[float] = None,
) -> list[tuple[float, bytes]]:
    """Sample frames from *path* and return ``[(timestamp_secs, jpeg_bytes)]``.

    Uses ffmpeg's ``fps`` filter to grab one frame every ``1/fps`` seconds,
    scaled so the longest side is at most *max_side*.  The number of frames
    is clamped to *max_frames* (and a hard internal ceiling) to protect
    frame-based providers from oversized image arrays.
    """
    fps = max(0.01, float(fps))
    cap = max(1, min(int(max_frames), _ABSOLUTE_MAX_FRAMES))
    out_frames: list[tuple[float, bytes]] = []

    with tempfile.TemporaryDirectory(prefix="otto_frames_") as tmp:
        tmp_dir = Path(tmp)
        vf = (
            f"fps={fps},"
            f"scale='if(gt(iw,ih),min({max_side},iw),-2)':"
            f"'if(gt(iw,ih),-2,min({max_side},ih))'"
        )
        cmd = [ffmpeg_path(), "-y"]
        if start_secs is not None and start_secs > 0:
            cmd += ["-ss", f"{start_secs:.3f}"]
        cmd += ["-i", str(path)]
        if end_secs is not None and end_secs > 0:
            dur = end_secs - (start_secs or 0)
            if dur > 0:
                cmd += ["-t", f"{dur:.3f}"]
        cmd += [
            "-vf", vf,
            "-frames:v", str(cap),
            "-q:v", "3",
            str(tmp_dir / "frame_%05d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("frame sampling failed: %s", exc)
            return []

        base = start_secs or 0.0
        for i, fp in enumerate(sorted(tmp_dir.glob("frame_*.jpg"))):
            try:
                out_frames.append((base + i / fps, fp.read_bytes()))
            except Exception:  # noqa: BLE001
                continue
    return out_frames


def frames_to_image_blocks(frames: list[tuple[float, bytes]]) -> list[dict]:
    """Convert sampled frames into OpenAI-style ``image_url`` content blocks.

    Each frame is prefixed with a text block carrying its ``MM:SS``
    timestamp so the model can refer to specific moments.
    """
    blocks: list[dict] = []
    for ts, jpeg in frames:
        b64 = base64.standard_b64encode(jpeg).decode()
        blocks.append({"type": "text", "text": f"[frame @ {_fmt_ts(ts)}]"})
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return blocks


def _fmt_ts(secs: float) -> str:
    secs = max(0, int(round(secs)))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Audio → transcript
# ---------------------------------------------------------------------------

# Whisper invents a stray word or two on non-speech audio (traffic, music), and
# ``stt``'s filters deliberately let a lone word through — they cannot reject
# one without also rejecting the genuine one-word commands ("Stop", "Yes") the
# voice pipeline depends on.  Video carries no such constraint: a one- or
# two-word transcript tells the model nothing either way, so it is dropped here
# rather than in the filter both subsystems share.
_MIN_TRANSCRIPT_WORDS = 3


def extract_audio_pcm16(
    path: str | Path,
    *,
    start_secs: Optional[float] = None,
    end_secs: Optional[float] = None,
) -> bytes:
    """Extract the audio track as 16 kHz mono int16 PCM (Whisper's format).

    Returns empty bytes when the clip has no audio or extraction fails.
    """
    cmd = [ffmpeg_path(), "-y"]
    if start_secs is not None and start_secs > 0:
        cmd += ["-ss", f"{start_secs:.3f}"]
    cmd += ["-i", str(path)]
    if end_secs is not None and end_secs > 0:
        dur = end_secs - (start_secs or 0)
        if dur > 0:
            cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        return proc.stdout or b""
    except Exception as exc:  # noqa: BLE001
        logger.warning("audio extraction failed: %s", exc)
        return b""


async def transcribe_video_audio(
    path: str | Path,
    *,
    start_secs: Optional[float] = None,
    end_secs: Optional[float] = None,
) -> str:
    """Transcribe a clip's audio track using the on-device Whisper stack.

    Degrades gracefully to an empty string when audio is absent or the STT
    model isn't available, so the visual path still works.
    """
    import asyncio

    pcm = await asyncio.to_thread(
        extract_audio_pcm16, path, start_secs=start_secs, end_secs=end_secs,
    )
    if not pcm:
        return ""
    try:
        from backend.voice import stt

        text = await stt.transcribe(pcm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("video audio transcription unavailable: %s", exc)
        return ""

    if len(text.split()) < _MIN_TRANSCRIPT_WORDS:
        logger.debug("dropping %r as noise, not a video transcript", text)
        return ""
    return text


# ---------------------------------------------------------------------------
# YouTube download (frame-based path only — Gemini takes the URL directly)
# ---------------------------------------------------------------------------


def download_youtube(url: str, dest_dir: str | Path, *, max_height: int = 720) -> Path:
    """Download a public YouTube video into *dest_dir* and return its path.

    Caps the resolution so frame sampling stays cheap.  Raises on failure.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yt-dlp is not installed — cannot download YouTube videos for "
            "frame-based providers. Install 'yt-dlp' or use the Gemini "
            "provider (which accepts YouTube URLs natively)."
        ) from exc

    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    p = Path(filename)
    if not p.exists():
        # merge_output_format may have rewritten the extension to .mp4
        alt = p.with_suffix(".mp4")
        if alt.exists():
            return alt
        raise RuntimeError(f"YouTube download produced no file for {url!r}")
    return p


def downscale_jpeg(jpeg_bytes: bytes, max_side: int) -> bytes:
    """Return a JPEG re-encoded with the longest side capped at *max_side*."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return jpeg_bytes
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / longest
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return jpeg_bytes
