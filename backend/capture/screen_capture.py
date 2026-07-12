"""Native screen capture for the Transcribe screenshot feature (macOS).

Provides three capabilities used by ``backend/routes/capture.py``:

- **Window enumeration** — list on-screen normal windows (app + title + id).
- **Capture** — grab the whole desktop or a single window as PNG bytes.
- **Change detection** — a small perceptual hash (dHash) so the frontend's
  transcript-anchored auto-capture can skip frames where the followed window
  has not visibly changed.

Everything degrades gracefully off macOS (and when the optional pyobjc / Pillow
dependencies are unavailable): callers get ``supported=False`` / empty results
rather than exceptions.
"""

from __future__ import annotations

import base64
import io
import logging
import subprocess
import sys
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"

# --- Optional native frameworks (guarded) ----------------------------------
try:  # Quartz / CoreGraphics window list + image capture
    from Quartz import (  # type: ignore
        CGRectNull,
        CGWindowListCopyWindowInfo,
        CGWindowListCreateImage,
        kCGNullWindowID,
        kCGWindowImageBoundsIgnoreFraming,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionIncludingWindow,
        kCGWindowListOptionOnScreenOnly,
    )

    try:
        from Quartz import CGPreflightScreenCaptureAccess  # type: ignore
    except Exception:  # pragma: no cover - older macOS without the symbol
        CGPreflightScreenCaptureAccess = None  # type: ignore

    try:
        from Quartz import CGRequestScreenCaptureAccess  # type: ignore
    except Exception:  # pragma: no cover
        CGRequestScreenCaptureAccess = None  # type: ignore

    from AppKit import (  # type: ignore
        NSBitmapImageFileTypePNG,
        NSBitmapImageRep,
    )

    _QUARTZ_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host frameworks
    _QUARTZ_AVAILABLE = False
    CGPreflightScreenCaptureAccess = None  # type: ignore
    CGRequestScreenCaptureAccess = None  # type: ignore

try:
    from PIL import Image  # type: ignore

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PIL_AVAILABLE = False


# Longest-side caps (px). Full frames sent to a vision model; thumbnails for the
# in-app window picker.
_MAX_SIDE = 1568
_THUMB_MAX_SIDE = 240


# ---------------------------------------------------------------------------
# Capability / permission
# ---------------------------------------------------------------------------


def supported() -> bool:
    """Whether native screen capture is available on this host."""
    return _IS_MACOS and _QUARTZ_AVAILABLE


def screen_recording_granted() -> Optional[bool]:
    """Best-effort Screen Recording permission check.

    Returns True/False when the preflight API is available, or None when the
    result is unknown (API missing / not macOS). The API is process-scoped and
    can lag the real capture, so treat None/True as "try anyway".
    """
    if not supported() or CGPreflightScreenCaptureAccess is None:
        return None
    try:
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:  # pragma: no cover - defensive
        return None


def request_screen_recording() -> None:
    """Trigger the macOS Screen Recording permission prompt (best effort)."""
    if not supported() or CGRequestScreenCaptureAccess is None:
        return
    try:
        CGRequestScreenCaptureAccess()
    except Exception:  # pragma: no cover - defensive
        pass


# ---------------------------------------------------------------------------
# Window enumeration
# ---------------------------------------------------------------------------


def list_windows(include_thumbnails: bool = False) -> list[dict]:
    """Return on-screen normal windows: ``[{window_id, app, title, thumb_b64?}]``.

    Filters to layer-0 (normal) windows with a non-zero area, largest first.
    Window titles require Screen Recording permission; app names always work.
    Thumbnails are best-effort and skipped silently when capture fails.
    """
    if not supported():
        return []
    try:
        info = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("window enumeration failed: %s", exc)
        return []

    windows: list[dict] = []
    for win in info or []:
        try:
            if int(win.get("kCGWindowLayer", 0)) != 0:
                continue
            bounds = win.get("kCGWindowBounds") or {}
            w = int(bounds.get("Width", 0))
            h = int(bounds.get("Height", 0))
            if w <= 0 or h <= 0:
                continue
            app = str(win.get("kCGWindowOwnerName", "") or "").strip()
            title = str(win.get("kCGWindowName", "") or "").strip()
            wid = int(win.get("kCGWindowNumber", 0))
            if not wid or not app:
                continue
            # Skip our own tiny/utility windows and desktop wallpaper owners.
            if app in {"Window Server", "Dock", "WindowManager"}:
                continue
            windows.append(
                {
                    "window_id": wid,
                    "app": app,
                    "title": title,
                    "_area": w * h,
                }
            )
        except Exception:  # pragma: no cover - skip malformed entries
            continue

    windows.sort(key=lambda d: d["_area"], reverse=True)

    if include_thumbnails:
        for win in windows:
            try:
                png = capture_window(win["window_id"])
                if png:
                    win["thumb_b64"] = _downscale_b64(png, _THUMB_MAX_SIDE)
            except Exception:  # pragma: no cover - thumbnails are optional
                pass

    for win in windows:
        win.pop("_area", None)
    return windows


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_window(window_id: int) -> Optional[bytes]:
    """Capture a single window by id as PNG bytes (reads the window's own pixels).

    Returns None if the window is gone or capture failed (e.g. permission).
    """
    if not supported():
        return None
    try:
        cg_image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            window_id,
            kCGWindowImageBoundsIgnoreFraming,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("window capture failed (%s): %s", window_id, exc)
        return None
    if cg_image is None:
        return None
    try:
        rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
        # Empty/blank capture (0x0) => permission denied or window vanished.
        if rep.pixelsWide() == 0 or rep.pixelsHigh() == 0:
            return None
        data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
        return bytes(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("PNG encode failed (%s): %s", window_id, exc)
        return None


def capture_desktop() -> Optional[bytes]:
    """Capture the whole main display as PNG bytes via the ``screencapture`` CLI.

    ``screencapture`` exits non-zero without Screen Recording permission.
    """
    if not _IS_MACOS:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            # -x: no capture sound. -t png: format. Default = full screen.
            proc = subprocess.run(
                ["screencapture", "-x", "-t", "png", tmp.name],
                capture_output=True,
                timeout=15,
            )
            if proc.returncode != 0:
                logger.warning(
                    "screencapture failed rc=%s: %s",
                    proc.returncode,
                    proc.stderr.decode(errors="replace").strip(),
                )
                return None
            with open(tmp.name, "rb") as fh:
                data = fh.read()
            return data or None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("desktop capture failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Downscale + perceptual hash
# ---------------------------------------------------------------------------


def _downscale_b64(png_bytes: bytes, max_side: int = _MAX_SIDE) -> str:
    """Return base64 PNG with the longest side capped at *max_side* px."""
    if not _PIL_AVAILABLE:
        return base64.b64encode(png_bytes).decode()
    try:
        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / longest
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:  # pragma: no cover - fall back to original bytes
        return base64.b64encode(png_bytes).decode()


def image_dimensions(png_bytes: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` of PNG bytes, or ``(0, 0)`` if unknown."""
    if not _PIL_AVAILABLE:
        return (0, 0)
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            return (int(img.size[0]), int(img.size[1]))
    except Exception:  # pragma: no cover
        return (0, 0)


def perceptual_hash(png_bytes: bytes) -> Optional[str]:
    """Compute a 64-bit dHash as a 16-char hex string.

    dHash: downscale to 9x8 grayscale and compare adjacent pixels horizontally.
    Robust to minor rendering noise; used for cheap change detection.
    """
    if not _PIL_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((9, 8))
        px = list(img.getdata())
        bits = 0
        idx = 0
        for row in range(8):
            base = row * 9
            for col in range(8):
                left = px[base + col]
                right = px[base + col + 1]
                bits = (bits << 1) | (1 if left > right else 0)
                idx += 1
        return f"{bits:016x}"
    except Exception:  # pragma: no cover - defensive
        return None


def hashes_similar(a: Optional[str], b: Optional[str], threshold: int = 4) -> bool:
    """True if two dHashes are within *threshold* Hamming distance (unchanged)."""
    if not a or not b:
        return False
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1") <= threshold
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# High-level capture used by the route
# ---------------------------------------------------------------------------


def capture(
    mode: str,
    window_id: Optional[int] = None,
    last_hash: Optional[str] = None,
    hash_threshold: int = 4,
) -> dict:
    """Capture desktop or a window and return a result dict for the API.

    Shapes:
    - ``{"unsupported": True}`` — not macOS / frameworks missing.
    - ``{"needs_permission": True}`` — Screen Recording denied / blank capture.
    - ``{"window_gone": True}`` — followed window no longer exists.
    - ``{"unchanged": True, "hash": ...}`` — deduped against *last_hash*.
    - ``{"image_b64", "mime_type", "width", "height", "hash"}`` — success.
    """
    if not supported():
        return {"unsupported": True}

    png: Optional[bytes]
    if mode == "window":
        if not window_id:
            return {"error": "window_id required for window capture"}
        png = capture_window(int(window_id))
        if png is None:
            # Distinguish "gone" from "permission" when we can.
            if screen_recording_granted() is False:
                return {"needs_permission": True}
            return {"window_gone": True}
    else:  # desktop
        png = capture_desktop()
        if png is None:
            return {"needs_permission": True}

    new_hash = perceptual_hash(png)
    if last_hash and hashes_similar(last_hash, new_hash, hash_threshold):
        return {"unchanged": True, "hash": new_hash}

    width, height = image_dimensions(png)
    return {
        "image_b64": _downscale_b64(png, _MAX_SIDE),
        "mime_type": "image/png",
        "width": width,
        "height": height,
        "hash": new_hash,
    }
