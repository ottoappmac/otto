"""Native video understanding via the Google Gemini SDK.

Gemini is the only provider that watches video natively: it samples both
frames and the audio track itself, supports public YouTube URLs, and lets
us pick a custom frame rate (``videoMetadata.fps``) and media resolution.

This module talks to the ``google-genai`` SDK directly (rather than
through LangChain) so we get full control over the File API upload flow
and the per-part ``videoMetadata`` that LangChain does not surface.

Small clips (< ~20 MB) are sent inline; larger/longer clips go through
the File API.  YouTube URLs are passed straight through with no download.
"""

from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Inline data is capped by the API at ~20 MB per request; above this we
# must use the File API.
_INLINE_MAX_BYTES = 18 * 1024 * 1024

_DEFAULT_MODEL = "gemini-2.5-flash"


def _media_resolution_enum(types, media_resolution: str):
    """Map our ``default``/``low`` string to the SDK's MediaResolution enum."""
    if (media_resolution or "").lower() == "low":
        return types.MediaResolution.MEDIA_RESOLUTION_LOW
    return types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED


def _client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def _guess_mime(path: str | Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "video/mp4"


def _wait_active(client, file_obj, *, timeout: float = 300.0):
    """Poll a File API upload until it becomes ACTIVE (or fails/times out)."""
    deadline = time.monotonic() + timeout
    while getattr(file_obj.state, "name", str(file_obj.state)) == "PROCESSING":
        if time.monotonic() > deadline:
            raise TimeoutError("Gemini File API processing timed out")
        time.sleep(2.0)
        file_obj = client.files.get(name=file_obj.name)
    state = getattr(file_obj.state, "name", str(file_obj.state))
    if state == "FAILED":
        raise RuntimeError("Gemini File API failed to process the video")
    return file_obj


def understand_video_sync(
    *,
    api_key: str,
    model: str,
    prompt: str,
    source: str,
    is_youtube: bool,
    fps: float = 1.0,
    media_resolution: str = "default",
    start_secs: Optional[float] = None,
    end_secs: Optional[float] = None,
    max_output_tokens: int = 16384,
    temperature: float = 0.0,
) -> str:
    """Ask Gemini about a video and return its text answer.

    *source* is either a local file path or a YouTube URL (when
    *is_youtube* is True).  Blocking — call via ``asyncio.to_thread``.
    """
    from google.genai import types

    client = _client(api_key)
    model = model or _DEFAULT_MODEL

    # Build the video Part with an optional custom frame rate + clip window.
    video_metadata_kwargs: dict = {}
    if fps and fps > 0:
        video_metadata_kwargs["fps"] = float(fps)
    if start_secs is not None and start_secs > 0:
        video_metadata_kwargs["start_offset"] = f"{int(start_secs)}s"
    if end_secs is not None and end_secs > 0:
        video_metadata_kwargs["end_offset"] = f"{int(end_secs)}s"
    video_metadata = types.VideoMetadata(**video_metadata_kwargs) if video_metadata_kwargs else None

    if is_youtube:
        part = types.Part(
            file_data=types.FileData(file_uri=source, mime_type="video/*"),
            video_metadata=video_metadata,
        )
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {source}")
        mime = _guess_mime(path)
        size = path.stat().st_size
        if size <= _INLINE_MAX_BYTES:
            data = path.read_bytes()
            part = types.Part(
                inline_data=types.Blob(data=data, mime_type=mime),
                video_metadata=video_metadata,
            )
        else:
            logger.info("Gemini File API upload: %s (%.1f MB)", path.name, size / 1e6)
            uploaded = client.files.upload(file=str(path))
            uploaded = _wait_active(client, uploaded)
            part = types.Part(
                file_data=types.FileData(file_uri=uploaded.uri, mime_type=mime),
                video_metadata=video_metadata,
            )

    # Per Gemini best practice, place the text prompt *after* the video part.
    contents = [
        types.Content(role="user", parts=[part, types.Part(text=prompt)]),
    ]

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        media_resolution=_media_resolution_enum(types, media_resolution),
    )

    resp = client.models.generate_content(model=model, contents=contents, config=config)
    return (getattr(resp, "text", None) or "").strip()
