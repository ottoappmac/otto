"""LangChain tools for viewing uploaded files (images) within a session."""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import logging
import mimetypes
import socket
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_image_block, create_text_block
from langchain_core.tools import tool

from backend.utils import is_resolved_path_allowed, remap_to_virtual_path

logger = logging.getLogger(__name__)

_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
_FETCH_TIMEOUT = 15.0  # seconds

_DESCRIBE_IMAGE_PROMPT = (
    "Describe this image in detail. Include all visible text, objects, "
    "colours, layout, charts, diagrams, and any other notable content."
)


def _describe_with_vlm(
    vision_llm: BaseChatModel,
    raw: bytes,
    mime: str,
    prompt: str = _DESCRIBE_IMAGE_PROMPT,
) -> str:
    """Call *vision_llm* to describe an image given its raw bytes.

    Constructs an OpenAI-style ``image_url`` message block (the format that
    ``MLXVLChatModel._extract_images`` expects) and invokes the model
    synchronously.  *prompt* is the instruction sent alongside the image —
    callers pass a targeted question to query a specific image.  Returns the
    model's text output.
    """
    _vlm_name = getattr(vision_llm, "model", None) or type(vision_llm).__name__
    logger.info("Vision: describing %d-byte %s image via VLM '%s'", len(raw), mime, _vlm_name)
    b64 = base64.standard_b64encode(raw).decode()
    data_url = f"data:{mime};base64,{b64}"
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ])
    try:
        import time as _time
        t0 = _time.monotonic()
        response = vision_llm.invoke([msg])
        elapsed = _time.monotonic() - t0
        content = response.content
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            result = "\n".join(parts).strip() or str(content)
        else:
            result = str(content).strip()
        logger.info("Vision: VLM '%s' described image in %.1fs (%d chars)", _vlm_name, elapsed, len(result))
        return result
    except Exception as exc:
        logger.warning("VLM image description failed via '%s': %s", _vlm_name, exc)
        return f"[Image description unavailable: {exc}]"


def _desc_cache_path(files_dir: Path) -> Path:
    """Path to the per-session image-description cache sidecar."""
    return files_dir / "doc_images" / "descriptions.json"


def _desc_cache_key(resolved: Path) -> str:
    """Cache key combining the resolved path and its mtime."""
    try:
        mtime = resolved.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return hashlib.sha1(f"{resolved}:{mtime}".encode()).hexdigest()


def _load_desc_cache(files_dir: Path) -> dict[str, str]:
    try:
        return json.loads(_desc_cache_path(files_dir).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_desc_cache(files_dir: Path, cache: dict[str, str]) -> None:
    path = _desc_cache_path(files_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as exc:
        logger.debug("Could not persist image-description cache: %s", exc)


def build_file_tools(files_dir: Path, vision_llm: Optional[BaseChatModel] = None) -> list:
    """Build tools that let the agent view uploaded images in the session.

    Args:
        files_dir: Absolute path to the session's files directory.
        vision_llm: Optional dedicated vision-language model.  When provided
            the image is described as text by *vision_llm* instead of being
            forwarded as a raw base64 image block.  Pass this when the main
            session LLM is text-only (e.g. a local MLX text model) so that
            ``view_image`` still returns useful output rather than an opaque
            binary block the model cannot interpret.

    Returns:
        List of LangChain tool callables to inject into the deep agent.
    """

    @tool
    def view_image(file_path: str, question: str = "") -> list[dict[str, Any]]:
        """View an image file from the session filesystem and return it as
        visual content the model can see.

        Use this tool when the user uploads an image and you need to see its
        contents (e.g. screenshots, photos, diagrams), or to inspect an image
        extracted from a document (paths under "/doc_images/...").  Text-based
        files (csv, json, code, etc.) should be read with the regular file
        tools instead.

        Args:
            file_path: Path to the image relative to the session root
                       (e.g. "/uploads/screenshot.png" or
                       "/doc_images/report_ab12/p3_img2.png").
            question: Optional specific question to ask about the image
                      (e.g. "What is the value of the third bar in this
                      chart?").  When omitted, a general description is
                      returned.
        """
        # Accept the real absolute files_dir spelling by remapping it to the
        # virtual form, mirroring the session file tools.
        file_path = remap_to_virtual_path(file_path, files_dir)
        vpath = file_path if file_path.startswith("/") else "/" + file_path
        if ".." in vpath or vpath.startswith("~"):
            return [create_text_block(text="Error: path traversal is not allowed.")]

        full = files_dir / vpath.lstrip("/")
        # Containment allows user-dropped images under /links/ to resolve
        # outside the session root, while blocking symlink escapes elsewhere.
        if not is_resolved_path_allowed(full, files_dir, vpath):
            return [create_text_block(text="Error: path is outside the session directory.")]

        resolved = full.resolve()
        if not resolved.is_file():
            return [create_text_block(text=f"Error: file not found: {file_path}")]

        mime, _ = mimetypes.guess_type(str(resolved))
        if mime not in _IMAGE_MIMES:
            hint = "Use read_file or shell tools for non-image files."
            if mime in ("application/pdf",) or resolved.suffix.lower() in (
                ".pdf", ".docx", ".pptx",
            ):
                hint = (
                    "This is a document, not an image. Use doc_reader or "
                    "doc_research on it first — they extract embedded images "
                    "and return their /doc_images/... paths, which you then "
                    "pass to view_image."
                )
            return [create_text_block(
                text=f"Error: not a supported image type ({mime}). {hint}"
            )]

        size = resolved.stat().st_size
        if size > _MAX_IMAGE_SIZE:
            return [create_text_block(
                text=f"Error: image is too large ({size / 1024 / 1024:.1f} MB, "
                     f"max {_MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB)."
            )]

        raw = resolved.read_bytes()
        q = question.strip()

        if vision_llm is not None:
            # Generic (no-question) descriptions are cached per path+mtime so
            # repeat views are free; targeted questions are always re-run.
            if not q:
                cache = _load_desc_cache(files_dir)
                key = _desc_cache_key(resolved)
                description = cache.get(key)
                if description is None:
                    description = _describe_with_vlm(vision_llm, raw, mime)
                    cache[key] = description
                    _save_desc_cache(files_dir, cache)
            else:
                description = _describe_with_vlm(vision_llm, raw, mime, prompt=q)
            header = f"Image: {file_path} ({mime}, {size / 1024:.0f} KB)"
            if q:
                header += f"\nQuestion: {q}"
            return [create_text_block(text=f"{header}\n\n{description}")]

        data = base64.standard_b64encode(raw).decode()
        text = f"Image: {file_path} ({mime}, {size / 1024:.0f} KB)"
        if q:
            text += f"\nQuestion: {q}"
        return [
            create_text_block(text=text),
            create_image_block(base64=data, mime_type=mime),
        ]

    @tool
    def load_image_from_url(url: str) -> list[dict[str, Any]]:
        """Fetch an image from a public URL and return it as visual content
        the model can see.

        Use this when the user provides a URL pointing to an image (PNG, JPEG,
        GIF, or WebP) and you need to inspect its contents.  For images already
        uploaded to the session, use view_image instead.

        Args:
            url: Fully-qualified HTTP or HTTPS URL of the image
                 (e.g. "https://example.com/photo.jpg").
        """
        import httpx
        from PIL import Image

        # 1. Scheme guard — only http/https
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return [create_text_block(text="Error: only http and https URLs are supported.")]

        hostname = parsed.hostname or ""
        if not hostname:
            return [create_text_block(text="Error: URL has no hostname.")]

        # 2. SSRF guard — block private/loopback/link-local addresses
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return [create_text_block(
                    text="Error: requests to private or internal addresses are not allowed."
                )]
        except OSError:
            return [create_text_block(text=f"Error: could not resolve host: {hostname}")]

        # 3. Streaming fetch — check Content-Type and size before reading body
        try:
            with httpx.Client(follow_redirects=True, timeout=_FETCH_TIMEOUT) as client:
                with client.stream("GET", url, headers={"User-Agent": "Otto/1.0"}) as resp:
                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "").split(";")[0].strip()
                    if content_type not in _IMAGE_MIMES:
                        return [create_text_block(
                            text=f"Error: URL does not point to a supported image type "
                                 f"({content_type or 'unknown'}). "
                                 "Supported types: PNG, JPEG, GIF, WebP."
                        )]

                    declared = int(resp.headers.get("content-length", 0))
                    if declared > _MAX_IMAGE_SIZE:
                        return [create_text_block(
                            text=f"Error: image exceeds the "
                                 f"{_MAX_IMAGE_SIZE // (1024 * 1024)} MB size limit."
                        )]

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        total += len(chunk)
                        if total > _MAX_IMAGE_SIZE:
                            return [create_text_block(
                                text=f"Error: image exceeds the "
                                     f"{_MAX_IMAGE_SIZE // (1024 * 1024)} MB size limit."
                            )]
                        chunks.append(chunk)

        except httpx.HTTPStatusError as exc:
            return [create_text_block(text=f"Error: HTTP {exc.response.status_code} fetching URL.")]
        except httpx.RequestError as exc:
            return [create_text_block(text=f"Error: network error fetching URL: {exc}")]

        raw = b"".join(chunks)

        # 4. Verify bytes are actually a valid image (guards against HTML/error pages
        #    served with an image Content-Type)
        try:
            img = Image.open(io.BytesIO(raw))
            img.verify()
        except Exception:
            return [create_text_block(
                text="Error: downloaded content is not a valid image."
            )]

        logger.debug("load_image_from_url: fetched %s (%s, %d KB)", url, content_type, total // 1024)

        if vision_llm is not None:
            description = _describe_with_vlm(vision_llm, raw, content_type)
            return [create_text_block(
                text=f"Image from URL: {url} ({content_type}, {total / 1024:.0f} KB)\n\n{description}"
            )]

        data = base64.standard_b64encode(raw).decode()
        return [
            create_text_block(
                text=f"Image from URL: {url} ({content_type}, {total / 1024:.0f} KB)"
            ),
            create_image_block(base64=data, mime_type=content_type),
        ]

    return [view_image, load_image_from_url]
