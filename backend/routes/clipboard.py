"""Read file/folder references off the host OS clipboard.

Exists because of a WebView gap the UI can't work around on its own: when
Finder copies a *folder*, the DataTransfer the webview receives on paste has
no usable ``file://`` entry in ``text/uri-list``, and ``getAsFile()`` hands
back a 0-byte ``File`` named after the folder (a directory has no Blob
content to represent). The frontend therefore can't tell a copied folder
from an empty file, and has no absolute path to symlink either way.

OS-level *drag and drop* doesn't have this problem — Tauri intercepts the
drop natively and passes real paths through — so this endpoint gives paste
the same ground truth, keeping the two input methods behaviourally
identical (see ``tryNativeClipboardPaste`` in ``ChatPage.tsx``).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/clipboard", tags=["clipboard"])

# Finder Cmd+C writes file URLs (NSFilenamesPboardType / NSURL), not a classic
# alias. AppKit is the reliable read; `the clipboard as alias` is the fallback
# for a single item if AppKit is unavailable.
_CLIPBOARD_PATHS_SCRIPT = """
use framework "AppKit"
use framework "Foundation"
use scripting additions
set pb to current application's NSPasteboard's generalPasteboard()
set opts to current application's NSDictionary's dictionaryWithObject:(current application's NSNumber's numberWithBool:true) forKey:(current application's NSPasteboardURLReadingFileURLsOnlyKey)
set theURLs to pb's readObjectsForClasses:{(current application's NSURL)} options:opts
if theURLs is missing value then return ""
if (theURLs's |count|()) is 0 then return ""
set pathList to {}
repeat with theURL in theURLs
	if (theURL's isFileURL as boolean) then
		set end of pathList to (theURL's |path|() as text)
	end if
end repeat
set oldDelims to AppleScript's text item delimiters
set AppleScript's text item delimiters to linefeed
set out to pathList as text
set AppleScript's text item delimiters to oldDelims
return out
"""

_CLIPBOARD_ALIAS_SCRIPT = """try
return POSIX path of (the clipboard as alias)
on error
return ""
end try"""

# osascript talks to the window server; if that's wedged we'd rather report
# an empty clipboard than leave a paste hanging with no feedback.
_TIMEOUT_S = 3.0


async def _run_osascript(script: str) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        return ""
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("osascript clipboard read timed out after %ss", _TIMEOUT_S)
        return ""
    return stdout.decode("utf-8", errors="replace").strip()


async def _clipboard_paths() -> list[str]:
    """Return POSIX paths of file references on the clipboard, if any."""
    if sys.platform != "darwin":
        return []
    raw = await _run_osascript(_CLIPBOARD_PATHS_SCRIPT)
    paths = [p.strip() for p in raw.splitlines() if p.strip()]
    if paths:
        return paths
    fallback = await _run_osascript(_CLIPBOARD_ALIAS_SCRIPT)
    return [fallback] if fallback else []


def _describe(path: str) -> dict:
    # AppleScript may return folder paths with a trailing slash; strip it so
    # the path matches what Tauri's drag-drop hands over for the same folder.
    normalized = path.rstrip("/") or path
    try:
        is_dir = Path(normalized).is_dir()
    except OSError:
        is_dir = False
    return {"path": normalized, "is_dir": is_dir}


@router.get("/file")
async def api_clipboard_file() -> JSONResponse:
    """Describe file/folder references currently on the clipboard.

    Returns empty ``path`` / ``items`` when the clipboard holds no file
    reference, which the caller treats as "nothing to attach".
    """
    paths = await _clipboard_paths()
    items = [_describe(p) for p in paths]
    body = {
        "path": items[0]["path"] if items else "",
        "is_dir": items[0]["is_dir"] if items else False,
        "items": items,
    }
    return JSONResponse(content=body, headers={"Cache-Control": "no-store"})
