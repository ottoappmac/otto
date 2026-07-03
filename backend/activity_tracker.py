"""User activity timeline — local, screenshot-free, on-device.

Background loop that polls the foreground macOS application every N seconds
and records ``(app, window title, browser URL, active document)`` into a
local SQLite DB with FTS5 indexing.  No pixels are captured — only the
metadata most useful for "what was I doing at 3pm last Tuesday".

The tracker is **opt-in** (``AppConfig.activity.enabled``) and runs
entirely on-device; nothing is ever sent over the network.

Design notes
------------
* SQLite + FTS5 — keyword + date queries cover ~95% of realistic recall
  needs without the LLM cost of an embedding pipeline.  We can layer
  ``sqlite-vec`` on later if usage warrants it.
* Lexically deduplicates consecutive identical rows — if you stay on the
  same window for 10 minutes we record one row whose ``duration_s`` is
  bumped on each tick rather than 40 identical rows.
* macOS-only for now (the AppKit / Accessibility / AppleScript stack we
  need lives in ``pyobjc``); no-op on other platforms so the rest of the
  backend keeps working.
* Soft-fails on missing accessibility permissions — the tracker logs a
  warning and keeps polling, so granting permission later "just works"
  without restarting the backend.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from backend.config import ActivityConfig, get_app_data_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------


# Patterns that defeat naive (app, title, url) deduplication because they
# tick along with the wall clock or message counts.  Stripping them lets
# the tracker collapse runs of "Inbox (3) — 14:32" / "Inbox (3) — 14:33"
# into a single span.
_TIME_PATTERNS = [
    # 14:32, 14:32:05, 14:32 PM, 2:32 pm, etc., usually preceded by " — "
    # or " - " or " | ".  Strip both the separator and the time.
    re.compile(r"\s*[—\-|·]\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?\s*$"),
    # Trailing unread/notification counts: " (3)", " [12]"
    re.compile(r"\s*[\(\[]\d+[\)\]]\s*$"),
    # Trailing "— Updating…" / "Loading…" spinners
    re.compile(r"\s*[—\-]\s*(Updating|Loading|Saving|Syncing)\.{0,3}\s*$", re.IGNORECASE),
]
_WHITESPACE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Collapse whitespace and strip volatile suffixes from a window title.

    The goal isn't perfect normalization — it's reducing the most common
    sources of pathological row growth so the dedup key becomes stable
    across consecutive ticks.
    """
    if not title:
        return ""
    s = title.strip()
    # Strip volatile suffixes; keep stripping while patterns continue to
    # match (a title can have multiple stacked annotations).
    changed = True
    while changed:
        changed = False
        for pat in _TIME_PATTERNS:
            new = pat.sub("", s)
            if new != s:
                s = new.strip()
                changed = True
    return _WHITESPACE.sub(" ", s).strip()


# Separator for entries inside a row's rolling context log.  Picked so
# it never collides with the ``"sel: ..." | "input: ..."`` separator
# that ``_get_active_macos`` uses inside a single snapshot.
_CTX_SEP = "\n"


def _merge_context(prev: str, new: str, *, max_chars: int) -> str:
    """Merge a new context snapshot into a rolling history within a row.

    Behaviour, in order:
      * Empty ``new`` → keep ``prev`` unchanged.
      * Empty ``prev`` → ``new`` becomes the first entry.
      * ``new`` exactly matches the most-recent entry → no change.
      * ``new`` extends the last entry (user typed more chars in the
        same field) → replace that entry rather than appending, so the
        log doesn't fill with prefixes of the same input.
      * ``new`` is a strict shortening of the last entry (user
        backspaced) → keep the longer existing version.
      * ``new`` matches some earlier entry verbatim → skip (no dup).
      * Otherwise → append as a new entry.

    When the merged buffer exceeds ``max_chars``, the oldest entries
    are trimmed off the front until it fits, then a hard slice is
    applied as a final safety net.  ``max_chars <= 0`` skips the
    rolling log entirely and returns ``new`` (overwrite semantics).
    """
    if max_chars <= 0:
        return new
    if not new:
        return prev
    if not prev:
        return new
    entries = prev.split(_CTX_SEP) if prev else []
    last = entries[-1] if entries else ""

    if new == last:
        return prev
    # Treat short tails as too noisy to merge into — only fold typing
    # extensions when there's a meaningful prefix to anchor on.
    _MIN_OVERLAP = 8
    if last and new.startswith(last) and len(last) >= _MIN_OVERLAP:
        entries[-1] = new
    elif last and last.startswith(new) and len(new) >= _MIN_OVERLAP:
        return prev
    elif new in entries:
        return prev
    else:
        entries.append(new)

    merged = _CTX_SEP.join(entries)
    while len(merged) > max_chars and len(entries) > 1:
        entries.pop(0)
        merged = _CTX_SEP.join(entries)
    if len(merged) > max_chars:
        merged = merged[-max_chars:]
    return merged


# ---------------------------------------------------------------------------
# Idle detection (macOS)
# ---------------------------------------------------------------------------


def _seconds_since_last_input() -> Optional[float]:
    """Return seconds since the last keyboard or mouse event on macOS.

    Returns ``None`` if the API isn't available (non-macOS, missing
    pyobjc).  Used to skip recording during AFK periods like screen-saver
    or overnight when the user isn't actually present.
    """
    try:
        from Quartz import (
            CGEventSourceSecondsSinceLastEventType,
            kCGEventSourceStateHIDSystemState,
            kCGAnyInputEventType,
        )
    except ImportError:
        return None
    try:
        return float(
            CGEventSourceSecondsSinceLastEventType(
                kCGEventSourceStateHIDSystemState,
                kCGAnyInputEventType,
            )
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    p = get_app_data_dir() / "activity.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def _open_db() -> Any:
    conn = sqlite3.connect(_db_path(), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Incremental auto-vacuum frees space back to the OS when retention
    # prunes old rows — without this the .db file grows monotonically
    # even as we delete data.  The PRAGMA only takes effect on a fresh
    # database, so existing files stay on auto_vacuum=NONE; that's fine
    # because the periodic VACUUM in ``_maybe_compact`` covers them.
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # Base table — created on first run, never altered destructively so
    # older databases keep working untouched.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           INTEGER NOT NULL,
            app          TEXT    NOT NULL,
            title        TEXT    DEFAULT '',
            url          TEXT    DEFAULT '',
            file_path    TEXT    DEFAULT '',
            duration_s   INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_activity_ts  ON activity(ts);
        CREATE INDEX IF NOT EXISTS idx_activity_app ON activity(app);
    """)

    # ── Tier 1 enrichment migration ──────────────────────────────────
    # Add the ``context`` column (selection text, focused-input value)
    # and rebuild the FTS index so the new column is searchable.  Both
    # halves are idempotent — they only do work the first time the new
    # schema is seen.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(activity)").fetchall()}
    if "context" not in cols:
        conn.execute("ALTER TABLE activity ADD COLUMN context TEXT DEFAULT ''")

    needs_fts_rebuild = False
    try:
        fts_cols = {row[1] for row in conn.execute("PRAGMA table_info(activity_fts)").fetchall()}
        if fts_cols and "context" not in fts_cols:
            needs_fts_rebuild = True
    except sqlite3.OperationalError:
        pass

    if needs_fts_rebuild:
        conn.executescript("""
            DROP TRIGGER IF EXISTS activity_ai;
            DROP TRIGGER IF EXISTS activity_ad;
            DROP TRIGGER IF EXISTS activity_au;
            DROP TABLE IF EXISTS activity_fts;
        """)

    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS activity_fts USING fts5(
            app, title, url, file_path, context,
            content='activity', content_rowid='id', tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS activity_ai AFTER INSERT ON activity BEGIN
          INSERT INTO activity_fts(rowid, app, title, url, file_path, context)
          VALUES (new.id, new.app, new.title, new.url, new.file_path, new.context);
        END;
        CREATE TRIGGER IF NOT EXISTS activity_ad AFTER DELETE ON activity BEGIN
          INSERT INTO activity_fts(activity_fts, rowid, app, title, url, file_path, context)
          VALUES ('delete', old.id, old.app, old.title, old.url, old.file_path, old.context);
        END;
        CREATE TRIGGER IF NOT EXISTS activity_au AFTER UPDATE ON activity BEGIN
          INSERT INTO activity_fts(activity_fts, rowid, app, title, url, file_path, context)
          VALUES ('delete', old.id, old.app, old.title, old.url, old.file_path, old.context);
          INSERT INTO activity_fts(rowid, app, title, url, file_path, context)
          VALUES (new.id, new.app, new.title, new.url, new.file_path, new.context);
        END;
    """)

    if needs_fts_rebuild:
        # Backfill the rebuilt FTS index from the existing rows so search
        # over historical activity still works post-migration.
        conn.execute("""
            INSERT INTO activity_fts(rowid, app, title, url, file_path, context)
            SELECT id, app, title, url, file_path, COALESCE(context, '')
            FROM activity
        """)


# ---------------------------------------------------------------------------
# macOS frontmost-app inspection
# ---------------------------------------------------------------------------


def _get_active_macos(
    field_val_max_len: int = 8000,
    *,
    browser_text_max_chars: int = 4000,
    ax_walk_max_chars: int = 2000,
    ax_walk_max_depth: int = 5,
) -> Optional[dict[str, str]]:
    """Return the frontmost macOS app + its window title / URL / doc path.

    Uses Quartz' CGWindowList to query the WindowServer directly — this is
    real-time, whereas ``NSWorkspace.frontmostApplication()`` depends on
    NSNotifications delivered via the Cocoa runloop, which is **not**
    spinning inside a pure-asyncio FastAPI process.  Without a runloop
    the NSWorkspace cache stays frozen at whatever was frontmost when
    the server started, so all polls return the same stale app.

    Returns ``None`` on any failure (missing permissions, no frontmost
    window, pyobjc unavailable).
    """
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except ImportError:
        return None

    try:
        opts = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
        windows = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) or []
    except Exception:
        return None

    app_name = ""
    pid = 0
    for w in windows:
        # Layer 0 = ordinary application windows; >0 = menubar, dock,
        # screen-saver, etc.  CGWindowList returns windows in z-order
        # so the first layer-0 entry is the frontmost.
        try:
            if int(w.get("kCGWindowLayer", 0)) != 0:
                continue
        except Exception:
            continue
        owner = w.get("kCGWindowOwnerName")
        if not owner:
            continue
        app_name = str(owner)
        try:
            pid = int(w.get("kCGWindowOwnerPID", 0))
        except Exception:
            pid = 0
        break
    if not app_name or pid <= 0:
        return None

    bundle_id = ""
    try:
        from AppKit import NSRunningApplication
        ra = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if ra is not None:
            bundle_id = str(ra.bundleIdentifier() or "")
    except Exception:
        pass

    # Tier 1 metadata enrichment — pull richer context from the
    # Accessibility tree.  All reads are best-effort and capped to a
    # few hundred chars so a single noisy app can't bloat the DB.
    title = ""
    ax_url = ""
    ax_doc = ""
    selection = ""
    field_val = ""
    ax_tree_text = ""
    focused_window = None
    try:
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
        )
        ax = AXUIElementCreateApplication(pid)
        err, focused = AXUIElementCopyAttributeValue(ax, "AXFocusedWindow", None)
        if err == 0 and focused is not None:
            focused_window = focused
            title = _ax_str(focused, "AXTitle", max_len=200)
            ax_url = _ax_str(focused, "AXURL", max_len=500)
            ax_doc = _ax_str(focused, "AXDocument", max_len=500)
        err, fe = AXUIElementCopyAttributeValue(ax, "AXFocusedUIElement", None)
        if err == 0 and fe is not None:
            # Never read the contents of a password field.
            sub = _ax_str(fe, "AXSubrole", max_len=64)
            if sub != "AXSecureTextField":
                selection = _ax_str(fe, "AXSelectedText", max_len=300)
                field_val = _ax_str(fe, "AXValue", max_len=field_val_max_len)
    except Exception:
        pass

    # Prefer the URL exposed via Accessibility (no AppleScript prompt)
    # and only fall back to AppleScript when AX didn't give us one.
    url = ax_url or _get_browser_url(bundle_id)
    file_path = _strip_file_uri(ax_doc) if ax_doc else ""

    # Phase 0 — browser DOM body text (only when the active app is a
    # supported browser).  The AppleScript bridge keeps this fast
    # (~10-50ms) and the cap stops a giant SPA from bloating storage.
    page_text = ""
    is_browser = _is_browser_bundle(bundle_id)
    if is_browser and browser_text_max_chars > 0:
        page_text = _get_browser_page_text(
            bundle_id, max_chars=browser_text_max_chars,
        )

    # Phase 0 — AX tree walk for non-browser apps.  Captures rich
    # context for Electron apps (Cursor, VS Code, Slack, Discord,
    # Notion) and AppKit apps where AXFocusedUIElement alone is thin.
    if (
        not is_browser
        and ax_walk_max_chars > 0
        and focused_window is not None
    ):
        try:
            ax_tree_text = _ax_walk_text(
                focused_window,
                max_chars=ax_walk_max_chars,
                max_depth=ax_walk_max_depth,
            )
        except Exception:
            ax_tree_text = ""

    # Build a compact, searchable context blob.  Skip duplicates of the
    # title / selection so the same string isn't repeated three times.
    context_bits: list[str] = []
    if selection:
        context_bits.append(f"sel: {selection}")
    if field_val and field_val not in (selection, title):
        context_bits.append(f"input: {field_val}")
    if page_text and page_text not in (title, selection, field_val):
        context_bits.append(f"page: {page_text}")
    if ax_tree_text and ax_tree_text not in (title, selection, field_val):
        context_bits.append(f"ui: {ax_tree_text}")
    context = " | ".join(context_bits)

    return {
        "app": app_name,
        "title": title,
        "url": url,
        "file_path": file_path,
        "context": context,
    }


def _ax_str(elem: Any, attr: str, *, max_len: int = 500) -> str:
    """Best-effort read of an Accessibility attribute as a clean string.

    Returns an empty string on any failure (attribute missing, type
    mismatch, AX not granted).  Caps the length so a single huge text
    field can't blow up a row's storage.
    """
    try:
        from ApplicationServices import AXUIElementCopyAttributeValue
    except ImportError:
        return ""
    try:
        err, val = AXUIElementCopyAttributeValue(elem, attr, None)
        if err != 0 or val is None:
            return ""
        s = str(val).strip()
        if max_len and len(s) > max_len:
            s = s[:max_len].rstrip() + "…"
        return s
    except Exception:
        return ""


def _ax_walk_text(
    root: Any,
    *,
    max_chars: int = 3000,
    max_depth: int = 25,
    max_nodes: int = 4000,
) -> str:
    """Recursively descend an Accessibility tree and harvest visible text.

    Captures *AXTitle*, *AXValue*, and *AXDescription* from every
    element down to *max_depth* levels, dedups, and joins with " ¦ "
    separators.  This is the Tier 2 fallback for apps where the
    focused element alone (Tier 0/1) doesn't yield useful context —
    Electron apps (Cursor, VS Code, Slack, Discord, Notion), AppKit
    apps with multi-pane layouts, and anything else exposing data
    only via deeper tree positions.

    Skips secure text fields, splash images, and obvious chrome
    (lines shorter than 3 chars or longer than 250).  Hard-bounded
    by *max_nodes* so a runaway tree (some Electron apps emit
    thousands of nodes per window) can't stall the loop.

    Returns empty string if the AX framework isn't importable or
    permission was denied.
    """
    if max_chars <= 0 or max_depth <= 0 or root is None:
        return ""
    try:
        from ApplicationServices import AXUIElementCopyAttributeValue
    except ImportError:
        return ""

    seen: set[str] = set()
    out: list[str] = []
    out_len = 0
    node_count = 0

    def _read(elem: Any, attr: str) -> str:
        try:
            err, val = AXUIElementCopyAttributeValue(elem, attr, None)
            if err != 0 or val is None:
                return ""
            s = str(val).strip()
            return s
        except Exception:
            return ""

    def visit(elem: Any, depth: int) -> None:
        nonlocal out_len, node_count
        if depth > max_depth or out_len >= max_chars or node_count >= max_nodes:
            return
        node_count += 1

        sub = _read(elem, "AXSubrole")
        if sub == "AXSecureTextField":
            return

        for attr in ("AXTitle", "AXValue", "AXDescription"):
            v = _read(elem, attr)
            if not v:
                continue
            if len(v) < 3 or len(v) > 500:
                continue
            if v in seen:
                continue
            seen.add(v)
            remaining = max_chars - out_len
            if remaining <= 0:
                return
            if len(v) > remaining:
                v = v[:remaining].rstrip() + "…"
            out.append(v)
            out_len += len(v) + 3

        try:
            err, kids = AXUIElementCopyAttributeValue(elem, "AXChildren", None)
        except Exception:
            return
        if err != 0 or not kids:
            return
        for k in kids:
            if out_len >= max_chars or node_count >= max_nodes:
                return
            visit(k, depth + 1)

    try:
        visit(root, 0)
    except Exception:
        return ""

    return " ¦ ".join(out)


def _strip_file_uri(s: str) -> str:
    """Turn ``file:///Users/foo/bar.txt`` into ``/Users/foo/bar.txt``."""
    if not s:
        return ""
    if s.startswith("file://"):
        from urllib.parse import unquote
        return unquote(s[len("file://"):])
    return s


def _is_browser_bundle(bundle_id: str) -> bool:
    """Detect Safari / Chrome / Arc / Brave by bundle id."""
    if not bundle_id:
        return False
    bundle = bundle_id.lower()
    return (
        "safari" in bundle
        or "google.chrome" in bundle
        or "company.thebrowser.browser" in bundle
        or bundle.endswith(".arc")
        or "brave.browser" in bundle
    )


def _get_browser_url(bundle_id: str) -> str:
    """Return the active tab URL for Safari / Chrome / Arc / Brave.

    Uses AppleScript via ``osascript`` — fast (~5ms) and the only
    reliable way without injecting into the browser.  Returns empty
    string on any failure or for non-browser apps.
    """
    if not bundle_id:
        return ""
    bundle = bundle_id.lower()
    if "safari" in bundle:
        script = 'tell application "Safari" to return URL of current tab of window 1'
    elif "google.chrome" in bundle:
        script = 'tell application "Google Chrome" to return URL of active tab of window 1'
    elif "company.thebrowser.browser" in bundle or "arc" in bundle:
        script = 'tell application "Arc" to return URL of active tab of window 1'
    elif "brave.browser" in bundle:
        script = 'tell application "Brave Browser" to return URL of active tab of window 1'
    else:
        return ""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


# JavaScript snippet executed inside the active browser tab.  Pulls
# title, top headings, and a slice of body text — joined with " ¦ "
# separators.  Stays small (a few hundred bytes) so AppleScript's
# argument size limits don't bite.  No external network access.
_BROWSER_PAGE_TEXT_JS = (
    "(()=>{"
    "try{"
    "const t=document.title||'';"
    "const hs=[...document.querySelectorAll('h1,h2')].slice(0,6)"
    ".map(e=>(e.innerText||'').trim()).filter(Boolean).join(' ¦ ');"
    "const body=((document.body&&document.body.innerText)||'')"
    ".replace(/\\s+/g,' ').trim().slice(0,__MAX__);"
    "return [t,hs,body].filter(Boolean).join(' ¦ ');"
    "}catch(e){return '';}"
    "})()"
)


def _get_browser_page_text(bundle_id: str, *, max_chars: int = 4000) -> str:
    """Return the visible body text of the active browser tab.

    Uses AppleScript ``do JavaScript`` (Safari) / ``execute javascript``
    (Chromium-family) to read ``document.body.innerText`` from the
    front document.  Output is title + headings + body, joined with
    " ¦ " separators and capped at *max_chars*.

    Returns empty string on any failure (timeout, permission denied,
    JavaScript disabled-from-Apple-Events, non-browser app).  Soft-fails
    silently so a slow page can't stall the polling loop.

    Notes:
      - Safari requires Develop → "Allow JavaScript from Apple Events".
      - Chrome/Brave require View → Developer → "Allow JavaScript from
        Apple Events" (Chrome ≥ 80).
      - Arc allows by default.
      - 3-second hard timeout — covers slow pages + osascript bridge.
    """
    if max_chars <= 0 or not bundle_id:
        return ""
    bundle = bundle_id.lower()
    js = _BROWSER_PAGE_TEXT_JS.replace("__MAX__", str(max_chars))
    if "safari" in bundle:
        script = (
            'tell application "Safari" to do JavaScript "'
            + js.replace("\\", "\\\\").replace('"', '\\"')
            + '" in current tab of window 1'
        )
    elif "google.chrome" in bundle:
        script = (
            'tell application "Google Chrome" to execute active tab of window 1 '
            'javascript "' + js.replace("\\", "\\\\").replace('"', '\\"') + '"'
        )
    elif "company.thebrowser.browser" in bundle or bundle.endswith(".arc"):
        script = (
            'tell application "Arc" to execute active tab of window 1 '
            'javascript "' + js.replace("\\", "\\\\").replace('"', '\\"') + '"'
        )
    elif "brave.browser" in bundle:
        script = (
            'tell application "Brave Browser" to execute active tab of window 1 '
            'javascript "' + js.replace("\\", "\\\\").replace('"', '\\"') + '"'
        )
    else:
        return ""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            return out[:max_chars]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------


class ActivityTracker:
    """Lifecycle wrapper around the background capture task.

    Reads the live ``ActivityConfig`` on every tick so the user can flip
    settings in the UI without restarting the backend — toggling
    ``enabled`` simply pauses/resumes capture, ``interval_secs`` retunes
    the poll cadence, and ``exclude_apps`` filters out the next sample
    that would have hit a sensitive app.
    """

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # Cached row used to coalesce repeated identical samples into a
        # single (app, title, url) span with a growing ``duration_s``.
        self._last: Optional[dict[str, Any]] = None
        self._last_id: Optional[int] = None
        self._cleanup_due_at: float = 0.0
        self._compact_due_at: float = 0.0

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if platform.system() != "Darwin":
            logger.info("ActivityTracker: skipped (only macOS supported)")
            return
        with _open_db() as conn:
            _ensure_schema(conn)
        self._stop.clear()
        self._task = asyncio.create_task(self._run())
        logger.info("ActivityTracker started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            except Exception:
                pass
            self._task = None
        # Flush any pending duration into the DB so we don't lose the
        # tail end of the current span on shutdown.
        self._flush_last()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cfg = await self._load_config()
                if not cfg.enabled:
                    await self._sleep(2.0)
                    continue
                await self._tick(cfg)
                await self._maybe_cleanup(cfg)
                await self._sleep(max(2, cfg.interval_secs))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ActivityTracker tick failed")
                await self._sleep(5.0)

    async def _sleep(self, secs: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=secs)
        except asyncio.TimeoutError:
            pass

    async def _load_config(self) -> ActivityConfig:
        from backend.config import AppConfig
        cfg = await AppConfig.aload()
        return cfg.activity

    # Warn once when idle reads implausibly high — usually means the
    # process lacks Input Monitoring permission or input arrives via a
    # remote-desktop channel that bypasses the HID system state.
    _warned_idle_stale: bool = False

    async def _tick(self, cfg: ActivityConfig) -> None:
        # Idle gate — if the user hasn't touched keyboard/mouse for a
        # while, flush the in-flight span and skip recording.  This is
        # the single biggest row-reducer for typical users.
        if cfg.idle_threshold_secs > 0:
            idle = await asyncio.to_thread(_seconds_since_last_input)
            if idle is not None and idle >= cfg.idle_threshold_secs:
                # Warn once when the idle counter looks stale (> 4 hours),
                # which typically means Input Monitoring permission is
                # missing or input is arriving via a remote channel (VNC,
                # Splashtop, etc.) that doesn't update the HID system state.
                # In that case every tick is silently skipped and no rows
                # are ever written to the DB.
                _4H = 4 * 3600
                if idle > _4H and not ActivityTracker._warned_idle_stale:
                    ActivityTracker._warned_idle_stale = True
                    logger.warning(
                        "ActivityTracker: idle counter is %.0fs (>4 h) — "
                        "no activity will be recorded. This usually means "
                        "Input Monitoring permission is not granted to this "
                        "process, or keyboard/mouse input is arriving via a "
                        "remote-desktop channel (VNC, Splashtop, etc.) that "
                        "bypasses the HID system state. "
                        "Set idle_threshold_secs=0 in settings to disable "
                        "the idle gate and record activity regardless.",
                        idle,
                    )
                self._flush_last(min_span_secs=cfg.min_span_secs)
                return

        sample = await asyncio.to_thread(
            _get_active_macos,
            cfg.field_val_max_chars,
            browser_text_max_chars=cfg.browser_text_max_chars,
            ax_walk_max_chars=cfg.ax_walk_max_chars,
            ax_walk_max_depth=cfg.ax_walk_max_depth,
        )
        if sample is None:
            return

        # Normalize the title before it enters the dedup signature so
        # clock-tickers ("Inbox — 14:32" → "Inbox — 14:33") collapse.
        sample["title"] = _normalize_title(sample.get("title", ""))

        excluded = [x.strip().lower() for x in cfg.exclude_apps if x.strip()]
        app_lower = sample["app"].lower()
        if any(ex in app_lower for ex in excluded):
            return

        now = int(time.time())
        sig = (sample["app"], sample["title"], sample["url"])
        same_window = (
            self._last is not None
            and sig == (self._last["app"], self._last["title"], self._last["url"])
        )

        # Time-shard long single-window spans so deep focus sessions don't
        # collapse into one row that loses the per-moment context history.
        span_capped = (
            same_window
            and cfg.max_span_secs > 0
            and self._last is not None
            and (now - int(self._last["started_at"])) >= cfg.max_span_secs
        )

        if same_window and not span_capped:
            self._bump_duration(
                now,
                context=sample.get("context", ""),
                max_ctx_chars=cfg.context_max_chars,
            )
            return

        self._flush_last(min_span_secs=cfg.min_span_secs)
        await asyncio.to_thread(self._insert_row, sample, now)
        self._last = {**sample, "started_at": now}

    def _insert_row(self, sample: dict[str, str], now: int) -> None:
        with _open_db() as conn:
            cur = conn.execute(
                "INSERT INTO activity (ts, app, title, url, file_path, context, duration_s) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    now,
                    sample["app"],
                    sample["title"],
                    sample["url"],
                    sample.get("file_path", ""),
                    sample.get("context", ""),
                ),
            )
            self._last_id = cur.lastrowid

    def _bump_duration(
        self,
        now: int,
        *,
        context: str = "",
        max_ctx_chars: int = 0,
    ) -> None:
        """Extend the in-flight span and grow its rolling context log.

        Dedup is keyed on ``(app, title, url)`` so the user can stay on
        the same window for many ticks.  Rather than overwriting the
        row's ``context`` with whatever was on screen at the *latest*
        tick (which loses everything that happened earlier in the
        span), we merge each new snapshot into a rolling log via
        :func:`_merge_context` — typing extensions collapse, distinct
        moments accumulate, and the buffer is trimmed from the front
        when it overflows ``max_ctx_chars``.

        Set ``max_ctx_chars=0`` to fall back to overwrite-only
        semantics (the old behaviour).
        """
        if self._last is None or self._last_id is None:
            return
        elapsed = now - int(self._last["started_at"])
        last_ctx = self._last.get("context", "") or ""
        merged = _merge_context(last_ctx, context, max_chars=max_ctx_chars)
        try:
            with _open_db() as conn:
                if merged != last_ctx:
                    conn.execute(
                        "UPDATE activity SET duration_s = ?, context = ? WHERE id = ?",
                        (elapsed, merged, self._last_id),
                    )
                    self._last["context"] = merged
                else:
                    conn.execute(
                        "UPDATE activity SET duration_s = ? WHERE id = ?",
                        (elapsed, self._last_id),
                    )
        except Exception:
            logger.debug("Failed to bump duration", exc_info=True)

    def _flush_last(self, *, min_span_secs: int = 0) -> None:
        """Finalize the in-flight span.

        If ``min_span_secs`` is set and the span is shorter than that
        threshold, the row is deleted instead of being saved — this
        cleans up the "alt-tab flicker" rows that otherwise dominate
        the table without carrying any signal.
        """
        if self._last is None or self._last_id is None:
            return
        now = int(time.time())
        elapsed = max(0, now - int(self._last["started_at"]))
        try:
            with _open_db() as conn:
                if min_span_secs > 0 and elapsed < min_span_secs:
                    conn.execute("DELETE FROM activity WHERE id = ?", (self._last_id,))
                else:
                    conn.execute(
                        "UPDATE activity SET duration_s = ? WHERE id = ?",
                        (elapsed, self._last_id),
                    )
        except Exception:
            pass
        self._last = None
        self._last_id = None

    async def _maybe_cleanup(self, cfg: ActivityConfig) -> None:
        now = time.time()
        if now >= self._cleanup_due_at:
            # Run retention + size-cap cleanup at most once per hour.
            self._cleanup_due_at = now + 3600
            if cfg.retain_days > 0:
                cutoff = int(now) - cfg.retain_days * 86400
                try:
                    await asyncio.to_thread(self._prune_older_than, cutoff)
                except Exception:
                    logger.debug("Activity retention prune failed", exc_info=True)
            if cfg.max_db_mb > 0:
                max_bytes = cfg.max_db_mb * 1024 * 1024
                if db_size_bytes() > max_bytes:
                    try:
                        await asyncio.to_thread(self._prune_to_size, max_bytes)
                    except Exception:
                        logger.debug("Activity size-cap prune failed", exc_info=True)

        # Run incremental_vacuum once a day so the file actually
        # shrinks after retention pruning.  Cheap (only touches free
        # pages) so we can afford it even on a non-fresh DB.
        if now >= self._compact_due_at:
            self._compact_due_at = now + 86400
            try:
                await asyncio.to_thread(self._compact)
            except Exception:
                logger.debug("Activity DB compaction failed", exc_info=True)

    def _prune_older_than(self, cutoff_ts: int) -> int:
        with _open_db() as conn:
            cur = conn.execute("DELETE FROM activity WHERE ts < ?", (cutoff_ts,))
            return cur.rowcount or 0

    def _prune_to_size(self, max_bytes: int) -> int:
        """Delete oldest rows in batches until the DB file is ≤ max_bytes.

        Each iteration deletes the 200 oldest rows, then checks the file
        size again.  After all deletes a forced ``incremental_vacuum`` is
        run so the freed pages are returned to the OS (otherwise SQLite
        only reclaims them lazily).

        Returns the total number of rows deleted.
        """
        total_deleted = 0
        while db_size_bytes() > max_bytes:
            with _open_db() as conn:
                cur = conn.execute(
                    "DELETE FROM activity WHERE id IN "
                    "(SELECT id FROM activity ORDER BY ts ASC LIMIT 200)"
                )
                deleted = cur.rowcount or 0
            if deleted == 0:
                break
            total_deleted += deleted
            # Vacuum after each batch so the file actually shrinks and
            # the next size check reflects reality.
            try:
                with _open_db() as conn:
                    conn.execute("PRAGMA incremental_vacuum")
            except Exception:
                pass
        if total_deleted:
            logger.info(
                "Activity DB size cap: pruned %d rows (cap=%d MB)",
                total_deleted,
                max_bytes // (1024 * 1024),
            )
        return total_deleted

    def _compact(self) -> None:
        with _open_db() as conn:
            # Only does work if auto_vacuum=INCREMENTAL was active when
            # the schema was created; on legacy DBs this is a no-op.
            conn.execute("PRAGMA incremental_vacuum")
            # Keep the FTS index tight too — much cheaper than VACUUM.
            try:
                conn.execute("INSERT INTO activity_fts(activity_fts) VALUES ('optimize')")
            except sqlite3.OperationalError:
                pass


tracker = ActivityTracker()


# ---------------------------------------------------------------------------
# Read API (used by routes + agent tools)
# ---------------------------------------------------------------------------


def _row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    out = {
        "id": r["id"],
        "ts": r["ts"],
        "app": r["app"],
        "title": r["title"],
        "url": r["url"],
        "file_path": r["file_path"],
        "duration_s": r["duration_s"],
    }
    # ``context`` is a Tier-1 enrichment column; older rows may pre-date
    # the migration and have no value, so guard the read.
    try:
        out["context"] = r["context"] or ""
    except (IndexError, KeyError):
        out["context"] = ""
    return out


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a free-text string for use as an FTS5 MATCH expression.

    FTS5 treats a handful of characters as syntax (apostrophe/single-quote,
    double-quote, parentheses, ``^``, ``*``, ``-``).  When the query comes
    from a user prompt it can easily contain apostrophes ("owner's",
    "don't") that the FTS5 parser rejects with "syntax error near '\"'\"".

    Strategy: strip apostrophes (contractions survive as single tokens,
    e.g. "owners") and remove other FTS5 operator characters that would
    cause parse errors when the user didn't intend them as operators.
    Boolean keywords (AND/OR/NOT) are left intact so intentional operators
    still work.
    """
    q = query.strip()
    # Remove apostrophes/single-quotes — most common culprit
    q = q.replace("'", "").replace("\u2019", "").replace("\u2018", "")
    # Remove characters that have special meaning in FTS5 syntax
    for ch in ('"', "(", ")", "^", "*"):
        q = q.replace(ch, " ")
    # Collapse extra whitespace left by the replacements
    q = " ".join(q.split())
    return q


def _build_search_clause(
    query: str | None,
    date_from: int | None,
    date_to: int | None,
    app: str | None,
) -> tuple[str, str, list[Any]]:
    """Compose the JOIN/WHERE fragment shared between read paths.

    Returns ``(join_sql, where_sql, args)`` where ``where_sql`` already
    includes the leading ``" WHERE "`` if any filters were specified
    (otherwise it's empty).
    """
    join_sql = ""
    args: list[Any] = []
    where: list[str] = []
    if query and query.strip():
        sanitized = _sanitize_fts_query(query)
        if sanitized:
            join_sql = " JOIN activity_fts f ON f.rowid = a.id"
            where.append("activity_fts MATCH ?")
            args.append(sanitized)
    if date_from is not None:
        where.append("a.ts >= ?")
        args.append(int(date_from))
    if date_to is not None:
        where.append("a.ts <= ?")
        args.append(int(date_to))
    if app:
        where.append("LOWER(a.app) = LOWER(?)")
        args.append(app)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    return join_sql, where_sql, args


def search_activity(
    query: str | None = None,
    *,
    date_from: int | None = None,
    date_to: int | None = None,
    app: str | None = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "ts",
) -> list[dict[str, Any]]:
    """Search the activity log.

    Args:
        query: FTS5 MATCH expression (free-text terms).  ``None`` returns
               all rows in the date range, sorted by recency.
        date_from / date_to: Unix timestamps bounding the search.
        app: Restrict to a specific app name (exact, case-insensitive).
        limit: Max rows to return.
        offset: Number of rows to skip — used with ``limit`` for paging.
        order_by: ``"ts"`` (chronological, newest first) or ``"rank"``
                  (FTS5 BM25 relevance — only valid when ``query`` is
                  also set; falls back to ``ts`` otherwise).
    """
    join_sql, where_sql, args = _build_search_clause(query, date_from, date_to, app)
    use_rank = order_by == "rank" and bool(query and query.strip())
    order_sql = " ORDER BY rank" if use_rank else " ORDER BY a.ts DESC"
    sql = f"SELECT a.* FROM activity a{join_sql}{where_sql}{order_sql} LIMIT ? OFFSET ?"
    args = args + [int(limit), max(0, int(offset))]
    with _open_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]


def count_activity(
    query: str | None = None,
    *,
    date_from: int | None = None,
    date_to: int | None = None,
    app: str | None = None,
) -> int:
    """Return the total number of rows matching ``search_activity`` filters.

    Cheap on FTS5 — runs the same WHERE/JOIN as the search itself but
    selects a count.  Used by the UI to render "X of N" counters and
    to decide whether to show a "Load more" button.
    """
    join_sql, where_sql, args = _build_search_clause(query, date_from, date_to, app)
    sql = f"SELECT COUNT(*) FROM activity a{join_sql}{where_sql}"
    with _open_db() as conn:
        cur = conn.execute(sql, args)
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0


def list_apps(*, days: int = 30) -> list[dict[str, Any]]:
    """Return a histogram of apps used over the last N days."""
    cutoff = int(time.time()) - days * 86400
    with _open_db() as conn:
        rows = conn.execute(
            "SELECT app, COUNT(*) AS samples, SUM(duration_s) AS seconds "
            "FROM activity WHERE ts >= ? GROUP BY app ORDER BY seconds DESC",
            (cutoff,),
        ).fetchall()
        return [
            {"app": r[0], "samples": int(r[1] or 0), "seconds": int(r[2] or 0)}
            for r in rows
        ]


def daily_summary(date_from: int, date_to: int) -> dict[str, Any]:
    """Return per-app totals and total tracked seconds within a range."""
    with _open_db() as conn:
        rows = conn.execute(
            "SELECT app, SUM(duration_s) AS seconds, COUNT(*) AS samples "
            "FROM activity WHERE ts BETWEEN ? AND ? GROUP BY app ORDER BY seconds DESC",
            (int(date_from), int(date_to)),
        ).fetchall()
        total = sum(int(r[1] or 0) for r in rows)
        return {
            "from": int(date_from),
            "to": int(date_to),
            "total_seconds": total,
            "apps": [
                {"app": r[0], "seconds": int(r[1] or 0), "samples": int(r[2] or 0)}
                for r in rows
            ],
        }


def clear_all() -> int:
    """Delete every row.  Returns the count removed."""
    with _open_db() as conn:
        cur = conn.execute("DELETE FROM activity")
        return cur.rowcount or 0


def db_size_bytes() -> int:
    try:
        return _db_path().stat().st_size
    except OSError:
        return 0
