"""Serve, and record, the raw.githubusercontent.com files the SWE-bench scorer reads.

While preparing each instance, the scorer fetches its requirements files from GitHub, even
with prebuilt images, where they only feed an environment script that is never run. That
breaks offline runs. This wraps ``requests.get`` for those URLs:

  online   fetch as normal and keep the answer in a cache folder
  offline  answer from the cache; for a URL never seen, answer an empty 200, which is
           harmless because the prebuilt image already contains the environment

``bench.sh warm`` fills the cache for the whole dataset. Folder: $SWEBENCH_HTTP_CACHE.
Offline is switched on with SWEBENCH_OFFLINE=1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

PREFIX = "https://raw.githubusercontent.com/"


def cache_dir() -> Path:
    return Path(os.environ.get("SWEBENCH_HTTP_CACHE") or Path.home() / ".cache" / "swebench-http")


def offline() -> bool:
    return os.environ.get("SWEBENCH_OFFLINE") == "1"


def _entry(url: str) -> Path:
    return cache_dir() / (hashlib.sha1(url.encode()).hexdigest() + ".json")


def _response(status: int, body: str, url: str) -> requests.Response:
    r = requests.models.Response()
    r.status_code, r._content, r.url, r.encoding = status, body.encode(), url, "utf-8"
    return r


def install() -> None:
    """Wrap requests.get once. Safe to call repeatedly."""
    real = requests.get
    if getattr(real, "_swebench_cached", False):
        return

    def get(url, *args, **kwargs):
        if not (isinstance(url, str) and url.startswith(PREFIX)):
            return real(url, *args, **kwargs)
        entry = _entry(url)
        if entry.exists():
            d = json.loads(entry.read_text())
            return _response(d["status"], d["body"], url)
        if offline():
            return _response(200, "", url)
        resp = real(url, *args, **kwargs)
        if resp.status_code in (200, 404):          # keep answers, not transient errors
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(json.dumps({"url": url, "status": resp.status_code, "body": resp.text}))
        return resp

    get._swebench_cached = True
    requests.get = get
