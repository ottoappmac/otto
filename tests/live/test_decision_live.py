"""One live call against a decision sidecar the user has already started.

Skipped when nothing is listening, so a machine without the model stays green.
"""

from __future__ import annotations

import socket

import pytest

from backend.config import AppConfig
from backend.decision_client import ask


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


@pytest.mark.live
def test_local_sidecar_answers_yes_no() -> None:
    cfg = AppConfig.load().decision
    port = int(cfg.port)
    if not _listening(port):
        pytest.skip(f"decision sidecar is not listening on {port}")
    from backend.decision_runtime import local_api_key

    result = ask(
        "The build log ends with FAILED tests/test_api.py",
        "Did the build fail?",
        base_url=f"http://127.0.0.1:{port}",
        api_key=local_api_key(),
    )
    assert result.answer in {"yes", "no"}
    assert 0.0 <= result.confidence <= 1.0
