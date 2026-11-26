"""LinkedIn built-in MCP: a thin launcher for the pinned ``mcp-server-linkedin`` package.

The tools themselves come from the third-party, Apache-2.0
``mcp-server-linkedin`` package (stickerdaniel/linkedin-mcp-server), pinned
and hash-locked in ``requirements.txt``. It drives LinkedIn through a
Patchright Chromium window signed in as the user. This file only fixes the
settings Otto needs before handing over to the package's own ``main()``:

* **Everything stays in this folder.** ``HOME`` points at ``./home`` and the
  browser profile lives in ``./state/profile``, so the session cookies,
  the downloaded Chromium and the package's caches are all under
  ``<app_data>/mcp_server/linkedin/``. That is also the only place the
  built-in Seatbelt profile lets this process write.
* **No calls to the package author's servers.** ``LINKEDIN_MCP_REMOTE_TOOLS=off``
  stops the remote tool catalog (extra tools fetched from, and run on,
  mcp.daniel.sticker.name). ``LINKEDIN_MCP_CHECK_FOR_UPDATES=off`` stops the
  daily PyPI version check. Both are forced, not defaulted.
* **No silent cookie import.** ``--no-auto-import`` keeps the package from
  reading the LinkedIn cookies of the user's everyday browser. Sign-in
  happens in the package's own window on the first tool call that needs it.
* **One browser per Otto connection.** ``--no-daemon`` keeps the browser a
  child of this process, so stopping the MCP in Otto closes it.

Remove the session with Tools -> LinkedIn -> Stop, then delete
``<app_data>/mcp_server/linkedin/state``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The package deletes the profile's parent directory when it rotates or clears
# a session, so give the profile a parent that holds nothing else.
STATE_DIR = HERE / "state"
PROFILE_DIR = STATE_DIR / "profile"
HOME_DIR = HERE / "home"

FORCED_ENV = {
    "LINKEDIN_MCP_REMOTE_TOOLS": "off",
    "LINKEDIN_MCP_CHECK_FOR_UPDATES": "off",
}

FIXED_ARGS = (
    "--no-auto-import",
    "--no-daemon",
    "--user-data-dir",
    str(PROFILE_DIR),
)


def prepare(environ: dict[str, str] | os._Environ = os.environ) -> list[str]:
    """Set the forced environment and return the package's argv."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    environ["HOME"] = str(HOME_DIR)
    environ.update(FORCED_ENV)
    return ["mcp-server-linkedin", *FIXED_ARGS]


def main() -> None:
    sys.argv = prepare()
    from linkedin_mcp_server.__main__ import main as package_main

    package_main()


if __name__ == "__main__":
    main()
