#!/usr/bin/env bash
# One-time dev environment installer for Otto — macOS and Linux.
#
# Installs the full toolchain (uv, Node, graphviz), the Python venv + deps,
# the Playwright browser, and builds the PyInstaller backend binary used by
# the packaged Tauri app. For day-to-day launching use ./start_app.sh,
# which shares the same toolchain bootstrap but skips the heavy release
# steps below.
#
# Windows is not natively supported; use WSL and run this from there, or
# follow the manual installation steps in README.md.
#
# Nothing here requires Homebrew (or any other system package manager):
# uv, Node, and graphviz all have brew-free fallbacks so this works on a
# clean machine / VM with no prior setup.

set -euo pipefail

OTTO_LOG_TAG="install"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/bootstrap.sh
source "$SCRIPT_DIR/scripts/lib/bootstrap.sh"
cd "$OTTO_ROOT"

UNAME_S="$(uname -s)"
case "$UNAME_S" in
    MINGW*|MSYS*|CYGWIN*)
        die "native Windows is not supported. Use WSL and re-run this script there, or see README.md." ;;
    Darwin|Linux) ;;
    *) die "Unsupported OS ($UNAME_S). See README.md for manual installation steps." ;;
esac

info "Installing Otto dev environment ($UNAME_S)…"

# ── Core toolchain (shared with start_app.sh via scripts/lib/bootstrap.sh) ──
ensure_uv
success "uv $(uv --version) found."

ensure_graphviz_optional \
    || info "Skipping graphviz (optional; no supported package manager found, or install failed)."

ensure_node
success "Node.js $(node -v) found."

ensure_venv
sync_python_deps
ensure_env_file

# ── Optional: pygraphviz for LangGraph draw_mermaid_png ───────────────
if uv pip install pygraphviz 2>/dev/null; then
    success "pygraphviz installed (LangGraph PNG output enabled)"
else
    info "Skipping pygraphviz (optional; install graphviz then 'uv pip install pygraphviz' to enable)"
fi

# ── Playwright MCP + browser (install-only; unused by the dev launcher) ──
info "Caching Playwright MCP server…"
npx -y @playwright/mcp@latest --help >/dev/null 2>&1 || true
info "Installing Playwright browser (Chromium)…"
npx -y @playwright/test install chromium

# ── Build backend binary for the packaged Tauri app (install-only) ────
# `uv run` resolves the project's own .venv interpreter regardless of PATH,
# so this works even where only `python3` exists (e.g. Debian/Ubuntu).
info "Building backend binary for Tauri app…"
uv pip show pyinstaller >/dev/null 2>&1 || uv pip install pyinstaller
uv run python scripts/build_backend.py

echo ""
echo "================================================"
echo "  Installation complete!"
echo "================================================"
echo ""
echo "Start everything (backend + Tauri app) with:"
echo "  ./start_app.sh"
echo ""
echo "Or run the pieces manually — activate the venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Start the backend (port 18081):"
echo "  PYTHONPATH=src python -m backend --port 18081 --reload"
echo ""
echo "Start the Tauri desktop app (in a separate terminal):"
echo "  cd app && npm install && npm run tauri dev"
echo ""
echo "Run tests:"
echo "  pytest"
