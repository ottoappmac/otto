#!/bin/bash
# Build Otto for macOS.
#
# Usage:
#   ./scripts/build_mac.sh          # full build (backend + frontend + Tauri)
#   ./scripts/build_mac.sh --skip-backend   # skip PyInstaller step (reuse existing binary)
#
# Output:
#   app/src-tauri/target/release/bundle/dmg/   — installable .dmg
#   app/src-tauri/target/release/bundle/macos/  — standalone .app

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKIP_BACKEND=false
for arg in "$@"; do
  case "$arg" in
    --skip-backend) SKIP_BACKEND=true ;;
  esac
done

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m==> %s\033[0m\n' "$*"; exit 1; }

# ---------- Preflight checks ------------------------------------------------

info "Checking prerequisites..."

command -v python3 &>/dev/null || fail "Python 3 not found"
command -v node    &>/dev/null || fail "Node.js not found — install from https://nodejs.org"
command -v npm     &>/dev/null || fail "npm not found"
command -v cargo   &>/dev/null || fail "Rust not found — run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

if [ ! -d ".venv" ]; then
  fail "Virtual environment not found — run ./scripts/install.sh first"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

ok "Prerequisites OK (Python $(python3 --version | cut -d' ' -f2), Node $(node --version), Rust $(rustc --version | cut -d' ' -f2))"

# ---------- 1. Build Python backend binary -----------------------------------

if [ "$SKIP_BACKEND" = true ]; then
  BINARY="app/src-tauri/resources/backend/otto-backend"
  if [ ! -f "$BINARY" ]; then
    fail "No backend binary at $BINARY — remove --skip-backend flag"
  fi
  warn "Skipping backend build (using existing binary)"
else
  info "Installing PyInstaller..."
  uv pip install pyinstaller 2>/dev/null || pip install pyinstaller

  info "Building Python backend binary (this takes a few minutes)..."
  python scripts/build_backend.py
  ok "Backend binary built"
fi

# ---------- 2. Install frontend dependencies ---------------------------------

info "Installing frontend dependencies..."
(cd app && npm install)
ok "Frontend dependencies installed"

# ---------- 3. Build Tauri app -----------------------------------------------

info "Building Tauri app (frontend + Rust shell + bundling)..."
(cd app && npm run tauri build)

ok "Build complete!"
echo ""

# ---------- Rename artifacts with platform prefix ----------------------------

DMG_DIR="app/src-tauri/target/release/bundle/dmg"
APP_DIR="app/src-tauri/target/release/bundle/macos"

info "Prefixing artifact names with MacOS_..."
if [ -d "$DMG_DIR" ]; then
  for f in "$DMG_DIR"/*.dmg; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    case "$base" in MacOS_*) continue ;; esac
    mv "$f" "$DMG_DIR/MacOS_$base"
  done
fi

# ---------- Summary ----------------------------------------------------------

echo ""
info "Output:"
if [ -d "$DMG_DIR" ]; then
  echo "   DMG:  $(ls "$DMG_DIR"/*.dmg 2>/dev/null || echo 'not found')"
fi
if [ -d "$APP_DIR" ]; then
  echo "   App:  ${APP_DIR}/Otto.app"
fi
echo ""
ok "Done! You can open the .dmg to install or run the .app directly."
