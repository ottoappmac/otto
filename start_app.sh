#!/usr/bin/env bash
# start_app.sh — Bootstrap the venv and launch backend + frontend in one command.
#
# Usage:
#   ./start_app.sh             — ensure deps (fast/idempotent), start services
#   ./start_app.sh --no-install  — skip dependency install, start services only
#
# Shares its toolchain bootstrap (uv/Node/venv/deps) with install.sh via
# scripts/lib/bootstrap.sh, but deliberately skips install.sh's heavy,
# one-time release steps (Playwright browser download, PyInstaller backend
# build) — the dev flow runs the Python backend directly and never uses the
# packaged bundle.

set -euo pipefail

OTTO_LOG_TAG="start"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/bootstrap.sh
source "$SCRIPT_DIR/scripts/lib/bootstrap.sh"

BACKEND_PORT="18081"

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_INSTALL=false
for arg in "$@"; do
    [[ "$arg" == "--no-install" ]] && SKIP_INSTALL=true
done

# ── Core toolchain (shared with install.sh via scripts/lib/bootstrap.sh) ──────
# ensure_uv / ensure_node install-if-missing and self-heal (incl. the Node
# dylib check) and validate Node >= 18, so no separate prerequisite gate is
# needed here.
ensure_uv
ensure_node

# ── .env bootstrap ────────────────────────────────────────────────────────────
ensure_env_file

# ── Port availability checks ─────────────────────────────────────────────────
check_port() {
    local port=$1 label=$2
    if lsof -ti :"$port" &>/dev/null; then
        local owner
        owner=$(lsof -ti :"$port" | head -1 | xargs ps -o comm= -p 2>/dev/null || echo "unknown")
        die "Port $port ($label) is already in use by '$owner'. Stop it first: lsof -ti :$port | xargs kill"
    fi
}

check_port "$BACKEND_PORT" "backend"
check_port 5173            "Vite (frontend)"

# ── Dependencies ──────────────────────────────────────────────────────────────
ensure_venv
if [[ "$SKIP_INSTALL" == false ]]; then
    sync_python_deps --system-certs --quiet

    info "Installing frontend dependencies (npm install)..."
    npm install --prefix "$APP_DIR" --silent
    success "Frontend dependencies installed."
else
    info "--no-install: skipping dependency installation."
fi

# ── Process-tree teardown ─────────────────────────────────────────────────────
# kill $PID only kills the immediate process. For frontend, npm spawns node →
# cargo → the Tauri binary. We walk the tree recursively so no orphans remain.
kill_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null) || true
    for child in $children; do
        kill_tree "$child"
    done
    kill "$pid" 2>/dev/null || true
}

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    info "Shutting down..."
    [[ -n "$BACKEND_PID" ]]  && kill_tree "$BACKEND_PID"
    [[ -n "$FRONTEND_PID" ]] && kill_tree "$FRONTEND_PID"
    wait "$BACKEND_PID"  2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
    success "All services stopped."
}
trap cleanup EXIT INT TERM

# ── Launch backend ────────────────────────────────────────────────────────────
info "Starting backend on port $BACKEND_PORT..."
VIRTUAL_ENV="$VENV_DIR" \
PATH="$VENV_DIR/bin:$PATH" \
PYTHONPATH="$OTTO_ROOT/src" \
    "$VENV_DIR/bin/python" -m backend --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!
success "Backend PID $BACKEND_PID"

# ── Launch frontend ───────────────────────────────────────────────────────────
# Tauri's build script validates that every `bundle.resources` source path
# exists — even for `tauri dev`. In dev we run the Python backend directly
# (above) and `spawn_backend` in lib.rs is gated to release builds, so the
# PyInstaller bundle at resources/backend/ is never produced. Ensure the
# directory exists so the dev build doesn't abort with
# "resource path `resources/backend` doesn't exist".
mkdir -p "$APP_DIR/src-tauri/resources/backend"

info "Starting Tauri frontend..."
(cd "$APP_DIR" && npm run tauri dev) &
FRONTEND_PID=$!
success "Frontend PID $FRONTEND_PID"

# ── Startup health check ──────────────────────────────────────────────────────
# Give both processes a moment to start, then verify they are still alive.
# A fast crash (e.g. port already in use) would otherwise look like a clean exit.
sleep 2
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    die "Backend exited immediately. Check the output above for errors (port in use? missing .env key?)."
fi
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    die "Frontend exited immediately. Check the output above for errors."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${_C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_C_RESET}"
echo -e "  ${_C_GREEN}Backend${_C_RESET}   → http://localhost:$BACKEND_PORT"
echo -e "  ${_C_GREEN}Frontend${_C_RESET}  → Tauri desktop app (Vite dev server)"
echo -e "  Press ${_C_BOLD}Ctrl-C${_C_RESET} to stop both services."
echo -e "${_C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_C_RESET}"
echo ""

# ── Stay alive until either service dies ─────────────────────────────────────
# `wait -n` is unreliable here: uvicorn --reload forks a child and the original
# PID exits immediately, which would trigger cleanup even though the service is
# still running. Poll with kill -0 instead — it checks liveness without signalling.
while true; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        warn "Backend process ($BACKEND_PID) has exited."
        break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        warn "Frontend process ($FRONTEND_PID) has exited."
        break
    fi
    sleep 3
done
