#!/usr/bin/env bash
# start_web.sh — Install deps and run OTTO from source in the browser.
#
# The Rust-free, Homebrew-free counterpart to ./start_app.sh: it launches the
# FastAPI backend plus the Vite dev server (browser UI at http://localhost:5173)
# instead of the native Tauri desktop app, so neither Rust nor the Xcode
# Command Line Tools are needed. Ideal for VMs, headless machines, or a quick
# dev loop.
#
# The install path never uses Homebrew (or any other system package manager):
# uv installs via its official script and Node installs from the official
# prebuilt tarball into ~/.local — both sudo-free. No .env file is created or
# required; configure your model and API keys from the in-app Settings.
#
# Usage:
#   ./start_web.sh              — ensure deps (fast/idempotent), start services
#   ./start_web.sh --no-install — skip dependency install, start services only

set -euo pipefail

OTTO_LOG_TAG="web"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/bootstrap.sh
source "$SCRIPT_DIR/scripts/lib/bootstrap.sh"

BACKEND_PORT="18081"
FRONTEND_PORT="5173"

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_INSTALL=false
for arg in "$@"; do
    [[ "$arg" == "--no-install" ]] && SKIP_INSTALL=true
done

# ── Node.js (Homebrew-free) ───────────────────────────────────────────────────
# Unlike bootstrap.sh's ensure_node, this deliberately never shells out to a
# system package manager. It reuses an existing working Node >= 18 if present,
# otherwise falls back to the official prebuilt tarball (install_node_portable).
ensure_node_no_brew() {
    if command -v node &>/dev/null; then
        # A broken Homebrew dylib link surfaces as a "Library not loaded" error
        # from `node -v`; treat that (or a failed invocation) as "reinstall".
        local node_ver
        if node_ver=$(node -v 2>/dev/null); then
            local node_major
            node_major=$(echo "$node_ver" | sed 's/v\([0-9]*\).*/\1/')
            if (( node_major >= 18 )); then
                return
            fi
            warn "Node.js is too old ($node_ver < 18) — installing a portable copy…"
        else
            warn "Node.js is present but not runnable — installing a portable copy…"
        fi
    else
        info "Node.js not found. Installing a portable copy (no package manager)…"
    fi

    install_node_portable

    local node_major
    node_major=$(node -v | sed 's/v\([0-9]*\).*/\1/')
    (( node_major >= 18 )) || die "Node.js 18+ required (found $(node -v))."
    command -v npm &>/dev/null || die "npm not found after Node install."
}

# ── Core toolchain (Homebrew-free) ────────────────────────────────────────────
ensure_uv          # official astral.sh installer; no package manager
ensure_node_no_brew

# ── Port availability checks ─────────────────────────────────────────────────
check_port() {
    local port=$1 label=$2
    if lsof -ti :"$port" &>/dev/null; then
        local owner
        owner=$(lsof -ti :"$port" | head -1 | xargs ps -o comm= -p 2>/dev/null || echo "unknown")
        die "Port $port ($label) is already in use by '$owner'. Stop it first: lsof -ti :$port | xargs kill"
    fi
}

check_port "$BACKEND_PORT"  "backend"
check_port "$FRONTEND_PORT" "Vite (frontend)"

# ── Dependencies ──────────────────────────────────────────────────────────────
# No ensure_env_file: the browser workflow needs no .env. Configure your model
# and API keys from the in-app Settings (stored in the OS keychain).
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
# kill $PID only kills the immediate process; npm spawns node → vite, so we
# walk the tree recursively to avoid leaving orphaned dev servers behind.
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

# ── Launch frontend (Vite dev server — browser, no Rust) ──────────────────────
info "Starting Vite dev server on port $FRONTEND_PORT..."
(cd "$APP_DIR" && npm run dev) &
FRONTEND_PID=$!
success "Frontend PID $FRONTEND_PID"

# ── Startup health check ──────────────────────────────────────────────────────
# Give both processes a moment to start, then verify they are still alive.
# A fast crash (e.g. port already in use) would otherwise look like a clean exit.
sleep 2
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    die "Backend exited immediately. Check the output above for errors (port in use?)."
fi
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    die "Frontend exited immediately. Check the output above for errors."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${_C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_C_RESET}"
echo -e "  ${_C_GREEN}Backend${_C_RESET}   → http://localhost:$BACKEND_PORT"
echo -e "  ${_C_GREEN}Frontend${_C_RESET}  → http://localhost:$FRONTEND_PORT  (open in your browser)"
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
