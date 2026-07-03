#!/usr/bin/env bash
# Shared bootstrap helpers for install.sh and start_app.sh.
#
# This file is meant to be *sourced*, not executed: it defines constants
# and functions and performs no work on its own. Keeping the toolchain
# bootstrap in one place means the brew-free fallbacks and version pins
# live in exactly one spot and can't drift between the two scripts.
#
# Nothing here requires Homebrew (or any other system package manager):
# uv, Node, and graphviz all fall back to official, sudo-free installers.

# ── Paths (resolved relative to this file, independent of caller CWD) ──
OTTO_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OTTO_ROOT="$(cd "$OTTO_LIB_DIR/../.." && pwd)"
VENV_DIR="$OTTO_ROOT/.venv"
APP_DIR="$OTTO_ROOT/app"

# ── Version pins ──────────────────────────────────────────────────────
PYTHON_VERSION="3.12"
# Pinned to match backend.exo_cli._NODE_VERSION / backend.node_provisioner.NODE_VERSION.
NODE_VERSION="v22.14.0"

# ── Logging ───────────────────────────────────────────────────────────
# Colours are disabled automatically when stdout is not a terminal (piped
# to a file/CI) so logs stay clean. The tag is read live from
# OTTO_LOG_TAG so each caller can label its output (defaults to "otto").
if [[ -t 1 ]]; then
    _C_RED=$'\033[0;31m'; _C_GREEN=$'\033[0;32m'; _C_YELLOW=$'\033[1;33m'
    _C_CYAN=$'\033[0;36m'; _C_BOLD=$'\033[1m'; _C_RESET=$'\033[0m'
else
    _C_RED=''; _C_GREEN=''; _C_YELLOW=''; _C_CYAN=''; _C_BOLD=''; _C_RESET=''
fi

info()    { echo -e "${_C_CYAN}${_C_BOLD}[${OTTO_LOG_TAG:-otto}]${_C_RESET} $*"; }
success() { echo -e "${_C_GREEN}${_C_BOLD}[${OTTO_LOG_TAG:-otto}]${_C_RESET} $*"; }
warn()    { echo -e "${_C_YELLOW}${_C_BOLD}[${OTTO_LOG_TAG:-otto}]${_C_RESET} $*"; }
die()     { echo -e "${_C_RED}${_C_BOLD}[${OTTO_LOG_TAG:-otto}]${_C_RESET} $*" >&2; exit 1; }

# ── uv ────────────────────────────────────────────────────────────────
ensure_uv() {
    if command -v uv &>/dev/null; then
        return
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        return
    fi
    info "uv not found. Installing…"
    curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || die "uv install failed. Install manually: https://astral.sh/uv"
}

# ── Node ──────────────────────────────────────────────────────────────
# Download + install Node's official prebuilt tarball into ~/.local — no
# package manager, no sudo. Mirrors backend/exo_cli.py's brew-free path.
install_node_portable() {
    info "Installing Node $NODE_VERSION from the official prebuilt tarball (no package manager needed)…"
    local os_tag arch_tag name node_root local_bin tool
    case "$(uname -s)" in
        Darwin) os_tag="darwin" ;;
        Linux)  os_tag="linux" ;;
        *) die "No portable Node build for $(uname -s)." ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) arch_tag="arm64" ;;
        *)             arch_tag="x64" ;;
    esac
    name="node-${NODE_VERSION}-${os_tag}-${arch_tag}"
    node_root="$HOME/.local/node"
    local_bin="$HOME/.local/bin"

    mkdir -p "$local_bin"
    rm -rf "$node_root"
    mkdir -p "$node_root"
    curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/${name}.tar.gz" \
        | tar xz --strip-components 1 -C "$node_root"

    for tool in node npm npx; do
        ln -sf "$node_root/bin/$tool" "$local_bin/$tool"
    done
    export PATH="$local_bin:$PATH"
}

# Homebrew can upgrade a shared library (e.g. llhttp) independently of
# Node, leaving Node's binary pointing at a dylib that no longer exists.
# Detect that ("Library not loaded") and self-heal before anything runs
# Node — brew reinstall first, then a portable install as a last resort.
heal_node_dylibs() {
    command -v node &>/dev/null || return 0
    local node_err
    node_err=$(node --version 2>&1 >/dev/null) || true
    echo "$node_err" | grep -q "Library not loaded" || return 0

    warn "Node.js has a broken dylib link (Homebrew dependency mismatch)."
    if command -v brew &>/dev/null; then
        warn "Running: brew reinstall node — this takes ~30 s …"
        brew reinstall node >/dev/null 2>&1 \
            && success "Node.js reinstalled successfully." \
            || warn "brew reinstall node failed; falling back to a portable install."
    fi
    node_err=$(node --version 2>&1 >/dev/null) || true
    if echo "$node_err" | grep -q "Library not loaded"; then
        install_node_portable
    fi
}

# Ensure a working Node.js >= 18 with npm is on PATH. Prefers a system
# package manager when present but never requires one — every branch
# degrades to the portable tarball on failure.
ensure_node() {
    if ! command -v node &>/dev/null; then
        info "Node.js not found. Installing…"
        if command -v brew &>/dev/null; then
            brew install node || install_node_portable
        elif command -v apt-get &>/dev/null; then
            { sudo apt-get update -y && sudo apt-get install -y nodejs npm; } || install_node_portable
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y nodejs npm || install_node_portable
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm nodejs npm || install_node_portable
        else
            install_node_portable
        fi
    fi

    heal_node_dylibs

    local node_major
    node_major=$(node -v | sed 's/v\([0-9]*\).*/\1/')
    if (( node_major < 18 )); then
        warn "Node.js is too old ($(node -v) < 18) — installing a portable copy…"
        install_node_portable
        node_major=$(node -v | sed 's/v\([0-9]*\).*/\1/')
        (( node_major >= 18 )) || die "Node.js 18+ required (found $(node -v))."
    fi
    command -v npm &>/dev/null || die "npm not found after Node install."
}

# ── graphviz (optional — LangGraph PNG output only) ───────────────────
# Returns non-zero (rather than dying) when no package manager is present
# or the install fails, so callers can treat it as best-effort.
ensure_graphviz_optional() {
    if command -v brew &>/dev/null; then
        brew list graphviz &>/dev/null 2>&1 && return 0
        info "Installing graphviz via Homebrew…"
        brew install graphviz && return 0
    elif command -v apt-get &>/dev/null; then
        dpkg -s graphviz &>/dev/null 2>&1 && return 0
        info "Installing graphviz via apt…"
        sudo apt-get update -y && sudo apt-get install -y graphviz && return 0
    elif command -v dnf &>/dev/null; then
        rpm -q graphviz &>/dev/null 2>&1 && return 0
        info "Installing graphviz via dnf…"
        sudo dnf install -y graphviz && return 0
    elif command -v pacman &>/dev/null; then
        pacman -Qi graphviz &>/dev/null 2>&1 && return 0
        info "Installing graphviz via pacman…"
        sudo pacman -S --noconfirm graphviz && return 0
    fi
    return 1
}

# ── Python venv + deps ────────────────────────────────────────────────
ensure_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating Python $PYTHON_VERSION virtual environment…"
        (cd "$OTTO_ROOT" && uv venv "$VENV_DIR" --python "$PYTHON_VERSION")
        success "Virtual environment created at $VENV_DIR"
    fi
}

# `uv sync --frozen` (not `uv pip install -e .`) installs deterministically
# against uv.lock, which pins numba 0.65.1 / llvmlite 0.47.0. Without the
# lock, uv's fresh resolver can pick numba 0.53.1 → llvmlite 0.36.0, which
# only supports Python <3.10 and fails to build from source on 3.11+. This
# also installs the `dev` dependency group. Extra args are forwarded
# (e.g. --system-certs, --quiet).
sync_python_deps() {
    info "Installing Python dependencies (uv sync --frozen)…"
    (cd "$OTTO_ROOT" && uv sync --frozen "$@")
    success "Python dependencies installed."
}

# ── .env ──────────────────────────────────────────────────────────────
ensure_env_file() {
    if [[ -f "$OTTO_ROOT/.env" ]]; then
        return
    fi
    if [[ -f "$OTTO_ROOT/.env.template" ]]; then
        warn ".env not found — copying from .env.template. Edit it to add your API keys."
        cp "$OTTO_ROOT/.env.template" "$OTTO_ROOT/.env"
    else
        warn ".env not found and no .env.template available. Backend may fail to start."
    fi
}
