#!/usr/bin/env bash
# Build the otto-audiotap system-audio capture helper (macOS 14.4+).
# Produces app/src-tauri/audiotap/.build/release/otto-audiotap, which
# tauri.conf.json bundles into the app as resources/audiotap/otto-audiotap.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/audiotap"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "otto-audiotap is macOS-only; skipping build on $(uname)." >&2
  exit 0
fi

echo "Building otto-audiotap (release)..." >&2
swift build -c release

BIN="$HERE/audiotap/.build/release/otto-audiotap"
if [[ ! -x "$BIN" ]]; then
  echo "Build did not produce $BIN" >&2
  exit 1
fi
echo "Built $BIN" >&2
