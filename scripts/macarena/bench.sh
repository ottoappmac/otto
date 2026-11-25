#!/usr/bin/env bash
# MacArena runs with Otto: macarena-agent (standard track) or Otto's macOS desktop agent (--native).
#
#   scripts/macarena/bench.sh setup [--vm macarena|osworld|both|none] [--yes] [--dry-run]
#                                                    install UTM, clone + pin MacArena, build its Python env,
#                                                    download the VM image(s). Asks before each download.
#   scripts/macarena/bench.sh check                  prerequisites
#   scripts/macarena/bench.sh grant-perms            ONE-TIME, after setup: boots the base VM and walks you through
#                                                    giving SSH-launched programs Accessibility + Screen Recording
#   scripts/macarena/bench.sh grant-apps [--vm macarena|osworld] [App ...]
#                                                    ONE-TIME: lets the tasks' osascript setup and grading talk to
#                                                    Contacts, Reminders, Calendar, Notes (or the apps you name);
#                                                    `run` checks this first and skips tasks that cannot work
#   scripts/macarena/bench.sh verify-image [--vm macarena|osworld] [--apps "App ..."]
#                                                    boots a throwaway COPY of the base image and checks every
#                                                    permission and app the tasks need; prints PASS/FAIL per line,
#                                                    changes nothing, deletes the copy. Not while a run is going.
#   scripts/macarena/bench.sh up   [--setup] [--vm osworld|macarena] [--a11y] [--native] [--memory off|live|<serve.db>]
#                                                    start the bridge (owns the VM) and Otto's backend;
#                                                    --setup runs `setup` first for whatever is missing;
#                                                    --native: Otto-native track (Otto's macOS desktop agent and
#                                                    its tools, run in the VM) instead of mouse/keyboard;
#                                                    --memory: procedural memory snapshot to serve read-only, or
#                                                      live: each task's run is recallable by the later tasks of
#                                                      the same run (off: no memory tool; the default)
#                                                    (scripts/procmem/build_snapshot.py; native track; default off)
#   scripts/macarena/bench.sh run  [run.py args]     run tasks and print the summary (needs `up` first)
#   scripts/macarena/bench.sh report <run folder>    summarise a finished run
#   scripts/macarena/bench.sh stop                   stop the bridge and the backend
#   scripts/macarena/bench.sh slots                  list the slots and whether each is up
#
# Two VMs at once: MACARENA_SLOT=2 runs a second bridge + backend pair beside slot 1 (the default), with its own
# ports, logs, desktop lock and benchmark home. macOS runs at most two macOS VMs at a time, so there are two slots.
# Give each slot half of the tasks with `run --shard 1/2` and `--shard 2/2` and its own --out; report both
# folders together with `report <folder> <folder>`. Both agents share one model server, so each is slower.
# MACARENA_DIR points at a MacArena checkout (default ../MacArena), pinned to MACARENA_COMMIT.
# MACARENA_PYTHON is MacArena's Python (default: <MacArena>/.venv/bin/python, which `setup` builds with uv).
# MACARENA_RECORD=1 saves a screenshot after every step and recording.mp4 (off by default; review only).
# MACARENA_SETUP_MODE=launch starts each task's setup without waiting (MacArena's way); default sync waits and checks it.
# MACARENA_FAKE=1 starts the bridge with no VM (blank screen, score 0): a wiring smoke test only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MA="${MACARENA_DIR:-$(dirname "$ROOT")/MacArena}"
RUNS="${MACARENA_RUNS:-$(dirname "$ROOT")/otto-macarena-runs}"
if [[ -n "${MACARENA_PYTHON:-}" ]]; then PY_MA="$MACARENA_PYTHON"
elif [[ -x "$MA/.venv/bin/python" ]]; then PY_MA="$MA/.venv/bin/python"
else PY_MA="python3"; fi
SLOT="${MACARENA_SLOT:-1}"
MAX_SLOTS=2                                # macOS's limit on macOS VMs running at once
[[ "$SLOT" =~ ^[0-9]+$ ]] && (( SLOT >= 1 && SLOT <= MAX_SLOTS )) \
  || { echo "MACARENA_SLOT must be 1 to $MAX_SLOTS (macOS runs at most $MAX_SLOTS macOS VMs at once)" >&2; exit 2; }
slot_state() { if (( $1 == 1 )); then printf '%s' "$RUNS"; else printf '%s/slot%s' "$RUNS" "$1"; fi; }
slot_bridge_port() { echo $(( 18095 + 100 * ($1 - 1) )); }
BRIDGE_PORT=$(slot_bridge_port "$SLOT")
API_PORT=$(( 18093 + 100 * (SLOT - 1) ))
# What one bridge + backend pair keeps to itself: pid file, logs, desktop lock, benchmark home, VM bundle. Slot 1
# keeps the folder it always used; slot N uses slotN/ inside it. Tokens, MacArena's env and the images are shared.
STATE="$(slot_state "$SLOT")"
UNIT="${MACARENA_STEP_UNIT:-actions}"   # actions (every action is a step) or turns (one step per agent turn; native track)
STEPS="${MACARENA_MAX_STEPS:-0}"        # 0 = no step limit; MACARENA_MAX_STEPS=15 is the paper's cap
TOKENS="$RUNS/tokens.env"

# What `setup` installs. The commit is MacArena's main as of 2026-11-14; change it on purpose, not by accident.
MACARENA_COMMIT="${MACARENA_COMMIT:-dcdc7d366da12641578ab90f085d69db91651c58}"
HF_BUCKET="hf://buckets/macpaw-research/MacArena"
UTM_DOCS="$HOME/Library/Containers/com.utmapp.UTM/Data/Documents"
UTM_BIN="/Applications/UTM.app/Contents/MacOS"
# Sizes of the .utm folders in the bucket (bytes, read from `hf buckets list` on 2026-11-14).
vm_bytes() { case "$1" in MacArena) echo 66936000000 ;; osworld) echo 54097000000 ;; esac; }

export PATH="$UTM_BIN:$PATH"                           # MacArena calls `utmctl` by name

# utmctl is the source of truth (the VM may live outside UTM's Documents folder); case-insensitive on the name.
utm_has() { utmctl list 2>/dev/null | awk 'NR>1{ $1=""; $2=""; sub(/^ +/,""); print }' | grep -qix "$1"; }

check() {
  local ok=1
  [[ -x "$ROOT/.venv/bin/python" ]] || { echo "missing: $ROOT/.venv (run ./start_web.sh once)"; ok=0; }
  [[ -d "$MA/evaluation_examples" ]] || { echo "missing: MacArena checkout at $MA  ->  scripts/macarena/bench.sh setup"; ok=0; }
  if [[ "${MACARENA_FAKE:-}" != 1 ]]; then
    [[ -d /Applications/UTM.app ]] || { echo "missing: UTM  ->  scripts/macarena/bench.sh setup"; ok=0; }
    (cd "$MA" 2>/dev/null && "$PY_MA" -c "import desktop_env.desktop_env" 2>/dev/null) \
      || { echo "missing: MacArena's Python env ($PY_MA cannot import desktop_env)  ->  scripts/macarena/bench.sh setup"; ok=0; }
    for n in MacArena osworld; do
      utm_has "$n" && echo "VM image '$n': present in UTM" || echo "VM image '$n': not in UTM  ->  scripts/macarena/bench.sh setup --vm $(printf '%s' "$n" | tr '[:upper:]' '[:lower:]')   (only the one your tasks need)"
    done
  fi
  [[ $ok == 1 ]] && echo "ok"; [[ $ok == 1 ]]
}

# ── setup ───────────────────────────────────────────────────────────────────

YES=0 DRY=0
say() { printf '%s\n' "$*"; }
run() { if [[ $DRY == 1 ]]; then say "  [dry-run] $*"; else "$@"; fi; }
confirm() {                                            # confirm "question"; --yes skips, a non-interactive shell refuses
  [[ $YES == 1 || $DRY == 1 ]] && return 0
  [[ -t 0 ]] || { say "  not a terminal: re-run with --yes to allow: $1" >&2; return 1; }
  read -r -p "  $1 [y/N] " a; [[ "$a" == [yY]* ]]
}
hf_cmd() { if [[ -x "$ROOT/.venv/bin/hf" ]]; then echo "$ROOT/.venv/bin/hf"; else echo "uvx --from huggingface_hub hf"; fi; }

setup() {
  local vm=macarena
  while [[ $# -gt 0 ]]; do case "$1" in
    --vm) vm="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;; --yes|-y) YES=1; shift ;; --dry-run) DRY=1; shift ;;
    *) echo "unknown: $1"; exit 2 ;; esac; done
  local want=()
  case "$vm" in macarena) want=(MacArena) ;; osworld) want=(osworld) ;; both) want=(MacArena osworld) ;; none) ;;
    *) echo "--vm must be macarena, osworld, both or none"; exit 2 ;; esac
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { say "setup needs an Apple Silicon Mac (UTM runs the macOS VMs)"; exit 1; }
  command -v uv >/dev/null || { say "missing: uv (https://docs.astral.sh/uv/)"; exit 1; }
  say "Licence notes: macOS in a VM is covered by Apple's licence; the macOSWorld tasks are CC BY-NC 4.0 (non-commercial)."

  say "1/4 UTM"
  if [[ -d /Applications/UTM.app ]]; then say "  already installed"
  else
    command -v brew >/dev/null || { say "  missing: Homebrew (or install UTM from https://github.com/utmapp/UTM)"; exit 1; }
    confirm "Install UTM with 'brew install --cask utm'?" && run brew install --cask utm || { say "  skipped (declined or failed; see above)"; }
  fi

  say "2/4 MacArena checkout (pinned to ${MACARENA_COMMIT:0:10})"
  if [[ -d "$MA/.git" ]]; then
    local head; head="$(git -C "$MA" rev-parse HEAD)"
    [[ "$head" == "$MACARENA_COMMIT" ]] && say "  present, at the pinned commit" \
      || say "  present at ${head:0:10}, not the pinned commit; left as is (results are only comparable at one commit)"
  else
    confirm "Clone https://github.com/MacPaw/MacArena into $MA ?" \
      && { run git clone https://github.com/MacPaw/MacArena "$MA"; run git -C "$MA" checkout --quiet "$MACARENA_COMMIT"; } || say "  skipped (declined or failed; see above)"
  fi

  say "3/4 MacArena Python env (uv, Python 3.11: its numpy pin has no 3.12 wheels)"
  if [[ ! -d "$MA" ]]; then say "  no checkout yet, skipped"
  elif [[ -x "$MA/.venv/bin/python" ]] && (cd "$MA" && "$MA/.venv/bin/python" -c "import desktop_env.desktop_env" 2>/dev/null); then say "  ready"
  else
    confirm "Create $MA/.venv and install MacArena's requirements (a few GB: torch, transformers, ...)?" \
      && {
        run uv venv "$MA/.venv" --python 3.11 --allow-existing
        # Two upstream problems, found on a real install: borb's wheel has duplicate zip entries, which uv refuses
        # (pip tolerates it), and tldextract is imported by desktop_env but missing from requirements.txt.
        if [[ $DRY == 0 ]]; then grep -v '^borb$' "$MA/requirements.txt" > "$RUNS/requirements-noborb.txt"; else : ; fi
        run uv pip install --python "$MA/.venv/bin/python" -r "$RUNS/requirements-noborb.txt" tldextract pip
        run "$MA/.venv/bin/python" -m pip install --quiet borb
      } || say "  skipped (declined or failed; see above)"
    if [[ $DRY == 0 && -x "$MA/.venv/bin/python" ]]; then
      if (cd "$MA" && "$MA/.venv/bin/python" -c "import desktop_env.desktop_env" 2>"$RUNS/import.err"); then say "  desktop_env imports"
      else tail -3 "$RUNS/import.err"; say "  desktop_env did not import: read the error above; MacArena also ships environment.yml (conda)"; fi
    fi
  fi

  say "4/4 VM image(s): ${want[*]:-none}"
  local n bytes free
  for n in ${want[@]+"${want[@]}"}; do
    if utm_has "$n"; then say "  $n: already in UTM"; continue; fi
    bytes="$(vm_bytes "$n")"; mkdir -p "$RUNS/vm"
    free=$(df -k "$RUNS/vm" | awk 'NR==2{print $4*1024}')
    say "  $n: $(( bytes / 1000000000 )) GB from $HF_BUCKET/$n.utm   (free: $(( free / 1000000000 )) GB; allow some spare room)"
    (( free > bytes + 40000000000 )) || { say "  not enough free disk space for the download plus UTM's copy; free some space and re-run"; return 1; }
    confirm "Download $n.utm (about $(( bytes / 1000000000 )) GB) from huggingface.co/buckets/macpaw-research/MacArena?" || { say "  skipped (declined or failed; see above)"; continue; }
    # sync is resumable: re-running after an interruption only fetches what is missing
    run $(hf_cmd) buckets sync "$HF_BUCKET/$n.utm" "$RUNS/vm/$n.utm"
    if [[ $DRY == 0 ]]; then
      # UTM opens a .utm in place (it registers the folder by reference, it does not copy it), and MacArena's provider
      # reads <UTM Documents>/<name>.utm/config.plist. So move the bundle (same disk: instant) into UTM's Documents.
      if [[ ! -d "$UTM_DOCS/$n.utm" ]]; then mkdir -p "$UTM_DOCS"; mv "$RUNS/vm/$n.utm" "$UTM_DOCS/$n.utm"; fi
      say "  opening it in UTM: accept the import prompt in the UTM window if one appears"
      open -a UTM "$UTM_DOCS/$n.utm"
      for _ in $(seq 1 60); do utm_has "$n" && break; sleep 5; done
      utm_has "$n" && say "  $n imported" || say "  not imported yet: finish the import in UTM, then run: scripts/macarena/bench.sh check"
      say "  the image now lives in $UTM_DOCS/$n.utm: do NOT delete it"
    fi
  done
  [[ $DRY == 1 ]] && { say "dry run: nothing was changed"; return 0; }
  say "--- check"; check || true
}

# ── Otto-native track: what the VM needs to run Otto's own desktop tools ─────

# One tarball, uploaded into every fresh VM by the bridge: Otto's toolkit files copied verbatim, the guest server,
# and the four packages the VM's Python lacks, pinned to this repo's venv versions and built for the VM's
# Python 3.12 / arm64 from wheels only (no network in the VM, nothing installed into its own environment).
native_bundle() {
  local nd="$RUNS/native" vers key deps st out="$STATE/native"
  vers="$("$ROOT/.venv/bin/python" -c 'import importlib.metadata as m; print(" ".join(f"{p}=={m.version(p)}" for p in ("pydantic", "langchain-core", "python-dotenv", "aiohttp")))')"
  mkdir -p "$nd"
  # Every transitive package at the host venv's version too (the host venv is Python 3.12, like the VM's).
  uv pip freeze --python "$ROOT/.venv/bin/python" | grep -E '^[A-Za-z0-9_.-]+==' >"$nd/constraints.txt"
  key="$( (printf '%s\n' "$vers"; cat "$nd/constraints.txt") | shasum | cut -c1-12)"
  deps="$nd/deps-$key"
  if [[ ! -d "$deps" ]]; then
    echo "building the VM's Python packages ($vers) ..."
    rm -rf "$deps.tmp"
    # shellcheck disable=SC2086
    uv pip install --quiet --target "$deps.tmp" --python-platform aarch64-apple-darwin --python-version 3.12 \
      --no-build --constraint "$nd/constraints.txt" $vers || { echo "could not build the VM's Python packages" >&2; exit 1; }
    mv "$deps.tmp" "$deps"
  fi
  # Stage and tarball per slot: each bridge uploads its tarball at every reset, so another slot's `up` must not
  # rewrite it mid-run. The built packages are shared.
  mkdir -p "$out"; st="$out/stage"; rm -rf "$st"
  mkdir -p "$st/src/tools/navigation/computer" "$st/src/utilities" "$st/src/macos_osascript" "$st/guest"
  : >"$st/src/tools/__init__.py"; : >"$st/src/tools/navigation/__init__.py"     # packages only: no other Otto code
  cp "$ROOT"/src/tools/navigation/computer/{__init__,base,macos_navigator,macos_tools,macos_ax_struct,macos_app_cards,macos_structured_tools}.py "$st/src/tools/navigation/computer/"
  # Otto's AppleScript tools: the macos-osascript server module (stdlib only without the MCP SDK) and its helpers.
  cp "$ROOT"/backend/builtin_mcps/macos_osascript/{server,_helpers,_desktop_lock}.py "$st/src/macos_osascript/"
  cp "$ROOT"/src/utilities/{__init__,environment,logger}.py "$st/src/utilities/"
  cp "$HERE/native_tools.py" "$st/"; cp "$HERE/guest/otto_tools_server.py" "$HERE/guest/permission_clicker.py" "$st/guest/"
  tar -czf "$out/bundle.tgz" -s ",^deps-$key,deps," -C "$st" src guest native_tools.py -C "$nd" "deps-$key"
  NATIVE_BUNDLE="$out/bundle.tgz"
  echo "native bundle: $NATIVE_BUNDLE ($(du -h "$NATIVE_BUNDLE" | cut -f1))"
}

# ── bridge + backend ────────────────────────────────────────────────────────

tokens() {
  mkdir -p "$RUNS"
  if [[ ! -f "$TOKENS" ]]; then
    umask 077
    printf 'MACARENA_AGENT_TOKEN=%s\nMACARENA_CONTROL_TOKEN=%s\n' "$(openssl rand -hex 16)" "$(openssl rand -hex 16)" >"$TOKENS"
  fi
  set -a; source "$TOKENS"; set +a
}

up() {
  local vm=osworld a11y=() do_setup=0 track=standard memory="${MACARENA_MEMORY:-off}"
  while [[ $# -gt 0 ]]; do case "$1" in
    --vm) vm="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;; --a11y) a11y=(--a11y); shift ;; --setup) do_setup=1; shift ;;
    --native) track=native; shift ;;
    --memory) memory="$2"; shift 2 ;;
    *) echo "unknown: $1"; exit 2 ;; esac; done
  if [[ $do_setup == 1 && "${MACARENA_FAKE:-}" != 1 ]]; then
    # `up --vm macarena` needs the MacArena image; `--vm osworld` needs the osworld one.
    setup --vm "$vm"
    if [[ -x "$MA/.venv/bin/python" && -z "${MACARENA_PYTHON:-}" ]]; then PY_MA="$MA/.venv/bin/python"; fi
  fi
  check; tokens; mkdir -p "$STATE"
  if slot_up "$SLOT"; then echo "slot $SLOT is already up (bridge on $BRIDGE_PORT): MACARENA_SLOT=$SLOT $0 stop first" >&2; exit 1; fi
  if [[ "${MACARENA_FAKE:-}" != 1 ]]; then
    local img; img=$([[ "$vm" == macarena ]] && echo MacArena || echo osworld)
    utm_has "$img" || { echo "the '$vm' VM is not in UTM: scripts/macarena/bench.sh setup --vm $vm" >&2; exit 1; }
  fi
  local fake=(); [[ "${MACARENA_FAKE:-}" == 1 ]] && fake=(--fake-env)
  # Screenshot pixels -> guest pyautogui points. On a Retina guest the VM reports the pixel size, so the
  # bridge's automatic scale is 1.0 but clicks land at 2x. Example: MACARENA_COORD_SCALE="0.5 0.5".
  [[ "${MACARENA_RECORD:-}" == 1 ]] && fake+=(--record)
  [[ -n "${MACARENA_COORD_SCALE:-}" ]] && fake+=(--coord-scale $MACARENA_COORD_SCALE)
  # Task setup: sync (default) waits for each task's pre_command and checks it; MACARENA_SETUP_MODE=launch starts it
  # without waiting, as MacArena does (bridge.py: SETUP_MODES).
  [[ -n "${MACARENA_SETUP_MODE:-}" ]] && fake+=(--setup-mode "$MACARENA_SETUP_MODE")
  if [[ $track == native ]]; then
    fake+=(--native)
    if [[ "${MACARENA_FAKE:-}" != 1 ]]; then native_bundle; fake+=(--native-bundle "$NATIVE_BUNDLE"); fi
  fi
  echo "slot $SLOT: starting the bridge on $BRIDGE_PORT (vm=$vm, track=$track, max_steps=$([[ $STEPS == 0 ]] && echo none || echo $STEPS)) ..."
  nohup "$PY_MA" "$HERE/bridge.py" --macarena-dir "$MA" --vm "$vm" --port "$BRIDGE_PORT" --max-steps "$STEPS" --step-unit "$UNIT" \
    --agent-token "$MACARENA_AGENT_TOKEN" --control-token "$MACARENA_CONTROL_TOKEN" ${a11y[@]+"${a11y[@]}"} ${fake[@]+"${fake[@]}"} \
    >"$STATE/bridge.log" 2>&1 &
  echo $! >"$STATE/bridge.pid"
  for _ in $(seq 1 120); do
    curl -fsS -H "Authorization: Bearer $MACARENA_CONTROL_TOKEN" "http://127.0.0.1:$BRIDGE_PORT/ctl/health" >/dev/null 2>&1 && break
    kill -0 "$(cat "$STATE/bridge.pid")" 2>/dev/null || { echo "bridge exited; see $STATE/bridge.log" >&2; exit 1; }
    sleep 1
  done
  if [[ "$memory" != off ]]; then
    [[ $track == native ]] || { echo "--memory needs --native (the standard track has no memory tool)" >&2; exit 2; }
    if [[ "$memory" != live ]]; then
      [[ -f "$memory" ]] || { echo "no procedure snapshot at $memory (scripts/procmem/build_snapshot.py)" >&2; exit 2; }
      memory="$(cd "$(dirname "$memory")" && pwd)/$(basename "$memory")"
    fi
  fi
  echo "slot $SLOT: starting Otto's backend on $API_PORT (procedural memory: $memory) ..."
  MACARENA_RUNS="$RUNS" MACARENA_STATE="$STATE" MACARENA_API_PORT="$API_PORT" MACARENA_TRACK="$track" MACARENA_MEMORY="$memory" \
    MACARENA_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" nohup perl -e 'use POSIX; setsid(); exec @ARGV' \
    "$HERE/start_backend.sh" >"$STATE/backend.log" 2>&1 &
  for _ in $(seq 1 90); do curl -fsS "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 && { echo "slot $SLOT: UP (vm=$vm, bridge $BRIDGE_PORT, backend $API_PORT, logs in $STATE)"; return; }; sleep 1; done
  echo "backend did not come up; see $STATE/backend.log" >&2; exit 1
}

slot_up() {                                # is slot $1's bridge running?
  local pid; pid="$(cat "$(slot_state "$1")/bridge.pid" 2>/dev/null)" || return 1
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

slots_up() {                               # how many slots have a bridge running
  local n=0 i; for i in $(seq 1 "$MAX_SLOTS"); do slot_up "$i" && n=$((n + 1)); done; echo "$n"
}

# ── one-time: let SSH-launched programs in the base VM see and control the screen ─────────────
#
# MacArena starts everything in the VM over SSH, and macOS attributes those programs' permission requests to
# /usr/libexec/sshd-keygen-wrapper (found in the VM's tccd log). Without Accessibility and Screen Recording for it,
# MacArena's accessibility tree fails (HTTP 500) and screenshots show only the wallpaper, no windows. The grant is
# a macOS security setting that needs the VM's admin password, so a person makes it, once, in the base image; every
# fresh copy then inherits it. This command boots the base VM, opens the pane, waits, checks, and shuts it down.
vm_ssh() { sshpass -p admin ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
             -o PreferredAuthentications=password -o LogLevel=ERROR -o ConnectTimeout=10 "admin@$1" "${@:2}"; }

vm_perms() {   # "ax=<bool> screen=<bool>" as an SSH-launched Python in the VM sees it
  vm_ssh "$1" '/opt/homebrew/Caskroom/miniconda/base/envs/server/bin/python3 -c "
from ApplicationServices import AXIsProcessTrusted
import Quartz
print(\"ax=%s screen=%s\" % (AXIsProcessTrusted(), Quartz.CGPreflightScreenCaptureAccess()))"'
}

grant_perms() {
  local vm=macarena ip="" i got ans
  utm_has "$vm" || { echo "the '$vm' VM is not in UTM: scripts/macarena/bench.sh setup --vm $vm" >&2; exit 1; }
  if [[ -n "$(utmctl list | awk -v n="$vm" 'NR>1 && $2=="started" && $3==n')" ]]; then
    echo "the base VM '$vm' is already running"
  else
    echo "starting the base VM '$vm' (not a copy: the change must land in the base image) ..."
    utmctl start "$vm" >/dev/null
  fi
  for i in $(seq 1 60); do ip="$(utmctl ip-address "$vm" 2>/dev/null | head -1)"; [[ -n "$ip" ]] && break; sleep 5; done
  [[ -n "$ip" ]] || { echo "no IP address for the VM" >&2; exit 1; }
  for i in $(seq 1 30); do vm_ssh "$ip" true 2>/dev/null && break; sleep 4; done
  vm_ssh "$ip" 'open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"'
  cat <<MSG

In the UTM window of the VM "$vm" (System Settings is open on Privacy & Security > Accessibility):

  1. Click "+", enter the VM's password when asked (MacArena's default is "admin").
  2. In the file chooser press Command-Shift-G, type  /usr/libexec/sshd-keygen-wrapper  and press Return, then Open.
  3. Make sure its switch is ON.
  4. Do the same under Privacy & Security > Screen & System Audio Recording
     (if macOS offers "Quit & Reopen", choose "Later").

MSG
  while true; do
    read -r -p "Press Return when both are done (q to give up): " ans
    [[ "$ans" == q ]] && { echo "gave up; the VM is still running"; exit 1; }
    got="$(vm_perms "$ip" 2>&1 | tail -1)"; echo "  the VM reports: $got"
    [[ "$got" == "ax=True screen=True" ]] && break
    echo "  not both granted yet (a permission shows as False until it is switched on)."
  done
  echo "granted. Shutting the VM down so the change is saved in the image ..."
  utmctl stop "$vm" >/dev/null 2>&1 || true
  echo "done: every copy of '$vm' now starts with the permission."
}

# ── one-time: let SSH-launched scripts talk to the data apps (Contacts, Reminders, Calendar, Notes) ──────
#
# Tasks set themselves up and are graded with `osascript` ("tell application Contacts ..."). The first time such a
# script asks a data app for its data, macOS shows "sshd-keygen-wrapper would like to access ..." and the script
# waits for a click that never comes, so the task's setup and grader silently fail. Answering the dialogs once in the
# base image fixes every copy. `run` checks this first and skips tasks that would fail (see preflight.py).
APPS_PROBE_DEFAULT=(Contacts Reminders Calendar Notes)

vm_probe() {   # exit 0 when `osascript` can query the app from an SSH session, within 15 s
  local q; q="$(app_query "$2")"
  vm_ssh "$1" "perl -e 'alarm 15; exec @ARGV' osascript -e 'tell application \"$2\" to $q' >/dev/null 2>&1"
}

# The question that makes macOS ask for the permission. "get name" is not enough for most apps: they answer it
# without a dialog, so the grant would be reported as done without ever being asked for. (Same list as preflight.py.)
app_query() {
  case "$1" in
    Contacts) echo "get count of people";; Reminders) echo "get count of lists";; Calendar) echo "get count of calendars";;
    Notes) echo "get count of folders";; "System Events") echo "get name of login items";; *) echo "get name";;
  esac
}

grant_apps() {
  local vm=macarena ip="" apps=() app i ans pending
  while [[ $# -gt 0 ]]; do case "$1" in
    --vm) vm="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    *) apps+=("$1"); shift ;; esac; done
  [[ ${#apps[@]} -gt 0 ]] || apps=("${APPS_PROBE_DEFAULT[@]}")
  utm_has "$vm" || { echo "the '$vm' VM is not in UTM: scripts/macarena/bench.sh setup --vm $vm" >&2; exit 1; }
  if [[ -n "$(utmctl list | awk -v n="$vm" 'NR>1 && $2=="started" && $3==n')" ]]; then
    echo "the base VM '$vm' is already running"
  else
    echo "starting the base VM '$vm' (not a copy: the change must land in the base image) ..."
    utmctl start "$vm" >/dev/null
  fi
  for i in $(seq 1 60); do ip="$(utmctl ip-address "$vm" 2>/dev/null | head -1)"; [[ -n "$ip" ]] && break; sleep 5; done
  [[ -n "$ip" ]] || { echo "no IP address for the VM" >&2; exit 1; }
  for i in $(seq 1 30); do vm_ssh "$ip" true 2>/dev/null && break; sleep 4; done
  for app in "${apps[@]}"; do      # ask once, in the background, so every dialog is on screen at the same time
    vm_ssh "$ip" "(perl -e 'alarm 120; exec @ARGV' osascript -e 'tell application \"$app\" to $(app_query "$app")' >/dev/null 2>&1 &)" || true
  done
  while true; do
    cat <<MSG

In the UTM window of the VM "$vm", macOS is asking for permission to control: ${apps[*]}
Click "OK" / "Allow" on each dialog ("sshd-keygen-wrapper would like to access ..."). If a dialog does not
appear, open System Settings > Privacy & Security > Automation (and Contacts / Calendars / Reminders) and switch
sshd-keygen-wrapper on.

MSG
    read -r -p "Press Return when done (q to give up): " ans
    [[ "$ans" == q ]] && { echo "gave up; the VM is still running"; exit 1; }
    pending=()
    for app in "${apps[@]}"; do
      if vm_probe "$ip" "$app"; then echo "  $app: ok"; else echo "  $app: still waiting"; pending+=("$app"); fi
    done
    [[ ${#pending[@]} -eq 0 ]] && break
    for app in "${pending[@]}"; do   # ask again for the ones still unanswered
      vm_ssh "$ip" "(perl -e 'alarm 120; exec @ARGV' osascript -e 'tell application \"$app\" to $(app_query "$app")' >/dev/null 2>&1 &)" || true
    done
  done
  echo "all answered. Shutting the VM down so the change is saved in the image ..."
  echo admin | vm_ssh "$ip" 'sudo -S shutdown -h now' >/dev/null 2>&1 || utmctl stop "$vm" >/dev/null 2>&1 || true
  echo "done: every copy of '$vm' now starts with these permissions."
}

# ── verify the base image: boots a throwaway COPY and checks everything the tasks rely on ──────────────────
#
# The permissions and installed apps in the base image decide what the benchmark can score, and they are set by
# hand (macOS only grants them to a person's click). This checks them without changing anything: it clones the
# base VM, boots the copy, tests each requirement and deletes the copy. Needs no password and never touches the base.
# Do not run it while a benchmark is running: a second VM slows the run and its timings.
SERVER_PY=/opt/homebrew/Caskroom/miniconda/base/envs/server/bin/python3

verify_image() {
  local vm=macarena extra=() apps=() ip="" i app copy fails=0 dirs=()
  while [[ $# -gt 0 ]]; do case "$1" in
    --vm) vm="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --apps) read -r -a extra <<<"$2"; shift 2 ;;
    *) echo "unknown: $1"; exit 2 ;; esac; done
  utm_has "$vm" || { echo "the '$vm' VM is not in UTM: scripts/macarena/bench.sh setup --vm $vm" >&2; exit 1; }
  if utmctl list | awk 'NR>1 && $2!="stopped"{n++} END{exit !n}'; then
    echo "a VM is running (a benchmark run?). Stop it first: a second VM would slow the run." >&2; exit 1
  fi
  case "$vm" in
    macarena) dirs=("$MA/evaluation_examples/macarena") ;;
    *) dirs=("$MA/evaluation_examples/osworld" "$MA/evaluation_examples/macosworld") ;;
  esac
  # Apps the tasks script with AppleScript, plus the data apps and any --apps you name.
  while IFS= read -r app; do [[ -n "$app" ]] && apps+=("$app"); done < <(
    { printf '%s\n' "${APPS_PROBE_DEFAULT[@]}" ${extra[@]+"${extra[@]}"}; "$PY_MA" "$HERE/preflight.py" apps "${dirs[@]}"; } | sort -u)

  copy="verify-$(date +%s)"
  # Double quotes: the copy's name is expanded now, because `copy` is local and gone when the EXIT trap runs.
  trap "utmctl stop '$copy' >/dev/null 2>&1; sleep 3; utmctl delete '$copy' >/dev/null 2>&1; echo 'copy $copy removed'" EXIT
  echo "cloning '$vm' into the throwaway copy '$copy' (the base is not touched) ..."
  utmctl clone "$vm" --name "$copy" >/dev/null
  utmctl start "$copy" >/dev/null
  for i in $(seq 1 60); do ip="$(utmctl ip-address "$copy" 2>/dev/null | head -1)"; [[ -n "$ip" ]] && break; sleep 5; done
  [[ -n "$ip" ]] || { echo "FAIL  the copy got no IP address"; exit 1; }
  for i in $(seq 1 40); do vm_ssh "$ip" true 2>/dev/null && break; sleep 4; done
  vm_ssh "$ip" true 2>/dev/null || { echo "FAIL  SSH (admin/admin) does not work in this image"; exit 1; }
  echo "PASS  boots and accepts SSH"

  bad() { echo "FAIL  $1"; fails=$((fails+1)); }
  local got; got="$(vm_perms "$ip" 2>&1 | tail -1)"
  [[ "$got" == *"ax=True"* ]] && echo "PASS  Accessibility for SSH-launched programs" \
    || bad "Accessibility not granted ($got)  ->  bench.sh grant-perms"
  [[ "$got" == *"screen=True"* ]] && echo "PASS  Screen Recording for SSH-launched programs" \
    || bad "Screen Recording not granted ($got)  ->  bench.sh grant-perms"

  if vm_ssh "$ip" "[ -x $SERVER_PY ]"; then
    vm_ssh "$ip" "$SERVER_PY -c 'import os; os.listdir(os.path.expanduser(\"~/Documents\"))'" >/dev/null 2>&1 \
      && echo "PASS  Full Disk Access (the grader server can list ~/Documents)" \
      || bad "Full Disk Access missing for $SERVER_PY: the grader cannot read files apps write in ~/Documents (README: Prerequisites)"
  else
    bad "the grader server's Python is not at $SERVER_PY, so Full Disk Access cannot be checked"
  fi

  for app in "${apps[@]}"; do
    if ! vm_ssh "$ip" "osascript -e 'id of application \"$app\"' >/dev/null 2>&1"; then
      bad "$app: not installed"
    elif vm_probe "$ip" "$app"; then echo "PASS  $app: installed and scriptable"
    else bad "$app: installed, but a script cannot control it (Automation not granted)  ->  bench.sh grant-apps \"$app\""
    fi
  done

  echo
  if [[ $fails == 0 ]]; then echo "image '$vm': all requirements met"; else echo "image '$vm': $fails requirement(s) missing (see FAIL lines)"; fi
  [[ $fails == 0 ]]
}

case "${1:-}" in
  verify-image) shift; verify_image "$@" ;;
  grant-perms) shift; grant_perms "$@" ;;
  grant-apps) shift; grant_apps "$@" ;;
  setup) shift; setup "$@" ;;
  check) check ;;
  up) shift; up "$@" ;;
  run) shift; tokens
       # Keep the Mac awake for the run: idle sleep froze a run for 16 minutes while its time limit kept counting.
       # (Closing the lid still sleeps it.) MACARENA_ALLOW_SLEEP=1 turns this off.
       awake="caffeinate -i -s"; [[ "${MACARENA_ALLOW_SLEEP:-0}" == 1 ]] && awake=""
       # How many VMs are running now, recorded with each result (agents share the model server, so it matters).
       par="${MACARENA_PARALLEL:-$(slots_up)}"; (( par >= 1 )) || par=1
       echo "slot $SLOT: running against bridge $BRIDGE_PORT, backend $API_PORT" >&2
       exec $awake "$ROOT/.venv/bin/python" "$HERE/run.py" --macarena-dir "$MA" --bridge "http://127.0.0.1:$BRIDGE_PORT" \
         --api "http://127.0.0.1:$API_PORT" --control-token "$MACARENA_CONTROL_TOKEN" --max-steps "$STEPS" --step-unit "$UNIT" \
         --parallel "$par" "$@" ;;
  slots) for i in $(seq 1 "$MAX_SLOTS"); do
           st="$(slot_state "$i")"; p=$(slot_bridge_port "$i")
           if slot_up "$i"; then echo "slot $i: up (bridge $p, backend $((p - 2)), logs in $st)"; else echo "slot $i: down"; fi
         done ;;
  report) shift; exec "$ROOT/.venv/bin/python" "$HERE/report.py" "$@" ;;
  stop) # by pid file, then by command line: a bridge left running keeps a VM (and its RAM) alive
        pat="scripts/macarena/bridge.py .*--port $BRIDGE_PORT"
        # One signal only: the bridge deletes its VM copy on the way out and a second one would interrupt that.
        if [[ -f "$STATE/bridge.pid" ]] && kill "$(cat "$STATE/bridge.pid")" 2>/dev/null; then :
        else pkill -f "$pat" 2>/dev/null || true; fi
        rm -f "$STATE/bridge.pid"
        # The bridge stops and deletes its VM copy on the way out; give it time to do that.
        for _ in $(seq 1 90); do pgrep -f "$pat" >/dev/null || break; sleep 1; done
        if pgrep -f "$pat" >/dev/null; then echo "bridge still shutting down (VM copy may remain; see utmctl list)"
        else echo "slot $SLOT: bridge stopped"; fi
        pkill -f "backend(\.py)? --host 127.0.0.1 --port $API_PORT" && echo "slot $SLOT: backend stopped" || echo "slot $SLOT: backend not running" ;;
  *) sed -n 2,31p "$0"; exit 2 ;;
esac
