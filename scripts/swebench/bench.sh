#!/usr/bin/env bash
# bench.sh: set up, run and score Otto on SWE-bench in one command.
#
#   scripts/swebench/bench.sh                 # smoke test: 2 Verified instances, then score
#   scripts/swebench/bench.sh check           # only check/set up prerequisites
#   scripts/swebench/bench.sh run --sample 50 # 50 repo-stratified Verified instances
#   scripts/swebench/bench.sh score --out DIR # score an earlier run folder
#   scripts/swebench/bench.sh warm            # one-time: download everything needed to run offline
#   scripts/swebench/bench.sh run --sample 10 --offline   # run with no internet (after warm)
#
# Options
#   --sample N | --ids a,b | --repo NAME   which instances (default: 2 fixed smoke ids)
#   --seed N                               with --sample: which random sample (default 0; same seed = same instances)
#   --dataset verified|lite|full           default verified
#   --workers N                            parallel instances (default auto: 8 for cloud models, the oMLX
#                                          max_concurrent_requests setting for oMLX, 1 for in-process MLX)
#   --score-batch N                        instances per scoring batch (default 50; 0 = all at once)
#   --retry-errors                         score: also re-score instances that errored
#   --status                               score: show counts and exit (no Docker needed)
#   --reset ids|errors|all                 score: mark instances pending again (history is kept)
#   --min-docker-gb N                      stop scoring if Docker's virtual disk has less than N GB free (default 30;
#                                          a full 500-instance run needs about 150 GB)
#   --env-mode auto|uv|docker              where the agent's Python environment lives (default auto: a host uv
#                                          virtualenv, or Docker for Python 3.5-3.7, repos with compiled
#                                          extensions, and host environments that fail to import the project)
#   --retry-failed                         run: redo instances that were skipped or crashed (not ones the
#                                          agent simply failed to solve)
#   --prebuilt                             score with the official prebuilt images (pulled and removed per batch);
#                                          use this for numbers you quote. Default builds images locally.
#   --image-source official|bundle         which prebuilt image set to use for the agent's Docker environment, the
#                                          gold check and scoring. bundle = LogicStar's compressed set (5 GB download,
#                                          ~35 GB on disk, loaded once by `warm`, Verified only). Default: official
#                                          (bundle with --offline).
#   --offline                              no network at all: never pulls, reads the dataset from the local copy
#                                          `warm` made, forces Docker environments, sets HF/uv offline switches, and
#                                          stops early (before any agent runs) if something is missing
#   --keep-images                          keep pulled prebuilt images after each batch
#   --validate-gold                        also score the dataset's correct patches, and report which instances
#                                          fail even with them (environment problems, not agent problems)
#   --gold-only                            run only the gold validation
#   --resume                               run/score: use the most recent run folder
#   --score-workers N                      parallel scoring containers (default auto: min(8, cores/2))
#   --timeout SECONDS                      per-instance cap (default auto: 900 cloud, 5400 local)
#   --tool-budget N                        max tool calls per instance (default 200)
#   --prompt-cache on|off                  in-process MLX prompt caching (default on; oMLX caches server-side)
#   --provider P --model M [--api-key K]   override the model (anthropic|openai|google|omlx|mlx)
#   --runs-dir DIR                         where everything is written (default: otto-swebench-runs/ next to the otto repo;
#                                          also settable with SWEBENCH_RUNS)
#   --out DIR                              this run's results folder (default <runs-dir>/out/<time>-<mode>)
#   --method local|sbcli                   scoring method (default local = Docker)
#   --no-score                             run only
#   --yes                                  allow installs that need Homebrew (Docker Desktop)
#   --keep-backend                         leave the benchmark backend running afterwards
#
# Model choice when you give none: your current Otto model if it is usable
# (oMLX server reachable, MLX model downloaded, or a cloud provider); otherwise
# a local MLX model sized to this Mac's RAM, downloaded automatically.
set -euo pipefail

OTTO_LOG_TAG="swebench"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/bootstrap.sh
source "$HERE/../lib/bootstrap.sh"

RUNS="${SWEBENCH_RUNS:-$(dirname "$OTTO_ROOT")/otto-swebench-runs}"   # sibling of the otto repo by default
PORT=18091
API="http://127.0.0.1:$PORT"
SMOKE_IDS="django__django-16661,django__django-16100"
REQS="$HERE/requirements.txt"

SEED=0; MODE="smoke"; DATASET="verified"; WORKERS=""; TIMEOUT=""; SCORE_WORKERS=""; METHOD="local"
PROMPT_CACHE="on"; TOOL_BUDGET=200; EFF_PROVIDER=""
OFFLINE=0; IMAGE_SOURCE=""; ENV_MODE=auto; RETRY_FAILED=0; PREBUILT=0; VALIDATE_GOLD=0; GOLD_ONLY=0; KEEP_IMAGES=0; MIN_DOCKER_GB=30; SCORE_BATCH=50; RETRY_ERRORS=0; SCORE_STATUS=0; SCORE_RESET=""; RESUME=0
SAMPLE=""; IDS=""; REPO=""; OUT=""; DO_SCORE=1; ASSUME_YES=0; KEEP=0
PROVIDER="${SWEBENCH_PROVIDER:-}"; MODEL="${SWEBENCH_MODEL:-}"; APIKEY="${SWEBENCH_API_KEY:-}"

if [[ $# -gt 0 && "$1" != -* ]]; then MODE="$1"; shift; fi
case "$MODE" in smoke|run|score|check|warm) ;; *) die "unknown mode '$MODE' (smoke|run|score|check|warm)";; esac
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample) SAMPLE="$2"; shift 2;;   --seed) SEED="$2"; shift 2;;   --ids) IDS="$2"; shift 2;;
        --repo) REPO="$2"; shift 2;;       --dataset) DATASET="$2"; shift 2;;
        --workers) WORKERS="$2"; shift 2;; --timeout) TIMEOUT="$2"; shift 2;;
        --provider) PROVIDER="$2"; shift 2;; --model) MODEL="$2"; shift 2;;
        --api-key) APIKEY="$2"; shift 2;;  --out) OUT="$2"; shift 2;;
        --runs-dir) RUNS="$2"; shift 2;;
        --score-workers) SCORE_WORKERS="$2"; shift 2;;
        --score-batch) SCORE_BATCH="$2"; shift 2;;  --retry-errors) RETRY_ERRORS=1; shift;;
        --status) SCORE_STATUS=1; shift;;           --reset) SCORE_RESET="$2"; shift 2;;
        --resume) RESUME=1; shift;;
        --env-mode) ENV_MODE="$2"; shift 2;;        --retry-failed) RETRY_FAILED=1; shift;;
        --offline) OFFLINE=1; shift;;               --image-source) IMAGE_SOURCE="$2"; shift 2;;
        --prebuilt) PREBUILT=1; shift;;             --keep-images) KEEP_IMAGES=1; shift;;
        --validate-gold) VALIDATE_GOLD=1; shift;;   --gold-only) GOLD_ONLY=1; shift;;
        --min-docker-gb) MIN_DOCKER_GB="$2"; shift 2;;
        --prompt-cache) PROMPT_CACHE="$2"; shift 2;;
        --tool-budget) TOOL_BUDGET="$2"; shift 2;;
        --method) METHOD="$2"; shift 2;;   --no-score) DO_SCORE=0; shift;;
        --yes) ASSUME_YES=1; shift;;       --keep-backend) KEEP=1; shift;;
        -h|--help) awk 'NR>1 && /^set -euo/{exit} NR>1{print}' "${BASH_SOURCE[0]}"; exit 0;;
        *) die "unknown option $1";;
    esac
done
[[ "$MODE" == "smoke" && -z "$SAMPLE$IDS$REPO" ]] && IDS="$SMOKE_IDS"
RUNS="${RUNS/#\~/$HOME}"; mkdir -p "$RUNS"; RUNS="$(cd "$RUNS" && pwd)"
if [[ -z "$OUT" && ( $RESUME -eq 1 || "$MODE" == "score" ) ]]; then
    OUT="$(ls -td "$RUNS"/out/*/ 2>/dev/null | while read -r d; do [[ -f "${d}preds.jsonl" ]] && { echo "${d%/}"; break; }; done)"
    [[ -n "$OUT" ]] && info "Using most recent run: $OUT"
fi
[[ -z "$OUT" ]] && OUT="$RUNS/out/$(date +%Y%m%d-%H%M%S)-$MODE"

BUNDLE_URL="https://huggingface.co/LogicStar/SWE-Bench-Verified-Compressed/resolve/main/saved.tar.zst"
case "$IMAGE_SOURCE" in ""|official|bundle) ;; *) die "--image-source must be official or bundle";; esac
if [[ $OFFLINE -eq 1 ]]; then
    [[ -z "$IMAGE_SOURCE" ]] && IMAGE_SOURCE=bundle
    [[ "$ENV_MODE" == "auto" ]] && ENV_MODE=docker      # host virtualenvs need PyPI and compilers
    ASSUME_YES=0                                        # no Homebrew installs
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 UV_OFFLINE=1 SWEBENCH_OFFLINE=1
fi
export SWEBENCH_HTTP_CACHE="$RUNS/http-cache"      # the scorer's GitHub lookups, recorded online, replayed offline
[[ "$MODE" == "warm" && -z "$IMAGE_SOURCE" ]] && IMAGE_SOURCE=bundle
case "$IMAGE_SOURCE" in bundle) IMAGE_NS=logicstar;; *) IMAGE_NS=swebench;; esac
# Use the local copy of the dataset when `warm` made one: no Hugging Face round trips, and no network needed.
DATASET_NAME="$DATASET"
LOCAL_DS="$RUNS/dataset/swe-bench_${DATASET_NAME}.jsonl"
if [[ "$MODE" != "warm" && -f "$LOCAL_DS" && "$DATASET" != *.jsonl ]]; then DATASET="$LOCAL_DS"; fi
[[ "$METHOD" == "sbcli" ]] && DATASET="$DATASET_NAME"   # the cloud scorer needs the dataset name, not a file
OUT="${OUT/#\~/$HOME}"
BACKEND_PID=""

cleanup() {
    if [[ -n "$BACKEND_PID" && $KEEP -eq 0 ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
        pkill -f "backend --host 127.0.0.1 --port $PORT" 2>/dev/null || true
        pkill -f "$RUNS/home/" 2>/dev/null || true       # any tool server the backend left behind
    fi
}
trap cleanup EXIT INT TERM

# ── prerequisites ────────────────────────────────────────────────────
check_platform() {
    [[ "$(uname -s)" == "Darwin" ]] || die "Otto's agent sandbox is macOS-only (Seatbelt)."
    local free_gb; free_gb=$(df -g "$HOME" | awk 'NR==2{print $4}')
    info "Free disk: ${free_gb} GB"
    (( free_gb >= 40 )) || warn "Less than 40 GB free. Repos, Python builds, Docker images and models are large."
}

setup_python() {
    if [[ $OFFLINE -eq 1 ]]; then
        command -v uv >/dev/null 2>&1 || die "offline: uv is not installed (it cannot be downloaded without a network)."
        [[ -d "$VENV_DIR" ]] || die "offline: Otto's Python environment ($VENV_DIR) does not exist yet. Run once while online."
    else
        ensure_uv
        ensure_venv
    fi
    if ! (cd "$OTTO_ROOT" && PYTHONPATH=src "$VENV_DIR/bin/python" -c "import fastapi, langgraph, mlx_lm") >/dev/null 2>&1; then
        [[ $OFFLINE -eq 1 ]] && die "offline: Otto's Python dependencies are incomplete. Run once while online."
        sync_python_deps --quiet
    fi
    info "Python environment ready."
}

docker_ready() { docker info >/dev/null 2>&1; }

setup_docker() {
    if docker_ready; then info "Docker is running."; return 0; fi
    if ! command -v docker >/dev/null 2>&1 && [[ ! -d /Applications/Docker.app ]]; then
        if [[ $ASSUME_YES -eq 1 ]] && command -v brew >/dev/null 2>&1; then
            info "Installing Docker Desktop with Homebrew (--yes given)…"
            brew install --cask docker || { warn "Docker install failed."; return 1; }
        else
            warn "Docker is not installed. Scoring needs it."
            warn "Install Docker Desktop (https://www.docker.com/products/docker-desktop/) or re-run with --yes to use Homebrew."
            return 1
        fi
    fi
    info "Starting Docker Desktop…"
    open -a Docker 2>/dev/null || true
    for _ in $(seq 1 60); do docker_ready && { info "Docker is running."; return 0; }; sleep 3; done
    warn "Docker did not become ready in 3 minutes (it may be waiting for you to accept its first-run prompt)."
    return 1
}

# Free space inside Docker's own virtual disk (not the Mac's disk). Image builds fail
# half way with "no space left on device" when this is full.
docker_free_gb() {
    local img="alpine"
    docker image inspect "$img" >/dev/null 2>&1 || docker pull -q "$img" >/dev/null 2>&1 || return 1
    docker run --rm "$img" df -k / 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}'
}

docker_space_check() {
    local free; free="$(docker_free_gb)" || { warn "Could not read Docker's free disk space; skipping the check."; return 0; }
    [[ -z "$free" ]] && { warn "Could not read Docker's free disk space; skipping the check."; return 0; }
    info "Docker virtual disk free: ${free} GB"
    if (( free < MIN_DOCKER_GB )); then
        warn "Docker's virtual disk has only ${free} GB free (minimum ${MIN_DOCKER_GB} GB). Image builds will fail."
        warn "Fix: Docker Desktop → Settings → Resources → Advanced → raise 'Virtual disk limit'"
        warn "     (a full 500-instance run needs about 150 GB), or free space with 'docker system df'."
        return 1
    fi
    (( free < 150 )) && warn "Under 150 GB free in Docker: fine for a small run, not enough for all 500 instances."
    return 0
}

ram_gb() { echo $(( $(sysctl -n hw.memsize) / 1073741824 )); }

omlx_has_model() {   # $1 = model id as configured (org/name or org--name)
    local want="${1//\//--}"
    curl -sf -m 3 "http://127.0.0.1:52414/v1/models" 2>/dev/null | grep -q "\"$want\""
}

hf_cached() {        # $1 = org/name
    local d="$HOME/.cache/huggingface/hub/models--${1//\//--}/snapshots"
    [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]
}

download_model() {   # $1 = org/name
    [[ $OFFLINE -eq 1 ]] && die "offline: model $1 is not downloaded. Run 'bench.sh warm' (or any online run) first."
    info "Downloading model $1 (this can take a while)…"
    HF_HUB_ENABLE_HF_TRANSFER=0 "$VENV_DIR/bin/hf" download "$1" \
        --include "*.safetensors" "*.json" "tokenizer*" "*.txt" "*.model" >/dev/null \
        || die "Model download failed. Check your network, or pass --provider/--model."
    success "Model downloaded."
}

choose_model() {
    if [[ -n "$PROVIDER" || -n "$MODEL" ]]; then
        [[ -z "$PROVIDER" ]] && PROVIDER="mlx"
        EFF_PROVIDER="$PROVIDER"
        info "Using model from options: $PROVIDER ${MODEL:-(config default)}"
    else
        # Look at what the real Otto config already uses.
        local cfg="$HOME/Library/Application Support/Otto/config.json" cur_p="" cur_m=""
        if [[ -f "$cfg" ]]; then
            read -r cur_p cur_m < <(python3 - "$cfg" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); p = d["llm"]["provider"]
m = d["omlx"].get("model_name", "") if p == "omlx" else (
    d["llm"].get("mlx", {}).get("hf_llm_model_id", "") if p == "mlx" else
    d["llm"].get(p, {}).get("model_name", ""))
print(p, m or "-")
PY
)
        fi
        case "$cur_p" in
            omlx) omlx_has_model "$cur_m" && { EFF_PROVIDER=omlx; info "Using your Otto model: omlx $cur_m"; return; };;
            mlx)  [[ "$cur_m" != "-" ]] && hf_cached "$cur_m" && { EFF_PROVIDER=mlx; info "Using your Otto model: mlx $cur_m"; return; };;
            anthropic|openai|google)
                EFF_PROVIDER="$cur_p"
                info "Using your Otto cloud model: $cur_p ${cur_m:--}. This is billed by the provider."; return;;
        esac
        PROVIDER="mlx"; EFF_PROVIDER=mlx
        if (( $(ram_gb) >= 32 )); then MODEL="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
        else MODEL="mlx-community/Qwen3-8B-4bit"; warn "Under 32 GB RAM: using a small model. Expect low scores."; fi
        info "No usable Otto model found. Defaulting to local MLX: $MODEL"
    fi
    if [[ "$PROVIDER" == "mlx" ]]; then
        [[ -z "$MODEL" ]] && die "--provider mlx needs --model <hf repo id>"
        hf_cached "$MODEL" || download_model "$MODEL"
    fi
}

# Pick parallelism and timeout from the model that will actually be used.
auto_tune() {
    local cores; cores=$(sysctl -n hw.ncpu)
    if [[ -z "$WORKERS" ]]; then
        case "$EFF_PROVIDER" in
            anthropic|openai|google) WORKERS=8;;
            omlx) WORKERS=$(python3 - <<'PY'
import json, os
try:
    print(json.load(open(os.path.expanduser("~/.omlx/settings.json")))["scheduler"]["max_concurrent_requests"])
except Exception:
    print(2)
PY
);;
            *) WORKERS=1;;
        esac
    fi
    if [[ -z "$TIMEOUT" ]]; then
        # A local server is shared by every worker, so wall-clock time per instance is mostly
        # spent waiting for the model. The tool budget is the real cap; this only stops hangs.
        case "$EFF_PROVIDER" in anthropic|openai|google) TIMEOUT=900;; *) TIMEOUT=5400;; esac
    fi
    if [[ -z "$SCORE_WORKERS" ]]; then
        SCORE_WORKERS=$(( cores / 2 )); (( SCORE_WORKERS > 8 )) && SCORE_WORKERS=8; (( SCORE_WORKERS < 1 )) && SCORE_WORKERS=1
    fi
    [[ "$PROMPT_CACHE" == "off" ]] && PC=0 || PC=1
    info "Parallelism: $WORKERS agent worker(s), $SCORE_WORKERS scoring worker(s); timeout ${TIMEOUT}s; tool budget $TOOL_BUDGET"
}

# ── backend ──────────────────────────────────────────────────────────
start_backend() {
    mkdir -p "$RUNS"
    pkill -f "backend --host 127.0.0.1 --port $PORT" 2>/dev/null || true
    sleep 1
    info "Starting benchmark backend on port ${PORT}…"
    SWEBENCH_RUNS="$RUNS" SWEBENCH_PROVIDER="$PROVIDER" SWEBENCH_MODEL="$MODEL" SWEBENCH_API_KEY="$APIKEY" \
        SWEBENCH_PROMPT_CACHE="$PC" SWEBENCH_TOOL_BUDGET="$TOOL_BUDGET" \
        nohup perl -e 'use POSIX; setsid(); exec @ARGV' "$HERE/start_backend.sh" \
        > "$RUNS/backend.log" 2>&1 < /dev/null &
    BACKEND_PID=$!
    for _ in $(seq 1 90); do
        curl -sf -m 2 "$API/api/health" >/dev/null 2>&1 && { success "Backend is up. Model: $(grep -m1 'benchmark model' "$RUNS/backend.log" | cut -d: -f2-)"; return; }
        sleep 2
    done
    tail -20 "$RUNS/backend.log" >&2
    die "Backend did not start. Log: $RUNS/backend.log"
}

# ── run and score ────────────────────────────────────────────────────
run_instances() {
    local args=(--dataset "$DATASET" --workers "$WORKERS" --timeout "$TIMEOUT" --out "$OUT" --work "$RUNS/work" --api "$API")
    [[ -n "$IDS" ]] && args+=(--ids "$IDS")
    [[ -n "$SAMPLE" ]] && args+=(--sample "$SAMPLE" --seed "$SEED")
    [[ -n "$REPO" ]] && args+=(--repo "$REPO")
    args+=(--env-mode "$ENV_MODE" --image-namespace "$IMAGE_NS")
    [[ $RETRY_FAILED -eq 1 ]] && args+=(--retry-failed)
    [[ $OFFLINE -eq 1 ]] && args+=(--no-pull)
    info "Running instances → $OUT"
    uv run --with-requirements "$REQS" python "$HERE/run.py" "${args[@]}"
}

score_run() {
    if [[ "$METHOD" == "local" ]] && ! setup_docker; then
        warn "Skipping scoring. Finished instances stay saved. When Docker is ready, run:"
        warn "  $HERE/bench.sh score --out $OUT"
        return 0
    fi
    if [[ "$METHOD" == "local" ]] && ! docker_space_check; then
        warn "Not scoring. Finished instances stay saved; re-run 'bench.sh score' after fixing Docker."
        return 1
    fi
    info "Scoring in batches of $SCORE_BATCH (first run builds Docker images; slow on Apple Silicon)…"
    local extra=()
    [[ $RETRY_ERRORS -eq 1 ]] && extra+=(--retry-errors)
    if [[ "$IMAGE_SOURCE" == "bundle" ]]; then extra+=(--namespace "$IMAGE_NS")
    elif [[ $PREBUILT -eq 1 ]]; then extra+=(--prebuilt); fi
    [[ $OFFLINE -eq 1 ]] && extra+=(--no-pull)
    [[ $KEEP_IMAGES -eq 1 ]] && extra+=(--keep-images)
    [[ $VALIDATE_GOLD -eq 1 ]] && extra+=(--validate-gold)
    [[ $GOLD_ONLY -eq 1 ]] && extra+=(--gold-only)
    uv run --with-requirements "$REQS" python "$HERE/score.py" --out "$OUT" --dataset "$DATASET" \
        --method "$METHOD" --workers "$SCORE_WORKERS" --batch "$SCORE_BATCH" ${extra[@]+"${extra[@]}"}
}

# ── offline: warm (one time, online) and preflight (every offline run) ──
bundle_image_count() { docker images --format '{{.Repository}}' | grep -c '^logicstar/sweb.eval' || true; }

warm_images() {
    if [[ "$IMAGE_SOURCE" == "official" ]]; then
        warn "Official images are pulled on demand (about 3 GB each) and removed after scoring unless you use --keep-images."
        warn "For offline use, run: bench.sh warm   (the default image source is the compressed bundle)."
        return 0
    fi
    [[ "$DATASET_NAME" == "verified" ]] || die "The compressed image bundle covers SWE-bench Verified only."
    local have want; have=$(bundle_image_count); want=$(wc -l < "$LOCAL_DS" | tr -d ' ')
    if (( have >= want )); then info "Image bundle already loaded ($have images)."; return 0; fi
    command -v zstd >/dev/null 2>&1 || die "zstd is needed to unpack the bundle: brew install zstd"
    local free; free="$(docker_free_gb || echo 0)"
    (( free >= 60 )) || die "Docker has ${free} GB free; the bundle needs about 35 GB. Raise Docker's virtual disk limit."
    info "Downloading and loading the image bundle (5.4 GB download, about 35 GB on disk, roughly 10 minutes)…"
    curl -fL -# "$BUNDLE_URL" | zstd -d --long=31 --stdout | docker load > "$RUNS/bundle-load.log"
    have=$(bundle_image_count)
    (( have >= want )) || die "Only $have of $want bundle images loaded. See $RUNS/bundle-load.log"
    success "Loaded $have images."
}

warm_all() {
    [[ $OFFLINE -eq 1 ]] && die "warm needs the internet; do not combine it with --offline."
    mkdir -p "$RUNS/dataset"
    info "Exporting the $DATASET_NAME dataset → $LOCAL_DS"
    uv run --with-requirements "$REQS" python "$HERE/data.py" export --dataset "$DATASET_NAME" --out "$LOCAL_DS"
    info "Recording the scorer's GitHub lookups for every instance (so scoring needs no network)…"
    uv run --with-requirements "$REQS" python "$HERE/data.py" warm-scorer-cache --dataset "$LOCAL_DS"
    info "Caching the benchmark's Python packages…"
    uv run --with-requirements "$REQS" python -c "import datasets, httpx, websockets, swebench"
    setup_docker || die "warm needs Docker."
    warm_images
    "$VENV_DIR/bin/python" - "$RUNS/warm.json" "$LOCAL_DS" "$IMAGE_SOURCE" "$(bundle_image_count)" <<'PY'
import json, sys, time
out, ds, src, n = sys.argv[1:5]
rows = sum(1 for _ in open(ds))
json.dump({"warmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_file": ds, "dataset_rows": rows,
           "image_source": src, "bundle_images": int(n)}, open(out, "w"), indent=2)
PY
    success "Warm. Run offline with:  bench.sh run --sample 10 --offline"
}

# Everything an offline run needs, checked before any agent starts, so a gap shows up now
# and not an hour in. Nothing here touches the network.
offline_preflight() {
    local bad=0
    [[ -f "$DATASET" && "$DATASET" == *.jsonl ]] || { warn "offline: no local dataset (expected $LOCAL_DS). Run 'bench.sh warm' while online."; bad=1; }
    docker_ready || { warn "offline: Docker is not running."; bad=1; }
    if docker_ready; then
        local n; n=$(docker images --format '{{.Repository}}' | grep -c "^${IMAGE_NS}/sweb.eval" || true)
        (( n > 0 )) || { warn "offline: no ${IMAGE_NS} images are loaded. Run 'bench.sh warm' while online."; bad=1; }
        info "offline: $n ${IMAGE_NS} images present."
    fi
    [[ -n "$(ls "$SWEBENCH_HTTP_CACHE" 2>/dev/null)" ]] \
        || warn "offline: the scorer's GitHub cache is empty (run 'bench.sh warm'); scoring still works with prebuilt images."
    (( bad == 0 )) || die "Not ready to run offline."
    if [[ "$MODE" == "smoke" || "$MODE" == "run" ]]; then
        local args=(--dataset "$DATASET" --out "$OUT" --work "$RUNS/work" --env-mode "$ENV_MODE"
                    --image-namespace "$IMAGE_NS" --no-pull --dry-run)
        [[ -n "$IDS" ]] && args+=(--ids "$IDS")
        [[ -n "$SAMPLE" ]] && args+=(--sample "$SAMPLE" --seed "$SEED")
        [[ -n "$REPO" ]] && args+=(--repo "$REPO")
        uv run --with-requirements "$REQS" python "$HERE/run.py" "${args[@]}" \
            || die "offline: some selected instances have no local image (see above)."
    fi
    success "Offline preflight passed."
}

# ── main ─────────────────────────────────────────────────────────────
check_platform
setup_python
case "$MODE" in
    warm)
        choose_model
        warm_all
        ;;
    check)
        choose_model
        if setup_docker; then docker_space_check || true; fi
        [[ $OFFLINE -eq 1 ]] && offline_preflight
        success "Prerequisites checked. Model: ${PROVIDER:-(your Otto config)} ${MODEL}"
        ;;
    score)
        auto_tune
        [[ -f "$OUT/preds.jsonl" ]] || die "No preds.jsonl in $OUT (use --out <run folder>)"
        if [[ $SCORE_STATUS -eq 1 || -n "$SCORE_RESET" ]]; then
            args=(--out "$OUT" --dataset "$DATASET")
            [[ $SCORE_STATUS -eq 1 ]] && args+=(--status)
            [[ -n "$SCORE_RESET" ]] && args+=(--reset "$SCORE_RESET")
            uv run --with-requirements "$REQS" python "$HERE/score.py" "${args[@]}"
        else
            score_run
        fi
        ;;
    smoke|run)
        choose_model
        auto_tune
        [[ $OFFLINE -eq 1 ]] && offline_preflight
        [[ $DO_SCORE -eq 1 && "$METHOD" == "local" ]] && { setup_docker || true; }
        start_backend
        run_instances
        cleanup; BACKEND_PID=""
        [[ $DO_SCORE -eq 1 ]] && score_run
        if [[ $DO_SCORE -eq 1 ]]; then
            success "Done. Outputs: $OUT (preds.jsonl, trajectories/, results.json)"
        else
            success "Done. Outputs: $OUT (preds.jsonl, trajectories/). Score later with: $HERE/bench.sh score --out $OUT"
        fi
        ;;
esac
