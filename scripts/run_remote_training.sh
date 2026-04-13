#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_remote_training.sh --host USER@WINDOWS_HOST [options]

Sync the current working tree and reviewed label file to a Windows 11 host, run a
chosen make target inside WSL, and optionally pull the resulting model artifacts
back to this machine.

Required:
  --host HOST                  SSH target for the Windows machine

Options:
  --target NAME               Make target to run remotely
                              Default: benchmark-suite
  --wsl-distro NAME           WSL distro name used on the remote host
                              Default: Ubuntu
  --remote-dir PATH           Absolute Linux path inside WSL for the repo clone
                              Default: /home/<wsl-user>/ask-seattle
  --labels PATH               Local reviewed label file to sync
                              Default: data/processed/tampermonkey_labels.jsonl
  --eval-subreddit NAME       Passed through as EVAL_SUBREDDIT
                              Default: seattle
  --split-strategy NAME       Passed through as SPLIT_STRATEGY
                              Default: random
  --split-seed INT            Passed through as SPLIT_SEED
                              Default: 13
  --make-arg KEY=VALUE        Extra make override. Can be repeated.
  --artifact-dir PATH         Artifact directory to pull back, relative to repo root
                              Default depends on the target
  --bootstrap                 Install required Ubuntu packages in WSL first
  --pull-artifacts            Pull remote artifacts back after the run
                              Default: enabled
  --no-pull-artifacts         Leave artifacts on the remote box only
  --install-model-deps        Force model extras even for non-suite targets
  --skip-model-deps           Skip semantic/transformer dependencies
  --torch-index-url URL       PyTorch wheel index URL for CUDA-enabled torch wheels
                              Example: https://download.pytorch.org/whl/cu128
  --run-timeout-seconds INT   Max remote target runtime before it is terminated
                              Default: 21600 (6 hours)
  -h, --help                  Show this help

Examples:
  scripts/run_remote_training.sh \
    --host gpu-win \
    --bootstrap \
    --target benchmark-suite \
    --eval-subreddit seattle \
    --torch-index-url https://download.pytorch.org/whl/cu128

  scripts/run_remote_training.sh \
    --host gpu-win \
    --target retrain \
    --eval-subreddit seattle
EOF
}

log() {
  printf '[remote-train] %s\n' "$*"
}

die() {
  printf '[remote-train] error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required local command: $1"
}

quote_shell() {
  printf '%q' "$1"
}

run_wsl_script() {
  local tty_flag="${1:-}"
  local script="${2:-}"
  local -a ssh_args=()
  if [[ "$tty_flag" == "--tty" ]]; then
    ssh_args=(-tt)
  else
    script="$tty_flag"
  fi
  ssh "${ssh_args[@]}" "$SSH_HOST" "wsl.exe -d \"$WSL_DISTRO\" -- bash -s --" <<EOF
set -euo pipefail
$script
EOF
}

ROOT_DIR=$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)

SSH_HOST=""
TARGET="benchmark-suite"
WSL_DISTRO="Ubuntu"
REMOTE_DIR=""
LABELS_PATH="$ROOT_DIR/data/processed/tampermonkey_labels.jsonl"
EVAL_SUBREDDIT="seattle"
SPLIT_STRATEGY="random"
SPLIT_SEED="13"
PULL_ARTIFACTS=1
BOOTSTRAP=0
FORCE_MODEL_DEPS=0
SKIP_MODEL_DEPS=0
TORCH_INDEX_URL=""
RUN_TIMEOUT_SECONDS="21600"
ARTIFACT_DIR_REL=""
MAKE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      SSH_HOST="${2:-}"
      shift 2
      ;;
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --wsl-distro)
      WSL_DISTRO="${2:-}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:-}"
      shift 2
      ;;
    --labels)
      LABELS_PATH="${2:-}"
      shift 2
      ;;
    --eval-subreddit)
      EVAL_SUBREDDIT="${2:-}"
      shift 2
      ;;
    --split-strategy)
      SPLIT_STRATEGY="${2:-}"
      shift 2
      ;;
    --split-seed)
      SPLIT_SEED="${2:-}"
      shift 2
      ;;
    --make-arg)
      MAKE_ARGS+=("${2:-}")
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_DIR_REL="${2:-}"
      shift 2
      ;;
    --bootstrap)
      BOOTSTRAP=1
      shift
      ;;
    --pull-artifacts)
      PULL_ARTIFACTS=1
      shift
      ;;
    --no-pull-artifacts)
      PULL_ARTIFACTS=0
      shift
      ;;
    --install-model-deps)
      FORCE_MODEL_DEPS=1
      shift
      ;;
    --skip-model-deps)
      SKIP_MODEL_DEPS=1
      shift
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:-}"
      shift 2
      ;;
    --run-timeout-seconds)
      RUN_TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$SSH_HOST" ]] || {
  usage
  die "--host is required"
}

case "$TARGET" in
  retrain|benchmark|benchmark-variants|benchmark-suite)
    ;;
  *)
    die "unsupported target: $TARGET"
    ;;
esac

case "$SPLIT_STRATEGY" in
  random|time)
    ;;
  *)
    die "unsupported split strategy: $SPLIT_STRATEGY"
    ;;
esac

require_command ssh
require_command rsync

[[ -f "$LABELS_PATH" ]] || die "reviewed label file not found: $LABELS_PATH"

if [[ -z "$REMOTE_DIR" ]]; then
  log "Detecting remote WSL home directory"
  REMOTE_HOME=$(run_wsl_script "printf '%s\n' \"\$HOME\"")
  REMOTE_DIR="$REMOTE_HOME/ask-seattle"
fi

REMOTE_RSYNC_PATH="wsl.exe -d \"$WSL_DISTRO\" -- rsync"
REMOTE_LABELS_REL="data/processed/$(basename "$LABELS_PATH")"
REMOTE_LABELS_PATH="$REMOTE_DIR/$REMOTE_LABELS_REL"

if [[ -z "$ARTIFACT_DIR_REL" ]]; then
  case "$TARGET" in
    retrain)
      ARTIFACT_DIR_REL="models/real-labels-precision-refresh"
      ;;
    benchmark)
      ARTIFACT_DIR_REL="models/benchmark-suite"
      ;;
    benchmark-variants)
      ARTIFACT_DIR_REL="models/benchmark-variants"
      ;;
    benchmark-suite)
      ARTIFACT_DIR_REL="models/benchmark-suite"
      ;;
  esac
fi

NEEDS_MODEL_DEPS=0
if [[ "$TARGET" == "benchmark-suite" || "$FORCE_MODEL_DEPS" -eq 1 ]]; then
  NEEDS_MODEL_DEPS=1
fi
if [[ "$SKIP_MODEL_DEPS" -eq 1 ]]; then
  NEEDS_MODEL_DEPS=0
fi

if [[ "$BOOTSTRAP" -eq 1 ]]; then
  log "Bootstrapping remote Ubuntu packages in $WSL_DISTRO"
  run_wsl_script --tty "sudo apt update && sudo apt install -y build-essential git make python3 python3-venv python3-pip rsync"
fi

log "Preparing remote directories under $REMOTE_DIR"
run_wsl_script "
mkdir -p $(quote_shell "$REMOTE_DIR")
mkdir -p $(quote_shell "$REMOTE_DIR/data/processed")
mkdir -p $(quote_shell "$REMOTE_DIR/models")
"

log "Syncing repository working tree to $SSH_HOST:$REMOTE_DIR"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'data/processed/' \
  --exclude 'models/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --rsync-path="$REMOTE_RSYNC_PATH" \
  "$ROOT_DIR/" "$SSH_HOST:$REMOTE_DIR/"

log "Syncing reviewed labels to $REMOTE_LABELS_PATH"
rsync -az \
  --rsync-path="$REMOTE_RSYNC_PATH" \
  "$LABELS_PATH" "$SSH_HOST:$REMOTE_LABELS_PATH"

log "Refreshing remote Python environment"
run_wsl_script "
cd $(quote_shell "$REMOTE_DIR")
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
if [[ $NEEDS_MODEL_DEPS -eq 1 ]]; then
  if [[ -n $(quote_shell "$TORCH_INDEX_URL") ]]; then
    python -m pip install torch torchvision torchaudio --index-url $(quote_shell "$TORCH_INDEX_URL")
  elif ! python -c 'import torch' >/dev/null 2>&1; then
    python -m pip install torch
  fi
  python -m pip install accelerate datasets sentence-transformers transformers
  python - <<'PY'
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({'torch_error': str(exc)}, indent=2))
else:
    print(
        json.dumps(
            {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count(),
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )
PY
fi
"

MAKE_CMD=(make "$TARGET" "LABELS=$REMOTE_LABELS_REL" "SPLIT_STRATEGY=$SPLIT_STRATEGY" "SPLIT_SEED=$SPLIT_SEED")
if [[ -n "$EVAL_SUBREDDIT" ]]; then
  MAKE_CMD+=("EVAL_SUBREDDIT=$EVAL_SUBREDDIT")
fi
for make_arg in "${MAKE_ARGS[@]}"; do
  MAKE_CMD+=("$make_arg")
done
printf -v MAKE_LINE '%q ' "${MAKE_CMD[@]}"

log "Running remote target: ${MAKE_CMD[*]}"
run_wsl_script "
cd $(quote_shell "$REMOTE_DIR")
source .venv/bin/activate
command -v timeout >/dev/null 2>&1 || { echo 'missing timeout command inside WSL' >&2; exit 1; }
timeout --foreground --signal=TERM --kill-after=300 $(quote_shell "${RUN_TIMEOUT_SECONDS}s") $MAKE_LINE
"

if [[ "$PULL_ARTIFACTS" -eq 1 ]]; then
  LOCAL_ARTIFACT_DIR="$ROOT_DIR/$ARTIFACT_DIR_REL"
  REMOTE_ARTIFACT_DIR="$REMOTE_DIR/$ARTIFACT_DIR_REL"
  mkdir -p "$(dirname "$LOCAL_ARTIFACT_DIR")"
  log "Pulling artifacts from $REMOTE_ARTIFACT_DIR"
  rsync -az \
    --rsync-path="$REMOTE_RSYNC_PATH" \
    "$SSH_HOST:$REMOTE_ARTIFACT_DIR/" "$LOCAL_ARTIFACT_DIR/"
fi

log "Remote run finished successfully"
