#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?missing target}"
COMMIT_SHA="${2:?missing commit sha}"
ORIGIN_URL="${3:?missing origin url}"
REMOTE_REPO_DIR="${4:?missing remote repo dir}"
REMOTE_LABELS_PATH="${5:?missing remote labels path}"
REMOTE_LOG_DIR="${6:?missing remote log dir}"
REMOTE_VENV_DIR="${7:?missing remote venv dir}"
RUN_ID="${8:?missing run id}"
RUN_TIMEOUT_SECONDS="${9:?missing run timeout seconds}"
shift 9
MAKE_ARGS=("$@")

mkdir -p "${REMOTE_LOG_DIR}"
exec > >(tee -a "${REMOTE_LOG_DIR}/run.log") 2>&1

STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
export HF_HOME="/workspace/.cache/huggingface"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export TRANSFORMERS_CACHE="${HF_HOME}"

cleanup_remote_inputs() {
  rm -f "${REMOTE_LABELS_PATH}" || true
  rmdir "$(dirname "${REMOTE_LABELS_PATH}")" 2>/dev/null || true
}

trap cleanup_remote_inputs EXIT

write_metadata() {
  local status="$1"
  local finished_at="$2"
  python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "target": os.environ["TARGET"],
    "commit_sha": os.environ["COMMIT_SHA"],
    "origin_url": os.environ["ORIGIN_URL"],
    "remote_repo_dir": os.environ["REMOTE_REPO_DIR"],
    "remote_labels_path": os.environ["REMOTE_LABELS_PATH"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "status": os.environ["STATUS"],
    "make_args": os.environ.get("MAKE_ARGS_SERIALIZED", "").split("\n") if os.environ.get("MAKE_ARGS_SERIALIZED") else [],
}
path = Path(os.environ["REMOTE_LOG_DIR"]) / "run_metadata.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

export TARGET COMMIT_SHA ORIGIN_URL REMOTE_REPO_DIR REMOTE_LABELS_PATH REMOTE_LOG_DIR STARTED_AT RUN_ID
export MAKE_ARGS_SERIALIZED="$(printf '%s\n' "${MAKE_ARGS[@]}")"
export RUN_TIMEOUT_SECONDS

if [[ -d "${REMOTE_REPO_DIR}" && ! -d "${REMOTE_REPO_DIR}/.git" ]]; then
  rm -rf "${REMOTE_REPO_DIR}"
fi

if [[ ! -d "${REMOTE_REPO_DIR}/.git" ]]; then
  git clone "${ORIGIN_URL}" "${REMOTE_REPO_DIR}"
fi

cd "${REMOTE_REPO_DIR}"
git fetch --all --prune
git checkout --detach "${COMMIT_SHA}"
git reset --hard "${COMMIT_SHA}"
git clean -fdx

if [[ -x "${REMOTE_VENV_DIR}/bin/python3" ]]; then
  if ! "${REMOTE_VENV_DIR}/bin/python3" - <<'PY' >/dev/null 2>&1
import sys

try:
    import torch
except Exception:
    sys.exit(1)

sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    rm -rf "${REMOTE_VENV_DIR}"
  fi
fi

if [[ ! -x "${REMOTE_VENV_DIR}/bin/python3" ]]; then
  python3 -m venv --system-site-packages "${REMOTE_VENV_DIR}"
fi

source "${REMOTE_VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install \
  "accelerate==1.13.0" \
  "datasets==4.8.4" \
  "peft==0.18.1" \
  "sentence-transformers==5.4.0" \
  "trl==1.1.0" \
  "transformers==4.56.2"
python - <<'PY'
import torch

print(
    "remote torch:",
    {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    },
)
PY

if ! python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false after bootstrap dependency setup")
x = torch.randn((64, 64), device="cuda")
y = x @ x
torch.cuda.synchronize()
print({"device": torch.cuda.get_device_name(0), "checksum": float(y.sum().item())})
PY
then
  STATUS="failed_gpu_smoke"
  FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  export STATUS FINISHED_AT
  write_metadata "${STATUS}" "${FINISHED_AT}"
  exit 1
fi

STATUS="running"
FINISHED_AT=""
export STATUS FINISHED_AT
write_metadata "${STATUS}" "${FINISHED_AT}"

command -v timeout >/dev/null 2>&1 || {
  STATUS="failed"
  FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  export STATUS FINISHED_AT
  write_metadata "${STATUS}" "${FINISHED_AT}"
  echo "missing required timeout command inside pod" >&2
  exit 1
}

if timeout --foreground --signal=TERM --kill-after=300 "${RUN_TIMEOUT_SECONDS}s" make "${TARGET}" "${MAKE_ARGS[@]}"; then
  STATUS="success"
else
  exit_code=$?
  if [[ "$exit_code" -eq 124 ]]; then
    STATUS="timed_out"
  else
    STATUS="failed"
  fi
  FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  export STATUS FINISHED_AT
  write_metadata "${STATUS}" "${FINISHED_AT}"
  exit 1
fi

FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
export STATUS="success" FINISHED_AT
write_metadata "${STATUS}" "${FINISHED_AT}"
