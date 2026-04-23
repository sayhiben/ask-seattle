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
ENV_CACHE_KEY="${10:?missing env cache key}"
shift 10
MAKE_ARGS=("$@")
ENV_STAMP_PATH="${REMOTE_VENV_DIR}/.ask-seattle-runpod-env.json"
REMOTE_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
REMOTE_TORCH_VERSION="2.9.1"

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

cleanup_remote_history() {
  if [[ "${TARGET}" != "benchmark" ]]; then
    rm -rf "${REMOTE_REPO_DIR}/models" || true
  fi
  rm -rf "${REMOTE_REPO_DIR}/data/processed" || true
  if [[ -d "/workspace/runpod-logs" ]]; then
    find /workspace/runpod-logs -mindepth 1 -maxdepth 1 -type d ! -name "${RUN_ID}" -exec rm -rf {} +
  fi
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

export TARGET COMMIT_SHA ORIGIN_URL REMOTE_REPO_DIR REMOTE_LABELS_PATH REMOTE_LOG_DIR STARTED_AT RUN_ID ENV_CACHE_KEY
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
if [[ "${TARGET}" == "benchmark" ]]; then
  git clean -fdx -e models/
else
  git clean -fdx
fi
cleanup_remote_history

cached_env_is_healthy() {
  if [[ ! -x "${REMOTE_VENV_DIR}/bin/python3" ]]; then
    return 1
  fi
  "${REMOTE_VENV_DIR}/bin/python3" - <<'PY' >/dev/null 2>&1
import re
import sys
from importlib import import_module


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


try:
    torch = import_module("torch")
except Exception:
    sys.exit(1)

version = _parse_major_minor(torch.__version__.split("+", 1)[0] or "")
if version is None or version < (2, 7):
    sys.exit(1)
cuda_version = _parse_major_minor(torch.version.cuda or "")
if cuda_version is None or cuda_version < (12, 8):
    sys.exit(1)
required = (
    "accelerate",
    "datasets",
    "google.protobuf",
    "peft",
    "sentence_transformers",
    "sentencepiece",
    "tiktoken",
    "trl",
    "transformers",
)
for module_name in required:
    try:
        import_module(module_name)
    except Exception:
        sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(1)
arch_list = set()
get_arch_list = getattr(torch.cuda, "get_arch_list", None)
if callable(get_arch_list):
    try:
        arch_list = set(get_arch_list())
    except Exception:
        arch_list = set()
device_capability = torch.cuda.get_device_capability(0)
required_arch = f"sm_{device_capability[0]}{device_capability[1]}"
if arch_list and required_arch not in arch_list:
    sys.exit(1)
sys.exit(0)
PY
}

stamp_matches() {
  if [[ ! -f "${ENV_STAMP_PATH}" ]]; then
    return 1
  fi
  CURRENT_ENV_CACHE_KEY="${ENV_CACHE_KEY}" ENV_STAMP_PATH="${ENV_STAMP_PATH}" python3 - <<'PY' >/dev/null 2>&1
import json
import os
from pathlib import Path

path = Path(os.environ["ENV_STAMP_PATH"])
current_key = os.environ["CURRENT_ENV_CACHE_KEY"]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("env_cache_key") == current_key else 1)
PY
}

write_env_stamp() {
  ENV_STAMP_PATH="${ENV_STAMP_PATH}" ENV_CACHE_KEY="${ENV_CACHE_KEY}" python3 - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

path = Path(os.environ["ENV_STAMP_PATH"])
path.write_text(
    json.dumps(
        {
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "env_cache_key": os.environ["ENV_CACHE_KEY"],
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
}

BOOTSTRAP_ENV=1
if stamp_matches && cached_env_is_healthy; then
  BOOTSTRAP_ENV=0
  echo "reusing cached RunPod environment (matching env cache key)"
elif [[ -x "${REMOTE_VENV_DIR}/bin/python3" ]]; then
  echo "cached RunPod environment is stale or unhealthy; rebuilding"
  rm -rf "${REMOTE_VENV_DIR}"
fi

if [[ "${BOOTSTRAP_ENV}" -eq 1 ]]; then
  if [[ ! -x "${REMOTE_VENV_DIR}/bin/python3" ]]; then
    python3 -m venv "${REMOTE_VENV_DIR}"
  fi
  source "${REMOTE_VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  if ! python - <<'PY'
import re
from importlib import import_module


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


try:
    torch = import_module("torch")
except Exception:
    raise SystemExit(1)

version = _parse_major_minor(torch.__version__.split("+", 1)[0] or "")
cuda_version = _parse_major_minor(torch.version.cuda or "")
raise SystemExit(0 if version is not None and version >= (2, 7) and cuda_version is not None and cuda_version >= (12, 8) else 1)
PY
  then
    python -m pip install --upgrade \
      --index-url "${REMOTE_TORCH_INDEX_URL}" \
      "torch==${REMOTE_TORCH_VERSION}"
  fi
  python -m pip install -e ".[dev]"
  python -m pip install \
    "accelerate==1.13.0" \
    "datasets==4.8.4" \
    "peft==0.18.1" \
    "protobuf>=5.0" \
    "sentence-transformers==5.4.0" \
    "sentencepiece>=0.2" \
    "tiktoken>=0.7" \
    "trl==1.1.0" \
    "transformers==4.56.2"
else
  source "${REMOTE_VENV_DIR}/bin/activate"
fi

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

if [[ "${BOOTSTRAP_ENV}" -eq 1 ]]; then
  write_env_stamp
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
