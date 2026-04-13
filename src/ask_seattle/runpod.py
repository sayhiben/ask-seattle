from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RUNPOD_REQUIRED_FEATURES: tuple[tuple[str, ...], ...] = (
    ("runpodctl", "ssh", "add-key", "--help"),
    ("runpodctl", "ssh", "list-keys", "--help"),
    ("runpodctl", "pod", "create", "--help"),
    ("runpodctl", "pod", "get", "--help"),
    ("runpodctl", "pod", "delete", "--help"),
    ("runpodctl", "network-volume", "list", "--help"),
    ("runpodctl", "network-volume", "create", "--help"),
    ("runpodctl", "datacenter", "list", "--help"),
)

REMOTE_LOG_ROOT = "/workspace/runpod-logs"
REMOTE_LABEL_ROOT = "/workspace/runpod-inputs"
REMOTE_VENV_DIR = "/workspace/.venv"
REMOTE_CACHE_ROOT = "/workspace/.cache"
RSYNC_FLAGS = ("-rlptz",)
DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
DEFAULT_RUNPOD_TEMPLATE_ID = "runpod-torch-v240"
DEFAULT_RUNPOD_VOLUME_MOUNT_PATH = "/workspace"
DEFAULT_RUNPOD_CONTAINER_DISK_GB = 50
DEFAULT_RUNPOD_VOLUME_SIZE_GB = 100
DEFAULT_RUNPOD_VOLUME_RETENTION_SECONDS = 3 * 24 * 60 * 60
DEFAULT_RUNPOD_POD_CREATE_ATTEMPTS = 3
DEFAULT_RUNPOD_POD_CREATE_RETRY_DELAY_SECONDS = 20
DEFAULT_RUNPOD_POD_CREATE_TIMEOUT_SECONDS = 300
DEFAULT_RUNPOD_POD_RECONCILE_TIMEOUT_SECONDS = 90
DEFAULT_RUNPOD_GPU_TYPES = (
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
)
DEFAULT_RUNPOD_DATA_CENTER_IDS = (
    "EU-RO-1",
    "US-NC-1",
    "US-KS-2",
    "US-IL-1",
    "US-GA-2",
)


class RunPodOrchestrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPodConfig:
    repo_root: Path
    repo_slug: str
    ssh_key_path: Path
    volume_name: str
    volume_size_gb: int
    volume_retention_seconds: int
    gpu_types: tuple[str, ...]
    data_center_ids: tuple[str, ...]
    template_id: str | None
    image: str
    remote_dir: str
    ssh_user: str
    container_disk_gb: int
    volume_mount_path: str
    labels_path: Path
    benchmark_meta_dir: Path
    split_strategy: str
    split_seed: int
    evaluation_subreddit: str | None
    benchmark_notes: str | None
    semantic_model_id: str
    semantic_secondary_model_id: str
    transformer_model_id: str
    transformer_secondary_model_id: str
    causal_lm_model_id: str
    no_pull_artifacts: bool = False
    remote_run_timeout_seconds: int = 21600
    pod_ready_timeout_seconds: int = 1800
    pod_create_attempts: int = DEFAULT_RUNPOD_POD_CREATE_ATTEMPTS
    pod_create_retry_delay_seconds: int = DEFAULT_RUNPOD_POD_CREATE_RETRY_DELAY_SECONDS
    pod_create_timeout_seconds: int = DEFAULT_RUNPOD_POD_CREATE_TIMEOUT_SECONDS
    evict_volume_on_capacity_failure: bool = False


@dataclass(frozen=True)
class NetworkVolume:
    volume_id: str
    name: str
    data_center_id: str
    size_gb: int


@dataclass(frozen=True)
class PodSshEndpoint:
    host: str
    port: int
    user: str


@dataclass(frozen=True)
class PodInfo:
    pod_id: str
    name: str
    desired_status: str
    ssh_endpoint: PodSshEndpoint | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runpod_train.py",
        description="Run ask-seattle training targets on an ephemeral RunPod Pod.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Bootstrap GitHub and RunPod prerequisites.")
    _add_common_arguments(bootstrap_parser, include_target=False)
    bootstrap_parser.set_defaults(func=bootstrap_command)

    cleanup_parser = subparsers.add_parser("cleanup", help="Delete the retained RunPod cache volume for this config.")
    _add_common_arguments(cleanup_parser, include_target=False)
    cleanup_parser.set_defaults(func=cleanup_command)

    run_parser = subparsers.add_parser("run", help="Run a make target on an ephemeral RunPod Pod.")
    _add_common_arguments(run_parser, include_target=True)
    run_parser.set_defaults(func=run_command)

    args = parser.parse_args(argv)
    return int(args.func(args))


def bootstrap_command(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    ensure_local_prerequisites()
    ensure_runpodctl_features()
    ensure_clean_worktree(config.repo_root)
    ensure_origin_remote(config.repo_root, config.repo_slug)
    push_current_head(config.repo_root)
    ensure_runpod_ssh_key(config.ssh_key_path)
    return 0


def cleanup_command(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    ensure_local_prerequisites()
    ensure_runpodctl_features()
    delete_cached_volume(config)
    return 0


def run_command(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    target = str(args.target)
    ensure_local_prerequisites()
    ensure_runpodctl_features()
    ensure_clean_worktree(config.repo_root)
    ensure_label_path_exists(config.labels_path)
    origin_url = ensure_origin_remote(config.repo_root, config.repo_slug)
    remote_origin_url = remote_clone_url(origin_url)
    ensure_runpod_ssh_key(config.ssh_key_path)
    commit_sha = push_current_head(config.repo_root)
    cleanup_expired_cached_volume(config)
    existing_volume_before_run = next((item for item in list_network_volumes() if item.name == config.volume_name), None)

    pod_name = build_pod_name(target=target, commit_sha=commit_sha)
    volume, gpu_id, data_center_id, pod = provision_volume_and_pod(config, pod_name=pod_name)
    run_id = build_run_id(target)
    log_dir = config.benchmark_meta_dir / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    local_metadata = {
        "run_id": run_id,
        "target": target,
        "commit_sha": commit_sha,
        "pod_id": pod.pod_id,
        "pod_name": pod.name,
        "gpu_id": gpu_id,
        "data_center_id": data_center_id,
        "network_volume_id": volume.volume_id,
        "repo_slug": config.repo_slug,
        "origin_url": origin_url,
        "remote_origin_url": remote_origin_url,
        "started_at": utc_now(),
    }
    write_json(log_dir / "run_metadata.local.json", local_metadata)

    ready_pod: PodInfo | None = None
    retain_volume = existing_volume_before_run is not None and existing_volume_before_run.volume_id == volume.volume_id
    cleanup_state: dict[str, Any] = {
        "pod_id": pod.pod_id,
        "volume_id": volume.volume_id,
        "delete_volume": existing_volume_before_run is None,
    }
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_signal(signum: int, frame: Any) -> None:
        cleanup_runpod_resources(
            pod_id=str(cleanup_state.get("pod_id") or ""),
            volume_id=str(cleanup_state.get("volume_id") or ""),
            delete_volume=bool(cleanup_state.get("delete_volume")),
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        ready_pod = wait_for_pod_ready(config, pod.pod_id, pod_name=pod.name)
        if ready_pod.ssh_endpoint is None:
            raise RunPodOrchestrationError(f"pod {pod.pod_id} never exposed an SSH endpoint")
        run_remote_gpu_smoke(ready_pod.ssh_endpoint)
        ensure_remote_rsync(ready_pod.ssh_endpoint)
        remote_labels_path = sync_labels_to_pod(config, ready_pod.ssh_endpoint, run_id)
        retain_volume = True
        cleanup_state["delete_volume"] = False
        run_remote_bootstrap(
            config,
            ssh_endpoint=ready_pod.ssh_endpoint,
            run_id=run_id,
            target=target,
            commit_sha=commit_sha,
            origin_url=remote_origin_url,
            remote_labels_path=remote_labels_path,
        )
        pull_remote_logs(config, ssh_endpoint=ready_pod.ssh_endpoint, run_id=run_id)
        if not config.no_pull_artifacts:
            pull_artifacts(config, ssh_endpoint=ready_pod.ssh_endpoint, target=target)
        local_metadata["finished_at"] = utc_now()
        local_metadata["status"] = "success"
        write_json(log_dir / "run_metadata.local.json", local_metadata)
        return 0
    except Exception:
        if ready_pod is not None and ready_pod.ssh_endpoint is not None:
            try:
                pull_remote_logs(config, ssh_endpoint=ready_pod.ssh_endpoint, run_id=run_id)
            except Exception:
                pass
        local_metadata["finished_at"] = utc_now()
        local_metadata["status"] = "failed"
        write_json(log_dir / "run_metadata.local.json", local_metadata)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        cleanup_runpod_resources(
            pod_id=pod.pod_id,
            volume_id=volume.volume_id,
            delete_volume=(not retain_volume and existing_volume_before_run is None),
        )
        if retain_volume:
            record_volume_retention(config, volume)
        elif existing_volume_before_run is None:
            delete_volume_lease(config)


def _add_common_arguments(parser: argparse.ArgumentParser, *, include_target: bool) -> None:
    parser.add_argument("--repo-root", default=".", help="Local repo root. Default: current directory.")
    parser.add_argument("--repo", required=True, help="GitHub repo slug, e.g. sayhiben/ask-seattle.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519.pub")
    parser.add_argument("--volume-name", required=True)
    parser.add_argument("--volume-size-gb", type=int, default=DEFAULT_RUNPOD_VOLUME_SIZE_GB)
    parser.add_argument("--volume-retention-seconds", type=int, default=DEFAULT_RUNPOD_VOLUME_RETENTION_SECONDS)
    parser.add_argument("--gpu-types", default=",".join(DEFAULT_RUNPOD_GPU_TYPES))
    parser.add_argument("--data-center-ids", default=",".join(DEFAULT_RUNPOD_DATA_CENTER_IDS))
    parser.add_argument("--template-id", default=DEFAULT_RUNPOD_TEMPLATE_ID)
    parser.add_argument("--image", default=DEFAULT_RUNPOD_IMAGE)
    parser.add_argument("--remote-dir", default="/workspace/ask-seattle")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--container-disk-gb", type=int, default=DEFAULT_RUNPOD_CONTAINER_DISK_GB)
    parser.add_argument("--volume-mount-path", default=DEFAULT_RUNPOD_VOLUME_MOUNT_PATH)
    parser.add_argument("--labels", default="data/processed/tampermonkey_labels.jsonl")
    parser.add_argument("--benchmark-meta-dir", default="models/runpod-meta")
    parser.add_argument("--split-strategy", default="random", choices=("random", "time"))
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--eval-subreddit")
    parser.add_argument("--benchmark-notes")
    parser.add_argument("--semantic-model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-secondary-model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--transformer-model-id", default="microsoft/deberta-v3-small")
    parser.add_argument("--transformer-secondary-model-id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--causal-lm-model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--no-pull-artifacts", action="store_true")
    parser.add_argument("--remote-run-timeout-seconds", type=int, default=21600)
    parser.add_argument("--pod-ready-timeout-seconds", type=int, default=1800)
    parser.add_argument("--pod-create-attempts", type=int, default=DEFAULT_RUNPOD_POD_CREATE_ATTEMPTS)
    parser.add_argument(
        "--pod-create-retry-delay-seconds",
        type=int,
        default=DEFAULT_RUNPOD_POD_CREATE_RETRY_DELAY_SECONDS,
    )
    parser.add_argument("--pod-create-timeout-seconds", type=int, default=DEFAULT_RUNPOD_POD_CREATE_TIMEOUT_SECONDS)
    parser.add_argument("--evict-volume-on-capacity-failure", action="store_true")
    if include_target:
        parser.add_argument(
            "--target",
            required=True,
            choices=("retrain", "benchmark", "benchmark-variants"),
        )


def config_from_args(args: argparse.Namespace) -> RunPodConfig:
    return RunPodConfig(
        repo_root=Path(args.repo_root).resolve(),
        repo_slug=str(args.repo),
        ssh_key_path=Path(os.path.expanduser(str(args.ssh_key_path))).resolve(),
        volume_name=str(args.volume_name),
        volume_size_gb=int(args.volume_size_gb),
        volume_retention_seconds=int(args.volume_retention_seconds),
        gpu_types=_comma_list(args.gpu_types),
        data_center_ids=_comma_list(args.data_center_ids),
        template_id=str(args.template_id).strip() or None,
        image=str(args.image),
        remote_dir=str(args.remote_dir),
        ssh_user=str(args.ssh_user),
        container_disk_gb=int(args.container_disk_gb),
        volume_mount_path=str(args.volume_mount_path),
        labels_path=Path(args.labels).resolve(),
        benchmark_meta_dir=Path(args.benchmark_meta_dir).resolve(),
        split_strategy=str(args.split_strategy),
        split_seed=int(args.split_seed),
        evaluation_subreddit=str(args.eval_subreddit) if args.eval_subreddit else None,
        benchmark_notes=str(args.benchmark_notes) if args.benchmark_notes else None,
        semantic_model_id=str(args.semantic_model_id),
        semantic_secondary_model_id=str(args.semantic_secondary_model_id),
        transformer_model_id=str(args.transformer_model_id),
        transformer_secondary_model_id=str(args.transformer_secondary_model_id),
        causal_lm_model_id=str(args.causal_lm_model_id),
        no_pull_artifacts=bool(args.no_pull_artifacts),
        remote_run_timeout_seconds=int(args.remote_run_timeout_seconds),
        pod_ready_timeout_seconds=int(args.pod_ready_timeout_seconds),
        pod_create_attempts=int(args.pod_create_attempts),
        pod_create_retry_delay_seconds=int(args.pod_create_retry_delay_seconds),
        pod_create_timeout_seconds=int(args.pod_create_timeout_seconds),
        evict_volume_on_capacity_failure=bool(args.evict_volume_on_capacity_failure),
    )


def _comma_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def ensure_local_prerequisites() -> None:
    for command in ("gh", "git", "runpodctl", "ssh", "rsync"):
        if shutil.which(command) is None:
            raise RunPodOrchestrationError(f"missing required local command: {command}")


def ensure_runpodctl_features() -> None:
    for command in RUNPOD_REQUIRED_FEATURES:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RunPodOrchestrationError(
                "runpodctl is missing required Pod/network-volume commands. "
                "Upgrade it first with `runpodctl update` and verify the current docs-aligned command surface."
            ) from exc


def ensure_clean_worktree(repo_root: Path) -> None:
    result = run_command_capture(("git", "status", "--short"), cwd=repo_root)
    if result.strip():
        raise RunPodOrchestrationError(
            "remote RunPod execution requires a clean working tree so the Pod can train from an exact pushed commit"
        )


def ensure_origin_remote(repo_root: Path, repo_slug: str) -> str:
    origin_url = run_command_capture(("git", "remote", "get-url", "origin"), cwd=repo_root, check=False).strip()
    if origin_url:
        return origin_url

    repo_view = subprocess.run(
        ("gh", "repo", "view", repo_slug, "--json", "sshUrl"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if repo_view.returncode == 0:
        repo_info = json.loads(repo_view.stdout or "{}")
        ssh_url = str(repo_info.get("sshUrl") or "").strip()
        if not ssh_url:
            raise RunPodOrchestrationError(f"failed to resolve sshUrl for GitHub repo {repo_slug}")
        _run_subprocess(("git", "remote", "add", "origin", ssh_url), cwd=repo_root)
        return ssh_url

    _run_subprocess(("gh", "repo", "create", repo_slug, "--public", "--source=.", "--remote=origin"), cwd=repo_root)
    ssh_url = run_command_capture(("git", "remote", "get-url", "origin"), cwd=repo_root).strip()
    if not ssh_url:
        raise RunPodOrchestrationError("origin remote was not configured after gh repo create")
    return ssh_url


def push_current_head(repo_root: Path) -> str:
    commit_sha = run_command_capture(("git", "rev-parse", "HEAD"), cwd=repo_root).strip()
    _run_subprocess(("git", "push", "-u", "origin", "HEAD"), cwd=repo_root)
    return commit_sha


def ensure_runpod_ssh_key(ssh_key_path: Path) -> None:
    if not ssh_key_path.exists():
        raise RunPodOrchestrationError(f"RunPod SSH public key not found: {ssh_key_path}")
    public_key = ssh_key_path.read_text(encoding="utf-8").strip()
    listed = run_command_capture(("runpodctl", "ssh", "list-keys"))
    if public_key and public_key in listed:
        return
    _run_subprocess(("runpodctl", "ssh", "add-key", "--key-file", str(ssh_key_path)))


def ensure_label_path_exists(label_path: Path) -> None:
    if not label_path.exists():
        raise RunPodOrchestrationError(f"reviewed label file not found: {label_path}")


def provision_volume_and_pod(config: RunPodConfig, *, pod_name: str) -> tuple[NetworkVolume, str, str, PodInfo]:
    last_error: Exception | None = None
    for attempt_index in range(config.pod_create_attempts):
        datacenters = list_datacenters()
        existing_volume = next((item for item in list_network_volumes() if item.name == config.volume_name), None)
        if existing_volume is not None:
            try:
                gpu_id, pod = create_pod_in_datacenter(
                    config,
                    datacenters=datacenters,
                    pod_name=pod_name,
                    data_center_id=existing_volume.data_center_id,
                    network_volume_id=existing_volume.volume_id,
                )
                return existing_volume, gpu_id, existing_volume.data_center_id, pod
            except subprocess.CalledProcessError as exc:
                if not is_retryable_pod_create_error(exc):
                    raise
                if not config.evict_volume_on_capacity_failure:
                    raise RunPodOrchestrationError(
                        "cached RunPod volume is pinned to a datacenter that could not allocate the requested GPU. "
                        "The cache volume was preserved to honor the retention policy. "
                        "Retry later or rerun with volume eviction enabled to relocate it. "
                        f"Provider output: {summarize_called_process_error(exc)}"
                    ) from exc
                delete_network_volume(existing_volume.volume_id)
                delete_volume_lease(config)
                last_error = exc

        for data_center_id in candidate_datacenters(datacenters, config.gpu_types, config.data_center_ids):
            volume = create_network_volume(config.volume_name, config.volume_size_gb, data_center_id)
            try:
                gpu_id, pod = create_pod_in_datacenter(
                    config,
                    datacenters=datacenters,
                    pod_name=pod_name,
                    data_center_id=data_center_id,
                    network_volume_id=volume.volume_id,
                )
                return volume, gpu_id, data_center_id, pod
            except subprocess.CalledProcessError as exc:
                delete_network_volume(volume.volume_id)
                last_error = exc
                if not is_retryable_pod_create_error(exc):
                    raise
            except Exception:
                delete_network_volume(volume.volume_id)
                raise

        if attempt_index + 1 < config.pod_create_attempts:
            time.sleep(config.pod_create_retry_delay_seconds)

    if last_error is not None:
        raise RunPodOrchestrationError(
            "no acceptable RunPod datacenter could create a pod with the requested GPU preferences after retries. "
            f"Last provider output: {summarize_exception(last_error)}"
        ) from last_error
    raise RunPodOrchestrationError(
        "no acceptable RunPod datacenter had the requested GPU availability for the configured preference list"
    )


def create_pod_in_datacenter(
    config: RunPodConfig,
    *,
    datacenters: list[dict[str, Any]],
    pod_name: str,
    data_center_id: str,
    network_volume_id: str,
) -> tuple[str, PodInfo]:
    gpu_ids = available_gpus_for_datacenter(
        datacenters,
        data_center_id=data_center_id,
        preferred_gpu_ids=config.gpu_types,
    )
    if not gpu_ids:
        raise RunPodOrchestrationError(
            f"no requested GPU type was available in datacenter {data_center_id}"
        )
    last_error: subprocess.CalledProcessError | None = None
    for gpu_id in gpu_ids:
        try:
            pod = create_pod(
                config,
                pod_name=pod_name,
                gpu_id=gpu_id,
                data_center_id=data_center_id,
                network_volume_id=network_volume_id,
            )
            return gpu_id, pod
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if not is_retryable_pod_create_error(exc):
                raise RunPodOrchestrationError(
                    f"RunPod could not create a pod in {data_center_id} with GPU {gpu_id}. "
                    f"Provider output: {summarize_called_process_error(exc)}"
                ) from exc
    if last_error is not None:
        raise last_error
    raise RunPodOrchestrationError(
        f"no requested GPU type could be provisioned in datacenter {data_center_id}"
    )


def create_pod(
    config: RunPodConfig,
    *,
    pod_name: str,
    gpu_id: str,
    data_center_id: str,
    network_volume_id: str,
) -> PodInfo:
    command = build_create_pod_command(
        config,
        pod_name=pod_name,
        gpu_id=gpu_id,
        data_center_id=data_center_id,
        network_volume_id=network_volume_id,
    )
    try:
        output = run_command_capture(command, timeout=config.pod_create_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        reconciled = reconcile_pod_by_name(
            pod_name,
            wait_timeout_seconds=DEFAULT_RUNPOD_POD_RECONCILE_TIMEOUT_SECONDS,
        )
        if reconciled is not None:
            return reconciled
        raise RunPodOrchestrationError(
            f"runpodctl pod create timed out after {config.pod_create_timeout_seconds} seconds for pod {pod_name} "
            f"in {data_center_id} with GPU {gpu_id}, and no matching pod could be reconciled afterward"
        ) from exc
    pod = parse_pod_info(json.loads(output))
    if not pod.pod_id:
        raise RunPodOrchestrationError("failed to parse pod id from runpodctl pod create output")
    return pod


def wait_for_pod_ready(config: RunPodConfig, pod_id: str, *, pod_name: str) -> PodInfo:
    deadline = time.monotonic() + config.pod_ready_timeout_seconds
    current_pod_id = pod_id
    while time.monotonic() < deadline:
        try:
            pod = get_pod(current_pod_id)
        except RunPodOrchestrationError:
            reconciled = reconcile_pod_by_name(pod_name, wait_timeout_seconds=15)
            if reconciled is None or not reconciled.pod_id:
                time.sleep(5)
                continue
            current_pod_id = reconciled.pod_id
            pod = reconciled
        if pod.ssh_endpoint is not None and pod.desired_status.upper() == "RUNNING":
            wait_for_ssh(pod.ssh_endpoint)
            return pod
        time.sleep(10)
    raise RunPodOrchestrationError(f"pod {pod_name} ({current_pod_id}) did not become SSH-ready before timeout")


def run_remote_gpu_smoke(ssh_endpoint: PodSshEndpoint) -> None:
    command = (
        "set -euo pipefail\n"
        "command -v nvidia-smi >/dev/null 2>&1\n"
        "nvidia-smi -L\n"
        "python3 - <<'PY'\n"
        "import json\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('torch.cuda.is_available() is false during RunPod GPU smoke test')\n"
        "x = torch.randn((64, 64), device='cuda')\n"
        "y = x @ x\n"
        "torch.cuda.synchronize()\n"
        "print(json.dumps({'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(0), 'checksum': float(y.sum().item())}, indent=2))\n"
        "PY"
    )
    try:
        _run_subprocess(
            (
                "ssh",
                "-p",
                str(ssh_endpoint.port),
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                f"{ssh_endpoint.user}@{ssh_endpoint.host}",
                f"bash -lc {shlex.quote(command)}",
            ),
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunPodOrchestrationError("RunPod GPU smoke test timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise RunPodOrchestrationError(
            "RunPod pod failed the GPU smoke test before training. "
            "This usually means the selected template/image or provider runtime did not expose CUDA correctly."
        ) from exc


def ensure_remote_rsync(ssh_endpoint: PodSshEndpoint) -> None:
    command = (
        "set -euo pipefail\n"
        "if ! command -v rsync >/dev/null 2>&1; then\n"
        "  export DEBIAN_FRONTEND=noninteractive\n"
        "  apt-get update\n"
        "  apt-get install -y rsync\n"
        "fi\n"
        "command -v rsync >/dev/null 2>&1\n"
    )
    try:
        _run_subprocess(
            (
                "ssh",
                "-p",
                str(ssh_endpoint.port),
                "-o",
                "StrictHostKeyChecking=no",
                f"{ssh_endpoint.user}@{ssh_endpoint.host}",
                f"bash -lc {shlex.quote(command)}",
            ),
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunPodOrchestrationError("timed out while preparing rsync inside the RunPod pod") from exc
    except subprocess.CalledProcessError as exc:
        raise RunPodOrchestrationError("failed to install or verify rsync inside the RunPod pod") from exc


def sync_labels_to_pod(config: RunPodConfig, ssh_endpoint: PodSshEndpoint, run_id: str) -> str:
    remote_labels_dir = f"{REMOTE_LABEL_ROOT}/{run_id}"
    remote_labels_path = f"{remote_labels_dir}/{config.labels_path.name}"
    remote_target = f"{ssh_endpoint.user}@{ssh_endpoint.host}:{remote_labels_path}"
    _run_subprocess(
        (
            "ssh",
            "-p",
            str(ssh_endpoint.port),
            "-o",
            "StrictHostKeyChecking=no",
            f"{ssh_endpoint.user}@{ssh_endpoint.host}",
            f"mkdir -p {shlex.quote(remote_labels_dir)}",
        )
    )
    _run_subprocess(
        (
            "rsync",
            *RSYNC_FLAGS,
            "-e",
            f"ssh -p {ssh_endpoint.port} -o StrictHostKeyChecking=no",
            str(config.labels_path),
            remote_target,
        )
    )
    return remote_labels_path


def run_remote_bootstrap(
    config: RunPodConfig,
    *,
    ssh_endpoint: PodSshEndpoint,
    run_id: str,
    target: str,
    commit_sha: str,
    origin_url: str,
    remote_labels_path: str,
) -> None:
    remote_log_dir = f"{REMOTE_LOG_ROOT}/{run_id}"
    remote_script = f"{REMOTE_CACHE_ROOT}/runpod-bootstrap/{run_id}/runpod_pod_bootstrap.sh"
    sync_remote_bootstrap_script(config, ssh_endpoint=ssh_endpoint, remote_script=remote_script)
    make_args = build_remote_make_args(config, target=target, remote_labels_path=remote_labels_path)
    remote_command = build_remote_bootstrap_command(
        remote_script=remote_script,
        target=target,
        commit_sha=commit_sha,
        origin_url=origin_url,
        remote_repo_dir=config.remote_dir,
        remote_labels_path=remote_labels_path,
        remote_log_dir=remote_log_dir,
        remote_venv_dir=REMOTE_VENV_DIR,
        run_id=run_id,
        run_timeout_seconds=config.remote_run_timeout_seconds,
        make_args=make_args,
    )
    command = [
        "ssh",
        "-p",
        str(ssh_endpoint.port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        f"{ssh_endpoint.user}@{ssh_endpoint.host}",
        f"bash -lc {shlex.quote(remote_command)}",
    ]
    try:
        _run_subprocess(tuple(command), timeout=config.remote_run_timeout_seconds + 900)
    except subprocess.TimeoutExpired as exc:
        raise RunPodOrchestrationError(
            f"remote RunPod target exceeded timeout after {config.remote_run_timeout_seconds} seconds"
        ) from exc


def sync_remote_bootstrap_script(
    config: RunPodConfig,
    *,
    ssh_endpoint: PodSshEndpoint,
    remote_script: str,
) -> None:
    remote_parent = str(Path(remote_script).parent)
    _run_subprocess(
        (
            "ssh",
            "-p",
            str(ssh_endpoint.port),
            "-o",
            "StrictHostKeyChecking=no",
            f"{ssh_endpoint.user}@{ssh_endpoint.host}",
            f"mkdir -p {shlex.quote(remote_parent)}",
        )
    )
    _run_subprocess(
        (
            "rsync",
            *RSYNC_FLAGS,
            "-e",
            f"ssh -p {ssh_endpoint.port} -o StrictHostKeyChecking=no",
            str(config.repo_root / "scripts" / "runpod_pod_bootstrap.sh"),
            f"{ssh_endpoint.user}@{ssh_endpoint.host}:{remote_script}",
        )
    )
    _run_subprocess(
        (
            "ssh",
            "-p",
            str(ssh_endpoint.port),
            "-o",
            "StrictHostKeyChecking=no",
            f"{ssh_endpoint.user}@{ssh_endpoint.host}",
            f"chmod +x {shlex.quote(remote_script)}",
        )
    )


def pull_artifacts(config: RunPodConfig, *, ssh_endpoint: PodSshEndpoint, target: str) -> None:
    for relative_dir in artifact_dirs_for_target(target):
        local_dir = config.repo_root / relative_dir
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        remote_path = f"{config.remote_dir}/{relative_dir}"
        if not remote_directory_exists(ssh_endpoint=ssh_endpoint, remote_path=remote_path):
            raise RunPodOrchestrationError(
                f"remote artifact directory missing after {target}: {remote_path}"
            )
        remote_dir = f"{ssh_endpoint.user}@{ssh_endpoint.host}:{remote_path}/"
        _run_subprocess(
            (
                "rsync",
                *RSYNC_FLAGS,
                "--delete",
                "-e",
                f"ssh -p {ssh_endpoint.port} -o StrictHostKeyChecking=no",
                remote_dir,
                str(local_dir.parent / Path(relative_dir).name),
            )
        )


def pull_remote_logs(config: RunPodConfig, *, ssh_endpoint: PodSshEndpoint, run_id: str) -> None:
    local_dir = config.benchmark_meta_dir / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{ssh_endpoint.user}@{ssh_endpoint.host}:{REMOTE_LOG_ROOT}/{run_id}/"
    _run_subprocess(
        (
            "rsync",
            *RSYNC_FLAGS,
            "-e",
            f"ssh -p {ssh_endpoint.port} -o StrictHostKeyChecking=no",
            remote_dir,
            str(local_dir),
        ),
        check=False,
    )


def remote_directory_exists(*, ssh_endpoint: PodSshEndpoint, remote_path: str) -> bool:
    result = subprocess.run(
        (
            "ssh",
            "-p",
            str(ssh_endpoint.port),
            "-o",
            "StrictHostKeyChecking=no",
            f"{ssh_endpoint.user}@{ssh_endpoint.host}",
            f"test -d {shlex.quote(remote_path)}",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def delete_pod(pod_id: str) -> None:
    if not pod_id:
        return
    subprocess.run(("runpodctl", "pod", "delete", pod_id), check=False, capture_output=True, text=True)


def artifact_dirs_for_target(target: str) -> tuple[str, ...]:
    if target == "retrain":
        return (
            "models/real-labels-precision-refresh",
            "models/benchmark-suite",
        )
    if target == "benchmark":
        return ("models/benchmark-suite",)
    if target == "benchmark-variants":
        return ("models/benchmark-variants",)
    raise RunPodOrchestrationError(f"unsupported RunPod target: {target}")


def build_remote_make_args(config: RunPodConfig, *, target: str, remote_labels_path: str) -> tuple[str, ...]:
    args = (
        f"LABELS={remote_labels_path}",
        f"SPLIT_STRATEGY={config.split_strategy}",
        f"SPLIT_SEED={config.split_seed}",
        f"SEMANTIC_MODEL_ID={config.semantic_model_id}",
        f"SEMANTIC_SECONDARY_MODEL_ID={config.semantic_secondary_model_id}",
        f"TRANSFORMER_MODEL_ID={config.transformer_model_id}",
        f"TRANSFORMER_SECONDARY_MODEL_ID={config.transformer_secondary_model_id}",
        f"CAUSAL_LM_MODEL_ID={config.causal_lm_model_id}",
    )
    extra: list[str] = list(args)
    if config.evaluation_subreddit:
        extra.append(f"EVAL_SUBREDDIT={config.evaluation_subreddit}")
    if target == "benchmark" and config.benchmark_notes:
        extra.append(f"BENCHMARK_NOTES={config.benchmark_notes}")
    return tuple(extra)


def build_remote_bootstrap_command(
    *,
    remote_script: str,
    target: str,
    commit_sha: str,
    origin_url: str,
    remote_repo_dir: str,
    remote_labels_path: str,
    remote_log_dir: str,
    remote_venv_dir: str,
    run_id: str,
    run_timeout_seconds: int,
    make_args: tuple[str, ...],
) -> str:
    command = [
        "bash",
        shlex.quote(remote_script),
        shlex.quote(target),
        shlex.quote(commit_sha),
        shlex.quote(origin_url),
        shlex.quote(remote_repo_dir),
        shlex.quote(remote_labels_path),
        shlex.quote(remote_log_dir),
        shlex.quote(remote_venv_dir),
        shlex.quote(run_id),
        shlex.quote(str(run_timeout_seconds)),
    ]
    command.extend(shlex.quote(item) for item in make_args)
    return " ".join(command)


def build_create_pod_command(
    config: RunPodConfig,
    *,
    pod_name: str,
    gpu_id: str,
    data_center_id: str,
    network_volume_id: str,
) -> tuple[str, ...]:
    command: list[str] = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        pod_name,
    ]
    if config.template_id:
        command.extend(("--template-id", config.template_id))
    else:
        command.extend(("--image", config.image))
    command.extend(
        (
            "--gpu-id",
            gpu_id,
            "--gpu-count",
            "1",
            "--container-disk-in-gb",
            str(config.container_disk_gb),
            "--ports",
            "22/tcp",
            "--cloud-type",
            "SECURE",
            "--global-networking",
            "--ssh",
            "--data-center-ids",
            data_center_id,
            "--network-volume-id",
            network_volume_id,
            "--volume-mount-path",
            config.volume_mount_path,
        )
    )
    return tuple(command)


def build_run_id(target: str) -> str:
    return f"{target}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def build_pod_name(*, target: str, commit_sha: str) -> str:
    return f"ask-seattle-{target}-{commit_sha[:7]}"


def wait_for_ssh(ssh_endpoint: PodSshEndpoint) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = subprocess.run(
            (
                "ssh",
                "-p",
                str(ssh_endpoint.port),
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                f"{ssh_endpoint.user}@{ssh_endpoint.host}",
                "true",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(5)
    raise RunPodOrchestrationError(
        f"pod at {ssh_endpoint.user}@{ssh_endpoint.host}:{ssh_endpoint.port} did not accept SSH before timeout"
    )


def list_datacenters() -> list[dict[str, Any]]:
    output = run_command_capture(("runpodctl", "datacenter", "list"))
    payload = json.loads(output or "[]")
    if not isinstance(payload, list):
        raise RunPodOrchestrationError("unexpected runpodctl datacenter list output")
    return payload


def list_network_volumes() -> list[NetworkVolume]:
    output = run_command_capture(("runpodctl", "network-volume", "list"))
    payload = json.loads(output or "[]")
    if not isinstance(payload, list):
        raise RunPodOrchestrationError("unexpected runpodctl network-volume list output")
    volumes: list[NetworkVolume] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        volume_id = str(item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        data_center_id = str(item.get("dataCenterId") or item.get("data_center_id") or "").strip()
        size_gb = int(item.get("size") or 0)
        if volume_id and name and data_center_id:
            volumes.append(
                NetworkVolume(
                    volume_id=volume_id,
                    name=name,
                    data_center_id=data_center_id,
                    size_gb=size_gb,
                )
            )
    return volumes


def list_pods(*, name: str | None = None, include_all: bool = False) -> list[PodInfo]:
    command: list[str] = ["runpodctl", "pod", "list"]
    if include_all:
        command.append("--all")
    if name:
        command.extend(("--name", name))
    output = run_command_capture(tuple(command), timeout=30)
    payload = json.loads(output or "[]")
    if not isinstance(payload, list):
        raise RunPodOrchestrationError("unexpected runpodctl pod list output")
    pods: list[PodInfo] = []
    for item in payload:
        if isinstance(item, dict):
            pods.append(parse_pod_info(item))
    return pods


def reconcile_pod_by_name(pod_name: str, *, wait_timeout_seconds: int) -> PodInfo | None:
    deadline = time.monotonic() + wait_timeout_seconds
    while time.monotonic() < deadline:
        candidates = [
            pod
            for pod in list_pods(name=pod_name, include_all=True)
            if pod.name == pod_name and pod.pod_id
        ]
        candidates.sort(key=lambda pod: (pod.desired_status.upper() != "RUNNING", pod.pod_id))
        for candidate in candidates:
            try:
                return get_pod(candidate.pod_id)
            except RunPodOrchestrationError:
                continue
        if candidates:
            return candidates[0]
        time.sleep(5)
    return None


def create_network_volume(name: str, size_gb: int, data_center_id: str) -> NetworkVolume:
    output = run_command_capture(
        (
            "runpodctl",
            "network-volume",
            "create",
            "--name",
            name,
            "--size",
            str(size_gb),
            "--data-center-id",
            data_center_id,
        )
    )
    payload = json.loads(output or "{}")
    if not isinstance(payload, dict):
        raise RunPodOrchestrationError("unexpected runpodctl network-volume create output")
    volume_id = str(payload.get("id") or "").strip()
    if not volume_id:
        raise RunPodOrchestrationError("failed to parse network volume id from create output")
    return NetworkVolume(
        volume_id=volume_id,
        name=str(payload.get("name") or name),
        data_center_id=str(payload.get("dataCenterId") or data_center_id),
        size_gb=int(payload.get("size") or size_gb),
    )


def delete_network_volume(volume_id: str) -> None:
    if not volume_id:
        return
    subprocess.run(("runpodctl", "network-volume", "delete", volume_id), check=False, capture_output=True, text=True)


def cleanup_runpod_resources(*, pod_id: str, volume_id: str, delete_volume: bool) -> None:
    delete_pod(pod_id)
    if delete_volume:
        delete_network_volume(volume_id)


def cleanup_expired_cached_volume(config: RunPodConfig) -> None:
    existing_volume = next((item for item in list_network_volumes() if item.name == config.volume_name), None)
    lease = load_volume_lease(config)
    if existing_volume is None:
        if lease is not None:
            delete_volume_lease(config)
        return
    if lease is None:
        record_volume_retention(config, existing_volume)
        return
    expires_at = parse_utc_timestamp(str(lease.get("expires_at") or ""))
    if expires_at is None:
        record_volume_retention(config, existing_volume)
        return
    if datetime.now(UTC) >= expires_at:
        delete_network_volume(existing_volume.volume_id)
        delete_volume_lease(config)


def delete_cached_volume(config: RunPodConfig) -> None:
    existing_volume = next((item for item in list_network_volumes() if item.name == config.volume_name), None)
    if existing_volume is not None:
        delete_network_volume(existing_volume.volume_id)
    delete_volume_lease(config)


def record_volume_retention(config: RunPodConfig, volume: NetworkVolume) -> None:
    expires_at = datetime.now(UTC).timestamp() + config.volume_retention_seconds
    write_json(
        volume_lease_path(config),
        {
            "data_center_id": volume.data_center_id,
            "expires_at": utc_timestamp(expires_at),
            "last_used_at": utc_now(),
            "retention_seconds": config.volume_retention_seconds,
            "size_gb": volume.size_gb,
            "volume_id": volume.volume_id,
            "volume_name": volume.name,
        },
    )


def load_volume_lease(config: RunPodConfig) -> dict[str, Any] | None:
    path = volume_lease_path(config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def delete_volume_lease(config: RunPodConfig) -> None:
    path = volume_lease_path(config)
    if path.exists():
        path.unlink()


def volume_lease_path(config: RunPodConfig) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.volume_name).strip("._") or "default"
    return config.benchmark_meta_dir / "volumes" / f"{slug}.json"


def candidate_datacenters(
    datacenters: list[dict[str, Any]],
    preferred_gpu_ids: tuple[str, ...],
    candidate_data_center_ids: tuple[str, ...],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for data_center_id in candidate_data_center_ids:
        if available_gpus_for_datacenter(
            datacenters,
            data_center_id=data_center_id,
            preferred_gpu_ids=preferred_gpu_ids,
        ):
            ordered.append(data_center_id)
    return tuple(ordered)


def select_datacenter(
    datacenters: list[dict[str, Any]],
    preferred_gpu_ids: tuple[str, ...],
    candidate_data_center_ids: tuple[str, ...],
) -> str | None:
    candidate_set = set(candidate_data_center_ids)
    ordered_datacenters = [dc for dc in datacenters if str(dc.get("id") or dc.get("name") or "") in candidate_set]
    ordered_datacenters.sort(
        key=lambda dc: candidate_data_center_ids.index(str(dc.get("id") or dc.get("name") or ""))
    )
    for gpu_id in preferred_gpu_ids:
        for datacenter in ordered_datacenters:
            if datacenter_has_gpu(datacenter, gpu_id):
                return str(datacenter.get("id") or datacenter.get("name"))
    return None


def first_available_gpu_for_datacenter(
    datacenters: list[dict[str, Any]],
    *,
    data_center_id: str,
    preferred_gpu_ids: tuple[str, ...],
) -> str | None:
    datacenter = next(
        (item for item in datacenters if str(item.get("id") or item.get("name") or "") == data_center_id),
        None,
    )
    if datacenter is None:
        return None
    for gpu_id in preferred_gpu_ids:
        if datacenter_has_gpu(datacenter, gpu_id):
            return gpu_id
    return None


def available_gpus_for_datacenter(
    datacenters: list[dict[str, Any]],
    *,
    data_center_id: str,
    preferred_gpu_ids: tuple[str, ...],
) -> tuple[str, ...]:
    datacenter = next(
        (item for item in datacenters if str(item.get("id") or item.get("name") or "") == data_center_id),
        None,
    )
    if datacenter is None:
        return ()
    return tuple(gpu_id for gpu_id in preferred_gpu_ids if datacenter_has_gpu(datacenter, gpu_id))


def is_retryable_pod_create_error(exc: subprocess.CalledProcessError) -> bool:
    text = f"{exc.stdout}\n{exc.stderr}".lower()
    return any(
        phrase in text
        for phrase in (
            "no longer any instances available",
            "requested specifications",
            "please refresh and try again",
        )
    )


def datacenter_has_gpu(datacenter: dict[str, Any], gpu_id: str) -> bool:
    availability = datacenter.get("gpuAvailability")
    if not isinstance(availability, list):
        return False
    for item in availability:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("gpuId") or item.get("id") or "").strip()
        if candidate_id != gpu_id:
            continue
        stock_status = str(item.get("stockStatus") or "").strip().lower()
        available_flag = item.get("available")
        if isinstance(available_flag, bool):
            return available_flag
        return stock_status not in {"unavailable", "out", "outofstock", "none"}
    return False


def get_pod(pod_id: str) -> PodInfo:
    output = run_command_capture(("runpodctl", "pod", "get", pod_id))
    return parse_pod_info(json.loads(output or "{}"))


def parse_pod_info(payload: dict[str, Any]) -> PodInfo:
    if not isinstance(payload, dict):
        raise RunPodOrchestrationError("unexpected pod payload type")
    pod_id = str(payload.get("id") or payload.get("podId") or "").strip()
    name = str(payload.get("name") or "").strip()
    desired_status = str(payload.get("desiredStatus") or payload.get("status") or "").strip() or "UNKNOWN"
    ssh_endpoint = extract_ssh_endpoint(payload)
    return PodInfo(pod_id=pod_id, name=name, desired_status=desired_status, ssh_endpoint=ssh_endpoint)


def extract_ssh_endpoint(payload: dict[str, Any]) -> PodSshEndpoint | None:
    ssh = payload.get("ssh") if isinstance(payload.get("ssh"), dict) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    host = str(
        ssh.get("ip")
        or runtime.get("publicIp")
        or payload.get("publicIp")
        or payload.get("sshHost")
        or payload.get("ipAddress")
        or ""
    ).strip()
    if not host:
        return None
    port = extract_ssh_port(ssh) or extract_ssh_port(runtime) or extract_ssh_port(payload)
    if port is None:
        return None
    return PodSshEndpoint(host=host, port=port, user="root")


def extract_ssh_port(payload: dict[str, Any]) -> int | None:
    direct_port = payload.get("port")
    if direct_port is not None:
        return int(direct_port)
    ports = payload.get("ports")
    if isinstance(ports, list):
        for item in ports:
            if not isinstance(item, dict):
                continue
            private_port = int(item.get("privatePort") or item.get("containerPort") or item.get("port") or 0)
            if private_port != 22:
                continue
            public_port = int(item.get("publicPort") or item.get("hostPort") or 0)
            if public_port:
                return public_port
    port_mappings = payload.get("portMappings")
    if isinstance(port_mappings, dict):
        value = port_mappings.get("22") or port_mappings.get("22/tcp")
        if value is not None:
            return int(value)
    ssh_port = payload.get("sshPort")
    if ssh_port is not None:
        return int(ssh_port)
    return None


def remote_clone_url(origin_url: str) -> str:
    stripped = origin_url.strip()
    if stripped.startswith("git@github.com:") and stripped.endswith(".git"):
        repo_slug = stripped.removeprefix("git@github.com:").removesuffix(".git")
        return f"https://github.com/{repo_slug}.git"
    return stripped


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def summarize_called_process_error(exc: subprocess.CalledProcessError) -> str:
    parts = [part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()]
    return " | ".join(parts) if parts else f"exit code {exc.returncode}"


def summarize_exception(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return summarize_called_process_error(exc)
    return str(exc).strip() or exc.__class__.__name__


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_subprocess(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, check=check, text=True, timeout=timeout)


def run_command_capture(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        if check:
            raise subprocess.CalledProcessError(
                exc.returncode,
                exc.cmd,
                output=exc.stdout,
                stderr=exc.stderr,
            ) from exc
        raise
    return result.stdout
