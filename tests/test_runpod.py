from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ask_seattle.runpod import (
    RunPodConfig,
    RunPodOrchestrationError,
    available_gpus_for_datacenter,
    build_create_pod_command,
    build_remote_bootstrap_command,
    build_remote_make_args,
    candidate_datacenters,
    cleanup_expired_cached_volume,
    datacenter_has_gpu,
    ensure_clean_worktree,
    extract_ssh_endpoint,
    first_available_gpu_for_datacenter,
    artifact_dirs_for_target,
    provision_volume_and_pod,
    remote_clone_url,
    select_datacenter,
)


def test_artifact_dirs_for_target() -> None:
    assert artifact_dirs_for_target("retrain") == (
        "models/real-labels-precision-refresh",
        "models/benchmark-suite",
    )
    assert artifact_dirs_for_target("benchmark") == ("models/benchmark-suite",)
    assert artifact_dirs_for_target("benchmark-variants") == ("models/benchmark-variants",)


def test_select_datacenter_prefers_candidate_order_and_gpu_priority() -> None:
    datacenters = [
        {
            "id": "US-GA-1",
            "gpuAvailability": [{"gpuId": "NVIDIA RTX A5000", "stockStatus": "High"}],
        },
        {
            "id": "US-KS-2",
            "gpuAvailability": [{"gpuId": "NVIDIA GeForce RTX 4090", "stockStatus": "High"}],
        },
    ]

    assert (
        select_datacenter(
            datacenters,
            preferred_gpu_ids=("NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"),
            candidate_data_center_ids=("US-KS-2", "US-GA-1"),
        )
        == "US-KS-2"
    )


def test_first_available_gpu_for_datacenter_respects_preference_order() -> None:
    datacenters = [
        {
            "id": "US-KS-2",
            "gpuAvailability": [
                {"gpuId": "NVIDIA RTX A5000", "stockStatus": "High"},
                {"gpuId": "NVIDIA GeForce RTX 4090", "stockStatus": "Low"},
            ],
        }
    ]

    assert (
        first_available_gpu_for_datacenter(
            datacenters,
            data_center_id="US-KS-2",
            preferred_gpu_ids=("NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"),
        )
        == "NVIDIA GeForce RTX 4090"
    )


def test_datacenter_has_gpu_ignores_unavailable_stock() -> None:
    datacenter = {
        "id": "US-KS-2",
        "gpuAvailability": [{"gpuId": "NVIDIA GeForce RTX 4090", "stockStatus": "Unavailable"}],
    }

    assert datacenter_has_gpu(datacenter, "NVIDIA GeForce RTX 4090") is False


def test_datacenter_has_gpu_accepts_blank_stock_status_when_gpu_is_listed() -> None:
    datacenter = {
        "id": "EU-RO-1",
        "gpuAvailability": [{"gpuId": "NVIDIA GeForce RTX 4090", "stockStatus": ""}],
    }

    assert datacenter_has_gpu(datacenter, "NVIDIA GeForce RTX 4090") is True


def test_candidate_datacenters_keeps_preferred_order_with_available_gpus() -> None:
    datacenters = [
        {"id": "EU-RO-1", "gpuAvailability": [{"gpuId": "NVIDIA RTX A5000", "stockStatus": ""}]},
        {"id": "US-NC-1", "gpuAvailability": [{"gpuId": "NVIDIA GeForce RTX 4090", "stockStatus": "High"}]},
        {"id": "US-GA-2", "gpuAvailability": [{"gpuId": "NVIDIA L4", "stockStatus": "High"}]},
    ]

    assert candidate_datacenters(
        datacenters,
        preferred_gpu_ids=("NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"),
        candidate_data_center_ids=("EU-RO-1", "US-NC-1", "US-GA-2"),
    ) == ("EU-RO-1", "US-NC-1")


def test_available_gpus_for_datacenter_returns_preference_order() -> None:
    datacenters = [
        {
            "id": "US-NC-1",
            "gpuAvailability": [
                {"gpuId": "NVIDIA GeForce RTX 4090", "stockStatus": "High"},
                {"gpuId": "NVIDIA RTX A5000", "stockStatus": "High"},
            ],
        }
    ]

    assert available_gpus_for_datacenter(
        datacenters,
        data_center_id="US-NC-1",
        preferred_gpu_ids=("NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"),
    ) == ("NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090")


def test_build_remote_make_args_includes_label_path_and_benchmark_notes() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA GeForce RTX 4090",),
        data_center_ids=("US-KS-2",),
        template_id="runpod-torch-v240",
        image="runpod/pytorch:test",
        remote_dir="/workspace/ask-seattle",
        ssh_user="root",
        container_disk_gb=50,
        volume_mount_path="/workspace",
        labels_path=Path("/tmp/labels.jsonl"),
        benchmark_meta_dir=Path("/tmp/meta"),
        split_strategy="random",
        split_seed=13,
        evaluation_subreddit="seattle",
        benchmark_notes="after labels",
        semantic_model_id="sentence-transformers/all-MiniLM-L6-v2",
        semantic_secondary_model_id="Qwen/Qwen3-Embedding-0.6B",
        transformer_model_id="microsoft/deberta-v3-small",
        transformer_secondary_model_id="answerdotai/ModernBERT-base",
        causal_lm_model_id="Qwen/Qwen3-1.7B",
        remote_run_timeout_seconds=21600,
    )

    args = build_remote_make_args(
        config,
        target="benchmark",
        remote_labels_path="/workspace/runpod-inputs/run/labels.jsonl",
    )

    assert "LABELS=/workspace/runpod-inputs/run/labels.jsonl" in args
    assert "EVAL_SUBREDDIT=seattle" in args
    assert "BENCHMARK_NOTES=after labels" in args


def test_build_remote_bootstrap_command_quotes_make_args() -> None:
    command = build_remote_bootstrap_command(
        remote_script="/workspace/ask-seattle/scripts/runpod_pod_bootstrap.sh",
        target="benchmark",
        commit_sha="abc123",
        origin_url="git@github.com:sayhiben/ask-seattle.git",
        remote_repo_dir="/workspace/ask-seattle",
        remote_labels_path="/workspace/runpod-inputs/run/labels.jsonl",
        remote_log_dir="/workspace/runpod-logs/run-id",
        remote_venv_dir="/workspace/.venv",
        run_id="run-id",
        run_timeout_seconds=21600,
        make_args=("LABELS=/workspace/runpod-inputs/run/labels.jsonl", "BENCHMARK_NOTES=after labels"),
    )

    assert "'BENCHMARK_NOTES=after labels'" in command
    assert "/workspace/ask-seattle/scripts/runpod_pod_bootstrap.sh" in command
    assert "21600" in command


def test_build_create_pod_command_prefers_template_id() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        data_center_ids=("EU-RO-1",),
        template_id="runpod-torch-v240",
        image="runpod/pytorch:test",
        remote_dir="/workspace/ask-seattle",
        ssh_user="root",
        container_disk_gb=50,
        volume_mount_path="/workspace",
        labels_path=Path("/tmp/labels.jsonl"),
        benchmark_meta_dir=Path("/tmp/meta"),
        split_strategy="random",
        split_seed=13,
        evaluation_subreddit=None,
        benchmark_notes=None,
        semantic_model_id="sentence-transformers/all-MiniLM-L6-v2",
        semantic_secondary_model_id="Qwen/Qwen3-Embedding-0.6B",
        transformer_model_id="microsoft/deberta-v3-small",
        transformer_secondary_model_id="answerdotai/ModernBERT-base",
        causal_lm_model_id="Qwen/Qwen3-1.7B",
        remote_run_timeout_seconds=21600,
    )

    command = build_create_pod_command(
        config,
        pod_name="ask-seattle-test",
        gpu_id="NVIDIA RTX A5000",
        data_center_id="EU-RO-1",
        network_volume_id="vol-123",
    )

    assert "--template-id" in command
    assert "runpod-torch-v240" in command
    assert "--image" not in command


def test_build_create_pod_command_falls_back_to_image_when_template_missing() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        data_center_ids=("EU-RO-1",),
        template_id=None,
        image="runpod/pytorch:test",
        remote_dir="/workspace/ask-seattle",
        ssh_user="root",
        container_disk_gb=50,
        volume_mount_path="/workspace",
        labels_path=Path("/tmp/labels.jsonl"),
        benchmark_meta_dir=Path("/tmp/meta"),
        split_strategy="random",
        split_seed=13,
        evaluation_subreddit=None,
        benchmark_notes=None,
        semantic_model_id="sentence-transformers/all-MiniLM-L6-v2",
        semantic_secondary_model_id="Qwen/Qwen3-Embedding-0.6B",
        transformer_model_id="microsoft/deberta-v3-small",
        transformer_secondary_model_id="answerdotai/ModernBERT-base",
        causal_lm_model_id="Qwen/Qwen3-1.7B",
        remote_run_timeout_seconds=21600,
    )

    command = build_create_pod_command(
        config,
        pod_name="ask-seattle-test",
        gpu_id="NVIDIA RTX A5000",
        data_center_id="EU-RO-1",
        network_volume_id="vol-123",
    )

    assert "--image" in command
    assert "runpod/pytorch:test" in command
    assert "--template-id" not in command


def test_extract_ssh_endpoint_reads_runtime_port_mapping() -> None:
    endpoint = extract_ssh_endpoint(
        {
            "runtime": {
                "publicIp": "1.2.3.4",
                "ports": [{"privatePort": 22, "publicPort": 32511}],
            }
        }
    )

    assert endpoint is not None
    assert endpoint.host == "1.2.3.4"
    assert endpoint.port == 32511
    assert endpoint.user == "root"


def test_extract_ssh_endpoint_reads_new_ssh_payload_shape() -> None:
    endpoint = extract_ssh_endpoint(
        {
            "ssh": {
                "ip": "203.57.40.238",
                "port": 10132,
            }
        }
    )

    assert endpoint is not None
    assert endpoint.host == "203.57.40.238"
    assert endpoint.port == 10132
    assert endpoint.user == "root"


def test_remote_clone_url_rewrites_github_ssh_to_https() -> None:
    assert (
        remote_clone_url("git@github.com:sayhiben/ask-seattle.git")
        == "https://github.com/sayhiben/ask-seattle.git"
    )


def test_ensure_clean_worktree_raises_for_dirty_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ask_seattle import runpod

    monkeypatch.setattr(runpod, "run_command_capture", lambda *args, **kwargs: " M src/ask_seattle/model.py\n")

    with pytest.raises(RunPodOrchestrationError):
        ensure_clean_worktree(tmp_path)


def test_cleanup_expired_cached_volume_deletes_volume_and_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ask_seattle import runpod

    config = RunPodConfig(
        repo_root=tmp_path,
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=tmp_path / "id.pub",
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        data_center_ids=("EU-RO-1",),
        template_id="runpod-torch-v240",
        image="runpod/pytorch:test",
        remote_dir="/workspace/ask-seattle",
        ssh_user="root",
        container_disk_gb=50,
        volume_mount_path="/workspace",
        labels_path=tmp_path / "labels.jsonl",
        benchmark_meta_dir=tmp_path / "meta",
        split_strategy="random",
        split_seed=13,
        evaluation_subreddit=None,
        benchmark_notes=None,
        semantic_model_id="sentence-transformers/all-MiniLM-L6-v2",
        semantic_secondary_model_id="Qwen/Qwen3-Embedding-0.6B",
        transformer_model_id="microsoft/deberta-v3-small",
        transformer_secondary_model_id="answerdotai/ModernBERT-base",
        causal_lm_model_id="Qwen/Qwen3-1.7B",
    )
    lease_path = config.benchmark_meta_dir / "volumes" / "ask-seattle-train-sayhiben.json"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text(
        '{"expires_at":"2020-01-01T00:00:00Z","volume_id":"vol-123","volume_name":"ask-seattle-train-sayhiben"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runpod,
        "list_network_volumes",
        lambda: [
            runpod.NetworkVolume(
                volume_id="vol-123",
                name="ask-seattle-train-sayhiben",
                data_center_id="EU-RO-1",
                size_gb=100,
            )
        ],
    )
    deleted: list[str] = []
    monkeypatch.setattr(runpod, "delete_network_volume", lambda volume_id: deleted.append(volume_id))

    cleanup_expired_cached_volume(config)

    assert deleted == ["vol-123"]
    assert not lease_path.exists()


def test_provision_volume_and_pod_preserves_existing_cache_without_eviction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ask_seattle import runpod

    config = RunPodConfig(
        repo_root=tmp_path,
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=tmp_path / "id.pub",
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        data_center_ids=("EU-RO-1",),
        template_id="runpod-torch-v240",
        image="runpod/pytorch:test",
        remote_dir="/workspace/ask-seattle",
        ssh_user="root",
        container_disk_gb=50,
        volume_mount_path="/workspace",
        labels_path=tmp_path / "labels.jsonl",
        benchmark_meta_dir=tmp_path / "meta",
        split_strategy="random",
        split_seed=13,
        evaluation_subreddit=None,
        benchmark_notes=None,
        semantic_model_id="sentence-transformers/all-MiniLM-L6-v2",
        semantic_secondary_model_id="Qwen/Qwen3-Embedding-0.6B",
        transformer_model_id="microsoft/deberta-v3-small",
        transformer_secondary_model_id="answerdotai/ModernBERT-base",
        causal_lm_model_id="Qwen/Qwen3-1.7B",
        evict_volume_on_capacity_failure=False,
    )

    monkeypatch.setattr(runpod, "list_datacenters", lambda: [])
    monkeypatch.setattr(
        runpod,
        "list_network_volumes",
        lambda: [
            runpod.NetworkVolume(
                volume_id="vol-123",
                name="ask-seattle-train-sayhiben",
                data_center_id="EU-RO-1",
                size_gb=100,
            )
        ],
    )

    def fail_create(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            ("runpodctl", "pod", "create"),
            output="",
            stderr="no longer any instances available",
        )

    monkeypatch.setattr(runpod, "create_pod_in_datacenter", fail_create)
    deleted: list[str] = []
    monkeypatch.setattr(runpod, "delete_network_volume", lambda volume_id: deleted.append(volume_id))

    with pytest.raises(RunPodOrchestrationError, match="preserved to honor the retention policy"):
        provision_volume_and_pod(config, pod_name="ask-seattle-test")

    assert deleted == []
