from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ask_seattle.runpod import (
    RunPodConfig,
    RunPodOrchestrationError,
    available_gpus_for_datacenter,
    build_create_pod_payload,
    build_create_pod_command,
    build_remote_env_cache_key,
    build_remote_bootstrap_command,
    build_remote_make_args,
    candidate_datacenters,
    candidate_gpu_ids_for_existing_volume,
    cleanup_remote_workspace,
    cleanup_expired_cached_volume,
    create_pod,
    datacenter_has_gpu,
    ensure_clean_worktree,
    extract_ssh_endpoint,
    first_available_gpu_for_datacenter,
    artifact_dirs_for_target,
    PodSshEndpoint,
    pull_artifacts,
    wait_for_pod_ready,
    provision_volume_and_pod,
    remote_clone_url,
    select_datacenter,
    is_retryable_pod_create_error,
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


def test_candidate_gpu_ids_for_existing_volume_appends_fallbacks_after_primary() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"),
        fallback_gpu_types=("NVIDIA L4", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A4000"),
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
    datacenters = [
        {
            "id": "EU-RO-1",
            "gpuAvailability": [
                {"gpuId": "NVIDIA RTX A5000", "stockStatus": "Unavailable"},
                {"gpuId": "NVIDIA L4", "stockStatus": "High"},
                {"gpuId": "NVIDIA RTX A4000", "stockStatus": "High"},
            ],
        }
    ]

    assert candidate_gpu_ids_for_existing_volume(
        config,
        datacenters=datacenters,
        data_center_id="EU-RO-1",
    ) == ("NVIDIA L4", "NVIDIA RTX A4000")


def test_is_retryable_pod_create_error_accepts_current_rest_capacity_message() -> None:
    exc = subprocess.CalledProcessError(
        1,
        ("POST", "/pods"),
        output='{"error":"create pod: There are no instances currently available","status":500}',
        stderr="",
    )

    assert is_retryable_pod_create_error(exc) is True


def test_is_retryable_pod_create_error_accepts_generic_provider_500_message() -> None:
    exc = subprocess.CalledProcessError(
        1,
        ("POST", "/pods"),
        output='{"error":"create pod: Something went wrong. Please try again later or contact support.","status":500}',
        stderr="",
    )

    assert is_retryable_pod_create_error(exc) is True


def test_build_remote_make_args_includes_label_path_and_benchmark_notes() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA GeForce RTX 4090",),
        fallback_gpu_types=("NVIDIA L4",),
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
    assert "SEMANTIC_TERTIARY_MODEL_ID=jinaai/jina-embeddings-v5-text-small-classification" in args
    assert "TRANSFORMER_TERTIARY_MODEL_ID=chandar-lab/NeoBERT" in args
    assert "TRANSFORMER_QUATERNARY_MODEL_ID=answerdotai/ModernBERT-large" in args


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
        env_cache_key="env-key-123",
        make_args=("LABELS=/workspace/runpod-inputs/run/labels.jsonl", "BENCHMARK_NOTES=after labels"),
    )

    assert "'BENCHMARK_NOTES=after labels'" in command
    assert "/workspace/ask-seattle/scripts/runpod_pod_bootstrap.sh" in command
    assert "21600" in command
    assert "env-key-123" in command


def test_build_remote_env_cache_key_changes_when_dependency_shape_changes(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='ask-seattle'\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "runpod_pod_bootstrap.sh").write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    config = RunPodConfig(
        repo_root=tmp_path,
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=tmp_path / "id.pub",
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        fallback_gpu_types=("NVIDIA L4",),
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
        remote_run_timeout_seconds=21600,
    )

    first_key = build_remote_env_cache_key(config)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='ask-seattle'\nversion='0.2.0'\n", encoding="utf-8")
    second_key = build_remote_env_cache_key(config)

    assert first_key != second_key


def test_build_create_pod_command_prefers_template_id() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        fallback_gpu_types=("NVIDIA L4",),
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
        fallback_gpu_types=("NVIDIA L4",),
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


def test_build_create_pod_payload_prefers_template_id() -> None:
    config = RunPodConfig(
        repo_root=Path("/tmp/repo"),
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=Path("/tmp/id.pub"),
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        fallback_gpu_types=("NVIDIA L4",),
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

    payload = build_create_pod_payload(
        config,
        pod_name="ask-seattle-test",
        gpu_id="NVIDIA RTX A5000",
        data_center_id="EU-RO-1",
        network_volume_id="vol-123",
    )

    assert payload["templateId"] == "runpod-torch-v240"
    assert "imageName" not in payload
    assert payload["gpuTypeIds"] == ["NVIDIA RTX A5000"]
    assert payload["dataCenterIds"] == ["EU-RO-1"]
    assert payload["networkVolumeId"] == "vol-123"


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
        fallback_gpu_types=("NVIDIA L4",),
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


def test_pull_artifacts_uses_uncompressed_rsync_for_large_directories(
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
        fallback_gpu_types=("NVIDIA L4",),
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
        remote_run_timeout_seconds=21600,
    )
    monkeypatch.setattr(runpod, "remote_directory_exists", lambda **kwargs: True)
    monkeypatch.setattr(runpod, "remote_directory_size", lambda **kwargs: "7.6G")
    monkeypatch.setattr(runpod, "local_directory_size", lambda path: "missing")
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(runpod, "_run_subprocess", fake_run)

    pull_artifacts(
        config,
        ssh_endpoint=PodSshEndpoint(host="1.2.3.4", port=2222, user="root"),
        target="benchmark",
    )

    rsync_commands = [command for command in commands if command[0] == "rsync"]
    assert len(rsync_commands) == 1
    assert "-rlpt" in rsync_commands[0]
    assert "--partial" in rsync_commands[0]
    assert "--inplace" in rsync_commands[0]
    assert "checkpoints_*/" in rsync_commands[0]
    assert "checkpoint-*/" in rsync_commands[0]
    assert "optimizer.pt" in rsync_commands[0]
    assert "-rlptz" not in rsync_commands[0]


def test_pull_artifacts_retries_retryable_rsync_failures(
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
        fallback_gpu_types=("NVIDIA L4",),
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
        remote_run_timeout_seconds=21600,
    )
    monkeypatch.setattr(runpod, "remote_directory_exists", lambda **kwargs: True)
    monkeypatch.setattr(runpod, "remote_directory_size", lambda **kwargs: "7.6G")
    monkeypatch.setattr(runpod, "local_directory_size", lambda path: "missing")
    monkeypatch.setattr(runpod.time, "sleep", lambda seconds: None)
    attempts = {"count": 0}

    def fake_run(command: tuple[str, ...], *args: object, **kwargs: object) -> None:
        if command[0] != "rsync":
            return
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise subprocess.CalledProcessError(12, command, output="stream error", stderr="")

    monkeypatch.setattr(runpod, "_run_subprocess", fake_run)

    pull_artifacts(
        config,
        ssh_endpoint=PodSshEndpoint(host="1.2.3.4", port=2222, user="root"),
        target="benchmark",
    )

    assert attempts["count"] == 2


def test_cleanup_remote_workspace_prunes_logs_inputs_and_repo_artifacts(
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
        fallback_gpu_types=("NVIDIA L4",),
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
        remote_run_timeout_seconds=21600,
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(runpod, "_run_subprocess", fake_run)

    cleanup_remote_workspace(
        config,
        ssh_endpoint=PodSshEndpoint(host="1.2.3.4", port=2222, user="root"),
        target="retrain",
    )

    assert len(commands) == 1
    assert commands[0][0] == "ssh"
    assert "rm -rf /workspace/ask-seattle/models" in commands[0][-1]
    assert "rm -rf /workspace/ask-seattle/data/processed" in commands[0][-1]
    assert "find /workspace/runpod-logs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +" in commands[0][-1]
    assert "find /workspace/runpod-inputs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +" in commands[0][-1]


def test_cleanup_remote_workspace_preserves_models_for_benchmark(
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
        fallback_gpu_types=("NVIDIA L4",),
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
        remote_run_timeout_seconds=21600,
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *args: object, **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(runpod, "_run_subprocess", fake_run)

    cleanup_remote_workspace(
        config,
        ssh_endpoint=PodSshEndpoint(host="1.2.3.4", port=2222, user="root"),
        target="benchmark",
    )

    assert len(commands) == 1
    assert commands[0][0] == "ssh"
    assert "rm -rf /workspace/ask-seattle/models" not in commands[0][-1]
    assert "rm -rf /workspace/ask-seattle/data/processed" in commands[0][-1]


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
        fallback_gpu_types=("NVIDIA L4",),
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


def test_provision_volume_and_pod_uses_same_datacenter_fallback_gpu_for_existing_volume(
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
        fallback_gpu_types=("NVIDIA L4", "NVIDIA RTX A4000"),
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

    monkeypatch.setattr(
        runpod,
        "list_datacenters",
        lambda: [
            {
                "id": "EU-RO-1",
                "gpuAvailability": [
                    {"gpuId": "NVIDIA RTX A5000", "stockStatus": "Unavailable"},
                    {"gpuId": "NVIDIA L4", "stockStatus": "High"},
                ],
            }
        ],
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
    captured: dict[str, tuple[str, ...]] = {}

    def fake_create_pod_in_datacenter(config, *, pod_name, data_center_id, network_volume_id, gpu_ids):
        captured["gpu_ids"] = gpu_ids
        return "NVIDIA L4", runpod.PodInfo(
            pod_id="pod-123",
            name=pod_name,
            desired_status="RUNNING",
            ssh_endpoint=None,
        )

    monkeypatch.setattr(runpod, "create_pod_in_datacenter", fake_create_pod_in_datacenter)

    volume, gpu_id, data_center_id, pod = provision_volume_and_pod(config, pod_name="ask-seattle-test")

    assert volume.volume_id == "vol-123"
    assert gpu_id == "NVIDIA L4"
    assert data_center_id == "EU-RO-1"
    assert pod.pod_id == "pod-123"
    assert captured["gpu_ids"] == ("NVIDIA L4",)


def test_provision_volume_and_pod_uses_fallback_gpu_for_new_volume_when_primary_unavailable(
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
        fallback_gpu_types=("NVIDIA L4",),
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

    monkeypatch.setattr(
        runpod,
        "list_datacenters",
        lambda: [
            {
                "id": "EU-RO-1",
                "gpuAvailability": [
                    {"gpuId": "NVIDIA RTX A5000", "stockStatus": "Unavailable"},
                    {"gpuId": "NVIDIA L4", "stockStatus": "High"},
                ],
            }
        ],
    )
    monkeypatch.setattr(runpod, "list_network_volumes", lambda: [])
    monkeypatch.setattr(
        runpod,
        "create_network_volume",
        lambda name, size_gb, data_center_id: runpod.NetworkVolume(
            volume_id="vol-456",
            name=name,
            data_center_id=data_center_id,
            size_gb=size_gb,
        ),
    )
    captured: dict[str, tuple[str, ...]] = {}

    def fake_create_pod_in_datacenter(config, *, pod_name, data_center_id, network_volume_id, gpu_ids):
        captured["gpu_ids"] = gpu_ids
        return "NVIDIA L4", runpod.PodInfo(
            pod_id="pod-456",
            name=pod_name,
            desired_status="RUNNING",
            ssh_endpoint=None,
        )

    monkeypatch.setattr(runpod, "create_pod_in_datacenter", fake_create_pod_in_datacenter)

    volume, gpu_id, data_center_id, pod = provision_volume_and_pod(config, pod_name="ask-seattle-test")

    assert volume.volume_id == "vol-456"
    assert gpu_id == "NVIDIA L4"
    assert data_center_id == "EU-RO-1"
    assert pod.pod_id == "pod-456"
    assert captured["gpu_ids"] == ("NVIDIA L4",)


def test_create_pod_reconciles_by_name_after_create_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ask_seattle import runpod

    config = RunPodConfig(
        repo_root=tmp_path,
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=tmp_path / "id.pub",
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        fallback_gpu_types=("NVIDIA L4",),
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
        pod_create_timeout_seconds=10,
    )

    def timeout_capture(*args, **kwargs):
        raise subprocess.TimeoutExpired(("POST", "/pods"), timeout=10)

    monkeypatch.setattr(runpod, "create_pod_via_api", timeout_capture)
    monkeypatch.setattr(
        runpod,
        "reconcile_pod_by_name",
        lambda pod_name, wait_timeout_seconds: runpod.PodInfo(
            pod_id="pod-123",
            name=pod_name,
            desired_status="RUNNING",
            ssh_endpoint=None,
        ),
    )

    pod = create_pod(
        config,
        pod_name="ask-seattle-test",
        gpu_id="NVIDIA RTX A5000",
        data_center_id="EU-RO-1",
        network_volume_id="vol-123",
    )

    assert pod.pod_id == "pod-123"


def test_wait_for_pod_ready_reconciles_after_missing_pod(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ask_seattle import runpod

    config = RunPodConfig(
        repo_root=tmp_path,
        repo_slug="sayhiben/ask-seattle",
        ssh_key_path=tmp_path / "id.pub",
        volume_name="ask-seattle-train-sayhiben",
        volume_size_gb=100,
        volume_retention_seconds=300,
        gpu_types=("NVIDIA RTX A5000",),
        fallback_gpu_types=("NVIDIA L4",),
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
        pod_ready_timeout_seconds=30,
    )

    state = {"calls": 0}

    def fake_get_pod(pod_id: str):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RunPodOrchestrationError("pod not found")
        return runpod.PodInfo(
            pod_id="pod-456",
            name="ask-seattle-test",
            desired_status="RUNNING",
            ssh_endpoint=runpod.PodSshEndpoint(host="1.2.3.4", port=1234, user="root"),
        )

    monkeypatch.setattr(runpod, "get_pod", fake_get_pod)
    monkeypatch.setattr(
        runpod,
        "reconcile_pod_by_name",
        lambda pod_name, wait_timeout_seconds: runpod.PodInfo(
            pod_id="pod-456",
            name=pod_name,
            desired_status="RUNNING",
            ssh_endpoint=runpod.PodSshEndpoint(host="1.2.3.4", port=1234, user="root"),
        ),
    )
    monkeypatch.setattr(runpod, "wait_for_ssh", lambda endpoint: None)

    pod = wait_for_pod_ready(config, "pod-old", pod_name="ask-seattle-test")

    assert pod.pod_id == "pod-456"
