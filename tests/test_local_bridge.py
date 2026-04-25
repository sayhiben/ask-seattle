from __future__ import annotations

import http.client
import json
import logging
from pathlib import Path
from threading import Thread
import threading
from typing import Any

import pytest

from ask_seattle.data import load_jsonl_records, write_jsonl_records
import ask_seattle.local_bridge as local_bridge
from ask_seattle.local_bridge import (
    AutoRetrainManager,
    BridgeConfig,
    LocalBridgeRequestHandler,
    find_label_record,
    resolve_bridge_path,
    upsert_label_record,
)


class FakeClassifier:
    classes_ = [0, 1]


class FakeModel:
    named_steps = {"classifier": FakeClassifier()}

    def __init__(self) -> None:
        self.last_rows: list[object] = []

    def predict_proba(self, texts: list[object]) -> list[list[float]]:
        self.last_rows = list(texts)
        return [[0.2, 0.8] for _ in texts]


class ScoredFakeModel(FakeModel):
    def __init__(self, positive_probability: float) -> None:
        super().__init__()
        self.positive_probability = positive_probability

    def predict_proba(self, texts: list[object]) -> list[list[float]]:
        self.last_rows = list(texts)
        negative_probability = 1 - self.positive_probability
        return [[negative_probability, self.positive_probability] for _ in texts]


class FakeServer:
    pass


class FakeAutoRetrain:
    def note_label_saved(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "scheduled": True,
            "in_progress": True,
            "retrain_every": 5,
            "training_records": 10,
            "last_retrain_training_records": 5,
            "labels_until_retrain": 0,
        }


def request_json(port: int, method: str, path: str, payload: dict[str, Any] | None = None) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload or {})
    connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    connection.close()
    return data


def test_bridge_check_and_train(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = None
    Handler.bridge_config.comparison_models = []
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    fake_model = FakeModel()
    Handler.bridge_config.bundle = {
        "model": fake_model,
        "model_name": "fake",
        "model_version": "test",
        "threshold": 0.7,
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        check = request_json(
            port,
            "POST",
            "/check",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": (
                    "Visiting Seattle soon and need help comparing neighborhoods, hotels, "
                    "transit access, and easy food options for a weekend trip."
                ),
                "crosspost_body": "Original poster asked for allergy relief recommendations after moving to Redmond.",
                "post_type": "image",
                "content_domain": "reddit.com",
                "is_crosspost": True,
            },
        )
        train = request_json(
            port,
            "POST",
            "/train",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "Visiting",
                "crosspost_body": "Original poster body from the source community.",
                "crosspost_title": "Source community title",
                "label": "ask",
                "post_type": "text",
                "content_domain": "reddit.com",
                "is_crosspost": False,
                "collected_at": "2026-04-10T20:00:00+00:00",
            },
        )
    finally:
        server.shutdown()

    assert check["ok"] is True
    assert check["result"]["label"] == "askseattle"
    assert check["result"]["confidence_band"] == "high"
    assert check["decider_result"] is None
    assert check["decision_context"]["policy"] == "stacked_transformer_decider"
    assert check["decision_context"]["decision_source"] == "primary_model"
    assert "stacked_decider_unavailable" in check["decision_context"]["review_reasons"]
    assert check["comparison_models"] == []
    assert check["comparisons"] == []
    assert "POST_TYPE:image" in fake_model.last_rows[0]["body"]
    assert "CONTENT_DOMAIN:reddit_com" in fake_model.last_rows[0]["body"]
    assert "CROSSPOST:yes" in fake_model.last_rows[0]["body"]
    assert "Original poster asked for allergy relief recommendations after moving to Redmond." in fake_model.last_rows[0]["body_raw"]
    assert train["ok"] is True
    assert train["saved"]["label"] == "askseattle"
    assert train["saved"]["post_type"] == "text"
    assert train["saved"]["content_domain"] == "reddit.com"
    assert train["saved"]["is_crosspost"] is False
    assert train["saved"]["selftext"] == "Visiting\n\nOriginal poster body from the source community."
    assert train["saved"]["crosspost_body"] == "Original poster body from the source community."
    assert train["saved"]["crosspost_title"] == "Source community title"
    assert train["saved"]["collected_at"] == "2026-04-10T20:00:00+00:00"
    assert load_jsonl_records(labels)[0]["id"] == "abc"


def test_bridge_check_scope_filters_explicit_out_of_scope_post_type(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = None
    Handler.bridge_config.comparison_models = []
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": FakeModel(),
        "model_name": "fake",
        "model_version": "test",
        "threshold": 0.7,
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        check = request_json(
            port,
            "POST",
            "/check",
            {
                "id": "abc",
                "title": "Seattle skyline",
                "selftext": "",
                "post_type": "image",
                "is_crosspost": False,
            },
        )
    finally:
        server.shutdown()

    assert check["ok"] is True
    assert check["result"]["model_name"] == "scope_filter_text_plus_crosspost"
    assert check["result"]["label"] == "not_askseattle"
    assert check["decider_result"] is None
    assert check["decision_context"]["decision_source"] == "scope_filter"
    assert check["decision_context"]["scope_included"] is False
    assert check["decision_context"]["scope_policy"] == "text_plus_crosspost"
    assert check["decision_context"]["post_type"] == "image"
    assert check["comparisons"] == []


def test_bridge_recorded_and_train_uses_last_label(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = None
    Handler.bridge_config.comparison_models = []
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": FakeModel(),
        "model_name": "fake",
        "model_version": "test",
        "threshold": 0.7,
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    payload = {
        "id": "abc",
        "permalink": "https://reddit.test/r/test/comments/abc/title/",
        "title": "Where should I stay?",
        "selftext": "Visiting",
    }
    try:
        before = request_json(port, "POST", "/recorded", payload)
        first = request_json(port, "POST", "/train", {**payload, "label": "askseattle"})
        second = request_json(port, "POST", "/train", {**payload, "label": "not_askseattle"})
        after = request_json(port, "POST", "/recorded", payload)
    finally:
        server.shutdown()

    records = load_jsonl_records(labels)
    assert before["recorded"] is False
    assert first["replaced"] is False
    assert second["replaced"] is True
    assert after["recorded"] is True
    assert after["record"]["label"] == "not_askseattle"
    assert len(records) == 1
    assert records[0]["label"] == "not_askseattle"


def test_bridge_train_reports_auto_retrain_status(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = None
    Handler.bridge_config.comparison_models = []
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = FakeAutoRetrain()
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": FakeModel(),
        "model_name": "fake",
        "model_version": "test",
        "threshold": 0.7,
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        train = request_json(
            port,
            "POST",
            "/train",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "Visiting",
                "label": "ask",
            },
        )
    finally:
        server.shutdown()

    assert train["ok"] is True
    assert train["auto_retrain"]["enabled"] is True
    assert train["auto_retrain"]["scheduled"] is True
    assert train["auto_retrain"]["in_progress"] is True


def test_bridge_check_returns_comparison_results(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    primary_model = FakeModel()
    semantic_model = FakeModel()
    transformer_model = FakeModel()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = tmp_path / "benchmark_suite_summary.json"
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": primary_model,
        "model_family": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "threshold": 0.7,
    }
    Handler.bridge_config.comparison_models = [
        {
            "name": "semantic_embedding",
            "model_family": "semantic_embedding",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "artifact_path": str(tmp_path / "semantic.joblib"),
            "bundle": {
                "model": semantic_model,
                "model_family": "tfidf",
                "model_name": "semantic_embedding_logreg",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
        {
            "name": "transformer_sequence_classifier",
            "model_family": "transformer_sequence_classifier",
            "model_id": "answerdotai/ModernBERT-base",
            "artifact_path": str(tmp_path / "transformer_bundle.joblib"),
            "bundle": {
                "model": transformer_model,
                "model_family": "tfidf",
                "model_name": "transformer_sequence_classifier",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
    ]

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        check = request_json(
            port,
            "POST",
            "/check",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "Visiting",
                "include_comparisons": True,
            },
        )
    finally:
        server.shutdown()

    assert check["ok"] is True
    assert check["result"]["model_name"] == "tfidf_logreg"
    assert [entry["name"] for entry in check["comparison_models"]] == [
        "semantic_embedding",
        "transformer_sequence_classifier",
    ]
    assert [entry["name"] for entry in check["comparisons"]] == [
        "semantic_embedding",
        "transformer_sequence_classifier",
    ]
    assert check["comparisons"][0]["result"]["model_name"] == "semantic_embedding_logreg"
    assert check["comparisons"][1]["result"]["model_name"] == "transformer_sequence_classifier"


def test_bridge_check_can_return_hybrid_decider_result_when_routed(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = tmp_path / "benchmark_suite_summary.json"
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.decider_policy = "hybrid_consensus"
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": ScoredFakeModel(0.7),
        "model_family": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "low_threshold": 0.75,
        "high_threshold": 0.9,
    }
    Handler.bridge_config.comparison_models = [
        {
            "name": "semantic_embedding",
            "model_family": "semantic_embedding",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "artifact_path": str(tmp_path / "semantic.joblib"),
            "bundle": {
                "model": ScoredFakeModel(0.95),
                "model_family": "tfidf",
                "model_name": "semantic_embedding_logreg",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
        {
            "name": "transformer_sequence_classifier",
            "model_family": "transformer_sequence_classifier",
            "model_id": "answerdotai/ModernBERT-base",
            "artifact_path": str(tmp_path / "transformer_bundle.joblib"),
            "bundle": {
                "model": ScoredFakeModel(0.95),
                "model_family": "tfidf",
                "model_name": "transformer_sequence_classifier",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
    ]

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        check = request_json(
            port,
            "POST",
            "/check",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "",
            },
        )
    finally:
        server.shutdown()

    assert check["ok"] is True
    assert check["result"]["label"] == "askseattle"
    assert check["decider_result"]["label"] == "askseattle"
    assert check["decider_result"]["confidence_band"] == "borderline"
    assert check["decider_result"]["score"] == pytest.approx(0.825)
    assert check["decision_context"]["decision_source"] == "hybrid_consensus"
    assert check["decision_context"]["review_priority"] == "high"
    assert "low_text" in check["decision_context"]["route_reasons"]
    assert "label_changed_by_hybrid" in check["decision_context"]["review_reasons"]
    assert len(check["comparisons"]) == 2


def test_bridge_check_primary_only_skips_hybrid_decider(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = tmp_path / "benchmark_suite_summary.json"
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.decider_policy = "primary_only"
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": ScoredFakeModel(0.7),
        "model_family": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "low_threshold": 0.75,
        "high_threshold": 0.9,
    }
    Handler.bridge_config.comparison_models = [
        {
            "name": "transformer_sequence_classifier",
            "model_family": "transformer_sequence_classifier",
            "model_id": "answerdotai/ModernBERT-base",
            "artifact_path": str(tmp_path / "transformer_bundle.joblib"),
            "bundle": {
                "model": ScoredFakeModel(0.95),
                "model_family": "tfidf",
                "model_name": "transformer_sequence_classifier",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
        {
            "name": "semantic_embedding",
            "model_family": "semantic_embedding",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "artifact_path": str(tmp_path / "semantic.joblib"),
            "bundle": {
                "model": ScoredFakeModel(0.95),
                "model_family": "tfidf",
                "model_name": "semantic_embedding_logreg",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
    ]

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        check = request_json(
            port,
            "POST",
            "/check",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "",
            },
        )
    finally:
        server.shutdown()

    assert check["ok"] is True
    assert check["result"]["label"] == "not_askseattle"
    assert check["decider_result"] is None
    assert check["decision_context"]["policy"] == "primary_only"
    assert check["decision_context"]["decision_source"] == "primary_model"
    assert check["comparisons"] == []


def test_bridge_check_comparison_returns_single_result(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    primary_model = FakeModel()
    semantic_model = FakeModel()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = tmp_path / "benchmark_suite_summary.json"
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.stacked_decider_model = None
    Handler.bridge_config.bundle = {
        "model": primary_model,
        "model_family": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "threshold": 0.7,
    }
    Handler.bridge_config.comparison_models = [
        {
            "name": "semantic_embedding",
            "display_name": "Semantic MiniLM",
            "model_family": "semantic_embedding",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "artifact_path": str(tmp_path / "semantic.joblib"),
            "bundle": {
                "model": semantic_model,
                "model_family": "tfidf",
                "model_name": "semantic_embedding_logreg",
                "model_version": "test",
                "threshold": 0.7,
            },
        },
    ]

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        response = request_json(
            port,
            "POST",
            "/check-comparison",
            {
                "name": "semantic_embedding",
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "Visiting",
            },
        )
    finally:
        server.shutdown()

    assert response["ok"] is True
    assert response["comparison"]["name"] == "semantic_embedding"
    assert response["comparison"]["result"]["model_name"] == "semantic_embedding_logreg"


def test_bridge_check_can_return_stacked_decider_result(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.comparison_suite_path = tmp_path / "benchmark_suite_summary.json"
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
    Handler.bridge_config.decider_policy = "stacked_transformer_decider"
    Handler.bridge_config.hybrid_policy = None
    Handler.bridge_config.bundle = {
        "model": ScoredFakeModel(0.7),
        "model_family": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "low_threshold": 0.75,
        "high_threshold": 0.9,
    }
    Handler.bridge_config.comparison_models = []
    Handler.bridge_config.stacked_decider_model = {
        "name": "stacked_transformer_decider",
        "display_name": "Stacked transformer decider",
        "model_family": "stacked_transformer_decider",
        "artifact_path": str(tmp_path / "stacked.joblib"),
        "bundle": {
            "model": ScoredFakeModel(0.96),
            "model_family": "tfidf",
            "model_name": "stacked_transformer_decider",
            "display_name": "Stacked transformer decider",
            "model_version": "test",
            "low_threshold": 0.8,
            "high_threshold": 0.9,
        },
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    try:
        check = request_json(
            port,
            "POST",
            "/check",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "",
            },
        )
    finally:
        server.shutdown()

    assert check["ok"] is True
    assert check["result"]["label"] == "askseattle"
    assert check["result"]["model_name"] == "stacked_transformer_decider"
    assert check["decider_result"]["label"] == "askseattle"
    assert check["decision_context"]["decision_source"] == "stacked_transformer_decider"
    assert check["decision_context"]["primary_result"]["label"] == "not_askseattle"
    assert "label_changed_by_stacked_decider" in check["decision_context"]["review_reasons"]


def test_load_comparison_models_filters_to_supported_active_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary_artifact = tmp_path / "tfidf.joblib"
    primary_artifact.write_text("primary", encoding="utf-8")
    neobert_artifact = tmp_path / "neobert.joblib"
    neobert_artifact.write_text("neobert", encoding="utf-8")
    modernbert_artifact = tmp_path / "modernbert-large.joblib"
    modernbert_artifact.write_text("modernbert", encoding="utf-8")
    semantic_artifact = tmp_path / "semantic.joblib"
    semantic_artifact.write_text("semantic", encoding="utf-8")
    causal_artifact = tmp_path / "causal.joblib"
    causal_artifact.write_text("causal", encoding="utf-8")

    suite_summary_path = tmp_path / "benchmark_suite_summary.json"
    suite_summary_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "semantic_minilm_tuned",
                        "display_name": "Semantic MiniLM",
                        "model_family": "semantic_embedding",
                        "artifact_path": str(semantic_artifact),
                        "status": "ok",
                    },
                    {
                        "name": "transformer_modernbert_large",
                        "display_name": "Transformer ModernBERT-large",
                        "model_family": "transformer_sequence_classifier",
                        "artifact_path": str(modernbert_artifact),
                        "status": "ok",
                    },
                    {
                        "name": "tfidf_recommended",
                        "display_name": "TF-IDF",
                        "model_family": "tfidf",
                        "artifact_path": str(primary_artifact),
                        "status": "ok",
                    },
                    {
                        "name": "causal_lm_qwen3_1_7b_lora",
                        "display_name": "Qwen3 LoRA",
                        "model_family": "causal_lm_classifier",
                        "artifact_path": str(causal_artifact),
                        "status": "ok",
                    },
                    {
                        "name": "transformer_neobert",
                        "display_name": "Transformer NeoBERT",
                        "model_family": "transformer_sequence_classifier",
                        "artifact_path": str(neobert_artifact),
                        "status": "ok",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load_model(path: Path) -> dict[str, Any]:
        artifact_name = Path(path).name
        if artifact_name == "neobert.joblib":
            return {"model_name": "transformer_neobert", "model_family": "transformer_sequence_classifier"}
        if artifact_name == "modernbert-large.joblib":
            return {"model_name": "transformer_modernbert_large", "model_family": "transformer_sequence_classifier"}
        raise AssertionError(f"unexpected artifact load: {artifact_name}")

    monkeypatch.setattr(local_bridge, "load_model", fake_load_model)

    loaded = local_bridge._load_comparison_models(
        primary_bundle={"artifact_path": str(primary_artifact), "model_family": "tfidf"},
        primary_model_path=primary_artifact,
        comparison_suite_path=suite_summary_path,
    )

    assert [entry["name"] for entry in loaded] == [
        "transformer_neobert",
        "transformer_modernbert_large",
    ]


def test_auto_retrain_note_label_saved_returns_status_without_deadlock(tmp_path: Path) -> None:
    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
    manager.bridge_config.label_path = tmp_path / "labels.jsonl"
    manager.split_strategy = "random"
    manager.split_seed = 13
    manager.evaluation_subreddit = None
    manager.retrain_every = 5
    manager.output_dir = tmp_path / "models"
    manager.reload_model_path = manager.output_dir / "tfidf_logreg.joblib"
    manager._state_lock = threading.Lock()
    manager._running = False
    manager._last_error = None
    manager._last_reload_at = None
    manager._last_summary = None
    manager._last_prepared_training_count = 5
    manager._last_retrain_training_count = 5
    manager._last_triggered_training_count = 5

    scheduled_counts: list[int] = []

    def fake_prepare_outputs() -> dict[str, int]:
        return {"training_records": 6}

    def fake_start_retrain(training_records: int) -> None:
        scheduled_counts.append(training_records)

    manager._prepare_outputs = fake_prepare_outputs  # type: ignore[method-assign]
    manager._start_retrain_locked = fake_start_retrain  # type: ignore[method-assign]

    status = manager.note_label_saved()

    assert scheduled_counts == []
    assert status["enabled"] is True
    assert status["scheduled"] is False
    assert status["in_progress"] is False
    assert status["training_records"] == 6
    assert status["labels_until_retrain"] == 4


def test_auto_retrain_note_label_saved_logs_status(tmp_path: Path, caplog) -> None:
    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
    manager.bridge_config.label_path = tmp_path / "labels.jsonl"
    manager.split_strategy = "random"
    manager.split_seed = 13
    manager.evaluation_subreddit = None
    manager.retrain_every = 5
    manager.output_dir = tmp_path / "models"
    manager.reload_model_path = manager.output_dir / "tfidf_logreg.joblib"
    manager._state_lock = threading.Lock()
    manager._running = False
    manager._last_error = None
    manager._last_reload_at = None
    manager._last_summary = None
    manager._last_prepared_training_count = 5
    manager._last_retrain_training_count = 5
    manager._last_triggered_training_count = 5

    def fake_prepare_outputs() -> dict[str, int]:
        return {"training_records": 6}

    manager._prepare_outputs = fake_prepare_outputs  # type: ignore[method-assign]

    caplog.set_level(logging.INFO, logger="ask_seattle.local_bridge")
    status = manager.note_label_saved()

    assert status["scheduled"] is False
    assert any(
        "auto retrain status event=label_saved_waiting_for_retrain" in record.message
        for record in caplog.records
    )


def test_auto_retrain_does_not_immediately_retry_failed_attempt(tmp_path: Path) -> None:
    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
    manager.split_strategy = "random"
    manager.split_seed = 13
    manager.evaluation_subreddit = None
    manager.retrain_every = 5
    manager._state_lock = threading.Lock()
    manager._running = False
    manager._last_error = "training failed"
    manager._last_reload_at = None
    manager._last_summary = None
    manager._last_prepared_training_count = 10
    manager._last_retrain_training_count = 5
    manager._last_triggered_training_count = 10

    scheduled_counts: list[int] = []

    def fake_start_retrain(training_records: int) -> None:
        scheduled_counts.append(training_records)

    manager._start_retrain_locked = fake_start_retrain  # type: ignore[method-assign]

    manager._schedule_followup_if_needed()

    assert scheduled_counts == []


def test_auto_retrain_snapshot_uses_label_lock(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    write_jsonl_records(
        labels,
        [
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "Visiting",
                "label": "askseattle",
            }
        ],
    )

    class CountingLock:
        def __init__(self) -> None:
            self.entries = 0

        def __enter__(self) -> None:
            self.entries += 1
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    lock = CountingLock()

    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
    manager.bridge_config.label_path = labels
    manager.bridge_config.label_lock = lock
    manager.split_strategy = "random"
    manager.split_seed = 13
    manager.evaluation_subreddit = None
    manager.output_dir = tmp_path / "models"

    snapshot_path = manager._snapshot_label_file()
    try:
        assert lock.entries == 1
        assert load_jsonl_records(snapshot_path) == load_jsonl_records(labels)
    finally:
        snapshot_path.unlink(missing_ok=True)


def test_auto_retrain_logs_training_summary(tmp_path: Path, monkeypatch, caplog) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text("{}\n", encoding="utf-8")

    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
    manager.bridge_config.label_path = tmp_path / "labels.jsonl"
    manager.bridge_config.label_lock = threading.Lock()
    manager.bridge_config.split_strategy = "random"
    manager.bridge_config.split_seed = 13
    manager.retrain_every = 5
    manager.evaluation_subreddit = "seattle"
    manager.split_strategy = "random"
    manager.split_seed = 13
    manager.output_dir = tmp_path / "models"
    manager.reload_model_path = manager.output_dir / "tfidf_logreg.joblib"
    manager._state_lock = threading.Lock()
    manager._running = True
    manager._last_error = None
    manager._last_reload_at = None
    manager._last_summary = None
    manager._last_prepared_training_count = 12
    manager._last_retrain_training_count = 8
    manager._last_triggered_training_count = 12

    summary = {
        "prepared_data": {"training_records": 12},
        "split": {"train": 8, "calibration": 2, "test": 2, "split_strategy": "random", "split_seed": 13},
        "calibration": {"available": True},
        "metrics": {
            "high_confidence_precision": 0.97,
            "high_confidence_recall": 0.75,
            "high_confidence_f1": 0.84,
        },
        "operating_metrics": {
            "auto_band": {"predicted_positive": 7},
        },
        "threshold_policy": {"low_threshold": 0.55, "high_threshold": 0.9},
        "production_ready": True,
        "production_ready_blocked_reason": None,
    }

    def fake_train_model_bundle_from_labels(*args, **kwargs):
        assert kwargs["split_strategy"] == "random"
        assert kwargs["split_seed"] == 13
        assert kwargs["evaluation_subreddit"] == "seattle"
        return summary

    monkeypatch.setattr(manager, "_snapshot_label_file", lambda: snapshot_path)
    monkeypatch.setattr(local_bridge, "train_model_bundle_from_labels", fake_train_model_bundle_from_labels)
    monkeypatch.setattr(local_bridge, "load_model", lambda *args, **kwargs: {"model_name": "fake"})

    caplog.set_level(logging.INFO, logger="ask_seattle.local_bridge")
    manager._run_retrain(training_records=12)

    assert any("starting auto retrain training_records=12" in record.message for record in caplog.records)
    assert any("auto retrain complete training_records=12" in record.message for record in caplog.records)
    assert any(
        "auto retrain summary training_records=12" in record.message
        and "high_confidence_precision=0.97" in record.message
        and "high_confidence_predictions=7" in record.message
        and "production_ready=True" in record.message
        for record in caplog.records
    )


def test_upsert_label_record_matches_permalink_when_id_missing(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    permalink = "https://reddit.test/r/test/comments/abc/title/"

    upsert_label_record(
        labels,
        {"id": "", "permalink": permalink, "title": "One", "selftext": "", "label": "askseattle"},
    )
    upsert_label_record(
        labels,
        {
            "id": "abc",
            "permalink": permalink,
            "title": "Two",
            "selftext": "",
            "label": "not_askseattle",
        },
    )

    records = load_jsonl_records(labels)
    found = find_label_record(labels, post_id="abc", permalink=permalink)
    assert len(records) == 1
    assert found is not None
    assert found["label"] == "not_askseattle"


def test_bridge_path_resolution_falls_back_to_project_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project_root"
    target = project_root / "data" / "processed" / "fixture.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n", encoding="utf-8")
    (project_root / "scripts").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(local_bridge, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(project_root / "scripts")

    resolved = resolve_bridge_path("data/processed/fixture.jsonl", must_exist=True)

    assert resolved == target.resolve()
