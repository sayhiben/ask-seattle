from __future__ import annotations

import http.client
import json
from pathlib import Path
from threading import Thread
import threading
from typing import Any

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

    def predict_proba(self, texts: list[object]) -> list[list[float]]:
        return [[0.2, 0.8] for _ in texts]


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
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
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
            {"id": "abc", "title": "Where should I stay?", "selftext": "Visiting"},
        )
        train = request_json(
            port,
            "POST",
            "/train",
            {
                "id": "abc",
                "title": "Where should I stay?",
                "selftext": "Visiting",
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
    assert train["ok"] is True
    assert train["saved"]["label"] == "askseattle"
    assert train["saved"]["post_type"] == "text"
    assert train["saved"]["content_domain"] == "reddit.com"
    assert train["saved"]["is_crosspost"] is False
    assert train["saved"]["collected_at"] == "2026-04-10T20:00:00+00:00"
    assert load_jsonl_records(labels)[0]["id"] == "abc"


def test_bridge_recorded_and_train_uses_last_label(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    labels = tmp_path / "labels.jsonl"

    class Handler(LocalBridgeRequestHandler):
        bridge_config = FakeServer()

    Handler.bridge_config = BridgeConfig.__new__(BridgeConfig)
    Handler.bridge_config.model_path = tmp_path / "fake.joblib"
    Handler.bridge_config.label_path = labels
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = None
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
    Handler.bridge_config.label_lock = threading.Lock()
    Handler.bridge_config.auto_retrain = FakeAutoRetrain()
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


def test_auto_retrain_note_label_saved_returns_status_without_deadlock(tmp_path: Path) -> None:
    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
    manager.bridge_config.label_path = tmp_path / "labels.jsonl"
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


def test_auto_retrain_does_not_immediately_retry_failed_attempt(tmp_path: Path) -> None:
    manager = AutoRetrainManager.__new__(AutoRetrainManager)
    manager.bridge_config = FakeServer()
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
    manager.output_dir = tmp_path / "models"

    snapshot_path = manager._snapshot_label_file()
    try:
        assert lock.entries == 1
        assert load_jsonl_records(snapshot_path) == load_jsonl_records(labels)
    finally:
        snapshot_path.unlink(missing_ok=True)


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
