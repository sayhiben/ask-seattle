from __future__ import annotations

import json
import logging
import tempfile
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ask_seattle.data import (
    load_jsonl_records,
    normalize_review_label,
    prepare_training_records,
    utc_now_iso,
    write_jsonl_records,
)
from ask_seattle.model import load_model
from ask_seattle.model import classify_post
from ask_seattle.training import train_model_bundle_from_labels

LOGGER = logging.getLogger("ask_seattle.local_bridge")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BridgeConfig:
    def __init__(
        self,
        *,
        model_path: Path,
        label_path: Path,
        comparison_suite_path: Path | None = None,
        retrain_every: int = 0,
        split_strategy: str = "random",
        split_seed: int = 13,
        evaluation_subreddit: str | None = None,
    ) -> None:
        self.model_path = resolve_bridge_path(model_path, must_exist=True)
        self.label_path = resolve_bridge_path(label_path, must_exist=False)
        self.comparison_suite_path = (
            resolve_bridge_path(comparison_suite_path, must_exist=False)
            if comparison_suite_path is not None
            else None
        )
        self.split_strategy = split_strategy
        self.split_seed = split_seed
        self.evaluation_subreddit = evaluation_subreddit
        self.label_lock = threading.Lock()
        LOGGER.info("loading model from %s", self.model_path)
        self.bundle = load_model(self.model_path)
        self.comparison_models = _load_comparison_models(
            primary_bundle=self.bundle,
            primary_model_path=self.model_path,
            comparison_suite_path=self.comparison_suite_path,
        )
        LOGGER.info("labels will append to %s", self.label_path)
        self.auto_retrain = None
        if retrain_every > 0:
            self.auto_retrain = AutoRetrainManager(
                bridge_config=self,
                retrain_every=retrain_every,
            )


def run_bridge(
    *,
    host: str,
    port: int,
    model_path: str | Path,
    label_path: str | Path,
    comparison_suite_path: str | Path | None = None,
    log_level: str = "INFO",
    retrain_every: int = 0,
    split_strategy: str = "random",
    split_seed: int = 13,
    evaluation_subreddit: str | None = None,
) -> None:
    configure_logging(log_level)
    config = BridgeConfig(
        model_path=Path(model_path),
        label_path=Path(label_path),
        comparison_suite_path=Path(comparison_suite_path) if comparison_suite_path else None,
        retrain_every=retrain_every,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )

    class RequestHandler(LocalBridgeRequestHandler):
        bridge_config = config

    server = ThreadingHTTPServer((host, port), RequestHandler)
    LOGGER.info(
        "starting local bridge host=%s port=%s model_path=%s label_path=%s split_strategy=%s split_seed=%s evaluation_subreddit=%s comparison_models=%s",
        host,
        port,
        config.model_path,
        config.label_path,
        config.split_strategy,
        config.split_seed,
        config.evaluation_subreddit or "",
        len(config.comparison_models),
    )
    print(
        "local_bridge="
        + json.dumps(
            {
                "host": host,
                "port": port,
                "model_path": str(config.model_path),
                "label_path": str(config.label_path),
                "comparison_suite_path": (
                    str(config.comparison_suite_path) if config.comparison_suite_path else None
                ),
                "comparison_models": _comparison_model_summaries(config.comparison_models),
                "split_strategy": config.split_strategy,
                "split_seed": config.split_seed,
                "evaluation_subreddit": config.evaluation_subreddit,
                "auto_retrain": config.auto_retrain.status_snapshot() if config.auto_retrain else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("local bridge stopped by keyboard interrupt")
        raise


class LocalBridgeRequestHandler(BaseHTTPRequestHandler):
    bridge_config: BridgeConfig
    server_version = "AskSeattleLocalBridge/0.1"

    def do_OPTIONS(self) -> None:
        LOGGER.debug("OPTIONS %s", self.path)
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        LOGGER.info("GET %s", self.path)
        if self.path == "/health":
            self._send_json(
                {
                    "ok": True,
                    "model_path": str(self.bridge_config.model_path),
                    "label_path": str(self.bridge_config.label_path),
                    "comparison_suite_path": (
                        str(self.bridge_config.comparison_suite_path)
                        if self.bridge_config.comparison_suite_path
                        else None
                    ),
                    "comparison_models": _comparison_model_summaries(self.bridge_config.comparison_models),
                    "split_strategy": self.bridge_config.split_strategy,
                    "split_seed": self.bridge_config.split_seed,
                    "evaluation_subreddit": self.bridge_config.evaluation_subreddit,
                    "auto_retrain": (
                        self.bridge_config.auto_retrain.status_snapshot()
                        if self.bridge_config.auto_retrain
                        else None
                    ),
                }
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            LOGGER.info(
                "POST %s id=%s title=%r selftext_chars=%s",
                self.path,
                payload.get("id") or "",
                _truncate(str(payload.get("title") or ""), 80),
                len(str(payload.get("selftext") or "")),
            )
            if self.path == "/check":
                self._handle_check(payload)
                return
            if self.path == "/check-comparison":
                self._handle_check_comparison(payload)
                return
            if self.path == "/train":
                self._handle_train(payload)
                return
            if self.path == "/recorded":
                self._handle_recorded(payload)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
        except ValueError as exc:
            LOGGER.warning("bad request path=%s error=%s", self.path, exc)
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            LOGGER.exception("bridge request failed path=%s", self.path)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug(format, *args)

    def _handle_check(self, payload: dict[str, Any]) -> None:
        title = _required_string(payload, "title")
        selftext = str(payload.get("selftext") or "")
        post_type = _optional_string(payload, "post_type")
        content_domain = _optional_string(payload, "content_domain")
        is_crosspost = _optional_bool(payload, "is_crosspost")
        post_id = _optional_string(payload, "id")
        permalink = _optional_string(payload, "permalink")
        time_source = _request_time_source(payload)
        result = classify_post(
            self.bridge_config.bundle,
            title=title,
            selftext=selftext,
            post_type=post_type,
            content_domain=content_domain,
            is_crosspost=is_crosspost,
            post_id=post_id,
            permalink=permalink,
            time_source=time_source,
        )
        LOGGER.info(
            "check result id=%s label=%s confidence_band=%s score=%.3f high_threshold=%.3f",
            result.post_id or "",
            result.label,
            result.confidence_band,
            result.score,
            result.high_threshold,
        )
        include_comparisons = bool(payload.get("include_comparisons") is True)
        comparisons = []
        if include_comparisons:
            comparisons = [
                _comparison_result_entry(
                    comparison,
                    title=title,
                    selftext=selftext,
                    post_type=post_type,
                    content_domain=content_domain,
                    is_crosspost=is_crosspost,
                    post_id=post_id,
                    permalink=permalink,
                    time_source=time_source,
                )
                for comparison in self.bridge_config.comparison_models
            ]
        self._send_json(
            {
                "ok": True,
                "result": asdict(result),
                "comparison_models": _comparison_model_summaries(self.bridge_config.comparison_models),
                "comparisons": comparisons,
            }
        )

    def _handle_check_comparison(self, payload: dict[str, Any]) -> None:
        comparison_name = _required_string(payload, "name")
        title = _required_string(payload, "title")
        selftext = str(payload.get("selftext") or "")
        post_type = _optional_string(payload, "post_type")
        content_domain = _optional_string(payload, "content_domain")
        is_crosspost = _optional_bool(payload, "is_crosspost")
        post_id = _optional_string(payload, "id")
        permalink = _optional_string(payload, "permalink")
        time_source = _request_time_source(payload)
        comparison = _find_comparison_model(self.bridge_config.comparison_models, comparison_name)
        if comparison is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"unknown comparison model: {comparison_name}")
            return
        entry = _comparison_result_entry(
            comparison,
            title=title,
            selftext=selftext,
            post_type=post_type,
            content_domain=content_domain,
            is_crosspost=is_crosspost,
            post_id=post_id,
            permalink=permalink,
            time_source=time_source,
        )
        LOGGER.info(
            "comparison result name=%s label=%s confidence_band=%s score=%s error=%s",
            comparison_name,
            ((entry.get("result") or {}).get("label") if isinstance(entry.get("result"), dict) else ""),
            ((entry.get("result") or {}).get("confidence_band") if isinstance(entry.get("result"), dict) else ""),
            ((entry.get("result") or {}).get("score") if isinstance(entry.get("result"), dict) else ""),
            entry.get("error") or "",
        )
        self._send_json({"ok": True, "comparison": entry})

    def _handle_train(self, payload: dict[str, Any]) -> None:
        title = _required_string(payload, "title")
        normalized_label = normalize_review_label(_required_string(payload, "label"))
        record = {
            "id": payload.get("id") or "",
            "created_utc": payload.get("created_utc") or "",
            "permalink": payload.get("permalink") or "",
            "title": title,
            "selftext": str(payload.get("selftext") or ""),
            "label": normalized_label,
            "source": "tampermonkey",
            "notes": payload.get("notes") or "",
            "collected_at": payload.get("collected_at") or utc_now_iso(),
        }
        for optional_field in (
            "subreddit",
            "post_type",
            "content_href",
            "content_domain",
            "is_crosspost",
            "capture_context",
        ):
            if optional_field in payload:
                record[optional_field] = payload.get(optional_field)
        with self.bridge_config.label_lock:
            result = upsert_label_record(self.bridge_config.label_path, record)
        auto_retrain = (
            self.bridge_config.auto_retrain.note_label_saved()
            if self.bridge_config.auto_retrain
            else None
        )
        LOGGER.info(
            "saved label id=%s label=%s path=%s replaced=%s total=%s",
            record["id"],
            record["label"],
            self.bridge_config.label_path,
            result["replaced"],
            result["total"],
        )
        self._send_json(
            {
                "ok": True,
                "saved": record,
                "label_path": str(self.bridge_config.label_path),
                "replaced": result["replaced"],
                "auto_retrain": auto_retrain,
            }
        )

    def _handle_recorded(self, payload: dict[str, Any]) -> None:
        with self.bridge_config.label_lock:
            record = find_label_record(
                self.bridge_config.label_path,
                post_id=_optional_string(payload, "id"),
                permalink=_optional_string(payload, "permalink"),
            )
        LOGGER.info(
            "recorded lookup id=%s found=%s",
            payload.get("id") or "",
            record is not None,
        )
        self._send_json({"ok": True, "recorded": record is not None, "record": record})

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = str(payload.get(key) or "").strip()
    return value or None


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return bool(value)


def _request_time_source(payload: dict[str, Any]) -> str | None:
    explicit = _optional_string(payload, "time_source")
    if explicit:
        return explicit
    if payload.get("created_utc") not in ("", None):
        return "created_utc"
    if payload.get("collected_at") not in ("", None):
        return "collected_at"
    return None


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level {log_level!r}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.setLevel(numeric_level)


class AutoRetrainManager:
    def __init__(
        self,
        *,
        bridge_config: BridgeConfig,
        retrain_every: int,
    ) -> None:
        self.bridge_config = bridge_config
        self.retrain_every = retrain_every
        self.output_dir = (
            self.bridge_config.model_path
            if self.bridge_config.model_path.is_dir()
            else self.bridge_config.model_path.parent
        )
        self.split_strategy = bridge_config.split_strategy
        self.split_seed = bridge_config.split_seed
        self.evaluation_subreddit = bridge_config.evaluation_subreddit
        self.reload_model_path = self.output_dir / "tfidf_logreg.joblib"
        self._state_lock = threading.Lock()
        self._running = False
        self._last_error: str | None = None
        self._last_reload_at: str | None = None
        self._last_summary: dict[str, Any] | None = None
        prepared = self._prepare_outputs()
        self._last_prepared_training_count = prepared["training_records"]
        self._last_retrain_training_count = prepared["training_records"]
        self._last_triggered_training_count = prepared["training_records"]
        LOGGER.info(
            "auto retrain enabled every=%s label_path=%s split_strategy=%s split_seed=%s baseline_training_rows=%s",
            self.retrain_every,
            self.bridge_config.label_path,
            self.split_strategy,
            self.split_seed,
            self._last_retrain_training_count,
        )

    def note_label_saved(self) -> dict[str, Any]:
        prepared = self._prepare_outputs()
        training_records = int(prepared["training_records"])
        with self._state_lock:
            self._last_prepared_training_count = training_records
            if self._running:
                status = self._status_snapshot_locked(scheduled=False)
                self._log_status("label_saved_while_retrain_in_progress", status)
                return status
            delta = training_records - self._last_triggered_training_count
            if delta < self.retrain_every:
                status = self._status_snapshot_locked(scheduled=False)
                self._log_status("label_saved_waiting_for_retrain", status)
                return status
            self._start_retrain_locked(training_records)
            status = self._status_snapshot_locked(scheduled=True)
            self._log_status("label_saved_triggered_retrain", status)
            return status

    def status_snapshot(self, *, scheduled: bool = False) -> dict[str, Any]:
        with self._state_lock:
            return self._status_snapshot_locked(scheduled=scheduled)

    def _status_snapshot_locked(self, *, scheduled: bool) -> dict[str, Any]:
        delta = self._last_prepared_training_count - self._last_triggered_training_count
        labels_until_retrain = max(self.retrain_every - max(delta, 0), 0)
        if self._running:
            labels_until_retrain = 0
        return {
            "enabled": True,
            "scheduled": scheduled,
            "in_progress": self._running,
            "retrain_every": self.retrain_every,
            "training_records": self._last_prepared_training_count,
            "last_retrain_training_records": self._last_retrain_training_count,
            "last_triggered_training_records": self._last_triggered_training_count,
            "labels_until_retrain": labels_until_retrain,
            "label_path": str(self.bridge_config.label_path),
            "output_dir": str(self.output_dir),
            "reload_model_path": str(self.reload_model_path),
            "split_strategy": self.split_strategy,
            "split_seed": self.split_seed,
            "evaluation_subreddit": self.evaluation_subreddit,
            "last_reload_at": self._last_reload_at,
            "last_error": self._last_error,
            "last_summary_path": str(self.output_dir / "training_summary.json"),
        }

    def _prepare_outputs(self) -> dict[str, int]:
        with self.bridge_config.label_lock:
            _, summary = prepare_training_records(self.bridge_config.label_path)
            return summary

    def _start_retrain_locked(self, training_records: int) -> None:
        self._running = True
        self._last_error = None
        self._last_triggered_training_count = training_records
        thread = threading.Thread(
            target=self._run_retrain,
            args=(training_records,),
            daemon=True,
            name="ask-seattle-auto-retrain",
        )
        thread.start()

    def _run_retrain(self, training_records: int) -> None:
        snapshot_path = self._snapshot_label_file()
        LOGGER.info(
            "starting auto retrain training_records=%s output_dir=%s split_strategy=%s split_seed=%s snapshot_path=%s",
            training_records,
            self.output_dir,
            self.split_strategy,
            self.split_seed,
            snapshot_path,
        )
        try:
            summary = train_model_bundle_from_labels(
                snapshot_path,
                self.output_dir,
                split_strategy=self.split_strategy,
                split_seed=self.split_seed,
                evaluation_subreddit=self.evaluation_subreddit,
            )
            self.bridge_config.bundle = load_model(self.reload_model_path)
            self.bridge_config.model_path = self.reload_model_path
            with self._state_lock:
                self._last_retrain_training_count = training_records
                self._last_summary = summary
                self._last_reload_at = utc_now_iso()
                self._last_error = None
                self._running = False
            LOGGER.info(
                "auto retrain complete training_records=%s reloaded_model=%s summary_path=%s",
                training_records,
                self.reload_model_path,
                self.output_dir / "training_summary.json",
            )
            self._log_training_summary(summary, training_records=training_records)
        except Exception as exc:
            LOGGER.exception(
                "auto retrain failed training_records=%s output_dir=%s snapshot_path=%s",
                training_records,
                self.output_dir,
                snapshot_path,
            )
            with self._state_lock:
                self._last_error = str(exc)
                self._running = False
        finally:
            snapshot_path.unlink(missing_ok=True)
        self._schedule_followup_if_needed()

    def _schedule_followup_if_needed(self) -> None:
        with self._state_lock:
            delta = self._last_prepared_training_count - self._last_triggered_training_count
            if self._running or delta < self.retrain_every:
                return
            self._start_retrain_locked(self._last_prepared_training_count)

    def _snapshot_label_file(self) -> Path:
        snapshot_dir = self.output_dir / ".snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="ask-seattle-auto-retrain-",
            suffix=".jsonl",
            dir=snapshot_dir,
            delete=False,
        ) as handle:
            snapshot_path = Path(handle.name)

        with self.bridge_config.label_lock:
            write_jsonl_records(snapshot_path, load_jsonl_records(self.bridge_config.label_path))
        return snapshot_path

    def _log_status(self, event: str, status: dict[str, Any]) -> None:
        LOGGER.info(
            "auto retrain status event=%s scheduled=%s in_progress=%s training_records=%s "
            "last_retrain_training_records=%s last_triggered_training_records=%s labels_until_retrain=%s "
            "split_strategy=%s split_seed=%s evaluation_subreddit=%s last_error=%s",
            event,
            status.get("scheduled"),
            status.get("in_progress"),
            status.get("training_records"),
            status.get("last_retrain_training_records"),
            status.get("last_triggered_training_records"),
            status.get("labels_until_retrain"),
            status.get("split_strategy"),
            status.get("split_seed"),
            status.get("evaluation_subreddit") or "",
            status.get("last_error") or "",
        )

    def _log_training_summary(self, summary: dict[str, Any], *, training_records: int) -> None:
        prepared_data = summary.get("prepared_data") or {}
        split = summary.get("split") or {}
        calibration = summary.get("calibration") or {}
        metrics = summary.get("metrics") or {}
        operating_metrics = summary.get("operating_metrics") or {}
        auto_band = operating_metrics.get("auto_band") or {}
        threshold_policy = summary.get("threshold_policy") or {}
        LOGGER.info(
            "auto retrain summary training_records=%s prepared_training_records=%s "
            "split_train=%s split_calibration=%s split_test=%s split_strategy=%s split_seed=%s "
            "calibration_available=%s high_confidence_precision=%s high_confidence_recall=%s "
            "high_confidence_f1=%s high_confidence_predictions=%s low_threshold=%s high_threshold=%s "
            "evaluation_subreddit=%s production_ready=%s blocked_reason=%s",
            training_records,
            prepared_data.get("training_records"),
            split.get("train"),
            split.get("calibration"),
            split.get("test"),
            split.get("split_strategy"),
            split.get("split_seed"),
            calibration.get("available"),
            metrics.get("high_confidence_precision"),
            metrics.get("high_confidence_recall"),
            metrics.get("high_confidence_f1"),
            auto_band.get("predicted_positive"),
            threshold_policy.get("low_threshold"),
            threshold_policy.get("high_threshold"),
            threshold_policy.get("evaluation_subreddit") or "",
            summary.get("production_ready"),
            summary.get("production_ready_blocked_reason"),
        )


def resolve_bridge_path(path: str | Path, *, must_exist: bool) -> Path:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
        checked = [resolved]
    else:
        cwd_candidate = (Path.cwd() / raw_path).resolve()
        project_candidate = (PROJECT_ROOT / raw_path).resolve()
        checked = [cwd_candidate]
        if project_candidate != cwd_candidate:
            checked.append(project_candidate)

        if cwd_candidate.exists():
            resolved = cwd_candidate
        elif (
            project_candidate.exists()
            or project_candidate.parent.exists()
            or raw_path.parts[:1] in {("data",), ("models",)}
        ):
            LOGGER.info(
                "resolved relative path %s against project root %s instead of cwd %s",
                raw_path,
                PROJECT_ROOT,
                Path.cwd(),
            )
            resolved = project_candidate
        else:
            resolved = cwd_candidate

    if must_exist and not resolved.exists():
        checked_paths = ", ".join(str(candidate) for candidate in checked)
        raise FileNotFoundError(f"Could not find {path!s}. Checked: {checked_paths}")

    return resolved


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _comparison_model_summaries(comparison_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": comparison["name"],
            "display_name": comparison.get("display_name"),
            "model_family": comparison["model_family"],
            "model_id": comparison.get("model_id"),
            "artifact_path": comparison.get("artifact_path"),
        }
        for comparison in comparison_models
    ]


def _find_comparison_model(
    comparison_models: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    for comparison in comparison_models:
        if str(comparison.get("name") or "") == name:
            return comparison
    return None


def _comparison_result_entry(
    comparison: dict[str, Any],
    *,
    title: str,
    selftext: str,
    post_type: str | None,
    content_domain: str | None,
    is_crosspost: bool | None,
    post_id: str | None,
    permalink: str | None,
    time_source: str | None,
) -> dict[str, Any]:
    payload = {
        "name": comparison["name"],
        "display_name": comparison.get("display_name"),
        "model_family": comparison["model_family"],
        "model_id": comparison.get("model_id"),
    }
    try:
        payload["result"] = asdict(
            classify_post(
                comparison["bundle"],
                title=title,
                selftext=selftext,
                post_type=post_type,
                content_domain=content_domain,
                is_crosspost=is_crosspost,
                post_id=post_id,
                permalink=permalink,
                time_source=time_source,
            )
        )
    except Exception as exc:
        LOGGER.exception(
            "comparison model check failed name=%s family=%s model_id=%s",
            comparison["name"],
            comparison["model_family"],
            comparison.get("model_id") or "",
        )
        payload["error"] = str(exc)
    return payload


def _bundle_family(bundle: dict[str, Any]) -> str:
    return str(bundle.get("model_family") or bundle.get("model_type") or "tfidf")


def _load_comparison_models(
    *,
    primary_bundle: dict[str, Any],
    primary_model_path: Path,
    comparison_suite_path: Path | None,
) -> list[dict[str, Any]]:
    if comparison_suite_path is None or not comparison_suite_path.exists():
        return []

    try:
        suite_summary = json.loads(comparison_suite_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("could not parse comparison suite summary at %s: %s", comparison_suite_path, exc)
        return []

    primary_artifact_candidates = {
        str(resolve_bridge_path(primary_model_path, must_exist=False)),
    }
    for candidate in (
        primary_bundle.get("artifact_path"),
        primary_bundle.get("model_dir"),
    ):
        if candidate:
            try:
                primary_artifact_candidates.add(str(resolve_bridge_path(str(candidate), must_exist=False)))
            except Exception:
                primary_artifact_candidates.add(str(candidate))
    loaded: list[dict[str, Any]] = []
    for entry in suite_summary.get("models") or []:
        if entry.get("status") != "ok":
            continue
        family = str(entry.get("model_family") or "").strip()
        artifact_path = entry.get("artifact_path")
        if not family or not artifact_path:
            continue
        try:
            resolved_artifact = resolve_bridge_path(str(artifact_path), must_exist=True)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            LOGGER.warning(
                "skipping comparison model name=%s family=%s artifact_path=%s error=%s",
                entry.get("name") or "",
                family,
                artifact_path,
                exc,
            )
            continue
        if str(resolved_artifact) in primary_artifact_candidates:
            continue
        try:
            bundle = load_model(resolved_artifact)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            LOGGER.warning(
                "skipping comparison model name=%s family=%s artifact_path=%s error=%s",
                entry.get("name") or "",
                family,
                artifact_path,
                exc,
            )
            continue
        loaded.append(
            {
                "name": str(entry.get("name") or bundle.get("model_name") or family),
                "display_name": entry.get("display_name") or bundle.get("display_name") or entry.get("name"),
                "model_family": family,
                "model_id": entry.get("model_id") or bundle.get("model_id"),
                "artifact_path": str(resolved_artifact),
                "bundle": bundle,
            }
        )
    if loaded:
        LOGGER.info(
            "loaded comparison models comparison_suite_path=%s comparisons=%s",
            comparison_suite_path,
            ",".join(comparison["name"] for comparison in loaded),
        )
    return loaded


def find_label_record(
    label_path: str | Path,
    *,
    post_id: str | None,
    permalink: str | None,
) -> dict[str, Any] | None:
    target_keys = _label_lookup_keys({"id": post_id, "permalink": permalink})
    if not target_keys:
        return None

    found: dict[str, Any] | None = None
    for record in load_jsonl_records(label_path):
        if _label_lookup_keys(record) & target_keys:
            found = record
    return found


def upsert_label_record(label_path: str | Path, new_record: dict[str, Any]) -> dict[str, int | bool]:
    records = load_jsonl_records(label_path)
    new_key = _label_record_key(new_record)
    replaced = False
    output: list[dict[str, Any]] = []
    new_keys = _label_lookup_keys(new_record)

    for record in records:
        if new_key and _label_lookup_keys(record) & new_keys:
            if not replaced:
                output.append(new_record)
                replaced = True
            continue
        output.append(record)

    if not replaced:
        output.append(new_record)

    write_jsonl_records(label_path, output)
    return {"replaced": replaced, "total": len(output)}


def _label_record_key(record: dict[str, Any]) -> str | None:
    post_id = str(record.get("id") or "").strip()
    if post_id:
        return f"id:{post_id}"
    permalink = str(record.get("permalink") or "").strip()
    if permalink:
        return f"permalink:{permalink}"
    return None


def _label_lookup_keys(record: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    post_id = str(record.get("id") or "").strip()
    permalink = str(record.get("permalink") or "").strip()
    if post_id:
        keys.add(f"id:{post_id}")
    if permalink:
        keys.add(f"permalink:{permalink}")
    return keys
