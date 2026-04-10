from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ask_seattle.data import append_jsonl_record, load_jsonl_records
from ask_seattle.moderation import ModerationDecision


REVIEW_FIELDS = [
    "created_at",
    "post_id",
    "permalink",
    "model_name",
    "model_version",
    "threshold",
    "score",
    "predicted_label",
    "should_flag",
    "title",
    "selftext",
    "review_label",
    "notes",
]


def daily_decision_log_path(log_dir: str | Path, now: datetime | None = None) -> Path:
    active_now = now or datetime.now(UTC)
    return Path(log_dir) / "decisions" / f"{active_now.date().isoformat()}.jsonl"


def write_decision_event(
    log_dir: str | Path,
    decision: ModerationDecision,
    *,
    title: str = "",
    selftext: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    event = asdict(decision)
    if title:
        event["title"] = title
    if selftext:
        event["selftext"] = selftext
    if extra:
        event.update(extra)

    log_path = daily_decision_log_path(log_dir)
    append_jsonl_record(log_path, event)
    return log_path


def export_review_csv(decision_log_path: str | Path, output_path: str | Path) -> dict[str, int]:
    records = load_jsonl_records(decision_log_path)
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in REVIEW_FIELDS})

    return {"exported": len(records)}
