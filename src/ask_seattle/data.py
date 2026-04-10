from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


POSITIVE_LABELS = {"1", "true", "yes", "ask", "askseattle", "ask_seattle"}
NEGATIVE_LABELS = {"0", "false", "no", "not", "not_askseattle", "not_ask_seattle"}
DELETED_TEXT_MARKERS = {"[deleted]", "[removed]", "[deleted by user]"}
LABELING_FIELDS = ["id", "created_utc", "permalink", "title", "selftext", "label", "notes"]


@dataclass(frozen=True)
class LabeledPost:
    title: str
    selftext: str
    label: int
    post_id: str | None = None
    subreddit: str | None = None
    permalink: str | None = None
    created_utc: float | None = None


@dataclass(frozen=True)
class RawPost:
    post_id: str
    created_utc: float
    permalink: str
    title: str
    selftext: str
    subreddit: str | None = None
    url: str | None = None
    content_status: str = "available"
    collected_at: str | None = None
    refreshed_at: str | None = None


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value

    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0

    msg = f"Unsupported label {value!r}; expected one of {sorted(POSITIVE_LABELS | NEGATIVE_LABELS)}"
    raise ValueError(msg)


def post_text(title: str, selftext: str | None = None) -> str:
    body = normalize_body(selftext)

    return f"TITLE: {str(title).strip()}\nBODY: {body.strip()}".strip()


def label_name(label: int) -> str:
    return "askseattle" if label == 1 else "not_askseattle"


def normalize_body(value: str | None) -> str:
    body = "" if value is None else str(value)
    if body.strip().lower() in DELETED_TEXT_MARKERS:
        return ""
    return body


def load_labeled_posts(path: str | Path) -> list[LabeledPost]:
    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        return _load_jsonl(data_path)
    if data_path.suffix.lower() == ".csv":
        return _load_csv(data_path)

    msg = f"Unsupported data file type for {data_path}; use .jsonl or .csv"
    raise ValueError(msg)


def _post_from_mapping(row: dict[str, Any], source: str) -> LabeledPost:
    try:
        title = row["title"]
        label = row["label"]
    except KeyError as exc:
        msg = f"{source} is missing required field {exc.args[0]!r}"
        raise ValueError(msg) from exc

    return LabeledPost(
        title=str(title),
        selftext=str(row.get("selftext") or row.get("body") or ""),
        label=normalize_label(label),
        post_id=str(row["id"]) if row.get("id") else None,
        subreddit=str(row["subreddit"]) if row.get("subreddit") else None,
        permalink=str(row["permalink"]) if row.get("permalink") else None,
        created_utc=float(row["created_utc"]) if row.get("created_utc") else None,
    )


def _load_jsonl(path: Path) -> list[LabeledPost]:
    posts: list[LabeledPost] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_number} is not valid JSON"
                raise ValueError(msg) from exc
            if not isinstance(row, dict):
                msg = f"{path}:{line_number} must be a JSON object"
                raise ValueError(msg)
            posts.append(_post_from_mapping(row, f"{path}:{line_number}"))
    return posts


def _load_csv(path: Path) -> list[LabeledPost]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_post_from_mapping(row, f"{path}:{index}") for index, row in enumerate(reader, start=2)]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def raw_post_from_mapping(row: dict[str, Any]) -> RawPost:
    post_id = str(row.get("id") or row.get("post_id") or "").strip()
    if not post_id:
        raise ValueError("Raw post record is missing id")

    return RawPost(
        post_id=post_id,
        created_utc=float(row.get("created_utc") or 0),
        permalink=str(row.get("permalink") or ""),
        title=str(row.get("title") or ""),
        selftext=normalize_body(str(row.get("selftext") or "")),
        subreddit=str(row["subreddit"]) if row.get("subreddit") else None,
        url=str(row["url"]) if row.get("url") else None,
        content_status=str(row.get("content_status") or "available"),
        collected_at=str(row["collected_at"]) if row.get("collected_at") else None,
        refreshed_at=str(row["refreshed_at"]) if row.get("refreshed_at") else None,
    )


def raw_post_to_record(post: RawPost) -> dict[str, Any]:
    return {
        "id": post.post_id,
        "created_utc": post.created_utc,
        "permalink": post.permalink,
        "title": post.title,
        "selftext": post.selftext,
        "subreddit": post.subreddit,
        "url": post.url,
        "content_status": post.content_status,
        "collected_at": post.collected_at,
        "refreshed_at": post.refreshed_at,
    }


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    data_path = Path(path)
    if not data_path.exists():
        return records

    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{data_path}:{line_number} is not valid JSON"
                raise ValueError(msg) from exc
            if not isinstance(record, dict):
                msg = f"{data_path}:{line_number} must be a JSON object"
                raise ValueError(msg)
            records.append(record)
    return records


def write_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def dedupe_raw_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        post = raw_post_from_mapping(record)
        by_id[post.post_id] = raw_post_to_record(post)
    return sorted(by_id.values(), key=lambda item: (float(item.get("created_utc") or 0), item["id"]))


def export_labeling_csv(raw_path: str | Path, output_path: str | Path) -> dict[str, int]:
    records = dedupe_raw_records(load_jsonl_records(raw_path))
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "created_utc": record.get("created_utc", ""),
                    "permalink": record.get("permalink", ""),
                    "title": record.get("title", ""),
                    "selftext": record.get("selftext", ""),
                    "label": "",
                    "notes": "",
                }
            )

    return {"exported": len(records)}


def import_labeling_csv(labeling_path: str | Path, output_path: str | Path) -> dict[str, int]:
    imported = 0
    skipped = 0
    records: list[dict[str, Any]] = []

    with Path(labeling_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"id", "title", "selftext", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{labeling_path} is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            raw_label = str(row.get("label") or "").strip()
            if not raw_label:
                skipped += 1
                continue

            try:
                normalized_label = normalize_label(raw_label)
            except ValueError as exc:
                raise ValueError(f"{labeling_path}:{row_number}: {exc}") from exc

            records.append(
                {
                    "id": row.get("id", ""),
                    "created_utc": row.get("created_utc", ""),
                    "permalink": row.get("permalink", ""),
                    "title": row.get("title", ""),
                    "selftext": normalize_body(row.get("selftext")),
                    "label": label_name(normalized_label),
                    "notes": row.get("notes", ""),
                }
            )
            imported += 1

    write_jsonl_records(output_path, records)
    return {"imported": imported, "skipped": skipped}
