from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


POSITIVE_LABELS = {"1", "true", "yes", "ask", "askseattle", "ask_seattle"}
NEGATIVE_LABELS = {"0", "false", "no", "not", "not_askseattle", "not_ask_seattle"}
DELETED_TEXT_MARKERS = {"[deleted]", "[removed]", "[deleted by user]"}
MEDIA_POST_TYPES = frozenset({"image", "link"})
LOW_TEXT_BODY_CHAR_THRESHOLD = 80
URL_PLACEHOLDER = "URL"
DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN = True
DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS = True
URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?"
)


@dataclass(frozen=True)
class LabeledPost:
    title: str
    selftext: str
    label: int
    post_id: str | None = None
    subreddit: str | None = None
    permalink: str | None = None
    post_type: str | None = None
    content_domain: str | None = None
    is_crosspost: bool | None = None
    created_utc: float | None = None
    time_key: float | None = None
    time_source: str | None = None
    text_hash: str | None = None


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


def normalize_review_label(value: Any) -> str:
    if isinstance(value, bool):
        return label_name(int(value))
    if isinstance(value, int) and value in {0, 1}:
        return label_name(value)

    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return "askseattle"
    if normalized in NEGATIVE_LABELS:
        return "not_askseattle"

    msg = (
        f"Unsupported review label {value!r}; expected one of "
        f"{sorted(POSITIVE_LABELS | NEGATIVE_LABELS)}"
    )
    raise ValueError(msg)


def post_metadata_text(
    *,
    title: str | None = None,
    selftext: str | None = None,
    post_type: str | None = None,
    content_domain: str | None = None,
    is_crosspost: Any = None,
    include_sparse_media_token: bool = DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    include_image_low_text_tokens: bool = DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
) -> str:
    normalized_title = str(title or "").strip()
    normalized_body = normalize_body(selftext).strip()
    has_body = bool(normalized_body)
    low_text = is_low_text_body(normalized_body)
    tokens = [
        f"HAS_BODY:{'yes' if has_body else 'no'}",
        f"TITLE_LEN_BUCKET:{title_length_bucket(normalized_title)}",
        f"BODY_LEN_BUCKET:{body_length_bucket(normalized_body)}",
        f"HAS_QUESTION_MARK:{'yes' if has_question_mark(normalized_title, normalized_body) else 'no'}",
        f"LOW_TEXT:{'yes' if low_text else 'no'}",
    ]

    normalized_post_type = _normalize_metadata_token(post_type)
    if normalized_post_type:
        tokens.append(f"POST_TYPE:{normalized_post_type}")

    normalized_domain = _normalize_metadata_token(content_domain)
    if normalized_domain:
        if normalized_domain.startswith("www_"):
            normalized_domain = normalized_domain[4:]
        tokens.append(f"CONTENT_DOMAIN:{normalized_domain}")

    normalized_crosspost = _normalize_boolean_token(is_crosspost)
    if normalized_crosspost is not None:
        tokens.append(f"CROSSPOST:{normalized_crosspost}")

    normalized_post_type = _normalize_metadata_token(post_type) or ""
    sparse_media = is_sparse_media_post(post_type=post_type, selftext=normalized_body)
    if include_sparse_media_token and sparse_media:
        tokens.append("SPARSE_MEDIA:yes")
    if include_image_low_text_tokens and normalized_post_type == "image" and not has_body:
        tokens.append("IMAGE_NO_BODY:yes")
    if include_image_low_text_tokens and normalized_post_type == "image" and low_text:
        tokens.append("LOW_TEXT_IMAGE:yes")

    return " ".join(tokens)


def post_text(
    title: str,
    selftext: str | None = None,
    *,
    post_type: str | None = None,
    content_domain: str | None = None,
    is_crosspost: Any = None,
    include_sparse_media_token: bool = DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    include_image_low_text_tokens: bool = DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
) -> str:
    body = normalize_body(selftext).strip()
    metadata = post_metadata_text(
        title=title,
        selftext=body,
        post_type=post_type,
        content_domain=content_domain,
        is_crosspost=is_crosspost,
        include_sparse_media_token=include_sparse_media_token,
        include_image_low_text_tokens=include_image_low_text_tokens,
    )

    parts = [f"TITLE: {str(title).strip()}"]
    if metadata:
        parts.append(metadata)
    parts.append(f"BODY: {body}")
    return "\n".join(parts).strip()


def label_name(label: int) -> str:
    return "askseattle" if label == 1 else "not_askseattle"


def normalize_body(value: str | None) -> str:
    body = "" if value is None else str(value)
    if body.strip().lower() in DELETED_TEXT_MARKERS:
        return ""
    return body


def normalize_urls_for_lexical_text(value: str | None, *, replacement: str = URL_PLACEHOLDER) -> str:
    text = "" if value is None else str(value)
    normalized = URL_PATTERN.sub(replacement, text)
    return re.sub(r"\s+", " ", normalized).strip()


def title_length_bucket(title: str | None) -> str:
    length = _normalized_text_length(title)
    if length < 40:
        return "short"
    if length < 90:
        return "medium"
    return "long"


def body_length_bucket(selftext: str | None) -> str:
    length = _normalized_text_length(normalize_body(selftext))
    if length == 0:
        return "none"
    if length < LOW_TEXT_BODY_CHAR_THRESHOLD:
        return "short"
    if length < 280:
        return "medium"
    return "long"


def is_low_text_body(selftext: str | None) -> bool:
    return body_length_bucket(selftext) in {"none", "short"}


def has_question_mark(title: str | None, selftext: str | None = None) -> bool:
    return "?" in str(title or "") or "?" in normalize_body(selftext)


def is_sparse_media_post(*, post_type: str | None = None, selftext: str | None = None) -> bool:
    normalized_post_type = _normalize_metadata_token(post_type)
    return bool(normalized_post_type in MEDIA_POST_TYPES and is_low_text_body(selftext))


def exact_text_hash(title: str, selftext: str | None = None) -> str:
    normalized = _normalize_text_for_hash(title, selftext)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def derive_time_key(row: dict[str, Any]) -> tuple[float | None, str | None]:
    explicit = _float_or_none(row.get("time_key"))
    if explicit is not None:
        return explicit, str(row.get("time_source") or "time_key")

    created_utc = _float_or_none(row.get("created_utc"))
    if created_utc is not None:
        return created_utc, "created_utc"

    for field_name in ("collected_at", "retrieved_at"):
        timestamp = _parse_timestamp(row.get(field_name))
        if timestamp is not None:
            return timestamp, field_name

    return None, None


def load_labeled_posts(path: str | Path) -> list[LabeledPost]:
    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        return _load_jsonl(data_path)

    msg = f"Unsupported data file type for {data_path}; use .jsonl"
    raise ValueError(msg)


def _post_from_mapping(row: dict[str, Any], source: str) -> LabeledPost:
    try:
        title = row["title"]
        label = row["label"]
    except KeyError as exc:
        msg = f"{source} is missing required field {exc.args[0]!r}"
        raise ValueError(msg) from exc

    selftext = normalize_body(str(row.get("selftext") or row.get("body") or ""))
    time_key, time_source = derive_time_key(row)

    return LabeledPost(
        title=str(title),
        selftext=selftext,
        label=normalize_label(normalize_review_label(label)),
        post_id=str(row["id"]) if row.get("id") else None,
        subreddit=str(row["subreddit"]) if row.get("subreddit") else None,
        permalink=str(row["permalink"]) if row.get("permalink") else None,
        post_type=str(row["post_type"]) if row.get("post_type") else None,
        content_domain=str(row["content_domain"]) if row.get("content_domain") else None,
        is_crosspost=_bool_or_none(row.get("is_crosspost")),
        created_utc=_float_or_none(row.get("created_utc")),
        time_key=time_key,
        time_source=time_source,
        text_hash=str(row.get("text_hash") or exact_text_hash(str(title), selftext)),
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


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


def prepare_training_records(input_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized_records = [
        _normalized_review_record(record)
        for record in _load_review_records(input_path)
        if str(record.get("label") or "").strip()
    ]

    identity_records: list[dict[str, Any]] = []
    identity_replaced = 0
    for record in normalized_records:
        before = len(identity_records)
        identity_records = _upsert_records(identity_records, record, _identity_keys)
        if len(identity_records) == before:
            identity_replaced += 1

    deduped_records: list[dict[str, Any]] = []
    text_hash_replaced = 0
    for record in identity_records:
        before = len(deduped_records)
        deduped_records = _upsert_records(deduped_records, record, _text_hash_keys)
        if len(deduped_records) == before:
            text_hash_replaced += 1

    ordered_training_records = _sorted_review_records(deduped_records)
    missing_time_key = sum(1 for record in deduped_records if "time_key" not in record)
    summary = {
        "loaded": len(normalized_records),
        "identity_replaced": identity_replaced,
        "text_hash_replaced": text_hash_replaced,
        "training_records": len(ordered_training_records),
        "missing_time_key": missing_time_key,
    }
    return ordered_training_records, summary


def prepare_training_posts(input_path: str | Path) -> tuple[list[LabeledPost], dict[str, int]]:
    records, summary = prepare_training_records(input_path)
    source = str(Path(input_path))
    posts = [_post_from_mapping(record, f"{source}:prepared") for record in records]
    return posts, summary


def _load_review_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        return load_jsonl_records(data_path)
    msg = f"Unsupported data file type for {data_path}; use .jsonl"
    raise ValueError(msg)


def _normalized_review_record(row: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("title") or "").strip()
    if not title:
        raise ValueError("Reviewed label record is missing title")

    normalized_label = normalize_review_label(row.get("label"))
    selftext = normalize_body(row.get("selftext") or row.get("body") or "")
    record = {
        "id": str(row.get("id") or "").strip(),
        "created_utc": row.get("created_utc") or "",
        "permalink": str(row.get("permalink") or "").strip(),
        "title": title,
        "selftext": selftext,
        "label": normalized_label,
        "notes": str(row.get("notes") or ""),
        "text_hash": exact_text_hash(title, selftext),
    }
    for optional_field in (
        "collected_at",
        "retrieved_at",
        "subreddit",
        "post_type",
        "content_domain",
        "content_href",
        "capture_context",
        "source",
        "is_crosspost",
        "time_source",
        "time_key",
    ):
        if optional_field in row and row.get(optional_field) not in ("", None):
            record[optional_field] = row.get(optional_field)

    time_key, time_source = derive_time_key(record)
    if time_key is not None:
        record["time_key"] = time_key
        record["time_source"] = time_source
    else:
        record.pop("time_key", None)
        record.pop("time_source", None)

    return record


def _normalize_text_for_hash(title: str, selftext: str | None = None) -> str:
    def collapse(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    return f"title:{collapse(str(title))}\nbody:{collapse(normalize_body(selftext))}"


def _normalized_text_length(value: str | None) -> int:
    collapsed = re.sub(r"\s+", " ", str(value or "").strip())
    return len(collapsed)


def _identity_keys(record: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    record_id = str(record.get("id") or "").strip()
    permalink = str(record.get("permalink") or "").strip()
    if record_id:
        keys.add(f"id:{record_id}")
    if permalink:
        keys.add(f"permalink:{permalink}")
    return keys


def _text_hash_keys(record: dict[str, Any]) -> set[str]:
    text_hash = str(record.get("text_hash") or "").strip()
    return {f"text_hash:{text_hash}"} if text_hash else set()


def _upsert_records(
    records: list[dict[str, Any]],
    new_record: dict[str, Any],
    key_fn: Any,
) -> list[dict[str, Any]]:
    new_keys = key_fn(new_record)
    if not new_keys:
        return [*records, new_record]

    output = [record for record in records if not key_fn(record) & new_keys]
    output.append(new_record)
    return output


def _sorted_review_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[float, str, str, str]:
        time_key = float(record.get("time_key")) if record.get("time_key") is not None else float("inf")
        return (
            time_key,
            str(record.get("collected_at") or ""),
            str(record.get("id") or ""),
            str(record.get("permalink") or ""),
        )

    return sorted(records, key=sort_key)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
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


def _normalize_boolean_token(value: Any) -> str | None:
    normalized = _bool_or_none(value)
    if normalized is None:
        return None
    return "yes" if normalized else "no"


def _normalize_metadata_token(value: Any) -> str | None:
    if value in ("", None):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or None


def _parse_timestamp(value: Any) -> float | None:
    if value in ("", None):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 1000 if numeric > 10_000_000_000 else numeric

    try:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(normalized)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.timestamp()
    except ValueError:
        return None
