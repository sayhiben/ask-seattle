from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ask_seattle.data import (
    DELETED_TEXT_MARKERS,
    RawPost,
    append_jsonl_record,
    dedupe_raw_records,
    load_jsonl_records,
    raw_post_from_mapping,
    raw_post_to_record,
    utc_now_iso,
    write_jsonl_records,
)


def reddit_from_env() -> Any:
    try:
        import praw
    except ImportError as exc:
        raise SystemExit("Install Reddit support with: python -m pip install -e '.[reddit]'") from exc

    return praw.Reddit(
        client_id=_required_env("REDDIT_CLIENT_ID"),
        client_secret=_required_env("REDDIT_CLIENT_SECRET"),
        username=_required_env("REDDIT_USERNAME"),
        password=_required_env("REDDIT_PASSWORD"),
        user_agent=_required_env("REDDIT_USER_AGENT"),
    )


def collect_submissions(
    reddit: Any,
    *,
    subreddit_name: str,
    output_path: str | Path,
    limit: int,
    stream: bool,
) -> dict[str, int]:
    output = Path(output_path)
    seen = {record["id"] for record in dedupe_raw_records(load_jsonl_records(output))}
    collected = 0
    skipped = 0
    subreddit = reddit.subreddit(subreddit_name)

    def append_if_new(submission: Any) -> None:
        nonlocal collected, skipped
        if submission.id in seen:
            skipped += 1
            return
        seen.add(submission.id)
        append_jsonl_record(output, raw_post_to_record(raw_post_from_submission(submission)))
        collected += 1

    for submission in subreddit.new(limit=limit):
        append_if_new(submission)

    if stream:
        for submission in subreddit.stream.submissions(skip_existing=True):
            append_if_new(submission)

    return {"collected": collected, "skipped": skipped}


def raw_post_from_submission(submission: Any) -> RawPost:
    title = str(getattr(submission, "title", "") or "")
    selftext = str(getattr(submission, "selftext", "") or "")
    permalink = str(getattr(submission, "permalink", "") or "")
    if permalink and permalink.startswith("/"):
        permalink = f"https://www.reddit.com{permalink}"
    content_status = "deleted_or_removed" if _submission_content_removed(submission) else "available"
    if content_status == "deleted_or_removed":
        title = ""
        selftext = ""

    return RawPost(
        post_id=str(getattr(submission, "id")),
        created_utc=float(getattr(submission, "created_utc", 0) or 0),
        permalink=permalink,
        title=title,
        selftext=selftext,
        subreddit=str(getattr(submission, "subreddit", "") or "") or None,
        url=str(getattr(submission, "url", "") or "") or None,
        content_status=content_status,
        collected_at=utc_now_iso(),
    )


def refresh_deleted_content(
    reddit: Any,
    *,
    raw_path: str | Path,
    output_path: str | Path | None = None,
    max_items: int | None = None,
) -> dict[str, int]:
    records = dedupe_raw_records(load_jsonl_records(raw_path))
    refreshed: list[dict[str, Any]] = []
    checked = 0
    purged = 0
    errors = 0

    for record in records:
        if max_items is not None and checked >= max_items:
            refreshed.append(record)
            continue

        post = raw_post_from_mapping(record)
        try:
            submission = reddit.submission(id=post.post_id)
            checked += 1
            if _submission_content_removed(submission):
                purged += 1
                refreshed.append(
                    raw_post_to_record(
                        RawPost(
                            post_id=post.post_id,
                            created_utc=post.created_utc,
                            permalink=post.permalink,
                            title="",
                            selftext="",
                            subreddit=post.subreddit,
                            url=post.url,
                            content_status="deleted_or_removed",
                            collected_at=post.collected_at,
                            refreshed_at=utc_now_iso(),
                        )
                    )
                )
                continue

            refreshed.append(
                raw_post_to_record(
                    RawPost(
                        post_id=post.post_id,
                        created_utc=float(getattr(submission, "created_utc", post.created_utc) or 0),
                        permalink=post.permalink,
                        title=str(getattr(submission, "title", post.title) or ""),
                        selftext=str(getattr(submission, "selftext", post.selftext) or ""),
                        subreddit=post.subreddit,
                        url=str(getattr(submission, "url", post.url) or ""),
                        content_status="available",
                        collected_at=post.collected_at,
                        refreshed_at=utc_now_iso(),
                    )
                )
            )
        except Exception:
            errors += 1
            refreshed.append(record)

    write_jsonl_records(output_path or raw_path, refreshed)
    return {"checked": checked, "purged": purged, "errors": errors, "total": len(records)}


def _submission_content_removed(submission: Any) -> bool:
    title = str(getattr(submission, "title", "") or "").strip().lower()
    selftext = str(getattr(submission, "selftext", "") or "").strip().lower()
    removed_by_category = getattr(submission, "removed_by_category", None)
    return bool(title in DELETED_TEXT_MARKERS or selftext in DELETED_TEXT_MARKERS or removed_by_category)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value
