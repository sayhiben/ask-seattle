from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ask_seattle.decision_log import write_decision_event
from ask_seattle.model import load_model
from ask_seattle.moderation import decide
from ask_seattle.reddit_data import reddit_from_env


def main() -> int:
    subreddit_name = _required_env("REDDIT_SUBREDDIT")
    model_path = _required_env("ASK_SEATTLE_MODEL")
    log_dir = Path(_required_env("ASK_SEATTLE_LOG_DIR"))
    no_write = os.getenv("ASK_SEATTLE_NO_WRITE", os.getenv("ASK_SEATTLE_DRY_RUN", "1"))
    if no_write != "1":
        raise SystemExit("This version only supports no-write shadow mode. Set ASK_SEATTLE_NO_WRITE=1.")

    bundle = load_model(model_path)
    reddit = reddit_from_env()

    subreddit = reddit.subreddit(subreddit_name)
    print(f"watching=r/{subreddit_name} no_write=1 model={model_path}", flush=True)

    for submission in subreddit.stream.submissions(skip_existing=True):
        event = process_submission(submission, bundle, log_dir)
        print(json.dumps(event), flush=True)

    return 0


def process_submission(submission: Any, bundle: dict, log_dir: str | Path) -> dict:
    permalink = str(getattr(submission, "permalink", "") or "")
    if permalink.startswith("/"):
        permalink = f"https://www.reddit.com{permalink}"

    title = str(getattr(submission, "title", "") or "")
    selftext = str(getattr(submission, "selftext", "") or "")
    decision = decide(
        bundle,
        title=title,
        selftext=selftext,
        post_id=str(getattr(submission, "id", "")),
        permalink=permalink,
    )
    log_path = write_decision_event(
        log_dir,
        decision,
        title=title,
        selftext=selftext,
        extra={"reddit_write_mode": "none"},
    )
    return {"decision": asdict(decision), "log_path": str(log_path)}


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value
