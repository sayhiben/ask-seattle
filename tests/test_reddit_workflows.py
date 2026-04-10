from pathlib import Path

from ask_seattle.data import load_jsonl_records, write_jsonl_records
from ask_seattle.reddit_data import collect_submissions, refresh_deleted_content
from ask_seattle.reddit_stream import process_submission


class FakeClassifier:
    classes_ = [0, 1]


class FakeModel:
    named_steps = {"classifier": FakeClassifier()}

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.9] for _ in texts]


class FakeSubmission:
    def __init__(self, post_id: str, title: str, selftext: str = "body") -> None:
        self.id = post_id
        self.title = title
        self.selftext = selftext
        self.created_utc = 1
        self.permalink = f"/r/test/comments/{post_id}/title/"
        self.subreddit = "test"
        self.url = f"https://reddit.test/{post_id}"
        self.removed_by_category = None
        self.mod = WriteTrap()

    def reply(self, message: str) -> None:
        raise AssertionError("no-write mode must not reply")


class WriteTrap:
    def remove(self, spam: bool = False) -> None:
        raise AssertionError("no-write mode must not remove")

    def approve(self) -> None:
        raise AssertionError("no-write mode must not approve")


class FakeStream:
    def submissions(self, skip_existing: bool = True) -> list[FakeSubmission]:
        return []


class FakeSubreddit:
    def __init__(self, submissions: list[FakeSubmission]) -> None:
        self._submissions = submissions
        self.stream = FakeStream()

    def new(self, limit: int) -> list[FakeSubmission]:
        return self._submissions[:limit]


class FakeReddit:
    def __init__(self, submissions: list[FakeSubmission]) -> None:
        self._submissions = {submission.id: submission for submission in submissions}
        self._subreddit = FakeSubreddit(submissions)

    def subreddit(self, name: str) -> FakeSubreddit:
        return self._subreddit

    def submission(self, id: str) -> FakeSubmission:
        return self._submissions[id]


def test_collect_submissions_writes_raw_jsonl_without_author(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    reddit = FakeReddit([FakeSubmission("a", "Where should I stay?")])

    result = collect_submissions(
        reddit,
        subreddit_name="test",
        output_path=output,
        limit=10,
        stream=False,
    )
    records = load_jsonl_records(output)

    assert result == {"collected": 1, "skipped": 0}
    assert records[0]["id"] == "a"
    assert "author" not in records[0]


def test_refresh_deleted_content_blanks_removed_text(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    submission = FakeSubmission("a", "Old title", "[removed]")
    reddit = FakeReddit([submission])
    write_jsonl_records(
        raw,
        [{"id": "a", "created_utc": 1, "permalink": "https://reddit.test/a", "title": "Old"}],
    )

    result = refresh_deleted_content(reddit, raw_path=raw)
    records = load_jsonl_records(raw)

    assert result["purged"] == 1
    assert records[0]["title"] == ""
    assert records[0]["selftext"] == ""


def test_process_submission_logs_decision_without_reddit_writes(tmp_path: Path) -> None:
    bundle = {
        "model": FakeModel(),
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "threshold": 0.85,
    }

    event = process_submission(FakeSubmission("a", "Where should I stay?"), bundle, tmp_path)

    assert event["decision"]["post_id"] == "a"
    assert event["decision"]["should_flag"] is True
    assert list((tmp_path / "decisions").glob("*.jsonl"))
