from pathlib import Path

import pytest

from ask_seattle.data import (
    exact_text_hash,
    load_labeled_posts,
    merge_crosspost_body,
    normalize_label,
    prepare_training_posts,
    post_metadata_text,
    post_text,
    repair_crosspost_records,
    write_jsonl_records,
)


def test_normalize_label_accepts_named_labels() -> None:
    assert normalize_label("askseattle") == 1
    assert normalize_label("not_askseattle") == 0


def test_normalize_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        normalize_label("maybe")


def test_post_text_drops_removed_body() -> None:
    assert (
        post_text("Question", "[removed]")
        == "TITLE: Question\n"
        "HAS_BODY:no TITLE_LEN_BUCKET:short BODY_LEN_BUCKET:none "
        "HAS_QUESTION_MARK:no LOW_TEXT:yes\n"
        "BODY:"
    )


def test_post_text_includes_normalized_metadata_tokens() -> None:
    assert (
        post_text(
            "Question",
            "",
            post_type="Image",
            content_domain="www.Instagram.com",
            is_crosspost=True,
        )
        == "TITLE: Question\n"
        "HAS_BODY:no TITLE_LEN_BUCKET:short BODY_LEN_BUCKET:none "
        "HAS_QUESTION_MARK:no LOW_TEXT:yes POST_TYPE:image "
        "CONTENT_DOMAIN:instagram_com CROSSPOST:yes SPARSE_MEDIA:yes "
        "IMAGE_NO_BODY:yes LOW_TEXT_IMAGE:yes\n"
        "BODY:"
    )
    assert (
        post_metadata_text(
            title="Question",
            selftext="Body here",
            post_type="link",
            content_domain="example.org",
            is_crosspost=False,
        )
        == "HAS_BODY:yes TITLE_LEN_BUCKET:short BODY_LEN_BUCKET:short "
        "HAS_QUESTION_MARK:no LOW_TEXT:yes POST_TYPE:link CONTENT_DOMAIN:example_org "
        "CROSSPOST:no SPARSE_MEDIA:yes"
    )


def test_post_metadata_text_can_disable_sparse_media_token_but_keep_image_low_text_tokens() -> None:
    assert (
        post_metadata_text(
            title="What is this?",
            selftext="",
            post_type="image",
            content_domain="i.redd.it",
            include_sparse_media_token=False,
        )
        == "HAS_BODY:no TITLE_LEN_BUCKET:short BODY_LEN_BUCKET:none "
        "HAS_QUESTION_MARK:yes LOW_TEXT:yes POST_TYPE:image CONTENT_DOMAIN:i_redd_it "
        "IMAGE_NO_BODY:yes LOW_TEXT_IMAGE:yes"
    )


def test_load_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "labels.jsonl"
    data_path.write_text(
        '{"id":"a","title":"Where should I stay?","selftext":"Visiting","label":"askseattle"}\n'
        '{"id":"b","title":"Power outage","selftext":"Update","label":"not_askseattle"}\n',
        encoding="utf-8",
    )

    posts = load_labeled_posts(data_path)

    assert len(posts) == 2
    assert posts[0].label == 1
    assert posts[1].label == 0


def test_prepare_training_posts_derives_time_key(tmp_path: Path) -> None:
    source = tmp_path / "captured.jsonl"
    write_jsonl_records(
        source,
        [
            {
                "id": "a",
                "title": "Where should I stay?",
                "selftext": "Visiting soon",
                "label": "askseattle",
                "collected_at": "2026-04-10T20:00:00+00:00",
                "post_type": "text",
                "content_domain": "reddit.com",
                "is_crosspost": False,
            },
            {
                "id": "b",
                "title": "Neighborhood question",
                "selftext": "Unsure whether this is redirectable",
                "label": "not_askseattle",
                "collected_at": "2026-04-10T21:00:00+00:00",
            },
        ],
    )

    posts, result = prepare_training_posts(source)
    assert result["training_records"] == 2
    assert result["missing_time_key"] == 0
    assert posts[0].time_source == "collected_at"
    assert posts[0].time_key is not None
    assert posts[0].post_type == "text"
    assert posts[0].content_domain == "reddit.com"
    assert posts[0].is_crosspost is False


def test_prepare_training_posts_last_write_wins_and_text_hash_dedupes(tmp_path: Path) -> None:
    source = tmp_path / "captured.jsonl"
    shared_hash = exact_text_hash("Where should I stay?", "Visiting soon")
    write_jsonl_records(
        source,
        [
            {
                "id": "a",
                "permalink": "https://reddit.test/a",
                "title": "Where should I stay?",
                "selftext": "Visiting soon",
                "label": "askseattle",
                "collected_at": "2026-04-10T20:00:00+00:00",
            },
            {
                "id": "a",
                "permalink": "https://reddit.test/a",
                "title": "Where should I stay?",
                "selftext": "Visiting soon",
                "label": "not_askseattle",
                "collected_at": "2026-04-10T20:05:00+00:00",
            },
            {
                "id": "b",
                "permalink": "https://reddit.test/b",
                "title": "Where should I stay?",
                "selftext": "Visiting soon",
                "label": "askseattle",
                "collected_at": "2026-04-10T20:10:00+00:00",
            },
        ],
    )

    posts, result = prepare_training_posts(source)

    assert result["identity_replaced"] == 1
    assert result["text_hash_replaced"] == 1
    assert len(posts) == 1
    assert posts[0].label == 1
    assert posts[0].text_hash == shared_hash


def test_prepare_training_posts_rejects_ambiguous_label(tmp_path: Path) -> None:
    source = tmp_path / "captured.jsonl"
    write_jsonl_records(
        source,
        [
            {
                "id": "a",
                "title": "Where should I stay?",
                "selftext": "Visiting soon",
                "label": "ambiguous",
                "collected_at": "2026-04-10T20:00:00+00:00",
            }
        ],
    )

    with pytest.raises(ValueError):
        prepare_training_posts(source)


def test_merge_crosspost_body_prefers_nonempty_and_dedupes_identical_text() -> None:
    assert merge_crosspost_body("", "Original body") == "Original body"
    assert merge_crosspost_body("Original body", "Original body") == "Original body"
    assert merge_crosspost_body("Crosspost note", "Original body") == "Crosspost note\n\nOriginal body"


def test_repair_crosspost_records_backfills_body_and_drops_safe_duplicate_original() -> None:
    records = [
        {
            "id": "cross",
            "permalink": "https://www.reddit.com/r/Seattle/comments/cross/example/",
            "title": "Need apartment advice",
            "selftext": "",
            "label": "askseattle",
            "post_type": "crosspost",
            "is_crosspost": True,
            "content_href": "/r/AskSeattle/comments/original/example/",
        },
        {
            "id": "original",
            "permalink": "https://www.reddit.com/r/AskSeattle/comments/original/example/",
            "title": "Need apartment advice",
            "selftext": "Looking for neighborhoods with easy transit access.",
            "label": "askseattle",
            "post_type": "text",
            "is_crosspost": False,
        },
    ]

    repaired, summary = repair_crosspost_records(records)

    assert len(repaired) == 1
    assert repaired[0]["id"] == "cross"
    assert repaired[0]["crosspost_body"] == "Looking for neighborhoods with easy transit access."
    assert repaired[0]["selftext"] == "Looking for neighborhoods with easy transit access."
    assert summary["crosspost_rows_hydrated"] == 1
    assert summary["crosspost_duplicates_removed"] == 1


def test_prepare_training_posts_uses_crosspost_body_and_collapses_original_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "captured.jsonl"
    write_jsonl_records(
        source,
        [
            {
                "id": "cross",
                "permalink": "https://www.reddit.com/r/Seattle/comments/cross/example/",
                "title": "Need apartment advice",
                "selftext": "",
                "label": "askseattle",
                "post_type": "crosspost",
                "is_crosspost": True,
                "content_href": "/r/AskSeattle/comments/original/example/",
                "collected_at": "2026-04-10T20:00:00+00:00",
            },
            {
                "id": "original",
                "permalink": "https://www.reddit.com/r/AskSeattle/comments/original/example/",
                "title": "Need apartment advice",
                "selftext": "Looking for neighborhoods with easy transit access.",
                "label": "askseattle",
                "post_type": "text",
                "is_crosspost": False,
                "collected_at": "2026-04-10T20:01:00+00:00",
            },
        ],
    )

    posts, summary = prepare_training_posts(source)

    assert len(posts) == 1
    assert posts[0].post_id == "cross"
    assert posts[0].selftext == "Looking for neighborhoods with easy transit access."
    assert summary["crosspost_rows_hydrated"] == 1
    assert summary["crosspost_duplicates_removed"] == 1


def test_repair_crosspost_records_ignores_distinct_reframed_crosspost_label_difference() -> None:
    records = [
        {
            "id": "cross",
            "permalink": "https://www.reddit.com/r/Seattle/comments/cross/example/",
            "title": "Does anyone in Seattle remember what this accident was like locally?",
            "selftext": "",
            "label": "askseattle",
            "post_type": "crosspost",
            "is_crosspost": True,
            "content_href": "/r/todayilearned/comments/original/example/",
        },
        {
            "id": "original",
            "permalink": "https://www.reddit.com/r/todayilearned/comments/original/example/",
            "title": "TIL that a maintenance decision contributed to Alaska Airlines Flight 261 crashing",
            "selftext": "",
            "label": "not_askseattle",
            "post_type": "link",
            "is_crosspost": False,
        },
    ]

    repaired, summary = repair_crosspost_records(records)

    assert len(repaired) == 2
    assert summary["crosspost_duplicates_removed"] == 0
    assert summary["crosspost_label_conflicts"] == 0
