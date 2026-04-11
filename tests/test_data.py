from pathlib import Path

import pytest

from ask_seattle.data import (
    exact_text_hash,
    load_labeled_posts,
    normalize_label,
    prepare_training_posts,
    post_text,
    write_jsonl_records,
)


def test_normalize_label_accepts_named_labels() -> None:
    assert normalize_label("askseattle") == 1
    assert normalize_label("not_askseattle") == 0


def test_normalize_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        normalize_label("maybe")


def test_post_text_drops_removed_body() -> None:
    assert post_text("Question", "[removed]") == "TITLE: Question\nBODY:"


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
