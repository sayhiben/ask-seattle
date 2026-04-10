from pathlib import Path

import pytest

from ask_seattle.data import (
    export_labeling_csv,
    import_labeling_csv,
    load_labeled_posts,
    normalize_label,
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


def test_export_and_import_labeling_csv(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    labeling_path = tmp_path / "labeling.csv"
    imported_path = tmp_path / "train.jsonl"
    write_jsonl_records(
        raw_path,
        [
            {
                "id": "a",
                "created_utc": 1,
                "permalink": "https://reddit.test/a",
                "title": "Where should I stay?",
                "selftext": "Visiting",
            },
            {
                "id": "b",
                "created_utc": 2,
                "permalink": "https://reddit.test/b",
                "title": "Power outage",
                "selftext": "Update",
            },
        ],
    )

    assert export_labeling_csv(raw_path, labeling_path) == {"exported": 2}
    text = labeling_path.read_text(encoding="utf-8")
    text = text.replace("Where should I stay?,Visiting,", "Where should I stay?,Visiting,askseattle")
    labeling_path.write_text(text, encoding="utf-8")

    result = import_labeling_csv(labeling_path, imported_path)
    posts = load_labeled_posts(imported_path)

    assert result == {"imported": 1, "skipped": 1}
    assert len(posts) == 1
    assert posts[0].label == 1
