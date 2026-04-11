from __future__ import annotations

from pathlib import Path

from ask_seattle.data import write_jsonl_records
from ask_seattle.training import train_model_bundle_from_labels


def test_train_model_bundle_from_labels_writes_time_split_summary_and_feature_audit(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    records = [
        {
            "id": f"a{index}",
            "title": f"Where should I stay {index}?",
            "selftext": "Visiting Seattle soon",
            "label": "askseattle",
            "created_utc": float(index * 2),
        }
        for index in range(8)
    ] + [
        {
            "id": f"n{index}",
            "title": f"Local update {index}",
            "selftext": "Policy discussion and civic news",
            "label": "not_askseattle",
            "created_utc": float(index * 2 + 1),
        }
        for index in range(8)
    ]
    write_jsonl_records(labels_path, records)

    summary = train_model_bundle_from_labels(labels_path, tmp_path)

    assert summary["split"]["split_strategy"] == "time"
    assert summary["split"]["calibration"] > 0
    assert summary["split"]["time_coverage"]["train"]["count"] > 0
    assert summary["prepared_data"]["training_records"] == 16
    assert "feature_audit" in summary
    assert "calibration" in summary
    assert "threshold_selection" in summary
    assert set(summary["metrics"]["confidence_band_counts"]) == {"high", "borderline", "low"}
    assert Path(tmp_path, "training_summary.json").exists()
