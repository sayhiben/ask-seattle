from __future__ import annotations

from pathlib import Path

from ask_seattle.model import (
    ConfidenceBandMetrics,
    DecisionThresholds,
    ThresholdSelection,
)
from ask_seattle.data import write_jsonl_records
import ask_seattle.training as training
from ask_seattle.training import benchmark_model_variants_from_labels, train_model_bundle_from_labels


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


def test_train_model_bundle_requires_test_precision_for_production_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    monkeypatch.setattr(
        training,
        "_select_thresholds_or_default",
        lambda *args, **kwargs: DecisionThresholds(
            low_threshold=0.5,
            high_threshold=0.8,
            high_threshold_selection=ThresholdSelection(
                threshold=0.8,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                support=2,
                production_ready=True,
            ),
            low_threshold_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
            high_threshold_sweep=[],
            low_threshold_sweep=[],
            abstain_enabled=True,
        ),
    )
    monkeypatch.setattr(
        training,
        "evaluate_decision_policy",
        lambda *args, **kwargs: ConfidenceBandMetrics(
            high_confidence_precision=0.5,
            high_confidence_recall=1.0,
            high_confidence_f1=2 / 3,
            support=2,
            band_counts={"high": 2, "borderline": 0, "low": 0},
        ),
    )

    summary = train_model_bundle_from_labels(labels_path, tmp_path)

    assert summary["production_ready"] is False
    assert summary["production_ready_blocked_reason"] == "high_precision_target_not_met_on_test"


def test_train_model_bundle_from_labels_can_evaluate_only_one_subreddit(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    records = [
        {
            "id": "ask0",
            "title": "Moving advice",
            "selftext": "Need recommendations",
            "label": "askseattle",
            "subreddit": "askseattle",
            "created_utc": 0.0,
        },
        {
            "id": "sea1",
            "title": "Traffic update",
            "selftext": "Road closure downtown",
            "label": "not_askseattle",
            "subreddit": "seattle",
            "created_utc": 1.0,
        },
        {
            "id": "sea2",
            "title": "Neighborhood advice",
            "selftext": "Where should I live?",
            "label": "askseattle",
            "subreddit": "seattle",
            "created_utc": 2.0,
        },
        {
            "id": "sea3",
            "title": "Best coffee",
            "selftext": "Need recommendations",
            "label": "askseattle",
            "subreddit": "seattle",
            "created_utc": 3.0,
        },
        {
            "id": "ask4",
            "title": "Late askseattle positive",
            "selftext": "Visiting next week",
            "label": "askseattle",
            "subreddit": "askseattle",
            "created_utc": 4.0,
        },
        {
            "id": "sea4",
            "title": "City budget update",
            "selftext": "Council discussion",
            "label": "not_askseattle",
            "subreddit": "seattle",
            "created_utc": 5.0,
        },
        {
            "id": "sea5",
            "title": "Weekend itinerary help",
            "selftext": "What should I do?",
            "label": "askseattle",
            "subreddit": "seattle",
            "created_utc": 6.0,
        },
    ]
    write_jsonl_records(labels_path, records)

    summary = train_model_bundle_from_labels(
        labels_path,
        tmp_path,
        evaluation_subreddit="seattle",
    )

    assert summary["split"]["split_strategy"] == "time_eval_subreddit"
    assert summary["split"]["evaluation_subreddit"] == "seattle"
    assert summary["threshold_policy"]["evaluation_subreddit"] == "seattle"


def test_benchmark_model_variants_writes_aggregate_summary(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    records = [
        {
            "id": f"ask{index}",
            "title": f"Where should I stay {index}?",
            "selftext": "Visiting Seattle soon and need recommendations",
            "label": "askseattle",
            "subreddit": "askseattle",
            "created_utc": float(index * 3),
        }
        for index in range(8)
    ] + [
        {
            "id": f"sea_pos{index}",
            "title": f"Best coffee {index}?",
            "selftext": "Any suggestions in Seattle?",
            "label": "askseattle",
            "subreddit": "seattle",
            "created_utc": float(index * 3 + 1),
        }
        for index in range(8)
    ] + [
        {
            "id": f"sea_neg{index}",
            "title": f"Local update {index}",
            "selftext": "Policy discussion and civic news",
            "label": "not_askseattle",
            "subreddit": "seattle",
            "created_utc": float(index * 3 + 2),
        }
        for index in range(8)
    ]
    write_jsonl_records(labels_path, records)

    summary = benchmark_model_variants_from_labels(
        labels_path,
        tmp_path / "variants",
        evaluation_subreddit="seattle",
    )

    assert summary["evaluation_subreddit"] == "seattle"
    assert [variant["name"] for variant in summary["variants"]] == [
        "legacy_baseline",
        "extra_stopwords_only",
        "lower_char_weight_only",
        "recommended",
    ]
    assert Path(tmp_path / "variants" / "variant_benchmark_summary.json").exists()
