from __future__ import annotations

from pathlib import Path

from ask_seattle.model import (
    ConfidenceBandMetrics,
    DecisionThresholds,
    ThresholdSelection,
)
from ask_seattle.data import write_jsonl_records
import ask_seattle.training as training
from ask_seattle.training import (
    OperatingMetrics,
    benchmark_model_suite_from_labels,
    benchmark_model_variants_from_labels,
    train_model_bundle_from_labels,
)


def test_train_model_bundle_from_labels_writes_random_split_summary_and_feature_audit(tmp_path: Path) -> None:
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

    assert summary["split"]["split_strategy"] == "random"
    assert summary["split"]["split_seed"] == 13
    assert summary["split"]["calibration"] > 0
    assert summary["split"]["time_coverage"] is None
    assert set(summary["split"]["coverage"]) == {"train", "calibration", "test"}
    assert "post_type" in summary["split"]["coverage"]["train"]
    assert summary["prepared_data"]["training_records"] == 16
    assert "feature_audit" in summary
    assert "training_balance" in summary
    assert summary["training_balance"]["strategy"] == "slice_aware_positive_weighting"
    assert "calibration" in summary
    assert "threshold_selection" in summary
    assert set(summary["metrics"]["confidence_band_counts"]) == {"high", "borderline", "low"}
    assert "slice_metrics" in summary["operating_metrics"]
    assert set(summary["operating_metrics"]["slice_metrics"]) == {"post_type", "low_text", "sparse_media"}
    assert Path(tmp_path, "training_summary.json").exists()


def test_train_model_bundle_from_labels_can_use_explicit_time_split(tmp_path: Path) -> None:
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

    summary = train_model_bundle_from_labels(labels_path, tmp_path, split_strategy="time")

    assert summary["split"]["split_strategy"] == "time"
    assert summary["split"]["split_seed"] is None
    assert summary["split"]["time_coverage"]["train"]["count"] > 0


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
    monkeypatch.setattr(
        training,
        "_operating_metrics_summary",
        lambda *args, **kwargs: OperatingMetrics(
            auto_band={"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "predicted_positive": 5, "support": 2},
            review_queue={"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "predicted_positive": 5, "support": 2},
            queue_counts={"high": 5, "borderline": 0, "low": 0},
            queue_rates={"auto_rate": 1.0, "review_rate": 1.0, "borderline_rate": 0.0},
            positive_prevalence=1.0,
            positive_count=2,
            total_count=2,
            slice_metrics={},
        ),
    )

    summary = train_model_bundle_from_labels(labels_path, tmp_path)

    assert summary["production_ready"] is False
    assert summary["production_ready_blocked_reason"] == "high_precision_target_not_met_on_test"


def test_train_model_bundle_requires_minimum_high_confidence_predictions_for_production_ready(
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
            high_confidence_precision=1.0,
            high_confidence_recall=1.0,
            high_confidence_f1=1.0,
            support=2,
            band_counts={"high": 2, "borderline": 0, "low": 0},
        ),
    )
    monkeypatch.setattr(
        training,
        "_operating_metrics_summary",
        lambda *args, **kwargs: OperatingMetrics(
            auto_band={"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted_positive": 2, "support": 2},
            review_queue={"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted_positive": 2, "support": 2},
            queue_counts={"high": 2, "borderline": 0, "low": 0},
            queue_rates={"auto_rate": 1.0, "review_rate": 1.0, "borderline_rate": 0.0},
            positive_prevalence=1.0,
            positive_count=2,
            total_count=2,
            slice_metrics={},
        ),
    )

    summary = train_model_bundle_from_labels(labels_path, tmp_path)

    assert summary["production_ready"] is False
    assert summary["production_ready_blocked_reason"] == "insufficient_high_confidence_test_predictions"
    assert summary["production_gate"]["minimum_high_confidence_test_predictions"] == 5


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

    assert summary["split"]["split_strategy"] == "random_eval_subreddit"
    assert summary["split"]["split_seed"] == 13
    assert summary["split"]["evaluation_subreddit"] == "seattle"
    assert summary["threshold_policy"]["evaluation_subreddit"] == "seattle"
    assert summary["threshold_policy"]["split_seed"] == 13
    assert summary["production_gate"]["minimum_high_confidence_test_predictions"] == 5


def test_slice_aware_positive_weighting_upweights_sparse_positive_slices() -> None:
    posts = [
        training.LabeledPost(
            title="What is this?",
            selftext="",
            label=1,
            post_type="image",
            content_domain="reddit.com",
            post_id="p1",
        ),
        training.LabeledPost(
            title="Moving advice",
            selftext=(
                "Need neighborhood recommendations for a move this summer. "
                "We are comparing commute options, grocery access, and parking."
            ),
            label=1,
            post_type="text",
            post_id="p2",
        ),
        training.LabeledPost(
            title="Weekend itinerary help",
            selftext=(
                "Looking for restaurants and day-trip ideas around Seattle. "
                "We want museums, ferry rides, and rainy-day backup plans."
            ),
            label=1,
            post_type="text",
            post_id="p3",
        ),
        training.LabeledPost(
            title="City budget update",
            selftext="Council discussion and civic news",
            label=0,
            post_type="link",
            post_id="n1",
        ),
    ]

    weighting = training._slice_aware_positive_weighting(posts)

    assert weighting.sample_weights[0] > 1.0
    assert weighting.sample_weights[1] == 1.0
    assert weighting.summary["bucket_weights"]["post_type"]["image"] > 1.0
    assert weighting.summary["bucket_weights"]["low_text"]["yes"] > 1.0


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
    assert summary["production_gate"]["minimum_high_confidence_test_predictions"] == 5
    assert [variant["name"] for variant in summary["variants"]] == [
        "legacy_baseline",
        "extra_stopwords_only",
        "lower_char_weight_only",
        "recommended",
    ]
    assert Path(tmp_path / "variants" / "variant_benchmark_summary.json").exists()


def test_benchmark_model_suite_writes_aggregate_summary(tmp_path: Path, monkeypatch) -> None:
    labels_path = tmp_path / "labels.jsonl"
    records = [
        {
            "id": f"sea_pos{index}",
            "title": f"Best coffee {index}?",
            "selftext": "Any suggestions in Seattle?",
            "label": "askseattle",
            "subreddit": "seattle",
            "created_utc": float(index * 2),
        }
        for index in range(8)
    ] + [
        {
            "id": f"sea_neg{index}",
            "title": f"Local update {index}",
            "selftext": "Policy discussion and civic news",
            "label": "not_askseattle",
            "subreddit": "seattle",
            "created_utc": float(index * 2 + 1),
        }
        for index in range(8)
    ]
    write_jsonl_records(labels_path, records)

    def fake_semantic(*, split, output_dir, model_id, prepared_data_summary=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_name": "semantic_embedding_logreg",
            "model_family": "semantic_embedding",
            "model_id": model_id,
            "artifact_path": str(output_dir / "semantic_embedding_logreg.joblib"),
            "metrics": {
                "high_confidence_precision": 0.9,
                "high_confidence_recall": 0.5,
                "high_confidence_f1": 0.64,
                "support": 2,
                "confidence_band_counts": {"high": 1, "borderline": 2, "low": 1},
            },
            "operating_metrics": {
                "auto_band": {"precision": 0.9, "recall": 0.5, "f1": 0.64, "predicted_positive": 1, "support": 2},
                "review_queue": {"precision": 0.6, "recall": 0.75, "f1": 0.67, "predicted_positive": 3, "support": 2},
                "queue_counts": {"high": 1, "borderline": 2, "low": 1},
                "queue_rates": {"auto_rate": 0.25, "review_rate": 0.75, "borderline_rate": 0.5},
                "positive_prevalence": 0.5,
                "positive_count": 2,
                "total_count": 4,
                "slice_metrics": {},
            },
            "threshold_policy": {"low_threshold": 0.3, "high_threshold": 0.7},
            "production_gate": {"high_precision_target": 0.95, "minimum_high_confidence_test_predictions": 5},
            "production_ready": False,
            "production_ready_blocked_reason": "high_precision_target_not_met_on_test",
        }
        (output_dir / "training_summary.json").write_text("{}", encoding="utf-8")
        return summary

    def fake_transformer(*, split, output_dir, model_id, prepared_data_summary=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_name": "transformer_sequence_classifier",
            "model_family": "transformer_sequence_classifier",
            "model_id": model_id,
            "artifact_path": str(output_dir / "transformer_model"),
            "metrics": {
                "high_confidence_precision": 0.95,
                "high_confidence_recall": 0.55,
                "high_confidence_f1": 0.7,
                "support": 2,
                "confidence_band_counts": {"high": 2, "borderline": 1, "low": 1},
            },
            "operating_metrics": {
                "auto_band": {"precision": 0.95, "recall": 0.55, "f1": 0.7, "predicted_positive": 2, "support": 2},
                "review_queue": {"precision": 0.7, "recall": 0.8, "f1": 0.75, "predicted_positive": 3, "support": 2},
                "queue_counts": {"high": 2, "borderline": 1, "low": 1},
                "queue_rates": {"auto_rate": 0.5, "review_rate": 0.75, "borderline_rate": 0.25},
                "positive_prevalence": 0.5,
                "positive_count": 2,
                "total_count": 4,
                "slice_metrics": {},
            },
            "threshold_policy": {"low_threshold": 0.25, "high_threshold": 0.6},
            "production_gate": {"high_precision_target": 0.95, "minimum_high_confidence_test_predictions": 5},
            "production_ready": True,
            "production_ready_blocked_reason": None,
        }
        (output_dir / "training_summary.json").write_text("{}", encoding="utf-8")
        return summary

    monkeypatch.setattr(training, "_train_semantic_embedding_bundle_for_split", fake_semantic)
    monkeypatch.setattr(training, "_train_transformer_bundle_for_split", fake_transformer)

    summary = benchmark_model_suite_from_labels(
        labels_path,
        tmp_path / "suite",
        evaluation_subreddit="seattle",
    )

    assert summary["evaluation_subreddit"] == "seattle"
    assert summary["production_gate"]["minimum_high_confidence_test_predictions"] == 5
    assert [model["name"] for model in summary["models"]] == [
        "tfidf_recommended",
        "semantic_embedding",
        "transformer_sequence_classifier",
    ]
    assert "metrics_reference" in summary
    assert "slice_metrics" in summary["metrics_reference"]
    assert summary["models"][1]["production_gate"]["minimum_high_confidence_test_predictions"] == 5
    assert Path(tmp_path / "suite" / "benchmark_suite_summary.json").exists()
