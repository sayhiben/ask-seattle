from __future__ import annotations

import json
from pathlib import Path

import pytest

from ask_seattle.model import (
    ConfidenceBandMetrics,
    DecisionThresholds,
    ThresholdSelection,
)
from ask_seattle.data import write_jsonl_records
import ask_seattle.training as training
from ask_seattle.training import (
    OperatingMetrics,
    SuiteModelSpec,
    benchmark_model_suite_from_labels,
    benchmark_model_variants_from_labels,
    retrain_all_from_labels,
    retrain_model_suite_from_labels,
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
    assert summary["operating_metrics"]["slice_metrics"]["post_type"]["support_status"] == "active"
    assert summary["operating_metrics"]["slice_metrics"]["low_text"]["support_status"] == "active"
    assert summary["operating_metrics"]["slice_metrics"]["sparse_media"]["support_status"] in {"active", "observational"}
    assert "constraint_metrics" in summary
    assert "ranking_metrics" in summary
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


def test_classification_metrics_handles_empty_slice() -> None:
    metrics = training._classification_metrics([], [])

    assert metrics == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "predicted_positive": 0,
        "support": 0,
    }


def test_tensor_to_float32_numpy_handles_bfloat16() -> None:
    torch = pytest.importorskip("torch")

    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    array = training._tensor_to_float32_numpy(tensor)

    assert array.dtype.name == "float32"
    assert array.tolist() == [[1.0, 2.0]]


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
                predicted_positive=2,
                support=2,
                production_ready=True,
            ),
            low_threshold_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
            high_threshold_sweep=[],
            low_threshold_sweep=[],
            abstain_enabled=True,
            minimum_high_confidence_calibration_predictions=5,
            high_threshold_fallback_used=False,
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
                predicted_positive=2,
                support=2,
                production_ready=True,
            ),
            low_threshold_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
            high_threshold_sweep=[],
            low_threshold_sweep=[],
            abstain_enabled=True,
            minimum_high_confidence_calibration_predictions=5,
            high_threshold_fallback_used=False,
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


def test_slice_aware_positive_weighting_tracks_support_gated_hard_slices() -> None:
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
    assert weighting.summary["bucket_weights"]["image_post"]["yes"] > 1.0
    assert weighting.summary["bucket_weights"]["low_text"]["yes"] > 1.0
    assert weighting.summary["bucket_weights"]["sparse_media"] == {}
    assert weighting.summary["bucket_weights"]["low_text_image"] == {}
    assert weighting.summary["slice_support_status"]["sparse_media"]["support_status"] == "observational"
    assert weighting.summary["slice_support_status"]["low_text_image"]["support_status"] == "observational"


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
    variant_names = [variant["name"] for variant in summary["variants"]]
    assert variant_names[0:2] == ["legacy_baseline", "recommended"]
    assert any(name.startswith("grid_c1_0") for name in variant_names)
    assert any(name.startswith("grid_c4_0") for name in variant_names)
    assert Path(tmp_path / "variants" / "variant_benchmark_summary.json").exists()


def test_train_model_bundle_from_labels_can_skip_benchmark(tmp_path: Path) -> None:
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

    summary = train_model_bundle_from_labels(labels_path, tmp_path / "model", evaluate_on_test=False)

    assert summary["benchmark_status"] == "not_run"
    assert summary["production_ready"] is False
    assert summary["production_ready_blocked_reason"] == "benchmark_not_run"
    assert "metrics" not in summary
    assert "operating_metrics" not in summary


def test_transformer_candidate_key_prefers_calibration_ready_candidate() -> None:
    ready_candidate = {
        "candidate_metrics": {"pr_auc": 0.91},
        "calibrated_constraint_metrics": {"review_recall_at_precision_75": {"recall": 0.72}},
        "calibrated_thresholds": DecisionThresholds(
            low_threshold=0.3,
            high_threshold=0.8,
            high_threshold_selection=ThresholdSelection(
                threshold=0.8,
                precision=0.97,
                recall=0.45,
                f1=0.61,
                predicted_positive=11,
                support=20,
                production_ready=True,
            ),
            low_threshold_metrics={"precision": 0.8, "recall": 0.8, "f1": 0.8, "predicted_positive": 20, "support": 20},
            high_threshold_sweep=[],
            low_threshold_sweep=[],
            abstain_enabled=True,
            minimum_high_confidence_calibration_predictions=5,
            high_threshold_fallback_used=False,
        ),
        "loss_mode": "balanced_cross_entropy",
    }
    higher_pr_auc_but_blocked = {
        "candidate_metrics": {"pr_auc": 0.99},
        "calibrated_constraint_metrics": {"review_recall_at_precision_75": {"recall": 0.95}},
        "calibrated_thresholds": DecisionThresholds(
            low_threshold=0.2,
            high_threshold=0.7,
            high_threshold_selection=ThresholdSelection(
                threshold=0.7,
                precision=0.93,
                recall=0.80,
                f1=0.86,
                predicted_positive=20,
                support=20,
                production_ready=False,
            ),
            low_threshold_metrics={"precision": 0.8, "recall": 0.9, "f1": 0.85, "predicted_positive": 25, "support": 20},
            high_threshold_sweep=[],
            low_threshold_sweep=[],
            abstain_enabled=True,
            minimum_high_confidence_calibration_predictions=5,
            high_threshold_fallback_used=True,
        ),
        "loss_mode": "plain_cross_entropy",
    }

    assert training._transformer_candidate_key(ready_candidate) > training._transformer_candidate_key(
        higher_pr_auc_but_blocked
    )


def test_retrain_model_suite_writes_training_only_summary(tmp_path: Path, monkeypatch) -> None:
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

    call_counts = {"semantic_minilm_tuned": 0, "transformer_modernbert_base": 0}

    def make_runner(name: str, family: str, model_id: str):
        def runner(*, split, output_dir, prepared_data_summary=None, evaluate_on_test=True, **kwargs):
            assert evaluate_on_test is False
            call_counts[name] += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"{name}.joblib"
            artifact_path.write_text(name, encoding="utf-8")
            return {
                "model_name": name,
                "model_family": family,
                "model_id": model_id,
                "artifact_path": str(artifact_path),
                "calibration": {
                    "available": True,
                    "method": "sigmoid",
                    "positive_count": 2,
                    "negative_count": 2,
                    "calibration_size": 4,
                },
                "threshold_selection": {
                    "high_threshold": 0.7,
                    "high_threshold_selection": {"production_ready": True},
                },
                "threshold_policy": {"low_threshold": 0.3, "high_threshold": 0.7},
                "benchmark_status": "not_run",
                "production_ready": False,
                "production_ready_blocked_reason": "benchmark_not_run",
                "split": training._split_summary(split),
                "prepared_data": prepared_data_summary or {},
            }

        return runner

    def fake_suite_specs(**kwargs):
        return [
            SuiteModelSpec(
                name="semantic_minilm_tuned",
                display_name="Semantic MiniLM",
                family="semantic_embedding",
                runner=make_runner("semantic_minilm_tuned", "semantic_embedding", kwargs["semantic_model_id"]),
                kwargs={
                    "config": training.SemanticModelConfig(
                        name="semantic_minilm_tuned",
                        display_name="Semantic MiniLM",
                        model_id=kwargs["semantic_model_id"],
                        backend="sentence_transformers",
                        prompt_modes=("plain",),
                        normalize_embeddings=(False,),
                        logistic_c_values=(1.0,),
                    )
                },
            ),
            SuiteModelSpec(
                name="transformer_modernbert_base",
                display_name="Transformer ModernBERT-base",
                family="transformer_sequence_classifier",
                runner=make_runner(
                    "transformer_modernbert_base",
                    "transformer_sequence_classifier",
                    kwargs["transformer_model_id"],
                ),
                kwargs={"model_id": kwargs["transformer_model_id"], "display_name": "Transformer ModernBERT-base"},
            ),
        ]

    monkeypatch.setattr(training, "_suite_model_specs", fake_suite_specs)

    summary = retrain_model_suite_from_labels(labels_path, tmp_path / "suite", evaluation_subreddit="seattle")

    assert call_counts == {"semantic_minilm_tuned": 1, "transformer_modernbert_base": 1}
    assert (tmp_path / "suite" / "suite_input.json").exists()
    assert (tmp_path / "suite" / "suite_training_summary.json").exists()
    assert [model["status"] for model in summary["models"]] == ["trained", "trained"]
    assert [model["benchmark_status"] for model in summary["models"]] == ["not_run", "not_run"]


def test_suite_model_specs_include_tfidf_and_transformer_candidates_only() -> None:
    specs = training._suite_model_specs(
        semantic_model_id="sentence-transformers/all-MiniLM-L6-v2",
        semantic_secondary_model_id="Qwen/Qwen3-Embedding-0.6B",
        semantic_tertiary_model_id="jinaai/jina-embeddings-v5-text-small-classification",
        transformer_model_id="answerdotai/ModernBERT-base",
        transformer_secondary_model_id="chandar-lab/NeoBERT",
        transformer_tertiary_model_id="answerdotai/ModernBERT-large",
        causal_lm_model_id="Qwen/Qwen3-1.7B",
    )

    names = [spec.name for spec in specs]

    assert names == [
        "tfidf_recommended",
        "transformer_modernbert_base",
        "transformer_neobert",
        "transformer_modernbert_large",
    ]
    assert "semantic_minilm_tuned" not in names
    assert "semantic_jina_embeddings_v5_text_small_classification" not in names
    assert "causal_lm_qwen3_1_7b_lora" not in names


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

    call_counts = {"semantic_minilm_tuned": 0, "transformer_modernbert_base": 0}

    def make_runner(name: str, family: str, model_id: str):
        def runner(*, split, output_dir, prepared_data_summary=None, evaluate_on_test=True, **kwargs):
            assert evaluate_on_test is False
            call_counts[name] += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"{name}.joblib"
            artifact_path.write_text(name, encoding="utf-8")
            return {
                "model_name": name,
                "model_family": family,
                "model_id": model_id,
                "artifact_path": str(artifact_path),
                "calibration": {
                    "available": True,
                    "method": "sigmoid",
                    "positive_count": 2,
                    "negative_count": 2,
                    "calibration_size": 4,
                },
                "threshold_selection": {
                    "high_threshold": 0.7,
                    "high_threshold_selection": {"production_ready": True},
                },
                "threshold_policy": {"low_threshold": 0.3, "high_threshold": 0.7},
                "benchmark_status": "not_run",
                "production_ready": False,
                "production_ready_blocked_reason": "benchmark_not_run",
                "split": training._split_summary(split),
                "prepared_data": prepared_data_summary or {},
            }

        return runner

    def fake_suite_specs(**kwargs):
        return [
            SuiteModelSpec(
                name="semantic_minilm_tuned",
                display_name="Semantic MiniLM",
                family="semantic_embedding",
                runner=make_runner("semantic_minilm_tuned", "semantic_embedding", kwargs["semantic_model_id"]),
                kwargs={
                    "config": training.SemanticModelConfig(
                        name="semantic_minilm_tuned",
                        display_name="Semantic MiniLM",
                        model_id=kwargs["semantic_model_id"],
                        backend="sentence_transformers",
                        prompt_modes=("plain",),
                        normalize_embeddings=(False,),
                        logistic_c_values=(1.0,),
                    )
                },
            ),
            SuiteModelSpec(
                name="transformer_modernbert_base",
                display_name="Transformer ModernBERT-base",
                family="transformer_sequence_classifier",
                runner=make_runner(
                    "transformer_modernbert_base",
                    "transformer_sequence_classifier",
                    kwargs["transformer_model_id"],
                ),
                kwargs={"model_id": kwargs["transformer_model_id"], "display_name": "Transformer ModernBERT-base"},
            ),
        ]

    monkeypatch.setattr(training, "_suite_model_specs", fake_suite_specs)
    monkeypatch.setattr(training, "load_model", lambda artifact_path: {"artifact_path": str(artifact_path)})

    def fake_score_rows(bundle, rows):
        artifact_path = str(bundle["artifact_path"])
        base = [0.8, 0.7, 0.2, 0.1] if "semantic_minilm_tuned" in artifact_path else [0.9, 0.6, 0.4, 0.1]
        repeats = (len(rows) + len(base) - 1) // len(base)
        scores = (base * repeats)[: len(rows)]
        if "semantic_minilm_tuned" in artifact_path:
            return scores
        return scores

    monkeypatch.setattr(training, "score_rows", fake_score_rows)

    retrain_model_suite_from_labels(labels_path, tmp_path / "suite", evaluation_subreddit="seattle")
    assert call_counts == {"semantic_minilm_tuned": 1, "transformer_modernbert_base": 1}

    summary = benchmark_model_suite_from_labels(
        labels_path,
        tmp_path / "suite",
        evaluation_subreddit="seattle",
        notes="after adding april labels",
    )

    assert summary["evaluation_subreddit"] == "seattle"
    assert summary["production_gate"]["minimum_high_confidence_test_predictions"] == 5
    assert [model["status"] for model in summary["models"]] == ["ok", "ok"]
    assert [model["result_source"] for model in summary["models"]] == ["benchmarked", "benchmarked"]
    assert "metrics_reference" in summary
    assert summary["benchmark_run"]["notes"] == "after adding april labels"
    assert "representation" in summary["benchmark_run"]
    assert Path(tmp_path / "suite" / "benchmark_suite_summary.json").exists()
    history_index = json.loads((tmp_path / "suite" / "benchmark_history.json").read_text())
    assert history_index["runs"][-1]["notes"] == "after adding april labels"
    archived_summary_path = Path(history_index["runs"][-1]["summary_path"])
    assert archived_summary_path.exists()
    benchmarked_summary = json.loads((tmp_path / "suite" / "semantic_minilm_tuned" / "training_summary.json").read_text())
    assert benchmarked_summary["benchmark_status"] == "complete"
    assert "metrics" in benchmarked_summary
    assert "runtime_environment" in benchmarked_summary
    assert "input_data" in summary


def test_benchmark_model_suite_skips_untrained_models(tmp_path: Path, monkeypatch) -> None:
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

    def fake_suite_specs(**kwargs):
        return [
            SuiteModelSpec(
                name="semantic_minilm_tuned",
                display_name="Semantic MiniLM",
                family="semantic_embedding",
                runner=lambda **kwargs: {},
                kwargs={
                    "config": training.SemanticModelConfig(
                        name="semantic_minilm_tuned",
                        display_name="Semantic MiniLM",
                        model_id=kwargs["semantic_model_id"],
                        backend="sentence_transformers",
                        prompt_modes=("plain",),
                        normalize_embeddings=(False,),
                        logistic_c_values=(1.0,),
                    )
                },
            ),
            SuiteModelSpec(
                name="transformer_modernbert_base",
                display_name="Transformer ModernBERT-base",
                family="transformer_sequence_classifier",
                runner=lambda **kwargs: {},
                kwargs={"model_id": kwargs["transformer_model_id"], "display_name": "Transformer ModernBERT-base"},
            ),
        ]

    monkeypatch.setattr(training, "_suite_model_specs", fake_suite_specs)

    summary = benchmark_model_suite_from_labels(labels_path, tmp_path / "suite", evaluation_subreddit="seattle")

    assert [model["status"] for model in summary["models"]] == ["skipped", "skipped"]
    assert all(model["reason"] == "not_trained" for model in summary["models"])


def test_benchmark_seed_sweep_writes_aggregate_summary(tmp_path: Path, monkeypatch) -> None:
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
        for index in range(10)
    ] + [
        {
            "id": f"sea_neg{index}",
            "title": f"Local update {index}",
            "selftext": "Policy discussion and civic news",
            "label": "not_askseattle",
            "subreddit": "seattle",
            "created_utc": float(index * 2 + 1),
        }
        for index in range(10)
    ]
    write_jsonl_records(labels_path, records)

    def make_runner(name: str, family: str, model_id: str):
        def runner(*, split, output_dir, prepared_data_summary=None, evaluate_on_test=True, **kwargs):
            assert evaluate_on_test is True
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"{name}.joblib"
            artifact_path.write_text(name, encoding="utf-8")
            seed = int(split.split_seed or 0)
            pr_auc = 0.8 + (seed / 1000.0)
            auto_recall = 0.3 + (seed / 1000.0)
            review_recall = 0.7 + (seed / 1000.0)
            return {
                "model_name": name,
                "model_family": family,
                "display_name": name,
                "model_id": model_id,
                "artifact_path": str(artifact_path),
                "calibration": {"available": True, "method": "sigmoid", "positive_count": 3, "negative_count": 3, "calibration_size": 6},
                "threshold_selection": {"high_threshold": 0.7, "high_threshold_selection": {"production_ready": True}},
                "threshold_policy": {"low_threshold": 0.3, "high_threshold": 0.7},
                "metrics": {"high_confidence_precision": 1.0, "high_confidence_recall": auto_recall, "high_confidence_f1": auto_recall, "support": 5, "confidence_band_counts": {"high": 2, "borderline": 1, "low": 1}},
                "operating_metrics": {
                    "auto_band": {"precision": 1.0, "recall": auto_recall, "f1": auto_recall, "predicted_positive": 2, "support": 5},
                    "review_queue": {"precision": 0.8, "recall": review_recall, "f1": review_recall, "predicted_positive": 4, "support": 5},
                    "queue_counts": {"high": 2, "borderline": 2, "low": 1},
                    "queue_rates": {"auto_rate": 0.2, "review_rate": 0.4, "borderline_rate": 0.2},
                    "positive_prevalence": 0.5,
                    "positive_count": 5,
                    "total_count": 10,
                    "slice_metrics": {},
                },
                "constraint_metrics": {
                    "auto_recall_at_precision_95": {"recall": auto_recall, "precision_target": 0.95, "threshold": 0.7, "precision": 1.0, "f1": auto_recall, "predicted_positive": 2, "support": 5, "target_met": True},
                    "review_recall_at_precision_75": {"recall": review_recall, "precision_target": 0.75, "threshold": 0.3, "precision": 0.8, "f1": review_recall, "predicted_positive": 4, "support": 5, "target_met": True},
                },
                "ranking_metrics": {"pr_auc": pr_auc},
                "benchmark_status": "complete",
                "production_ready": True,
                "production_ready_blocked_reason": None,
                "split": training._split_summary(split),
                "prepared_data": prepared_data_summary or {},
            }
        return runner

    def fake_suite_specs(**kwargs):
        return [
            SuiteModelSpec(
                name="transformer_modernbert_base",
                display_name="Transformer ModernBERT-base",
                family="transformer_sequence_classifier",
                runner=make_runner("transformer_modernbert_base", "transformer_sequence_classifier", kwargs["transformer_secondary_model_id"]),
                kwargs={"model_id": kwargs["transformer_secondary_model_id"], "display_name": "Transformer ModernBERT-base"},
            ),
            SuiteModelSpec(
                name="transformer_neobert",
                display_name="Transformer NeoBERT",
                family="transformer_sequence_classifier",
                runner=make_runner("transformer_neobert", "transformer_sequence_classifier", kwargs["transformer_tertiary_model_id"]),
                kwargs={"model_id": kwargs["transformer_tertiary_model_id"], "display_name": "Transformer NeoBERT"},
            ),
        ]

    monkeypatch.setattr(training, "_suite_model_specs", fake_suite_specs)

    summary = training.benchmark_seed_sweep_from_labels(
        labels_path,
        tmp_path / "suite",
        evaluation_subreddit="seattle",
        split_seeds=(13, 21),
        model_names=("transformer_modernbert_base", "transformer_neobert"),
    )

    assert summary["split_seeds"] == [13, 21]
    assert summary["selected_models"] == ["transformer_modernbert_base", "transformer_neobert"]
    assert len(summary["seed_runs"]) == 2
    assert len(summary["model_aggregates"]) == 2
    assert summary["best_model_selection_order"] == ["ready_rate", "min_auto_precision", "mean_auto_recall", "mean_pr_auc"]
    assert Path(tmp_path / "suite" / "seed_sweeps" / "seed_sweep_summary.json").exists()
    aggregate = next(item for item in summary["model_aggregates"] if item["name"] == "transformer_modernbert_base")
    assert aggregate["metric_summary"]["pr_auc"]["count"] == 2
    assert aggregate["metric_summary"]["review_recall"]["mean"] > 0.7
    assert aggregate["production_ready_runs"] == 2
    assert aggregate["ready_rate"] == 1.0
    assert aggregate["min_auto_precision"] == 1.0
    assert aggregate["mean_pr_auc"] > 0.8
    assert aggregate["std_pr_auc"] > 0.0


def test_benchmark_seed_sweep_aggregates_prioritize_ready_rate_then_stability() -> None:
    runs = [
        {
            "seed": 13,
            "models": [
                {
                    "name": "transformer_modernbert_base",
                    "display_name": "Transformer ModernBERT-base",
                    "model_family": "transformer_sequence_classifier",
                    "model_id": "answerdotai/ModernBERT-base",
                    "production_ready": True,
                    "ranking_metrics": {"pr_auc": 0.90},
                    "operating_metrics": {
                        "auto_band": {"precision": 0.97, "recall": 0.40},
                        "review_queue": {"precision": 0.82, "recall": 0.80},
                    },
                    "constraint_metrics": {
                        "auto_recall_at_precision_95": {"recall": 0.45},
                        "review_recall_at_precision_75": {"recall": 0.80},
                    },
                },
                {
                    "name": "transformer_modernbert_large",
                    "display_name": "Transformer ModernBERT-large",
                    "model_family": "transformer_sequence_classifier",
                    "model_id": "answerdotai/ModernBERT-large",
                    "production_ready": True,
                    "ranking_metrics": {"pr_auc": 0.95},
                    "operating_metrics": {
                        "auto_band": {"precision": 0.96, "recall": 0.60},
                        "review_queue": {"precision": 0.81, "recall": 0.88},
                    },
                    "constraint_metrics": {
                        "auto_recall_at_precision_95": {"recall": 0.63},
                        "review_recall_at_precision_75": {"recall": 0.88},
                    },
                },
            ],
        },
        {
            "seed": 21,
            "models": [
                {
                    "name": "transformer_modernbert_base",
                    "display_name": "Transformer ModernBERT-base",
                    "model_family": "transformer_sequence_classifier",
                    "model_id": "answerdotai/ModernBERT-base",
                    "production_ready": True,
                    "ranking_metrics": {"pr_auc": 0.89},
                    "operating_metrics": {
                        "auto_band": {"precision": 0.96, "recall": 0.38},
                        "review_queue": {"precision": 0.83, "recall": 0.79},
                    },
                    "constraint_metrics": {
                        "auto_recall_at_precision_95": {"recall": 0.43},
                        "review_recall_at_precision_75": {"recall": 0.79},
                    },
                },
                {
                    "name": "transformer_modernbert_large",
                    "display_name": "Transformer ModernBERT-large",
                    "model_family": "transformer_sequence_classifier",
                    "model_id": "answerdotai/ModernBERT-large",
                    "production_ready": False,
                    "ranking_metrics": {"pr_auc": 0.97},
                    "operating_metrics": {
                        "auto_band": {"precision": 0.89, "recall": 0.70},
                        "review_queue": {"precision": 0.80, "recall": 0.90},
                    },
                    "constraint_metrics": {
                        "auto_recall_at_precision_95": {"recall": 0.59},
                        "review_recall_at_precision_75": {"recall": 0.90},
                    },
                },
            ],
        },
    ]

    aggregates = training._benchmark_seed_sweep_aggregates(
        runs,
        model_names=("transformer_modernbert_base", "transformer_modernbert_large"),
    )

    assert [aggregate["name"] for aggregate in aggregates] == [
        "transformer_modernbert_base",
        "transformer_modernbert_large",
    ]
    assert aggregates[0]["production_ready_runs"] == 2
    assert aggregates[0]["ready_rate"] == 1.0
    assert aggregates[0]["min_auto_precision"] == 0.96
    assert aggregates[0]["mean_pr_auc"] == pytest.approx(0.895)
    assert aggregates[0]["std_pr_auc"] == pytest.approx(0.005)


def test_retrain_all_trains_operational_and_suite_without_benchmark(tmp_path: Path, monkeypatch) -> None:
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

    monkeypatch.setattr(
        training,
        "retrain_model_suite_from_labels",
        lambda *args, **kwargs: {"models": [{"name": "semantic_minilm_tuned", "benchmark_status": "not_run"}]},
    )

    summary = retrain_all_from_labels(
        labels_path,
        operational_output_dir=tmp_path / "operational",
        benchmark_output_dir=tmp_path / "suite",
        evaluation_subreddit="seattle",
    )

    assert summary["operational_model"]["benchmark_status"] == "not_run"
    assert summary["suite"]["models"][0]["benchmark_status"] == "not_run"


def test_benchmark_model_suite_benchmarks_only_available_artifacts(tmp_path: Path, monkeypatch) -> None:
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

    call_counts = {"semantic_minilm_tuned": 0, "transformer_modernbert_base": 0}

    def make_runner(name: str, family: str, model_id: str):
        def runner(*, split, output_dir, prepared_data_summary=None, evaluate_on_test=True, **kwargs):
            assert evaluate_on_test is False
            call_counts[name] += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"{name}.joblib"
            artifact_path.write_text(name, encoding="utf-8")
            return {
                "model_name": name,
                "model_family": family,
                "model_id": model_id,
                "artifact_path": str(artifact_path),
                "calibration": {
                    "available": True,
                    "method": "sigmoid",
                    "positive_count": 2,
                    "negative_count": 2,
                    "calibration_size": 4,
                },
                "threshold_selection": {
                    "high_threshold": 0.7,
                    "high_threshold_selection": {"production_ready": True},
                },
                "threshold_policy": {"low_threshold": 0.3, "high_threshold": 0.7},
                "benchmark_status": "not_run",
                "production_ready": False,
                "production_ready_blocked_reason": "benchmark_not_run",
                "split": training._split_summary(split),
                "prepared_data": prepared_data_summary or {},
            }

        return runner

    def fake_suite_specs(**kwargs):
        return [
            SuiteModelSpec(
                name="semantic_minilm_tuned",
                display_name="Semantic MiniLM",
                family="semantic_embedding",
                runner=make_runner("semantic_minilm_tuned", "semantic_embedding", kwargs["semantic_model_id"]),
                kwargs={
                    "config": training.SemanticModelConfig(
                        name="semantic_minilm_tuned",
                        display_name="Semantic MiniLM",
                        model_id=kwargs["semantic_model_id"],
                        backend="sentence_transformers",
                        prompt_modes=("plain",),
                        normalize_embeddings=(False,),
                        logistic_c_values=(1.0,),
                    )
                },
            ),
            SuiteModelSpec(
                name="transformer_modernbert_base",
                display_name="Transformer ModernBERT-base",
                family="transformer_sequence_classifier",
                runner=make_runner(
                    "transformer_modernbert_base",
                    "transformer_sequence_classifier",
                    kwargs["transformer_model_id"],
                ),
                kwargs={"model_id": kwargs["transformer_model_id"], "display_name": "Transformer ModernBERT-base"},
            ),
        ]

    monkeypatch.setattr(training, "_suite_model_specs", fake_suite_specs)
    monkeypatch.setattr(training, "load_model", lambda artifact_path: {"artifact_path": str(artifact_path)})
    monkeypatch.setattr(
        training,
        "score_rows",
        lambda bundle, rows: ([0.8, 0.7, 0.2, 0.1] * ((len(rows) + 3) // 4))[: len(rows)],
    )

    retrain_model_suite_from_labels(labels_path, tmp_path / "suite", evaluation_subreddit="seattle")
    assert call_counts == {"semantic_minilm_tuned": 1, "transformer_modernbert_base": 1}

    (tmp_path / "suite" / "transformer_modernbert_base" / "transformer_modernbert_base.joblib").unlink()

    second = benchmark_model_suite_from_labels(labels_path, tmp_path / "suite", evaluation_subreddit="seattle")

    assert call_counts == {"semantic_minilm_tuned": 1, "transformer_modernbert_base": 1}
    assert second["models"][0]["status"] == "ok"
    assert second["models"][0]["result_source"] == "benchmarked"
    assert second["models"][1]["status"] == "skipped"
    assert second["models"][1]["reason"] == "not_trained"


def test_resolve_suite_artifact_path_prefers_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    repo_like_root = tmp_path / "repo"
    artifact = repo_like_root / "models" / "benchmark-suite" / "tfidf_recommended" / "tfidf_logreg.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("ok", encoding="utf-8")
    monkeypatch.chdir(repo_like_root)

    resolved = training._resolve_suite_artifact_path(
        {"artifact_path": "models/benchmark-suite/tfidf_recommended/tfidf_logreg.joblib"},
        repo_like_root / "models" / "benchmark-suite" / "tfidf_recommended",
    )

    assert resolved == artifact.resolve()


def test_causal_lm_training_dataset_removes_raw_text_columns() -> None:
    class FakeTokenizer:
        eos_token_id = 99

        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
            max_length: int,
        ) -> dict[str, list[int]]:
            token_ids = [min(ord(ch), 255) for ch in text][:max_length]
            if add_special_tokens:
                token_ids = [1] + token_ids
            return {"input_ids": token_ids}

    dataset = training._causal_lm_training_dataset(
        [
            {
                "prompt": "Prompt text",
                "target": " askseattle",
                "label": 1,
                "example_weight": 1.5,
            }
        ],
        tokenizer=FakeTokenizer(),
        prompt_max_length=32,
        target_max_length=8,
        sequence_max_length=40,
    )

    assert set(dataset.column_names) == {"input_ids", "attention_mask", "labels", "example_weight"}


def test_resolve_causal_lm_runtime_profile_prefers_cpu_on_mps_by_default() -> None:
    assert (
        training._resolve_causal_lm_runtime_profile(
            detected_runtime="mps",
            requested_runtime_profile=None,
            model_id="Qwen/Qwen3-1.7B",
        )
        == "cpu_fallback"
    )
    assert (
        training._resolve_causal_lm_runtime_profile(
            detected_runtime="mps",
            requested_runtime_profile="mps",
            model_id="Qwen/Qwen3-1.7B",
        )
        == "mps"
    )


def test_resolve_semantic_encoder_device_prefers_cpu_for_hf_embedding_on_mps() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMPSBackend:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeBackends:
        mps = FakeMPSBackend()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

    config = training.SemanticModelConfig(
        name="semantic_qwen3_embedding_0_6b",
        display_name="Semantic Qwen3-Embedding",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        backend="hf_embedding",
        config_version="v2_title_body_metadata",
        prompt_modes=("plain",),
        normalize_embeddings=(False,),
        logistic_c_values=(1.0,),
    )

    assert training._resolve_semantic_encoder_device(config, FakeTorch()) == "cpu"


def test_causal_lm_prompt_v3_uses_compact_contextual_fields() -> None:
    row = {
        "title": "Who is this on the mural?",
        "body_raw": "Found this in Ballard and trying to identify the artist.",
        "post_type": "image",
        "content_domain": "i.redd.it",
        "has_question_mark": "yes",
        "is_low_text": "no",
        "is_crosspost": "no",
    }

    prompt = training.causal_lm_prompt_for_row(
        row,
        prompt_template_version=training.DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
    )

    assert "Return exactly one label: askseattle or not_askseattle." in prompt
    assert "Do not use subreddit name." in prompt
    assert "Title: Who is this on the mural?" in prompt
    assert "Post type: image" in prompt
    assert "Has question mark: yes" in prompt
    assert "Crosspost: no" in prompt
    assert "Sparse media:" not in prompt
    assert prompt.endswith("Label:")


def test_causal_lm_prompt_v4_adds_image_low_text_guidance() -> None:
    row = {
        "title": "Anyone know what this plant is?",
        "body_raw": "",
        "post_type": "image",
        "content_domain": "i.redd.it",
        "has_body": "no",
        "has_question_mark": "yes",
        "is_low_text": "yes",
        "is_sparse_media": True,
        "is_crosspost": "no",
    }

    prompt = training.causal_lm_prompt_for_row(
        row,
        prompt_template_version="v4_image_low_text",
    )

    assert "Title-only image posts can still be askseattle" in prompt
    assert "Has body: no" in prompt
    assert "Sparse media: yes" in prompt
    assert prompt.endswith("Label:")


def test_suite_summary_matches_spec_checks_causal_lm_prompt_template_version() -> None:
    spec = SuiteModelSpec(
        name="causal_lm_qwen3_1_7b_lora",
        display_name="Decoder Qwen3-1.7B LoRA",
        family="causal_lm_classifier",
        runner=lambda **kwargs: {},
        kwargs={
            "model_id": "Qwen/Qwen3-1.7B",
            "display_name": "Decoder Qwen3-1.7B LoRA",
            "prompt_template_version": training.DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
            "config_version": "v2_compact_prompt_two_epoch",
        },
    )

    matching_summary = {
        "model_family": "causal_lm_classifier",
        "model_id": "Qwen/Qwen3-1.7B",
        "prompt_template_version": training.DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
        "config_version": "v2_compact_prompt_two_epoch",
    }
    stale_summary = {
        "model_family": "causal_lm_classifier",
        "model_id": "Qwen/Qwen3-1.7B",
        "prompt_template_version": "v1_binary_label_completion",
        "config_version": "v2_compact_prompt_two_epoch",
    }

    assert training._suite_summary_matches_spec(matching_summary, spec) is True
    assert training._suite_summary_matches_spec(stale_summary, spec) is False


def test_suite_summary_matches_spec_checks_semantic_config_version() -> None:
    spec = SuiteModelSpec(
        name="semantic_minilm_tuned",
        display_name="Semantic MiniLM",
        family="semantic_embedding",
        runner=lambda **kwargs: {},
        kwargs={
            "config": training.SemanticModelConfig(
                name="semantic_minilm_tuned",
                display_name="Semantic MiniLM",
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                backend="sentence_transformers",
                config_version="v2_title_body_metadata",
                prompt_modes=("plain", "task_prefix"),
                normalize_embeddings=(False, True),
                logistic_c_values=(0.25, 1.0),
            )
        },
    )

    matching_summary = {
        "model_family": "semantic_embedding",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "config_version": "v2_title_body_metadata",
    }
    stale_summary = {
        "model_family": "semantic_embedding",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "config_version": "v1_flat_text",
    }

    assert training._suite_summary_matches_spec(matching_summary, spec) is True
    assert training._suite_summary_matches_spec(stale_summary, spec) is False


def test_semantic_component_texts_split_title_body_and_prefixes() -> None:
    posts = [
        training.LabeledPost(title="Who is this?", selftext="Seen near Fremont.", label=1),
    ]
    config = training.SemanticModelConfig(
        name="semantic_qwen3_embedding_0_6b",
        display_name="Semantic Qwen3-Embedding",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        backend="hf_embedding",
        config_version="v2_title_body_metadata",
        prompt_modes=("plain", "short_task_prefix"),
        normalize_embeddings=(False, True),
        logistic_c_values=(1.0,),
        prompt_prefix="Long prefix",
        short_prompt_prefix="Short prefix",
    )

    title_texts, body_texts = training._semantic_component_texts(
        posts,
        prompt_mode="short_task_prefix",
        config=config,
    )

    assert title_texts == ["Short prefix Title: Who is this?"]
    assert body_texts == ["Short prefix Body: Seen near Fremont."]


def test_semantic_component_texts_support_jina_document_component() -> None:
    posts = [
        training.LabeledPost(title="Who is this?", selftext="Seen near Fremont.", label=1),
    ]
    config = training.SemanticModelConfig(
        name="semantic_jina_embeddings_v5_text_small_classification",
        display_name="Semantic Jina v5 Text Small Classification",
        model_id="jinaai/jina-embeddings-v5-text-small-classification",
        backend="hf_embedding",
        config_version="v4_title_body_metadata_jina_document_component",
        prompt_modes=("plain", "jina_document_component"),
        normalize_embeddings=(False, True),
        logistic_c_values=(1.0,),
        prompt_prefix="Document:",
    )

    title_texts, body_texts = training._semantic_component_texts(
        posts,
        prompt_mode="jina_document_component",
        config=config,
    )

    assert title_texts == ["Document: Title: Who is this?"]
    assert body_texts == ["Document: Body: Seen near Fremont."]


def test_transformer_candidate_profiles_keep_long_context_profiles_bounded_to_supported_runs() -> None:
    modernbert_base_default = training._transformer_candidate_profiles("answerdotai/ModernBERT-base")
    modernbert_base_cuda = training._transformer_candidate_profiles("answerdotai/ModernBERT-base", allow_long_context=True)
    neobert_default = training._transformer_candidate_profiles("chandar-lab/NeoBERT")
    neobert_cuda = training._transformer_candidate_profiles("chandar-lab/NeoBERT", allow_long_context=True)
    modernbert_large_cuda = training._transformer_candidate_profiles(
        "answerdotai/ModernBERT-large",
        allow_long_context=True,
    )

    assert [profile["name"] for profile in modernbert_base_default] == [
        "baseline",
        "precision_tuned",
        "balanced_tuned",
    ]
    assert [profile["name"] for profile in modernbert_base_cuda] == [
        "baseline",
        "precision_tuned",
        "balanced_tuned",
        "long_context",
    ]
    assert [profile["name"] for profile in neobert_default] == ["baseline", "precision_tuned"]
    assert [profile["name"] for profile in neobert_cuda] == [
        "baseline",
        "precision_tuned",
        "long_context",
        "precision_long_context",
    ]
    assert [profile["name"] for profile in modernbert_large_cuda] == [
        "baseline",
        "precision_tuned",
        "balanced_tuned",
        "long_context",
        "precision_long_context",
    ]


def test_semantic_component_texts_fill_empty_values_with_placeholders() -> None:
    posts = [
        training.LabeledPost(title="", selftext="", label=1),
    ]
    config = training.SemanticModelConfig(
        name="semantic_qwen3_embedding_0_6b",
        display_name="Semantic Qwen3-Embedding",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        backend="hf_embedding",
        config_version="v2_title_body_metadata",
        prompt_modes=("plain",),
        normalize_embeddings=(False, True),
        logistic_c_values=(1.0,),
    )

    title_texts, body_texts = training._semantic_component_texts(
        posts,
        prompt_mode="plain",
        config=config,
    )

    assert title_texts == ["[no title]"]
    assert body_texts == ["[no body]"]


def test_move_token_batch_to_device_preserves_integer_tensor_types() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object | None]] = []

        def to(self, *, device: object, dtype: object | None = None) -> "FakeTensor":
            self.calls.append((device, dtype))
            return self

    class FakeTorch:
        long = "long-dtype"

    input_ids = FakeTensor()
    attention_mask = FakeTensor()
    other = FakeTensor()

    moved = training._move_token_batch_to_device(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special": other,
        },
        device="mps",
        torch_module=FakeTorch(),
    )

    assert moved["input_ids"] is input_ids
    assert moved["attention_mask"] is attention_mask
    assert moved["special"] is other
    assert input_ids.calls == [("mps", "long-dtype")]
    assert attention_mask.calls == [("mps", "long-dtype")]
    assert other.calls == [("mps", None)]


def test_suite_summary_matches_spec_checks_tfidf_config_version() -> None:
    spec = SuiteModelSpec(
        name="tfidf_recommended",
        display_name="TF-IDF",
        family="tfidf",
        runner=lambda **kwargs: {},
        kwargs={
            "variant": training.VariantConfig(
                name="recommended",
                extra_word_stopwords=training.DEFAULT_EXTRA_WORD_STOPWORDS,
                char_weight=training.DEFAULT_CHAR_WEIGHT,
                metadata_weight=training.DEFAULT_METADATA_WEIGHT,
                tfidf_config_version=training.DEFAULT_TFIDF_CONFIG_VERSION,
                normalize_urls=training.DEFAULT_TFIDF_URL_NORMALIZATION,
                strip_urls=training.DEFAULT_TFIDF_STRIP_URLS,
            )
        },
    )

    matching_summary = {
        "model_family": "tfidf",
        "variant": {
            "name": "recommended",
            "tfidf_config_version": training.DEFAULT_TFIDF_CONFIG_VERSION,
            "normalize_urls": training.DEFAULT_TFIDF_URL_NORMALIZATION,
            "strip_urls": training.DEFAULT_TFIDF_STRIP_URLS,
        },
    }
    stale_summary = {
        "model_family": "tfidf",
        "variant": {
            "name": "recommended",
            "tfidf_config_version": "v1_mixed_metadata",
            "normalize_urls": training.DEFAULT_TFIDF_URL_NORMALIZATION,
            "strip_urls": training.DEFAULT_TFIDF_STRIP_URLS,
        },
    }

    assert training._suite_summary_matches_spec(matching_summary, spec) is True
    assert training._suite_summary_matches_spec(stale_summary, spec) is False
