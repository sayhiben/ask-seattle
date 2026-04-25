from __future__ import annotations

import json
from pathlib import Path

import pytest

from ask_seattle.data import LabeledPost
from ask_seattle.hybrid_policy import (
    build_benchmark_weighted_hybrid_policy,
    hybrid_decider_response,
)
from ask_seattle.model import CheckResult, DatasetSplit
import ask_seattle.training as training


def test_build_benchmark_weighted_hybrid_policy_prefers_matching_history_runs() -> None:
    policy = build_benchmark_weighted_hybrid_policy(
        active_models=[
            {"name": "tfidf_recommended", "display_name": "TF-IDF"},
            {"name": "transformer_neobert", "display_name": "NeoBERT"},
            {"name": "transformer_modernbert_large", "display_name": "ModernBERT-large"},
        ],
        primary_model_name="tfidf_recommended",
        split_strategy="random_eval_subreddit",
        evaluation_subreddit="seattle",
        benchmark_history={
            "runs": [
                {
                    "split": {
                        "split_strategy": "time",
                        "evaluation_subreddit": "seattle",
                    },
                    "models": [
                        {"name": "tfidf_recommended", "status": "ok"},
                        {"name": "transformer_neobert", "status": "ok"},
                        {"name": "transformer_modernbert_large", "status": "ok"},
                    ],
                },
                {
                    "split": {
                        "split_strategy": "random_eval_subreddit",
                        "evaluation_subreddit": "seattle",
                    },
                    "models": [
                        {
                            "name": "tfidf_recommended",
                            "display_name": "TF-IDF",
                            "status": "ok",
                            "production_ready": False,
                            "auto_recall_at_precision_95": 0.15,
                            "review_recall_at_precision_75": 0.70,
                            "pr_auc": 0.80,
                        },
                        {
                            "name": "transformer_neobert",
                            "display_name": "NeoBERT",
                            "status": "ok",
                            "production_ready": True,
                            "auto_recall_at_precision_95": 0.63,
                            "review_recall_at_precision_75": 0.88,
                            "pr_auc": 0.92,
                        },
                        {
                            "name": "transformer_modernbert_large",
                            "display_name": "ModernBERT-large",
                            "status": "ok",
                            "production_ready": True,
                            "auto_recall_at_precision_95": 0.60,
                            "review_recall_at_precision_75": 0.87,
                            "pr_auc": 0.90,
                        },
                    ],
                },
            ]
        },
    )

    by_name = {item["name"]: item for item in policy["weights"]}

    assert policy["source"] == "benchmark_history"
    assert policy["matched_run_count"] == 1
    assert pytest.approx(sum(item["weight"] for item in policy["weights"])) == 1.0
    assert by_name["transformer_neobert"]["weight"] > by_name["transformer_modernbert_large"]["weight"]
    assert by_name["transformer_modernbert_large"]["weight"] > by_name["tfidf_recommended"]["weight"]


def test_hybrid_decider_response_uses_benchmark_weighted_policy() -> None:
    decider_result, decision_context = hybrid_decider_response(
        policy="hybrid_consensus",
        primary_result={
            "model_name": "tfidf_logreg",
            "display_name": "TF-IDF",
            "score": 0.70,
            "score_raw": 0.70,
            "score_calibrated": 0.70,
            "label": "not_askseattle",
            "confidence_band": "low",
            "low_threshold": 0.75,
            "high_threshold": 0.90,
            "time_source": None,
            "created_at": "2026-04-21T00:00:00Z",
        },
        primary_model_name="tfidf_recommended",
        row={"post_type": "image", "is_low_text": "yes", "is_sparse_media": True},
        comparisons=[
            {
                "name": "transformer_neobert",
                "display_name": "NeoBERT",
                "result": {"score": 0.95, "label": "askseattle", "confidence_band": "high"},
            },
            {
                "name": "transformer_modernbert_large",
                "display_name": "ModernBERT-large",
                "result": {"score": 0.85, "label": "askseattle", "confidence_band": "borderline"},
            },
        ],
        route_reasons=["image_post", "low_text"],
        hybrid_policy={
            "source": "benchmark_history",
            "weights": [
                {"name": "tfidf_recommended", "display_name": "TF-IDF", "weight": 0.10},
                {
                    "name": "transformer_neobert",
                    "display_name": "NeoBERT",
                    "weight": 0.45,
                },
                {"name": "transformer_modernbert_large", "display_name": "ModernBERT-large", "weight": 0.45},
            ],
        },
        min_comparison_results=2,
    )

    applied = {
        item["name"]: item["weight"] for item in decision_context["hybrid_policy"]["applied_weights"]
    }

    assert decider_result is not None
    assert decider_result["label"] == "askseattle"
    assert decider_result["confidence_band"] == "borderline"
    assert decider_result["score"] == pytest.approx(0.88)
    assert decision_context["decision_source"] == "hybrid_consensus"
    assert decision_context["hybrid_weight_source"] == "benchmark_history"
    assert decision_context["primary_weight"] == pytest.approx(0.10)
    assert applied == {
        "tfidf_recommended": pytest.approx(0.10),
        "transformer_neobert": pytest.approx(0.45),
        "transformer_modernbert_large": pytest.approx(0.45),
    }


def test_hybrid_decider_response_can_apply_hybrid_thresholds_without_routing() -> None:
    decider_result, decision_context = hybrid_decider_response(
        policy="hybrid_consensus",
        primary_result={
            "model_name": "tfidf_logreg",
            "display_name": "TF-IDF",
            "score": 0.70,
            "score_raw": 0.70,
            "score_calibrated": 0.70,
            "label": "not_askseattle",
            "confidence_band": "low",
            "low_threshold": 0.75,
            "high_threshold": 0.90,
            "time_source": None,
            "created_at": "2026-04-24T00:00:00Z",
        },
        primary_model_name="tfidf_recommended",
        row={"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
        comparisons=[],
        route_reasons=[],
        hybrid_policy={
            "source": "benchmarked_policy",
            "threshold_policy": {"low_threshold": 0.65, "high_threshold": 0.80},
            "weights": [
                {"name": "tfidf_recommended", "display_name": "TF-IDF", "weight": 1.0},
            ],
        },
        min_comparison_results=2,
    )

    assert decider_result is not None
    assert decider_result["label"] == "askseattle"
    assert decider_result["confidence_band"] == "borderline"
    assert decider_result["score"] == pytest.approx(0.70)
    assert decider_result["low_threshold"] == pytest.approx(0.65)
    assert decider_result["high_threshold"] == pytest.approx(0.80)
    assert decision_context["decision_source"] == "hybrid_primary_only"
    assert decision_context["routed"] is False
    assert decision_context["hybrid_threshold_source"] == "hybrid_policy"
    assert decision_context["hybrid_calibration_source"] == "identity"


def test_benchmark_hybrid_policy_entry_uses_history_weighting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = DatasetSplit(
        train=[
            LabeledPost("t0", "body", 1, post_id="t0"),
            LabeledPost("t1", "body", 0, post_id="t1"),
        ],
        calibration=[
            LabeledPost("c0", "body", 1, post_id="c0", post_type="text"),
            LabeledPost("c1", "body", 0, post_id="c1", post_type="text"),
        ],
        test=[
            LabeledPost("p0", "body", 1, post_id="p0", post_type="text"),
            LabeledPost("p1", "", 1, post_id="p1", post_type="image"),
            LabeledPost("p2", "body", 0, post_id="p2", post_type="text"),
            LabeledPost("p3", "body", 1, post_id="p3", post_type="text"),
            LabeledPost("p4", "", 0, post_id="p4", post_type="link"),
            LabeledPost("p5", "body", 1, post_id="p5", post_type="text"),
        ],
        split_strategy="random_eval_subreddit",
        split_seed=13,
        evaluation_subreddit="seattle",
    )
    rows = [
        {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
        {"post_type": "image", "is_low_text": "yes", "is_sparse_media": True},
        {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
        {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
        {"post_type": "link", "is_low_text": "yes", "is_sparse_media": True},
        {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
    ]
    scores_by_name = {
        "tfidf_recommended": [0.92, 0.70, 0.40, 0.80, 0.30, 0.78],
        "transformer_neobert": [0.87, 0.93, 0.25, 0.84, 0.12, 0.81],
        "transformer_modernbert_large": [0.86, 0.97, 0.45, 0.90, 0.20, 0.88],
    }

    def fake_payload(*, name: str, split: DatasetSplit, summary: dict, output_dir: Path) -> dict:
        return {
            "name": name,
            "display_name": summary.get("display_name") or name,
            "bundle": {
                "model_name": f"{name}_bundle",
                "model_family": "tfidf" if name == "tfidf_recommended" else "transformer_sequence_classifier",
                "model_id": None if name == "tfidf_recommended" else name,
                "low_threshold": 0.75,
                "high_threshold": 0.90,
            },
            "train_rows": [
                {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
                {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
            ],
            "calibration_rows": [
                {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
                {"post_type": "text", "is_low_text": "no", "is_sparse_media": False},
            ],
            "test_rows": rows,
            "calibration_scores": scores_by_name[name][:2],
            "scores": scores_by_name[name],
        }

    def fake_check_result_from_score(bundle: dict, row: dict, score: float) -> CheckResult:
        low_threshold = float(bundle["low_threshold"])
        high_threshold = float(bundle["high_threshold"])
        if score >= high_threshold:
            label = "askseattle"
            confidence_band = "high"
        elif score >= low_threshold:
            label = "askseattle"
            confidence_band = "borderline"
        else:
            label = "not_askseattle"
            confidence_band = "low"
        return CheckResult(
            post_id=None,
            permalink=None,
            model_name=str(bundle["model_name"]),
            display_name=str(bundle["model_name"]),
            model_version="test",
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            score=float(score),
            score_raw=float(score),
            score_calibrated=float(score),
            label=label,
            confidence_band=confidence_band,
            time_source=None,
            created_at="2026-04-21T00:00:00Z",
        )

    monkeypatch.setattr(training, "_hybrid_policy_model_payload", fake_payload)
    monkeypatch.setattr(training, "check_result_from_score", fake_check_result_from_score)

    benchmark_history_path = tmp_path / "benchmark_history.json"
    benchmark_history_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "split": {
                            "split_strategy": "random_eval_subreddit",
                            "evaluation_subreddit": "seattle",
                        },
                        "models": [
                            {
                                "name": "tfidf_recommended",
                                "display_name": "TF-IDF",
                                "status": "ok",
                                "production_ready": False,
                                "auto_recall_at_precision_95": 0.15,
                                "review_recall_at_precision_75": 0.70,
                                "pr_auc": 0.80,
                            },
                            {
                                "name": "transformer_neobert",
                                "display_name": "NeoBERT",
                                "status": "ok",
                                "production_ready": True,
                                "auto_recall_at_precision_95": 0.63,
                                "review_recall_at_precision_75": 0.88,
                                "pr_auc": 0.92,
                            },
                            {
                                "name": "transformer_modernbert_large",
                                "display_name": "ModernBERT-large",
                                "status": "ok",
                                "production_ready": True,
                                "auto_recall_at_precision_95": 0.60,
                                "review_recall_at_precision_75": 0.87,
                                "pr_auc": 0.90,
                            },
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    hybrid_entry = training._benchmark_hybrid_policy_entry(
        benchmark_dir=tmp_path,
        split=split,
        prepared_data_summary={"training_records": 8},
        benchmarked_summaries={
            "tfidf_recommended": {
                "display_name": "TF-IDF",
                "threshold_policy": {"low_threshold": 0.75, "high_threshold": 0.90},
                "calibration": {
                    "available": True,
                    "method": "sigmoid",
                    "positive_count": 1,
                    "negative_count": 1,
                    "calibration_size": 2,
                },
                "threshold_selection": {"high_threshold_selection": {"production_ready": True}},
            },
            "transformer_neobert": {"display_name": "NeoBERT"},
            "transformer_modernbert_large": {"display_name": "ModernBERT-large"},
        },
        comparison_suite_path=tmp_path / "benchmark_suite_summary.json",
        benchmark_history_path=benchmark_history_path,
    )

    assert hybrid_entry is not None
    assert hybrid_entry["name"] == "hybrid_consensus_policy"
    assert hybrid_entry["result_source"] == "benchmarked_policy"
    assert hybrid_entry["artifact_path"] is not None
    assert hybrid_entry["policy_metadata"]["hybrid_policy"]["source"] == "benchmark_history"
    assert hybrid_entry["policy_metadata"]["routed_count"] == 4
    assert hybrid_entry["policy_metadata"]["label_changed_count"] == 1
    assert hybrid_entry["policy_metadata"]["decision_source_counts"]["hybrid_consensus"] == 4
    assert hybrid_entry["policy_metadata"]["decision_source_counts"]["hybrid_primary_only"] == 2
