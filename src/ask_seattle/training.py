from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ask_seattle import __version__
from ask_seattle.data import LabeledPost, prepare_training_posts
from ask_seattle.model import (
    CalibrationResult,
    DecisionThresholds,
    ThresholdSelection,
    apply_probability_calibrator,
    build_inference_row,
    evaluate_decision_policy,
    fit_sigmoid_calibrator,
    positive_probabilities,
    save_model,
    select_decision_thresholds,
    split_labeled_posts,
    tfidf_feature_audit,
    train_model,
)

DEFAULT_HIGH_PRECISION_TARGET = 0.95
DEFAULT_CALIBRATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2


def train_model_bundle(
    posts: list[LabeledPost],
    output_dir: str | Path,
    *,
    prepared_data_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    split = split_labeled_posts(
        posts,
        calibration_size=DEFAULT_CALIBRATION_SIZE,
        test_size=DEFAULT_TEST_SIZE,
    )
    calibration_rows = [build_inference_row(title=post.title, selftext=post.selftext) for post in split.calibration]
    test_rows = [build_inference_row(title=post.title, selftext=post.selftext) for post in split.test]
    y_calibration = [post.label for post in split.calibration]
    y_test = [post.label for post in split.test]

    model = train_model(split.train)
    raw_calibration_scores = positive_probabilities(model, calibration_rows)
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        calibration=calibration,
    )
    raw_test_scores = positive_probabilities(model, test_rows)
    calibrated_test_scores = apply_probability_calibrator(calibrator, raw_test_scores)
    band_metrics = evaluate_decision_policy(
        y_test,
        calibrated_test_scores,
        low_threshold=thresholds.low_threshold,
        high_threshold=thresholds.high_threshold,
    )

    artifact_path = artifact_dir / "tfidf_logreg.joblib"
    threshold_policy = _decision_policy(
        split=split,
        calibration=calibration,
        thresholds=thresholds,
    )
    save_model(
        model,
        artifact_path,
        calibrator=calibrator,
        decision_policy=threshold_policy,
    )

    production_ready = (
        calibration.available
        and thresholds.high_threshold_selection.production_ready
        and band_metrics.high_confidence_precision >= DEFAULT_HIGH_PRECISION_TARGET
    )
    blocked_reason = None
    if not production_ready:
        if not calibration.available:
            blocked_reason = "calibration_unavailable"
        elif band_metrics.high_confidence_precision < DEFAULT_HIGH_PRECISION_TARGET:
            blocked_reason = "high_precision_target_not_met_on_test"
        else:
            blocked_reason = "high_precision_target_not_met_on_calibration"

    summary = {
        "version": __version__,
        "model_name": "tfidf_logreg",
        "artifact_path": str(artifact_path),
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "split": {
            "train": len(split.train),
            "calibration": len(split.calibration),
            "test": len(split.test),
            "split_strategy": split.split_strategy,
            "excluded_for_time_split": split.excluded_for_time_split,
            "time_coverage": split.time_coverage,
        },
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(thresholds),
        "metrics": {
            "high_confidence_precision": band_metrics.high_confidence_precision,
            "high_confidence_recall": band_metrics.high_confidence_recall,
            "high_confidence_f1": band_metrics.high_confidence_f1,
            "support": band_metrics.support,
            "confidence_band_counts": band_metrics.band_counts,
        },
        "threshold_policy": threshold_policy,
        "feature_audit": tfidf_feature_audit(model),
        "production_ready": production_ready,
        "production_ready_blocked_reason": blocked_reason,
    }
    if prepared_data_summary is not None:
        summary["prepared_data"] = prepared_data_summary
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def train_model_bundle_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    posts, prepared_data_summary = prepare_training_posts(input_path)
    return train_model_bundle(
        posts,
        output_dir,
        prepared_data_summary=prepared_data_summary,
    )


def _select_thresholds_or_default(
    y_calibration: list[int],
    probabilities: list[float],
    *,
    high_precision_target: float,
    calibration: CalibrationResult,
) -> DecisionThresholds:
    if calibration.available:
        return select_decision_thresholds(
            y_calibration,
            probabilities,
            auto_precision_target=high_precision_target,
        )

    support = Counter(y_calibration)[1]
    default_threshold = 0.85
    return DecisionThresholds(
        low_threshold=default_threshold,
        high_threshold=default_threshold,
        high_threshold_selection=_empty_threshold_selection(default_threshold, support=support),
        low_threshold_metrics={
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": support,
        },
        high_threshold_sweep=[],
        low_threshold_sweep=[],
        abstain_enabled=False,
    )


def _threshold_summary(thresholds: DecisionThresholds) -> dict[str, Any]:
    return {
        "low_threshold": thresholds.low_threshold,
        "high_threshold": thresholds.high_threshold,
        "abstain_enabled": thresholds.abstain_enabled,
        "high_threshold_selection": asdict(thresholds.high_threshold_selection),
        "low_threshold_metrics": thresholds.low_threshold_metrics,
        "high_threshold_sweep": thresholds.high_threshold_sweep,
        "low_threshold_sweep": thresholds.low_threshold_sweep,
    }


def _decision_policy(
    *,
    split: Any,
    calibration: CalibrationResult,
    thresholds: DecisionThresholds,
) -> dict[str, Any]:
    return {
        "low_threshold": thresholds.low_threshold,
        "high_threshold": thresholds.high_threshold,
        "calibration_method": calibration.method,
        "split_strategy": split.split_strategy,
        "time_coverage": split.time_coverage,
    }


def _empty_threshold_selection(threshold: float, *, support: int) -> ThresholdSelection:
    return ThresholdSelection(
        threshold=threshold,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        support=support,
        production_ready=False,
    )
