from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import json
import logging
from math import ceil
from pathlib import Path
import sys
import types
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from ask_seattle import __version__
from ask_seattle.data import (
    DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    LabeledPost,
    body_length_bucket,
    has_question_mark,
    is_sparse_media_post,
    is_low_text_body,
    normalize_body,
    normalize_urls_for_lexical_text,
    post_metadata_text,
    post_text,
    title_length_bucket,
)

DEFAULT_THRESHOLD_GRID = tuple(round(index / 100, 2) for index in range(5, 100, 5))
DEFAULT_SPLIT_STRATEGY = "random"
DEFAULT_SPLIT_SEED = 13
DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION = "v3_compact_contextual_fields"
DEFAULT_TFIDF_CONFIG_VERSION = "v8_tfidf_bootstrap_precision_stable_sparse_observational"
DEFAULT_REVIEW_PRECISION_TARGET = 0.75
DEFAULT_MIN_HIGH_CONFIDENCE_CALIBRATION_PREDICTIONS = 5
DEFAULT_THRESHOLD_BOOTSTRAP_SAMPLE_COUNT = 200
DEFAULT_THRESHOLD_BOOTSTRAP_PRECISION_PERCENTILE = 0.20
DEFAULT_THRESHOLD_BOOTSTRAP_SEED = 13
DEFAULT_LOW_TEXT_HIGH_THRESHOLD_DELTA = 0.03
DEFAULT_IMAGE_HIGH_THRESHOLD_DELTA = 0.04
DEFAULT_SPARSE_MEDIA_HIGH_THRESHOLD_DELTA = 0.05
LOGGER = logging.getLogger("ask_seattle.model")
WORD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "with",
        "without",
    }
)
DEFAULT_EXTRA_WORD_STOPWORDS = frozenset({"just", "one", "some"})
DEFAULT_CHAR_WEIGHT = 0.25
DEFAULT_METADATA_WEIGHT = 0.4
DEFAULT_TFIDF_URL_NORMALIZATION = True
DEFAULT_TFIDF_STRIP_URLS = False
STACKED_TRANSFORMER_DECIDER_MODEL_FAMILY = "stacked_transformer_decider"


class TextFieldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def fit(self, rows: list[dict[str, str]], y: list[int] | None = None) -> TextFieldExtractor:
        return self

    def transform(self, rows: list[dict[str, str]]) -> list[str]:
        return [str(row.get(self.field_name) or "") for row in rows]


@dataclass(frozen=True)
class DatasetSplit:
    train: list[LabeledPost]
    calibration: list[LabeledPost]
    test: list[LabeledPost]
    split_strategy: str
    split_seed: int | None = None
    excluded_for_time_split: int = 0
    time_coverage: dict[str, dict[str, Any]] | None = None
    evaluation_subreddit: str | None = None

    @property
    def validation(self) -> list[LabeledPost]:
        return self.calibration


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive: int
    support: int
    production_ready: bool
    bootstrap_precision_p20: float | None = None
    bootstrap_precision_mean: float | None = None
    bootstrap_precision_min: float | None = None
    bootstrap_predicted_positive_p20: int | None = None
    bootstrap_predicted_positive_mean: float | None = None
    bootstrap_predicted_positive_min: int | None = None
    bootstrap_sample_count: int = 0
    bootstrap_ready: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DecisionThresholds:
    low_threshold: float
    high_threshold: float
    high_threshold_selection: ThresholdSelection
    low_threshold_metrics: dict[str, float | int]
    high_threshold_sweep: list[dict[str, float | int]]
    low_threshold_sweep: list[dict[str, float | int]]
    abstain_enabled: bool
    minimum_high_confidence_calibration_predictions: int
    high_threshold_fallback_used: bool


@dataclass(frozen=True)
class CalibrationResult:
    available: bool
    method: str | None
    brier_score: float | None
    log_loss: float | None
    positive_count: int
    negative_count: int
    calibration_size: int


@dataclass(frozen=True)
class ConfidenceBandMetrics:
    high_confidence_precision: float
    high_confidence_recall: float
    high_confidence_f1: float
    support: int
    band_counts: dict[str, int]


@dataclass(frozen=True)
class CheckResult:
    post_id: str | None
    permalink: str | None
    model_name: str
    display_name: str
    model_version: str
    low_threshold: float
    high_threshold: float
    score: float
    score_raw: float
    score_calibrated: float
    label: str
    confidence_band: str
    time_source: str | None
    created_at: str


def transformer_requires_trust_remote_code(model_id: str) -> bool:
    return "neobert" in str(model_id).lower()


def transformer_load_options(model_id: str) -> dict[str, Any]:
    if transformer_requires_trust_remote_code(model_id):
        return {"trust_remote_code": True}
    return {}


def ensure_transformer_custom_code_support(*, trust_remote_code: bool) -> None:
    if not trust_remote_code:
        return
    _install_xformers_swiglu_fallback()


def _install_xformers_swiglu_fallback() -> None:
    try:
        import xformers.ops  # noqa: F401
        return
    except Exception:
        pass

    import torch.nn.functional as functional
    from torch import nn

    class SwiGLU(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            *,
            bias: bool = True,
        ) -> None:
            super().__init__()
            self.w12 = nn.Linear(in_features, hidden_features * 2, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, inputs: Any) -> Any:
            gated_inputs, linear_inputs = self.w12(inputs).chunk(2, dim=-1)
            return self.w3(functional.silu(gated_inputs) * linear_inputs)

    xformers_module = sys.modules.get("xformers")
    if xformers_module is None:
        xformers_module = types.ModuleType("xformers")
        sys.modules["xformers"] = xformers_module

    ops_module = types.ModuleType("xformers.ops")
    ops_module.SwiGLU = SwiGLU
    xformers_module.ops = ops_module
    sys.modules["xformers.ops"] = ops_module


def build_pipeline(*, min_df: int = 2) -> Pipeline:
    return build_pipeline_with_config(
        min_df=min_df,
        extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
        char_weight=DEFAULT_CHAR_WEIGHT,
        metadata_weight=DEFAULT_METADATA_WEIGHT,
        normalize_urls=DEFAULT_TFIDF_URL_NORMALIZATION,
        strip_urls=DEFAULT_TFIDF_STRIP_URLS,
        classifier_c=1.0,
        classifier_class_weight="balanced",
    )


def build_pipeline_with_config(
    *,
    min_df: int = 2,
    extra_word_stopwords: set[str] | frozenset[str] | None = None,
    char_weight: float = DEFAULT_CHAR_WEIGHT,
    metadata_weight: float = DEFAULT_METADATA_WEIGHT,
    normalize_urls: bool = DEFAULT_TFIDF_URL_NORMALIZATION,
    strip_urls: bool = DEFAULT_TFIDF_STRIP_URLS,
    classifier_c: float = 1.0,
    classifier_class_weight: str | dict[int, float] | None = "balanced",
) -> Pipeline:
    word_stopwords = sorted(WORD_STOPWORDS | set(extra_word_stopwords or ()))
    if normalize_urls:
        title_field = "title_lexical_stripped" if strip_urls else "title_lexical"
        body_field = "body_lexical_stripped" if strip_urls else "body_lexical"
        char_field = "text_lexical_stripped" if strip_urls else "text_lexical"
    else:
        title_field = "title"
        body_field = "body_raw"
        char_field = "text_raw"
    features = FeatureUnion(
        [
            (
                "title_word",
                Pipeline(
                    [
                        ("extractor", TextFieldExtractor(title_field)),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 3),
                                min_df=min_df,
                                max_df=0.95,
                                strip_accents="unicode",
                                stop_words=word_stopwords,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "body_word",
                Pipeline(
                    [
                        ("extractor", TextFieldExtractor(body_field)),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=min_df,
                                max_df=0.98,
                                strip_accents="unicode",
                                stop_words=word_stopwords,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "char_wb",
                Pipeline(
                    [
                        ("extractor", TextFieldExtractor(char_field)),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=min_df,
                                max_df=0.99,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "metadata_token",
                Pipeline(
                    [
                        ("extractor", TextFieldExtractor("metadata_text")),
                        (
                            "vectorizer",
                            CountVectorizer(
                                binary=True,
                                lowercase=False,
                                min_df=min_df,
                                preprocessor=None,
                                tokenizer=str.split,
                                token_pattern=None,
                            ),
                        ),
                    ]
                ),
            ),
        ],
        transformer_weights={
            "title_word": 2.0,
            "body_word": 1.0,
            "char_wb": char_weight,
            "metadata_token": metadata_weight,
        },
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    class_weight=classifier_class_weight,
                    C=classifier_c,
                    max_iter=2_000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_model(
    posts: list[LabeledPost],
    *,
    sample_weight: list[float] | None = None,
    extra_word_stopwords: set[str] | frozenset[str] | None = None,
    char_weight: float = DEFAULT_CHAR_WEIGHT,
    metadata_weight: float = DEFAULT_METADATA_WEIGHT,
    normalize_urls: bool = DEFAULT_TFIDF_URL_NORMALIZATION,
    strip_urls: bool = DEFAULT_TFIDF_STRIP_URLS,
    min_df: int | None = None,
    classifier_c: float = 1.0,
    classifier_class_weight: str | dict[int, float] | None = "balanced",
    include_sparse_media_token: bool = DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    include_image_low_text_tokens: bool = DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
) -> Pipeline:
    _validate_posts(posts)
    effective_word_stopwords = (
        DEFAULT_EXTRA_WORD_STOPWORDS if extra_word_stopwords is None else set(extra_word_stopwords)
    )
    model = build_pipeline_with_config(
        min_df=_default_min_df(posts) if min_df is None else int(min_df),
        extra_word_stopwords=effective_word_stopwords,
        char_weight=char_weight,
        metadata_weight=metadata_weight,
        normalize_urls=normalize_urls,
        strip_urls=strip_urls,
        classifier_c=classifier_c,
        classifier_class_weight=classifier_class_weight,
    )
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight
    model.fit(
        _rows(
            posts,
            include_sparse_media_token=include_sparse_media_token,
            include_image_low_text_tokens=include_image_low_text_tokens,
        ),
        _labels(posts),
        **fit_kwargs,
    )
    return model


def split_labeled_posts(
    posts: list[LabeledPost],
    *,
    calibration_size: float,
    test_size: float,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
) -> DatasetSplit:
    _validate_posts(posts)
    if not 0 < calibration_size < 1 or not 0 < test_size < 1:
        raise ValueError("calibration_size and test_size must be between 0 and 1")
    if calibration_size + test_size >= 1:
        raise ValueError("calibration_size + test_size must be less than 1")
    if split_strategy not in {"random", "time"}:
        raise ValueError("split_strategy must be one of {'random', 'time'}")

    if split_strategy == "time":
        return _time_split(
            posts,
            calibration_size=calibration_size,
            test_size=test_size,
            evaluation_subreddit=evaluation_subreddit,
        )

    return _random_split(
        posts,
        calibration_size=calibration_size,
        test_size=test_size,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )


def threshold_sweep(
    y_true: list[int],
    probabilities: list[float],
    thresholds: tuple[float, ...] | None = None,
) -> list[dict[str, float | int]]:
    resolved_thresholds = _resolve_thresholds(probabilities, thresholds)
    sweep: list[dict[str, float | int]] = []
    for threshold in resolved_thresholds:
        predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        sweep.append({"threshold": threshold} | _binary_metrics(y_true, predictions))
    return sweep


def _bootstrap_sample_indices(size: int, sample_count: int, *, seed: int) -> np.ndarray:
    if size <= 0 or sample_count <= 0:
        return np.empty((0, 0), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, size, size=(sample_count, size), dtype=np.int64)


def _bootstrap_percentile_floor(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(np.asarray(values))
    index = int(np.floor((len(ordered) - 1) * float(percentile)))
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered[index])


def _bootstrap_threshold_diagnostics(
    y_true: list[int],
    probabilities: list[float],
    *,
    threshold: float,
    sample_indices: np.ndarray,
    precision_percentile: float,
) -> dict[str, float | int]:
    if sample_indices.size == 0:
        return {
            "bootstrap_precision_p20": None,
            "bootstrap_precision_mean": None,
            "bootstrap_precision_min": None,
            "bootstrap_predicted_positive_p20": None,
            "bootstrap_predicted_positive_mean": None,
            "bootstrap_predicted_positive_min": None,
            "bootstrap_sample_count": 0,
        }

    y_array = np.asarray(y_true, dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=np.float32)
    sampled_labels = y_array[sample_indices]
    sampled_predictions = probability_array[sample_indices] >= float(threshold)
    predicted_positive = sampled_predictions.sum(axis=1).astype(np.int64)
    true_positive = np.logical_and(sampled_predictions, sampled_labels == 1).sum(axis=1).astype(np.int64)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_positive > 0,
    )
    return {
        "bootstrap_precision_p20": _bootstrap_percentile_floor(precision, precision_percentile),
        "bootstrap_precision_mean": float(precision.mean()),
        "bootstrap_precision_min": float(precision.min()),
        "bootstrap_predicted_positive_p20": int(_bootstrap_percentile_floor(predicted_positive, precision_percentile)),
        "bootstrap_predicted_positive_mean": float(np.asarray(predicted_positive, dtype=np.float64).mean()),
        "bootstrap_predicted_positive_min": int(np.asarray(predicted_positive).min()),
        "bootstrap_sample_count": int(sample_indices.shape[0]),
    }


def select_threshold(
    y_true: list[int],
    probabilities: list[float],
    *,
    min_precision: float = 0.95,
    minimum_predictions: int = 0,
    thresholds: tuple[float, ...] | None = None,
    bootstrap_sample_count: int = DEFAULT_THRESHOLD_BOOTSTRAP_SAMPLE_COUNT,
    bootstrap_precision_percentile: float = DEFAULT_THRESHOLD_BOOTSTRAP_PRECISION_PERCENTILE,
    bootstrap_seed: int = DEFAULT_THRESHOLD_BOOTSTRAP_SEED,
) -> ThresholdSelection:
    sweep = threshold_sweep(y_true, probabilities, thresholds)
    use_bootstrap_gate = minimum_predictions > 0 and bootstrap_sample_count > 0
    sample_indices = (
        _bootstrap_sample_indices(
            len(y_true),
            bootstrap_sample_count,
            seed=bootstrap_seed,
        )
        if use_bootstrap_gate
        else np.empty((0, 0), dtype=np.int64)
    )
    for row in sweep:
        row.update(
            {
                "bootstrap_precision_p20": None,
                "bootstrap_precision_mean": None,
                "bootstrap_precision_min": None,
                "bootstrap_predicted_positive_p20": None,
                "bootstrap_predicted_positive_mean": None,
                "bootstrap_predicted_positive_min": None,
                "bootstrap_sample_count": 0,
                "bootstrap_ready": False,
            }
        )
        if not use_bootstrap_gate or float(row["precision"]) < min_precision:
            continue
        row.update(
            _bootstrap_threshold_diagnostics(
                y_true,
                probabilities,
                threshold=float(row["threshold"]),
                sample_indices=sample_indices,
                precision_percentile=bootstrap_precision_percentile,
            )
        )
        row["bootstrap_ready"] = (
            int(row.get("predicted_positive") or 0) >= int(minimum_predictions)
            and float(row.get("bootstrap_precision_p20") or 0.0) >= float(min_precision)
        )
    precision_ready = [row for row in sweep if float(row["precision"]) >= min_precision]
    ready = [row for row in precision_ready if int(row.get("predicted_positive") or 0) >= int(minimum_predictions)]
    bootstrap_ready = [row for row in ready if bool(row.get("bootstrap_ready"))]
    if not use_bootstrap_gate:
        candidates = precision_ready or sweep
        fallback_used = False
        fallback_reason = None if precision_ready else "high_precision_target_not_met"
    elif bootstrap_ready:
        candidates = bootstrap_ready
        fallback_used = False
        fallback_reason = None
    elif ready:
        candidates = ready
        fallback_used = True
        fallback_reason = "bootstrap_precision_target_not_met"
    elif precision_ready:
        candidates = precision_ready
        fallback_used = True
        fallback_reason = "minimum_high_confidence_calibration_predictions_not_met"
    else:
        candidates = sweep
        fallback_used = False
        fallback_reason = "high_precision_target_not_met"
    selected = max(
        candidates,
        key=lambda row: (
            float(row["recall"]),
            float(row.get("bootstrap_precision_p20") or -1.0),
            int(row.get("predicted_positive") or 0),
            float(row["threshold"]),
        ),
    )

    return ThresholdSelection(
        threshold=float(selected["threshold"]),
        precision=float(selected["precision"]),
        recall=float(selected["recall"]),
        f1=float(selected["f1"]),
        predicted_positive=int(selected.get("predicted_positive") or 0),
        support=int(selected["support"]),
        production_ready=(bool(precision_ready) and not use_bootstrap_gate) or (bool(bootstrap_ready) and not fallback_used),
        bootstrap_precision_p20=(
            float(selected["bootstrap_precision_p20"])
            if selected.get("bootstrap_precision_p20") is not None
            else None
        ),
        bootstrap_precision_mean=(
            float(selected["bootstrap_precision_mean"])
            if selected.get("bootstrap_precision_mean") is not None
            else None
        ),
        bootstrap_precision_min=(
            float(selected["bootstrap_precision_min"])
            if selected.get("bootstrap_precision_min") is not None
            else None
        ),
        bootstrap_predicted_positive_p20=(
            int(selected["bootstrap_predicted_positive_p20"])
            if selected.get("bootstrap_predicted_positive_p20") is not None
            else None
        ),
        bootstrap_predicted_positive_mean=(
            float(selected["bootstrap_predicted_positive_mean"])
            if selected.get("bootstrap_predicted_positive_mean") is not None
            else None
        ),
        bootstrap_predicted_positive_min=(
            int(selected["bootstrap_predicted_positive_min"])
            if selected.get("bootstrap_predicted_positive_min") is not None
            else None
        ),
        bootstrap_sample_count=int(selected.get("bootstrap_sample_count") or 0),
        bootstrap_ready=bool(selected.get("bootstrap_ready")),
        fallback_reason=fallback_reason,
    )


def select_decision_thresholds(
    y_true: list[int],
    probabilities: list[float],
    *,
    auto_precision_target: float,
    review_precision_target: float = DEFAULT_REVIEW_PRECISION_TARGET,
    minimum_high_confidence_calibration_predictions: int = DEFAULT_MIN_HIGH_CONFIDENCE_CALIBRATION_PREDICTIONS,
    thresholds: tuple[float, ...] | None = None,
    bootstrap_sample_count: int = DEFAULT_THRESHOLD_BOOTSTRAP_SAMPLE_COUNT,
    bootstrap_precision_percentile: float = DEFAULT_THRESHOLD_BOOTSTRAP_PRECISION_PERCENTILE,
    bootstrap_seed: int = DEFAULT_THRESHOLD_BOOTSTRAP_SEED,
) -> DecisionThresholds:
    high_threshold_selection = select_threshold(
        y_true,
        probabilities,
        min_precision=auto_precision_target,
        minimum_predictions=minimum_high_confidence_calibration_predictions,
        thresholds=thresholds,
        bootstrap_sample_count=bootstrap_sample_count,
        bootstrap_precision_percentile=bootstrap_precision_percentile,
        bootstrap_seed=bootstrap_seed,
    )
    high_threshold_sweep = threshold_sweep(y_true, probabilities, thresholds)
    low_threshold_sweep = high_threshold_sweep
    low_ready = [
        row for row in low_threshold_sweep if float(row["precision"]) >= review_precision_target
    ]
    best_low = max(
        low_ready or low_threshold_sweep,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            float(row["f1"]),
            -float(row["threshold"]),
        ),
    )
    low_threshold = min(float(best_low["threshold"]), high_threshold_selection.threshold)
    low_metrics = _binary_metrics(
        y_true,
        [1 if probability >= low_threshold else 0 for probability in probabilities],
    )

    return DecisionThresholds(
        low_threshold=low_threshold,
        high_threshold=high_threshold_selection.threshold,
        high_threshold_selection=high_threshold_selection,
        low_threshold_metrics=low_metrics,
        high_threshold_sweep=high_threshold_sweep,
        low_threshold_sweep=low_threshold_sweep,
        abstain_enabled=low_threshold < high_threshold_selection.threshold,
        minimum_high_confidence_calibration_predictions=minimum_high_confidence_calibration_predictions,
        high_threshold_fallback_used=high_threshold_selection.fallback_reason
        in {
            "bootstrap_precision_target_not_met",
            "minimum_high_confidence_calibration_predictions_not_met",
        },
    )


def fit_sigmoid_calibrator(
    y_true: list[int],
    probabilities: list[float],
) -> tuple[LogisticRegression | None, CalibrationResult]:
    class_counts = Counter(y_true)
    if len(class_counts) < 2:
        return None, CalibrationResult(
            available=False,
            method=None,
            brier_score=None,
            log_loss=None,
            positive_count=class_counts.get(1, 0),
            negative_count=class_counts.get(0, 0),
            calibration_size=len(y_true),
        )

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit([[probability] for probability in probabilities], y_true)
    calibrated = apply_probability_calibrator(calibrator, probabilities)
    return calibrator, CalibrationResult(
        available=True,
        method="sigmoid",
        brier_score=float(brier_score_loss(y_true, calibrated)),
        log_loss=float(log_loss(y_true, calibrated, labels=[0, 1])),
        positive_count=class_counts.get(1, 0),
        negative_count=class_counts.get(0, 0),
        calibration_size=len(y_true),
    )


def apply_probability_calibrator(
    calibrator: LogisticRegression | None,
    probabilities: list[float],
) -> list[float]:
    if calibrator is None:
        return list(probabilities)
    calibrated = calibrator.predict_proba([[probability] for probability in probabilities])
    return [float(row[1]) for row in calibrated]


def evaluate_decision_policy(
    y_true: list[int],
    probabilities: list[float],
    *,
    low_threshold: float,
    high_threshold: float,
    rows: list[dict[str, Any]] | None = None,
) -> ConfidenceBandMetrics:
    auto_predictions = [
        1
        if probability >= _effective_high_threshold_for_row(row, high_threshold=high_threshold)
        else 0
        for probability, row in zip(probabilities, rows or [{}] * len(probabilities), strict=True)
    ]
    metrics = _binary_metrics(y_true, auto_predictions)
    band_counts = Counter(
        confidence_band_for_row(
            row,
            probability,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        for probability, row in zip(probabilities, rows or [{}] * len(probabilities), strict=True)
    )
    return ConfidenceBandMetrics(
        high_confidence_precision=float(metrics["precision"]),
        high_confidence_recall=float(metrics["recall"]),
        high_confidence_f1=float(metrics["f1"]),
        support=int(metrics["support"]),
        band_counts={
            "high": band_counts.get("high", 0),
            "borderline": band_counts.get("borderline", 0),
            "low": band_counts.get("low", 0),
        },
    )


def tfidf_feature_audit(model: Pipeline, *, limit: int = 20) -> dict[str, list[dict[str, float | str]]]:
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    records = _feature_records(features, classifier.coef_[0])

    return {
        "word_stopwords": _configured_word_stopwords(features),
        "top_positive": _rank_feature_records(records, limit=limit, reverse=True),
        "top_negative": _rank_feature_records(records, limit=limit, reverse=False),
        "top_positive_by_channel": {
            channel: _rank_feature_records(
                [record for record in records if str(record["channel"]) == channel],
                limit=limit,
                reverse=True,
            )
            for channel, _transformer in features.transformer_list
        },
        "top_negative_by_channel": {
            channel: _rank_feature_records(
                [record for record in records if str(record["channel"]) == channel],
                limit=limit,
                reverse=False,
            )
            for channel, _transformer in features.transformer_list
        },
    }


def save_model(
    model: Pipeline,
    path: str | Path,
    *,
    threshold: float | None = None,
    calibrator: LogisticRegression | None = None,
    decision_policy: dict[str, Any] | None = None,
    representation_config: dict[str, bool] | None = None,
) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    high_threshold = float(
        decision_policy.get("high_threshold", threshold if threshold is not None else 0.85)
        if decision_policy
        else (threshold if threshold is not None else 0.85)
    )
    low_threshold = float(decision_policy.get("low_threshold", high_threshold) if decision_policy else high_threshold)
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_family": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": __version__,
        "tfidf_config_version": DEFAULT_TFIDF_CONFIG_VERSION,
        "threshold": high_threshold,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "threshold_policy": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "review_precision_target": (
                decision_policy.get("review_precision_target") if decision_policy else DEFAULT_REVIEW_PRECISION_TARGET
            ),
            "high_precision_target": decision_policy.get("high_precision_target") if decision_policy else None,
            "calibration_method": decision_policy.get("calibration_method") if decision_policy else None,
            "split_strategy": decision_policy.get("split_strategy") if decision_policy else "manual",
            "split_seed": decision_policy.get("split_seed") if decision_policy else None,
            "evaluation_subreddit": decision_policy.get("evaluation_subreddit") if decision_policy else None,
            "time_coverage": decision_policy.get("time_coverage") if decision_policy else None,
        },
        "calibration_method": decision_policy.get("calibration_method") if decision_policy else None,
        "split_strategy": decision_policy.get("split_strategy") if decision_policy else "manual",
        "split_seed": decision_policy.get("split_seed") if decision_policy else None,
        "evaluation_subreddit": decision_policy.get("evaluation_subreddit") if decision_policy else None,
        "time_coverage": decision_policy.get("time_coverage") if decision_policy else None,
        "calibrator": calibrator,
        "representation_config": _normalize_representation_config(representation_config),
        "positive_label": 1,
        "version": __version__,
    }
    joblib.dump(bundle, model_path)


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if model_path.is_dir():
        return _load_transformer_bundle(model_path)

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        msg = f"{path} is not an ask-seattle model bundle"
        raise ValueError(msg)
    model_family = str(bundle.get("model_family") or bundle.get("model_type") or "")
    if model_family == STACKED_TRANSFORMER_DECIDER_MODEL_FAMILY:
        return _load_stacked_transformer_bundle_from_joblib(bundle, source_path=model_path)
    if model_family == "hybrid_decider_policy":
        return _load_hybrid_policy_bundle_from_joblib(bundle)
    if model_family == "transformer_sequence_classifier":
        return _load_transformer_bundle_from_joblib(bundle, source_path=model_path)
    if model_family == "causal_lm_classifier":
        return _load_causal_lm_bundle_from_joblib(bundle, source_path=model_path)
    if model_family == "semantic_embedding" and "classifier" in bundle:
        return _load_semantic_bundle(bundle)
    if "model" in bundle:
        return _normalize_tfidf_bundle(bundle)
    msg = f"{path} is not an ask-seattle model bundle"
    raise ValueError(msg)


def score_post_raw(
    bundle: dict[str, Any],
    *,
    title: str,
    selftext: str = "",
    post_type: str | None = None,
    content_domain: str | None = None,
    is_crosspost: bool | None = None,
) -> float:
    representation_config = _bundle_representation_config(bundle)
    return raw_score_rows(
        bundle,
        [
            build_inference_row(
                title=title,
                selftext=selftext,
                post_type=post_type,
                content_domain=content_domain,
                is_crosspost=is_crosspost,
                include_sparse_media_token=representation_config["include_sparse_media_token"],
                include_image_low_text_tokens=representation_config["include_image_low_text_tokens"],
            )
        ],
    )[0]


def score_post(
    bundle: dict[str, Any],
    *,
    title: str,
    selftext: str = "",
    post_type: str | None = None,
    content_domain: str | None = None,
    is_crosspost: bool | None = None,
) -> float:
    representation_config = _bundle_representation_config(bundle)
    return score_rows(
        bundle,
        [
            build_inference_row(
                title=title,
                selftext=selftext,
                post_type=post_type,
                content_domain=content_domain,
                is_crosspost=is_crosspost,
                include_sparse_media_token=representation_config["include_sparse_media_token"],
                include_image_low_text_tokens=representation_config["include_image_low_text_tokens"],
            )
        ],
    )[0]


def score_rows(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    raw = raw_score_rows(bundle, rows)
    return apply_probability_calibrator(bundle.get("calibrator"), raw)


def raw_score_rows(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    family = str(bundle.get("model_family") or bundle.get("model_type") or "tfidf")
    if family == "tfidf":
        model = bundle["model"]
        return positive_probabilities(model, rows)
    if family == "semantic_embedding":
        return _semantic_positive_probabilities(bundle, rows)
    if family == "transformer_sequence_classifier":
        return _transformer_positive_probabilities(bundle, rows)
    if family == "causal_lm_classifier":
        return _causal_lm_positive_probabilities(bundle, rows)
    if family == STACKED_TRANSFORMER_DECIDER_MODEL_FAMILY:
        return _stacked_transformer_positive_probabilities(bundle, rows)
    raise ValueError(f"Unsupported model family: {family}")


def stacked_transformer_feature_names(component_names: list[str]) -> list[str]:
    return [
        *[f"score_{name}" for name in component_names],
        "score_mean",
        "score_max",
        "score_min",
        "score_spread",
        "score_top_gap",
        "post_type_text",
        "post_type_link",
        "post_type_image",
        "post_type_other",
        "is_low_text",
        "is_sparse_media",
        "is_crosspost_yes",
    ]


def stacked_transformer_feature_matrix(
    rows: list[dict[str, Any]],
    component_scores_by_name: dict[str, list[float]],
    *,
    component_names: list[str],
) -> np.ndarray:
    if not component_names:
        raise ValueError("stacked transformer feature matrix requires at least one component model")
    column_scores = [
        np.asarray(component_scores_by_name.get(name) or [], dtype=np.float32)
        for name in component_names
    ]
    expected_size = len(rows)
    if any(scores.shape[0] != expected_size for scores in column_scores):
        raise ValueError("stacked transformer component scores must match row count")
    score_matrix = np.column_stack(column_scores).astype(np.float32)
    sorted_scores = np.sort(score_matrix, axis=1)
    top_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2] if score_matrix.shape[1] > 1 else sorted_scores[:, -1]
    post_type_text: list[float] = []
    post_type_link: list[float] = []
    post_type_image: list[float] = []
    post_type_other: list[float] = []
    low_text: list[float] = []
    sparse_media: list[float] = []
    crosspost_yes: list[float] = []
    for row in rows:
        post_type = str(row.get("post_type") or "").strip().lower()
        post_type_text.append(1.0 if post_type == "text" else 0.0)
        post_type_link.append(1.0 if post_type == "link" else 0.0)
        post_type_image.append(1.0 if post_type == "image" else 0.0)
        post_type_other.append(1.0 if post_type not in {"text", "link", "image"} else 0.0)
        low_text.append(1.0 if str(row.get("is_low_text") or "").strip().lower() == "yes" else 0.0)
        sparse_media.append(1.0 if bool(row.get("is_sparse_media")) else 0.0)
        crosspost_yes.append(1.0 if str(row.get("is_crosspost") or "").strip().lower() == "yes" else 0.0)
    return np.column_stack(
        [
            score_matrix,
            np.mean(score_matrix, axis=1, dtype=np.float32),
            np.max(score_matrix, axis=1),
            np.min(score_matrix, axis=1),
            np.max(score_matrix, axis=1) - np.min(score_matrix, axis=1),
            top_scores - second_scores,
            np.asarray(post_type_text, dtype=np.float32),
            np.asarray(post_type_link, dtype=np.float32),
            np.asarray(post_type_image, dtype=np.float32),
            np.asarray(post_type_other, dtype=np.float32),
            np.asarray(low_text, dtype=np.float32),
            np.asarray(sparse_media, dtype=np.float32),
            np.asarray(crosspost_yes, dtype=np.float32),
        ]
    ).astype(np.float32)


def positive_probabilities(model: Pipeline, rows: list[dict[str, str]]) -> list[float]:
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(rows)
    positive_index = list(classifier.classes_).index(1)
    return [float(row[positive_index]) for row in probabilities]


def _semantic_positive_probabilities(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    classifier = bundle.get("classifier")
    if classifier is None:
        raise ValueError("Semantic embedding bundle is missing classifier")
    feature_layout = str(bundle.get("feature_layout") or "single_text")
    backend = str(bundle.get("backend") or "sentence_transformers")
    if feature_layout == "title_body_metadata_v1":
        title_weight = float(bundle.get("title_weight") or 1.0)
        body_weight = float(bundle.get("body_weight") or 1.0)
        title_texts = _semantic_runtime_component_texts(bundle, rows, component="title")
        body_texts = _semantic_runtime_component_texts(bundle, rows, component="body")
        if backend == "sentence_transformers":
            encoder = bundle.get("encoder")
            if encoder is None:
                raise ValueError("Semantic embedding bundle is missing encoder")
            title_embeddings = np.asarray(
                encoder.encode(
                    title_texts,
                    show_progress_bar=False,
                    normalize_embeddings=bool(bundle.get("normalize_embeddings")),
                ),
                dtype=np.float32,
            )
            body_embeddings = np.asarray(
                encoder.encode(
                    body_texts,
                    show_progress_bar=False,
                    normalize_embeddings=bool(bundle.get("normalize_embeddings")),
                ),
                dtype=np.float32,
            )
        elif backend == "hf_embedding":
            title_embeddings = _hf_embedding_runtime_embeddings(bundle, title_texts)
            body_embeddings = _hf_embedding_runtime_embeddings(bundle, body_texts)
        else:
            raise ValueError(f"Unsupported semantic embedding backend: {backend}")
        metadata_features = _semantic_runtime_metadata_features(bundle, rows)
        feature_matrix = np.hstack(
            [title_embeddings * title_weight, body_embeddings * body_weight, metadata_features]
        ).astype(np.float32)
        probabilities = classifier.predict_proba(feature_matrix)
    else:
        texts = _semantic_runtime_texts(bundle, rows)
        if backend == "sentence_transformers":
            encoder = bundle.get("encoder")
            if encoder is None:
                raise ValueError("Semantic embedding bundle is missing encoder")
            embeddings = encoder.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=bool(bundle.get("normalize_embeddings")),
            )
        elif backend == "hf_embedding":
            embeddings = _hf_embedding_runtime_embeddings(bundle, texts)
        else:
            raise ValueError(f"Unsupported semantic embedding backend: {backend}")
        probabilities = classifier.predict_proba(embeddings)
    positive_index = list(classifier.classes_).index(1)
    return [float(row[positive_index]) for row in probabilities]


def _transformer_positive_probabilities(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "Transformer inference requires torch. Install optional model dependencies with "
            "`python -m pip install -e \".[dev,models]\"`."
        ) from exc

    model = bundle.get("model")
    tokenizer = bundle.get("tokenizer")
    if model is None or tokenizer is None:
        raise ValueError("Transformer bundle is missing model or tokenizer")
    device = _bundle_runtime_device(bundle, torch)
    try:
        return _transformer_positive_probabilities_on_device(bundle, rows, device=device, torch_module=torch)
    except (NotImplementedError, RuntimeError) as exc:
        if device == "mps" and _should_retry_transformer_inference_on_cpu(exc):
            LOGGER.warning(
                "transformer inference leaving mps model_id=%s reason=%s retrying_runtime=cpu",
                bundle.get("model_id") or "",
                str(exc),
            )
            _clear_torch_inference_memory(torch)
            return _transformer_positive_probabilities_on_device(bundle, rows, device="cpu", torch_module=torch)
        raise


def _transformer_positive_probabilities_on_device(
    bundle: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    device: str,
    torch_module: Any,
) -> list[float]:
    model = bundle.get("model")
    tokenizer = bundle.get("tokenizer")
    if model is None or tokenizer is None:
        raise ValueError("Transformer bundle is missing model or tokenizer")
    batch_size = _transformer_inference_batch_size(bundle)

    titles = [str(row.get("title") or "") for row in rows]
    bodies = [str(row.get("body") or "") for row in rows]
    model.to(device)
    model.eval()
    probabilities: list[float] = []
    max_length = int(bundle.get("max_length") or 384)
    padding_strategy: bool | str = "max_length" if device == "mps" else True
    with torch_module.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_titles = titles[start : start + batch_size]
            batch_bodies = bodies[start : start + batch_size]
            encoded = tokenizer(
                batch_titles,
                batch_bodies,
                truncation=True,
                max_length=max_length,
                padding=padding_strategy,
                return_tensors="pt",
            )
            encoded = _move_token_batch_to_device(encoded, device=device, torch_module=torch_module)
            outputs = model(**encoded)
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            probabilities.extend(_positive_scores_from_logits(logits.detach().cpu().numpy()))
    return probabilities


def _should_retry_transformer_inference_on_cpu(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    message = str(exc).lower()
    retry_signals = (
        "mps",
        "metal",
        "placeholder storage",
        "not implemented",
        "out of memory",
        "invalid buffer",
    )
    return any(signal in message for signal in retry_signals)


def _clear_torch_inference_memory(torch_module: Any) -> None:
    gc.collect()
    cuda_backend = getattr(torch_module, "cuda", None)
    if cuda_backend is not None and hasattr(cuda_backend, "is_available") and cuda_backend.is_available():
        empty_cache = getattr(cuda_backend, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and hasattr(mps_backend, "is_available") and mps_backend.is_available():
        mps_module = getattr(torch_module, "mps", None)
        empty_cache = getattr(mps_module, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _transformer_inference_batch_size(bundle: dict[str, Any]) -> int:
    training_args = bundle.get("training_args")
    explicit_value = bundle.get("inference_batch_size")
    raw_value = explicit_value or bundle.get("per_device_eval_batch_size") or (
        training_args.get("per_device_eval_batch_size")
        if isinstance(training_args, dict)
        else None
    )
    try:
        if explicit_value is not None:
            return max(1, int(raw_value))
        return max(8, int(raw_value or 8))
    except (TypeError, ValueError):
        return 8


def _causal_lm_positive_probabilities(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "Causal-LM inference requires torch. Install optional model dependencies with "
            "`python -m pip install -e \".[dev,models]\"`."
        ) from exc

    model = bundle.get("model")
    tokenizer = bundle.get("tokenizer")
    if model is None or tokenizer is None:
        raise ValueError("Causal-LM bundle is missing model or tokenizer")

    device = _bundle_runtime_device(bundle, torch)
    model.to(device)
    model.eval()
    prompt_template_version = str(
        bundle.get("prompt_template_version")
        or DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION
    )
    prompts = [
        causal_lm_prompt_for_row(row, prompt_template_version=prompt_template_version)
        for row in rows
    ]
    ask_scores = _causal_lm_completion_scores(model, tokenizer, prompts, completion=" askseattle", device=device)
    not_scores = _causal_lm_completion_scores(model, tokenizer, prompts, completion=" not_askseattle", device=device)
    probabilities: list[float] = []
    for ask_score, not_score in zip(ask_scores, not_scores, strict=True):
        probabilities.append(_safe_binary_completion_probability(ask_score, not_score))
    return probabilities


def _stacked_transformer_positive_probabilities(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    classifier = bundle.get("model")
    components = list(bundle.get("component_models") or [])
    if classifier is None or not components:
        raise ValueError("Stacked transformer decider bundle is missing classifier or component models")
    component_names = [str(component.get("name") or "") for component in components if str(component.get("name") or "")]
    if not component_names:
        raise ValueError("Stacked transformer decider bundle is missing component model names")
    component_scores_by_name: dict[str, list[float]] = {}
    for component in components:
        name = str(component.get("name") or "").strip()
        component_bundle = component.get("bundle")
        if not name or not isinstance(component_bundle, dict):
            raise ValueError("Stacked transformer decider component bundle is incomplete")
        component_scores_by_name[name] = score_rows(component_bundle, rows)
    feature_matrix = stacked_transformer_feature_matrix(
        rows,
        component_scores_by_name,
        component_names=component_names,
    )
    probabilities = classifier.predict_proba(feature_matrix)
    positive_index = list(classifier.classes_).index(1)
    return [float(row[positive_index]) for row in probabilities]


def confidence_band_for_score(score: float, *, low_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= low_threshold:
        return "borderline"
    return "low"


def _effective_high_threshold_for_row(
    row: dict[str, Any] | None,
    *,
    high_threshold: float,
) -> float:
    if not isinstance(row, dict):
        return high_threshold
    adjusted = float(high_threshold)
    if str(row.get("is_low_text") or "").strip().lower() == "yes":
        adjusted += DEFAULT_LOW_TEXT_HIGH_THRESHOLD_DELTA
    if str(row.get("post_type") or "").strip().lower() == "image":
        adjusted += DEFAULT_IMAGE_HIGH_THRESHOLD_DELTA
    if bool(row.get("is_sparse_media")):
        adjusted += DEFAULT_SPARSE_MEDIA_HIGH_THRESHOLD_DELTA
    return min(adjusted, 0.99)


def confidence_band_for_row(
    row: dict[str, Any] | None,
    score: float,
    *,
    low_threshold: float,
    high_threshold: float,
) -> str:
    return confidence_band_for_score(
        score,
        low_threshold=low_threshold,
        high_threshold=_effective_high_threshold_for_row(row, high_threshold=high_threshold),
    )


def effective_high_threshold_for_row(
    row: dict[str, Any] | None,
    *,
    high_threshold: float,
) -> float:
    return _effective_high_threshold_for_row(row, high_threshold=high_threshold)


def check_result_from_score(
    bundle: dict[str, Any],
    *,
    row: dict[str, Any],
    score: float,
    score_raw: float | None = None,
    post_id: str | None = None,
    permalink: str | None = None,
    time_source: str | None = None,
) -> CheckResult:
    high_threshold = float(bundle.get("high_threshold") or bundle.get("threshold") or 0.85)
    low_threshold = float(bundle.get("low_threshold") or high_threshold)
    low_threshold = min(low_threshold, high_threshold)
    calibrated_score = float(score)
    raw_score = float(score_raw if score_raw is not None else score)
    effective_high_threshold = effective_high_threshold_for_row(row, high_threshold=high_threshold)
    label = "askseattle" if calibrated_score >= low_threshold else "not_askseattle"
    return CheckResult(
        post_id=post_id,
        permalink=permalink,
        model_name=str(bundle.get("model_name") or bundle.get("model_type") or "unknown"),
        display_name=str(bundle.get("display_name") or bundle.get("model_name") or bundle.get("model_type") or "unknown"),
        model_version=str(bundle.get("model_version") or bundle.get("version") or "unknown"),
        low_threshold=low_threshold,
        high_threshold=effective_high_threshold,
        score=calibrated_score,
        score_raw=raw_score,
        score_calibrated=calibrated_score,
        label=label,
        confidence_band=confidence_band_for_row(
            row,
            calibrated_score,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        ),
        time_source=time_source,
        created_at=datetime.now(tz=UTC).isoformat(),
    )


def classify_post(
    bundle: dict[str, Any],
    *,
    title: str,
    selftext: str = "",
    post_type: str | None = None,
    content_domain: str | None = None,
    is_crosspost: bool | None = None,
    post_id: str | None = None,
    permalink: str | None = None,
    time_source: str | None = None,
) -> CheckResult:
    representation_config = _bundle_representation_config(bundle)
    row = build_inference_row(
        title=title,
        selftext=selftext,
        post_type=post_type,
        content_domain=content_domain,
        is_crosspost=is_crosspost,
        include_sparse_media_token=representation_config["include_sparse_media_token"],
        include_image_low_text_tokens=representation_config["include_image_low_text_tokens"],
    )

    raw_score = raw_score_rows(bundle, [row])[0]
    calibrated_score = score_rows(bundle, [row])[0]
    return check_result_from_score(
        bundle,
        row=row,
        score=calibrated_score,
        score_raw=raw_score,
        post_id=post_id,
        permalink=permalink,
        time_source=time_source,
    )


def _validate_posts(posts: list[LabeledPost]) -> None:
    if not posts:
        raise ValueError("Training data is empty")
    labels = set(_labels(posts))
    if labels != {0, 1}:
        raise ValueError("Training data must include both askseattle and not_askseattle examples")


def _rows(
    posts: list[LabeledPost],
    *,
    include_sparse_media_token: bool = DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    include_image_low_text_tokens: bool = DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
) -> list[dict[str, str]]:
    return [
        build_inference_row(
            title=post.title,
            selftext=post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
            include_sparse_media_token=include_sparse_media_token,
            include_image_low_text_tokens=include_image_low_text_tokens,
        )
        for post in posts
    ]


def _labels(posts: list[LabeledPost]) -> list[int]:
    return [post.label for post in posts]


def build_inference_row(
    *,
    title: str,
    selftext: str = "",
    post_type: str | None = None,
    content_domain: str | None = None,
    is_crosspost: bool | None = None,
    include_sparse_media_token: bool = DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    include_image_low_text_tokens: bool = DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
) -> dict[str, str]:
    body = normalize_body(selftext)
    normalized_title = str(title).strip()
    normalized_body = body.strip()
    title_lexical = normalize_urls_for_lexical_text(normalized_title)
    body_lexical = normalize_urls_for_lexical_text(normalized_body)
    title_lexical_stripped = normalize_urls_for_lexical_text(normalized_title, replacement="")
    body_lexical_stripped = normalize_urls_for_lexical_text(normalized_body, replacement="")
    metadata = post_metadata_text(
        title=normalized_title,
        selftext=body,
        post_type=post_type,
        content_domain=content_domain,
        is_crosspost=is_crosspost,
        include_sparse_media_token=include_sparse_media_token,
        include_image_low_text_tokens=include_image_low_text_tokens,
    )
    sparse_media = is_sparse_media_post(post_type=post_type, selftext=body)
    crosspost_value = "unknown"
    if is_crosspost is True:
        crosspost_value = "yes"
    elif is_crosspost is False:
        crosspost_value = "no"
    raw_text = "\n".join(part for part in (normalized_title, normalized_body) if part).strip()
    lexical_text = "\n".join(part for part in (title_lexical, body_lexical) if part).strip()
    lexical_text_stripped = "\n".join(
        part for part in (title_lexical_stripped, body_lexical_stripped) if part
    ).strip()
    return {
        "title": normalized_title,
        "title_lexical": title_lexical,
        "title_lexical_stripped": title_lexical_stripped,
        "body": "\n".join(part for part in (metadata, normalized_body) if part).strip(),
        "body_raw": normalized_body,
        "body_lexical": body_lexical,
        "body_lexical_stripped": body_lexical_stripped,
        "metadata_text": metadata,
        "text_raw": raw_text,
        "text_lexical": lexical_text,
        "text_lexical_stripped": lexical_text_stripped,
        "text": post_text(
            normalized_title,
            body,
            post_type=post_type,
            content_domain=content_domain,
            is_crosspost=is_crosspost,
            include_sparse_media_token=include_sparse_media_token,
            include_image_low_text_tokens=include_image_low_text_tokens,
        ),
        "post_type": str(post_type or "").strip(),
        "content_domain": str(content_domain or "").strip(),
        "title_length_bucket": title_length_bucket(title),
        "body_length_bucket": body_length_bucket(body),
        "has_body": "yes" if body.strip() else "no",
        "has_question_mark": "yes" if has_question_mark(title, body) else "no",
        "is_low_text": "yes" if is_low_text_body(body) else "no",
        "is_crosspost": crosspost_value,
        "is_sparse_media": sparse_media,
    }


def _time_split(
    posts: list[LabeledPost],
    *,
    calibration_size: float,
    test_size: float,
    evaluation_subreddit: str | None = None,
) -> DatasetSplit:
    eligible_posts = [
        post for post in posts if post.time_key is not None or post.created_utc is not None
    ]
    if len(eligible_posts) < 3:
        raise ValueError("Need at least 3 dated examples for time-based train/calibration/test splits")

    normalized_subreddit = _canonical_subreddit_name(evaluation_subreddit)
    if normalized_subreddit:
        return _time_split_for_evaluation_subreddit(
            posts,
            eligible_posts=eligible_posts,
            calibration_size=calibration_size,
            test_size=test_size,
            evaluation_subreddit=normalized_subreddit,
        )

    ordered_posts = sorted(
        eligible_posts,
        key=_post_sort_key,
    )

    test_count = max(1, ceil(len(ordered_posts) * test_size))
    calibration_count = max(1, ceil(len(ordered_posts) * calibration_size))
    train_count = len(ordered_posts) - calibration_count - test_count
    minimum_train_count = _minimum_train_count_with_both_classes(ordered_posts)

    while train_count < minimum_train_count and (calibration_count > 0 or test_count > 0):
        if calibration_count >= test_count and calibration_count > 0:
            calibration_count -= 1
        elif test_count > 0:
            test_count -= 1
        train_count = len(ordered_posts) - calibration_count - test_count

    if train_count < minimum_train_count:
        raise ValueError("Not enough dated examples to keep both labels in the chronological train split")

    train_posts = ordered_posts[:train_count]
    calibration_posts = ordered_posts[train_count : train_count + calibration_count]
    test_posts = ordered_posts[train_count + calibration_count :]
    return DatasetSplit(
        train=train_posts,
        calibration=calibration_posts,
        test=test_posts,
        split_strategy="time",
        split_seed=None,
        excluded_for_time_split=len(posts) - len(ordered_posts),
        time_coverage={
            "train": _time_coverage(train_posts),
            "calibration": _time_coverage(calibration_posts),
            "test": _time_coverage(test_posts),
        },
    )


def _time_split_for_evaluation_subreddit(
    posts: list[LabeledPost],
    *,
    eligible_posts: list[LabeledPost],
    calibration_size: float,
    test_size: float,
    evaluation_subreddit: str,
) -> DatasetSplit:
    ordered_posts = sorted(eligible_posts, key=_post_sort_key)
    evaluation_posts = [
        post for post in ordered_posts if _canonical_subreddit_name(post.subreddit) == evaluation_subreddit
    ]
    if len(evaluation_posts) < 3:
        raise ValueError(
            f"Need at least 3 dated examples in subreddit {evaluation_subreddit!r} "
            "for time-based train/calibration/test splits"
        )

    test_count = max(1, ceil(len(evaluation_posts) * test_size))
    calibration_count = max(1, ceil(len(evaluation_posts) * calibration_size))

    while True:
        evaluation_train_count = len(evaluation_posts) - calibration_count - test_count
        if evaluation_train_count < 0:
            calibration_count, test_count = _shrink_later_split_counts(calibration_count, test_count)
            continue

        calibration_posts = evaluation_posts[
            evaluation_train_count : evaluation_train_count + calibration_count
        ]
        test_posts = evaluation_posts[evaluation_train_count + calibration_count :]
        first_holdout = calibration_posts[0] if calibration_posts else (test_posts[0] if test_posts else None)
        if first_holdout is None:
            raise ValueError(
                f"Not enough dated examples in subreddit {evaluation_subreddit!r} "
                "to build chronological evaluation slices"
            )

        train_cutoff = _post_sort_key(first_holdout)
        train_posts = [post for post in ordered_posts if _post_sort_key(post) < train_cutoff]
        if {post.label for post in train_posts} == {0, 1}:
            return DatasetSplit(
                train=train_posts,
                calibration=calibration_posts,
                test=test_posts,
                split_strategy="time_eval_subreddit",
                split_seed=None,
                excluded_for_time_split=len(posts) - len(ordered_posts),
                time_coverage={
                    "train": _time_coverage(train_posts),
                    "calibration": _time_coverage(calibration_posts),
                    "test": _time_coverage(test_posts),
                },
                evaluation_subreddit=evaluation_subreddit,
            )

        if calibration_count == 0 and test_count == 0:
            break
        calibration_count, test_count = _shrink_later_split_counts(calibration_count, test_count)

    raise ValueError(
        f"Not enough dated examples before the {evaluation_subreddit!r} evaluation window "
        "to keep both labels in the chronological train split"
    )


def _random_split(
    posts: list[LabeledPost],
    *,
    calibration_size: float,
    test_size: float,
    split_seed: int,
    evaluation_subreddit: str | None = None,
) -> DatasetSplit:
    ordered_posts = sorted(posts, key=_post_sort_key)
    normalized_subreddit = _canonical_subreddit_name(evaluation_subreddit)
    evaluation_posts = (
        ordered_posts
        if normalized_subreddit is None
        else [post for post in ordered_posts if _canonical_subreddit_name(post.subreddit) == normalized_subreddit]
    )
    if len(evaluation_posts) < 3:
        if normalized_subreddit is None:
            raise ValueError("Need at least 3 examples for random train/calibration/test splits")
        raise ValueError(
            f"Need at least 3 examples in subreddit {normalized_subreddit!r} "
            "for random train/calibration/test splits"
        )

    calibration_count = max(1, ceil(len(evaluation_posts) * calibration_size))
    test_count = max(1, ceil(len(evaluation_posts) * test_size))
    calibration_count, test_count = _fit_random_holdout_counts(
        evaluation_posts,
        calibration_count=calibration_count,
        test_count=test_count,
    )
    if calibration_count + test_count <= 0:
        raise ValueError("Not enough examples to build random calibration/test splits")

    calibration_posts, test_posts = _random_holdout_split(
        evaluation_posts,
        calibration_count=calibration_count,
        test_count=test_count,
        split_seed=split_seed,
    )
    holdout_posts = set(calibration_posts) | set(test_posts)
    train_posts = [post for post in ordered_posts if post not in holdout_posts]
    if {post.label for post in train_posts} != {0, 1}:
        raise ValueError("Random split must leave both labels in the training split")

    return DatasetSplit(
        train=train_posts,
        calibration=calibration_posts,
        test=test_posts,
        split_strategy="random_eval_subreddit" if normalized_subreddit else "random",
        split_seed=split_seed,
        excluded_for_time_split=0,
        time_coverage=None,
        evaluation_subreddit=normalized_subreddit,
    )


def _time_coverage(posts: list[LabeledPost]) -> dict[str, Any]:
    if not posts:
        return {"count": 0, "first_time_key": None, "last_time_key": None, "first_at": None, "last_at": None}

    time_keys = [
        float(post.time_key if post.time_key is not None else post.created_utc or 0)
        for post in posts
        if post.time_key is not None or post.created_utc is not None
    ]
    if not time_keys:
        return {"count": len(posts), "first_time_key": None, "last_time_key": None, "first_at": None, "last_at": None}

    return {
        "count": len(posts),
        "first_time_key": time_keys[0],
        "last_time_key": time_keys[-1],
        "first_at": _timestamp_to_iso(time_keys[0]),
        "last_at": _timestamp_to_iso(time_keys[-1]),
    }


def _timestamp_to_iso(timestamp: float | int | None) -> str | None:
    if timestamp in (None, ""):
        return None
    return datetime.fromtimestamp(float(timestamp), tz=UTC).isoformat()


def _post_sort_key(post: LabeledPost) -> tuple[float, str, str, str]:
    return (
        float(post.time_key if post.time_key is not None else post.created_utc or 0),
        post.post_id or "",
        post.permalink or "",
        post.text_hash or "",
    )


def _resolve_thresholds(
    probabilities: list[float],
    thresholds: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if thresholds is not None:
        return thresholds

    derived = {0.0, 1.0, *DEFAULT_THRESHOLD_GRID}
    derived.update(max(0.0, min(1.0, round(float(probability), 6))) for probability in probabilities)
    return tuple(sorted(derived))


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(sum(y_pred)),
        "support": Counter(y_true)[1],
    }


def _feature_names(features: FeatureUnion) -> list[str]:
    names: list[str] = []
    for branch_name, transformer in features.transformer_list:
        if isinstance(transformer, Pipeline):
            vectorizer = transformer.named_steps["vectorizer"]
            branch_names = vectorizer.get_feature_names_out()
        else:
            branch_names = transformer.get_feature_names_out()
        names.extend(f"{branch_name}:{feature_name}" for feature_name in branch_names)
    return names


def _configured_word_stopwords(features: FeatureUnion) -> list[str]:
    configured: set[str] = set()
    for channel, transformer in features.transformer_list:
        if channel not in {"title_word", "body_word"}:
            continue
        if not isinstance(transformer, Pipeline):
            continue
        vectorizer = transformer.named_steps["vectorizer"]
        configured.update(str(word) for word in vectorizer.get_stop_words() or ())
    return sorted(configured)


def _canonical_subreddit_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().lstrip("/")
    if normalized.startswith("r/"):
        normalized = normalized[2:]
    return normalized or None


def _shrink_later_split_counts(calibration_count: int, test_count: int) -> tuple[int, int]:
    if calibration_count >= test_count and calibration_count > 0:
        return calibration_count - 1, test_count
    if test_count > 0:
        return calibration_count, test_count - 1
    return calibration_count, test_count


def _fit_random_holdout_counts(
    posts: list[LabeledPost],
    *,
    calibration_count: int,
    test_count: int,
) -> tuple[int, int]:
    max_holdout_capacity = len(posts) - len({post.label for post in posts})
    while calibration_count + test_count > max_holdout_capacity and (calibration_count > 0 or test_count > 0):
        calibration_count, test_count = _shrink_later_split_counts(calibration_count, test_count)
    return calibration_count, test_count


def _random_holdout_split(
    posts: list[LabeledPost],
    *,
    calibration_count: int,
    test_count: int,
    split_seed: int,
) -> tuple[list[LabeledPost], list[LabeledPost]]:
    posts_by_label = _shuffled_posts_by_label(posts, seed=split_seed)
    holdout_count = calibration_count + test_count
    holdout_allocations = _allocate_label_counts(
        {label: len(bucket) for label, bucket in posts_by_label.items()},
        target_count=holdout_count,
        reserve_one_for_remaining=True,
    )
    holdout_posts_by_label = {
        label: list(bucket[: holdout_allocations.get(label, 0)])
        for label, bucket in posts_by_label.items()
    }
    test_allocations = _allocate_label_counts(
        {label: len(bucket) for label, bucket in holdout_posts_by_label.items()},
        target_count=test_count,
        reserve_one_for_remaining=False,
    )
    test_posts: list[LabeledPost] = []
    calibration_posts: list[LabeledPost] = []
    for label, bucket in holdout_posts_by_label.items():
        label_test_count = test_allocations.get(label, 0)
        test_posts.extend(bucket[:label_test_count])
        calibration_posts.extend(bucket[label_test_count:])

    calibration_rng = np.random.default_rng(split_seed + 1)
    test_rng = np.random.default_rng(split_seed + 2)
    calibration_rng.shuffle(calibration_posts)
    test_rng.shuffle(test_posts)
    return calibration_posts, test_posts


def _shuffled_posts_by_label(posts: list[LabeledPost], *, seed: int) -> dict[int, list[LabeledPost]]:
    grouped: dict[int, list[LabeledPost]] = {}
    for post in posts:
        grouped.setdefault(post.label, []).append(post)
    for label, bucket in grouped.items():
        rng = np.random.default_rng(seed + label)
        rng.shuffle(bucket)
    return grouped


def _allocate_label_counts(
    label_counts: dict[int, int],
    *,
    target_count: int,
    reserve_one_for_remaining: bool,
) -> dict[int, int]:
    capacities = {
        label: max(count - 1, 0) if reserve_one_for_remaining else count
        for label, count in label_counts.items()
    }
    allocations = {label: 0 for label in label_counts}
    target_count = min(target_count, sum(capacities.values()))
    if target_count <= 0:
        return allocations

    eligible = [label for label, capacity in capacities.items() if capacity > 0]
    if target_count >= len(eligible):
        for label in eligible:
            allocations[label] += 1
            capacities[label] -= 1
        target_count -= len(eligible)

    if target_count <= 0:
        return allocations

    total_capacity = sum(capacities.values())
    if total_capacity <= 0:
        return allocations

    provisional: dict[int, float] = {
        label: target_count * (capacities[label] / total_capacity)
        for label in label_counts
    }
    for label in label_counts:
        addition = min(int(provisional[label]), capacities[label])
        allocations[label] += addition
        capacities[label] -= addition
        target_count -= addition

    while target_count > 0:
        candidates = [
            label
            for label, capacity in capacities.items()
            if capacity > 0
        ]
        if not candidates:
            break
        label = max(
            candidates,
            key=lambda item: (
                provisional[item] - int(provisional[item]),
                capacities[item],
                -item,
            ),
        )
        allocations[label] += 1
        capacities[label] -= 1
        target_count -= 1

    return allocations


def _feature_records(features: FeatureUnion, coefficients: list[float] | Any) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for full_feature, weight in zip(_feature_names(features), coefficients, strict=True):
        channel, feature = full_feature.split(":", 1)
        records.append(
            {
                "channel": channel,
                "feature": feature,
                "full_feature": full_feature,
                "weight": round(float(weight), 6),
            }
        )
    return records


def _rank_feature_records(
    records: list[dict[str, float | str]],
    *,
    limit: int,
    reverse: bool,
) -> list[dict[str, float | str]]:
    return sorted(records, key=lambda record: float(record["weight"]), reverse=reverse)[:limit]


def _default_min_df(posts: list[LabeledPost]) -> int:
    if len(posts) < 50:
        return 1
    if len(posts) < 500:
        return 2
    if len(posts) < 2_000:
        return 3
    return 5


def _minimum_train_count_with_both_classes(posts: list[LabeledPost]) -> int:
    labels_seen: set[int] = set()
    for index, post in enumerate(posts, start=1):
        labels_seen.add(post.label)
        if labels_seen == {0, 1}:
            return index
    raise ValueError("Dated examples must include both askseattle and not_askseattle labels")


def _normalize_tfidf_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    bundle.setdefault("model_family", "tfidf")
    bundle.setdefault("model_type", "tfidf")
    bundle.setdefault("model_name", "tfidf_logreg")
    bundle.setdefault("model_version", str(bundle.get("version") or __version__))
    bundle.setdefault("tfidf_config_version", DEFAULT_TFIDF_CONFIG_VERSION)
    _apply_threshold_policy_defaults(bundle)
    _apply_representation_defaults(bundle)
    return bundle


def _load_semantic_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(bundle)
    backend = str(bundle.get("backend") or "sentence_transformers")
    normalized["backend"] = backend
    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError(
                "Semantic embedding inference requires sentence-transformers. Install optional model "
                "dependencies with `python -m pip install -e \".[dev,models]\"`."
            ) from exc
        normalized["encoder"] = SentenceTransformer(str(bundle["model_id"]))
    elif backend == "hf_embedding":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ValueError(
                "Transformer-backed embedding inference requires transformers. Install optional model "
                "dependencies with `python -m pip install -e \".[dev,models]\"`."
            ) from exc
        normalized["tokenizer"] = AutoTokenizer.from_pretrained(str(bundle["model_id"]), use_fast=False)
        if normalized["tokenizer"].pad_token is None and normalized["tokenizer"].eos_token is not None:
            normalized["tokenizer"].pad_token = normalized["tokenizer"].eos_token
        normalized["encoder_model"] = AutoModel.from_pretrained(str(bundle["model_id"]))
    else:
        raise ValueError(f"Unsupported semantic embedding backend: {backend}")
    normalized.setdefault("model_family", "semantic_embedding")
    normalized.setdefault("model_name", "semantic_embedding_logreg")
    normalized.setdefault("model_version", str(bundle.get("version") or __version__))
    _apply_threshold_policy_defaults(normalized)
    _apply_representation_defaults(normalized)
    return normalized


def _load_transformer_bundle_from_joblib(bundle: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    model_dir = _resolve_bundle_model_dir(bundle, source_path=source_path, family_label="transformer")
    if model_dir is None:
        raise ValueError(f"{source_path} is missing transformer artifact metadata")
    normalized = dict(bundle)
    normalized["artifact_path"] = str(model_dir)
    normalized["model_dir"] = str(model_dir)
    return _load_transformer_runtime_bundle(normalized, model_dir=model_dir)


def _load_transformer_bundle(model_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metadata_path = model_dir.parent / "transformer_bundle.joblib"
    if metadata_path.exists():
        loaded = joblib.load(metadata_path)
        if isinstance(loaded, dict):
            metadata = dict(loaded)
    elif (model_dir.parent / "transformer_metadata.json").exists():
        metadata = json.loads((model_dir.parent / "transformer_metadata.json").read_text(encoding="utf-8"))

    metadata.setdefault("artifact_path", str(model_dir))
    metadata.setdefault("model_family", "transformer_sequence_classifier")
    metadata.setdefault("model_name", "transformer_sequence_classifier")
    metadata.setdefault("model_version", str(metadata.get("version") or __version__))
    return _load_transformer_runtime_bundle(metadata, model_dir=model_dir)


def _load_transformer_runtime_bundle(metadata: dict[str, Any], *, model_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "Transformer inference requires transformers. Install optional model dependencies with "
            "`python -m pip install -e \".[dev,models]\"`."
        ) from exc

    normalized = dict(metadata)
    load_options = dict(
        normalized.get("load_options")
        or transformer_load_options(str(normalized.get("model_id") or model_dir))
    )
    trust_remote_code = bool(load_options.get("trust_remote_code"))
    ensure_transformer_custom_code_support(trust_remote_code=trust_remote_code)
    normalized["tokenizer"] = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False, **load_options)
    normalized["model"] = AutoModelForSequenceClassification.from_pretrained(str(model_dir), **load_options)
    normalized.setdefault("model_family", "transformer_sequence_classifier")
    normalized.setdefault("model_name", "transformer_sequence_classifier")
    normalized.setdefault("model_version", str(normalized.get("version") or __version__))
    normalized.setdefault("max_length", _transformer_max_length(model_dir))
    _apply_threshold_policy_defaults(normalized)
    _apply_representation_defaults(normalized)
    return normalized


def _load_causal_lm_bundle_from_joblib(bundle: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    model_dir = _resolve_bundle_model_dir(bundle, source_path=source_path, family_label="causal-lm")
    if model_dir is None:
        raise ValueError(f"{source_path} is missing causal-lm artifact metadata")
    normalized = dict(bundle)
    normalized["artifact_path"] = str(model_dir)
    normalized["model_dir"] = str(model_dir)
    return _load_causal_lm_runtime_bundle(normalized, model_dir=model_dir)


def _load_stacked_transformer_bundle_from_joblib(bundle: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    normalized = dict(bundle)
    components = []
    for component in bundle.get("component_models") or []:
        if not isinstance(component, dict):
            continue
        artifact_reference = component.get("artifact_path")
        if not artifact_reference:
            raise ValueError(f"{source_path} is missing stacked decider component artifact metadata")
        resolved_artifact = _resolve_bundle_reference_path(
            artifact_reference,
            source_path=source_path,
            label="stacked-transformer component",
        )
        loaded_bundle = load_model(resolved_artifact)
        if str(loaded_bundle.get("model_family") or "") != "transformer_sequence_classifier":
            raise ValueError(
                f"{source_path} component {resolved_artifact} is not a transformer_sequence_classifier bundle"
            )
        components.append(
            {
                "name": str(component.get("name") or loaded_bundle.get("model_name") or resolved_artifact.stem),
                "display_name": component.get("display_name") or loaded_bundle.get("display_name"),
                "model_id": component.get("model_id") or loaded_bundle.get("model_id"),
                "artifact_path": str(resolved_artifact),
                "bundle": loaded_bundle,
            }
        )
    normalized["component_models"] = components
    normalized.setdefault("model_family", STACKED_TRANSFORMER_DECIDER_MODEL_FAMILY)
    normalized.setdefault("model_name", STACKED_TRANSFORMER_DECIDER_MODEL_FAMILY)
    normalized.setdefault("model_version", str(normalized.get("version") or __version__))
    _apply_threshold_policy_defaults(normalized)
    _apply_representation_defaults(normalized)
    return normalized


def _load_hybrid_policy_bundle_from_joblib(bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(bundle)
    normalized.setdefault("model_family", "hybrid_decider_policy")
    normalized.setdefault("model_name", "hybrid_consensus_policy")
    normalized.setdefault("display_name", "Hybrid consensus policy")
    normalized.setdefault("model_version", str(normalized.get("version") or __version__))
    _apply_threshold_policy_defaults(normalized)
    _apply_representation_defaults(normalized)
    return normalized


def _resolve_bundle_model_dir(
    bundle: dict[str, Any],
    *,
    source_path: Path,
    family_label: str,
) -> Path | None:
    raw_model_dir = bundle.get("artifact_path") or bundle.get("model_dir")
    if not raw_model_dir:
        return None
    candidate = Path(str(raw_model_dir))
    if not candidate.is_absolute():
        resolved = (source_path.parent / candidate).resolve()
        if resolved.exists():
            return resolved
        raise ValueError(
            f"{source_path} references missing {family_label} artifact directory {resolved}"
        )
    if candidate.exists():
        return candidate
    sibling = (source_path.parent / candidate.name).resolve()
    if sibling.exists():
        LOGGER.info(
            "rebasing stale absolute %s artifact path bundle=%s from=%s to=%s",
            family_label,
            str(source_path),
            str(candidate),
            str(sibling),
        )
        return sibling
    raise ValueError(
        f"{source_path} references missing {family_label} artifact directory {candidate}"
    )


def _resolve_bundle_reference_path(reference: Any, *, source_path: Path, label: str) -> Path:
    candidate = Path(str(reference))
    if not candidate.is_absolute():
        resolved = (source_path.parent / candidate).resolve()
        if resolved.exists():
            return resolved
        raise ValueError(f"{source_path} references missing {label} {resolved}")
    if candidate.exists():
        return candidate
    sibling = (source_path.parent / candidate.name).resolve()
    if sibling.exists():
        LOGGER.info(
            "rebasing stale absolute %s path bundle=%s from=%s to=%s",
            label,
            str(source_path),
            str(candidate),
            str(sibling),
        )
        return sibling
    raise ValueError(f"{source_path} references missing {label} {candidate}")


def _load_causal_lm_runtime_bundle(metadata: dict[str, Any], *, model_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "Causal-LM inference requires transformers. Install optional model dependencies with "
            "`python -m pip install -e \".[dev,models]\"`."
        ) from exc

    normalized = dict(metadata)
    normalized["tokenizer"] = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    if normalized["tokenizer"].pad_token is None:
        normalized["tokenizer"].pad_token = normalized["tokenizer"].eos_token
    normalized["model"] = AutoModelForCausalLM.from_pretrained(str(model_dir))
    normalized.setdefault("model_family", "causal_lm_classifier")
    normalized.setdefault("model_name", "causal_lm_classifier")
    normalized.setdefault("model_version", str(normalized.get("version") or __version__))
    _apply_threshold_policy_defaults(normalized)
    _apply_representation_defaults(normalized)
    return normalized


def _transformer_max_length(model_dir: Path) -> int:
    training_summary_path = model_dir.parent / "training_summary.json"
    if training_summary_path.exists():
        try:
            summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 384
        training_args = summary.get("training_args") or {}
        max_length = training_args.get("max_length")
        if isinstance(max_length, int) and max_length > 0:
            return max_length
    return 384


def _apply_threshold_policy_defaults(bundle: dict[str, Any]) -> None:
    threshold_policy = bundle.get("threshold_policy")
    legacy_policy = bundle.get("decision_policy")
    policy = threshold_policy if isinstance(threshold_policy, dict) else legacy_policy if isinstance(legacy_policy, dict) else {}
    high_threshold = float(
        bundle.get("high_threshold")
        or bundle.get("threshold")
        or policy.get("high_threshold")
        or 0.85
    )
    low_threshold = float(bundle.get("low_threshold") or policy.get("low_threshold") or high_threshold)
    bundle.setdefault("threshold", high_threshold)
    bundle.setdefault("high_threshold", high_threshold)
    bundle.setdefault("low_threshold", low_threshold)
    bundle["threshold_policy"] = {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "review_precision_target": policy.get("review_precision_target", DEFAULT_REVIEW_PRECISION_TARGET),
        "high_precision_target": policy.get("high_precision_target"),
        "minimum_high_confidence_calibration_predictions": policy.get(
            "minimum_high_confidence_calibration_predictions",
            DEFAULT_MIN_HIGH_CONFIDENCE_CALIBRATION_PREDICTIONS,
        ),
        "high_threshold_fallback_used": bool(policy.get("high_threshold_fallback_used", False)),
        "calibration_method": policy.get("calibration_method") or bundle.get("calibration_method"),
        "split_strategy": policy.get("split_strategy") or bundle.get("split_strategy") or "manual",
        "split_seed": _first_defined(policy.get("split_seed"), bundle.get("split_seed")),
        "evaluation_subreddit": policy.get("evaluation_subreddit") or bundle.get("evaluation_subreddit"),
        "time_coverage": policy.get("time_coverage") or bundle.get("time_coverage"),
    }


def _normalize_representation_config(value: dict[str, Any] | None) -> dict[str, bool]:
    payload = value if isinstance(value, dict) else {}
    return {
        "include_sparse_media_token": bool(
            payload.get("include_sparse_media_token", DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN)
        ),
        "include_image_low_text_tokens": bool(
            payload.get("include_image_low_text_tokens", DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS)
        ),
    }


def _apply_representation_defaults(bundle: dict[str, Any]) -> None:
    bundle["representation_config"] = _normalize_representation_config(bundle.get("representation_config"))


def _bundle_representation_config(bundle: dict[str, Any]) -> dict[str, bool]:
    _apply_representation_defaults(bundle)
    return dict(bundle["representation_config"])


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _semantic_runtime_texts(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    prompt_mode = str(bundle.get("prompt_mode") or "plain")
    prompt_prefix = str(bundle.get("prompt_prefix") or "")
    texts = [str(row.get("text") or "") for row in rows]
    if prompt_mode == "task_prefix" and prompt_prefix:
        return [f"{prompt_prefix}\n{text}" for text in texts]
    if prompt_mode == "short_task_prefix" and prompt_prefix:
        return [f"{prompt_prefix} {text}".strip() for text in texts]
    if prompt_mode == "document_prefix":
        document_prefix = prompt_prefix or "Document:"
        return [f"{document_prefix} {text}".strip() for text in texts]
    if prompt_mode == "jina_document_component":
        document_prefix = prompt_prefix or "Document:"
        return [f"{document_prefix} {text}".strip() for text in texts]
    return texts


def _semantic_runtime_component_texts(
    bundle: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    component: str,
) -> list[str]:
    prompt_mode = str(bundle.get("prompt_mode") or "plain")
    prompt_prefix = str(bundle.get("prompt_prefix") or "")
    short_prompt_prefix = str(bundle.get("short_prompt_prefix") or prompt_prefix)
    component_label = "Title" if component == "title" else "Body"
    raw_texts = [
        _semantic_runtime_component_fallback_text(
            str(row.get("title") or "").strip() if component == "title" else str(row.get("body_raw") or "").strip(),
            component=component,
        )
        for row in rows
    ]
    if prompt_mode == "plain":
        return raw_texts
    if prompt_mode == "task_prefix":
        return [
            f"{prompt_prefix}\n{component_label}: {text}" if prompt_prefix else f"{component_label}: {text}"
            for text in raw_texts
        ]
    if prompt_mode == "short_task_prefix":
        return [
            f"{short_prompt_prefix} {component_label}: {text}".strip()
            if short_prompt_prefix
            else f"{component_label}: {text}"
            for text in raw_texts
        ]
    if prompt_mode == "document_prefix":
        document_prefix = prompt_prefix or "Document:"
        return [f"{document_prefix} {text}".strip() for text in raw_texts]
    if prompt_mode == "jina_document_component":
        document_prefix = prompt_prefix or "Document:"
        return [f"{document_prefix} {component_label}: {text}".strip() for text in raw_texts]
    raise ValueError(f"Unsupported semantic prompt mode: {prompt_mode}")


def _semantic_runtime_component_fallback_text(text: str, *, component: str) -> str:
    normalized = str(text).strip()
    if normalized:
        return normalized
    if component == "title":
        return "[no title]"
    return "[no body]"


def _semantic_runtime_metadata_features(bundle: dict[str, Any], rows: list[dict[str, str]]) -> np.ndarray:
    vectorizer = bundle.get("metadata_vectorizer")
    if vectorizer is None:
        raise ValueError("Semantic embedding bundle is missing metadata_vectorizer")
    matrix = vectorizer.transform([str(row.get("metadata_text") or "") for row in rows])
    return np.asarray(matrix.toarray(), dtype=np.float32)


def _hf_embedding_runtime_embeddings(bundle: dict[str, Any], texts: list[str]) -> np.ndarray:
    import torch

    tokenizer = bundle.get("tokenizer")
    model = bundle.get("encoder_model")
    if tokenizer is None or model is None:
        raise ValueError("Transformer-backed semantic bundle is missing tokenizer or encoder model")
    device = _bundle_runtime_device(bundle, torch)
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    batch_size = int(bundle.get("encode_batch_size") or 8)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = _move_token_batch_to_device(batch, device=device, torch_module=torch)
            result = model(**batch)
            hidden = result.last_hidden_state
            pooling = str(bundle.get("pooling") or "mean")
            if pooling == "last_token":
                pooled = _last_token_pool(hidden, batch["attention_mask"])
            else:
                pooled = _mean_pool(hidden, batch["attention_mask"])
            outputs.append(_tensor_to_float32_numpy(pooled))
    embeddings = np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)
    if bool(bundle.get("normalize_embeddings")) and embeddings.size:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
    return embeddings.astype(np.float32)


def _mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * expanded).sum(dim=1)
    counts = expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _last_token_pool(hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    token_counts = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, token_counts]


def _tensor_to_float32_numpy(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32)


def _move_token_batch_to_device(
    batch: dict[str, Any],
    *,
    device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    integer_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if key in integer_keys:
            moved[key] = value.to(device=device, dtype=torch_module.long)
        else:
            moved[key] = value.to(device=device)
    return moved


def causal_lm_prompt_for_row(
    row: dict[str, Any],
    *,
    prompt_template_version: str = DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
) -> str:
    if prompt_template_version == "v1_binary_label_completion":
        return _causal_lm_prompt_v1(str(row.get("text") or ""))
    if prompt_template_version == "v2_contextual_fields":
        return _causal_lm_prompt_v2(row)
    if prompt_template_version == "v3_compact_contextual_fields":
        return _causal_lm_prompt_v3(row)
    if prompt_template_version == "v4_image_low_text":
        return _causal_lm_prompt_v4(row)
    raise ValueError(f"Unsupported causal-LM prompt template version: {prompt_template_version}")


def _causal_lm_prompt_v1(text: str) -> str:
    return (
        "Classify the following Reddit post as askseattle or not_askseattle.\n"
        "Respond with exactly one label.\n\n"
        f"{text}\n\n"
        "Label:"
    )


def _causal_lm_prompt_v2(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip() or "(none)"
    body = str(row.get("body_raw") or "").strip() or "(none)"
    post_type = str(row.get("post_type") or "").strip() or "unknown"
    content_domain = str(row.get("content_domain") or "").strip() or "unknown"
    has_body = str(row.get("has_body") or "unknown")
    has_question = str(row.get("has_question_mark") or "unknown")
    low_text = str(row.get("is_low_text") or "unknown")
    sparse_media = "yes" if row.get("is_sparse_media") else "no"
    crosspost = str(row.get("is_crosspost") or "unknown")
    title_length = str(row.get("title_length_bucket") or "unknown")
    body_length = str(row.get("body_length_bucket") or "unknown")

    return (
        "You are classifying a Reddit post for a binary moderation workflow.\n"
        "Return exactly one label: askseattle or not_askseattle.\n\n"
        "Use the title, body, and metadata together.\n"
        "Choose askseattle when the post is primarily asking for local help, recommendations, identification, "
        "explanations, or advice that fits an ask-style local question.\n"
        "Choose not_askseattle when the post is primarily news, discussion, opinion, promotion, media sharing, "
        "or another post that is not mainly asking for that kind of local help.\n"
        "Image or link posts can still be askseattle if the user is clearly asking what, where, who, why, "
        "or for recommendations or guidance.\n"
        "Do not use subreddit name.\n\n"
        f"Title: {title}\n"
        f"Body: {body}\n"
        f"Post type: {post_type}\n"
        f"Content domain: {content_domain}\n"
        f"Has body: {has_body}\n"
        f"Has question mark: {has_question}\n"
        f"Low text: {low_text}\n"
        f"Sparse media: {sparse_media}\n"
        f"Crosspost: {crosspost}\n"
        f"Title length: {title_length}\n"
        f"Body length: {body_length}\n\n"
        "Label:"
    )


def _causal_lm_prompt_v3(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip() or "(none)"
    body = str(row.get("body_raw") or "").strip() or "(none)"
    post_type = str(row.get("post_type") or "").strip() or "unknown"
    content_domain = str(row.get("content_domain") or "").strip() or "unknown"
    has_question = str(row.get("has_question_mark") or "unknown")
    low_text = str(row.get("is_low_text") or "unknown")
    crosspost = str(row.get("is_crosspost") or "unknown")

    return (
        "Classify this Reddit post for a binary moderation workflow.\n"
        "Return exactly one label: askseattle or not_askseattle.\n"
        "Use the title, body, and metadata together.\n"
        "Choose askseattle when the post is mainly asking for local help, recommendations, identification, explanation, or advice.\n"
        "Choose not_askseattle when the post is mainly news, discussion, opinion, promotion, or media sharing without that primary ask.\n"
        "Do not use subreddit name.\n\n"
        f"Title: {title}\n"
        f"Body: {body}\n"
        f"Post type: {post_type}\n"
        f"Content domain: {content_domain}\n"
        f"Has question mark: {has_question}\n"
        f"Low text: {low_text}\n"
        f"Crosspost: {crosspost}\n\n"
        "Label:"
    )


def _causal_lm_prompt_v4(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip() or "(none)"
    body = str(row.get("body_raw") or "").strip() or "(none)"
    post_type = str(row.get("post_type") or "").strip() or "unknown"
    content_domain = str(row.get("content_domain") or "").strip() or "unknown"
    has_body = str(row.get("has_body") or "unknown")
    has_question = str(row.get("has_question_mark") or "unknown")
    low_text = str(row.get("is_low_text") or "unknown")
    sparse_media = "yes" if row.get("is_sparse_media") else "no"
    crosspost = str(row.get("is_crosspost") or "unknown")

    return (
        "Classify this Reddit post for a binary moderation workflow.\n"
        "Return exactly one label: askseattle or not_askseattle.\n"
        "Use the title, body, and metadata together.\n"
        "Choose askseattle when the post is mainly asking for local help, recommendations, identification, explanation, or advice.\n"
        "Choose not_askseattle when the post is mainly news, discussion, opinion, promotion, or media sharing without that primary ask.\n"
        "Title-only image posts can still be askseattle when the title is clearly asking for local help, identification, explanation, or recommendations.\n"
        "Do not use subreddit name.\n\n"
        f"Title: {title}\n"
        f"Body: {body}\n"
        f"Post type: {post_type}\n"
        f"Content domain: {content_domain}\n"
        f"Has body: {has_body}\n"
        f"Has question mark: {has_question}\n"
        f"Low text: {low_text}\n"
        f"Sparse media: {sparse_media}\n"
        f"Crosspost: {crosspost}\n\n"
        "Label:"
    )


def _causal_lm_completion_scores(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    completion: str,
    *,
    device: str,
) -> list[float]:
    import torch

    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    scores: list[float] = []
    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
            logits = model(input_ids=input_ids).logits
            token_log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            total = 0.0
            start = len(prompt_ids) - 1
            for offset, token_id in enumerate(completion_ids):
                total += float(token_log_probs[0, start + offset, token_id].item())
            scores.append(total)
    return scores


def _safe_binary_completion_probability(positive_score: float, negative_score: float) -> float:
    positive_finite = np.isfinite(positive_score)
    negative_finite = np.isfinite(negative_score)

    if positive_finite and not negative_finite:
        return 1.0
    if negative_finite and not positive_finite:
        return 0.0
    if not positive_finite and not negative_finite:
        return 0.5

    stabilizer = max(positive_score, negative_score)
    positive_weight = float(np.exp(positive_score - stabilizer))
    negative_weight = float(np.exp(negative_score - stabilizer))
    denominator = positive_weight + negative_weight
    if not np.isfinite(denominator) or denominator <= 0:
        return 0.5
    return positive_weight / denominator


def _bundle_runtime_device(bundle: dict[str, Any], torch_module: Any) -> str:
    detected = _torch_runtime_device(torch_module)
    if detected != "mps":
        return detected

    family = str(bundle.get("model_family") or bundle.get("model_type") or "")
    backend = str(bundle.get("backend") or "sentence_transformers")

    if family in {
        "causal_lm_classifier",
        "transformer_sequence_classifier",
    }:
        LOGGER.debug(
            "using cpu runtime for bundle family=%s model_id=%s reason=%s",
            family,
            bundle.get("model_id") or "",
            "neural bridge inference is forced off MPS on this machine",
        )
        return "cpu"

    if family == "semantic_embedding" and backend in {"hf_embedding", "sentence_transformers"}:
        LOGGER.debug(
            "using cpu runtime for bundle family=%s model_id=%s reason=%s",
            family,
            bundle.get("model_id") or "",
            "semantic embedding inference is forced off MPS on this machine",
        )
        return "cpu"

    return detected


def _torch_runtime_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _positive_scores_from_logits(logits: Any) -> list[float]:
    scores = np.asarray(logits)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Expected binary classification logits with shape [batch, 2]")
    stabilized = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(stabilized)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return [float(row[1]) for row in probabilities]
