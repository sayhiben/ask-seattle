from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from math import ceil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from ask_seattle import __version__
from ask_seattle.data import (
    LabeledPost,
    body_length_bucket,
    is_sparse_media_post,
    normalize_body,
    post_metadata_text,
    post_text,
    title_length_bucket,
)

DEFAULT_THRESHOLD_GRID = tuple(round(index / 100, 2) for index in range(5, 100, 5))
SPARSE_MEDIA_HIGH_THRESHOLD_DELTA = 0.1
DEFAULT_SPLIT_STRATEGY = "random"
DEFAULT_SPLIT_SEED = 13
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
DEFAULT_EXTRA_WORD_STOPWORDS = frozenset()
DEFAULT_CHAR_WEIGHT = 0.25


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
    support: int
    production_ready: bool


@dataclass(frozen=True)
class DecisionThresholds:
    low_threshold: float
    high_threshold: float
    high_threshold_selection: ThresholdSelection
    low_threshold_metrics: dict[str, float | int]
    high_threshold_sweep: list[dict[str, float | int]]
    low_threshold_sweep: list[dict[str, float | int]]
    abstain_enabled: bool


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


def build_pipeline(*, min_df: int = 2) -> Pipeline:
    return build_pipeline_with_config(
        min_df=min_df,
        extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
        char_weight=DEFAULT_CHAR_WEIGHT,
    )


def build_pipeline_with_config(
    *,
    min_df: int = 2,
    extra_word_stopwords: set[str] | frozenset[str] | None = None,
    char_weight: float = DEFAULT_CHAR_WEIGHT,
) -> Pipeline:
    word_stopwords = sorted(WORD_STOPWORDS | set(extra_word_stopwords or ()))
    features = FeatureUnion(
        [
            (
                "title_word",
                Pipeline(
                    [
                        ("extractor", TextFieldExtractor("title")),
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
                        ("extractor", TextFieldExtractor("body")),
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
                        ("extractor", TextFieldExtractor("text")),
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
        ],
        transformer_weights={
            "title_word": 2.0,
            "body_word": 1.0,
            "char_wb": char_weight,
        },
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
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
) -> Pipeline:
    _validate_posts(posts)
    effective_word_stopwords = (
        DEFAULT_EXTRA_WORD_STOPWORDS if extra_word_stopwords is None else set(extra_word_stopwords)
    )
    model = build_pipeline_with_config(
        min_df=_default_min_df(posts),
        extra_word_stopwords=effective_word_stopwords,
        char_weight=char_weight,
    )
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight
    model.fit(_rows(posts), _labels(posts), **fit_kwargs)
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
    return [
        {"threshold": threshold}
        | _binary_metrics(y_true, [1 if probability >= threshold else 0 for probability in probabilities])
        for threshold in resolved_thresholds
    ]


def select_threshold(
    y_true: list[int],
    probabilities: list[float],
    *,
    min_precision: float = 0.95,
    thresholds: tuple[float, ...] | None = None,
) -> ThresholdSelection:
    sweep = threshold_sweep(y_true, probabilities, thresholds)
    ready = [row for row in sweep if float(row["precision"]) >= min_precision]
    candidates = ready or sweep
    selected = max(
        candidates,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            float(row["f1"]),
            float(row["threshold"]),
        ),
    )

    return ThresholdSelection(
        threshold=float(selected["threshold"]),
        precision=float(selected["precision"]),
        recall=float(selected["recall"]),
        f1=float(selected["f1"]),
        support=int(selected["support"]),
        production_ready=bool(ready),
    )


def select_decision_thresholds(
    y_true: list[int],
    probabilities: list[float],
    *,
    auto_precision_target: float,
    thresholds: tuple[float, ...] | None = None,
) -> DecisionThresholds:
    high_threshold_selection = select_threshold(
        y_true,
        probabilities,
        min_precision=auto_precision_target,
        thresholds=thresholds,
    )
    high_threshold_sweep = threshold_sweep(y_true, probabilities, thresholds)
    low_threshold_sweep = high_threshold_sweep
    best_low = max(
        low_threshold_sweep,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
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
        "model_name": "tfidf_logreg",
        "model_version": __version__,
        "threshold": high_threshold,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "threshold_policy": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
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
    if "model" in bundle:
        return _normalize_tfidf_bundle(bundle)
    if bundle.get("model_family") == "semantic_embedding" and "classifier" in bundle:
        return _load_semantic_bundle(bundle)
    if bundle.get("model_family") == "transformer_sequence_classifier":
        return _load_transformer_bundle_from_joblib(bundle, source_path=model_path)
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
    return raw_score_rows(
        bundle,
        [
            build_inference_row(
                title=title,
                selftext=selftext,
                post_type=post_type,
                content_domain=content_domain,
                is_crosspost=is_crosspost,
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
    return score_rows(
        bundle,
        [
            build_inference_row(
                title=title,
                selftext=selftext,
                post_type=post_type,
                content_domain=content_domain,
                is_crosspost=is_crosspost,
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
    raise ValueError(f"Unsupported model family: {family}")


def positive_probabilities(model: Pipeline, rows: list[dict[str, str]]) -> list[float]:
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(rows)
    positive_index = list(classifier.classes_).index(1)
    return [float(row[positive_index]) for row in probabilities]


def _semantic_positive_probabilities(bundle: dict[str, Any], rows: list[dict[str, str]]) -> list[float]:
    encoder = bundle.get("encoder")
    classifier = bundle.get("classifier")
    if encoder is None or classifier is None:
        raise ValueError("Semantic embedding bundle is missing encoder or classifier")
    embeddings = encoder.encode([str(row.get("text") or "") for row in rows], show_progress_bar=False)
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

    titles = [str(row.get("title") or "") for row in rows]
    bodies = [str(row.get("body") or "") for row in rows]
    encoded = tokenizer(
        titles,
        bodies,
        truncation=True,
        max_length=int(bundle.get("max_length") or 384),
        padding=True,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
    return _positive_scores_from_logits(logits.detach().cpu().numpy())


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
    if row and row.get("is_sparse_media"):
        return min(0.99, high_threshold + SPARSE_MEDIA_HIGH_THRESHOLD_DELTA)
    return high_threshold


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
    high_threshold = float(bundle.get("high_threshold") or bundle.get("threshold") or 0.85)
    low_threshold = float(bundle.get("low_threshold") or high_threshold)
    low_threshold = min(low_threshold, high_threshold)
    row = build_inference_row(
        title=title,
        selftext=selftext,
        post_type=post_type,
        content_domain=content_domain,
        is_crosspost=is_crosspost,
    )

    raw_score = raw_score_rows(bundle, [row])[0]
    calibrated_score = score_rows(bundle, [row])[0]
    label = "askseattle" if calibrated_score >= low_threshold else "not_askseattle"

    return CheckResult(
        post_id=post_id,
        permalink=permalink,
        model_name=str(bundle.get("model_name") or bundle.get("model_type") or "unknown"),
        model_version=str(bundle.get("model_version") or bundle.get("version") or "unknown"),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
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


def _validate_posts(posts: list[LabeledPost]) -> None:
    if not posts:
        raise ValueError("Training data is empty")
    labels = set(_labels(posts))
    if labels != {0, 1}:
        raise ValueError("Training data must include both askseattle and not_askseattle examples")


def _rows(posts: list[LabeledPost]) -> list[dict[str, str]]:
    return [
        build_inference_row(
            title=post.title,
            selftext=post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
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
) -> dict[str, str]:
    body = normalize_body(selftext)
    metadata = post_metadata_text(
        title=title,
        selftext=body,
        post_type=post_type,
        content_domain=content_domain,
        is_crosspost=is_crosspost,
    )
    sparse_media = is_sparse_media_post(post_type=post_type, selftext=body)
    return {
        "title": str(title).strip(),
        "body": "\n".join(part for part in (metadata, body.strip()) if part).strip(),
        "text": post_text(
            title,
            body,
            post_type=post_type,
            content_domain=content_domain,
            is_crosspost=is_crosspost,
        ),
        "post_type": str(post_type or "").strip(),
        "content_domain": str(content_domain or "").strip(),
        "title_length_bucket": title_length_bucket(title),
        "body_length_bucket": body_length_bucket(body),
        "has_body": "yes" if body.strip() else "no",
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
    return 1 if len(posts) < 50 else 2


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
    _apply_threshold_policy_defaults(bundle)
    return bundle


def _load_semantic_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ValueError(
            "Semantic embedding inference requires sentence-transformers. Install optional model "
            "dependencies with `python -m pip install -e \".[dev,models]\"`."
        ) from exc

    normalized = dict(bundle)
    normalized["encoder"] = SentenceTransformer(str(bundle["model_id"]))
    normalized.setdefault("model_family", "semantic_embedding")
    normalized.setdefault("model_name", "semantic_embedding_logreg")
    normalized.setdefault("model_version", str(bundle.get("version") or __version__))
    _apply_threshold_policy_defaults(normalized)
    return normalized


def _load_transformer_bundle_from_joblib(bundle: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    model_dir = bundle.get("artifact_path") or bundle.get("model_dir")
    if not model_dir:
        raise ValueError(f"{source_path} is missing transformer artifact metadata")
    if not Path(model_dir).is_absolute():
        model_dir = str((source_path.parent / str(model_dir)).resolve())
    normalized = dict(bundle)
    normalized["artifact_path"] = str(model_dir)
    return _load_transformer_runtime_bundle(normalized, model_dir=Path(model_dir))


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
    normalized["tokenizer"] = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    normalized["model"] = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    normalized.setdefault("model_family", "transformer_sequence_classifier")
    normalized.setdefault("model_name", "transformer_sequence_classifier")
    normalized.setdefault("model_version", str(normalized.get("version") or __version__))
    normalized.setdefault("max_length", _transformer_max_length(model_dir))
    _apply_threshold_policy_defaults(normalized)
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
        "calibration_method": policy.get("calibration_method") or bundle.get("calibration_method"),
        "split_strategy": policy.get("split_strategy") or bundle.get("split_strategy") or "manual",
        "split_seed": _first_defined(policy.get("split_seed"), bundle.get("split_seed")),
        "evaluation_subreddit": policy.get("evaluation_subreddit") or bundle.get("evaluation_subreddit"),
        "time_coverage": policy.get("time_coverage") or bundle.get("time_coverage"),
    }


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _positive_scores_from_logits(logits: Any) -> list[float]:
    scores = np.asarray(logits)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Expected binary classification logits with shape [batch, 2]")
    stabilized = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(stabilized)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return [float(row[1]) for row in probabilities]
