from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from ask_seattle import __version__
from ask_seattle.data import LabeledPost, post_text

DEFAULT_THRESHOLD_GRID = tuple(round(index / 100, 2) for index in range(5, 100, 5))


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    classification_report: str
    confusion_matrix: list[list[int]]
    precision: float
    recall: float
    f1: float
    support: int
    threshold_sweep: list[dict[str, float | int]]


@dataclass(frozen=True)
class DatasetSplit:
    train: list[LabeledPost]
    validation: list[LabeledPost]
    test: list[LabeledPost]


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    precision: float
    recall: float
    f1: float
    support: int
    production_ready: bool


@dataclass(frozen=True)
class ModelSelection:
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    production_ready: bool


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=30_000,
                    min_df=1,
                    max_df=0.95,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=40_000,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_model(posts: list[LabeledPost]) -> Pipeline:
    _validate_posts(posts)
    model = build_pipeline()
    model.fit(_texts(posts), _labels(posts))
    return model


def train_and_evaluate(
    posts: list[LabeledPost],
    *,
    threshold: float,
    test_size: float,
    random_state: int,
) -> tuple[Pipeline, EvaluationResult | None]:
    _validate_posts(posts)

    labels = _labels(posts)
    class_counts = Counter(labels)
    test_count = ceil(len(posts) * test_size)
    train_count = len(posts) - test_count
    can_stratify = (
        len(posts) >= 4
        and min(class_counts.values()) >= 2
        and test_count >= 2
        and train_count >= 2
    )

    if not can_stratify:
        return train_model(posts), None

    train_posts, test_posts = train_test_split(
        posts,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = train_model(train_posts)
    evaluation = evaluate_model(model, test_posts, threshold=threshold)

    final_model = train_model(posts)
    return final_model, evaluation


def split_labeled_posts(
    posts: list[LabeledPost],
    *,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    _validate_posts(posts)
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    labels = _labels(posts)
    class_counts = Counter(labels)
    if min(class_counts.values()) < 3:
        raise ValueError("Need at least 3 examples per class for train/validation/test splits")

    train_validation_posts, test_posts = train_test_split(
        posts,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    validation_fraction = validation_size / (1 - test_size)
    train_posts, validation_posts = train_test_split(
        train_validation_posts,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=_labels(train_validation_posts),
    )

    return DatasetSplit(train=train_posts, validation=validation_posts, test=test_posts)


def evaluate_model(model: Pipeline, posts: list[LabeledPost], *, threshold: float) -> EvaluationResult:
    y_true = _labels(posts)
    probabilities = positive_probabilities(model, _texts(posts))
    y_pred = [1 if probability >= threshold else 0 for probability in probabilities]

    metrics = _binary_metrics(y_true, y_pred)

    return EvaluationResult(
        threshold=threshold,
        classification_report=classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["not_askseattle", "askseattle"],
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        support=int(metrics["support"]),
        threshold_sweep=[
            {"threshold": sweep_threshold}
            | _binary_metrics(
                y_true,
                [1 if probability >= sweep_threshold else 0 for probability in probabilities],
            )
            for sweep_threshold in (0.5, 0.65, 0.75, 0.85, 0.9)
        ],
    )


def threshold_sweep(
    y_true: list[int],
    probabilities: list[float],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> list[dict[str, float | int]]:
    return [
        {"threshold": threshold}
        | _binary_metrics(y_true, [1 if probability >= threshold else 0 for probability in probabilities])
        for threshold in thresholds
    ]


def select_threshold(
    y_true: list[int],
    probabilities: list[float],
    *,
    min_precision: float = 0.95,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
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


def choose_active_model(selections: list[ModelSelection]) -> ModelSelection | None:
    ready = [selection for selection in selections if selection.production_ready]
    if not ready:
        return None

    return max(
        ready,
        key=lambda selection: (
            selection.recall,
            selection.precision,
            selection.f1,
            selection.model_name,
        ),
    )


def save_model(model: Pipeline, path: str | Path, *, threshold: float) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": __version__,
        "threshold": threshold,
        "positive_label": 1,
        "version": __version__,
    }
    joblib.dump(bundle, model_path)


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if model_path.is_dir():
        from ask_seattle.transformer_model import load_transformer_bundle

        return load_transformer_bundle(model_path)

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        msg = f"{path} is not an ask-seattle model bundle"
        raise ValueError(msg)
    bundle.setdefault("model_type", "tfidf")
    bundle.setdefault("model_name", "tfidf_logreg")
    bundle.setdefault("model_version", str(bundle.get("version") or __version__))
    return bundle


def score_post(bundle: dict[str, Any], *, title: str, selftext: str = "") -> float:
    return score_texts(bundle, [post_text(title, selftext)])[0]


def score_texts(bundle: dict[str, Any], texts: list[str]) -> list[float]:
    if bundle.get("model_type") == "transformer":
        from ask_seattle.transformer_model import transformer_positive_probabilities

        return transformer_positive_probabilities(bundle, texts)

    model = bundle["model"]
    return positive_probabilities(model, texts)


def positive_probabilities(model: Pipeline, texts: list[str]) -> list[float]:
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(texts)
    positive_index = list(classifier.classes_).index(1)
    return [float(row[positive_index]) for row in probabilities]


def _validate_posts(posts: list[LabeledPost]) -> None:
    if not posts:
        raise ValueError("Training data is empty")
    labels = set(_labels(posts))
    if labels != {0, 1}:
        raise ValueError("Training data must include both askseattle and not_askseattle examples")


def _texts(posts: list[LabeledPost]) -> list[str]:
    return [post_text(post.title, post.selftext) for post in posts]


def _labels(posts: list[LabeledPost]) -> list[int]:
    return [post.label for post in posts]


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
