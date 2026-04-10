from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ask_seattle import __version__
from ask_seattle.data import LabeledPost, post_text
from ask_seattle.model import (
    ModelSelection,
    choose_active_model,
    positive_probabilities,
    save_model,
    select_threshold,
    split_labeled_posts,
    train_model,
)
from ask_seattle.transformer_model import (
    DEFAULT_BASE_MODEL,
    load_transformer_bundle,
    train_transformer_model,
    update_transformer_metadata,
)


def train_all_models(
    posts: list[LabeledPost],
    output_dir: str | Path,
    *,
    min_precision: float = 0.95,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    include_transformer: bool = True,
    transformer_base_model: str = DEFAULT_BASE_MODEL,
    transformer_epochs: int = 2,
    transformer_batch_size: int = 8,
    production_ready_blocked_reason: str | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    split = split_labeled_posts(
        posts,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    y_test = [post.label for post in split.test]
    test_texts = [post_text(post.title, post.selftext) for post in split.test]
    candidates: list[ModelSelection] = []
    candidate_reports: list[dict[str, Any]] = []

    tfidf_model = train_model(split.train)
    tfidf_scores = positive_probabilities(tfidf_model, test_texts)
    tfidf_selection = select_threshold(
        y_test,
        tfidf_scores,
        min_precision=min_precision,
    )
    tfidf_path = artifact_dir / "tfidf_logreg.joblib"
    save_model(tfidf_model, tfidf_path, threshold=tfidf_selection.threshold)
    tfidf_candidate = ModelSelection(
        model_name="tfidf_logreg",
        threshold=tfidf_selection.threshold,
        precision=tfidf_selection.precision,
        recall=tfidf_selection.recall,
        f1=tfidf_selection.f1,
        production_ready=tfidf_selection.production_ready and production_ready_blocked_reason is None,
    )
    candidates.append(tfidf_candidate)
    candidate_reports.append(
        {
            **asdict(tfidf_candidate),
            "artifact_path": str(tfidf_path),
            "support": tfidf_selection.support,
        }
    )

    if include_transformer:
        transformer_dir = artifact_dir / "transformer_sequence_classifier"
        train_transformer_model(
            split.train,
            split.validation,
            transformer_dir,
            threshold=0.5,
            base_model=transformer_base_model,
            epochs=transformer_epochs,
            batch_size=transformer_batch_size,
        )
        transformer_bundle = load_transformer_bundle(transformer_dir)
        transformer_scores = _score_transformer_bundle(transformer_bundle, test_texts)
        transformer_selection = select_threshold(
            y_test,
            transformer_scores,
            min_precision=min_precision,
        )
        transformer_candidate = ModelSelection(
            model_name="transformer_sequence_classifier",
            threshold=transformer_selection.threshold,
            precision=transformer_selection.precision,
            recall=transformer_selection.recall,
            f1=transformer_selection.f1,
            production_ready=(
                transformer_selection.production_ready and production_ready_blocked_reason is None
            ),
        )
        update_transformer_metadata(
            transformer_dir,
            {
                "threshold": transformer_selection.threshold,
                "production_ready": transformer_candidate.production_ready,
            },
        )
        candidates.append(transformer_candidate)
        candidate_reports.append(
            {
                **asdict(transformer_candidate),
                "artifact_path": str(transformer_dir),
                "support": transformer_selection.support,
            }
        )

    active = choose_active_model(candidates)
    summary = {
        "version": __version__,
        "min_precision": min_precision,
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "random_state": random_state,
        },
        "candidates": candidate_reports,
        "active_model": asdict(active) if active else None,
        "production_ready": active is not None,
        "production_ready_blocked_reason": production_ready_blocked_reason,
    }
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _score_transformer_bundle(bundle: dict[str, Any], texts: list[str]) -> list[float]:
    from ask_seattle.transformer_model import transformer_positive_probabilities

    return transformer_positive_probabilities(bundle, texts)
