from __future__ import annotations

import gc
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from ask_seattle import __version__
from ask_seattle.data import LabeledPost, post_text, prepare_training_posts
from ask_seattle.model import (
    CalibrationResult,
    DEFAULT_CHAR_WEIGHT,
    DEFAULT_EXTRA_WORD_STOPWORDS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    DatasetSplit,
    DecisionThresholds,
    ThresholdSelection,
    apply_probability_calibrator,
    build_inference_row,
    confidence_band_for_row,
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
DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS = 5
DEFAULT_CALIBRATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TRANSFORMER_MODEL_ID = "microsoft/deberta-v3-small"
DEFAULT_MAX_SLICE_POSITIVE_WEIGHT = 2.0
LOGGER = logging.getLogger("ask_seattle.training")


@dataclass(frozen=True)
class VariantConfig:
    name: str
    extra_word_stopwords: frozenset[str]
    char_weight: float


@dataclass(frozen=True)
class OperatingMetrics:
    auto_band: dict[str, float | int]
    review_queue: dict[str, float | int]
    queue_counts: dict[str, int]
    queue_rates: dict[str, float]
    positive_prevalence: float
    positive_count: int
    total_count: int
    slice_metrics: dict[str, Any]


@dataclass(frozen=True)
class SliceAwareWeighting:
    sample_weights: list[float]
    summary: dict[str, Any]


class OptionalModelDependencyError(RuntimeError):
    pass


def train_model_bundle(
    posts: list[LabeledPost],
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    prepared_data_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    split = split_labeled_posts(
        posts,
        calibration_size=DEFAULT_CALIBRATION_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )
    return _train_model_bundle_for_split(
        split=split,
        output_dir=output_dir,
        variant=VariantConfig(
            name="recommended",
            extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
            char_weight=DEFAULT_CHAR_WEIGHT,
        ),
        prepared_data_summary=prepared_data_summary,
    )


def train_model_bundle_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
) -> dict[str, Any]:
    posts, prepared_data_summary = prepare_training_posts(input_path)
    return train_model_bundle(
        posts,
        output_dir,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
        prepared_data_summary=prepared_data_summary,
    )


def benchmark_model_variants_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
) -> dict[str, Any]:
    posts, prepared_data_summary = prepare_training_posts(input_path)
    split = split_labeled_posts(
        posts,
        calibration_size=DEFAULT_CALIBRATION_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )

    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        VariantConfig(name="legacy_baseline", extra_word_stopwords=frozenset(), char_weight=0.5),
        VariantConfig(
            name="extra_stopwords_only",
            extra_word_stopwords=frozenset({"just", "one", "some"}),
            char_weight=0.5,
        ),
        VariantConfig(name="lower_char_weight_only", extra_word_stopwords=frozenset(), char_weight=DEFAULT_CHAR_WEIGHT),
        VariantConfig(
            name="recommended",
            extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
            char_weight=DEFAULT_CHAR_WEIGHT,
        ),
    ]
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_dir = benchmark_dir / variant.name
        summary = _train_model_bundle_for_split(
            split=split,
            output_dir=variant_dir,
            variant=variant,
            prepared_data_summary=prepared_data_summary,
        )
        results.append(
            {
                "name": variant.name,
                "artifact_path": summary["artifact_path"],
                "summary_path": str(variant_dir / "training_summary.json"),
                "extra_word_stopwords": sorted(variant.extra_word_stopwords),
                "char_weight": variant.char_weight,
                "production_ready": summary["production_ready"],
                "production_ready_blocked_reason": summary["production_ready_blocked_reason"],
                "metrics": summary["metrics"],
                "operating_metrics": summary["operating_metrics"],
                "production_gate": summary["production_gate"],
                "threshold_policy": summary["threshold_policy"],
                "feature_audit": summary["feature_audit"],
            }
        )

    aggregate = {
        "version": __version__,
        "benchmark_output_dir": str(benchmark_dir),
        "evaluation_subreddit": split.evaluation_subreddit,
        "production_gate": _production_gate_summary(),
        "prepared_data": prepared_data_summary,
        "split": {
            "train": len(split.train),
            "calibration": len(split.calibration),
            "test": len(split.test),
            "split_strategy": split.split_strategy,
            "split_seed": split.split_seed,
            "evaluation_subreddit": split.evaluation_subreddit,
            "excluded_for_time_split": split.excluded_for_time_split,
            "time_coverage": split.time_coverage,
        },
        "variants": results,
    }
    (benchmark_dir / "variant_benchmark_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def benchmark_model_suite_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    semantic_model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    transformer_model_id: str = DEFAULT_TRANSFORMER_MODEL_ID,
) -> dict[str, Any]:
    posts, prepared_data_summary = prepare_training_posts(input_path)
    split = split_labeled_posts(
        posts,
        calibration_size=DEFAULT_CALIBRATION_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )

    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    tfidf_summary = _train_model_bundle_for_split(
        split=split,
        output_dir=benchmark_dir / "tfidf_recommended",
        variant=VariantConfig(
            name="recommended",
            extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
            char_weight=DEFAULT_CHAR_WEIGHT,
        ),
        prepared_data_summary=prepared_data_summary,
    )
    results.append(_suite_entry_from_summary("tfidf_recommended", tfidf_summary))

    for suite_name, runner in (
        (
            "semantic_embedding",
            lambda: _train_semantic_embedding_bundle_for_split(
                split=split,
                output_dir=benchmark_dir / "semantic_embedding",
                model_id=semantic_model_id,
                prepared_data_summary=prepared_data_summary,
            ),
        ),
        (
            "transformer_sequence_classifier",
            lambda: _train_transformer_bundle_for_split(
                split=split,
                output_dir=benchmark_dir / "transformer_sequence_classifier",
                model_id=transformer_model_id,
                prepared_data_summary=prepared_data_summary,
            ),
        ),
    ):
        try:
            summary = runner()
        except OptionalModelDependencyError as exc:
            results.append(_suite_unavailable_entry(suite_name, str(exc)))
        else:
            results.append(_suite_entry_from_summary(suite_name, summary))

    aggregate = {
        "version": __version__,
        "benchmark_output_dir": str(benchmark_dir),
        "evaluation_subreddit": split.evaluation_subreddit,
        "production_gate": _production_gate_summary(),
        "prepared_data": prepared_data_summary,
        "split": _split_summary(split),
        "metrics_reference": _metrics_reference(),
        "models": results,
    }
    (benchmark_dir / "benchmark_suite_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


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


def _train_model_bundle_for_split(
    *,
    split: DatasetSplit,
    output_dir: str | Path,
    variant: VariantConfig,
    prepared_data_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _inference_rows(split.train)
    calibration_rows = _inference_rows(split.calibration)
    test_rows = _inference_rows(split.test)
    y_calibration = [post.label for post in split.calibration]
    y_test = [post.label for post in split.test]
    slice_weighting = _slice_aware_positive_weighting(split.train, rows=train_rows)

    model = train_model(
        split.train,
        sample_weight=slice_weighting.sample_weights,
        extra_word_stopwords=variant.extra_word_stopwords,
        char_weight=variant.char_weight,
    )
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
        rows=test_rows,
    )
    operating_metrics = _operating_metrics_summary(
        y_test,
        calibrated_test_scores,
        rows=test_rows,
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

    production_ready, blocked_reason = _production_ready_status(
        calibration=calibration,
        thresholds=thresholds,
        high_confidence_precision=band_metrics.high_confidence_precision,
        high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
    )

    summary = {
        "version": __version__,
        "model_name": "tfidf_logreg",
        "model_family": "tfidf",
        "variant": {
            "name": variant.name,
            "extra_word_stopwords": sorted(variant.extra_word_stopwords),
            "char_weight": variant.char_weight,
        },
        "artifact_path": str(artifact_path),
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(thresholds),
        "metrics": {
            "high_confidence_precision": band_metrics.high_confidence_precision,
            "high_confidence_recall": band_metrics.high_confidence_recall,
            "high_confidence_f1": band_metrics.high_confidence_f1,
            "support": band_metrics.support,
            "confidence_band_counts": band_metrics.band_counts,
        },
        "operating_metrics": asdict(operating_metrics),
        "threshold_policy": threshold_policy,
        "feature_audit": tfidf_feature_audit(model),
        "training_balance": slice_weighting.summary,
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


def _train_semantic_embedding_bundle_for_split(
    *,
    split: DatasetSplit,
    output_dir: str | Path,
    model_id: str,
    prepared_data_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise OptionalModelDependencyError(
            "Semantic embedding benchmarks require sentence-transformers. "
            "Install with `python -m pip install -e \".[dev,models]\"`."
        ) from exc

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(model_id)

    train_rows = _inference_rows(split.train)
    slice_weighting = _slice_aware_positive_weighting(split.train, rows=train_rows)
    train_embeddings = encoder.encode(_texts(split.train), show_progress_bar=False)
    calibration_embeddings = encoder.encode(_texts(split.calibration), show_progress_bar=False)
    test_embeddings = encoder.encode(_texts(split.test), show_progress_bar=False)
    test_rows = _inference_rows(split.test)
    y_train = [post.label for post in split.train]
    y_calibration = [post.label for post in split.calibration]
    y_test = [post.label for post in split.test]

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        solver="liblinear",
    )
    classifier.fit(train_embeddings, y_train, sample_weight=slice_weighting.sample_weights)
    raw_calibration_scores = [float(row[1]) for row in classifier.predict_proba(calibration_embeddings)]
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        calibration=calibration,
    )

    raw_test_scores = [float(row[1]) for row in classifier.predict_proba(test_embeddings)]
    calibrated_test_scores = apply_probability_calibrator(calibrator, raw_test_scores)
    band_metrics = evaluate_decision_policy(
        y_test,
        calibrated_test_scores,
        low_threshold=thresholds.low_threshold,
        high_threshold=thresholds.high_threshold,
        rows=test_rows,
    )
    operating_metrics = _operating_metrics_summary(
        y_test,
        calibrated_test_scores,
        rows=test_rows,
        low_threshold=thresholds.low_threshold,
        high_threshold=thresholds.high_threshold,
    )

    threshold_policy = _decision_policy(split=split, calibration=calibration, thresholds=thresholds)
    artifact_path = artifact_dir / "semantic_embedding_logreg.joblib"
    joblib.dump(
        {
            "model_family": "semantic_embedding",
            "model_name": "semantic_embedding_logreg",
            "model_id": model_id,
            "embedding_dimension": int(np.asarray(train_embeddings).shape[1]),
            "classifier": classifier,
            "calibrator": calibrator,
            "threshold_policy": threshold_policy,
            "version": __version__,
        },
        artifact_path,
    )

    production_ready, blocked_reason = _production_ready_status(
        calibration=calibration,
        thresholds=thresholds,
        high_confidence_precision=band_metrics.high_confidence_precision,
        high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
    )

    summary = {
        "version": __version__,
        "model_name": "semantic_embedding_logreg",
        "model_family": "semantic_embedding",
        "model_id": model_id,
        "artifact_path": str(artifact_path),
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(thresholds),
        "metrics": {
            "high_confidence_precision": band_metrics.high_confidence_precision,
            "high_confidence_recall": band_metrics.high_confidence_recall,
            "high_confidence_f1": band_metrics.high_confidence_f1,
            "support": band_metrics.support,
            "confidence_band_counts": band_metrics.band_counts,
        },
        "operating_metrics": asdict(operating_metrics),
        "threshold_policy": threshold_policy,
        "embedding_summary": {
            "embedding_dimension": int(np.asarray(train_embeddings).shape[1]),
            "train_examples": len(split.train),
            "model_id": model_id,
        },
        "training_balance": slice_weighting.summary,
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


def _train_transformer_bundle_for_split(
    *,
    split: DatasetSplit,
    output_dir: str | Path,
    model_id: str,
    prepared_data_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise OptionalModelDependencyError(
            "Transformer benchmarks require transformers, datasets, accelerate, and torch. "
            "Install with `python -m pip install -e \".[dev,models]\"`."
        ) from exc

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    use_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    use_cpu = use_mps
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 16
    gradient_accumulation_steps = 1
    if use_mps and hasattr(torch.mps, "empty_cache"):
        gc.collect()
        torch.mps.empty_cache()

    train_inference_rows = _inference_rows(split.train)
    slice_weighting = _slice_aware_positive_weighting(split.train, rows=train_inference_rows)
    train_rows = _sequence_classification_rows(split.train, example_weights=slice_weighting.sample_weights)
    calibration_rows = _sequence_classification_rows(split.calibration)
    test_rows = _sequence_classification_rows(split.test)
    test_inference_rows = _inference_rows(split.test)
    y_train = [row["label"] for row in train_rows]
    y_calibration = [row["label"] for row in calibration_rows]
    y_test = [row["label"] for row in test_rows]

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    previous_transformers_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            id2label={0: "not_askseattle", 1: "askseattle"},
            label2id={"not_askseattle": 0, "askseattle": 1},
        )
    finally:
        transformers_logging.set_verbosity(previous_transformers_verbosity)
    LOGGER.info(
        "initialized transformer sequence-classification model model_id=%s tokenizer_backend=slow",
        model_id,
    )

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["title"], batch["body"], truncation=True, max_length=384)

    train_dataset = Dataset.from_list(train_rows).map(tokenize, batched=True)
    calibration_dataset = Dataset.from_list(calibration_rows).map(tokenize, batched=True)
    test_dataset = Dataset.from_list(test_rows).map(tokenize, batched=True)

    class_weights = torch.tensor(_balanced_class_weights(y_train), dtype=torch.float32)

    class WeightedSequenceClassificationTrainer(Trainer):
        def __init__(self, *args: Any, class_weights: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
        ) -> Any:
            labels = inputs["labels"]
            example_weight = inputs.pop("example_weight", None)
            outputs = model(**inputs)
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                reduction="none",
            )
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            if example_weight is not None:
                loss = loss * example_weight.to(logits.device).view(-1)
            loss = loss.mean()
            if return_outputs:
                return loss, outputs
            return loss

    training_args = TrainingArguments(
        output_dir=str(artifact_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        weight_decay=0.01,
        use_cpu=use_cpu,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )
    trainer = WeightedSequenceClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=calibration_dataset,
        processing_class=tokenizer,
        class_weights=class_weights,
    )
    trainer.train()
    if use_mps and hasattr(torch.mps, "empty_cache"):
        gc.collect()
        torch.mps.empty_cache()

    raw_calibration_scores = _positive_scores_from_logits(trainer.predict(calibration_dataset).predictions)
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        calibration=calibration,
    )
    raw_test_scores = _positive_scores_from_logits(trainer.predict(test_dataset).predictions)
    calibrated_test_scores = apply_probability_calibrator(calibrator, raw_test_scores)
    band_metrics = evaluate_decision_policy(
        y_test,
        calibrated_test_scores,
        low_threshold=thresholds.low_threshold,
        high_threshold=thresholds.high_threshold,
        rows=test_inference_rows,
    )
    operating_metrics = _operating_metrics_summary(
        y_test,
        calibrated_test_scores,
        rows=test_inference_rows,
        low_threshold=thresholds.low_threshold,
        high_threshold=thresholds.high_threshold,
    )

    model_dir = artifact_dir / "transformer_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    threshold_policy = _decision_policy(split=split, calibration=calibration, thresholds=thresholds)
    metadata_path = artifact_dir / "transformer_metadata.json"
    bundle_path = artifact_dir / "transformer_bundle.joblib"
    training_args_summary = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": 3,
        "max_length": 384,
        "input_format": "title_body_pair",
        "body_includes_metadata_tokens": True,
        "class_weighting": "balanced_cross_entropy",
        "runtime_profile": "cpu_fallback_on_mps" if use_cpu else "default",
        "class_weights": {
            "not_askseattle": float(class_weights[0].item()),
            "askseattle": float(class_weights[1].item()),
        },
    }
    joblib.dump(
        {
            "model_family": "transformer_sequence_classifier",
            "model_name": "transformer_sequence_classifier",
            "model_id": model_id,
            "artifact_path": str(model_dir.resolve()),
            "model_dir": str(model_dir.resolve()),
            "calibrator": calibrator,
            "threshold_policy": threshold_policy,
            "training_args": training_args_summary,
            "version": __version__,
        },
        bundle_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model_family": "transformer_sequence_classifier",
                "model_name": "transformer_sequence_classifier",
                "model_id": model_id,
                "artifact_path": str(model_dir),
                "bundle_path": str(bundle_path),
                "calibrator_bundle_path": str(bundle_path),
                "threshold_policy": threshold_policy,
                "training_args": training_args_summary,
                "version": __version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    production_ready, blocked_reason = _production_ready_status(
        calibration=calibration,
        thresholds=thresholds,
        high_confidence_precision=band_metrics.high_confidence_precision,
        high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
    )
    summary = {
        "version": __version__,
        "model_name": "transformer_sequence_classifier",
        "model_family": "transformer_sequence_classifier",
        "model_id": model_id,
        "artifact_path": str(bundle_path),
        "model_dir": str(model_dir),
        "artifact_metadata_path": str(metadata_path),
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(thresholds),
        "metrics": {
            "high_confidence_precision": band_metrics.high_confidence_precision,
            "high_confidence_recall": band_metrics.high_confidence_recall,
            "high_confidence_f1": band_metrics.high_confidence_f1,
            "support": band_metrics.support,
            "confidence_band_counts": band_metrics.band_counts,
        },
        "operating_metrics": asdict(operating_metrics),
        "threshold_policy": threshold_policy,
        "training_args": training_args_summary,
        "training_balance": slice_weighting.summary,
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


def _positive_scores_from_logits(logits: Any) -> list[float]:
    scores = np.asarray(logits)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Expected binary classification logits with shape [batch, 2]")
    stabilized = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(stabilized)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return [float(row[1]) for row in probabilities]


def _sequence_classification_rows(
    posts: list[LabeledPost],
    *,
    example_weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_weights = example_weights or [1.0] * len(posts)
    for post, example_weight in zip(posts, resolved_weights, strict=True):
        inference_row = build_inference_row(
            title=post.title,
            selftext=post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
        )
        rows.append(
            {
                "title": inference_row["title"],
                "body": inference_row["body"],
                "text": inference_row["text"],
                "label": post.label,
                "example_weight": float(example_weight),
            }
        )
    return rows


def _texts(posts: list[LabeledPost]) -> list[str]:
    return [
        post_text(
            post.title,
            post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
        )
        for post in posts
    ]


def _inference_rows(posts: list[LabeledPost]) -> list[dict[str, Any]]:
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


def _slice_bucket_values(row: dict[str, Any]) -> dict[str, str]:
    post_type = str(row.get("post_type") or "").strip().lower()
    if post_type not in {"self", "link", "image"}:
        post_type = "other_or_unknown"
    return {
        "post_type": post_type,
        "low_text": "yes" if row.get("body_length_bucket") in {"none", "short"} else "no",
        "sparse_media": "yes" if bool(row.get("is_sparse_media")) else "no",
    }


def _coverage_summary(posts: list[LabeledPost]) -> dict[str, Any]:
    rows = _inference_rows(posts)
    labels = [post.label for post in posts]
    coverage: dict[str, dict[str, Any]] = {
        "post_type": {},
        "low_text": {},
        "sparse_media": {},
    }
    for label, row in zip(labels, rows, strict=True):
        label_name = "askseattle" if label == 1 else "not_askseattle"
        for slice_name, bucket in _slice_bucket_values(row).items():
            bucket_summary = coverage[slice_name].setdefault(
                bucket,
                {"askseattle": 0, "not_askseattle": 0, "total": 0},
            )
            bucket_summary[label_name] += 1
            bucket_summary["total"] += 1
    return coverage


def _slice_aware_positive_weighting(
    posts: list[LabeledPost],
    *,
    rows: list[dict[str, Any]] | None = None,
) -> SliceAwareWeighting:
    resolved_rows = rows or _inference_rows(posts)
    labels = [post.label for post in posts]
    positive_bucket_counts: dict[str, Counter[str]] = {
        "post_type": Counter(),
        "low_text": Counter(),
        "sparse_media": Counter(),
    }

    for label, row in zip(labels, resolved_rows, strict=True):
        if label != 1:
            continue
        for slice_name, bucket in _slice_bucket_values(row).items():
            positive_bucket_counts[slice_name][bucket] += 1

    bucket_weights: dict[str, dict[str, float]] = {}
    for slice_name, counts in positive_bucket_counts.items():
        if not counts:
            bucket_weights[slice_name] = {}
            continue
        max_count = max(counts.values())
        bucket_weights[slice_name] = {
            bucket: round(
                min(
                    DEFAULT_MAX_SLICE_POSITIVE_WEIGHT,
                    float((max_count / count) ** 0.5),
                ),
                4,
            )
            for bucket, count in counts.items()
            if count > 0
        }

    sample_weights: list[float] = []
    for label, row in zip(labels, resolved_rows, strict=True):
        if label != 1:
            sample_weights.append(1.0)
            continue
        row_bucket_weights = [
            bucket_weights.get(slice_name, {}).get(bucket, 1.0)
            for slice_name, bucket in _slice_bucket_values(row).items()
        ]
        sample_weights.append(max([1.0, *row_bucket_weights]))

    positive_weights = [weight for weight, label in zip(sample_weights, labels, strict=True) if label == 1]
    summary = {
        "strategy": "slice_aware_positive_weighting",
        "max_slice_positive_weight": DEFAULT_MAX_SLICE_POSITIVE_WEIGHT,
        "bucket_weights": bucket_weights,
        "train_positive_bucket_counts": {
            slice_name: dict(counts)
            for slice_name, counts in positive_bucket_counts.items()
        },
        "sample_weight_summary": {
            "mean": round(float(sum(sample_weights) / len(sample_weights)), 4) if sample_weights else 0.0,
            "max": round(float(max(sample_weights)), 4) if sample_weights else 0.0,
            "positive_mean": round(float(sum(positive_weights) / len(positive_weights)), 4)
            if positive_weights
            else 0.0,
            "positive_max": round(float(max(positive_weights)), 4) if positive_weights else 0.0,
        },
    }
    return SliceAwareWeighting(sample_weights=sample_weights, summary=summary)


def _suite_entry_from_summary(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "ok",
        "model_name": summary["model_name"],
        "model_family": summary["model_family"],
        "model_id": summary.get("model_id"),
        "artifact_path": summary["artifact_path"],
        "summary_path": str(Path(summary["artifact_path"]).parent / "training_summary.json")
        if summary["model_family"] != "transformer_sequence_classifier"
        else str(Path(summary["artifact_path"]).parent / "training_summary.json"),
        "production_ready": summary["production_ready"],
        "production_ready_blocked_reason": summary["production_ready_blocked_reason"],
        "production_gate": summary.get("production_gate"),
        "metrics": summary["metrics"],
        "operating_metrics": summary["operating_metrics"],
        "threshold_policy": summary["threshold_policy"],
    }


def _suite_unavailable_entry(name: str, error: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "unavailable",
        "error": error,
    }


def _split_summary(split: DatasetSplit) -> dict[str, Any]:
    return {
        "train": len(split.train),
        "calibration": len(split.calibration),
        "test": len(split.test),
        "split_strategy": split.split_strategy,
        "split_seed": split.split_seed,
        "evaluation_subreddit": split.evaluation_subreddit,
        "excluded_for_time_split": split.excluded_for_time_split,
        "time_coverage": split.time_coverage,
        "label_counts": {
            "train": _label_counts(split.train),
            "calibration": _label_counts(split.calibration),
            "test": _label_counts(split.test),
        },
        "coverage": {
            "train": _coverage_summary(split.train),
            "calibration": _coverage_summary(split.calibration),
            "test": _coverage_summary(split.test),
        },
    }


def _label_counts(posts: list[LabeledPost]) -> dict[str, int]:
    labels = Counter(post.label for post in posts)
    return {
        "not_askseattle": labels.get(0, 0),
        "askseattle": labels.get(1, 0),
    }


def _operating_metrics_summary(
    y_true: list[int],
    probabilities: list[float],
    *,
    rows: list[dict[str, Any]],
    low_threshold: float,
    high_threshold: float,
) -> OperatingMetrics:
    auto_predictions = [
        1
        if confidence_band_for_row(
            row,
            probability,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        == "high"
        else 0
        for probability, row in zip(probabilities, rows, strict=True)
    ]
    review_predictions = [1 if probability >= low_threshold else 0 for probability in probabilities]
    queue_counts = Counter(
        confidence_band_for_row(
            row,
            probability,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        for probability, row in zip(probabilities, rows, strict=True)
    )
    total_count = len(y_true)
    positive_count = Counter(y_true).get(1, 0)
    return OperatingMetrics(
        auto_band=_classification_metrics(y_true, auto_predictions),
        review_queue=_classification_metrics(y_true, review_predictions),
        queue_counts={
            "high": queue_counts.get("high", 0),
            "borderline": queue_counts.get("borderline", 0),
            "low": queue_counts.get("low", 0),
        },
        queue_rates={
            "auto_rate": _safe_rate(queue_counts.get("high", 0), total_count),
            "review_rate": _safe_rate(
                queue_counts.get("high", 0) + queue_counts.get("borderline", 0),
                total_count,
            ),
            "borderline_rate": _safe_rate(queue_counts.get("borderline", 0), total_count),
        },
        positive_prevalence=_safe_rate(positive_count, total_count),
        positive_count=positive_count,
        total_count=total_count,
        slice_metrics=_slice_metrics_summary(
            y_true,
            probabilities,
            rows=rows,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        ),
    )


def _slice_metrics_summary(
    y_true: list[int],
    probabilities: list[float],
    *,
    rows: list[dict[str, Any]],
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    slices: dict[str, dict[str, Any]] = {}
    slices["post_type"] = _slice_group_summary(
        y_true,
        probabilities,
        rows=rows,
        buckets={
            "self": lambda row: row.get("post_type") == "self",
            "link": lambda row: row.get("post_type") == "link",
            "image": lambda row: row.get("post_type") == "image",
            "other_or_unknown": lambda row: row.get("post_type") not in {"self", "link", "image"},
        },
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    slices["low_text"] = _slice_group_summary(
        y_true,
        probabilities,
        rows=rows,
        buckets={
            "yes": lambda row: row.get("body_length_bucket") in {"none", "short"},
            "no": lambda row: row.get("body_length_bucket") not in {"none", "short"},
        },
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    slices["sparse_media"] = _slice_group_summary(
        y_true,
        probabilities,
        rows=rows,
        buckets={
            "yes": lambda row: bool(row.get("is_sparse_media")),
            "no": lambda row: not bool(row.get("is_sparse_media")),
        },
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    return slices


def _slice_group_summary(
    y_true: list[int],
    probabilities: list[float],
    *,
    rows: list[dict[str, Any]],
    buckets: dict[str, Any],
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, predicate in buckets.items():
        indices = [index for index, row in enumerate(rows) if predicate(row)]
        subset_rows = [rows[index] for index in indices]
        subset_y_true = [y_true[index] for index in indices]
        subset_probabilities = [probabilities[index] for index in indices]
        summary[name] = _operating_metrics_without_slices(
            subset_y_true,
            subset_probabilities,
            rows=subset_rows,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
    return summary


def _operating_metrics_without_slices(
    y_true: list[int],
    probabilities: list[float],
    *,
    rows: list[dict[str, Any]],
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    auto_predictions = [
        1
        if confidence_band_for_row(
            row,
            probability,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        == "high"
        else 0
        for probability, row in zip(probabilities, rows, strict=True)
    ]
    review_predictions = [1 if probability >= low_threshold else 0 for probability in probabilities]
    queue_counts = Counter(
        confidence_band_for_row(
            row,
            probability,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        for probability, row in zip(probabilities, rows, strict=True)
    )
    total_count = len(y_true)
    positive_count = Counter(y_true).get(1, 0)
    return {
        "auto_band": _classification_metrics(y_true, auto_predictions),
        "review_queue": _classification_metrics(y_true, review_predictions),
        "queue_counts": {
            "high": queue_counts.get("high", 0),
            "borderline": queue_counts.get("borderline", 0),
            "low": queue_counts.get("low", 0),
        },
        "queue_rates": {
            "auto_rate": _safe_rate(queue_counts.get("high", 0), total_count),
            "review_rate": _safe_rate(
                queue_counts.get("high", 0) + queue_counts.get("borderline", 0),
                total_count,
            ),
            "borderline_rate": _safe_rate(queue_counts.get("borderline", 0), total_count),
        },
        "positive_prevalence": _safe_rate(positive_count, total_count),
        "positive_count": positive_count,
        "total_count": total_count,
    }


def _classification_metrics(y_true: list[int], predictions: list[int]) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(sum(predictions)),
        "support": Counter(y_true).get(1, 0),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _balanced_class_weights(labels: list[int]) -> list[float]:
    counts = Counter(labels)
    total = len(labels)
    classes = (0, 1)
    return [float(total / (len(classes) * counts[label])) for label in classes]


def _production_gate_summary() -> dict[str, float | int]:
    return {
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "minimum_high_confidence_test_predictions": DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS,
    }


def _production_ready_status(
    *,
    calibration: CalibrationResult,
    thresholds: DecisionThresholds,
    high_confidence_precision: float,
    high_confidence_predictions: int,
) -> tuple[bool, str | None]:
    blocked_reason = _production_ready_blocked_reason(
        calibration=calibration,
        thresholds=thresholds,
        high_confidence_precision=high_confidence_precision,
        high_confidence_predictions=high_confidence_predictions,
    )
    return blocked_reason is None, blocked_reason


def _production_ready_blocked_reason(
    *,
    calibration: CalibrationResult,
    thresholds: DecisionThresholds,
    high_confidence_precision: float,
    high_confidence_predictions: int,
) -> str | None:
    if (
        calibration.available
        and thresholds.high_threshold_selection.production_ready
        and high_confidence_precision >= DEFAULT_HIGH_PRECISION_TARGET
        and high_confidence_predictions >= DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS
    ):
        return None
    if not calibration.available:
        return "calibration_unavailable"
    if not thresholds.high_threshold_selection.production_ready:
        return "high_precision_target_not_met_on_calibration"
    if high_confidence_precision < DEFAULT_HIGH_PRECISION_TARGET:
        return "high_precision_target_not_met_on_test"
    if high_confidence_predictions < DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS:
        return "insufficient_high_confidence_test_predictions"
    return "production_gate_unsatisfied"


def _metrics_reference() -> dict[str, str]:
    return {
        "auto_band": "Posts with calibrated score >= high_threshold. This is the strict bucket for precision-first automation.",
        "review_queue": "Posts with calibrated score >= low_threshold. This combines high + borderline and behaves like a human-review queue.",
        "queue_counts": "Count of held-out test posts in each confidence band: high, borderline, low.",
        "queue_rates": "Share of the held-out test set in the auto band, review queue, and borderline band.",
        "positive_prevalence": "Fraction of held-out test posts labeled askseattle.",
        "slice_metrics": "Per-cohort operating metrics for post type, low-text posts, and sparse-media posts on the held-out test slice.",
        "production_gate": "A run is production-ready only if calibration is available, the calibration slice can hit the high-precision target, the held-out test auto band also clears that precision target, and the held-out test auto band contains at least the minimum number of high-confidence predictions.",
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
        "split_seed": split.split_seed,
        "evaluation_subreddit": split.evaluation_subreddit,
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
