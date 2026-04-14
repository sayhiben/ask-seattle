from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from ask_seattle import __version__
from ask_seattle.data import (
    DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
    LabeledPost,
    is_sparse_media_post,
    post_text,
    prepare_training_posts,
)
from ask_seattle.model import (
    CalibrationResult,
    DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
    DEFAULT_CHAR_WEIGHT,
    DEFAULT_EXTRA_WORD_STOPWORDS,
    DEFAULT_MIN_HIGH_CONFIDENCE_CALIBRATION_PREDICTIONS,
    DEFAULT_REVIEW_PRECISION_TARGET,
    DEFAULT_METADATA_WEIGHT,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TFIDF_CONFIG_VERSION,
    DEFAULT_TFIDF_URL_NORMALIZATION,
    DEFAULT_TFIDF_STRIP_URLS,
    DatasetSplit,
    DecisionThresholds,
    ThresholdSelection,
    apply_probability_calibrator,
    build_inference_row,
    causal_lm_prompt_for_row,
    confidence_band_for_row,
    evaluate_decision_policy,
    fit_sigmoid_calibrator,
    positive_probabilities,
    load_model,
    save_model,
    score_rows,
    select_decision_thresholds,
    split_labeled_posts,
    threshold_sweep,
    tfidf_feature_audit,
    ensure_transformer_custom_code_support,
    transformer_load_options,
    train_model,
)

DEFAULT_HIGH_PRECISION_TARGET = 0.95
DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS = 5
DEFAULT_CALIBRATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_SECONDARY_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_SEMANTIC_TERTIARY_MODEL_ID = "jinaai/jina-embeddings-v5-text-small-classification"
DEFAULT_TRANSFORMER_MODEL_ID = "microsoft/deberta-v3-small"
DEFAULT_TRANSFORMER_SECONDARY_MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_TRANSFORMER_TERTIARY_MODEL_ID = "chandar-lab/NeoBERT"
DEFAULT_TRANSFORMER_QUATERNARY_MODEL_ID = "answerdotai/ModernBERT-large"
DEFAULT_CAUSAL_LM_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_MAX_SLICE_POSITIVE_WEIGHT = 2.0
DEFAULT_TFIDF_REVIEW_PRECISION_TARGET = 0.70
DEFAULT_BENCHMARK_SEED_SWEEP = (13, 21, 34)
DEFAULT_BENCHMARK_SEED_MODELS = (
    "semantic_qwen3_embedding_0_6b",
    "transformer_modernbert_base",
    "transformer_neobert",
    "transformer_modernbert_large",
    "causal_lm_qwen3_1_7b_lora",
)
SPARSE_MEDIA_ACTIVE_TRAIN_POSITIVES = 10
SPARSE_MEDIA_ACTIVE_TEST_POSITIVES = 5
MPS_PROACTIVE_AVAILABLE_MEMORY_FLOOR_BYTES = 8 * 1024**3
MPS_PROACTIVE_AVAILABLE_MEMORY_DROP_BYTES = 12 * 1024**3
LOGGER = logging.getLogger("ask_seattle.training")


@dataclass(frozen=True)
class VariantConfig:
    name: str
    extra_word_stopwords: frozenset[str]
    char_weight: float
    metadata_weight: float = DEFAULT_METADATA_WEIGHT
    tfidf_config_version: str = DEFAULT_TFIDF_CONFIG_VERSION
    normalize_urls: bool = DEFAULT_TFIDF_URL_NORMALIZATION
    strip_urls: bool = DEFAULT_TFIDF_STRIP_URLS
    review_precision_target: float = DEFAULT_TFIDF_REVIEW_PRECISION_TARGET
    min_df: int | None = None
    classifier_c: float = 1.0
    classifier_class_weight: str | dict[int, float] | None = "balanced"
    max_slice_positive_weight: float = DEFAULT_MAX_SLICE_POSITIVE_WEIGHT


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


@dataclass(frozen=True)
class SemanticModelConfig:
    name: str
    display_name: str
    model_id: str
    backend: str
    prompt_modes: tuple[str, ...]
    normalize_embeddings: tuple[bool, ...]
    logistic_c_values: tuple[float, ...]
    title_weight_values: tuple[float, ...] = (1.0,)
    body_weight_values: tuple[float, ...] = (1.0,)
    config_version: str = "v1"
    encode_batch_size: int = 16
    prompt_prefix: str = ""
    short_prompt_prefix: str = ""
    pooling: str = "mean"
    feature_layout: str = "title_body_metadata_v1"


@dataclass(frozen=True)
class SuiteModelSpec:
    name: str
    display_name: str
    family: str
    runner: Callable[..., dict[str, Any]]
    kwargs: dict[str, Any]


class OptionalModelDependencyError(RuntimeError):
    pass


class MPSFallbackRequested(RuntimeError):
    pass


def _representation_config_for_split(split: DatasetSplit) -> dict[str, bool]:
    sparse_train_positive = sum(
        1
        for post in split.train
        if post.label == 1 and is_sparse_media_post(post_type=post.post_type, selftext=post.selftext)
    )
    sparse_test_positive = sum(
        1
        for post in split.test
        if post.label == 1 and is_sparse_media_post(post_type=post.post_type, selftext=post.selftext)
    )
    include_sparse_media_token = (
        sparse_train_positive >= SPARSE_MEDIA_ACTIVE_TRAIN_POSITIVES
        and sparse_test_positive >= SPARSE_MEDIA_ACTIVE_TEST_POSITIVES
    )
    return {
        "include_sparse_media_token": include_sparse_media_token,
        "include_image_low_text_tokens": DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_environment_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            "accelerate": _package_version("accelerate"),
            "datasets": _package_version("datasets"),
            "peft": _package_version("peft"),
            "scikit_learn": _package_version("scikit-learn"),
            "sentence_transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "trl": _package_version("trl"),
        },
    }


def _input_data_metadata(input_path: str | Path) -> dict[str, Any]:
    resolved = Path(input_path).resolve()
    return {
        "path": str(resolved),
        "fingerprint": _file_sha256(resolved),
    }


def _enable_cuda_tf32(torch_module: Any) -> dict[str, Any]:
    summary = {
        "device": _torch_runtime_device(torch_module),
        "float32_matmul_precision": None,
        "matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
        "enabled": False,
    }
    if summary["device"] != "cuda":
        return summary
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("high")
        summary["float32_matmul_precision"] = "high"
    if hasattr(torch_module.backends, "cuda") and hasattr(torch_module.backends.cuda, "matmul"):
        torch_module.backends.cuda.matmul.allow_tf32 = True
        summary["matmul_allow_tf32"] = bool(torch_module.backends.cuda.matmul.allow_tf32)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.allow_tf32 = True
        summary["cudnn_allow_tf32"] = bool(torch_module.backends.cudnn.allow_tf32)
    summary["enabled"] = True
    LOGGER.info(
        "enabled cuda tf32 float32_matmul_precision=%s matmul_allow_tf32=%s cudnn_allow_tf32=%s",
        summary["float32_matmul_precision"] or "unchanged",
        summary["matmul_allow_tf32"],
        summary["cudnn_allow_tf32"],
    )
    return summary


def _portable_artifact_reference(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return path.name


def _constraint_metrics_template() -> dict[str, dict[str, float | int | bool | None]]:
    return {
        "auto_recall_at_precision_95": {
            "precision_target": 0.95,
            "threshold": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_positive": 0,
            "support": 0,
            "target_met": False,
        },
        "review_recall_at_precision_75": {
            "precision_target": 0.75,
            "threshold": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_positive": 0,
            "support": 0,
            "target_met": False,
        },
    }


def train_model_bundle(
    posts: list[LabeledPost],
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    prepared_data_summary: dict[str, int] | None = None,
    evaluate_on_test: bool = True,
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
        evaluate_on_test=evaluate_on_test,
    )


def train_model_bundle_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    evaluate_on_test: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_data = _input_data_metadata(input_path)
    LOGGER.info(
        "starting tfidf retrain input=%s output_dir=%s split_strategy=%s split_seed=%s evaluation_subreddit=%s",
        str(input_path),
        str(output_dir),
        split_strategy,
        split_seed,
        evaluation_subreddit or "all",
    )
    posts, prepared_data_summary = prepare_training_posts(input_path)
    LOGGER.info(
        "prepared reviewed labels records=%s deduped=%s",
        prepared_data_summary.get("loaded_records", 0),
        prepared_data_summary.get("training_records", len(posts)),
    )
    summary = train_model_bundle(
        posts,
        output_dir,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
        prepared_data_summary=prepared_data_summary,
        evaluate_on_test=evaluate_on_test,
    )
    summary["input_data"] = input_data
    summary_path = Path(output_dir) / "training_summary.json"
    if summary_path.exists():
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info(
        "completed tfidf retrain artifact=%s elapsed=%s",
        summary["artifact_path"],
        _format_elapsed(time.perf_counter() - started_at),
    )
    return summary


def benchmark_model_variants_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_data = _input_data_metadata(input_path)
    LOGGER.info(
        "starting variant benchmark input=%s output_dir=%s split_strategy=%s split_seed=%s evaluation_subreddit=%s",
        str(input_path),
        str(output_dir),
        split_strategy,
        split_seed,
        evaluation_subreddit or "all",
    )
    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    suite_input_path = benchmark_dir.parent / "benchmark-suite" / "suite_input.json"
    reused_suite_manifest = suite_input_path.exists()
    if reused_suite_manifest:
        split, prepared_data_summary = _load_suite_input_manifest(suite_input_path)
        LOGGER.info(
            "variant benchmark reusing suite manifest path=%s train=%s calibration=%s test=%s",
            str(suite_input_path),
            len(split.train),
            len(split.calibration),
            len(split.test),
        )
    else:
        posts, prepared_data_summary = prepare_training_posts(input_path)
        split = split_labeled_posts(
            posts,
            calibration_size=DEFAULT_CALIBRATION_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            split_strategy=split_strategy,
            split_seed=split_seed,
            evaluation_subreddit=evaluation_subreddit,
        )
    variants = [
        VariantConfig(
            name="legacy_baseline",
            extra_word_stopwords=frozenset(),
            char_weight=0.5,
            normalize_urls=False,
            strip_urls=False,
            review_precision_target=DEFAULT_TFIDF_REVIEW_PRECISION_TARGET,
            classifier_c=1.0,
            min_df=2,
        ),
        VariantConfig(
            name="recommended",
            extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
            char_weight=DEFAULT_CHAR_WEIGHT,
            metadata_weight=DEFAULT_METADATA_WEIGHT,
            normalize_urls=True,
            strip_urls=DEFAULT_TFIDF_STRIP_URLS,
            review_precision_target=DEFAULT_TFIDF_REVIEW_PRECISION_TARGET,
            classifier_c=1.0,
            min_df=None,
        ),
    ]
    for classifier_c in (1.0, 4.0):
        for char_weight in (0.1, 0.25):
            for metadata_weight in (0.4, 0.5, 0.75):
                for min_df in (3, 5):
                    for max_slice_positive_weight in (2.0, 3.0):
                        variants.append(
                            VariantConfig(
                                name=(
                                    f"grid_c{str(classifier_c).replace('.', '_')}"
                                    f"_char{str(char_weight).replace('.', '_')}"
                                    f"_meta{str(metadata_weight).replace('.', '_')}"
                                    f"_mindf{min_df}"
                                    f"_slice{str(max_slice_positive_weight).replace('.', '_')}"
                                ),
                                extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
                                char_weight=char_weight,
                                metadata_weight=metadata_weight,
                                normalize_urls=True,
                                strip_urls=False,
                                review_precision_target=DEFAULT_TFIDF_REVIEW_PRECISION_TARGET,
                                classifier_c=classifier_c,
                                min_df=min_df,
                                max_slice_positive_weight=max_slice_positive_weight,
                            )
                        )
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_started_at = time.perf_counter()
        LOGGER.info(
            "variant benchmark start variant=%s stopwords=%s char_weight=%s normalize_urls=%s strip_urls=%s",
            variant.name,
            len(variant.extra_word_stopwords),
            variant.char_weight,
            variant.normalize_urls,
            variant.strip_urls,
        )
        variant_dir = benchmark_dir / variant.name
        summary = _train_model_bundle_for_split(
            split=split,
            output_dir=variant_dir,
            variant=variant,
            prepared_data_summary=prepared_data_summary,
        )
        LOGGER.info(
            "variant benchmark complete variant=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
            variant.name,
            float(summary["operating_metrics"]["auto_band"]["precision"]),
            float(summary["operating_metrics"]["auto_band"]["recall"]),
            float(summary["operating_metrics"]["review_queue"]["precision"]),
            float(summary["operating_metrics"]["review_queue"]["recall"]),
            _format_elapsed(time.perf_counter() - variant_started_at),
        )
        results.append(
            {
                "name": variant.name,
                "artifact_path": summary["artifact_path"],
                "summary_path": str(variant_dir / "training_summary.json"),
                "extra_word_stopwords": sorted(variant.extra_word_stopwords),
                "char_weight": variant.char_weight,
                "metadata_weight": variant.metadata_weight,
                "tfidf_config_version": variant.tfidf_config_version,
                "normalize_urls": variant.normalize_urls,
                "strip_urls": variant.strip_urls,
                "review_precision_target": variant.review_precision_target,
                "classifier_c": variant.classifier_c,
                "classifier_class_weight": variant.classifier_class_weight,
                "min_df": variant.min_df,
                "max_slice_positive_weight": variant.max_slice_positive_weight,
                "production_ready": summary["production_ready"],
                "production_ready_blocked_reason": summary["production_ready_blocked_reason"],
                "metrics": summary["metrics"],
                "operating_metrics": summary["operating_metrics"],
                "production_gate": summary["production_gate"],
                "threshold_policy": summary["threshold_policy"],
                "constraint_metrics": summary.get("constraint_metrics", _constraint_metrics_template()),
                "ranking_metrics": summary.get("ranking_metrics", {"pr_auc": 0.0}),
                "feature_audit": summary["feature_audit"],
            }
        )

    aggregate = {
        "version": __version__,
        "benchmark_output_dir": str(benchmark_dir),
        "input_data": input_data,
        "runtime_environment": _runtime_environment_metadata(),
        "evaluation_subreddit": split.evaluation_subreddit,
        "reused_suite_manifest": reused_suite_manifest,
        "production_gate": _production_gate_summary(),
        "prepared_data": prepared_data_summary,
        "suite_input_path": str(suite_input_path) if reused_suite_manifest else None,
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
    LOGGER.info(
        "variant benchmark summary written path=%s elapsed=%s",
        str(benchmark_dir / "variant_benchmark_summary.json"),
        _format_elapsed(time.perf_counter() - started_at),
    )
    return aggregate


def retrain_model_suite_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    semantic_model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    semantic_secondary_model_id: str = DEFAULT_SEMANTIC_SECONDARY_MODEL_ID,
    semantic_tertiary_model_id: str = DEFAULT_SEMANTIC_TERTIARY_MODEL_ID,
    transformer_model_id: str = DEFAULT_TRANSFORMER_MODEL_ID,
    transformer_secondary_model_id: str = DEFAULT_TRANSFORMER_SECONDARY_MODEL_ID,
    transformer_tertiary_model_id: str = DEFAULT_TRANSFORMER_TERTIARY_MODEL_ID,
    transformer_quaternary_model_id: str = DEFAULT_TRANSFORMER_QUATERNARY_MODEL_ID,
    causal_lm_model_id: str = DEFAULT_CAUSAL_LM_MODEL_ID,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_data = _input_data_metadata(input_path)
    LOGGER.info(
        "starting suite retrain input=%s output_dir=%s split_strategy=%s split_seed=%s evaluation_subreddit=%s",
        str(input_path),
        str(output_dir),
        split_strategy,
        split_seed,
        evaluation_subreddit or "all",
    )
    split, prepared_data_summary, suite_input_path = _prepare_suite_input_from_labels(
        input_path=input_path,
        output_dir=output_dir,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )
    benchmark_dir = Path(output_dir)
    specs = _suite_model_specs(
        semantic_model_id=semantic_model_id,
        semantic_secondary_model_id=semantic_secondary_model_id,
        semantic_tertiary_model_id=semantic_tertiary_model_id,
        transformer_model_id=transformer_model_id,
        transformer_secondary_model_id=transformer_secondary_model_id,
        transformer_tertiary_model_id=transformer_tertiary_model_id,
        transformer_quaternary_model_id=transformer_quaternary_model_id,
        causal_lm_model_id=causal_lm_model_id,
    )
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        model_started_at = time.perf_counter()
        existing_summary = _load_resumable_suite_summary(
            spec=spec,
            output_dir=benchmark_dir / spec.name,
            split=split,
            prepared_data_summary=prepared_data_summary,
            suite_manifest_path=suite_input_path,
        )
        if existing_summary is not None:
            LOGGER.info(
                "suite retrain model reused index=%s/%s name=%s family=%s elapsed=%s",
                index,
                len(specs),
                spec.name,
                spec.family,
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            results.append(_suite_training_entry_from_summary(spec, existing_summary, result_source="reused"))
            continue
        LOGGER.info(
            "suite retrain model start index=%s/%s name=%s family=%s display_name=%s",
            index,
            len(specs),
            spec.name,
            spec.family,
            spec.display_name,
        )
        _clear_torch_memory()
        try:
            summary = spec.runner(
                split=split,
                output_dir=benchmark_dir / spec.name,
                prepared_data_summary=prepared_data_summary,
                evaluate_on_test=False,
                **spec.kwargs,
            )
        except OptionalModelDependencyError as exc:
            LOGGER.warning(
                "suite retrain model unavailable name=%s family=%s error=%s elapsed=%s",
                spec.name,
                spec.family,
                str(exc),
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            results.append(_suite_skipped_entry(spec, reason="dependency_unavailable", error=str(exc)))
        except Exception:
            LOGGER.exception(
                "suite retrain model failed name=%s family=%s elapsed=%s",
                spec.name,
                spec.family,
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            raise
        else:
            summary = _stamp_suite_model_summary(
                spec=spec,
                output_dir=benchmark_dir / spec.name,
                summary=summary,
                suite_manifest_path=suite_input_path,
                split=split,
                prepared_data_summary=prepared_data_summary,
            )
            LOGGER.info(
                "suite retrain model complete name=%s family=%s artifact=%s elapsed=%s",
                spec.name,
                spec.family,
                summary["artifact_path"],
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            results.append(_suite_training_entry_from_summary(spec, summary, result_source="trained"))
        finally:
            _clear_torch_memory()

    aggregate = {
        "version": __version__,
        "benchmark_output_dir": str(benchmark_dir),
        "suite_input_path": str(suite_input_path),
        "evaluation_subreddit": split.evaluation_subreddit,
        "input_data": input_data,
        "runtime_environment": _runtime_environment_metadata(),
        "prepared_data": prepared_data_summary,
        "split": _split_summary(split),
        "models": results,
    }
    (benchmark_dir / "suite_training_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "suite retrain summary written path=%s elapsed=%s",
        str(benchmark_dir / "suite_training_summary.json"),
        _format_elapsed(time.perf_counter() - started_at),
    )
    return aggregate


def retrain_all_from_labels(
    input_path: str | Path,
    *,
    operational_output_dir: str | Path,
    benchmark_output_dir: str | Path,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    semantic_model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    semantic_secondary_model_id: str = DEFAULT_SEMANTIC_SECONDARY_MODEL_ID,
    semantic_tertiary_model_id: str = DEFAULT_SEMANTIC_TERTIARY_MODEL_ID,
    transformer_model_id: str = DEFAULT_TRANSFORMER_MODEL_ID,
    transformer_secondary_model_id: str = DEFAULT_TRANSFORMER_SECONDARY_MODEL_ID,
    transformer_tertiary_model_id: str = DEFAULT_TRANSFORMER_TERTIARY_MODEL_ID,
    transformer_quaternary_model_id: str = DEFAULT_TRANSFORMER_QUATERNARY_MODEL_ID,
    causal_lm_model_id: str = DEFAULT_CAUSAL_LM_MODEL_ID,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_data = _input_data_metadata(input_path)
    LOGGER.info(
        "starting full retrain input=%s operational_output_dir=%s benchmark_output_dir=%s split_strategy=%s split_seed=%s evaluation_subreddit=%s",
        str(input_path),
        str(operational_output_dir),
        str(benchmark_output_dir),
        split_strategy,
        split_seed,
        evaluation_subreddit or "all",
    )
    operational_summary = train_model_bundle_from_labels(
        input_path,
        operational_output_dir,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
        evaluate_on_test=False,
    )
    suite_summary = retrain_model_suite_from_labels(
        input_path,
        benchmark_output_dir,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
        semantic_model_id=semantic_model_id,
        semantic_secondary_model_id=semantic_secondary_model_id,
        semantic_tertiary_model_id=semantic_tertiary_model_id,
        transformer_model_id=transformer_model_id,
        transformer_secondary_model_id=transformer_secondary_model_id,
        transformer_tertiary_model_id=transformer_tertiary_model_id,
        transformer_quaternary_model_id=transformer_quaternary_model_id,
        causal_lm_model_id=causal_lm_model_id,
    )
    summary = {
        "version": __version__,
        "input_path": str(input_path),
        "input_data": input_data,
        "runtime_environment": _runtime_environment_metadata(),
        "operational_output_dir": str(operational_output_dir),
        "benchmark_output_dir": str(benchmark_output_dir),
        "split_strategy": split_strategy,
        "split_seed": split_seed,
        "evaluation_subreddit": evaluation_subreddit,
        "operational_model": operational_summary,
        "suite": suite_summary,
    }
    LOGGER.info("completed full retrain elapsed=%s", _format_elapsed(time.perf_counter() - started_at))
    return summary


def benchmark_model_suite_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = DEFAULT_SPLIT_SEED,
    evaluation_subreddit: str | None = None,
    semantic_model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    semantic_secondary_model_id: str = DEFAULT_SEMANTIC_SECONDARY_MODEL_ID,
    semantic_tertiary_model_id: str = DEFAULT_SEMANTIC_TERTIARY_MODEL_ID,
    transformer_model_id: str = DEFAULT_TRANSFORMER_MODEL_ID,
    transformer_secondary_model_id: str = DEFAULT_TRANSFORMER_SECONDARY_MODEL_ID,
    transformer_tertiary_model_id: str = DEFAULT_TRANSFORMER_TERTIARY_MODEL_ID,
    transformer_quaternary_model_id: str = DEFAULT_TRANSFORMER_QUATERNARY_MODEL_ID,
    causal_lm_model_id: str = DEFAULT_CAUSAL_LM_MODEL_ID,
    notes: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_data = _input_data_metadata(input_path)
    LOGGER.info(
        "starting benchmark suite evaluation input=%s output_dir=%s split_strategy=%s split_seed=%s evaluation_subreddit=%s",
        str(input_path),
        str(output_dir),
        split_strategy,
        split_seed,
        evaluation_subreddit or "all",
    )
    split, prepared_data_summary, suite_input_path = _prepare_suite_input_from_labels(
        input_path=input_path,
        output_dir=output_dir,
        split_strategy=split_strategy,
        split_seed=split_seed,
        evaluation_subreddit=evaluation_subreddit,
    )
    benchmark_dir = Path(output_dir)
    LOGGER.info(
        "benchmark suite manifest ready path=%s train=%s calibration=%s test=%s",
        str(suite_input_path),
        len(split.train),
        len(split.calibration),
        len(split.test),
    )

    results: list[dict[str, Any]] = []
    specs = _suite_model_specs(
        semantic_model_id=semantic_model_id,
        semantic_secondary_model_id=semantic_secondary_model_id,
        semantic_tertiary_model_id=semantic_tertiary_model_id,
        transformer_model_id=transformer_model_id,
        transformer_secondary_model_id=transformer_secondary_model_id,
        transformer_tertiary_model_id=transformer_tertiary_model_id,
        transformer_quaternary_model_id=transformer_quaternary_model_id,
        causal_lm_model_id=causal_lm_model_id,
    )
    for index, spec in enumerate(specs, start=1):
        model_started_at = time.perf_counter()
        existing_summary = _load_resumable_suite_summary(
            spec=spec,
            output_dir=benchmark_dir / spec.name,
            split=split,
            prepared_data_summary=prepared_data_summary,
            suite_manifest_path=suite_input_path,
        )
        if existing_summary is None:
            LOGGER.warning(
                "benchmark suite model skipped index=%s/%s name=%s family=%s reason=%s elapsed=%s",
                index,
                len(specs),
                spec.name,
                spec.family,
                "compatible trained artifact not found",
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            results.append(_suite_skipped_entry(spec, reason="not_trained", error="compatible trained artifact not found"))
            continue
        LOGGER.info(
            "benchmark suite model start index=%s/%s name=%s family=%s display_name=%s",
            index,
            len(specs),
            spec.name,
            spec.family,
            spec.display_name,
        )
        _clear_torch_memory()
        try:
            summary = _benchmark_existing_suite_model(
                spec=spec,
                split=split,
                prepared_data_summary=prepared_data_summary,
                output_dir=benchmark_dir / spec.name,
                trained_summary=existing_summary,
            )
        except OptionalModelDependencyError as exc:
            LOGGER.warning(
                "benchmark suite model skipped name=%s family=%s error=%s elapsed=%s",
                spec.name,
                spec.family,
                str(exc),
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            results.append(_suite_skipped_entry(spec, reason="dependency_unavailable", error=str(exc)))
        except Exception:
            LOGGER.exception(
                "benchmark suite model failed name=%s family=%s elapsed=%s",
                spec.name,
                spec.family,
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            raise
        else:
            LOGGER.info(
                "benchmark suite model complete name=%s family=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
                spec.name,
                spec.family,
                float(summary["operating_metrics"]["auto_band"]["precision"]),
                float(summary["operating_metrics"]["auto_band"]["recall"]),
                float(summary["operating_metrics"]["review_queue"]["precision"]),
                float(summary["operating_metrics"]["review_queue"]["recall"]),
                _format_elapsed(time.perf_counter() - model_started_at),
            )
            results.append(_suite_entry_from_summary(spec, summary, result_source="benchmarked"))
        finally:
            _clear_torch_memory()

    aggregate = {
        "version": __version__,
        "benchmark_output_dir": str(benchmark_dir),
        "suite_input_path": str(suite_input_path),
        "evaluation_subreddit": split.evaluation_subreddit,
        "input_data": input_data,
        "runtime_environment": _runtime_environment_metadata(),
        "benchmark_run": _benchmark_run_metadata(
            benchmark_dir=benchmark_dir,
            suite_input_path=suite_input_path,
            split=split,
            prepared_data_summary=prepared_data_summary,
            input_data=input_data,
            notes=notes,
        ),
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
    LOGGER.info(
        "benchmark suite summary written path=%s elapsed=%s",
        str(benchmark_dir / "benchmark_suite_summary.json"),
        _format_elapsed(time.perf_counter() - started_at),
    )
    _archive_benchmark_suite_summary(benchmark_dir=benchmark_dir, aggregate=aggregate)
    return aggregate


def benchmark_seed_sweep_from_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seeds: tuple[int, ...] = DEFAULT_BENCHMARK_SEED_SWEEP,
    evaluation_subreddit: str | None = None,
    model_names: tuple[str, ...] = DEFAULT_BENCHMARK_SEED_MODELS,
    semantic_model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
    semantic_secondary_model_id: str = DEFAULT_SEMANTIC_SECONDARY_MODEL_ID,
    semantic_tertiary_model_id: str = DEFAULT_SEMANTIC_TERTIARY_MODEL_ID,
    transformer_model_id: str = DEFAULT_TRANSFORMER_MODEL_ID,
    transformer_secondary_model_id: str = DEFAULT_TRANSFORMER_SECONDARY_MODEL_ID,
    transformer_tertiary_model_id: str = DEFAULT_TRANSFORMER_TERTIARY_MODEL_ID,
    transformer_quaternary_model_id: str = DEFAULT_TRANSFORMER_QUATERNARY_MODEL_ID,
    causal_lm_model_id: str = DEFAULT_CAUSAL_LM_MODEL_ID,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    benchmark_dir = Path(output_dir)
    sweep_dir = benchmark_dir / "seed_sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    posts, prepared_data_summary = prepare_training_posts(input_path)
    selected_seeds = tuple(dict.fromkeys(int(seed) for seed in split_seeds))
    selected_model_names = tuple(dict.fromkeys(str(name) for name in model_names))
    input_data = _input_data_metadata(input_path)
    specs = _suite_model_specs(
        semantic_model_id=semantic_model_id,
        semantic_secondary_model_id=semantic_secondary_model_id,
        semantic_tertiary_model_id=semantic_tertiary_model_id,
        transformer_model_id=transformer_model_id,
        transformer_secondary_model_id=transformer_secondary_model_id,
        transformer_tertiary_model_id=transformer_tertiary_model_id,
        transformer_quaternary_model_id=transformer_quaternary_model_id,
        causal_lm_model_id=causal_lm_model_id,
    )
    selected_specs = _select_seed_sweep_specs(specs, model_names=selected_model_names)
    LOGGER.info(
        "starting benchmark seed sweep input=%s output_dir=%s split_strategy=%s seeds=%s evaluation_subreddit=%s models=%s",
        str(input_path),
        str(output_dir),
        split_strategy,
        list(selected_seeds),
        evaluation_subreddit or "all",
        list(selected_model_names),
    )
    runs: list[dict[str, Any]] = []
    for seed in selected_seeds:
        split = split_labeled_posts(
            posts,
            calibration_size=DEFAULT_CALIBRATION_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            split_strategy=split_strategy,
            split_seed=seed,
            evaluation_subreddit=evaluation_subreddit,
        )
        seed_result_dir = sweep_dir / f"seed_{seed}"
        seed_models: list[dict[str, Any]] = []
        for spec in selected_specs:
            model_started_at = time.perf_counter()
            LOGGER.info(
                "benchmark seed sweep model start seed=%s name=%s family=%s display_name=%s",
                seed,
                spec.name,
                spec.family,
                spec.display_name,
            )
            _clear_torch_memory()
            try:
                summary = spec.runner(
                    split=split,
                    output_dir=seed_result_dir / spec.name,
                    prepared_data_summary=prepared_data_summary,
                    evaluate_on_test=True,
                    **spec.kwargs,
                )
            finally:
                _clear_torch_memory()
            seed_models.append(_suite_entry_from_summary(spec, summary, result_source="seed_sweep"))
            LOGGER.info(
                "benchmark seed sweep model complete seed=%s name=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
                seed,
                spec.name,
                float(summary["operating_metrics"]["auto_band"]["precision"]),
                float(summary["operating_metrics"]["auto_band"]["recall"]),
                float(summary["operating_metrics"]["review_queue"]["precision"]),
                float(summary["operating_metrics"]["review_queue"]["recall"]),
                _format_elapsed(time.perf_counter() - model_started_at),
            )
        runs.append(
            {
                "seed": seed,
                "split": _split_summary(split),
                "models": seed_models,
            }
        )
    aggregate = {
        "version": __version__,
        "benchmark_output_dir": str(benchmark_dir),
        "output_path": str((sweep_dir / "seed_sweep_summary.json").resolve()),
        "prepared_data": prepared_data_summary,
        "input_data": input_data,
        "runtime_environment": _runtime_environment_metadata(),
        "selected_models": list(selected_model_names),
        "split_strategy": split_strategy,
        "evaluation_subreddit": evaluation_subreddit,
        "split_seeds": list(selected_seeds),
        "seed_runs": runs,
        "model_aggregates": _benchmark_seed_sweep_aggregates(
            runs,
            model_names=selected_model_names,
        ),
    }
    (sweep_dir / "seed_sweep_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "benchmark seed sweep summary written path=%s elapsed=%s",
        str(sweep_dir / "seed_sweep_summary.json"),
        _format_elapsed(time.perf_counter() - started_at),
    )
    return aggregate


def _select_thresholds_or_default(
    y_calibration: list[int],
    probabilities: list[float],
    *,
    high_precision_target: float,
    review_precision_target: float,
    calibration: CalibrationResult,
) -> DecisionThresholds:
    if calibration.available:
        return select_decision_thresholds(
            y_calibration,
            probabilities,
            auto_precision_target=high_precision_target,
            review_precision_target=review_precision_target,
            minimum_high_confidence_calibration_predictions=DEFAULT_MIN_HIGH_CONFIDENCE_CALIBRATION_PREDICTIONS,
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
        minimum_high_confidence_calibration_predictions=DEFAULT_MIN_HIGH_CONFIDENCE_CALIBRATION_PREDICTIONS,
        high_threshold_fallback_used=False,
    )


def _suite_model_specs(
    *,
    semantic_model_id: str,
    semantic_secondary_model_id: str,
    semantic_tertiary_model_id: str,
    transformer_model_id: str,
    transformer_secondary_model_id: str,
    transformer_tertiary_model_id: str,
    transformer_quaternary_model_id: str,
    causal_lm_model_id: str,
) -> list[SuiteModelSpec]:
    return [
        SuiteModelSpec(
            name="tfidf_recommended",
            display_name="TF-IDF",
            family="tfidf",
            runner=_train_model_bundle_for_split,
            kwargs={
                "variant": VariantConfig(
                    name="recommended",
                    extra_word_stopwords=DEFAULT_EXTRA_WORD_STOPWORDS,
                    char_weight=DEFAULT_CHAR_WEIGHT,
                    metadata_weight=DEFAULT_METADATA_WEIGHT,
                    normalize_urls=DEFAULT_TFIDF_URL_NORMALIZATION,
                    strip_urls=DEFAULT_TFIDF_STRIP_URLS,
                    review_precision_target=DEFAULT_TFIDF_REVIEW_PRECISION_TARGET,
                    classifier_c=1.0,
                ),
            },
        ),
        SuiteModelSpec(
            name="semantic_minilm_tuned",
            display_name="Semantic MiniLM",
            family="semantic_embedding",
            runner=_train_semantic_embedding_bundle_for_split,
            kwargs={
                "config": SemanticModelConfig(
                    name="semantic_minilm_tuned",
                    display_name="Semantic MiniLM",
                    model_id=semantic_model_id,
                    backend="sentence_transformers",
                    config_version="v3_title_body_metadata_weighted",
                    prompt_modes=("plain", "task_prefix", "short_task_prefix"),
                    normalize_embeddings=(False, True),
                    logistic_c_values=(4.0, 16.0, 64.0),
                    title_weight_values=(1.0, 1.5, 2.0),
                    body_weight_values=(1.0, 0.75),
                    encode_batch_size=16,
                    prompt_prefix="Classify Reddit post intent:",
                    short_prompt_prefix="Classify askseattle intent.",
                    pooling="sentence_transformers",
                ),
            },
        ),
        SuiteModelSpec(
            name="semantic_qwen3_embedding_0_6b",
            display_name="Semantic Qwen3-Embedding",
            family="semantic_embedding",
            runner=_train_semantic_embedding_bundle_for_split,
            kwargs={
                "config": SemanticModelConfig(
                    name="semantic_qwen3_embedding_0_6b",
                    display_name="Semantic Qwen3-Embedding",
                    model_id=semantic_secondary_model_id,
                    backend="hf_embedding",
                    config_version="v3_title_body_metadata_weighted",
                    prompt_modes=("plain", "short_task_prefix"),
                    normalize_embeddings=(False, True),
                    logistic_c_values=(4.0, 8.0, 16.0),
                    title_weight_values=(1.0, 1.5, 2.0),
                    body_weight_values=(1.0, 0.75),
                    encode_batch_size=8,
                    prompt_prefix="Instruct: classify the Reddit post as askseattle or not_askseattle.\nQuery:",
                    short_prompt_prefix="Classify askseattle intent.",
                    pooling="last_token",
                ),
            },
        ),
        SuiteModelSpec(
            name="semantic_jina_embeddings_v5_text_small_classification",
            display_name="Semantic Jina v5 Text Small Classification",
            family="semantic_embedding",
            runner=_train_semantic_embedding_bundle_for_split,
            kwargs={
                "config": SemanticModelConfig(
                    name="semantic_jina_embeddings_v5_text_small_classification",
                    display_name="Semantic Jina v5 Text Small Classification",
                    model_id=semantic_tertiary_model_id,
                    backend="hf_embedding",
                    config_version="v5_title_body_metadata_weighted_jina_document_component",
                    prompt_modes=("plain", "short_task_prefix", "jina_document_component"),
                    normalize_embeddings=(False, True),
                    logistic_c_values=(4.0, 16.0, 64.0),
                    title_weight_values=(1.0, 1.5, 2.0),
                    body_weight_values=(1.0, 0.75),
                    encode_batch_size=8,
                    prompt_prefix="Document:",
                    short_prompt_prefix="Classify askseattle intent.",
                    pooling="last_token",
                ),
            },
        ),
        SuiteModelSpec(
            name="transformer_deberta_v3_small",
            display_name="Transformer DeBERTa-v3-small",
            family="transformer_sequence_classifier",
            runner=_train_transformer_bundle_for_split,
            kwargs={
                "model_id": transformer_model_id,
                "display_name": "Transformer DeBERTa-v3-small",
                "config_version": "v3_pr_auc_precision_profiles",
            },
        ),
        SuiteModelSpec(
            name="transformer_modernbert_base",
            display_name="Transformer ModernBERT-base",
            family="transformer_sequence_classifier",
            runner=_train_transformer_bundle_for_split,
            kwargs={
                "model_id": transformer_secondary_model_id,
                "display_name": "Transformer ModernBERT-base",
                "config_version": "v4_pr_auc_precision_grid",
            },
        ),
        SuiteModelSpec(
            name="transformer_neobert",
            display_name="Transformer NeoBERT",
            family="transformer_sequence_classifier",
            runner=_train_transformer_bundle_for_split,
            kwargs={
                "model_id": transformer_tertiary_model_id,
                "display_name": "Transformer NeoBERT",
                "config_version": "v5_pr_auc_precision_grid_remote_code",
            },
        ),
        SuiteModelSpec(
            name="transformer_modernbert_large",
            display_name="Transformer ModernBERT-large",
            family="transformer_sequence_classifier",
            runner=_train_transformer_bundle_for_split,
            kwargs={
                "model_id": transformer_quaternary_model_id,
                "display_name": "Transformer ModernBERT-large",
                "config_version": "v4_pr_auc_precision_grid",
            },
        ),
        SuiteModelSpec(
            name="causal_lm_qwen3_1_7b_lora",
            display_name="Decoder Qwen3-1.7B LoRA",
            family="causal_lm_classifier",
            runner=_train_causal_lm_bundle_for_split,
            kwargs={
                "model_id": causal_lm_model_id,
                "display_name": "Decoder Qwen3-1.7B LoRA",
                "prompt_template_version": (
                    DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
                    "v4_image_low_text",
                ),
                "config_version": "v3_prompt_grid_precision_selection",
            },
        ),
    ]


def _prepare_suite_input_from_labels(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    split_strategy: str,
    split_seed: int,
    evaluation_subreddit: str | None,
) -> tuple[DatasetSplit, dict[str, int], Path]:
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
    suite_input_path = benchmark_dir / "suite_input.json"
    _write_suite_input_manifest(
        split=split,
        prepared_data_summary=prepared_data_summary,
        output_path=suite_input_path,
    )
    loaded_split, loaded_prepared_data_summary = _load_suite_input_manifest(suite_input_path)
    return loaded_split, loaded_prepared_data_summary, suite_input_path


def _write_suite_input_manifest(
    *,
    split: DatasetSplit,
    prepared_data_summary: dict[str, int],
    output_path: Path,
) -> None:
    LOGGER.info(
        "writing suite manifest path=%s train=%s calibration=%s test=%s",
        str(output_path),
        len(split.train),
        len(split.calibration),
        len(split.test),
    )
    payload = {
        "version": __version__,
        "prepared_data": prepared_data_summary,
        "split": {
            "split_strategy": split.split_strategy,
            "split_seed": split.split_seed,
            "evaluation_subreddit": split.evaluation_subreddit,
            "excluded_for_time_split": split.excluded_for_time_split,
            "time_coverage": split.time_coverage,
        },
        "records": {
            "train": _serialize_posts(split.train),
            "calibration": _serialize_posts(split.calibration),
            "test": _serialize_posts(split.test),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_suite_input_manifest(path: Path) -> tuple[DatasetSplit, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or {}
    split_meta = payload.get("split") or {}
    split = DatasetSplit(
        train=[_post_from_suite_record(record) for record in records.get("train") or []],
        calibration=[_post_from_suite_record(record) for record in records.get("calibration") or []],
        test=[_post_from_suite_record(record) for record in records.get("test") or []],
        split_strategy=str(split_meta.get("split_strategy") or DEFAULT_SPLIT_STRATEGY),
        split_seed=split_meta.get("split_seed"),
        excluded_for_time_split=int(split_meta.get("excluded_for_time_split") or 0),
        time_coverage=split_meta.get("time_coverage"),
        evaluation_subreddit=split_meta.get("evaluation_subreddit"),
    )
    prepared_data = payload.get("prepared_data") or {}
    LOGGER.info(
        "loaded suite manifest path=%s train=%s calibration=%s test=%s",
        str(path),
        len(split.train),
        len(split.calibration),
        len(split.test),
    )
    return split, {str(key): int(value) for key, value in prepared_data.items()}


def _benchmark_run_metadata(
    *,
    benchmark_dir: Path,
    suite_input_path: Path,
    split: DatasetSplit,
    prepared_data_summary: dict[str, int],
    input_data: dict[str, Any],
    notes: str | None,
) -> dict[str, Any]:
    run_timestamp = datetime.now(UTC)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    clean_notes = str(notes).strip() if notes is not None and str(notes).strip() else None
    return {
        "run_id": run_id,
        "created_at": run_timestamp.isoformat().replace("+00:00", "Z"),
        "notes": clean_notes,
        "representation": _benchmark_representation(split=split, prepared_data_summary=prepared_data_summary),
        "input_data_fingerprint": input_data.get("fingerprint"),
        "suite_manifest_fingerprint": _file_sha256(suite_input_path),
        "latest_summary_path": str((benchmark_dir / "benchmark_suite_summary.json").resolve()),
    }


def _benchmark_representation(*, split: DatasetSplit, prepared_data_summary: dict[str, int]) -> str:
    split_label = split.split_strategy
    if split_label == "random_eval_subreddit":
        split_label = "random"
    split_text = f"{split_label} split"
    if split.split_seed is not None:
        split_text = f"{split_text} (seed {split.split_seed})"
    eval_text = (
        f"/r/{split.evaluation_subreddit} evaluation"
        if split.evaluation_subreddit
        else "all-subreddit evaluation"
    )
    return (
        f"{split_text}, {eval_text}, "
        f"{int(prepared_data_summary.get('training_records', 0))} prepared records, "
        f"{len(split.train)}/{len(split.calibration)}/{len(split.test)} train/calibration/test"
    )


def _archive_benchmark_suite_summary(*, benchmark_dir: Path, aggregate: dict[str, Any]) -> None:
    run_meta = aggregate.get("benchmark_run")
    if not isinstance(run_meta, dict):
        return
    run_id = run_meta.get("run_id")
    if not run_id:
        return
    history_dir = benchmark_dir / "history" / str(run_id)
    history_dir.mkdir(parents=True, exist_ok=True)
    archived_summary_path = history_dir / "benchmark_suite_summary.json"
    archived_summary = dict(aggregate)
    archived_summary["benchmark_run"] = dict(run_meta)
    archived_summary["benchmark_run"]["archived_summary_path"] = str(archived_summary_path.resolve())
    archived_summary_path.write_text(
        json.dumps(archived_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    index_entry = {
        "run_id": run_meta.get("run_id"),
        "created_at": run_meta.get("created_at"),
        "notes": run_meta.get("notes"),
        "representation": run_meta.get("representation"),
        "suite_manifest_fingerprint": run_meta.get("suite_manifest_fingerprint"),
        "summary_path": str(archived_summary_path.resolve()),
        "prepared_data": {
            "loaded": int((aggregate.get("prepared_data") or {}).get("loaded", 0)),
            "training_records": int((aggregate.get("prepared_data") or {}).get("training_records", 0)),
        },
        "split": {
            "train": int((aggregate.get("split") or {}).get("train", 0)),
            "calibration": int((aggregate.get("split") or {}).get("calibration", 0)),
            "test": int((aggregate.get("split") or {}).get("test", 0)),
            "split_strategy": (aggregate.get("split") or {}).get("split_strategy"),
            "split_seed": (aggregate.get("split") or {}).get("split_seed"),
            "evaluation_subreddit": (aggregate.get("split") or {}).get("evaluation_subreddit"),
        },
        "models": _benchmark_history_models_snapshot(aggregate.get("models")),
    }
    history_index_path = benchmark_dir / "benchmark_history.json"
    existing_runs: list[dict[str, Any]] = []
    if history_index_path.exists():
        try:
            existing_payload = json.loads(history_index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing_payload = {}
        maybe_runs = existing_payload.get("runs") if isinstance(existing_payload, dict) else None
        if isinstance(maybe_runs, list):
            existing_runs = [entry for entry in maybe_runs if isinstance(entry, dict)]
    existing_runs.append(index_entry)
    history_index_path.write_text(
        json.dumps({"runs": existing_runs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _benchmark_history_models_snapshot(models: Any) -> list[dict[str, Any]]:
    if not isinstance(models, list):
        return []
    snapshot: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        auto_band = model.get("operating_metrics", {}).get("auto_band", {}) if isinstance(model.get("operating_metrics"), dict) else {}
        review_queue = model.get("operating_metrics", {}).get("review_queue", {}) if isinstance(model.get("operating_metrics"), dict) else {}
        constraint_metrics = model.get("constraint_metrics") if isinstance(model.get("constraint_metrics"), dict) else {}
        ranking_metrics = model.get("ranking_metrics") if isinstance(model.get("ranking_metrics"), dict) else {}
        snapshot.append(
            {
                "name": model.get("name"),
                "display_name": model.get("display_name"),
                "status": model.get("status"),
                "production_ready": model.get("production_ready"),
                "production_ready_blocked_reason": model.get("production_ready_blocked_reason"),
                "auto_precision": auto_band.get("precision"),
                "auto_recall": auto_band.get("recall"),
                "review_precision": review_queue.get("precision"),
                "review_recall": review_queue.get("recall"),
                "auto_recall_at_precision_95": (constraint_metrics.get("auto_recall_at_precision_95") or {}).get("recall")
                if isinstance(constraint_metrics.get("auto_recall_at_precision_95"), dict)
                else None,
                "review_recall_at_precision_75": (constraint_metrics.get("review_recall_at_precision_75") or {}).get("recall")
                if isinstance(constraint_metrics.get("review_recall_at_precision_75"), dict)
                else None,
                "pr_auc": ranking_metrics.get("pr_auc"),
            }
        )
    return snapshot


def _select_seed_sweep_specs(specs: list[SuiteModelSpec], *, model_names: tuple[str, ...]) -> list[SuiteModelSpec]:
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in model_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown seed-sweep model names: {', '.join(missing)}")
    return [by_name[name] for name in model_names]


def _benchmark_seed_sweep_aggregates(
    runs: list[dict[str, Any]],
    *,
    model_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    metric_extractors: dict[str, tuple[str, ...]] = {
        "pr_auc": ("ranking_metrics", "pr_auc"),
        "auto_precision": ("operating_metrics", "auto_band", "precision"),
        "auto_recall": ("operating_metrics", "auto_band", "recall"),
        "review_precision": ("operating_metrics", "review_queue", "precision"),
        "review_recall": ("operating_metrics", "review_queue", "recall"),
        "auto_recall_at_precision_95": ("constraint_metrics", "auto_recall_at_precision_95", "recall"),
        "review_recall_at_precision_75": ("constraint_metrics", "review_recall_at_precision_75", "recall"),
    }
    aggregates: list[dict[str, Any]] = []
    for model_name in model_names:
        model_runs: list[dict[str, Any]] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            seed = int(run.get("seed"))
            for model in run.get("models") or []:
                if isinstance(model, dict) and model.get("name") == model_name:
                    model_runs.append({"seed": seed, "model": model})
                    break
        if not model_runs:
            continue
        sample = model_runs[0]["model"]
        aggregates.append(
            {
                "name": model_name,
                "display_name": sample.get("display_name"),
                "model_family": sample.get("model_family"),
                "model_id": sample.get("model_id"),
                "production_ready_rate": _safe_rate(
                    sum(1 for item in model_runs if bool(item["model"].get("production_ready"))),
                    len(model_runs),
                ),
                "metric_summary": {
                    metric_name: _seed_sweep_metric_summary(
                        [_nested_metric_value(item["model"], path) for item in model_runs]
                    )
                    for metric_name, path in metric_extractors.items()
                },
                "per_seed": [
                    {
                        "seed": item["seed"],
                        "production_ready": bool(item["model"].get("production_ready")),
                        "pr_auc": _nested_metric_value(item["model"], metric_extractors["pr_auc"]),
                        "auto_recall_at_precision_95": _nested_metric_value(
                            item["model"],
                            metric_extractors["auto_recall_at_precision_95"],
                        ),
                        "review_recall_at_precision_75": _nested_metric_value(
                            item["model"],
                            metric_extractors["review_recall_at_precision_75"],
                        ),
                    }
                    for item in model_runs
                ],
            }
        )
    return aggregates


def _nested_metric_value(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    return float(current)


def _seed_sweep_metric_summary(values: list[float | None]) -> dict[str, Any]:
    realized = [float(value) for value in values if value is not None]
    if not realized:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    array = np.asarray(realized, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _serialize_posts(posts: list[LabeledPost]) -> list[dict[str, Any]]:
    return [
        {
            "title": post.title,
            "selftext": post.selftext,
            "label": post.label,
            "post_id": post.post_id,
            "subreddit": post.subreddit,
            "permalink": post.permalink,
            "post_type": post.post_type,
            "content_domain": post.content_domain,
            "is_crosspost": post.is_crosspost,
            "created_utc": post.created_utc,
            "time_key": post.time_key,
            "time_source": post.time_source,
            "text_hash": post.text_hash,
        }
        for post in posts
    ]


def _post_from_suite_record(record: dict[str, Any]) -> LabeledPost:
    return LabeledPost(
        title=str(record["title"]),
        selftext=str(record.get("selftext") or ""),
        label=int(record["label"]),
        post_id=record.get("post_id"),
        subreddit=record.get("subreddit"),
        permalink=record.get("permalink"),
        post_type=record.get("post_type"),
        content_domain=record.get("content_domain"),
        is_crosspost=record.get("is_crosspost"),
        created_utc=record.get("created_utc"),
        time_key=record.get("time_key"),
        time_source=record.get("time_source"),
        text_hash=record.get("text_hash"),
    )


def _load_resumable_suite_summary(
    *,
    spec: SuiteModelSpec,
    output_dir: Path,
    split: DatasetSplit,
    prepared_data_summary: dict[str, int],
    suite_manifest_path: Path,
) -> dict[str, Any] | None:
    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        LOGGER.warning("ignoring unreadable benchmark model summary path=%s error=%s", str(summary_path), str(exc))
        return None
    if not _suite_summary_matches_spec(summary, spec):
        LOGGER.info("existing benchmark model summary incompatible with spec name=%s path=%s", spec.name, str(summary_path))
        return None
    artifact_path = _resolve_suite_artifact_path(summary, summary_path.parent)
    if artifact_path is None or not artifact_path.exists():
        LOGGER.info(
            "existing benchmark model summary missing artifact name=%s path=%s artifact=%s",
            spec.name,
            str(summary_path),
            str(artifact_path) if artifact_path is not None else "missing",
        )
        return None
    current_manifest_fingerprint = _file_sha256(suite_manifest_path)
    suite_resume = summary.get("suite_resume") if isinstance(summary.get("suite_resume"), dict) else {}
    if suite_resume.get("manifest_fingerprint") == current_manifest_fingerprint:
        return summary
    if _legacy_suite_summary_matches_run(summary, split=split, prepared_data_summary=prepared_data_summary):
        LOGGER.info(
            "reusing benchmark model summary via legacy compatibility name=%s path=%s",
            spec.name,
            str(summary_path),
        )
        return summary
    LOGGER.info("existing benchmark model summary does not match current split name=%s path=%s", spec.name, str(summary_path))
    return None


def _stamp_suite_model_summary(
    *,
    spec: SuiteModelSpec,
    output_dir: Path,
    summary: dict[str, Any],
    suite_manifest_path: Path,
    split: DatasetSplit,
    prepared_data_summary: dict[str, int],
) -> dict[str, Any]:
    stamped_summary = dict(summary)
    config = spec.kwargs.get("config")
    if stamped_summary.get("config_version") is None:
        if config is not None and hasattr(config, "config_version"):
            stamped_summary["config_version"] = config.config_version
        elif spec.kwargs.get("config_version") is not None:
            stamped_summary["config_version"] = spec.kwargs.get("config_version")
    stamped_summary["suite_resume"] = {
        "summary_version": 1,
        "manifest_fingerprint": _file_sha256(suite_manifest_path),
        "spec_name": spec.name,
        "display_name": spec.display_name,
        "model_family": spec.family,
        "expected_model_id": _expected_model_id_for_spec(spec),
        "split_signature": _legacy_split_signature(split),
        "prepared_training_records": int(prepared_data_summary.get("training_records", 0)),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(stamped_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stamped_summary


def _suite_summary_matches_spec(summary: dict[str, Any], spec: SuiteModelSpec) -> bool:
    if summary.get("model_family") != spec.family:
        return False
    expected_model_id = _expected_model_id_for_spec(spec)
    if expected_model_id is not None and summary.get("model_id") != expected_model_id:
        return False
    config = spec.kwargs.get("config")
    if config is not None and hasattr(config, "config_version"):
        if summary.get("config_version") != config.config_version:
            return False
    expected_config_version = spec.kwargs.get("config_version")
    if expected_config_version is not None and summary.get("config_version") != expected_config_version:
        return False
    expected_prompt_template_version = spec.kwargs.get("prompt_template_version")
    if expected_prompt_template_version is not None:
        actual_prompt_template_version = summary.get("prompt_template_version")
        if actual_prompt_template_version is None and isinstance(summary.get("training_args"), dict):
            actual_prompt_template_version = summary["training_args"].get("prompt_template_version")
        if isinstance(expected_prompt_template_version, (tuple, list, set)):
            if actual_prompt_template_version not in expected_prompt_template_version:
                return False
        elif actual_prompt_template_version != expected_prompt_template_version:
            return False
    variant = spec.kwargs.get("variant")
    if variant is not None:
        actual_variant = summary.get("variant", {})
        if actual_variant.get("name") != variant.name:
            return False
        if actual_variant.get("tfidf_config_version") != variant.tfidf_config_version:
            return False
        if bool(actual_variant.get("normalize_urls")) != bool(variant.normalize_urls):
            return False
        if bool(actual_variant.get("strip_urls")) != bool(variant.strip_urls):
            return False
        if float(actual_variant.get("max_slice_positive_weight", DEFAULT_MAX_SLICE_POSITIVE_WEIGHT)) != float(
            variant.max_slice_positive_weight
        ):
            return False
    return True


def _legacy_suite_summary_matches_run(
    summary: dict[str, Any],
    *,
    split: DatasetSplit,
    prepared_data_summary: dict[str, int],
) -> bool:
    expected_signature = _legacy_split_signature(split)
    actual_signature = _legacy_split_signature_from_summary(summary.get("split"))
    if actual_signature != expected_signature:
        return False
    summary_prepared = summary.get("prepared_data")
    if isinstance(summary_prepared, dict):
        if int(summary_prepared.get("training_records", 0)) != int(prepared_data_summary.get("training_records", 0)):
            return False
    return True


def _legacy_split_signature(split: DatasetSplit) -> dict[str, Any]:
    return {
        "train": len(split.train),
        "calibration": len(split.calibration),
        "test": len(split.test),
        "split_strategy": split.split_strategy,
        "split_seed": split.split_seed,
        "evaluation_subreddit": split.evaluation_subreddit,
        "excluded_for_time_split": split.excluded_for_time_split,
        "label_counts": {
            "train": _label_counts(split.train),
            "calibration": _label_counts(split.calibration),
            "test": _label_counts(split.test),
        },
    }


def _legacy_split_signature_from_summary(split_summary: Any) -> dict[str, Any] | None:
    if not isinstance(split_summary, dict):
        return None
    return {
        "train": split_summary.get("train"),
        "calibration": split_summary.get("calibration"),
        "test": split_summary.get("test"),
        "split_strategy": split_summary.get("split_strategy"),
        "split_seed": split_summary.get("split_seed"),
        "evaluation_subreddit": split_summary.get("evaluation_subreddit"),
        "excluded_for_time_split": split_summary.get("excluded_for_time_split"),
        "label_counts": split_summary.get("label_counts"),
    }


def _expected_model_id_for_spec(spec: SuiteModelSpec) -> str | None:
    config = spec.kwargs.get("config")
    if config is not None and hasattr(config, "model_id"):
        return str(config.model_id)
    model_id = spec.kwargs.get("model_id")
    return str(model_id) if model_id is not None else None


def _resolve_suite_artifact_path(summary: dict[str, Any], base_dir: Path) -> Path | None:
    artifact_path = summary.get("artifact_path")
    if not artifact_path:
        return None
    path = Path(str(artifact_path))
    candidates = [path]
    if not path.is_absolute():
        candidates = [
            (Path.cwd() / path),
            (base_dir / path),
        ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold_summary(
    thresholds: DecisionThresholds,
    *,
    review_precision_target: float,
    high_precision_target: float,
) -> dict[str, Any]:
    return {
        "low_threshold": thresholds.low_threshold,
        "high_threshold": thresholds.high_threshold,
        "review_precision_target": review_precision_target,
        "high_precision_target": high_precision_target,
        "low_threshold_strategy": "max_recall_subject_to_review_precision_target",
        "high_threshold_strategy": "max_recall_subject_to_high_precision_target_and_minimum_calibration_predictions",
        "minimum_high_confidence_calibration_predictions": thresholds.minimum_high_confidence_calibration_predictions,
        "high_threshold_fallback_used": thresholds.high_threshold_fallback_used,
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
    evaluate_on_test: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "tfidf training start variant=%s output_dir=%s train=%s calibration=%s test=%s",
        variant.name,
        str(artifact_dir),
        len(split.train),
        len(split.calibration),
        len(split.test),
    )

    y_calibration = [post.label for post in split.calibration]
    y_test = [post.label for post in split.test]
    representation_config = _representation_config_for_split(split)
    train_rows = _inference_rows(split.train, representation_config=representation_config)
    calibration_rows = _inference_rows(split.calibration, representation_config=representation_config)
    test_rows = _inference_rows(split.test, representation_config=representation_config)
    slice_weighting = _slice_aware_positive_weighting(
        split.train,
        rows=train_rows,
        max_slice_positive_weight=variant.max_slice_positive_weight,
    )

    model = train_model(
        split.train,
        sample_weight=slice_weighting.sample_weights,
        extra_word_stopwords=variant.extra_word_stopwords,
        char_weight=variant.char_weight,
        metadata_weight=variant.metadata_weight,
        normalize_urls=variant.normalize_urls,
        strip_urls=variant.strip_urls,
        min_df=variant.min_df,
        classifier_c=variant.classifier_c,
        classifier_class_weight=variant.classifier_class_weight,
        include_sparse_media_token=representation_config["include_sparse_media_token"],
        include_image_low_text_tokens=representation_config["include_image_low_text_tokens"],
    )
    raw_calibration_scores = positive_probabilities(model, calibration_rows)
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        review_precision_target=variant.review_precision_target,
        calibration=calibration,
    )
    artifact_path = artifact_dir / "tfidf_logreg.joblib"
    threshold_policy = _decision_policy(
        split=split,
        calibration=calibration,
        thresholds=thresholds,
        review_precision_target=variant.review_precision_target,
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
    )
    save_model(
        model,
        artifact_path,
        calibrator=calibrator,
        decision_policy=threshold_policy,
        representation_config=representation_config,
    )

    summary = {
        "version": __version__,
        "model_name": "tfidf_logreg",
        "model_family": "tfidf",
        "runtime_environment": _runtime_environment_metadata(),
        "variant": {
            "name": variant.name,
            "extra_word_stopwords": sorted(variant.extra_word_stopwords),
            "char_weight": variant.char_weight,
            "metadata_weight": variant.metadata_weight,
            "tfidf_config_version": variant.tfidf_config_version,
            "normalize_urls": variant.normalize_urls,
            "strip_urls": variant.strip_urls,
            "review_precision_target": variant.review_precision_target,
            "classifier_c": variant.classifier_c,
            "classifier_class_weight": variant.classifier_class_weight,
            "min_df": variant.min_df,
            "max_slice_positive_weight": variant.max_slice_positive_weight,
        },
        "artifact_path": str(artifact_path),
        "representation_config": representation_config,
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(
            thresholds,
            review_precision_target=variant.review_precision_target,
            high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        ),
        "threshold_policy": threshold_policy,
        "feature_audit": tfidf_feature_audit(model),
        "training_balance": slice_weighting.summary,
        "benchmark_status": "not_run",
        "production_ready": False,
        "production_ready_blocked_reason": "benchmark_not_run",
    }
    if prepared_data_summary is not None:
        summary["prepared_data"] = prepared_data_summary
    if evaluate_on_test:
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
            train_rows=train_rows,
            train_labels=[post.label for post in split.train],
            rows=test_rows,
            low_threshold=thresholds.low_threshold,
            high_threshold=thresholds.high_threshold,
        )
        production_ready, blocked_reason = _production_ready_status(
            calibration=calibration,
            thresholds=thresholds,
            high_confidence_precision=band_metrics.high_confidence_precision,
            high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
        )
        summary.update(
            {
                "ranking_metrics": _ranking_metrics_summary(y_test, calibrated_test_scores),
                "constraint_metrics": _constraint_metrics_summary(y_test, calibrated_test_scores),
                "metrics": {
                    "high_confidence_precision": band_metrics.high_confidence_precision,
                    "high_confidence_recall": band_metrics.high_confidence_recall,
                    "high_confidence_f1": band_metrics.high_confidence_f1,
                    "support": band_metrics.support,
                    "confidence_band_counts": band_metrics.band_counts,
                },
                "operating_metrics": asdict(operating_metrics),
                "benchmark_status": "complete",
                "production_ready": production_ready,
                "production_ready_blocked_reason": blocked_reason,
            }
        )
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if evaluate_on_test:
        LOGGER.info(
            "tfidf training complete variant=%s artifact=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
            variant.name,
            str(artifact_path),
            float(summary["operating_metrics"]["auto_band"]["precision"]),
            float(summary["operating_metrics"]["auto_band"]["recall"]),
            float(summary["operating_metrics"]["review_queue"]["precision"]),
            float(summary["operating_metrics"]["review_queue"]["recall"]),
            _format_elapsed(time.perf_counter() - started_at),
        )
    else:
        LOGGER.info(
            "tfidf training complete variant=%s artifact=%s calibration_available=%s high_threshold=%s elapsed=%s",
            variant.name,
            str(artifact_path),
            calibration.available,
            thresholds.high_threshold,
            _format_elapsed(time.perf_counter() - started_at),
        )
    return summary


def _train_semantic_embedding_bundle_for_split(
    *,
    split: DatasetSplit,
    output_dir: str | Path,
    model_id: str | None = None,
    config: SemanticModelConfig | None = None,
    prepared_data_summary: dict[str, int] | None = None,
    evaluate_on_test: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    semantic_config = config or SemanticModelConfig(
        name="semantic_embedding",
        display_name="Semantic",
        model_id=str(model_id or DEFAULT_SEMANTIC_MODEL_ID),
        backend="sentence_transformers",
        config_version="v3_title_body_metadata_weighted",
        prompt_modes=("plain",),
        normalize_embeddings=(False,),
        logistic_c_values=(1.0,),
        title_weight_values=(1.0,),
        body_weight_values=(1.0,),
        encode_batch_size=16,
        prompt_prefix="",
        pooling="sentence_transformers",
    )
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    representation_config = _representation_config_for_split(split)
    train_rows = _inference_rows(split.train, representation_config=representation_config)
    calibration_rows = _inference_rows(split.calibration, representation_config=representation_config)
    slice_weighting = _slice_aware_positive_weighting(split.train, rows=train_rows)
    y_train = [post.label for post in split.train]
    y_calibration = [post.label for post in split.calibration]
    metadata_vectorizer = CountVectorizer(
        binary=True,
        lowercase=False,
        min_df=1,
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=None,
    )
    metadata_train = np.asarray(
        metadata_vectorizer.fit_transform([str(row.get("metadata_text") or "") for row in train_rows]).toarray(),
        dtype=np.float32,
    )
    metadata_calibration = np.asarray(
        metadata_vectorizer.transform([str(row.get("metadata_text") or "") for row in calibration_rows]).toarray(),
        dtype=np.float32,
    )
    LOGGER.info(
        "semantic training start name=%s model_id=%s backend=%s train=%s calibration=%s test=%s prompt_modes=%s normalize_options=%s title_weights=%s body_weights=%s c_values=%s",
        semantic_config.name,
        semantic_config.model_id,
        semantic_config.backend,
        len(split.train),
        len(split.calibration),
        len(split.test),
        list(semantic_config.prompt_modes),
        list(semantic_config.normalize_embeddings),
        list(semantic_config.title_weight_values),
        list(semantic_config.body_weight_values),
        list(semantic_config.logistic_c_values),
    )

    best_candidate: dict[str, Any] | None = None
    for prompt_mode in semantic_config.prompt_modes:
        LOGGER.info(
            "semantic encoder load name=%s model_id=%s prompt_mode=%s backend=%s",
            semantic_config.name,
            semantic_config.model_id,
            prompt_mode,
            semantic_config.backend,
        )
        encoder = _load_semantic_encoder(semantic_config)
        train_title_texts, train_body_texts = _semantic_component_texts(
            split.train,
            prompt_mode=prompt_mode,
            config=semantic_config,
        )
        calibration_title_texts, calibration_body_texts = _semantic_component_texts(
            split.calibration,
            prompt_mode=prompt_mode,
            config=semantic_config,
        )
        LOGGER.info(
            "semantic encoding start name=%s prompt_mode=%s split=train_title examples=%s normalize=false",
            semantic_config.name,
            prompt_mode,
            len(split.train),
        )
        train_title_embeddings = _semantic_embeddings(
            encoder,
            semantic_config,
            texts=train_title_texts,
            normalize=False,
        )
        LOGGER.info(
            "semantic encoding start name=%s prompt_mode=%s split=train_body examples=%s normalize=false",
            semantic_config.name,
            prompt_mode,
            len(split.train),
        )
        train_body_embeddings = _semantic_embeddings(
            encoder,
            semantic_config,
            texts=train_body_texts,
            normalize=False,
        )
        LOGGER.info(
            "semantic encoding start name=%s prompt_mode=%s split=calibration_title examples=%s normalize=false",
            semantic_config.name,
            prompt_mode,
            len(split.calibration),
        )
        calibration_title_embeddings = _semantic_embeddings(
            encoder,
            semantic_config,
            texts=calibration_title_texts,
            normalize=False,
        )
        LOGGER.info(
            "semantic encoding start name=%s prompt_mode=%s split=calibration_body examples=%s normalize=false",
            semantic_config.name,
            prompt_mode,
            len(split.calibration),
        )
        calibration_body_embeddings = _semantic_embeddings(
            encoder,
            semantic_config,
            texts=calibration_body_texts,
            normalize=False,
        )
        for normalize_embeddings in semantic_config.normalize_embeddings:
            normalized_train_title = _normalize_embedding_matrix(train_title_embeddings, enabled=normalize_embeddings)
            normalized_train_body = _normalize_embedding_matrix(train_body_embeddings, enabled=normalize_embeddings)
            normalized_calibration_title = _normalize_embedding_matrix(
                calibration_title_embeddings,
                enabled=normalize_embeddings,
            )
            normalized_calibration_body = _normalize_embedding_matrix(
                calibration_body_embeddings,
                enabled=normalize_embeddings,
            )
            for title_weight in semantic_config.title_weight_values:
                for body_weight in semantic_config.body_weight_values:
                    weighted_train = np.hstack(
                        [
                            normalized_train_title * float(title_weight),
                            normalized_train_body * float(body_weight),
                            metadata_train,
                        ]
                    ).astype(np.float32)
                    weighted_calibration = np.hstack(
                        [
                            normalized_calibration_title * float(title_weight),
                            normalized_calibration_body * float(body_weight),
                            metadata_calibration,
                        ]
                    ).astype(np.float32)
                    for logistic_c in semantic_config.logistic_c_values:
                        classifier = LogisticRegression(
                            class_weight="balanced",
                            max_iter=2_000,
                            solver="liblinear",
                            C=logistic_c,
                        )
                        classifier.fit(weighted_train, y_train, sample_weight=slice_weighting.sample_weights)
                        calibration_scores = [float(row[1]) for row in classifier.predict_proba(weighted_calibration)]
                        candidate = {
                            "prompt_mode": prompt_mode,
                            "normalize_embeddings": normalize_embeddings,
                            "title_weight": float(title_weight),
                            "body_weight": float(body_weight),
                            "logistic_c": logistic_c,
                            "classifier": classifier,
                            "calibration_scores": calibration_scores,
                            "encoder": encoder,
                            "metadata_vectorizer": metadata_vectorizer,
                        }
                        candidate_metrics = _classification_metrics(
                            y_calibration,
                            [1 if score >= 0.5 else 0 for score in calibration_scores],
                        )
                        candidate_metrics["pr_auc"] = _safe_average_precision(y_calibration, calibration_scores)
                        candidate["candidate_metrics"] = candidate_metrics
                        candidate["constraint_metrics"] = _constraint_metrics_summary(y_calibration, calibration_scores)
                        LOGGER.info(
                            "semantic candidate evaluated name=%s prompt_mode=%s normalize=%s title_weight=%s body_weight=%s logistic_c=%s precision=%.3f recall=%.3f f1=%.3f pr_auc=%.3f auto95_recall=%.3f review75_recall=%.3f predicted_positive=%s",
                            semantic_config.name,
                            prompt_mode,
                            normalize_embeddings,
                            float(title_weight),
                            float(body_weight),
                            logistic_c,
                            float(candidate_metrics["precision"]),
                            float(candidate_metrics["recall"]),
                            float(candidate_metrics["f1"]),
                            float(candidate_metrics["pr_auc"]),
                            float(candidate["constraint_metrics"]["auto_recall_at_precision_95"]["recall"]),
                            float(candidate["constraint_metrics"]["review_recall_at_precision_75"]["recall"]),
                            int(candidate_metrics["predicted_positive"]),
                        )
                        if best_candidate is None or _semantic_candidate_key(candidate) > _semantic_candidate_key(best_candidate):
                            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("semantic tuning did not produce a candidate")

    encoder = best_candidate["encoder"]
    metadata_vectorizer = best_candidate["metadata_vectorizer"]
    normalize_embeddings = bool(best_candidate["normalize_embeddings"])
    prompt_mode = str(best_candidate["prompt_mode"])
    title_weight = float(best_candidate["title_weight"])
    body_weight = float(best_candidate["body_weight"])
    classifier = best_candidate["classifier"]
    LOGGER.info(
        "semantic candidate selected name=%s prompt_mode=%s normalize=%s title_weight=%s body_weight=%s logistic_c=%s",
        semantic_config.name,
        prompt_mode,
        normalize_embeddings,
        float(best_candidate["title_weight"]),
        float(best_candidate["body_weight"]),
        best_candidate["logistic_c"],
    )
    LOGGER.info(
        "semantic encoding start name=%s prompt_mode=%s split=calibration_title examples=%s normalize=%s",
        semantic_config.name,
        prompt_mode,
        len(split.calibration),
        normalize_embeddings,
    )
    calibration_title_texts, calibration_body_texts = _semantic_component_texts(
        split.calibration,
        prompt_mode=prompt_mode,
        config=semantic_config,
    )
    calibration_title_embeddings = _semantic_embeddings(
        encoder,
        semantic_config,
        texts=calibration_title_texts,
        normalize=normalize_embeddings,
    )
    LOGGER.info(
        "semantic encoding start name=%s prompt_mode=%s split=calibration_body examples=%s normalize=%s",
        semantic_config.name,
        prompt_mode,
        len(split.calibration),
        normalize_embeddings,
    )
    calibration_body_embeddings = _semantic_embeddings(
        encoder,
        semantic_config,
        texts=calibration_body_texts,
        normalize=normalize_embeddings,
    )
    calibration_features = np.hstack(
        [
            calibration_title_embeddings * title_weight,
            calibration_body_embeddings * body_weight,
            metadata_calibration,
        ]
    ).astype(np.float32)
    raw_calibration_scores = [float(row[1]) for row in classifier.predict_proba(calibration_features)]
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
        calibration=calibration,
    )
    threshold_policy = _decision_policy(
        split=split,
        calibration=calibration,
        thresholds=thresholds,
        review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
    )
    artifact_path = artifact_dir / "semantic_embedding_logreg.joblib"
    joblib.dump(
        {
            "model_family": "semantic_embedding",
            "model_name": semantic_config.name,
            "display_name": semantic_config.display_name,
            "model_id": semantic_config.model_id,
            "backend": semantic_config.backend,
            "config_version": semantic_config.config_version,
            "feature_layout": semantic_config.feature_layout,
            "prompt_mode": prompt_mode,
            "prompt_prefix": semantic_config.prompt_prefix,
            "short_prompt_prefix": semantic_config.short_prompt_prefix,
            "normalize_embeddings": normalize_embeddings,
            "pooling": semantic_config.pooling,
            "title_weight": title_weight,
            "body_weight": body_weight,
            "encode_batch_size": semantic_config.encode_batch_size,
            "embedding_dimension": int(np.asarray(calibration_title_embeddings).shape[1]),
            "metadata_vectorizer": metadata_vectorizer,
            "metadata_feature_count": int(metadata_calibration.shape[1]),
            "classifier": classifier,
            "calibrator": calibrator,
            "threshold_policy": threshold_policy,
            "representation_config": representation_config,
            "version": __version__,
        },
        artifact_path,
    )

    summary = {
        "version": __version__,
        "model_name": semantic_config.name,
        "model_family": "semantic_embedding",
        "runtime_environment": _runtime_environment_metadata(),
        "display_name": semantic_config.display_name,
        "model_id": semantic_config.model_id,
        "artifact_path": str(artifact_path),
        "representation_config": representation_config,
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(
            thresholds,
            review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
            high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        ),
        "threshold_policy": threshold_policy,
        "config_version": semantic_config.config_version,
        "embedding_summary": {
            "embedding_dimension": int(np.asarray(calibration_title_embeddings).shape[1]),
            "train_examples": len(split.train),
            "model_id": semantic_config.model_id,
            "backend": semantic_config.backend,
            "feature_layout": semantic_config.feature_layout,
            "prompt_mode": prompt_mode,
            "prompt_prefix": semantic_config.prompt_prefix,
            "short_prompt_prefix": semantic_config.short_prompt_prefix,
            "normalize_embeddings": normalize_embeddings,
            "pooling": semantic_config.pooling,
            "metadata_feature_count": int(metadata_calibration.shape[1]),
            "logistic_c": float(best_candidate["logistic_c"]),
            "title_weight": title_weight,
            "body_weight": body_weight,
            "candidate_metrics": best_candidate["candidate_metrics"],
            "constraint_metrics": best_candidate["constraint_metrics"],
        },
        "training_balance": slice_weighting.summary,
        "benchmark_status": "not_run",
        "production_ready": False,
        "production_ready_blocked_reason": "benchmark_not_run",
    }
    if prepared_data_summary is not None:
        summary["prepared_data"] = prepared_data_summary
    if evaluate_on_test:
        test_rows = _inference_rows(split.test, representation_config=representation_config)
        y_test = [post.label for post in split.test]
        LOGGER.info(
            "semantic encoding start name=%s prompt_mode=%s split=test_title examples=%s normalize=%s",
            semantic_config.name,
            prompt_mode,
            len(split.test),
            normalize_embeddings,
        )
        test_title_texts, test_body_texts = _semantic_component_texts(
            split.test,
            prompt_mode=prompt_mode,
            config=semantic_config,
        )
        test_title_embeddings = _semantic_embeddings(
            encoder,
            semantic_config,
            texts=test_title_texts,
            normalize=normalize_embeddings,
        )
        LOGGER.info(
            "semantic encoding start name=%s prompt_mode=%s split=test_body examples=%s normalize=%s",
            semantic_config.name,
            prompt_mode,
            len(split.test),
            normalize_embeddings,
        )
        test_body_embeddings = _semantic_embeddings(
            encoder,
            semantic_config,
            texts=test_body_texts,
            normalize=normalize_embeddings,
        )
        test_metadata = np.asarray(
            metadata_vectorizer.transform([str(row.get("metadata_text") or "") for row in test_rows]).toarray(),
            dtype=np.float32,
        )
        test_features = np.hstack(
            [
                test_title_embeddings * title_weight,
                test_body_embeddings * body_weight,
                test_metadata,
            ]
        ).astype(np.float32)
        raw_test_scores = [float(row[1]) for row in classifier.predict_proba(test_features)]
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
            train_rows=train_rows,
            train_labels=y_train,
            rows=test_rows,
            low_threshold=thresholds.low_threshold,
            high_threshold=thresholds.high_threshold,
        )
        production_ready, blocked_reason = _production_ready_status(
            calibration=calibration,
            thresholds=thresholds,
            high_confidence_precision=band_metrics.high_confidence_precision,
            high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
        )
        summary.update(
            {
                "ranking_metrics": _ranking_metrics_summary(y_test, calibrated_test_scores),
                "constraint_metrics": _constraint_metrics_summary(y_test, calibrated_test_scores),
                "metrics": {
                    "high_confidence_precision": band_metrics.high_confidence_precision,
                    "high_confidence_recall": band_metrics.high_confidence_recall,
                    "high_confidence_f1": band_metrics.high_confidence_f1,
                    "support": band_metrics.support,
                    "confidence_band_counts": band_metrics.band_counts,
                },
                "operating_metrics": asdict(operating_metrics),
                "benchmark_status": "complete",
                "production_ready": production_ready,
                "production_ready_blocked_reason": blocked_reason,
            }
        )
    if semantic_config.backend == "hf_embedding":
        semantic_model = encoder.get("model") if isinstance(encoder, dict) else None
        if semantic_model is not None and hasattr(semantic_model, "to"):
            semantic_model.to("cpu")
    _clear_torch_memory()
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if evaluate_on_test:
        LOGGER.info(
            "semantic training complete name=%s artifact=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
            semantic_config.name,
            str(artifact_path),
            float(summary["operating_metrics"]["auto_band"]["precision"]),
            float(summary["operating_metrics"]["auto_band"]["recall"]),
            float(summary["operating_metrics"]["review_queue"]["precision"]),
            float(summary["operating_metrics"]["review_queue"]["recall"]),
            _format_elapsed(time.perf_counter() - started_at),
        )
    else:
        LOGGER.info(
            "semantic training complete name=%s artifact=%s calibration_available=%s high_threshold=%s elapsed=%s",
            semantic_config.name,
            str(artifact_path),
            calibration.available,
            thresholds.high_threshold,
            _format_elapsed(time.perf_counter() - started_at),
        )
    return summary


def _load_semantic_encoder(config: SemanticModelConfig) -> Any:
    if config.backend == "sentence_transformers":
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalModelDependencyError(
                "Semantic embedding benchmarks require sentence-transformers. "
                "Install with `python -m pip install -e \".[dev,models]\"`."
            ) from exc
        _enable_cuda_tf32(torch)
        return SentenceTransformer(config.model_id)
    if config.backend == "hf_embedding":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise OptionalModelDependencyError(
                "Transformer-backed embedding benchmarks require transformers and torch. "
                "Install with `python -m pip install -e \".[dev,models]\"`."
            ) from exc
        _enable_cuda_tf32(torch)
        device = _resolve_semantic_encoder_device(config, torch)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return {
            "tokenizer": tokenizer,
            "model": AutoModel.from_pretrained(config.model_id),
            "device": device,
        }
    raise ValueError(f"Unsupported semantic backend: {config.backend}")


def _resolve_semantic_encoder_device(config: SemanticModelConfig, torch_module: Any) -> str:
    device = _torch_runtime_device(torch_module)
    if config.backend == "hf_embedding" and device == "mps":
        LOGGER.info(
            "semantic hf embedding bypassing mps model_id=%s reason=%s fallback_device=cpu",
            config.model_id,
            "mps_backend_instability",
        )
        return "cpu"
    return device


def _semantic_component_texts(
    posts: list[LabeledPost],
    *,
    prompt_mode: str,
    config: SemanticModelConfig,
) -> tuple[list[str], list[str]]:
    title_texts = [_semantic_component_fallback_text(str(post.title).strip(), component="title") for post in posts]
    body_texts = [_semantic_component_fallback_text(str(post.selftext or "").strip(), component="body") for post in posts]
    if prompt_mode == "plain":
        return title_texts, body_texts
    if prompt_mode == "task_prefix":
        title_prefix = config.prompt_prefix or ""
        return (
            [f"{title_prefix}\nTitle: {text}" if title_prefix else f"Title: {text}" for text in title_texts],
            [f"{title_prefix}\nBody: {text}" if title_prefix else f"Body: {text}" for text in body_texts],
        )
    if prompt_mode == "short_task_prefix":
        short_prefix = config.short_prompt_prefix or config.prompt_prefix or ""
        return (
            [f"{short_prefix} Title: {text}".strip() if short_prefix else f"Title: {text}" for text in title_texts],
            [f"{short_prefix} Body: {text}".strip() if short_prefix else f"Body: {text}" for text in body_texts],
        )
    if prompt_mode == "document_prefix":
        document_prefix = config.prompt_prefix or "Document:"
        return (
            [f"{document_prefix} {text}".strip() for text in title_texts],
            [f"{document_prefix} {text}".strip() for text in body_texts],
        )
    if prompt_mode == "jina_document_component":
        document_prefix = config.prompt_prefix or "Document:"
        return (
            [f"{document_prefix} Title: {text}".strip() for text in title_texts],
            [f"{document_prefix} Body: {text}".strip() for text in body_texts],
        )
    raise ValueError(f"Unsupported semantic prompt mode: {prompt_mode}")


def _semantic_component_fallback_text(text: str, *, component: str) -> str:
    normalized = str(text).strip()
    if normalized:
        return normalized
    if component == "title":
        return "[no title]"
    return "[no body]"


def _semantic_texts(
    posts: list[LabeledPost],
    *,
    prompt_mode: str,
    config: SemanticModelConfig,
) -> list[str]:
    texts = _texts(posts)
    if prompt_mode == "plain":
        return texts
    if prompt_mode == "task_prefix":
        return [f"{config.prompt_prefix}\n{text}" if config.prompt_prefix else text for text in texts]
    if prompt_mode == "short_task_prefix":
        return [f"{config.short_prompt_prefix} {text}".strip() if config.short_prompt_prefix else text for text in texts]
    if prompt_mode == "document_prefix":
        document_prefix = config.prompt_prefix or "Document:"
        return [f"{document_prefix} {text}".strip() for text in texts]
    if prompt_mode == "jina_document_component":
        document_prefix = config.prompt_prefix or "Document:"
        return [f"{document_prefix} {text}".strip() for text in texts]
    raise ValueError(f"Unsupported semantic prompt mode: {prompt_mode}")


def _semantic_embeddings(
    encoder: Any,
    config: SemanticModelConfig,
    *,
    texts: list[str],
    normalize: bool,
) -> np.ndarray:
    LOGGER.info(
        "semantic encode backend=%s model_id=%s texts=%s batch_size=%s pooling=%s normalize=%s",
        config.backend,
        config.model_id,
        len(texts),
        config.encode_batch_size,
        config.pooling,
        normalize,
    )
    started_at = time.perf_counter()
    if config.backend == "sentence_transformers":
        encoded = encoder.encode(
            texts,
            show_progress_bar=False,
            batch_size=config.encode_batch_size,
            normalize_embeddings=normalize,
        )
        output = np.asarray(encoded, dtype=np.float32)
        LOGGER.info(
            "semantic encode complete backend=%s model_id=%s texts=%s elapsed=%s",
            config.backend,
            config.model_id,
            len(texts),
            _format_elapsed(time.perf_counter() - started_at),
        )
        return output

    encoded = _hf_embedding_encode(
        encoder,
        texts,
        pooling=config.pooling,
        batch_size=config.encode_batch_size,
    )
    output = _normalize_embedding_matrix(encoded, enabled=normalize)
    LOGGER.info(
        "semantic encode complete backend=%s model_id=%s texts=%s elapsed=%s",
        config.backend,
        config.model_id,
        len(texts),
        _format_elapsed(time.perf_counter() - started_at),
    )
    return output


def _transformer_candidate_profiles(model_id: str, *, allow_long_context: bool = False) -> list[dict[str, Any]]:
    normalized = str(model_id).lower()
    if "modernbert-large" in normalized:
        profiles = [
            {"name": "baseline", "learning_rate": 2e-5, "weight_decay": 0.01, "max_length": 384},
            {"name": "precision_tuned", "learning_rate": 1.0e-5, "weight_decay": 0.03, "max_length": 384},
            {"name": "balanced_tuned", "learning_rate": 1.5e-5, "weight_decay": 0.02, "max_length": 384},
        ]
        if allow_long_context:
            profiles.append({"name": "long_context", "learning_rate": 1.25e-5, "weight_decay": 0.02, "max_length": 512})
        return profiles
    if "neobert" in normalized:
        return [
            {"name": "baseline", "learning_rate": 2e-5, "weight_decay": 0.01, "max_length": 384},
            {"name": "precision_tuned", "learning_rate": 1.0e-5, "weight_decay": 0.03, "max_length": 384},
            {"name": "long_context", "learning_rate": 1.5e-5, "weight_decay": 0.02, "max_length": 512},
        ]
    if "modernbert" in normalized:
        profiles = [
            {"name": "baseline", "learning_rate": 2e-5, "weight_decay": 0.01, "max_length": 384},
            {"name": "precision_tuned", "learning_rate": 1.0e-5, "weight_decay": 0.03, "max_length": 384},
            {"name": "balanced_tuned", "learning_rate": 1.5e-5, "weight_decay": 0.02, "max_length": 384},
        ]
        if allow_long_context:
            profiles.append({"name": "long_context", "learning_rate": 1.25e-5, "weight_decay": 0.02, "max_length": 512})
        return profiles
    if "deberta-v3-small" in normalized:
        return [
            {"name": "baseline", "learning_rate": 2e-5, "weight_decay": 0.01, "max_length": 256},
            {"name": "balanced_tuned", "learning_rate": 1.5e-5, "weight_decay": 0.02, "max_length": 384},
            {"name": "precision_tuned", "learning_rate": 1.0e-5, "weight_decay": 0.03, "max_length": 256},
        ]
    return [{"name": "baseline", "learning_rate": 2e-5, "weight_decay": 0.01, "max_length": 256}]


def _causal_lm_candidate_profiles() -> list[dict[str, Any]]:
    return [
        {
            "name": "v3_baseline",
            "prompt_template_version": DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
            "learning_rate": 5e-5,
            "lora_rank": 8,
            "num_train_epochs": 2,
        },
        {
            "name": "v3_precision",
            "prompt_template_version": DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
            "learning_rate": 2e-5,
            "lora_rank": 8,
            "num_train_epochs": 3,
        },
        {
            "name": "v4_low_text",
            "prompt_template_version": "v4_image_low_text",
            "learning_rate": 5e-5,
            "lora_rank": 8,
            "num_train_epochs": 2,
        },
        {
            "name": "v4_capacity",
            "prompt_template_version": "v4_image_low_text",
            "learning_rate": 2e-5,
            "lora_rank": 16,
            "num_train_epochs": 3,
        },
    ]


def _hf_embedding_encode(
    encoder: dict[str, Any],
    texts: list[str],
    *,
    pooling: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    tokenizer = encoder["tokenizer"]
    model = encoder["model"]
    device = encoder["device"]
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch_texts = texts[start : start + batch_size]
            if _should_log_progress(batch_index, total_batches):
                LOGGER.info(
                    "hf embedding progress device=%s batch=%s/%s examples=%s",
                    device,
                    batch_index,
                    total_batches,
                    len(texts),
                )
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
            if pooling == "last_token":
                pooled = _last_token_pool(hidden, batch["attention_mask"])
            else:
                pooled = _mean_pool(hidden, batch["attention_mask"])
            outputs.append(_tensor_to_float32_numpy(pooled))
    return np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)


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


def _mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * expanded).sum(dim=1)
    counts = expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _last_token_pool(hidden_states: Any, attention_mask: Any) -> Any:
    token_counts = attention_mask.sum(dim=1) - 1
    import torch

    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, token_counts]


def _tensor_to_float32_numpy(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32)


def _normalize_embedding_matrix(embeddings: np.ndarray, *, enabled: bool) -> np.ndarray:
    if not enabled or embeddings.size == 0:
        return np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return np.asarray(embeddings / norms, dtype=np.float32)


def _precision_first_candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    metrics = candidate.get("candidate_metrics", {})
    constraint_metrics = candidate.get("constraint_metrics", {})
    auto_band = constraint_metrics.get("auto_recall_at_precision_95", {})
    review_queue = constraint_metrics.get("review_recall_at_precision_75", {})
    return (
        float(metrics.get("pr_auc", 0.0)),
        float(auto_band.get("recall", 0.0)),
        float(review_queue.get("recall", 0.0)),
        float(review_queue.get("precision", 0.0)),
        float(metrics.get("recall", 0.0)),
        float(metrics.get("precision", 0.0)),
    )


def _semantic_candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    return _precision_first_candidate_key(candidate) + (-float(candidate["logistic_c"]),)


def _transformer_candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, int]:
    return (
        *_precision_first_candidate_key(candidate),
        1 if str(candidate.get("loss_mode")) == "plain_cross_entropy" else 0,
    )


def _train_transformer_bundle_for_split(
    *,
    split: DatasetSplit,
    output_dir: str | Path,
    model_id: str,
    display_name: str | None = None,
    prepared_data_summary: dict[str, int] | None = None,
    runtime_profile: str | None = None,
    config_version: str = "v2_pr_auc_early_stop",
    evaluate_on_test: bool = True,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise OptionalModelDependencyError(
            "Transformer benchmarks require transformers, datasets, accelerate, and torch. "
            "Install with `python -m pip install -e \".[dev,models]\"`."
        ) from exc

    started_at = time.perf_counter()
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    detected_runtime = _torch_runtime_device(torch)
    cuda_runtime = _enable_cuda_tf32(torch)
    effective_runtime_profile = runtime_profile or detected_runtime
    use_mps = effective_runtime_profile == "mps"
    use_cpu = effective_runtime_profile in {"cpu", "cpu_fallback"}
    load_options = transformer_load_options(model_id)
    trust_remote_code = bool(load_options.get("trust_remote_code"))
    per_device_train_batch_size = 8 if not use_cpu else 1
    per_device_eval_batch_size = 16 if not use_cpu else 2
    gradient_accumulation_steps = 1 if not use_cpu else 8
    candidate_profiles = _transformer_candidate_profiles(
        model_id,
        allow_long_context=(effective_runtime_profile == "cuda"),
    )
    default_max_length = int(candidate_profiles[0]["max_length"])
    max_length = default_max_length
    model_dtype = None
    num_train_epochs = 2
    if use_mps:
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 2
        gradient_accumulation_steps = 8
        model_dtype = torch.float16
    if use_mps and hasattr(torch.mps, "empty_cache"):
        gc.collect()
        torch.mps.empty_cache()
    LOGGER.info(
        "transformer training start model_id=%s display_name=%s output_dir=%s runtime_profile=%s train=%s calibration=%s test=%s batch_train=%s batch_eval=%s grad_accum=%s epochs=%s candidate_profiles=%s default_max_length=%s dtype=%s",
        model_id,
        display_name or model_id,
        str(artifact_dir),
        effective_runtime_profile,
        len(split.train),
        len(split.calibration),
        len(split.test),
        per_device_train_batch_size,
        per_device_eval_batch_size,
        gradient_accumulation_steps,
        num_train_epochs,
        [profile["name"] for profile in candidate_profiles],
        default_max_length,
        str(model_dtype).replace("torch.", "") if model_dtype is not None else "default",
    )

    representation_config = _representation_config_for_split(split)
    train_inference_rows = _inference_rows(split.train, representation_config=representation_config)
    slice_weighting = _slice_aware_positive_weighting(split.train, rows=train_inference_rows)
    train_rows = _sequence_classification_rows(
        split.train,
        example_weights=slice_weighting.sample_weights,
        representation_config=representation_config,
    )
    calibration_rows = _sequence_classification_rows(
        split.calibration,
        representation_config=representation_config,
    )
    y_train = [row["label"] for row in train_rows]
    y_calibration = [row["label"] for row in calibration_rows]

    ensure_transformer_custom_code_support(trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **load_options)

    class_weights = torch.tensor(_balanced_class_weights(y_train), dtype=torch.float32)

    class WeightedSequenceClassificationTrainer(Trainer):
        def __init__(self, *args: Any, class_weights: Any | None, loss_mode: str, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.loss_mode = loss_mode

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
                weight=(
                    self.class_weights.to(device=logits.device, dtype=logits.dtype)
                    if self.loss_mode == "balanced_cross_entropy" and self.class_weights is not None
                    else None
                ),
                reduction="none",
            )
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            if example_weight is not None:
                loss = loss * example_weight.to(logits.device).view(-1)
            loss = loss.mean()
            if return_outputs:
                return loss, outputs
            return loss

    class ProgressLoggingCallback(TrainerCallback):
        def __init__(self, label: str) -> None:
            self.label = label
            self._last_logged_step = 0
            self._initial_memory_snapshot = _mps_memory_snapshot(torch) if use_mps else None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            LOGGER.info("%s trainer started total_steps=%s", self.label, state.max_steps)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step or 0)
            total = int(state.max_steps or 0)
            interval = _progress_log_interval(total)
            if use_mps and step and step % 4 == 0 and hasattr(torch.mps, "empty_cache"):
                gc.collect()
                torch.mps.empty_cache()
            if use_mps and step and step % 8 == 0:
                snapshot = _mps_memory_snapshot(torch)
                if snapshot is not None and self._initial_memory_snapshot is not None:
                    LOGGER.info(
                        "%s mps memory step=%s current=%s driver=%s available=%s recommended=%s",
                        self.label,
                        step,
                        _format_bytes(snapshot["current_allocated_bytes"]),
                        _format_bytes(snapshot["driver_allocated_bytes"]),
                        _format_bytes(snapshot["available_system_bytes"]),
                        _format_bytes(snapshot["recommended_max_bytes"]),
                    )
                    if _should_proactively_fallback_from_mps(
                        initial_snapshot=self._initial_memory_snapshot,
                        current_snapshot=snapshot,
                    ):
                        raise MPSFallbackRequested(
                            "mps memory headroom dropped below the safety threshold; switching to cpu_fallback"
                        )
            if step and (step == 1 or step == total or step - self._last_logged_step >= interval):
                self._last_logged_step = step
                LOGGER.info("%s trainer progress step=%s/%s epoch=%.2f", self.label, step, total, float(state.epoch or 0.0))

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            LOGGER.info("%s trainer finished total_steps=%s", self.label, state.max_steps)

    def compute_pr_auc(eval_pred: Any) -> dict[str, float]:
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        scores = _positive_scores_from_logits(predictions)
        return {"pr_auc": _safe_average_precision(y_calibration, scores)}

    best_candidate: dict[str, Any] | None = None
    candidate_results: dict[str, dict[str, float | int | str]] = {}
    loss_modes = ("plain_cross_entropy", "balanced_cross_entropy")
    previous_transformers_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        for profile in candidate_profiles:
            max_length = int(profile["max_length"])
            def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
                return tokenizer(batch["title"], batch["body"], truncation=True, max_length=max_length)

            LOGGER.info("transformer tokenization start model_id=%s profile=%s split=train examples=%s", model_id, profile["name"], len(train_rows))
            train_dataset = Dataset.from_list(train_rows).map(tokenize, batched=True)
            LOGGER.info(
                "transformer tokenization start model_id=%s profile=%s split=calibration examples=%s",
                model_id,
                profile["name"],
                len(calibration_rows),
            )
            calibration_dataset = Dataset.from_list(calibration_rows).map(tokenize, batched=True)
            for loss_mode in loss_modes:
                LOGGER.info(
                    "transformer candidate start model_id=%s profile=%s loss_mode=%s",
                    model_id,
                    profile["name"],
                    loss_mode,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    num_labels=2,
                    id2label={0: "not_askseattle", 1: "askseattle"},
                    label2id={"not_askseattle": 0, "askseattle": 1},
                    torch_dtype=model_dtype,
                    **load_options,
                )
                candidate_dir = artifact_dir / f"checkpoints_{profile['name']}_{loss_mode}"
                training_args = TrainingArguments(
                    output_dir=str(candidate_dir),
                    learning_rate=float(profile["learning_rate"]),
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    num_train_epochs=num_train_epochs,
                    weight_decay=float(profile["weight_decay"]),
                    use_cpu=use_cpu,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    metric_for_best_model="pr_auc",
                    greater_is_better=True,
                    logging_strategy="no",
                    optim="adafactor" if use_mps else "adamw_torch",
                    report_to=[],
                )
                trainer = WeightedSequenceClassificationTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=calibration_dataset,
                    processing_class=tokenizer,
                    class_weights=class_weights if loss_mode == "balanced_cross_entropy" else None,
                    loss_mode=loss_mode,
                    callbacks=[
                        ProgressLoggingCallback(f"transformer {display_name or model_id} {profile['name']} {loss_mode}"),
                        EarlyStoppingCallback(early_stopping_patience=1),
                    ],
                    compute_metrics=compute_pr_auc,
                )
                LOGGER.info(
                    "transformer trainer fit start model_id=%s profile=%s loss_mode=%s",
                    model_id,
                    profile["name"],
                    loss_mode,
                )
                try:
                    trainer.train()
                except (RuntimeError, MPSFallbackRequested) as exc:
                    if effective_runtime_profile == "mps" and _should_retry_transformer_on_cpu(exc):
                        LOGGER.warning(
                            "transformer training leaving mps model_id=%s reason=%s retrying_runtime=cpu_fallback",
                            model_id,
                            str(exc),
                        )
                        _clear_torch_memory()
                        return _train_transformer_bundle_for_split(
                            split=split,
                            output_dir=output_dir,
                            model_id=model_id,
                            display_name=display_name,
                            prepared_data_summary=prepared_data_summary,
                            runtime_profile="cpu_fallback",
                            config_version=config_version,
                            evaluate_on_test=evaluate_on_test,
                        )
                    raise
                LOGGER.info(
                    "transformer trainer fit complete model_id=%s profile=%s loss_mode=%s",
                    model_id,
                    profile["name"],
                    loss_mode,
                )
                raw_calibration_scores = _positive_scores_from_logits(trainer.predict(calibration_dataset).predictions)
                candidate_pr_auc = _safe_average_precision(y_calibration, raw_calibration_scores)
                candidate = {
                    "profile": dict(profile),
                    "loss_mode": loss_mode,
                    "trainer": trainer,
                    "raw_calibration_scores": raw_calibration_scores,
                    "pr_auc": candidate_pr_auc,
                }
                candidate["candidate_metrics"] = {
                    **_classification_metrics(
                        y_calibration,
                        [1 if score >= 0.5 else 0 for score in raw_calibration_scores],
                    ),
                    "pr_auc": candidate_pr_auc,
                }
                candidate["constraint_metrics"] = _constraint_metrics_summary(y_calibration, raw_calibration_scores)
                candidate_results[f"{profile['name']}:{loss_mode}"] = {
                    "profile_name": str(profile["name"]),
                    "learning_rate": float(profile["learning_rate"]),
                    "weight_decay": float(profile["weight_decay"]),
                    "max_length": int(profile["max_length"]),
                    "loss_mode": loss_mode,
                    "pr_auc": float(candidate_pr_auc),
                    "auto_recall_at_precision_95": float(
                        candidate["constraint_metrics"]["auto_recall_at_precision_95"]["recall"]
                    ),
                    "review_recall_at_precision_75": float(
                        candidate["constraint_metrics"]["review_recall_at_precision_75"]["recall"]
                    ),
                }
                LOGGER.info(
                    "transformer candidate evaluated model_id=%s profile=%s loss_mode=%s pr_auc=%.3f auto95_recall=%.3f review75_recall=%.3f",
                    model_id,
                    profile["name"],
                    loss_mode,
                    candidate_pr_auc,
                    float(candidate["constraint_metrics"]["auto_recall_at_precision_95"]["recall"]),
                    float(candidate["constraint_metrics"]["review_recall_at_precision_75"]["recall"]),
                )
                if best_candidate is None or _transformer_candidate_key(candidate) > _transformer_candidate_key(best_candidate):
                    if best_candidate is not None:
                        _clear_torch_memory()
                    best_candidate = candidate
                else:
                    del trainer
                    del model
                    _clear_torch_memory()
    finally:
        transformers_logging.set_verbosity(previous_transformers_verbosity)

    if best_candidate is None:
        raise RuntimeError("transformer candidate search did not produce a candidate")

    trainer = best_candidate["trainer"]
    selected_loss_mode = str(best_candidate["loss_mode"])
    selected_profile = dict(best_candidate["profile"])
    max_length = int(selected_profile["max_length"])
    raw_calibration_scores = list(best_candidate["raw_calibration_scores"])
    if use_mps and hasattr(torch.mps, "empty_cache"):
        gc.collect()
        torch.mps.empty_cache()

    LOGGER.info(
        "transformer candidate selected model_id=%s profile=%s loss_mode=%s pr_auc=%.3f",
        model_id,
        selected_profile["name"],
        selected_loss_mode,
        float(best_candidate["pr_auc"]),
    )
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
        calibration=calibration,
    )

    model_dir = artifact_dir / "transformer_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    threshold_policy = _decision_policy(
        split=split,
        calibration=calibration,
        thresholds=thresholds,
        review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
    )
    metadata_path = artifact_dir / "transformer_metadata.json"
    bundle_path = artifact_dir / "transformer_bundle.joblib"
    training_args_summary = {
        "learning_rate": float(selected_profile["learning_rate"]),
        "candidate_profile": selected_profile,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "max_length": max_length,
        "input_format": "title_body_pair",
        "body_includes_metadata_tokens": True,
        "class_weighting": selected_loss_mode,
        "runtime_profile": effective_runtime_profile,
        "optimizer": "adafactor" if use_mps else "adamw_torch",
        "weight_decay": float(selected_profile["weight_decay"]),
        "trust_remote_code": trust_remote_code,
        "cuda_matmul": cuda_runtime,
        "representation_config": representation_config,
        "class_weights": {
            "not_askseattle": float(class_weights[0].item()),
            "askseattle": float(class_weights[1].item()),
        },
        "early_stopping": {"metric": "pr_auc", "patience": 1},
        "best_checkpoint_restored": True,
        "candidate_results": candidate_results,
    }
    joblib.dump(
        {
            "model_family": "transformer_sequence_classifier",
            "model_name": _slugify_model_name(display_name or f"transformer {Path(model_id).name}"),
            "display_name": display_name or model_id,
            "model_id": model_id,
            "artifact_path": _portable_artifact_reference(model_dir, base_dir=artifact_dir),
            "model_dir": _portable_artifact_reference(model_dir, base_dir=artifact_dir),
            "load_options": load_options,
            "calibrator": calibrator,
            "threshold_policy": threshold_policy,
            "representation_config": representation_config,
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
                "artifact_path": _portable_artifact_reference(model_dir, base_dir=artifact_dir),
                "bundle_path": _portable_artifact_reference(bundle_path, base_dir=artifact_dir),
                "calibrator_bundle_path": _portable_artifact_reference(bundle_path, base_dir=artifact_dir),
                "load_options": load_options,
                "threshold_policy": threshold_policy,
                "representation_config": representation_config,
                "training_args": training_args_summary,
                "version": __version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": __version__,
        "model_name": _slugify_model_name(display_name or f"transformer {Path(model_id).name}"),
        "model_family": "transformer_sequence_classifier",
        "runtime_environment": _runtime_environment_metadata(),
        "display_name": display_name or model_id,
        "model_id": model_id,
        "artifact_path": str(bundle_path),
        "model_dir": str(model_dir),
        "artifact_metadata_path": str(metadata_path),
        "representation_config": representation_config,
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(
            thresholds,
            review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
            high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        ),
        "threshold_policy": threshold_policy,
        "config_version": config_version,
        "training_args": training_args_summary,
        "training_balance": slice_weighting.summary,
        "benchmark_status": "not_run",
        "production_ready": False,
        "production_ready_blocked_reason": "benchmark_not_run",
    }
    if prepared_data_summary is not None:
        summary["prepared_data"] = prepared_data_summary
    if evaluate_on_test:
        test_rows = _sequence_classification_rows(split.test, representation_config=representation_config)
        test_inference_rows = _inference_rows(split.test, representation_config=representation_config)
        y_test = [row["label"] for row in test_rows]
        previous_transformers_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            LOGGER.info("transformer tokenization start model_id=%s split=test examples=%s", model_id, len(test_rows))
            test_dataset = Dataset.from_list(test_rows).map(tokenize, batched=True)
        finally:
            transformers_logging.set_verbosity(previous_transformers_verbosity)
        raw_test_scores = _positive_scores_from_logits(trainer.predict(test_dataset).predictions)
        LOGGER.info("transformer test scoring complete model_id=%s examples=%s", model_id, len(test_rows))
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
            train_rows=train_inference_rows,
            train_labels=[post.label for post in split.train],
            rows=test_inference_rows,
            low_threshold=thresholds.low_threshold,
            high_threshold=thresholds.high_threshold,
        )
        production_ready, blocked_reason = _production_ready_status(
            calibration=calibration,
            thresholds=thresholds,
            high_confidence_precision=band_metrics.high_confidence_precision,
            high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
        )
        summary.update(
            {
                "ranking_metrics": _ranking_metrics_summary(y_test, calibrated_test_scores),
                "constraint_metrics": _constraint_metrics_summary(y_test, calibrated_test_scores),
                "metrics": {
                    "high_confidence_precision": band_metrics.high_confidence_precision,
                    "high_confidence_recall": band_metrics.high_confidence_recall,
                    "high_confidence_f1": band_metrics.high_confidence_f1,
                    "support": band_metrics.support,
                    "confidence_band_counts": band_metrics.band_counts,
                },
                "operating_metrics": asdict(operating_metrics),
                "benchmark_status": "complete",
                "production_ready": production_ready,
                "production_ready_blocked_reason": blocked_reason,
            }
        )
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if evaluate_on_test:
        LOGGER.info(
            "transformer training complete model_id=%s artifact=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
            model_id,
            str(bundle_path),
            float(summary["operating_metrics"]["auto_band"]["precision"]),
            float(summary["operating_metrics"]["auto_band"]["recall"]),
            float(summary["operating_metrics"]["review_queue"]["precision"]),
            float(summary["operating_metrics"]["review_queue"]["recall"]),
            _format_elapsed(time.perf_counter() - started_at),
        )
    else:
        LOGGER.info(
            "transformer training complete model_id=%s artifact=%s calibration_available=%s high_threshold=%s elapsed=%s",
            model_id,
            str(bundle_path),
            calibration.available,
            thresholds.high_threshold,
            _format_elapsed(time.perf_counter() - started_at),
        )
    return summary


def _train_causal_lm_bundle_for_split(
    *,
    split: DatasetSplit,
    output_dir: str | Path,
    model_id: str,
    display_name: str | None = None,
    prepared_data_summary: dict[str, int] | None = None,
    runtime_profile: str | None = None,
    prompt_template_version: str = DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
    config_version: str = "v2_compact_prompt_two_epoch",
    evaluate_on_test: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    except ImportError as exc:
        raise OptionalModelDependencyError(
            "Causal-LM benchmarks require transformers, datasets, torch, and peft. "
            "Install with `python -m pip install -e \".[dev,models]\"`."
        ) from exc

    started_at = time.perf_counter()
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    detected_runtime = _torch_runtime_device(torch)
    cuda_runtime = _enable_cuda_tf32(torch)
    effective_runtime_profile = _resolve_causal_lm_runtime_profile(
        detected_runtime=detected_runtime,
        requested_runtime_profile=runtime_profile,
        model_id=model_id,
    )
    device = "cpu" if effective_runtime_profile == "cpu_fallback" else effective_runtime_profile
    use_mps = device == "mps"
    model_dtype = torch.float16 if use_mps else None
    prompt_max_length = 224 if use_mps else 320
    target_max_length = 24 if use_mps else 32
    sequence_max_length = 320 if use_mps else 448
    gradient_accumulation_steps = 8 if use_mps else 4
    if effective_runtime_profile == "cpu_fallback":
        prompt_max_length = 224
        target_max_length = 24
        sequence_max_length = 320
        gradient_accumulation_steps = 8
        if detected_runtime == "mps" and runtime_profile is None:
            LOGGER.info(
                "causal lm training bypassing mps model_id=%s reason=%s fallback_runtime=cpu_fallback",
                model_id,
                "qwen causal-lm fine-tuning is not stable on this MPS stack",
            )
    LOGGER.info(
        "causal lm training start model_id=%s display_name=%s output_dir=%s runtime_profile=%s train=%s calibration=%s test=%s candidate_count=%s grad_accum=%s sequence_max_length=%s dtype=%s",
        model_id,
        display_name or model_id,
        str(artifact_dir),
        effective_runtime_profile,
        len(split.train),
        len(split.calibration),
        len(split.test),
        len(_causal_lm_candidate_profiles()),
        gradient_accumulation_steps,
        sequence_max_length,
        str(model_dtype).replace("torch.", "") if model_dtype is not None else "default",
    )

    representation_config = _representation_config_for_split(split)
    train_inference_rows = _inference_rows(split.train, representation_config=representation_config)
    slice_weighting = _slice_aware_positive_weighting(split.train, rows=train_inference_rows)
    y_calibration = [post.label for post in split.calibration]

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("causal lm tokenizer ready model_id=%s prompt_max_length=%s target_max_length=%s", model_id, prompt_max_length, target_max_length)

    class WeightedCausalLMTrainer(Trainer):
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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            valid_mask = shift_labels.ne(-100)
            token_loss = token_loss * valid_mask
            per_example = token_loss.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            if example_weight is not None:
                per_example = per_example * example_weight.to(per_example.device).view(-1)
            loss = per_example.mean()
            if return_outputs:
                return loss, outputs
            return loss

    class ProgressLoggingCallback(TrainerCallback):
        def __init__(self, label: str) -> None:
            self.label = label
            self._last_logged_step = 0
            self._initial_memory_snapshot = _mps_memory_snapshot(torch) if use_mps else None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            LOGGER.info("%s trainer started total_steps=%s", self.label, state.max_steps)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step or 0)
            total = int(state.max_steps or 0)
            interval = _progress_log_interval(total)
            if use_mps and step and step % 4 == 0 and hasattr(torch.mps, "empty_cache"):
                gc.collect()
                torch.mps.empty_cache()
            if use_mps and step and step % 8 == 0:
                snapshot = _mps_memory_snapshot(torch)
                if snapshot is not None and self._initial_memory_snapshot is not None:
                    LOGGER.info(
                        "%s mps memory step=%s current=%s driver=%s available=%s recommended=%s",
                        self.label,
                        step,
                        _format_bytes(snapshot["current_allocated_bytes"]),
                        _format_bytes(snapshot["driver_allocated_bytes"]),
                        _format_bytes(snapshot["available_system_bytes"]),
                        _format_bytes(snapshot["recommended_max_bytes"]),
                    )
                    if _should_proactively_fallback_from_mps(
                        initial_snapshot=self._initial_memory_snapshot,
                        current_snapshot=snapshot,
                    ):
                        raise MPSFallbackRequested(
                            "mps memory headroom dropped below the safety threshold; switching to cpu_fallback"
                        )
            if step and (step == 1 or step == total or step - self._last_logged_step >= interval):
                self._last_logged_step = step
                LOGGER.info("%s trainer progress step=%s/%s epoch=%.2f", self.label, step, total, float(state.epoch or 0.0))

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            LOGGER.info("%s trainer finished total_steps=%s", self.label, state.max_steps)

    best_candidate: dict[str, Any] | None = None
    candidate_results: dict[str, dict[str, Any]] = {}
    for candidate_profile in _causal_lm_candidate_profiles():
        candidate_prompt_template_version = str(candidate_profile["prompt_template_version"])
        candidate_train_rows = _causal_lm_rows(
            split.train,
            example_weights=slice_weighting.sample_weights,
            prompt_template_version=candidate_prompt_template_version,
            representation_config=representation_config,
        )
        candidate_calibration_rows = _causal_lm_rows(
            split.calibration,
            prompt_template_version=candidate_prompt_template_version,
            representation_config=representation_config,
        )
        candidate_train_dataset = _causal_lm_training_dataset(
            candidate_train_rows,
            tokenizer=tokenizer,
            prompt_max_length=prompt_max_length,
            target_max_length=target_max_length,
            sequence_max_length=sequence_max_length,
        )
        calibration_prompts = [row["prompt"] for row in candidate_calibration_rows]
        LOGGER.info(
            "causal lm candidate start model_id=%s candidate=%s prompt_template=%s lr=%s lora_rank=%s epochs=%s",
            model_id,
            candidate_profile["name"],
            candidate_prompt_template_version,
            float(candidate_profile["learning_rate"]),
            int(candidate_profile["lora_rank"]),
            int(candidate_profile["num_train_epochs"]),
        )
        causal_lm_load_kwargs: dict[str, Any] = {}
        if model_dtype is not None:
            causal_lm_load_kwargs["dtype"] = model_dtype
        base_model = AutoModelForCausalLM.from_pretrained(model_id, **causal_lm_load_kwargs)
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(candidate_profile["lora_rank"]),
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
        )
        model = get_peft_model(base_model, lora_config)
        training_args = TrainingArguments(
            output_dir=str(artifact_dir / f"checkpoints_{candidate_profile['name']}"),
            learning_rate=float(candidate_profile["learning_rate"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=int(candidate_profile["num_train_epochs"]),
            weight_decay=0.0,
            save_strategy="no",
            logging_strategy="no",
            optim="adafactor" if use_mps else "adamw_torch",
            report_to=[],
            use_cpu=(device == "cpu"),
            remove_unused_columns=False,
        )
        trainer = WeightedCausalLMTrainer(
            model=model,
            args=training_args,
            train_dataset=candidate_train_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressLoggingCallback(f"causal lm {display_name or model_id} {candidate_profile['name']}")],
        )
        LOGGER.info("causal lm trainer fit start model_id=%s candidate=%s", model_id, candidate_profile["name"])
        try:
            trainer.train()
        except (RuntimeError, MPSFallbackRequested) as exc:
            if effective_runtime_profile == "mps" and _should_retry_transformer_on_cpu(exc):
                LOGGER.warning(
                    "causal lm training leaving mps model_id=%s reason=%s retrying_runtime=cpu_fallback",
                    model_id,
                    str(exc),
                )
                _clear_torch_memory()
                return _train_causal_lm_bundle_for_split(
                    split=split,
                    output_dir=output_dir,
                    model_id=model_id,
                    display_name=display_name,
                    prepared_data_summary=prepared_data_summary,
                    runtime_profile="cpu_fallback",
                    prompt_template_version=prompt_template_version,
                    config_version=config_version,
                    evaluate_on_test=evaluate_on_test,
                )
            raise
        LOGGER.info("causal lm trainer fit complete model_id=%s candidate=%s", model_id, candidate_profile["name"])

        merged_model = trainer.model.merge_and_unload()
        candidate_model_dir = artifact_dir / f"causal_lm_model_{candidate_profile['name']}"
        if candidate_model_dir.exists():
            shutil.rmtree(candidate_model_dir)
        merged_model.save_pretrained(str(candidate_model_dir))
        tokenizer.save_pretrained(str(candidate_model_dir))

        raw_calibration_scores = _causal_lm_candidate_probabilities(
            merged_model,
            tokenizer,
            calibration_prompts,
            device=device,
        )
        candidate_metrics = _classification_metrics(
            y_calibration,
            [1 if score >= 0.5 else 0 for score in raw_calibration_scores],
        )
        candidate_metrics["pr_auc"] = _safe_average_precision(y_calibration, raw_calibration_scores)
        constraint_metrics = _constraint_metrics_summary(y_calibration, raw_calibration_scores)
        candidate = {
            "profile": dict(candidate_profile),
            "prompt_template_version": candidate_prompt_template_version,
            "model": merged_model,
            "model_dir": candidate_model_dir,
            "raw_calibration_scores": raw_calibration_scores,
            "candidate_metrics": candidate_metrics,
            "constraint_metrics": constraint_metrics,
        }
        candidate_results[str(candidate_profile["name"])] = {
            "prompt_template_version": candidate_prompt_template_version,
            "learning_rate": float(candidate_profile["learning_rate"]),
            "lora_rank": int(candidate_profile["lora_rank"]),
            "num_train_epochs": int(candidate_profile["num_train_epochs"]),
            "pr_auc": float(candidate_metrics["pr_auc"]),
            "auto_recall_at_precision_95": float(constraint_metrics["auto_recall_at_precision_95"]["recall"]),
            "review_recall_at_precision_75": float(constraint_metrics["review_recall_at_precision_75"]["recall"]),
        }
        LOGGER.info(
            "causal lm candidate evaluated model_id=%s candidate=%s pr_auc=%.3f auto95_recall=%.3f review75_recall=%.3f",
            model_id,
            candidate_profile["name"],
            float(candidate_metrics["pr_auc"]),
            float(constraint_metrics["auto_recall_at_precision_95"]["recall"]),
            float(constraint_metrics["review_recall_at_precision_75"]["recall"]),
        )
        if best_candidate is None or _precision_first_candidate_key(candidate) > _precision_first_candidate_key(best_candidate):
            if best_candidate is not None:
                previous_model_dir = Path(best_candidate["model_dir"])
                if previous_model_dir.exists():
                    shutil.rmtree(previous_model_dir)
                previous_model = best_candidate.get("model")
                if previous_model is not None and hasattr(previous_model, "to"):
                    previous_model.to("cpu")
            best_candidate = candidate
        else:
            if candidate_model_dir.exists():
                shutil.rmtree(candidate_model_dir)
            if hasattr(merged_model, "to"):
                merged_model.to("cpu")
        del trainer
        del model
        del base_model
        _clear_torch_memory()

    if best_candidate is None:
        raise RuntimeError("causal lm tuning did not produce a candidate")

    raw_calibration_scores = list(best_candidate["raw_calibration_scores"])
    calibrator, calibration = fit_sigmoid_calibrator(y_calibration, raw_calibration_scores)
    thresholds = _select_thresholds_or_default(
        y_calibration,
        apply_probability_calibrator(calibrator, raw_calibration_scores),
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
        calibration=calibration,
    )

    threshold_policy = _decision_policy(
        split=split,
        calibration=calibration,
        thresholds=thresholds,
        review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
        high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
    )
    selected_prompt_template_version = str(best_candidate["prompt_template_version"])
    selected_profile = dict(best_candidate["profile"])
    model_dir = artifact_dir / "causal_lm_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.move(str(best_candidate["model_dir"]), str(model_dir))
    artifact_path = artifact_dir / "causal_lm_bundle.joblib"
    training_args_summary = {
        "learning_rate": float(selected_profile["learning_rate"]),
        "candidate_profile": selected_profile,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": int(selected_profile["num_train_epochs"]),
        "max_length": sequence_max_length,
        "prompt_template_version": selected_prompt_template_version,
        "runtime_profile": effective_runtime_profile,
        "torch_dtype": str(model_dtype).replace("torch.", "") if model_dtype is not None else "default",
        "cuda_matmul": cuda_runtime,
        "optimizer": "adafactor" if use_mps else "adamw_torch",
        "representation_config": representation_config,
        "lora": {
            "r": int(selected_profile["lora_rank"]),
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": "all-linear",
        },
        "candidate_results": candidate_results,
    }
    joblib.dump(
        {
            "model_family": "causal_lm_classifier",
            "model_name": _slugify_model_name(display_name or f"decoder {Path(model_id).name}"),
            "display_name": display_name or model_id,
            "model_id": model_id,
            "artifact_path": _portable_artifact_reference(model_dir, base_dir=artifact_dir),
            "model_dir": _portable_artifact_reference(model_dir, base_dir=artifact_dir),
            "calibrator": calibrator,
            "threshold_policy": threshold_policy,
            "training_args": training_args_summary,
            "representation_config": representation_config,
            "prompt_template_version": selected_prompt_template_version,
            "version": __version__,
        },
        artifact_path,
    )

    summary = {
        "version": __version__,
        "model_name": _slugify_model_name(display_name or f"decoder {Path(model_id).name}"),
        "model_family": "causal_lm_classifier",
        "runtime_environment": _runtime_environment_metadata(),
        "display_name": display_name or model_id,
        "model_id": model_id,
        "artifact_path": str(artifact_path),
        "model_dir": str(model_dir),
        "representation_config": representation_config,
        "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
        "production_gate": _production_gate_summary(),
        "split": _split_summary(split),
        "calibration": asdict(calibration),
        "threshold_selection": _threshold_summary(
            thresholds,
            review_precision_target=DEFAULT_REVIEW_PRECISION_TARGET,
            high_precision_target=DEFAULT_HIGH_PRECISION_TARGET,
        ),
        "threshold_policy": threshold_policy,
        "config_version": config_version,
        "prompt_template_version": selected_prompt_template_version,
        "training_args": training_args_summary,
        "training_balance": slice_weighting.summary,
        "benchmark_status": "not_run",
        "production_ready": False,
        "production_ready_blocked_reason": "benchmark_not_run",
    }
    if prepared_data_summary is not None:
        summary["prepared_data"] = prepared_data_summary
    if evaluate_on_test:
        test_rows = _causal_lm_rows(
            split.test,
            prompt_template_version=selected_prompt_template_version,
            representation_config=representation_config,
        )
        test_prompts = [row["prompt"] for row in test_rows]
        test_inference_rows = _inference_rows(split.test, representation_config=representation_config)
        y_test = [row["label"] for row in test_rows]
        LOGGER.info("causal lm test scoring start model_id=%s examples=%s", model_id, len(test_prompts))
        raw_test_scores = _causal_lm_candidate_probabilities(
            best_candidate["model"],
            tokenizer,
            test_prompts,
            device=device,
        )
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
            train_rows=train_inference_rows,
            train_labels=[post.label for post in split.train],
            rows=test_inference_rows,
            low_threshold=thresholds.low_threshold,
            high_threshold=thresholds.high_threshold,
        )
        production_ready, blocked_reason = _production_ready_status(
            calibration=calibration,
            thresholds=thresholds,
            high_confidence_precision=band_metrics.high_confidence_precision,
            high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
        )
        summary.update(
            {
                "ranking_metrics": _ranking_metrics_summary(y_test, calibrated_test_scores),
                "constraint_metrics": _constraint_metrics_summary(y_test, calibrated_test_scores),
                "metrics": {
                    "high_confidence_precision": band_metrics.high_confidence_precision,
                    "high_confidence_recall": band_metrics.high_confidence_recall,
                    "high_confidence_f1": band_metrics.high_confidence_f1,
                    "support": band_metrics.support,
                    "confidence_band_counts": band_metrics.band_counts,
                },
                "operating_metrics": asdict(operating_metrics),
                "benchmark_status": "complete",
                "production_ready": production_ready,
                "production_ready_blocked_reason": blocked_reason,
            }
        )
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if evaluate_on_test:
        LOGGER.info(
            "causal lm training complete model_id=%s artifact=%s auto_precision=%.3f auto_recall=%.3f review_precision=%.3f review_recall=%.3f elapsed=%s",
            model_id,
            str(artifact_path),
            float(summary["operating_metrics"]["auto_band"]["precision"]),
            float(summary["operating_metrics"]["auto_band"]["recall"]),
            float(summary["operating_metrics"]["review_queue"]["precision"]),
            float(summary["operating_metrics"]["review_queue"]["recall"]),
            _format_elapsed(time.perf_counter() - started_at),
        )
    else:
        LOGGER.info(
            "causal lm training complete model_id=%s artifact=%s calibration_available=%s high_threshold=%s elapsed=%s",
            model_id,
            str(artifact_path),
            calibration.available,
            thresholds.high_threshold,
            _format_elapsed(time.perf_counter() - started_at),
        )
    if hasattr(best_candidate["model"], "to"):
        best_candidate["model"].to("cpu")
    _clear_torch_memory()
    return summary


def _causal_lm_rows(
    posts: list[LabeledPost],
    *,
    example_weights: list[float] | None = None,
    prompt_template_version: str = DEFAULT_CAUSAL_LM_PROMPT_TEMPLATE_VERSION,
    representation_config: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_weights = example_weights or [1.0] * len(posts)
    resolved_representation_config = representation_config or {
        "include_sparse_media_token": DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
        "include_image_low_text_tokens": DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    }
    for post, example_weight in zip(posts, resolved_weights, strict=True):
        inference_row = build_inference_row(
            title=post.title,
            selftext=post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
            include_sparse_media_token=resolved_representation_config["include_sparse_media_token"],
            include_image_low_text_tokens=resolved_representation_config["include_image_low_text_tokens"],
        )
        prompt = causal_lm_prompt_for_row(
            inference_row,
            prompt_template_version=prompt_template_version,
        )
        rows.append(
            {
                "prompt": prompt,
                "target": f" {('askseattle' if post.label == 1 else 'not_askseattle')}",
                "label": post.label,
                "example_weight": float(example_weight),
            }
        )
    return rows


def _causal_lm_training_dataset(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    prompt_max_length: int,
    target_max_length: int,
    sequence_max_length: int,
) -> Dataset:
    original_columns = list(rows[0].keys()) if rows else []
    return Dataset.from_list(rows).map(
        _tokenize_causal_lm_training_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prompt_max_length": prompt_max_length,
            "target_max_length": target_max_length,
            "sequence_max_length": sequence_max_length,
        },
        remove_columns=original_columns,
    )


def _tokenize_causal_lm_training_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    prompt_max_length: int,
    target_max_length: int,
    sequence_max_length: int,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        row["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
    )["input_ids"]
    target_ids = tokenizer(
        row["target"],
        add_special_tokens=False,
        truncation=True,
        max_length=target_max_length,
    )["input_ids"] + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + target_ids)[-sequence_max_length:]
    attention_mask = [1] * len(input_ids)
    label_count = min(len(target_ids), len(input_ids))
    labels = [-100] * (len(input_ids) - label_count) + input_ids[-label_count:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_weight": float(row.get("example_weight", 1.0)),
    }


def _causal_lm_prompt(text: str) -> str:
    return (
        "Classify the following Reddit post as askseattle or not_askseattle.\n"
        "Respond with exactly one label.\n\n"
        f"{text}\n\n"
        "Label:"
    )


def _causal_lm_candidate_probabilities(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    device: str,
) -> list[float]:
    LOGGER.info(
        "causal lm candidate scoring start prompts=%s device=%s completions=2",
        len(prompts),
        device,
    )
    started_at = time.perf_counter()
    model.to(device)
    model.eval()
    ask_scores = _causal_lm_completion_scores(
        model,
        tokenizer,
        prompts,
        completion=" askseattle",
        device=device,
    )
    not_scores = _causal_lm_completion_scores(
        model,
        tokenizer,
        prompts,
        completion=" not_askseattle",
        device=device,
    )
    probabilities: list[float] = []
    for ask_score, not_score in zip(ask_scores, not_scores, strict=True):
        stabilizer = max(ask_score, not_score)
        ask_prob = float(np.exp(ask_score - stabilizer))
        not_prob = float(np.exp(not_score - stabilizer))
        probabilities.append(ask_prob / (ask_prob + not_prob))
    LOGGER.info(
        "causal lm candidate scoring complete prompts=%s device=%s elapsed=%s",
        len(prompts),
        device,
        _format_elapsed(time.perf_counter() - started_at),
    )
    return probabilities


def _causal_lm_completion_scores(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    completion: str,
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
    representation_config: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_weights = example_weights or [1.0] * len(posts)
    resolved_representation_config = representation_config or {
        "include_sparse_media_token": DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
        "include_image_low_text_tokens": DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    }
    for post, example_weight in zip(posts, resolved_weights, strict=True):
        inference_row = build_inference_row(
            title=post.title,
            selftext=post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
            include_sparse_media_token=resolved_representation_config["include_sparse_media_token"],
            include_image_low_text_tokens=resolved_representation_config["include_image_low_text_tokens"],
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


def _texts(
    posts: list[LabeledPost],
    *,
    representation_config: dict[str, bool] | None = None,
) -> list[str]:
    resolved_representation_config = representation_config or {
        "include_sparse_media_token": DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
        "include_image_low_text_tokens": DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    }
    return [
        post_text(
            post.title,
            post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
            include_sparse_media_token=resolved_representation_config["include_sparse_media_token"],
            include_image_low_text_tokens=resolved_representation_config["include_image_low_text_tokens"],
        )
        for post in posts
    ]


def _inference_rows(
    posts: list[LabeledPost],
    *,
    representation_config: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    resolved_representation_config = representation_config or {
        "include_sparse_media_token": DEFAULT_INCLUDE_SPARSE_MEDIA_TOKEN,
        "include_image_low_text_tokens": DEFAULT_INCLUDE_IMAGE_LOW_TEXT_TOKENS,
    }
    return [
        build_inference_row(
            title=post.title,
            selftext=post.selftext,
            post_type=post.post_type,
            content_domain=post.content_domain,
            is_crosspost=post.is_crosspost,
            include_sparse_media_token=resolved_representation_config["include_sparse_media_token"],
            include_image_low_text_tokens=resolved_representation_config["include_image_low_text_tokens"],
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
    max_slice_positive_weight: float = DEFAULT_MAX_SLICE_POSITIVE_WEIGHT,
) -> SliceAwareWeighting:
    resolved_rows = rows or _inference_rows(posts)
    labels = [post.label for post in posts]
    positive_bucket_counts: dict[str, Counter[str]] = {
        "image_post": Counter(),
        "low_text": Counter(),
    }

    for label, row in zip(labels, resolved_rows, strict=True):
        if label != 1:
            continue
        for slice_name, bucket in _weighting_bucket_values(row).items():
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
                    max_slice_positive_weight,
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
            for slice_name, bucket in _weighting_bucket_values(row).items()
        ]
        sample_weights.append(max([1.0, *row_bucket_weights]))

    positive_weights = [weight for weight, label in zip(sample_weights, labels, strict=True) if label == 1]
    summary = {
        "strategy": "slice_aware_positive_weighting",
        "max_slice_positive_weight": float(max_slice_positive_weight),
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


def _weighting_bucket_values(row: dict[str, Any]) -> dict[str, str]:
    return {
        "image_post": "yes" if row.get("post_type") == "image" else "no",
        "low_text": "yes" if row.get("body_length_bucket") in {"none", "short"} else "no",
    }


def _suite_entry_from_summary(
    spec: SuiteModelSpec,
    summary: dict[str, Any],
    *,
    result_source: str = "trained",
) -> dict[str, Any]:
    artifact_path = Path(summary["artifact_path"])
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "status": "ok",
        "result_source": result_source,
        "model_name": summary["model_name"],
        "model_family": summary["model_family"],
        "model_id": summary.get("model_id"),
        "artifact_path": summary["artifact_path"],
        "summary_path": str((artifact_path if artifact_path.is_dir() else artifact_path.parent) / "training_summary.json"),
        "production_ready": summary["production_ready"],
        "production_ready_blocked_reason": summary["production_ready_blocked_reason"],
        "production_gate": summary.get("production_gate"),
        "metrics": summary["metrics"],
        "operating_metrics": summary["operating_metrics"],
        "constraint_metrics": summary.get("constraint_metrics", _constraint_metrics_template()),
        "ranking_metrics": summary.get("ranking_metrics", {"pr_auc": 0.0}),
        "threshold_policy": summary["threshold_policy"],
    }


def _suite_training_entry_from_summary(
    spec: SuiteModelSpec,
    summary: dict[str, Any],
    *,
    result_source: str,
) -> dict[str, Any]:
    artifact_path = Path(summary["artifact_path"])
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "status": "trained",
        "result_source": result_source,
        "model_name": summary["model_name"],
        "model_family": summary["model_family"],
        "model_id": summary.get("model_id"),
        "artifact_path": summary["artifact_path"],
        "summary_path": str((artifact_path if artifact_path.is_dir() else artifact_path.parent) / "training_summary.json"),
        "benchmark_status": summary.get("benchmark_status", "not_run"),
    }


def _suite_skipped_entry(spec: SuiteModelSpec, *, reason: str, error: str) -> dict[str, Any]:
    config = spec.kwargs.get("config")
    model_id = getattr(config, "model_id", None) or spec.kwargs.get("model_id")
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "model_family": spec.family,
        "model_id": model_id,
        "status": "skipped",
        "reason": reason,
        "error": error,
    }


def _benchmark_existing_suite_model(
    *,
    spec: SuiteModelSpec,
    split: DatasetSplit,
    prepared_data_summary: dict[str, int],
    output_dir: Path,
    trained_summary: dict[str, Any],
) -> dict[str, Any]:
    artifact_path = _resolve_suite_artifact_path(trained_summary, output_dir)
    if artifact_path is None or not artifact_path.exists():
        raise FileNotFoundError(f"missing trained artifact for {spec.name}")

    try:
        bundle = load_model(artifact_path)
    except ValueError as exc:
        if "Install optional model dependencies" in str(exc):
            raise OptionalModelDependencyError(str(exc)) from exc
        raise

    representation_config = (
        trained_summary.get("representation_config")
        if isinstance(trained_summary.get("representation_config"), dict)
        else bundle.get("representation_config")
        if isinstance(bundle.get("representation_config"), dict)
        else _representation_config_for_split(split)
    )
    test_rows = _inference_rows(split.test, representation_config=representation_config)
    y_test = [post.label for post in split.test]
    threshold_policy = trained_summary.get("threshold_policy") or bundle.get("threshold_policy") or {}
    low_threshold = float(
        threshold_policy.get("low_threshold")
        or bundle.get("low_threshold")
        or bundle.get("threshold")
        or 0.85
    )
    high_threshold = float(
        threshold_policy.get("high_threshold")
        or bundle.get("high_threshold")
        or bundle.get("threshold")
        or low_threshold
    )
    calibrated_test_scores = score_rows(bundle, test_rows)
    band_metrics = evaluate_decision_policy(
        y_test,
        calibrated_test_scores,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        rows=test_rows,
    )
    operating_metrics = _operating_metrics_summary(
        y_test,
        calibrated_test_scores,
        train_rows=_inference_rows(split.train, representation_config=representation_config),
        train_labels=[post.label for post in split.train],
        rows=test_rows,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    calibration = _calibration_result_from_summary(trained_summary.get("calibration"))
    production_ready, blocked_reason = _production_ready_status_from_trained_summary(
        calibration=calibration,
        calibration_high_threshold_ready=_threshold_selection_ready_from_summary(
            trained_summary.get("threshold_selection")
        ),
        high_confidence_precision=band_metrics.high_confidence_precision,
        high_confidence_predictions=int(operating_metrics.auto_band["predicted_positive"]),
    )
    benchmarked_summary = dict(trained_summary)
    benchmarked_summary.update(
        {
            "high_precision_target": DEFAULT_HIGH_PRECISION_TARGET,
            "production_gate": _production_gate_summary(),
            "runtime_environment": _runtime_environment_metadata(),
            "split": _split_summary(split),
            "representation_config": representation_config,
            "ranking_metrics": _ranking_metrics_summary(y_test, calibrated_test_scores),
            "constraint_metrics": _constraint_metrics_summary(y_test, calibrated_test_scores),
            "metrics": {
                "high_confidence_precision": band_metrics.high_confidence_precision,
                "high_confidence_recall": band_metrics.high_confidence_recall,
                "high_confidence_f1": band_metrics.high_confidence_f1,
                "support": band_metrics.support,
                "confidence_band_counts": band_metrics.band_counts,
            },
            "operating_metrics": asdict(operating_metrics),
            "production_ready": production_ready,
            "production_ready_blocked_reason": blocked_reason,
            "benchmark_status": "complete",
        }
    )
    if prepared_data_summary is not None:
        benchmarked_summary["prepared_data"] = prepared_data_summary
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(benchmarked_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return benchmarked_summary


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
    train_rows: list[dict[str, Any]],
    train_labels: list[int],
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
            train_rows=train_rows,
            train_labels=train_labels,
            y_true=y_true,
            probabilities=probabilities,
            rows=rows,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        ),
    )


def _slice_metrics_summary(
    *,
    train_rows: list[dict[str, Any]],
    train_labels: list[int],
    y_true: list[int],
    probabilities: list[float],
    rows: list[dict[str, Any]],
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    slices: dict[str, dict[str, Any]] = {}
    support = _slice_support_summary(
        train_rows=train_rows,
        train_labels=train_labels,
        test_rows=rows,
        test_labels=y_true,
    )
    slices["post_type"] = _slice_group_summary(
        slice_name="post_type",
        y_true=y_true,
        probabilities=probabilities,
        rows=rows,
        buckets={
            "self": lambda row: row.get("post_type") == "self",
            "link": lambda row: row.get("post_type") == "link",
            "image": lambda row: row.get("post_type") == "image",
            "other_or_unknown": lambda row: row.get("post_type") not in {"self", "link", "image"},
        },
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        support=support["post_type"],
    )
    slices["low_text"] = _slice_group_summary(
        slice_name="low_text",
        y_true=y_true,
        probabilities=probabilities,
        rows=rows,
        buckets={
            "yes": lambda row: row.get("body_length_bucket") in {"none", "short"},
            "no": lambda row: row.get("body_length_bucket") not in {"none", "short"},
        },
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        support=support["low_text"],
    )
    slices["sparse_media"] = _slice_group_summary(
        slice_name="sparse_media",
        y_true=y_true,
        probabilities=probabilities,
        rows=rows,
        buckets={
            "yes": lambda row: bool(row.get("is_sparse_media")),
            "no": lambda row: not bool(row.get("is_sparse_media")),
        },
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        support=support["sparse_media"],
    )
    return slices


def _slice_group_summary(
    *,
    slice_name: str,
    y_true: list[int],
    probabilities: list[float],
    rows: list[dict[str, Any]],
    buckets: dict[str, Any],
    low_threshold: float,
    high_threshold: float,
    support: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "support_status": support["support_status"],
        "support": support["support"],
        "buckets": {},
    }
    for name, predicate in buckets.items():
        indices = [index for index, row in enumerate(rows) if predicate(row)]
        subset_rows = [rows[index] for index in indices]
        subset_y_true = [y_true[index] for index in indices]
        subset_probabilities = [probabilities[index] for index in indices]
        summary["buckets"][name] = _operating_metrics_without_slices(
            subset_y_true,
            subset_probabilities,
            rows=subset_rows,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
    return summary


def _slice_support_summary(
    *,
    train_rows: list[dict[str, Any]],
    train_labels: list[int],
    test_rows: list[dict[str, Any]],
    test_labels: list[int],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for slice_name, rows_and_labels in {
        "post_type": (train_rows, train_labels, test_rows, test_labels),
        "low_text": (train_rows, train_labels, test_rows, test_labels),
        "sparse_media": (train_rows, train_labels, test_rows, test_labels),
    }.items():
        train_bucket_counts = _positive_bucket_counts(slice_name, train_rows, train_labels)
        test_bucket_counts = _positive_bucket_counts(slice_name, test_rows, test_labels)
        sparse_train_yes = int(train_bucket_counts.get("yes", 0))
        sparse_test_yes = int(test_bucket_counts.get("yes", 0))
        support_status = (
            "active"
            if slice_name != "sparse_media"
            or (
                sparse_train_yes >= SPARSE_MEDIA_ACTIVE_TRAIN_POSITIVES
                and sparse_test_yes >= SPARSE_MEDIA_ACTIVE_TEST_POSITIVES
            )
            else "observational"
        )
        summary[slice_name] = {
            "support_status": support_status,
            "support": {
                "train_positive_total": int(sum(train_bucket_counts.values())),
                "test_positive_total": int(sum(test_bucket_counts.values())),
                "train_positive_by_bucket": dict(train_bucket_counts),
                "test_positive_by_bucket": dict(test_bucket_counts),
            },
        }
    return summary


def _positive_bucket_counts(slice_name: str, rows: list[dict[str, Any]], labels: list[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row, label in zip(rows, labels, strict=True):
        if label != 1:
            continue
        counts[_slice_bucket_value(slice_name, row)] += 1
    return counts


def _slice_bucket_value(slice_name: str, row: dict[str, Any]) -> str:
    if slice_name == "post_type":
        post_type = str(row.get("post_type") or "").strip().lower()
        return post_type if post_type in {"self", "link", "image"} else "other_or_unknown"
    if slice_name == "low_text":
        return "yes" if row.get("body_length_bucket") in {"none", "short"} else "no"
    if slice_name == "sparse_media":
        return "yes" if bool(row.get("is_sparse_media")) else "no"
    raise ValueError(f"Unsupported slice name: {slice_name}")


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
    if not y_true:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_positive": int(sum(predictions)),
            "support": 0,
        }
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


def _ranking_metrics_summary(y_true: list[int], probabilities: list[float]) -> dict[str, float]:
    return {"pr_auc": _safe_average_precision(y_true, probabilities)}


def _constraint_metrics_summary(y_true: list[int], probabilities: list[float]) -> dict[str, dict[str, float | int | bool | None]]:
    return {
        "auto_recall_at_precision_95": _best_recall_at_precision(
            y_true,
            probabilities,
            precision_target=0.95,
        ),
        "review_recall_at_precision_75": _best_recall_at_precision(
            y_true,
            probabilities,
            precision_target=0.75,
        ),
    }


def _best_recall_at_precision(
    y_true: list[int],
    probabilities: list[float],
    *,
    precision_target: float,
) -> dict[str, float | int | bool | None]:
    sweep = threshold_sweep(y_true, probabilities)
    ready = [row for row in sweep if float(row["precision"]) >= precision_target]
    if not ready:
        return {
            "precision_target": precision_target,
            "threshold": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_positive": 0,
            "support": Counter(y_true).get(1, 0),
            "target_met": False,
        }
    selected = max(
        ready,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            float(row["f1"]),
            -float(row["threshold"]),
        ),
    )
    threshold = float(selected["threshold"])
    predicted_positive = int(sum(1 for probability in probabilities if probability >= threshold))
    return {
        "precision_target": precision_target,
        "threshold": threshold,
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f1": float(selected["f1"]),
        "predicted_positive": predicted_positive,
        "support": int(selected["support"]),
        "target_met": True,
    }


def _safe_average_precision(y_true: list[int], probabilities: list[float]) -> float:
    if len(set(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, probabilities))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _torch_runtime_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clear_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(remaining_seconds):02d}s"
    hours, remaining_minutes = divmod(int(minutes), 60)
    return f"{hours}h{remaining_minutes:02d}m"


def _progress_log_interval(total_steps: int) -> int:
    if total_steps <= 0:
        return 10
    return max(1, total_steps // 10)


def _should_log_progress(current: int, total: int) -> bool:
    if total <= 0:
        return current == 1
    interval = _progress_log_interval(total)
    return current == 1 or current == total or current % interval == 0


def _is_mps_oom_error(exc: BaseException) -> bool:
    return "MPS backend out of memory" in str(exc)


def _should_retry_transformer_on_cpu(exc: BaseException) -> bool:
    return _is_mps_oom_error(exc) or isinstance(exc, MPSFallbackRequested)


def _resolve_causal_lm_runtime_profile(
    *,
    detected_runtime: str,
    requested_runtime_profile: str | None,
    model_id: str,
) -> str:
    del model_id
    if requested_runtime_profile is not None:
        return requested_runtime_profile
    if detected_runtime == "mps":
        return "cpu_fallback"
    return detected_runtime


def _mps_memory_snapshot(torch_module: Any) -> dict[str, int] | None:
    if not hasattr(torch_module, "mps"):
        return None
    try:
        current_allocated_bytes = int(torch_module.mps.current_allocated_memory())
        driver_allocated_bytes = int(torch_module.mps.driver_allocated_memory())
        recommended_max_bytes = int(torch_module.mps.recommended_max_memory())
        available_system_bytes = _darwin_available_memory_bytes()
    except Exception:
        return None
    return {
        "current_allocated_bytes": current_allocated_bytes,
        "driver_allocated_bytes": driver_allocated_bytes,
        "recommended_max_bytes": recommended_max_bytes,
        "available_system_bytes": available_system_bytes,
    }


def _should_proactively_fallback_from_mps(
    *,
    initial_snapshot: dict[str, int],
    current_snapshot: dict[str, int],
) -> bool:
    current_available = int(current_snapshot.get("available_system_bytes", 0))
    initial_available = int(initial_snapshot.get("available_system_bytes", current_available))
    available_drop = max(0, initial_available - current_available)
    current_driver = int(current_snapshot.get("driver_allocated_bytes", 0))
    recommended_max = int(current_snapshot.get("recommended_max_bytes", 0))
    return (
        current_available <= MPS_PROACTIVE_AVAILABLE_MEMORY_FLOOR_BYTES
        or available_drop >= MPS_PROACTIVE_AVAILABLE_MEMORY_DROP_BYTES
        or (recommended_max > 0 and current_driver >= recommended_max)
    )


def _darwin_available_memory_bytes() -> int:
    vm_stat_output = subprocess.check_output(["vm_stat"], text=True)
    page_size_output = subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True)
    page_size = int(page_size_output.strip())
    values: dict[str, int] = {}
    for line in vm_stat_output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            values[key.strip()] = int(digits)
    return page_size * (
        values.get("Pages free", 0)
        + values.get("Pages inactive", 0)
        + values.get("Pages speculative", 0)
    )


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    if gib >= 1:
        return f"{gib:.2f}GiB"
    mib = value / (1024**2)
    return f"{mib:.0f}MiB"


def _slugify_model_name(value: str) -> str:
    return (
        str(value).strip().lower().replace("/", " ").replace("-", "_").replace(".", "_").replace(" ", "_")
    )


def _balanced_class_weights(labels: list[int]) -> list[float]:
    counts = Counter(labels)
    total = len(labels)
    classes = (0, 1)
    return [float(total / (len(classes) * counts[label])) for label in classes]


def _calibration_result_from_summary(value: Any) -> CalibrationResult:
    if not isinstance(value, dict):
        return CalibrationResult(
            available=False,
            method=None,
            brier_score=None,
            log_loss=None,
            positive_count=0,
            negative_count=0,
            calibration_size=0,
        )
    return CalibrationResult(
        available=bool(value.get("available")),
        method=value.get("method"),
        brier_score=value.get("brier_score"),
        log_loss=value.get("log_loss"),
        positive_count=int(value.get("positive_count") or 0),
        negative_count=int(value.get("negative_count") or 0),
        calibration_size=int(value.get("calibration_size") or 0),
    )


def _threshold_selection_ready_from_summary(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    high_threshold = value.get("high_threshold_selection")
    if not isinstance(high_threshold, dict):
        return False
    return bool(high_threshold.get("production_ready"))


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


def _production_ready_status_from_trained_summary(
    *,
    calibration: CalibrationResult,
    calibration_high_threshold_ready: bool,
    high_confidence_precision: float,
    high_confidence_predictions: int,
) -> tuple[bool, str | None]:
    if (
        calibration.available
        and calibration_high_threshold_ready
        and high_confidence_precision >= DEFAULT_HIGH_PRECISION_TARGET
        and high_confidence_predictions >= DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS
    ):
        return True, None
    if not calibration.available:
        return False, "calibration_unavailable"
    if not calibration_high_threshold_ready:
        return False, "high_precision_target_not_met_on_calibration"
    if high_confidence_precision < DEFAULT_HIGH_PRECISION_TARGET:
        return False, "high_precision_target_not_met_on_test"
    if high_confidence_predictions < DEFAULT_MIN_HIGH_CONFIDENCE_TEST_PREDICTIONS:
        return False, "insufficient_high_confidence_test_predictions"
    return False, "production_gate_unsatisfied"


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
        "slice_metrics": "Per-cohort operating metrics for post type, low-text posts, and sparse-media posts on the held-out test slice, including slice support counts and support_status.",
        "constraint_metrics": "Best recall achievable on the held-out test slice while meeting fixed precision constraints for the auto bucket (0.95) and review queue (0.75).",
        "ranking_metrics": "Threshold-independent ranking quality on the held-out test slice. pr_auc is average precision.",
        "production_gate": "A run is production-ready only if calibration is available, the calibration slice can hit the high-precision target, the held-out test auto band also clears that precision target, and the held-out test auto band contains at least the minimum number of high-confidence predictions.",
    }


def _decision_policy(
    *,
    split: Any,
    calibration: CalibrationResult,
    thresholds: DecisionThresholds,
    review_precision_target: float,
    high_precision_target: float,
) -> dict[str, Any]:
    return {
        "low_threshold": thresholds.low_threshold,
        "high_threshold": thresholds.high_threshold,
        "review_precision_target": review_precision_target,
        "high_precision_target": high_precision_target,
        "minimum_high_confidence_calibration_predictions": (
            thresholds.minimum_high_confidence_calibration_predictions
        ),
        "high_threshold_fallback_used": thresholds.high_threshold_fallback_used,
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
        predicted_positive=0,
        support=support,
        production_ready=False,
    )
