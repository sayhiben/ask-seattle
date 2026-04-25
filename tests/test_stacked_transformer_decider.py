from __future__ import annotations

import joblib
from pathlib import Path

import pytest
import torch

from ask_seattle.data import LabeledPost
import ask_seattle.model as model_module
from ask_seattle.model import DatasetSplit, build_inference_row, load_model, raw_score_rows, score_rows
import ask_seattle.training as training


class _FakeClassifier:
    classes_ = [0, 1]


class _ConstantScoreModel:
    named_steps = {"classifier": _FakeClassifier()}

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, rows):  # type: ignore[no-untyped-def]
        negative_probability = 1.0 - self.positive_probability
        return [[negative_probability, self.positive_probability] for _ in rows]


class _MetaClassifier:
    classes_ = [0, 1]

    def predict_proba(self, features):  # type: ignore[no-untyped-def]
        probabilities = []
        for row in features:
            score = max(0.0, min(1.0, float((row[0] + row[1] + row[2] + row[5]) / 4.0)))
            probabilities.append([1.0 - score, score])
        return probabilities


class _BatchingTokenizer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, titles, bodies, **kwargs):  # type: ignore[no-untyped-def]
        assert len(titles) == len(bodies)
        self.batch_sizes.append(len(titles))
        batch_size = len(titles)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class _BatchingModel:
    def to(self, device):  # type: ignore[no-untyped-def]
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **encoded):  # type: ignore[no-untyped-def]
        batch_size = int(encoded["input_ids"].shape[0])
        logits = torch.tensor([[0.0, 1.0]] * batch_size, dtype=torch.float32)
        return type("Outputs", (), {"logits": logits})()


def test_score_rows_supports_stacked_transformer_decider_bundle() -> None:
    bundle = {
        "model_family": "stacked_transformer_decider",
        "model_name": "stacked_transformer_decider",
        "model": _MetaClassifier(),
        "component_models": [
            {
                "name": "transformer_modernbert_base",
                "bundle": {
                    "model": _ConstantScoreModel(0.92),
                    "model_family": "tfidf",
                    "model_name": "base",
                    "low_threshold": 0.5,
                    "high_threshold": 0.8,
                },
            },
            {
                "name": "transformer_neobert",
                "bundle": {
                    "model": _ConstantScoreModel(0.88),
                    "model_family": "tfidf",
                    "model_name": "neobert",
                    "low_threshold": 0.5,
                    "high_threshold": 0.8,
                },
            },
            {
                "name": "transformer_modernbert_large",
                "bundle": {
                    "model": _ConstantScoreModel(0.83),
                    "model_family": "tfidf",
                    "model_name": "large",
                    "low_threshold": 0.5,
                    "high_threshold": 0.8,
                },
            },
        ],
    }
    rows = [
        build_inference_row(title="Where should I stay?", selftext="", post_type="image"),
        build_inference_row(title="City council update", selftext="Budget meeting tonight", post_type="text"),
    ]

    scores = score_rows(bundle, rows)

    assert len(scores) == 2
    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] > 0.8


def test_raw_score_rows_batches_transformer_inference() -> None:
    tokenizer = _BatchingTokenizer()
    bundle = {
        "model_family": "transformer_sequence_classifier",
        "model_name": "transformer_test",
        "tokenizer": tokenizer,
        "model": _BatchingModel(),
        "inference_batch_size": 2,
        "max_length": 32,
    }
    rows = [
        build_inference_row(title=f"Question {index}", selftext="body", post_type="text")
        for index in range(5)
    ]

    scores = raw_score_rows(bundle, rows)

    assert len(scores) == 5
    assert tokenizer.batch_sizes == [2, 2, 1]


def test_raw_score_rows_uses_reasonable_implicit_transformer_batch_size() -> None:
    tokenizer = _BatchingTokenizer()
    bundle = {
        "model_family": "transformer_sequence_classifier",
        "model_name": "transformer_test",
        "tokenizer": tokenizer,
        "model": _BatchingModel(),
        "training_args": {"per_device_eval_batch_size": 2},
        "max_length": 32,
    }
    rows = [
        build_inference_row(title=f"Question {index}", selftext="body", post_type="text")
        for index in range(9)
    ]

    scores = raw_score_rows(bundle, rows)

    assert len(scores) == 9
    assert tokenizer.batch_sizes == [8, 1]


def test_train_stacked_transformer_decider_for_split_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = DatasetSplit(
        train=[
            LabeledPost("Where should I stay downtown?", "", 1, post_id="tr0", post_type="image"),
            LabeledPost("Local policy update", "Budget discussion", 0, post_id="tr1", post_type="text"),
            LabeledPost("Which neighborhood for a weekend trip?", "", 1, post_id="tr2", post_type="link"),
            LabeledPost("Mayor press conference", "Transit funding", 0, post_id="tr3", post_type="text"),
        ],
        calibration=[
            LabeledPost("Best area for first-time visitors?", "", 1, post_id="ca0", post_type="image"),
            LabeledPost("Lane closure notice", "Construction next week", 0, post_id="ca1", post_type="text"),
            LabeledPost("Need hotel and transit advice", "", 1, post_id="ca2", post_type="link"),
            LabeledPost("Council recap", "Committee agenda", 0, post_id="ca3", post_type="text"),
        ],
        test=[
            LabeledPost("Where to stay without a car?", "", 1, post_id="te0", post_type="image"),
            LabeledPost("City budget hearing", "Public comment", 0, post_id="te1", post_type="text"),
            LabeledPost("Visiting and need neighborhood tips", "", 1, post_id="te2", post_type="link"),
            LabeledPost("Local election roundup", "Endorsements", 0, post_id="te3", post_type="text"),
        ],
        split_strategy="random",
        split_seed=13,
        evaluation_subreddit="seattle",
    )
    component_names = list(training.STACKED_TRANSFORMER_COMPONENT_NAMES)
    component_summaries: dict[str, dict[str, object]] = {}
    for name in component_names:
        artifact_path = tmp_path / f"{name}.joblib"
        artifact_path.write_text(name, encoding="utf-8")
        component_summaries[name] = {
            "display_name": name,
            "artifact_path": str(artifact_path),
        }

    score_lookup = {
        "transformer_modernbert_base": {
            "Where should I stay downtown?": 0.94,
            "Local policy update": 0.10,
            "Which neighborhood for a weekend trip?": 0.91,
            "Mayor press conference": 0.08,
            "Best area for first-time visitors?": 0.93,
            "Lane closure notice": 0.12,
            "Need hotel and transit advice": 0.88,
            "Council recap": 0.09,
            "Where to stay without a car?": 0.92,
            "City budget hearing": 0.11,
            "Visiting and need neighborhood tips": 0.90,
            "Local election roundup": 0.07,
        },
        "transformer_neobert": {
            "Where should I stay downtown?": 0.89,
            "Local policy update": 0.16,
            "Which neighborhood for a weekend trip?": 0.87,
            "Mayor press conference": 0.10,
            "Best area for first-time visitors?": 0.88,
            "Lane closure notice": 0.15,
            "Need hotel and transit advice": 0.85,
            "Council recap": 0.11,
            "Where to stay without a car?": 0.89,
            "City budget hearing": 0.12,
            "Visiting and need neighborhood tips": 0.86,
            "Local election roundup": 0.09,
        },
        "transformer_modernbert_large": {
            "Where should I stay downtown?": 0.91,
            "Local policy update": 0.20,
            "Which neighborhood for a weekend trip?": 0.90,
            "Mayor press conference": 0.14,
            "Best area for first-time visitors?": 0.92,
            "Lane closure notice": 0.18,
            "Need hotel and transit advice": 0.89,
            "Council recap": 0.16,
            "Where to stay without a car?": 0.93,
            "City budget hearing": 0.19,
            "Visiting and need neighborhood tips": 0.91,
            "Local election roundup": 0.13,
        },
    }

    def fake_load_model(path: Path) -> dict[str, object]:
        model_name = Path(path).stem
        return {
            "model_family": "transformer_sequence_classifier",
            "model_name": model_name,
            "model_id": model_name,
            "display_name": model_name,
            "representation_config": {
                "include_sparse_media_token": False,
                "include_image_low_text_tokens": True,
            },
        }

    def fake_score_rows(bundle: dict[str, object], rows):  # type: ignore[no-untyped-def]
        model_name = str(bundle["model_id"])
        return [float(score_lookup[model_name][str(row["title"])]) for row in rows]

    def fake_oof_train_scores(*, split, artifact_dir, component_payloads, representation_config):  # type: ignore[no-untyped-def]
        rows = training._inference_rows(split.train, representation_config=representation_config)
        return (
            {
                payload["name"]: [
                    float(score_lookup[payload["name"]][str(row["title"])]) for row in rows
                ]
                for payload in component_payloads
            },
            {
                "meta_training_source": "oof_component_scores",
                "requested_fold_count": 3,
                "actual_fold_count": 3,
                "inner_calibration_size": 0.2,
                "folds": [{"fold_index": 0, "holdout_size": 2}],
                "component_training_mode": "locked_from_full_component_summary",
            },
        )

    monkeypatch.setattr(training, "load_model", fake_load_model)
    monkeypatch.setattr(training, "score_rows", fake_score_rows)
    monkeypatch.setattr(training, "_stacked_transformer_oof_train_scores", fake_oof_train_scores)

    summary = training._train_stacked_transformer_decider_for_split(
        split=split,
        output_dir=tmp_path / "stacked",
        component_summaries=component_summaries,
        evaluate_on_test=True,
    )

    artifact_path = Path(summary["artifact_path"])
    artifact = joblib.load(artifact_path)

    assert summary["model_name"] == "stacked_transformer_decider"
    assert summary["model_family"] == "stacked_transformer_decider"
    assert summary["training_args"]["component_model_names"] == component_names
    assert summary["training_args"]["meta_training_source"] == "oof_component_scores"
    assert summary["oof_training"]["actual_fold_count"] == 3
    assert summary["benchmark_status"] == "complete"
    assert "constraint_metrics" in summary
    assert artifact["model_family"] == "stacked_transformer_decider"
    assert artifact["meta_training_source"] == "oof_component_scores"
    assert [component["name"] for component in artifact["component_models"]] == component_names


def test_load_model_prefers_stacked_loader_before_tfidf_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "stacked_transformer_decider.joblib"
    joblib.dump(
        {
            "model_family": "stacked_transformer_decider",
            "model_name": "stacked_transformer_decider",
            "model": object(),
            "component_models": [
                {
                    "name": "transformer_modernbert_base",
                    "artifact_path": "/tmp/component.joblib",
                }
            ],
        },
        bundle_path,
    )

    def fake_stacked_loader(bundle, *, source_path):  # type: ignore[no-untyped-def]
        return {"loader": "stacked", "source_path": str(source_path)}

    def fail_tfidf_loader(bundle):  # type: ignore[no-untyped-def]
        raise AssertionError("stacked bundle should not route through tfidf normalization")

    monkeypatch.setattr(model_module, "_load_stacked_transformer_bundle_from_joblib", fake_stacked_loader)
    monkeypatch.setattr(model_module, "_normalize_tfidf_bundle", fail_tfidf_loader)

    loaded = load_model(bundle_path)

    assert loaded == {"loader": "stacked", "source_path": str(bundle_path)}


def test_load_model_rebases_remote_stacked_component_paths_to_local_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_path = tmp_path / "models" / "benchmark-suite" / "transformer_neobert" / "transformer_bundle.joblib"
    component_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_family": "transformer_sequence_classifier",
            "model_name": "transformer_neobert",
            "artifact_path": str(component_path.parent),
        },
        component_path,
    )
    bundle_path = (
        tmp_path / "models" / "benchmark-suite" / "stacked_transformer_decider" / "stacked_transformer_decider.joblib"
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_family": "stacked_transformer_decider",
            "model_name": "stacked_transformer_decider",
            "model": object(),
            "component_models": [
                {
                    "name": "transformer_neobert",
                    "artifact_path": "/workspace/ask-seattle/models/benchmark-suite/transformer_neobert/transformer_bundle.joblib",
                }
            ],
        },
        bundle_path,
    )

    def fake_transformer_loader(bundle, *, source_path):  # type: ignore[no-untyped-def]
        return {
            "model_family": "transformer_sequence_classifier",
            "model_name": bundle.get("model_name"),
            "artifact_path": str(source_path),
        }

    monkeypatch.setattr(model_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_module, "_load_transformer_bundle_from_joblib", fake_transformer_loader)

    loaded = load_model(bundle_path)

    assert loaded["component_models"][0]["artifact_path"] == str(component_path)
    assert loaded["component_models"][0]["bundle"]["artifact_path"] == str(component_path)


def test_stacked_transformer_outer_holdout_folds_are_stratified() -> None:
    posts = [
        LabeledPost(f"Positive {index}", "", 1, post_id=f"p{index}", post_type="text")
        for index in range(3)
    ] + [
        LabeledPost(f"Negative {index}", "", 0, post_id=f"n{index}", post_type="text")
        for index in range(3)
    ]

    folds = training._stacked_transformer_outer_holdout_folds(
        posts,
        requested_fold_count=3,
        seed=13,
    )

    assert len(folds) == 3
    assert sorted(index for fold in folds for index in fold) == list(range(6))
    for fold in folds:
        labels = [posts[index].label for index in fold]
        assert labels.count(1) == 1
        assert labels.count(0) == 1


def test_stacked_transformer_fold_training_kwargs_reuses_selected_component_profile() -> None:
    payload = {
        "name": "transformer_neobert",
        "display_name": "Transformer NeoBERT",
        "bundle": {
            "model_id": "chandar-lab/NeoBERT",
        },
        "summary": {
            "model_id": "chandar-lab/NeoBERT",
            "display_name": "Transformer NeoBERT",
            "config_version": "v7_bootstrap_precision_grid",
            "training_args": {
                "candidate_profile": {
                    "name": "balanced",
                    "learning_rate": 3e-5,
                    "weight_decay": 0.01,
                    "max_length": 256,
                },
                "class_weighting": "balanced_cross_entropy",
            },
        },
    }

    original_runtime_resolver = training._resolve_stacked_transformer_oof_runtime_profile
    training._resolve_stacked_transformer_oof_runtime_profile = lambda: "cuda"
    try:
        kwargs = training._stacked_transformer_fold_training_kwargs(
            payload,
            representation_config={
                "include_sparse_media_token": False,
                "include_image_low_text_tokens": True,
            },
        )
    finally:
        training._resolve_stacked_transformer_oof_runtime_profile = original_runtime_resolver

    assert kwargs["model_id"] == "chandar-lab/NeoBERT"
    assert kwargs["display_name"] == "Transformer NeoBERT"
    assert kwargs["config_version"] == "v7_bootstrap_precision_grid"
    assert kwargs["runtime_profile"] == "cuda"
    assert kwargs["locked_candidate_profile"]["max_length"] == 256
    assert kwargs["locked_loss_mode"] == "balanced_cross_entropy"
