import sys
from pathlib import Path

import pytest

from ask_seattle.data import LabeledPost
import ask_seattle.model as model_module
from ask_seattle.model import (
    _default_min_df,
    _bundle_runtime_device,
    _load_causal_lm_bundle_from_joblib,
    _safe_binary_completion_probability,
    _load_transformer_bundle_from_joblib,
    _semantic_runtime_component_texts,
    _tensor_to_float32_numpy,
    _move_token_batch_to_device,
    _install_xformers_swiglu_fallback,
    build_inference_row,
    build_pipeline,
    classify_post,
    select_decision_thresholds,
    select_threshold,
    split_labeled_posts,
    transformer_load_options,
    tfidf_feature_audit,
    train_model,
)


def test_threshold_selection_prefers_recall_above_precision_gate() -> None:
    selection = select_threshold(
        [1, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.1],
        min_precision=0.95,
        thresholds=(0.5, 0.75, 0.85),
    )

    assert selection.production_ready is True
    assert selection.threshold == 0.75
    assert selection.precision == 1.0
    assert selection.recall == 1.0


def test_threshold_selection_prefers_stricter_threshold_with_stronger_bootstrap_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = {
        0.75: {
            "bootstrap_precision_p20": 0.96,
            "bootstrap_precision_mean": 0.98,
            "bootstrap_precision_min": 0.90,
            "bootstrap_predicted_positive_p20": 2,
            "bootstrap_predicted_positive_mean": 2.0,
            "bootstrap_predicted_positive_min": 2,
            "bootstrap_sample_count": 200,
        },
        0.85: {
            "bootstrap_precision_p20": 1.0,
            "bootstrap_precision_mean": 1.0,
            "bootstrap_precision_min": 1.0,
            "bootstrap_predicted_positive_p20": 2,
            "bootstrap_predicted_positive_mean": 2.0,
            "bootstrap_predicted_positive_min": 2,
            "bootstrap_sample_count": 200,
        },
    }

    monkeypatch.setattr(
        model_module,
        "_bootstrap_threshold_diagnostics",
        lambda *args, threshold, **kwargs: diagnostics[threshold],
    )

    selection = select_threshold(
        [1, 1, 0, 0],
        [0.9, 0.86, 0.2, 0.1],
        min_precision=0.95,
        minimum_predictions=1,
        thresholds=(0.75, 0.85),
    )

    assert selection.production_ready is True
    assert selection.threshold == 0.85
    assert selection.bootstrap_ready is True
    assert selection.bootstrap_precision_p20 == 1.0


def test_split_labeled_posts_is_random_by_default_and_deterministic() -> None:
    posts = [
        LabeledPost(
            title=f"post {index}",
            selftext="body",
            label=index % 2,
            post_id=f"p{index}",
            time_key=float(index),
        )
        for index in range(12)
    ]

    split = split_labeled_posts(
        posts,
        calibration_size=0.25,
        test_size=0.25,
    )
    repeated = split_labeled_posts(
        posts,
        calibration_size=0.25,
        test_size=0.25,
    )
    alternate_seed = split_labeled_posts(
        posts,
        calibration_size=0.25,
        test_size=0.25,
        split_seed=99,
    )

    assert split.split_strategy == "random"
    assert split.split_seed == 13
    assert [post.post_id for post in split.train] == [post.post_id for post in repeated.train]
    assert [post.post_id for post in split.calibration] == [post.post_id for post in repeated.calibration]
    assert [post.post_id for post in split.test] == [post.post_id for post in repeated.test]
    assert [post.post_id for post in split.train] != [post.post_id for post in alternate_seed.train]


def test_semantic_runtime_component_texts_fill_empty_values_with_placeholders() -> None:
    bundle = {"prompt_mode": "plain"}
    rows = [{"title": "", "body_raw": ""}]

    assert _semantic_runtime_component_texts(bundle, rows, component="title") == ["[no title]"]
    assert _semantic_runtime_component_texts(bundle, rows, component="body") == ["[no body]"]


def test_semantic_runtime_component_texts_support_document_prefix() -> None:
    bundle = {"prompt_mode": "document_prefix", "prompt_prefix": "Document:"}
    rows = [{"title": "Where to park?", "body_raw": "Need Capitol Hill advice"}]

    assert _semantic_runtime_component_texts(bundle, rows, component="title") == ["Document: Where to park?"]
    assert _semantic_runtime_component_texts(bundle, rows, component="body") == ["Document: Need Capitol Hill advice"]


def test_semantic_runtime_component_texts_support_jina_document_component() -> None:
    bundle = {"prompt_mode": "jina_document_component", "prompt_prefix": "Document:"}
    rows = [{"title": "Where to park?", "body_raw": "Need Capitol Hill advice"}]

    assert _semantic_runtime_component_texts(bundle, rows, component="title") == ["Document: Title: Where to park?"]
    assert _semantic_runtime_component_texts(bundle, rows, component="body") == ["Document: Body: Need Capitol Hill advice"]


def test_split_labeled_posts_can_use_explicit_time_strategy() -> None:
    posts = [
        LabeledPost(
            title=f"post {index}",
            selftext="body",
            label=index % 2,
            post_id=f"p{index}",
            time_key=float(index),
        )
        for index in range(12)
    ]

    split = split_labeled_posts(
        posts,
        calibration_size=0.25,
        test_size=0.25,
        split_strategy="time",
    )

    assert split.split_strategy == "time"
    assert [post.post_id for post in split.train] == ["p0", "p1", "p2", "p3", "p4", "p5"]
    assert [post.post_id for post in split.calibration] == ["p6", "p7", "p8"]
    assert [post.post_id for post in split.test] == ["p9", "p10", "p11"]


def test_select_decision_thresholds_orders_low_and_high() -> None:
    thresholds = select_decision_thresholds(
        [1, 1, 0, 0],
        [0.9, 0.6, 0.4, 0.2],
        auto_precision_target=0.95,
        thresholds=(0.2, 0.4, 0.6, 0.8),
    )

    assert thresholds.high_threshold == 0.6
    assert thresholds.low_threshold <= thresholds.high_threshold
    assert thresholds.abstain_enabled is False


def test_select_decision_thresholds_uses_review_precision_target_for_low_threshold() -> None:
    thresholds = select_decision_thresholds(
        [1, 1, 0, 1, 0],
        [0.9, 0.45, 0.4, 0.39, 0.1],
        auto_precision_target=0.95,
        review_precision_target=0.8,
        thresholds=(0.1, 0.39, 0.4, 0.45),
    )

    assert thresholds.low_threshold == 0.45
    assert thresholds.low_threshold <= thresholds.high_threshold


def test_select_decision_thresholds_falls_back_when_precision_ready_threshold_lacks_support() -> None:
    thresholds = select_decision_thresholds(
        [1, 0, 1, 0],
        [0.91, 0.62, 0.61, 0.1],
        auto_precision_target=0.95,
        minimum_high_confidence_calibration_predictions=2,
        thresholds=(0.61, 0.62, 0.91),
    )

    assert thresholds.high_threshold == 0.91
    assert thresholds.high_threshold_selection.production_ready is False
    assert thresholds.high_threshold_selection.predicted_positive == 1
    assert thresholds.minimum_high_confidence_calibration_predictions == 2
    assert thresholds.high_threshold_fallback_used is True


def test_select_decision_thresholds_falls_back_when_bootstrap_precision_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "_bootstrap_threshold_diagnostics",
        lambda *args, **kwargs: {
            "bootstrap_precision_p20": 0.90,
            "bootstrap_precision_mean": 0.92,
            "bootstrap_precision_min": 0.80,
            "bootstrap_predicted_positive_p20": 2,
            "bootstrap_predicted_positive_mean": 2.0,
            "bootstrap_predicted_positive_min": 2,
            "bootstrap_sample_count": 200,
        },
    )

    thresholds = select_decision_thresholds(
        [1, 1, 0],
        [0.91, 0.83, 0.1],
        auto_precision_target=0.95,
        minimum_high_confidence_calibration_predictions=2,
        thresholds=(0.8,),
    )

    assert thresholds.high_threshold == 0.8
    assert thresholds.high_threshold_selection.production_ready is False
    assert thresholds.high_threshold_selection.bootstrap_ready is False
    assert thresholds.high_threshold_selection.fallback_reason == "bootstrap_precision_target_not_met"
    assert thresholds.high_threshold_fallback_used is True


def test_tensor_to_float32_numpy_handles_bfloat16() -> None:
    torch = pytest.importorskip("torch")

    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    array = _tensor_to_float32_numpy(tensor)

    assert array.dtype.name == "float32"
    assert array.tolist() == [[1.0, 2.0]]


def test_transformer_bundle_loader_rebases_stale_remote_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "transformer_bundle.joblib"
    bundle_path.write_text("bundle", encoding="utf-8")
    local_model_dir = tmp_path / "transformer_model"
    local_model_dir.mkdir()
    seen: dict[str, object] = {}

    def fake_runtime_loader(metadata: dict[str, object], *, model_dir: Path) -> dict[str, object]:
        seen["artifact_path"] = metadata["artifact_path"]
        seen["model_dir"] = metadata["model_dir"]
        seen["resolved_model_dir"] = model_dir
        return {"model_dir": str(model_dir)}

    monkeypatch.setattr(model_module, "_load_transformer_runtime_bundle", fake_runtime_loader)

    loaded = _load_transformer_bundle_from_joblib(
        {"artifact_path": "/workspace/ask-seattle/models/benchmark-suite/x/transformer_model"},
        source_path=bundle_path,
    )

    assert loaded["model_dir"] == str(local_model_dir)
    assert seen["artifact_path"] == str(local_model_dir)
    assert seen["model_dir"] == str(local_model_dir)
    assert seen["resolved_model_dir"] == local_model_dir


def test_transformer_load_options_enable_remote_code_for_neobert() -> None:
    assert transformer_load_options("chandar-lab/NeoBERT") == {"trust_remote_code": True}
    assert transformer_load_options("answerdotai/ModernBERT-base") == {}


def test_install_xformers_swiglu_fallback_registers_compatible_module(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    monkeypatch.delitem(sys.modules, "xformers", raising=False)
    monkeypatch.delitem(sys.modules, "xformers.ops", raising=False)

    _install_xformers_swiglu_fallback()

    from xformers.ops import SwiGLU

    module = SwiGLU(4, 8, 4, bias=False)
    inputs = torch.randn(2, 3, 4)
    outputs = module(inputs)

    assert outputs.shape == (2, 3, 4)
    assert hasattr(module, "w12")
    assert hasattr(module, "w3")


def test_causal_lm_bundle_loader_resolves_relative_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "causal_lm_bundle.joblib"
    bundle_path.write_text("bundle", encoding="utf-8")
    local_model_dir = tmp_path / "causal_lm_model"
    local_model_dir.mkdir()
    seen: dict[str, object] = {}

    def fake_runtime_loader(metadata: dict[str, object], *, model_dir: Path) -> dict[str, object]:
        seen["artifact_path"] = metadata["artifact_path"]
        seen["model_dir"] = metadata["model_dir"]
        seen["resolved_model_dir"] = model_dir
        return {"model_dir": str(model_dir)}

    monkeypatch.setattr(model_module, "_load_causal_lm_runtime_bundle", fake_runtime_loader)

    loaded = _load_causal_lm_bundle_from_joblib(
        {"artifact_path": "causal_lm_model"},
        source_path=bundle_path,
    )

    assert loaded["model_dir"] == str(local_model_dir)
    assert seen["artifact_path"] == str(local_model_dir)
    assert seen["model_dir"] == str(local_model_dir)
    assert seen["resolved_model_dir"] == local_model_dir


def test_threshold_selection_uses_observed_probabilities_when_grid_is_too_coarse() -> None:
    selection = select_threshold(
        [1, 0],
        [0.961, 0.95],
        min_precision=1.0,
    )

    assert selection.production_ready is True
    assert selection.threshold == 0.961
    assert selection.precision == 1.0
    assert selection.recall == 1.0


def test_split_labeled_posts_shrinks_later_slices_to_keep_both_labels_in_train() -> None:
    posts = [
        LabeledPost(title=f"post {index}", selftext="body", label=0, post_id=f"p{index}", time_key=float(index))
        for index in range(3)
    ] + [
        LabeledPost(title="post 3", selftext="body", label=1, post_id="p3", time_key=3.0),
        LabeledPost(title="post 4", selftext="body", label=1, post_id="p4", time_key=4.0),
    ]

    split = split_labeled_posts(
        posts,
        calibration_size=0.2,
        test_size=0.2,
        split_strategy="time",
    )

    assert {post.label for post in split.train} == {0, 1}
    assert [post.post_id for post in split.train] == ["p0", "p1", "p2", "p3"]
    assert [post.post_id for post in split.calibration] == []
    assert [post.post_id for post in split.test] == ["p4"]


def test_split_labeled_posts_can_train_on_mixed_data_and_evaluate_on_one_subreddit() -> None:
    posts = [
        LabeledPost(
            title="Moving advice",
            selftext="Need recommendations",
            label=1,
            post_id="ask0",
            subreddit="askseattle",
            time_key=0.0,
        ),
        LabeledPost(
            title="Traffic update",
            selftext="Road closure downtown",
            label=0,
            post_id="sea1",
            subreddit="seattle",
            time_key=1.0,
        ),
        LabeledPost(
            title="Neighborhood advice",
            selftext="Where should I live?",
            label=1,
            post_id="sea2",
            subreddit="seattle",
            time_key=2.0,
        ),
        LabeledPost(
            title="Best coffee",
            selftext="Need recommendations",
            label=1,
            post_id="sea3",
            subreddit="seattle",
            time_key=3.0,
        ),
        LabeledPost(
            title="Late askseattle positive",
            selftext="Visiting next week",
            label=1,
            post_id="ask4",
            subreddit="askseattle",
            time_key=4.0,
        ),
        LabeledPost(
            title="City budget update",
            selftext="Council discussion",
            label=0,
            post_id="sea4",
            subreddit="seattle",
            time_key=5.0,
        ),
        LabeledPost(
            title="Weekend itinerary help",
            selftext="What should I do?",
            label=1,
            post_id="sea5",
            subreddit="seattle",
            time_key=6.0,
        ),
    ]

    split = split_labeled_posts(
        posts,
        calibration_size=0.2,
        test_size=0.2,
        evaluation_subreddit="seattle",
    )

    assert split.split_strategy == "random_eval_subreddit"
    assert split.split_seed == 13
    assert split.evaluation_subreddit == "seattle"
    assert all(post.subreddit == "seattle" for post in split.calibration)
    assert all(post.subreddit == "seattle" for post in split.test)
    assert {post.label for post in split.train} == {0, 1}
    assert any(post.subreddit == "askseattle" for post in split.train)


def test_build_pipeline_excludes_common_function_words_from_word_vocab() -> None:
    rows = [
        build_inference_row(
            title="The best pizza and coffee",
            selftext="The recommendations are in and the food is good with some just one choice",
        ),
        build_inference_row(
            title="The city council and transit update",
            selftext="The discussion is about policy and local news with some just one followup",
        ),
    ]
    model = build_pipeline(min_df=1)
    model.fit(rows, [1, 0])

    features = model.named_steps["features"]
    title_vectorizer = features.transformer_list[0][1].named_steps["vectorizer"]
    body_vectorizer = features.transformer_list[1][1].named_steps["vectorizer"]

    assert "some" in title_vectorizer.get_stop_words()
    assert "just" in title_vectorizer.get_stop_words()
    assert "one" in title_vectorizer.get_stop_words()
    assert "some" in body_vectorizer.get_stop_words()
    assert "just" in body_vectorizer.get_stop_words()
    assert "one" in body_vectorizer.get_stop_words()
    assert "the" not in title_vectorizer.vocabulary_
    assert "and" not in title_vectorizer.vocabulary_
    assert "the" not in body_vectorizer.vocabulary_
    assert "is" not in body_vectorizer.vocabulary_
    assert "best" in title_vectorizer.vocabulary_
    assert "recommendations" in body_vectorizer.vocabulary_


def test_build_pipeline_uses_separate_metadata_channel_and_keeps_char_ngrams_off_metadata_tokens() -> None:
    rows = [
        build_inference_row(
            title="What is this? https://example.com/post",
            selftext="See https://example.com/image for context",
            post_type="image",
            content_domain="www.instagram.com",
            is_crosspost=True,
        ),
        build_inference_row(
            title="Traffic update",
            selftext="Road closure downtown",
            post_type="link",
            content_domain="reddit.com",
            is_crosspost=False,
        ),
    ]

    model = build_pipeline(min_df=1)
    model.fit(rows, [1, 0])

    features = model.named_steps["features"]
    char_vectorizer = features.transformer_list[2][1].named_steps["vectorizer"]
    metadata_vectorizer = features.transformer_list[3][1].named_steps["vectorizer"]

    assert "HAS_BODY:yes" in metadata_vectorizer.get_feature_names_out()
    assert "POST_TYPE:image" in metadata_vectorizer.get_feature_names_out()
    assert not any(":" in feature for feature in char_vectorizer.get_feature_names_out())
    assert "https" not in char_vectorizer.get_feature_names_out()
    assert "http" not in char_vectorizer.get_feature_names_out()


def test_build_inference_row_separates_metadata_and_raw_text_views() -> None:
    row = build_inference_row(
        title="Looking for ideas https://example.com/post",
        selftext="See https://example.com/image",
        post_type="image",
        content_domain="www.instagram.com",
        is_crosspost=True,
    )

    assert "HAS_BODY:yes" in row["body"]
    assert "HAS_BODY:yes" in row["metadata_text"]
    assert "TITLE_LEN_BUCKET:medium" in row["body"]
    assert "BODY_LEN_BUCKET:short" in row["body"]
    assert "HAS_QUESTION_MARK:no" in row["body"]
    assert "LOW_TEXT:yes" in row["body"]
    assert "POST_TYPE:image" in row["body"]
    assert "CONTENT_DOMAIN:instagram_com" in row["body"]
    assert "CROSSPOST:yes" in row["body"]
    assert "SPARSE_MEDIA:yes" in row["body"]
    assert "LOW_TEXT_IMAGE:yes" in row["body"]
    assert row["body_length_bucket"] == "short"
    assert row["is_sparse_media"] is True
    assert row["body_raw"] == "See https://example.com/image"
    assert row["body_lexical"] == "See URL"
    assert row["body_lexical_stripped"] == "See"
    assert row["title_lexical"] == "Looking for ideas URL"
    assert row["title_lexical_stripped"] == "Looking for ideas"
    assert row["text_raw"] == "Looking for ideas https://example.com/post\nSee https://example.com/image"
    assert row["text_lexical"] == "Looking for ideas URL\nSee URL"
    assert row["text_lexical_stripped"] == "Looking for ideas\nSee"
    assert row["text"].startswith("TITLE: Looking for ideas https://example.com/post")


def test_build_inference_row_can_disable_sparse_media_token_for_runtime_representation() -> None:
    row = build_inference_row(
        title="Who is this?",
        selftext="",
        post_type="image",
        content_domain="i.redd.it",
        include_sparse_media_token=False,
    )

    assert "SPARSE_MEDIA:yes" not in row["metadata_text"]
    assert "IMAGE_NO_BODY:yes" in row["metadata_text"]
    assert "LOW_TEXT_IMAGE:yes" in row["metadata_text"]
    assert row["is_sparse_media"] is True


def test_default_min_df_scales_with_corpus_size() -> None:
    assert _default_min_df([]) == 1
    assert _default_min_df([LabeledPost(title="a", selftext="", label=0)] * 49) == 1
    assert _default_min_df([LabeledPost(title="a", selftext="", label=0)] * 50) == 2
    assert _default_min_df([LabeledPost(title="a", selftext="", label=0)] * 500) == 3
    assert _default_min_df([LabeledPost(title="a", selftext="", label=0)] * 2000) == 5


def test_tfidf_feature_audit_includes_channel_breakdown_and_stopwords() -> None:
    posts = [
        LabeledPost(title="Where should I stay?", selftext="Visiting soon", label=1, post_id="a1"),
        LabeledPost(title="Best coffee near Fremont", selftext="Need recommendations", label=1, post_id="a2"),
        LabeledPost(title="City council budget update", selftext="Local policy discussion", label=0, post_id="n1"),
        LabeledPost(title="Traffic alert downtown", selftext="Lane closures this morning", label=0, post_id="n2"),
    ]

    audit = tfidf_feature_audit(train_model(posts), limit=3)

    assert "word_stopwords" in audit
    assert "the" in audit["word_stopwords"]
    assert "some" in audit["word_stopwords"]
    assert "just" in audit["word_stopwords"]
    assert "one" in audit["word_stopwords"]
    assert set(audit["top_positive_by_channel"]) == {"title_word", "body_word", "char_wb", "metadata_token"}
    assert set(audit["top_negative_by_channel"]) == {"title_word", "body_word", "char_wb", "metadata_token"}
    assert all("channel" in row and "full_feature" in row for row in audit["top_positive"])


class FakeModel:
    named_steps = {"classifier": None}

    def __init__(self, positive_probability: float = 0.9) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, texts: list[object]) -> list[list[float]]:
        negative_probability = 1 - self.positive_probability
        return [[negative_probability, self.positive_probability] for _ in texts]


class FakeClassifier:
    classes_ = [0, 1]


def test_classify_post_returns_high_confidence_match() -> None:
    model = FakeModel()
    model.named_steps["classifier"] = FakeClassifier()
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "threshold": 0.85,
    }

    result = classify_post(bundle, title="Where should I stay?", post_id="abc", permalink="https://reddit.test")

    assert result.label == "askseattle"
    assert result.confidence_band == "high"
    assert result.post_id == "abc"
    assert result.permalink == "https://reddit.test"
    assert result.model_name == "tfidf_logreg"


def test_classify_post_accepts_metadata_features() -> None:
    model = FakeModel()
    model.named_steps["classifier"] = FakeClassifier()
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "threshold": 0.85,
    }

    result = classify_post(
        bundle,
        title="Where should I stay?",
        selftext="",
        post_type="link",
        content_domain="example.com",
        is_crosspost=False,
    )

    assert result.label == "askseattle"


def test_classify_post_returns_borderline_match_between_thresholds() -> None:
    model = FakeModel()
    model.named_steps["classifier"] = FakeClassifier()
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "low_threshold": 0.6,
        "high_threshold": 0.95,
    }

    result = classify_post(bundle, title="Neighborhood advice?", post_id="abc")

    assert result.confidence_band == "borderline"
    assert result.label == "askseattle"


def test_classify_post_uses_stricter_effective_high_threshold_for_sparse_media() -> None:
    model = FakeModel(positive_probability=0.9)
    model.named_steps["classifier"] = FakeClassifier()
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "low_threshold": 0.6,
        "high_threshold": 0.85,
    }

    result = classify_post(
        bundle,
        title="What is this?",
        selftext="",
        post_type="image",
        content_domain="reddit.com",
    )

    assert result.label == "askseattle"
    assert result.confidence_band == "borderline"
    assert result.high_threshold == pytest.approx(0.97)


def test_bundle_runtime_device_keeps_causal_lm_off_mps() -> None:
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

    assert (
        _bundle_runtime_device(
            {"model_family": "causal_lm_classifier", "model_id": "Qwen/Qwen3-1.7B"},
            FakeTorch(),
        )
        == "cpu"
    )


def test_bundle_runtime_device_keeps_transformer_backed_semantic_model_off_mps() -> None:
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

    assert (
        _bundle_runtime_device(
            {
                "model_family": "semantic_embedding",
                "backend": "hf_embedding",
                "model_id": "Qwen/Qwen3-Embedding-0.6B",
            },
            FakeTorch(),
        )
        == "cpu"
    )


def test_bundle_runtime_device_keeps_sentence_transformers_semantic_model_off_mps() -> None:
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

    assert (
        _bundle_runtime_device(
            {
                "model_family": "semantic_embedding",
                "backend": "sentence_transformers",
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            },
            FakeTorch(),
        )
        == "cpu"
    )


def test_bundle_runtime_device_keeps_transformer_sequence_classifier_off_mps() -> None:
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

    assert (
        _bundle_runtime_device(
            {
                "model_family": "transformer_sequence_classifier",
                "model_id": "answerdotai/ModernBERT-base",
            },
            FakeTorch(),
        )
        == "cpu"
    )


def test_safe_binary_completion_probability_handles_nonfinite_scores() -> None:
    assert _safe_binary_completion_probability(float("-inf"), float("-inf")) == 0.5
    assert _safe_binary_completion_probability(0.0, float("-inf")) == 1.0
    assert _safe_binary_completion_probability(float("-inf"), 0.0) == 0.0


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

    moved = _move_token_batch_to_device(
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
