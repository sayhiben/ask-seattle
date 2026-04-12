from ask_seattle.data import LabeledPost
from ask_seattle.model import (
    build_inference_row,
    build_pipeline,
    classify_post,
    select_decision_thresholds,
    select_threshold,
    split_labeled_posts,
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

    assert "some" not in title_vectorizer.get_stop_words()
    assert "just" not in title_vectorizer.get_stop_words()
    assert "one" not in title_vectorizer.get_stop_words()
    assert "some" not in body_vectorizer.get_stop_words()
    assert "just" not in body_vectorizer.get_stop_words()
    assert "one" not in body_vectorizer.get_stop_words()
    assert "the" not in title_vectorizer.vocabulary_
    assert "and" not in title_vectorizer.vocabulary_
    assert "the" not in body_vectorizer.vocabulary_
    assert "is" not in body_vectorizer.vocabulary_
    assert "best" in title_vectorizer.vocabulary_
    assert "recommendations" in body_vectorizer.vocabulary_


def test_build_inference_row_includes_metadata_tokens_in_body_and_text() -> None:
    row = build_inference_row(
        title="Looking for ideas",
        selftext="",
        post_type="image",
        content_domain="www.instagram.com",
        is_crosspost=True,
    )

    assert "HAS_BODY:no" in row["body"]
    assert "TITLE_LEN_BUCKET:short" in row["body"]
    assert "BODY_LEN_BUCKET:none" in row["body"]
    assert "HAS_QUESTION_MARK:no" in row["body"]
    assert "LOW_TEXT:yes" in row["body"]
    assert "POST_TYPE:image" in row["body"]
    assert "CONTENT_DOMAIN:instagram_com" in row["body"]
    assert "CROSSPOST:yes" in row["body"]
    assert "SPARSE_MEDIA:yes" in row["body"]
    assert row["body_length_bucket"] == "none"
    assert row["is_sparse_media"] is True
    assert row["text"].startswith("TITLE: Looking for ideas")


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
    assert "some" not in audit["word_stopwords"]
    assert "just" not in audit["word_stopwords"]
    assert "one" not in audit["word_stopwords"]
    assert set(audit["top_positive_by_channel"]) == {"title_word", "body_word", "char_wb"}
    assert set(audit["top_negative_by_channel"]) == {"title_word", "body_word", "char_wb"}
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


def test_classify_post_downgrades_sparse_media_high_confidence_to_borderline() -> None:
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
