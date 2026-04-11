from ask_seattle.data import LabeledPost
from ask_seattle.model import classify_post, select_decision_thresholds, select_threshold, split_labeled_posts


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


def test_split_labeled_posts_is_chronological() -> None:
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
    )

    assert {post.label for post in split.train} == {0, 1}
    assert [post.post_id for post in split.train] == ["p0", "p1", "p2", "p3"]
    assert [post.post_id for post in split.calibration] == []
    assert [post.post_id for post in split.test] == ["p4"]


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
