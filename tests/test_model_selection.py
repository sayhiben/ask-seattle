from ask_seattle.data import LabeledPost
from ask_seattle.model import select_threshold, split_labeled_posts


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


def test_split_labeled_posts_is_deterministic() -> None:
    posts = [
        LabeledPost(title=f"ask {index}", selftext="", label=1, post_id=f"a{index}")
        for index in range(6)
    ] + [
        LabeledPost(title=f"news {index}", selftext="", label=0, post_id=f"n{index}")
        for index in range(6)
    ]

    first = split_labeled_posts(posts, validation_size=0.25, test_size=0.25, random_state=7)
    second = split_labeled_posts(posts, validation_size=0.25, test_size=0.25, random_state=7)

    assert [post.post_id for post in first.train] == [post.post_id for post in second.train]
    assert [post.post_id for post in first.validation] == [
        post.post_id for post in second.validation
    ]
    assert [post.post_id for post in first.test] == [post.post_id for post in second.test]
