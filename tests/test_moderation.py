from ask_seattle.moderation import decide


class FakeModel:
    named_steps = {"classifier": None}

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.9] for _ in texts]


class FakeClassifier:
    classes_ = [0, 1]


def test_decide_removes_when_score_meets_threshold() -> None:
    model = FakeModel()
    model.named_steps["classifier"] = FakeClassifier()
    bundle = {
        "model": model,
        "model_type": "tfidf",
        "model_name": "tfidf_logreg",
        "model_version": "test",
        "threshold": 0.85,
    }

    decision = decide(bundle, title="Where should I stay?", post_id="abc", permalink="https://reddit.test")

    assert decision.should_flag is True
    assert decision.label == "askseattle"
    assert decision.post_id == "abc"
    assert decision.permalink == "https://reddit.test"
    assert decision.model_name == "tfidf_logreg"
    assert decision.removal_message is None
