from __future__ import annotations

from dataclasses import dataclass

from ask_seattle.data import utc_now_iso
from ask_seattle.model import score_post


@dataclass(frozen=True)
class ModerationDecision:
    post_id: str | None
    permalink: str | None
    model_name: str
    model_version: str
    threshold: float
    score: float
    predicted_label: str
    should_flag: bool
    created_at: str

    @property
    def label(self) -> str:
        return self.predicted_label

    @property
    def should_remove(self) -> bool:
        return self.should_flag

    @property
    def removal_message(self) -> None:
        return None


def decide(
    bundle: dict,
    *,
    title: str,
    selftext: str = "",
    post_id: str | None = None,
    permalink: str | None = None,
    threshold: float | None = None,
) -> ModerationDecision:
    active_threshold = float(bundle.get("threshold", 0.85) if threshold is None else threshold)
    score = score_post(bundle, title=title, selftext=selftext)
    should_flag = score >= active_threshold

    return ModerationDecision(
        post_id=post_id,
        permalink=permalink,
        model_name=str(bundle.get("model_name") or bundle.get("model_type") or "unknown"),
        model_version=str(bundle.get("model_version") or bundle.get("version") or "unknown"),
        threshold=active_threshold,
        score=score,
        predicted_label="askseattle" if should_flag else "not_askseattle",
        should_flag=should_flag,
        created_at=utc_now_iso(),
    )
