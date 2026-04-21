from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HYBRID_POLICY_NAME = "hybrid_consensus_policy"
HYBRID_POLICY_DISPLAY_NAME = "Hybrid consensus policy"
HYBRID_POLICY_MODEL_FAMILY = "hybrid_decider_policy"
HYBRID_WEIGHT_FORMULA_VERSION = "v1_benchmark_weighted_precision_first"

HYBRID_WEIGHT_READY_RATE_FLOOR = 0.2
HYBRID_WEIGHT_READY_RATE_SCALE = 0.8
HYBRID_WEIGHT_STRICT_RECALL = 0.55
HYBRID_WEIGHT_REVIEW_RECALL = 0.30
HYBRID_WEIGHT_PR_AUC = 0.15

HYBRID_LEGACY_PRIMARY_WEIGHT = 2.0
HYBRID_LEGACY_COMPARISON_WEIGHT = 1.0
HYBRID_MIN_COMPARISON_RESULTS = 2


def hybrid_route_reasons(*, row: dict[str, Any], primary_result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(primary_result.get("confidence_band") or "").strip().lower() == "borderline":
        reasons.append("primary_borderline")
    post_type = str(row.get("post_type") or "").strip().lower()
    if post_type == "image":
        reasons.append("image_post")
    elif post_type == "link":
        reasons.append("link_post")
    if str(row.get("is_low_text") or "").strip().lower() == "yes":
        reasons.append("low_text")
    if bool(row.get("is_sparse_media")):
        reasons.append("sparse_media")
    return list(dict.fromkeys(reasons))


def build_benchmark_weighted_hybrid_policy(
    *,
    active_models: list[dict[str, Any]],
    primary_model_name: str | None,
    split_strategy: str | None,
    evaluation_subreddit: str | None,
    benchmark_history_path: Path | None = None,
    comparison_suite_path: Path | None = None,
    benchmark_history: dict[str, Any] | None = None,
    suite_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = [
        {
            "name": str(model.get("name") or "").strip(),
            "display_name": str(model.get("display_name") or model.get("name") or "").strip(),
        }
        for model in active_models
        if str(model.get("name") or "").strip()
    ]
    active = _dedupe_active_models(active)
    if not active:
        return _uniform_fallback_policy(
            active_models=[],
            primary_model_name=primary_model_name,
            split_strategy=split_strategy,
            evaluation_subreddit=evaluation_subreddit,
        )

    history_payload = benchmark_history if isinstance(benchmark_history, dict) else _load_json_payload(
        benchmark_history_path
    )
    suite_payload = suite_summary if isinstance(suite_summary, dict) else _load_json_payload(comparison_suite_path)

    matched_runs = _matching_history_runs(
        history_payload=history_payload,
        active_model_names=[model["name"] for model in active],
        split_strategy=split_strategy or _suite_split_strategy(suite_payload),
        evaluation_subreddit=evaluation_subreddit or _suite_evaluation_subreddit(suite_payload),
    )
    if matched_runs:
        weights = _weights_from_history_runs(active=active, runs=matched_runs)
        return {
            "policy_name": HYBRID_POLICY_NAME,
            "display_name": HYBRID_POLICY_DISPLAY_NAME,
            "model_family": HYBRID_POLICY_MODEL_FAMILY,
            "weight_formula_version": HYBRID_WEIGHT_FORMULA_VERSION,
            "source": "benchmark_history",
            "source_path": str(benchmark_history_path) if benchmark_history_path is not None else None,
            "matched_run_count": len(matched_runs),
            "fallback_used": False,
            "primary_model_name": primary_model_name,
            "split_strategy": split_strategy or _suite_split_strategy(suite_payload),
            "evaluation_subreddit": evaluation_subreddit or _suite_evaluation_subreddit(suite_payload),
            "active_model_names": [model["name"] for model in active],
            "weights": weights,
        }

    suite_weights = _weights_from_suite_summary(active=active, suite_payload=suite_payload)
    if suite_weights is not None:
        return {
            "policy_name": HYBRID_POLICY_NAME,
            "display_name": HYBRID_POLICY_DISPLAY_NAME,
            "model_family": HYBRID_POLICY_MODEL_FAMILY,
            "weight_formula_version": HYBRID_WEIGHT_FORMULA_VERSION,
            "source": "suite_summary_fallback",
            "source_path": str(comparison_suite_path) if comparison_suite_path is not None else None,
            "matched_run_count": 1,
            "fallback_used": True,
            "primary_model_name": primary_model_name,
            "split_strategy": split_strategy or _suite_split_strategy(suite_payload),
            "evaluation_subreddit": evaluation_subreddit or _suite_evaluation_subreddit(suite_payload),
            "active_model_names": [model["name"] for model in active],
            "weights": suite_weights,
        }

    return _uniform_fallback_policy(
        active_models=active,
        primary_model_name=primary_model_name,
        split_strategy=split_strategy,
        evaluation_subreddit=evaluation_subreddit,
    )


def hybrid_policy_response(
    hybrid_policy: dict[str, Any] | None,
    *,
    applied_weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(hybrid_policy, dict):
        return None
    payload = {
        "policy_name": hybrid_policy.get("policy_name") or HYBRID_POLICY_NAME,
        "display_name": hybrid_policy.get("display_name") or HYBRID_POLICY_DISPLAY_NAME,
        "model_family": hybrid_policy.get("model_family") or HYBRID_POLICY_MODEL_FAMILY,
        "weight_formula_version": hybrid_policy.get("weight_formula_version") or HYBRID_WEIGHT_FORMULA_VERSION,
        "source": hybrid_policy.get("source") or "uniform_fallback",
        "source_path": hybrid_policy.get("source_path"),
        "matched_run_count": int(hybrid_policy.get("matched_run_count") or 0),
        "fallback_used": bool(hybrid_policy.get("fallback_used")),
        "primary_model_name": hybrid_policy.get("primary_model_name"),
        "split_strategy": hybrid_policy.get("split_strategy"),
        "evaluation_subreddit": hybrid_policy.get("evaluation_subreddit"),
        "active_model_names": list(hybrid_policy.get("active_model_names") or []),
        "weights": list(hybrid_policy.get("weights") or []),
    }
    if applied_weights:
        by_name = {str(item.get("name") or ""): item for item in payload["weights"] if isinstance(item, dict)}
        payload["applied_weights"] = [
            {
                "name": name,
                "display_name": (by_name.get(name) or {}).get("display_name"),
                "weight": float(weight),
            }
            for name, weight in applied_weights.items()
        ]
    return payload


def hybrid_decider_response(
    *,
    policy: str,
    primary_result: dict[str, Any],
    primary_model_name: str | None,
    row: dict[str, Any],
    comparisons: list[dict[str, Any]],
    route_reasons: list[str],
    hybrid_policy: dict[str, Any] | None,
    min_comparison_results: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    review_reasons = list(route_reasons)
    successful = [entry for entry in comparisons if isinstance(entry.get("result"), dict)]
    positive_votes = [
        entry
        for entry in successful
        if str((entry.get("result") or {}).get("label") or "") == "askseattle"
    ]
    negative_votes = [
        entry
        for entry in successful
        if str((entry.get("result") or {}).get("label") or "") == "not_askseattle"
    ]
    high_positive_votes = [
        entry
        for entry in positive_votes
        if str((entry.get("result") or {}).get("confidence_band") or "") == "high"
    ]
    decision_context: dict[str, Any] = {
        "policy": policy,
        "decision_source": "primary_model",
        "routed": bool(route_reasons),
        "route_reasons": route_reasons,
        "review_priority": "normal",
        "review_reasons": [],
        "primary_result": primary_result,
        "effective_high_threshold": float(primary_result.get("high_threshold") or 0.0),
        "successful_comparison_count": len(successful),
        "comparison_error_count": sum(1 for entry in comparisons if entry.get("error")),
        "positive_vote_count": len(positive_votes),
        "negative_vote_count": len(negative_votes),
        "high_positive_vote_count": len(high_positive_votes),
        "used_comparison_names": [str(entry.get("name") or "") for entry in successful],
        "hybrid_policy": hybrid_policy_response(hybrid_policy),
    }
    if positive_votes and negative_votes:
        review_reasons.append("comparison_disagreement")
    if policy != "hybrid_consensus" or not route_reasons:
        decision_context["review_reasons"] = list(dict.fromkeys(review_reasons))
        decision_context["review_priority"] = _review_priority(decision_context["review_reasons"])
        return None, decision_context
    if len(successful) < int(min_comparison_results):
        review_reasons.append("insufficient_comparison_support")
        decision_context["review_reasons"] = list(dict.fromkeys(review_reasons))
        decision_context["review_priority"] = _review_priority(decision_context["review_reasons"])
        return None, decision_context

    weighted_score, applied_weights, weight_source = _weighted_hybrid_score(
        primary_result=primary_result,
        primary_model_name=primary_model_name,
        successful=successful,
        hybrid_policy=hybrid_policy,
    )
    low_threshold = float(primary_result.get("low_threshold") or 0.0)
    high_threshold = float(primary_result.get("high_threshold") or low_threshold)
    label = "askseattle" if weighted_score >= low_threshold else "not_askseattle"
    confidence_band = "low"
    if weighted_score >= high_threshold:
        confidence_band = "high"
    elif weighted_score >= low_threshold:
        confidence_band = "borderline"
    decider_result = {
        **primary_result,
        "model_name": HYBRID_POLICY_NAME,
        "display_name": "Hybrid consensus",
        "score": float(weighted_score),
        "score_raw": float(weighted_score),
        "score_calibrated": float(weighted_score),
        "label": label,
        "confidence_band": confidence_band,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    if str(primary_result.get("label") or "") != label:
        review_reasons.append("label_changed_by_hybrid")
    elif str(primary_result.get("confidence_band") or "") != confidence_band:
        review_reasons.append("confidence_changed_by_hybrid")
    decision_context.update(
        {
            "decision_source": "hybrid_consensus",
            "review_reasons": list(dict.fromkeys(review_reasons)),
            "review_priority": _review_priority(review_reasons),
            "hybrid_score": float(weighted_score),
            "primary_weight": float(applied_weights.get(primary_model_name or "", 0.0)),
            "comparison_weight": None,
            "hybrid_weight_source": weight_source,
            "hybrid_policy": hybrid_policy_response(hybrid_policy, applied_weights=applied_weights),
        }
    )
    return decider_result, decision_context


def _dedupe_active_models(active: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for model in active:
        name = model["name"]
        if name in seen:
            continue
        seen.add(name)
        deduped.append(model)
    return deduped


def _uniform_fallback_policy(
    *,
    active_models: list[dict[str, Any]],
    primary_model_name: str | None,
    split_strategy: str | None,
    evaluation_subreddit: str | None,
) -> dict[str, Any]:
    count = len(active_models)
    uniform_weight = 1.0 / float(count) if count else 0.0
    return {
        "policy_name": HYBRID_POLICY_NAME,
        "display_name": HYBRID_POLICY_DISPLAY_NAME,
        "model_family": HYBRID_POLICY_MODEL_FAMILY,
        "weight_formula_version": HYBRID_WEIGHT_FORMULA_VERSION,
        "source": "uniform_fallback",
        "source_path": None,
        "matched_run_count": 0,
        "fallback_used": True,
        "primary_model_name": primary_model_name,
        "split_strategy": split_strategy,
        "evaluation_subreddit": evaluation_subreddit,
        "active_model_names": [model["name"] for model in active_models],
        "weights": [
            {
                "name": model["name"],
                "display_name": model.get("display_name"),
                "weight": uniform_weight,
                "raw_weight": uniform_weight,
                "ready_rate": 0.0,
                "auto_recall_at_precision_95": 0.0,
                "review_recall_at_precision_75": 0.0,
                "pr_auc": 0.0,
                "eligibility": 1.0,
            }
            for model in active_models
        ],
    }


def _load_json_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _suite_split_strategy(payload: dict[str, Any] | None) -> str | None:
    split = payload.get("split") if isinstance(payload, dict) else None
    return str(split.get("split_strategy") or "").strip() or None if isinstance(split, dict) else None


def _suite_evaluation_subreddit(payload: dict[str, Any] | None) -> str | None:
    split = payload.get("split") if isinstance(payload, dict) else None
    if isinstance(split, dict):
        value = str(split.get("evaluation_subreddit") or "").strip()
        return value or None
    value = str(payload.get("evaluation_subreddit") or "").strip() if isinstance(payload, dict) else ""
    return value or None


def _matching_history_runs(
    *,
    history_payload: dict[str, Any] | None,
    active_model_names: list[str],
    split_strategy: str | None,
    evaluation_subreddit: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(history_payload, dict):
        return []
    runs = history_payload.get("runs")
    if not isinstance(runs, list):
        return []
    requested = set(active_model_names)
    normalized_split = str(split_strategy or "").strip() or None
    normalized_subreddit = _normalize_subreddit(evaluation_subreddit)
    matched: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        split = run.get("split")
        if not isinstance(split, dict):
            continue
        if normalized_split is not None and str(split.get("split_strategy") or "").strip() != normalized_split:
            continue
        if normalized_subreddit is not None and _normalize_subreddit(split.get("evaluation_subreddit")) != normalized_subreddit:
            continue
        models = run.get("models")
        if not isinstance(models, list):
            continue
        available = {
            str(model.get("name") or "").strip()
            for model in models
            if isinstance(model, dict) and str(model.get("status") or "") == "ok"
        }
        if requested.issubset(available):
            matched.append(run)
    return matched


def _weights_from_history_runs(
    *,
    active: list[dict[str, str]],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics_by_name: dict[str, dict[str, list[float]]] = {
        model["name"]: {
            "ready": [],
            "auto_recall_at_precision_95": [],
            "review_recall_at_precision_75": [],
            "pr_auc": [],
        }
        for model in active
    }
    for run in runs:
        model_index = {
            str(model.get("name") or "").strip(): model
            for model in run.get("models") or []
            if isinstance(model, dict)
        }
        for model in active:
            payload = model_index.get(model["name"])
            if payload is None:
                continue
            metrics_by_name[model["name"]]["ready"].append(1.0 if bool(payload.get("production_ready")) else 0.0)
            metrics_by_name[model["name"]]["auto_recall_at_precision_95"].append(
                _float_or_zero(payload.get("auto_recall_at_precision_95"))
            )
            metrics_by_name[model["name"]]["review_recall_at_precision_75"].append(
                _float_or_zero(payload.get("review_recall_at_precision_75"))
            )
            metrics_by_name[model["name"]]["pr_auc"].append(_float_or_zero(payload.get("pr_auc")))
    rows: list[dict[str, Any]] = []
    for model in active:
        metrics = metrics_by_name[model["name"]]
        rows.append(
            _weight_row(
                name=model["name"],
                display_name=model.get("display_name"),
                ready_rate=_mean(metrics["ready"]),
                auto_recall_at_precision_95=_mean(metrics["auto_recall_at_precision_95"]),
                review_recall_at_precision_75=_mean(metrics["review_recall_at_precision_75"]),
                pr_auc=_mean(metrics["pr_auc"]),
            )
        )
    return _normalize_weight_rows(rows)


def _weights_from_suite_summary(
    *,
    active: list[dict[str, str]],
    suite_payload: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not isinstance(suite_payload, dict):
        return None
    models = suite_payload.get("models")
    if not isinstance(models, list):
        return None
    by_name = {
        str(model.get("name") or "").strip(): model
        for model in models
        if isinstance(model, dict) and str(model.get("status") or "") == "ok"
    }
    if not all(model["name"] in by_name for model in active):
        return None
    rows = [
        _weight_row(
            name=model["name"],
            display_name=model.get("display_name"),
            ready_rate=1.0 if bool(by_name[model["name"]].get("production_ready")) else 0.0,
            auto_recall_at_precision_95=_constraint_recall(
                by_name[model["name"]],
                "auto_recall_at_precision_95",
            ),
            review_recall_at_precision_75=_constraint_recall(
                by_name[model["name"]],
                "review_recall_at_precision_75",
            ),
            pr_auc=_ranking_metric(by_name[model["name"]], "pr_auc"),
        )
        for model in active
    ]
    return _normalize_weight_rows(rows)


def _weight_row(
    *,
    name: str,
    display_name: str | None,
    ready_rate: float,
    auto_recall_at_precision_95: float,
    review_recall_at_precision_75: float,
    pr_auc: float,
) -> dict[str, Any]:
    eligibility = HYBRID_WEIGHT_READY_RATE_FLOOR + (HYBRID_WEIGHT_READY_RATE_SCALE * float(ready_rate))
    raw_weight = float(eligibility) * (
        HYBRID_WEIGHT_STRICT_RECALL * float(auto_recall_at_precision_95)
        + HYBRID_WEIGHT_REVIEW_RECALL * float(review_recall_at_precision_75)
        + HYBRID_WEIGHT_PR_AUC * float(pr_auc)
    )
    return {
        "name": name,
        "display_name": display_name,
        "weight": 0.0,
        "raw_weight": raw_weight,
        "ready_rate": float(ready_rate),
        "auto_recall_at_precision_95": float(auto_recall_at_precision_95),
        "review_recall_at_precision_75": float(review_recall_at_precision_75),
        "pr_auc": float(pr_auc),
        "eligibility": float(eligibility),
    }


def _normalize_weight_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(float(row.get("raw_weight") or 0.0) for row in rows)
    if total <= 0:
        count = len(rows)
        uniform = 1.0 / float(count) if count else 0.0
        return [{**row, "weight": uniform, "raw_weight": uniform} for row in rows]
    return [{**row, "weight": float(row.get("raw_weight") or 0.0) / total} for row in rows]


def _constraint_recall(model: dict[str, Any], key: str) -> float:
    constraint_metrics = model.get("constraint_metrics")
    if not isinstance(constraint_metrics, dict):
        return 0.0
    payload = constraint_metrics.get(key)
    if not isinstance(payload, dict):
        return 0.0
    return _float_or_zero(payload.get("recall"))


def _ranking_metric(model: dict[str, Any], key: str) -> float:
    ranking_metrics = model.get("ranking_metrics")
    if not isinstance(ranking_metrics, dict):
        return 0.0
    return _float_or_zero(ranking_metrics.get(key))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_subreddit(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _review_priority(review_reasons: list[str]) -> str:
    reason_set = set(review_reasons)
    if not reason_set:
        return "normal"
    if {"label_changed_by_hybrid", "comparison_disagreement"} & reason_set:
        return "high"
    return "priority"


def _weighted_hybrid_score(
    *,
    primary_result: dict[str, Any],
    primary_model_name: str | None,
    successful: list[dict[str, Any]],
    hybrid_policy: dict[str, Any] | None,
) -> tuple[float, dict[str, float], str]:
    primary_score = float(primary_result.get("score") or 0.0)
    resolved_primary_name = str(primary_model_name or primary_result.get("model_name") or "").strip()
    if isinstance(hybrid_policy, dict):
        configured = {
            str(item.get("name") or "").strip(): float(item.get("weight") or 0.0)
            for item in hybrid_policy.get("weights") or []
            if isinstance(item, dict)
        }
        participants = [(resolved_primary_name, primary_score)] + [
            (str(entry.get("name") or "").strip(), float((entry.get("result") or {}).get("score") or 0.0))
            for entry in successful
        ]
        if participants and all(name in configured for name, _ in participants):
            total = sum(configured[name] for name, _ in participants)
            if total > 0:
                applied = {name: configured[name] / total for name, _ in participants}
                score = sum(applied[name] * value for name, value in participants)
                return float(score), applied, str(hybrid_policy.get("source") or "benchmark_history")

    denominator = HYBRID_LEGACY_PRIMARY_WEIGHT + (HYBRID_LEGACY_COMPARISON_WEIGHT * len(successful))
    legacy_score = (
        primary_score * HYBRID_LEGACY_PRIMARY_WEIGHT
        + sum(float((entry.get("result") or {}).get("score") or 0.0) for entry in successful)
    ) / float(denominator or 1.0)
    applied_weights = {
        resolved_primary_name: HYBRID_LEGACY_PRIMARY_WEIGHT / float(denominator or 1.0),
    }
    for entry in successful:
        name = str(entry.get("name") or "").strip()
        applied_weights[name] = HYBRID_LEGACY_COMPARISON_WEIGHT / float(denominator or 1.0)
    return float(legacy_score), applied_weights, "legacy_primary_weight"
