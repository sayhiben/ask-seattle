# Bridge API Reference

Use this page when you need the exact localhost HTTP contract used by the Tampermonkey helper or local tooling.

The bridge is local-only by design.

Default base URL:

- `http://localhost:8765`

## General Behavior

- CORS is open for local browser use.
- `OPTIONS` is supported.
- successful responses return JSON with `"ok": true`
- failures return JSON with `"ok": false` and an `"error"` message

## `GET /health`

Returns bridge health and startup configuration.

Example response:

This example is illustrative. Real `comparison_models` arrays depend on which supported benchmark-suite artifacts currently exist and load successfully. The bridge only exposes the active suite set: `tfidf_recommended` plus the three transformer models.

```json
{
  "ok": true,
  "model_path": "/abs/path/to/tfidf_logreg.joblib",
  "label_path": "/abs/path/to/tampermonkey_labels.jsonl",
  "comparison_suite_path": "/abs/path/to/benchmark_suite_summary.json",
  "split_strategy": "random",
  "split_seed": 13,
  "decider_policy": "hybrid_consensus",
  "comparison_models": [
    {
      "name": "transformer_modernbert_base",
      "display_name": "Transformer ModernBERT-base",
      "model_family": "transformer_sequence_classifier",
      "model_id": "answerdotai/ModernBERT-base",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_modernbert_base/transformer_bundle.joblib"
    },
    {
      "name": "transformer_neobert",
      "display_name": "Transformer NeoBERT",
      "model_family": "transformer_sequence_classifier",
      "model_id": "chandar-lab/NeoBERT",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_neobert/transformer_bundle.joblib"
    },
    {
      "name": "transformer_modernbert_large",
      "display_name": "Transformer ModernBERT-large",
      "model_family": "transformer_sequence_classifier",
      "model_id": "answerdotai/ModernBERT-large",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_modernbert_large/transformer_bundle.joblib"
    }
  ],
  "hybrid_policy": {
    "policy_name": "hybrid_consensus_policy",
    "display_name": "Hybrid consensus policy",
    "model_family": "hybrid_decider_policy",
    "weight_formula_version": "v1_benchmark_weighted_precision_first",
    "source": "benchmark_history",
    "source_path": "/abs/path/to/models/benchmark-suite/benchmark_history.json",
    "matched_run_count": 3,
    "fallback_used": false,
    "primary_model_name": "tfidf_recommended",
    "split_strategy": "random_eval_subreddit",
    "evaluation_subreddit": "seattle",
    "active_model_names": [
      "tfidf_recommended",
      "transformer_modernbert_base",
      "transformer_neobert",
      "transformer_modernbert_large"
    ],
    "weights": [
      {"name": "tfidf_recommended", "display_name": "TF-IDF", "weight": 0.06},
      {"name": "transformer_modernbert_base", "display_name": "Transformer ModernBERT-base", "weight": 0.45},
      {"name": "transformer_neobert", "display_name": "Transformer NeoBERT", "weight": 0.40},
      {"name": "transformer_modernbert_large", "display_name": "Transformer ModernBERT-large", "weight": 0.09}
    ]
  },
  "auto_retrain": null
}
```

## `POST /check`

Classify a post payload.

Required request fields:

- `title`

Optional request fields:

- `selftext`
- `id`
- `permalink`
- `created_utc`
- `collected_at`
- `time_source`
- `post_type`
- `content_domain`
- `is_crosspost`
- `include_comparisons`

Example request:

```json
{
  "id": "abc123",
  "permalink": "https://www.reddit.com/r/example/comments/abc123/example/",
  "title": "Where should I stay for a weekend trip?",
  "selftext": "Looking for hotel and food recommendations.",
  "collected_at": "2026-04-10T20:00:00+00:00",
  "post_type": "image",
  "content_domain": "instagram.com",
  "is_crosspost": false
}
```

If those optional metadata fields are present, the bridge includes them in the model input for scoring.

`include_comparisons` defaults to `false`. That means the normal fast `/check` path returns the active bridge model result plus `comparison_models` metadata, without waiting for every benchmark-suite model to score the same post.

The default bridge policy is `hybrid_consensus`. That keeps the primary bridge model result in `result`, but it can also compute a routed `decider_result` for borderline, low-text, image, link, or sparse-media posts when at least two comparison models are loaded successfully.

When the benchmark suite history exists, the routed hybrid score uses benchmark-informed weights derived from the active comparison set on comparable prior suite runs. The bridge prefers `benchmark_history.json`, falls back to the latest `benchmark_suite_summary.json`, and only falls back again to uniform weights if neither source is usable.

For low-text, image, and sparse-media posts, the bridge also applies a stricter effective high-confidence threshold. Those posts can still score positive, but they need a stronger score to return `confidence_band: "high"`.

Example response:

This example is illustrative. Thresholds, scores, timestamps, and version strings vary by trained artifact.

```json
{
  "ok": true,
  "result": {
    "post_id": "abc123",
    "permalink": "https://www.reddit.com/r/example/comments/abc123/example/",
    "model_name": "tfidf_logreg",
    "display_name": "tfidf_logreg",
    "model_version": "<varies>",
    "low_threshold": "<varies>",
    "high_threshold": "<effective per-post threshold>",
    "score": "<varies>",
    "score_raw": "<varies>",
    "score_calibrated": "<varies>",
    "label": "askseattle",
    "confidence_band": "high",
    "time_source": "collected_at",
    "created_at": "<varies>"
  },
  "decider_result": null,
  "decision_context": {
    "policy": "hybrid_consensus",
    "decision_source": "primary_model",
    "routed": true,
    "route_reasons": ["image_post"],
    "review_priority": "priority",
    "review_reasons": ["image_post", "insufficient_comparison_support"],
    "effective_high_threshold": "<effective per-post threshold>",
    "successful_comparison_count": 0,
    "comparison_error_count": 0,
    "positive_vote_count": 0,
    "negative_vote_count": 0,
    "high_positive_vote_count": 0,
    "used_comparison_names": [],
    "hybrid_policy": {
      "policy_name": "hybrid_consensus_policy",
      "source": "benchmark_history",
      "weights": [
        {"name": "tfidf_recommended", "weight": 0.06},
        {"name": "transformer_modernbert_base", "weight": 0.45},
        {"name": "transformer_neobert", "weight": 0.40},
        {"name": "transformer_modernbert_large", "weight": 0.09}
      ]
    }
  },
  "comparison_models": [
    {
      "name": "transformer_modernbert_base",
      "display_name": "Transformer ModernBERT-base",
      "model_family": "transformer_sequence_classifier",
      "model_id": "answerdotai/ModernBERT-base",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_modernbert_base/transformer_bundle.joblib"
    },
    {
      "name": "transformer_neobert",
      "display_name": "Transformer NeoBERT",
      "model_family": "transformer_sequence_classifier",
      "model_id": "chandar-lab/NeoBERT",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_neobert/transformer_bundle.joblib"
    },
    {
      "name": "transformer_modernbert_large",
      "display_name": "Transformer ModernBERT-large",
      "model_family": "transformer_sequence_classifier",
      "model_id": "answerdotai/ModernBERT-large",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_modernbert_large/transformer_bundle.joblib"
    }
  ],
  "comparisons": []
}
```

Response fields:

- `result`
  - always the primary bridge model verdict
- `decider_result`
  - optional routed verdict from the bridge decider policy
  - `null` when the bridge keeps the primary verdict only
- `decision_context`
  - routing and review metadata for the current policy
  - includes `hybrid_policy` when the bridge has a resolved weight policy for the active comparison set
- `comparison_models`
  - loaded comparison-model metadata
- `comparisons`
  - fully scored comparison entries

When the full benchmark suite artifacts exist, the bridge includes all available supported comparison models from the suite summary in `comparison_models`. If you set `include_comparisons: true`, the bridge also includes fully scored comparison entries in `comparisons`. Under `hybrid_consensus`, the bridge may also populate `comparisons` even when `include_comparisons` is `false` if the post was routed through the hybrid decider. The current expected full set is four models total: TF-IDF plus three encoder transformers. When the active bridge model is TF-IDF, the comparison list normally contains the three transformer models.

If one comparison model fails during scoring, the bridge now keeps the main `result` and returns an `error` field for that comparison entry instead of failing the whole `/check` request.

On Apple Silicon, the bridge keeps all neural comparison models off MPS during `/check` and `/check-comparison` because the current MPS stack is not stable enough for those families in local bridge inference.

`decision_context.review_priority` can be:

- `normal`
- `priority`
- `high`

`decision_context.review_reasons` currently includes bridge-routing and comparison signals such as:

- `primary_borderline`
- `image_post`
- `link_post`
- `low_text`
- `sparse_media`
- `comparison_disagreement`
- `label_changed_by_hybrid`
- `confidence_changed_by_hybrid`
- `insufficient_comparison_support`

If the bridge produced a routed `decider_result`, `decision_context` can also include:

- `hybrid_score`
- `primary_weight`
- `hybrid_weight_source`
- `hybrid_policy.applied_weights`

## `POST /check-comparison`

Classify a post payload with one named comparison model.

Required request fields:

- `name`
- `title`

Optional request fields:

- `selftext`
- `id`
- `permalink`
- `created_utc`
- `collected_at`
- `time_source`
- `post_type`
- `content_domain`
- `is_crosspost`

Example response:

```json
{
  "ok": true,
  "comparison": {
    "name": "transformer_modernbert_base",
    "display_name": "Transformer ModernBERT-base",
    "model_family": "transformer_sequence_classifier",
    "model_id": "answerdotai/ModernBERT-base",
    "result": {
      "model_name": "transformer_modernbert_base",
      "score": "<varies>",
      "label": "askseattle",
      "confidence_band": "borderline"
    }
  }
}
```

This endpoint exists so the userscript can render model cards incrementally instead of waiting for every comparison model to finish inside one `/check` request.

## `POST /train`

Append or update a reviewed label record.

Required request fields:

- `title`
- `label`

Accepted label families:

- positive: `1`, `true`, `yes`, `ask`, `askseattle`, `ask_seattle`
- negative: `0`, `false`, `no`, `not`, `not_askseattle`, `not_ask_seattle`

Optional request fields:

- `id`
- `created_utc`
- `permalink`
- `selftext`
- `notes`
- `collected_at`
- `subreddit`
- `post_type`
- `content_href`
- `content_domain`
- `is_crosspost`
- `capture_context`

Response fields:

- `saved`
- `label_path`
- `replaced`
- `auto_retrain`

`replaced` is `true` when an existing record with the same identity was overwritten.

## `POST /recorded`

Check whether the current post already exists in the reviewed label file.

Useful request fields:

- `id`
- `permalink`

Response fields:

- `recorded`
- `record`

The lookup is last-write-wins. If multiple historical records exist for the same identity, the most recent surviving one is returned.

## Errors

Typical error cases:

- missing required fields
- invalid JSON request body
- unknown endpoint
- bridge startup with a missing model file

Example error response:

```json
{
  "ok": false,
  "error": "missing required field: title"
}
```

Next:

- [CLI reference](cli.md)
- [Reviewed data and artifacts reference](data-format.md)
