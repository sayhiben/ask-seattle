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

This example is illustrative. Real `comparison_models` arrays depend on which supported benchmark-suite artifacts currently exist and load successfully. The bridge only exposes the active side-by-side comparison set: the three transformer bundles. The primary TF-IDF bundle and the stacked decider are surfaced separately.

```json
{
  "ok": true,
  "model_path": "/abs/path/to/tfidf_logreg.joblib",
  "label_path": "/abs/path/to/tampermonkey_labels.jsonl",
  "comparison_suite_path": "/abs/path/to/benchmark_suite_summary.json",
  "split_strategy": "random",
  "split_seed": 13,
  "decider_policy": "stacked_transformer_decider",
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
  "stacked_decider_model": {
    "name": "stacked_transformer_decider",
    "display_name": "Stacked transformer decider",
    "model_family": "stacked_transformer_decider",
    "model_id": null,
    "artifact_path": "/abs/path/to/models/benchmark-suite/stacked_transformer_decider/stacked_transformer_decider.joblib"
  },
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
- `crosspost_title`
- `crosspost_body`
- `include_comparisons`

Example request:

```json
{
  "id": "abc123",
  "permalink": "https://www.reddit.com/r/example/comments/abc123/example/",
  "title": "Where should I stay for a weekend trip?",
  "selftext": "Looking for hotel and food recommendations.",
  "crosspost_body": "Original ask-style body from the linked source subreddit.",
  "collected_at": "2026-04-10T20:00:00+00:00",
  "post_type": "image",
  "content_domain": "instagram.com",
  "is_crosspost": false
}
```

If those optional metadata fields are present, the bridge includes them in the model input for scoring.

When `crosspost_body` is present, the bridge appends it to `selftext` before scoring. That lets crossposts carry the original embedded post body through the normal `/check` path even when the outer crosspost shell has little or no body text of its own.

`include_comparisons` defaults to `false`. That means the normal fast `/check` path returns the effective bridge verdict plus `comparison_models` metadata, without waiting for every benchmark-suite model to score the same post.

The default bridge policy is `stacked_transformer_decider`. That returns the trained stacked transformer policy in `result` when the suite artifact is available, while keeping the primary TF-IDF bridge verdict under `decision_context.primary_result` for audit and fallback.

If you start the bridge with `DECIDER_POLICY=hybrid_consensus`, it can also compute a routed `decider_result` for borderline, low-text, image, link, or sparse-media posts when at least two comparison models are loaded successfully.

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
    "model_name": "stacked_transformer_decider",
    "display_name": "Stacked transformer decider",
    "model_version": "<varies>",
    "low_threshold": "<stacked low threshold>",
    "high_threshold": "<stacked high threshold>",
    "score": "<varies>",
    "score_raw": "<varies>",
    "score_calibrated": "<varies>",
    "label": "askseattle",
    "confidence_band": "high",
    "time_source": "collected_at",
    "created_at": "<varies>"
  },
  "decider_result": {
    "post_id": "abc123",
    "permalink": "https://www.reddit.com/r/example/comments/abc123/example/",
    "model_name": "stacked_transformer_decider",
    "display_name": "Stacked transformer decider",
    "model_version": "<varies>",
    "low_threshold": "<stacked low threshold>",
    "high_threshold": "<stacked high threshold>",
    "score": "<varies>",
    "score_raw": "<varies>",
    "score_calibrated": "<varies>",
    "label": "askseattle",
    "confidence_band": "high",
    "time_source": "collected_at",
    "created_at": "<varies>"
  },
  "decision_context": {
    "policy": "stacked_transformer_decider",
    "decision_source": "stacked_transformer_decider",
    "routed": false,
    "route_reasons": [],
    "review_priority": "high",
    "review_reasons": ["label_changed_by_stacked_decider"],
    "effective_high_threshold": "<stacked high threshold>",
    "primary_result": {
      "model_name": "tfidf_logreg",
      "display_name": "tfidf_logreg",
      "label": "not_askseattle",
      "confidence_band": "low",
      "score": "<varies>"
    },
    "stacked_decider_model": {
      "name": "stacked_transformer_decider",
      "display_name": "Stacked transformer decider",
      "model_family": "stacked_transformer_decider",
      "artifact_path": "/abs/path/to/models/benchmark-suite/stacked_transformer_decider/stacked_transformer_decider.joblib"
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
  - always the effective deployed bridge verdict for the active policy
- `decider_result`
  - optional verdict produced by the active decider policy
  - under `stacked_transformer_decider`, this usually matches `result`
  - under `hybrid_consensus`, this only appears when a post was actually routed
  - `null` when the bridge keeps the primary verdict only
- `decision_context`
  - routing and review metadata for the current policy
  - includes `hybrid_policy` when the bridge has a resolved weight policy for the active comparison set
  - includes `primary_result` so callers can audit the TF-IDF fallback result even when `result` came from the stacked or hybrid policy
- `comparison_models`
  - loaded comparison-model metadata
- `comparisons`
  - fully scored comparison entries

When the full benchmark suite artifacts exist, the bridge includes all available supported comparison models from the suite summary in `comparison_models`. If you set `include_comparisons: true`, the bridge also includes fully scored comparison entries in `comparisons`. Under `hybrid_consensus`, the bridge may also populate `comparisons` even when `include_comparisons` is `false` if the post was routed through the hybrid decider. The current expected full suite is five artifact-backed models total, but the side-by-side comparison list normally contains only the three transformer models because TF-IDF is the primary fallback model and the stacked decider is surfaced separately as the active policy artifact.

If one comparison model fails during scoring, the bridge now keeps the main `result` and returns an `error` field for that comparison entry instead of failing the whole `/check` request.

On Apple Silicon, the bridge keeps all neural comparison models off MPS during `/check` and `/check-comparison` because the current MPS stack is not stable enough for those families in local bridge inference.

`decision_context.review_priority` can be:

- `normal`
- `elevated`
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
- `label_changed_by_stacked_decider`
- `confidence_changed_by_stacked_decider`
- `stacked_decider_unavailable`
- `stacked_decider_failed`
- `stacked_decider_missing_result`

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
- `crosspost_title`
- `crosspost_body`
- `capture_context`

Response fields:

- `saved`
- `label_path`
- `replaced`
- `auto_retrain`

`replaced` is `true` when an existing record with the same identity was overwritten.

When `crosspost_body` is present, the bridge appends it to `selftext` before saving the reviewed label row, and also preserves the raw `crosspost_body` field in the JSONL record.

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
