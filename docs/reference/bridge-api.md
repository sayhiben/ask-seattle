# Bridge API Reference

Use this page when you need the exact localhost HTTP contract used by the Tampermonkey helper or local tooling.

The bridge is local-only by design.

Default base URL:

- `http://127.0.0.1:8765`

## General Behavior

- CORS is open for local browser use.
- `OPTIONS` is supported.
- successful responses return JSON with `"ok": true`
- failures return JSON with `"ok": false` and an `"error"` message

## `GET /health`

Returns bridge health and startup configuration.

Example response:

```json
{
  "ok": true,
  "model_path": "/abs/path/to/tfidf_logreg.joblib",
  "label_path": "/abs/path/to/tampermonkey_labels.jsonl",
  "comparison_suite_path": "/abs/path/to/benchmark_suite_summary.json",
  "split_strategy": "random",
  "split_seed": 13,
  "comparison_models": [
    {
      "name": "semantic_embedding",
      "model_family": "semantic_embedding",
      "model_id": "sentence-transformers/all-MiniLM-L6-v2",
      "artifact_path": "/abs/path/to/semantic_embedding_logreg.joblib"
    },
    {
      "name": "transformer_sequence_classifier",
      "model_family": "transformer_sequence_classifier",
      "model_id": "microsoft/deberta-v3-small",
      "artifact_path": "/abs/path/to/transformer_bundle.joblib"
    }
  ],
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

For sparse image and link posts, the bridge also applies a stricter effective high-confidence threshold. Those posts can still score positive, but they need a stronger score to return `confidence_band: "high"`.

Example response:

This example is illustrative. Thresholds, scores, timestamps, and version strings vary by trained artifact.

```json
{
  "ok": true,
  "result": {
    "post_id": "abc123",
    "permalink": "https://www.reddit.com/r/example/comments/abc123/example/",
    "model_name": "tfidf_logreg",
    "model_version": "<varies>",
    "low_threshold": "<varies>",
    "high_threshold": "<varies>",
    "score": "<varies>",
    "score_raw": "<varies>",
    "score_calibrated": "<varies>",
    "label": "askseattle",
    "confidence_band": "high",
    "time_source": "collected_at",
    "created_at": "<varies>"
  },
  "comparisons": [
    {
      "name": "semantic_embedding",
      "model_family": "semantic_embedding",
      "model_id": "sentence-transformers/all-MiniLM-L6-v2",
      "result": {
        "model_name": "semantic_embedding_logreg",
        "score": "<varies>",
        "label": "askseattle",
        "confidence_band": "borderline"
      }
    },
    {
      "name": "transformer_sequence_classifier",
      "model_family": "transformer_sequence_classifier",
      "model_id": "microsoft/deberta-v3-small",
      "result": {
        "model_name": "transformer_sequence_classifier",
        "score": "<varies>",
        "label": "askseattle",
        "confidence_band": "high"
      }
    }
  ]
}
```

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
