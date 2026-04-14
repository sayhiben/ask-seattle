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

This example is illustrative. Real `comparison_models` arrays depend on which benchmark-suite artifacts currently exist and load successfully.

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
      "name": "semantic_minilm_tuned",
      "display_name": "Semantic MiniLM",
      "model_family": "semantic_embedding",
      "model_id": "sentence-transformers/all-MiniLM-L6-v2",
      "artifact_path": "/abs/path/to/models/benchmark-suite/semantic_minilm_tuned/semantic_embedding_logreg.joblib"
    },
    {
      "name": "semantic_qwen3_embedding_0_6b",
      "display_name": "Semantic Qwen3-Embedding",
      "model_family": "semantic_embedding",
      "model_id": "Qwen/Qwen3-Embedding-0.6B",
      "artifact_path": "/abs/path/to/models/benchmark-suite/semantic_qwen3_embedding_0_6b/semantic_embedding_logreg.joblib"
    },
    {
      "name": "semantic_jina_embeddings_v5_text_small_classification",
      "display_name": "Semantic Jina v5 Text Small Classification",
      "model_family": "semantic_embedding",
      "model_id": "jinaai/jina-embeddings-v5-text-small-classification",
      "artifact_path": "/abs/path/to/models/benchmark-suite/semantic_jina_embeddings_v5_text_small_classification/semantic_embedding_logreg.joblib"
    },
    {
      "name": "transformer_deberta_v3_small",
      "display_name": "Transformer DeBERTa-v3-small",
      "model_family": "transformer_sequence_classifier",
      "model_id": "microsoft/deberta-v3-small",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_deberta_v3_small/transformer_bundle.joblib"
    },
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
    },
    {
      "name": "causal_lm_qwen3_1_7b_lora",
      "display_name": "Causal LM Qwen3-1.7B",
      "model_family": "causal_lm_classifier",
      "model_id": "Qwen/Qwen3-1.7B",
      "artifact_path": "/abs/path/to/models/benchmark-suite/causal_lm_qwen3_1_7b_lora/causal_lm_bundle.joblib"
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
    "display_name": "tfidf_logreg",
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
  "comparison_models": [
    {
      "name": "semantic_minilm_tuned",
      "display_name": "Semantic MiniLM",
      "model_family": "semantic_embedding",
      "model_id": "sentence-transformers/all-MiniLM-L6-v2",
      "artifact_path": "/abs/path/to/models/benchmark-suite/semantic_minilm_tuned/semantic_embedding_logreg.joblib"
    },
    {
      "name": "semantic_qwen3_embedding_0_6b",
      "display_name": "Semantic Qwen3-Embedding",
      "model_family": "semantic_embedding",
      "model_id": "Qwen/Qwen3-Embedding-0.6B",
      "artifact_path": "/abs/path/to/models/benchmark-suite/semantic_qwen3_embedding_0_6b/semantic_embedding_logreg.joblib"
    },
    {
      "name": "semantic_jina_embeddings_v5_text_small_classification",
      "display_name": "Semantic Jina v5 Text Small Classification",
      "model_family": "semantic_embedding",
      "model_id": "jinaai/jina-embeddings-v5-text-small-classification",
      "artifact_path": "/abs/path/to/models/benchmark-suite/semantic_jina_embeddings_v5_text_small_classification/semantic_embedding_logreg.joblib"
    },
    {
      "name": "transformer_deberta_v3_small",
      "display_name": "Transformer DeBERTa-v3-small",
      "model_family": "transformer_sequence_classifier",
      "model_id": "microsoft/deberta-v3-small",
      "artifact_path": "/abs/path/to/models/benchmark-suite/transformer_deberta_v3_small/transformer_bundle.joblib"
    },
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
    },
    {
      "name": "causal_lm_qwen3_1_7b_lora",
      "display_name": "Causal LM Qwen3-1.7B",
      "model_family": "causal_lm_classifier",
      "model_id": "Qwen/Qwen3-1.7B",
      "artifact_path": "/abs/path/to/models/benchmark-suite/causal_lm_qwen3_1_7b_lora/causal_lm_bundle.joblib"
    }
  ],
  "comparisons": []
}
```

When the full benchmark suite artifacts exist, the bridge includes all available comparison models from the suite summary in `comparison_models`. If you set `include_comparisons: true`, the bridge also includes fully scored comparison entries in `comparisons`. The current expected full set is nine models total: TF-IDF, three semantic models, four encoder transformers, and one decoder-LLM.

If one comparison model fails during scoring, the bridge now keeps the main `result` and returns an `error` field for that comparison entry instead of failing the whole `/check` request.

On Apple Silicon, the bridge also keeps the transformer-backed semantic comparison models off MPS during `/check` because those families are not stable on the current MPS stack.

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
