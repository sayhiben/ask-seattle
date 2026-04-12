# Reviewed Data And Artifacts Reference

Use this page when you need the local schema for reviewed labels, training preparation rules, or model artifacts.

## Reviewed Label File

Canonical path:

- `data/processed/tampermonkey_labels.jsonl`

The file is append-or-update JSONL written by the bridge.

Minimum useful record:

```json
{
  "id": "abc123",
  "title": "Where should I stay?",
  "selftext": "Visiting next month.",
  "label": "askseattle"
}
```

## Required Fields

Training requires these logical fields:

- `title`
- `label`

Useful identity fields:

- `id`
- `permalink`

Useful time fields:

- `created_utc`
- `collected_at`

## Optional Fields

The browser helper may also send:

- `subreddit`
- `post_type`
- `content_href`
- `content_domain`
- `is_crosspost`
- `capture_context`
- `notes`

Training and bridge inference reuse the content-facing subset of those fields when available:

- `post_type`
- `content_domain`
- `is_crosspost`

Those fields are folded into the shared model text as normalized metadata tokens such as `POST_TYPE:image` and `CONTENT_DOMAIN:instagram_com`.

The shared text also includes lightweight structural tokens derived from the visible text:

- `TITLE_LEN_BUCKET:short|medium|long`
- `BODY_LEN_BUCKET:none|short|medium|long`
- `HAS_QUESTION_MARK:yes|no`
- `LOW_TEXT:yes|no`
- `SPARSE_MEDIA:yes` for link or image posts with low body text

## Label Normalization

Positive labels normalize to `askseattle`:

- `1`
- `true`
- `yes`
- `ask`
- `askseattle`
- `ask_seattle`

Negative labels normalize to `not_askseattle`:

- `0`
- `false`
- `no`
- `not`
- `not_askseattle`
- `not_ask_seattle`

## Preparation Rules

Before fitting the model, training applies these steps:

1. normalize labels
2. normalize body text
3. compute an exact text hash from normalized title + body
4. dedupe by identity:
   - `id`
   - `permalink`
5. dedupe again by exact text hash
6. derive `time_key` and `time_source`
7. build train, calibration, and test splits according to the requested split strategy

The dedupe behavior is last-write-wins.

## Time Derivation

`time_key` and `time_source` are derived in this order:

1. explicit `time_key`
2. `created_utc`
3. `collected_at`
4. `retrieved_at`

If no usable time field exists, the record can still participate in the default random split. Missing time fields only block participation when you explicitly choose `SPLIT_STRATEGY=time`.

## Model Artifacts

Training writes these files into the output directory:

- `tfidf_logreg.joblib`
- `training_summary.json`

The benchmark suite writes:

- `tfidf_recommended/training_summary.json`
- `semantic_embedding/training_summary.json`
- `transformer_sequence_classifier/training_summary.json`
- `benchmark_suite_summary.json`

## `tfidf_logreg.joblib`

The saved bundle includes:

- the fitted model pipeline
- the calibrator
- low and high thresholds
- threshold policy metadata
- model version metadata

## `training_summary.json`

Important sections:

- `prepared_data`
  - counts after normalization and dedupe
- `split`
  - train, calibration, and test counts
  - `split_strategy` and `split_seed`
  - optional `evaluation_subreddit` when calibration/test were restricted to one subreddit
  - time coverage when `split_strategy` is time-based
  - `coverage` by cohort and label for:
    - `post_type`
    - `low_text`
    - `sparse_media`
- `calibration`
  - calibrator availability and metrics
- `production_gate`
  - the held-out production-readiness requirements, including the precision target and the minimum number of high-confidence test predictions
- `threshold_selection`
  - low/high thresholds and threshold sweeps
- `metrics`
  - held-out high-confidence precision, recall, F1, and band counts
- `operating_metrics`
  - stable cross-model metrics for the strict auto bucket, the broader review queue, and queue rates
  - includes `slice_metrics` for:
    - post type
    - low-text vs richer-text posts
    - sparse-media vs non-sparse-media posts
- `training_balance`
  - the slice-aware positive weighting strategy used during fitting
  - bucket weights for underrepresented positive cohorts
  - train-split positive cohort counts and sample-weight summary
- `feature_audit`
  - top positive and negative TF-IDF features
  - top positive and negative features by channel
  - the custom word stopword list used by the title/body vectorizers
- `production_ready`
- `production_ready_blocked_reason`

Neural benchmark summaries replace `feature_audit` with model-specific metadata such as `embedding_summary` or `training_args`, but keep the same `split`, `calibration`, `threshold_selection`, `metrics`, and `operating_metrics` structure so results can be compared consistently.

Transformer benchmark summaries also include the current input and loss setup under `training_args`, including:

- `input_format`
- `max_length`
- `body_includes_metadata_tokens`
- `class_weighting`
- `class_weights`

## Storage Rules

Reviewed label files and model artifacts are intentionally untracked.

Ignored paths:

- `data/processed/`
- `models/`

Next:

- [How to retrain](../how-to/retrain.md)
- [Bridge API reference](bridge-api.md)
