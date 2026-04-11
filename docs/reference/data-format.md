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
7. sort records by time for chronological splitting

The dedupe behavior is last-write-wins.

## Time Derivation

`time_key` and `time_source` are derived in this order:

1. explicit `time_key`
2. `created_utc`
3. `collected_at`
4. `retrieved_at`

If no usable time field exists, the record cannot participate in the chronological split.

## Model Artifacts

Training writes these files into the output directory:

- `tfidf_logreg.joblib`
- `training_summary.json`

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
  - time coverage
- `calibration`
  - calibrator availability and metrics
- `threshold_selection`
  - low/high thresholds and threshold sweeps
- `metrics`
  - held-out high-confidence precision, recall, F1, and band counts
- `feature_audit`
  - top positive and negative TF-IDF features
- `production_ready`
- `production_ready_blocked_reason`

## Storage Rules

Reviewed label files and model artifacts are intentionally untracked.

Ignored paths:

- `data/processed/`
- `models/`

Next:

- [How to retrain](../how-to/retrain.md)
- [Bridge API reference](bridge-api.md)
