# Reviewed Data And Artifacts Reference

Use this page when you need the local schema for reviewed labels, training preparation rules, or model artifacts.

## Reviewed Label File

Canonical path:

- `data/processed/tampermonkey_labels.jsonl`

This file is local contributor state, not repository content. The public GitHub repo contains code and docs only. Remote training paths may sync the local reviewed label file to a remote machine for one run, but they do not commit or fetch corpora through GitHub.

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
- `crosspost_title`
- `crosspost_body`
- `capture_context`
- `notes`

Training and bridge inference reuse the content-facing subset of those fields when available:

- `post_type`
- `content_domain`
- `is_crosspost`
- `crosspost_body`

For crossposts, `crosspost_body` is the embedded original-post body captured by the browser helper when it can see or hydrate that source post. During normalization, the effective training/inference body becomes:

- visible outer `selftext` when present
- plus `crosspost_body` appended when present

That keeps normal text posts unchanged while letting crossposts behave more like the discovered post the reviewer actually labeled.

Those fields are normalized into metadata tokens such as `POST_TYPE:image` and `CONTENT_DOMAIN:instagram_com`.

Neural model families still consume those tokens through their shared text or prompt inputs. The operational TF-IDF model now keeps them in a separate exact-token metadata channel so word and character n-grams stay focused on natural title/body text.

The transformer benchmark families consume those tokens through their paired title/body inputs so the comparison suite still sees the same content metadata contract as the operational model.

The shared text also includes lightweight structural tokens derived from the visible text:

- `TITLE_LEN_BUCKET:short|medium|long`
- `BODY_LEN_BUCKET:none|short|medium|long`
- `HAS_QUESTION_MARK:yes|no`
- `LOW_TEXT:yes|no`
- `SPARSE_MEDIA:yes` for link or image posts with low body text
- `IMAGE_NO_BODY:yes` for image posts with no visible body text
- `LOW_TEXT_IMAGE:yes` for image posts that are otherwise low-text

`SPARSE_MEDIA:yes` is now representation-gated. It remains present in slice metrics and coverage summaries, but low-support splits keep it out of model inputs until the train/test positive counts are high enough to trust it.

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

1. repair crosspost rows:
   - backfill `crosspost_body` and effective `selftext` from paired originals when `content_href` matches another record permalink
   - drop the paired original row when that match is safe and labels agree
2. normalize labels
3. normalize body text
4. compute an exact text hash from normalized title + body
5. dedupe by identity:
   - `id`
   - `permalink`
6. dedupe again by exact text hash
7. derive `time_key` and `time_source`
8. build train, calibration, and test splits according to the requested split strategy

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

- `suite_training_summary.json`
- `tfidf_recommended/training_summary.json`
- `transformer_modernbert_base/training_summary.json`
- `transformer_neobert/training_summary.json`
- `transformer_modernbert_large/training_summary.json`
- `stacked_transformer_decider/training_summary.json`
- `suite_input.json`
- `benchmark_suite_summary.json`
- `benchmark_history.json`
- `history/<run_id>/benchmark_suite_summary.json`
- `seed_sweeps/seed_sweep_summary.json`

The optional RunPod remote wrapper also writes local pulled metadata and logs under:

- `models/runpod-meta/<run_id>/`

`benchmark_suite_summary.json` can now contain two kinds of rows under `models`:

- artifact-backed model rows such as `tfidf_recommended`, the transformer bundles, or `stacked_transformer_decider`
- a derived policy row named `hybrid_consensus_policy`

That hybrid policy row is benchmark-only:

- `artifact_path` is `null`
- `result_source` is `benchmarked_policy`
- `policy_metadata` records the active hybrid weighting source, routed-rate diagnostics, and review-reason counts

## `suite_input.json`

The benchmark suite persists one shared manifest before running any model families.

The manifest includes:

- the prepared and deduped records used for the run
- the train, calibration, and test assignments
- `split_strategy`
- `split_seed`
- optional `evaluation_subreddit`
- prepared-data summary counts

Every benchmark-suite model family consumes that same manifest so the shared five-model comparison remains apples-to-apples.

`make retrain` writes or refreshes this manifest before training the suite models. `make benchmark` loads the same manifest later and only benchmarks compatible trained artifacts for that manifest.

## `tfidf_logreg.joblib`

The saved bundle includes:

- the fitted model pipeline
- the calibrator
- low and high thresholds
- threshold policy metadata
- representation metadata describing whether sparse-media and image-specific markers were included in model inputs
- model version metadata

## `training_summary.json`

Important sections:

- `input_data`
  - the local reviewed-label path plus a SHA-256 fingerprint for the corpus snapshot used by the run
- `runtime_environment`
  - Python/platform metadata plus package versions for the model stack used to write the summary
- `prepared_data`
  - counts after normalization and dedupe
- `benchmark_run`
  - benchmark-only metadata for the latest suite evaluation
  - includes:
    - `run_id`
    - `created_at`
    - optional `notes`
    - `representation`
    - `input_data_fingerprint`
    - `suite_manifest_fingerprint`
- `suite_resume`
  - benchmark-suite-only metadata used to decide whether an existing per-model artifact can be reused on a later run
- `benchmark_status`
  - `not_run` after retraining only
  - `complete` after held-out benchmarking finishes for that artifact
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
  - includes the current review precision target for low-threshold selection and the high precision target for strict auto selection
  - includes `minimum_high_confidence_calibration_predictions`
  - includes `high_threshold_fallback_used`
- `metrics`
  - held-out high-confidence precision, recall, F1, and band counts
- `operating_metrics`
  - stable cross-model metrics for the strict auto bucket, the broader review queue, and queue rates
  - includes `ranking_metrics.pr_auc`
  - includes `constraint_metrics` such as:
    - `auto_recall_at_precision_95`
    - `review_recall_at_precision_75`
  - includes `slice_metrics` for:
    - post type
    - low-text vs richer-text posts
    - sparse-media vs non-sparse-media posts
  - each slice now includes support counts and `support_status`
- `training_balance`
  - the slice-aware positive weighting strategy used during fitting
  - bucket weights for the active tuning levers only (`image` and `low_text`)
  - train-split positive cohort counts and sample-weight summary
- `feature_audit`
  - top positive and negative TF-IDF features
  - top positive and negative features by channel
  - the operational TF-IDF channels now include `metadata_token` alongside `title_word`, `body_word`, and `char_wb`
  - lexical TF-IDF channels normalize visible URLs to `URL`, so raw URL syntax is less likely to surface as a learned term
  - the custom word stopword list used by the title/body vectorizers, including the current default extra exclusions for `just`, `one`, and `some`
- `production_ready`
- `production_ready_blocked_reason`

Neural benchmark summaries replace `feature_audit` with model-specific metadata such as `training_args`, but keep the same `split`, `calibration`, `threshold_selection`, `metrics`, and `operating_metrics` structure so results can be compared consistently.

Neural summaries now also record the active `representation_config`, so later benchmark or inference runs can reconstruct the same metadata-token policy that the model was trained with.

For `stacked_transformer_decider`, `training_args` and `oof_training` also record how many out-of-fold component-training folds were used and whether the meta-model was fit from true OOF component probabilities or an explicit fallback path.

Training-only suite summaries intentionally omit `metrics` and `operating_metrics` until a later benchmark step writes them.

The selected-model seed sweep writes `seed_sweeps/seed_sweep_summary.json`, which includes:

- the selected model names
- the evaluated split seeds
- one per-seed run block with per-model metrics
- per-model aggregate mean/std summaries for:
  - `pr_auc`
  - `auto_precision`
  - `auto_recall`
  - `review_precision`
  - `review_recall`
  - `auto_recall_at_precision_95`
  - `review_recall_at_precision_75`

## `benchmark_history.json`

The suite benchmark also appends a compact historical index.

Each entry includes:

- run id
- timestamp
- optional notes
- a short human-readable description of what the benchmark represents
- split summary
- prepared-data counts
- a condensed per-model metrics snapshot

This file is meant for longitudinal tracking. The full immutable snapshot for each indexed run lives under `history/<run_id>/benchmark_suite_summary.json`.

Transformer benchmark summaries also include the current input and loss setup under `training_args`, including:

- `input_format`
- `max_length`
- `body_includes_metadata_tokens`
- `class_weighting`
- `class_weights`
- `candidate_profile`
- `candidate_results`
- `cuda_matmul`

## Storage Rules

Reviewed label files and model artifacts are intentionally untracked.

Ignored paths:

- `data/processed/`
- `models/`

Next:

- [How to retrain](../how-to/retrain.md)
- [Bridge API reference](bridge-api.md)
