# How To Retrain From Reviewed Labels

Use this page when you want to retrain the local models or benchmark the trained suite.

If you want to run those same targets on remote hardware instead of on the MacBook, see [How to run training on RunPod](runpod-training.md) for the preferred remote path or [How to run training on a remote Windows WSL box](remote-wsl-training.md) for the no-cloud fallback.

## Normal Retrain

```bash
make retrain
```

That expands to:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli retrain-all \
  --data data/processed/tampermonkey_labels.jsonl \
  --operational-output-dir models/real-labels-precision-refresh \
  --benchmark-output-dir models/benchmark-suite
```

This retrains:

- the operational TF-IDF model under `models/real-labels-precision-refresh/`
- the five-model comparison suite under `models/benchmark-suite/`

It does not run held-out benchmarks.

If you want to train on mixed reviewed labels but calibrate and test only on one subreddit, add `EVAL_SUBREDDIT`:

```bash
make retrain EVAL_SUBREDDIT=seattle
```

## Run Benchmarks On Trained Models

```bash
make benchmark
```

That expands to:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli benchmark-suite \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/benchmark-suite
```

Use this when you want a fresh held-out evaluation run for the trained suite models without retraining them.

If a model is missing or incompatible for the current `suite_input.json`, the benchmark logs a warning and skips it.

If you want a benchmark note stored in the history:

```bash
make benchmark EVAL_SUBREDDIT=seattle BENCHMARK_NOTES="after adding april labels"
```

To run the same make targets on RunPod, keep the target the same and add `REMOTE=runpod`:

```bash
make runpod-bootstrap
make retrain REMOTE=runpod EVAL_SUBREDDIT=seattle
make benchmark REMOTE=runpod EVAL_SUBREDDIT=seattle
```

The RunPod path now defaults to the official `runpod-torch-v240` template, prefers 48 GB cards (`RTX A6000`, `RTX 6000 Ada`, `L40S`) before falling back to the `4090`, and runs a hard GPU smoke test before syncing labels or starting training.
Successful RunPod cache volumes are retained for 3 days by default so repeated runs can reuse the checkout, venv, and model caches. Pods are still deleted at the end of every run. The retained venv is now reused directly unless the dependency environment key changes or the cached venv fails a health check. Expired retained volumes are pruned separately with `make runpod-prune-volumes` instead of being deleted automatically right before reuse.

To run the same make targets on your Windows GPU box over WSL, keep the target the same and add `REMOTE=wsl`:

```bash
make retrain REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
make benchmark REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

Both the WSL and RunPod remote wrappers now terminate the remote target after 6 hours by default. Override that with `REMOTE_RUN_TIMEOUT=<seconds>` if your run needs a larger budget.

Target only one subreddit for evaluation:

```bash
make benchmark EVAL_SUBREDDIT=seattle
```

## Compare Lightweight Variants

```bash
make benchmark-variants EVAL_SUBREDDIT=seattle
```

That expands to:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli benchmark-variants \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/benchmark-variants \
  --eval-subreddit seattle
```

Use this when you want to compare:

- the legacy baseline
- the current recommended default
- a TF-IDF tuning grid over `C`, `char_weight`, `metadata_weight`, `min_df`, and `max_slice_positive_weight`

If `models/benchmark-suite/suite_input.json` already exists, the variants run reuses that exact manifest. Otherwise it creates one fresh split and uses it for every variant in the sweep.

## Run A Selected-Model Seed Sweep

```bash
make benchmark-seed-sweep EVAL_SUBREDDIT=seattle
```

This is the stability pass for the top neural candidates. It retrains and benchmarks the selected comparison models across multiple deterministic split seeds and writes:

- `models/benchmark-suite/seed_sweeps/seed_sweep_summary.json`

By default it evaluates:

- `transformer_modernbert_base`
- `transformer_neobert`
- `transformer_modernbert_large`

Use it before promoting a model family change based on one benchmark run.

The aggregate summary now also reports:

- `production_ready_runs`
- `ready_rate`
- `min_auto_precision`
- `min_auto_recall`
- `mean_pr_auc`
- `std_pr_auc`

Treat the first model in `model_aggregates` as the current winner. That list is now ordered by:

1. `ready_rate`
2. `min_auto_precision`
3. mean auto recall
4. `mean_pr_auc`

## Compare The Full Five-Model Suite

Install the optional model dependencies first:

```bash
python -m pip install -e ".[dev,models]"
```

Then retrain:

```bash
make retrain EVAL_SUBREDDIT=seattle
```

Then benchmark:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli benchmark-suite \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/benchmark-suite \
  --split-strategy random \
  --split-seed 13 \
  --eval-subreddit seattle
```

`make benchmark-suite` is an alias for `make benchmark`.

The suite uses one shared split across all model families. Retraining writes:

- `suite_input.json`
- `tfidf_recommended/training_summary.json`
- `transformer_modernbert_base/training_summary.json`
- `transformer_neobert/training_summary.json`
- `transformer_modernbert_large/training_summary.json`
- `stacked_transformer_decider/training_summary.json`
- `suite_training_summary.json`

Benchmarking writes:

- `benchmark_suite_summary.json`
- `benchmark_history.json`
- `history/<run_id>/benchmark_suite_summary.json`

If TF-IDF plus at least two comparison models benchmark successfully for the current manifest, `benchmark_suite_summary.json` also includes one derived `hybrid_consensus_policy` row. That row reports the optional routed bridge policy on the same held-out split and does not correspond to a separately trained artifact.

The default five-model suite is:

- TF-IDF baseline
- ModernBERT-base sequence classifier
- NeoBERT sequence classifier
- ModernBERT-large sequence classifier
- stacked transformer decider trained from the three transformer scores plus shared post-shape features

Important implementation details:

- every family consumes the same persisted `suite_input.json` manifest
- rerunning `make retrain` resumes from any compatible completed per-model artifact already on disk for that manifest
- `make benchmark` never retrains missing models; it only benchmarks the compatible trained artifacts already present
- the stacked transformer decider is trained after the three transformer bundles and owns its own calibrator plus low/high thresholds
- the stacked transformer decider now fits its meta-model on out-of-fold transformer probabilities from the suite train split, then calibrates that stacked score on the normal suite calibration split
- the bridge hybrid policy weights now come from comparable benchmark history when available, then fall back to the latest suite summary, then to uniform weights
- `make benchmark-seed-sweep` is intentionally separate from `make benchmark`; it retrains only the selected comparison models across multiple seeds so the default retrain/benchmark contract stays simple
- the encoder transformer family uses title/body pair encoding, fits a sigmoid calibrator for every candidate, keeps the better candidate by calibrated strict-threshold readiness first, restores the best epoch checkpoint, and runs a small config grid for ModernBERT-base, NeoBERT, and ModernBERT-large
- that grid now includes a CUDA-only 512-token precision NeoBERT candidate and 48 GB CUDA-only ModernBERT-large long-context and precision-long-context candidates
- CUDA neural training now enables TF32 matmul when available to reduce remote runtime cost

## What Training Does

The training command:

1. reads the reviewed label JSONL file
2. normalizes labels
3. dedupes by identity and exact text hash
4. derives `time_key` and `time_source`
5. performs a deterministic random train/calibration/test split by default
6. fits the TF-IDF + logistic regression model
7. fits a sigmoid probability calibrator
8. selects low and high thresholds
9. writes trained artifacts and training summaries

That retrain step does not compute held-out test metrics. Benchmarking is a separate later step.

The raw `ask-seattle train` command still exists if you want a TF-IDF-only train-plus-benchmark run.

`make retrain` specifically runs the split flow instead:

- retrain all models first
- benchmark them second only when you explicitly run `make benchmark`

If `EVAL_SUBREDDIT` is set, training still uses mixed reviewed data, but the calibration and test slices are restricted to the named subreddit.

Default split policy:

- `SPLIT_STRATEGY=random`
- `SPLIT_SEED=13`

Use `SPLIT_STRATEGY=time` when you intentionally want future-facing evaluation over a longer collection window:

```bash
make benchmark EVAL_SUBREDDIT=seattle SPLIT_STRATEGY=time
```

The current default model applies these conservative refinements relative to the older baseline:

- metadata is fitted in its own exact-token feature channel instead of being mixed into the TF-IDF word and character channels
- `char_wb` now only sees natural title/body text, not synthetic metadata markers
- lexical title/body text normalizes visible URLs to `URL`, so raw transport syntax like `https`, `www`, and `://` does not carry direct weight
- the TF-IDF word stopword list also excludes `just`, `one`, and `some`, because that benchmarked better than leaving them active on the current `/r/seattle` split
- default `min_df` now scales with corpus size so larger label sets suppress more brittle low-support phrases
- high-threshold selection now requires the precision target, at least `5` calibration predictions in the strict bucket, and a bootstrap precision `p20` check on the calibration slice; if calibration cannot satisfy the stricter gate, the summary records the fallback reason explicitly
- TF-IDF review-threshold selection now maximizes review recall subject to `review precision >= 0.70`, while the strict auto bucket still targets `high precision >= 0.95`
- slice-aware positive weighting now boosts under-covered positives from existing labels without changing the capture format: `image` and `low_text` are active immediately, while `sparse_media` and `low_text_image` are tracked with minimum-support gates and stay observational until the split has enough positive examples

The shared post representation still includes normalized content metadata when available:

- `HAS_BODY`
- `POST_TYPE`
- `CONTENT_DOMAIN`
- `CROSSPOST`
- `TITLE_LEN_BUCKET`
- `BODY_LEN_BUCKET`
- `HAS_QUESTION_MARK`
- `LOW_TEXT`

## Bridge Decider Policy

`make bridge` now defaults to:

```bash
make bridge DECIDER_POLICY=stacked_transformer_decider
```

That does not change the operational retrain path. The TF-IDF bundle is still the cheap local model that bridge auto-retrain refreshes. What changes is the deployed `/check` verdict:

- when the stacked transformer decider artifact exists, `/check` returns that policy in `result`
- the primary TF-IDF verdict remains available under `decision_context.primary_result`
- if the stacked artifact is missing or fails, the bridge falls back to the TF-IDF verdict and records the reason in `decision_context.review_reasons`

The stacked decider is only refreshed when you rerun `make retrain` and then `make benchmark`. Bridge auto-retrain does not retrain the benchmark suite artifacts in the background.

If you want the routed bridge-side hybrid instead, use:

```bash
make bridge DECIDER_POLICY=hybrid_consensus
```

That policy is benchmark-weighted and still useful for comparison work on routed hard slices, but it is no longer the default top-line verdict.

If you want the bridge to expose the raw primary-model verdict only, use:

```bash
make bridge DECIDER_POLICY=primary_only
```
- `SPARSE_MEDIA`
- `IMAGE_NO_BODY`
- `LOW_TEXT_IMAGE`

`SPARSE_MEDIA` still appears in slice metrics and summaries, but model inputs only include it once the shared split has enough positive support to trust that signal. For the operational TF-IDF model specifically, the active metadata tokens live in a separate metadata feature channel instead of being mixed into the natural-language body and character channels.

The same TF-IDF path also normalizes visible URLs to a neutral `URL` token before vectorization. Domain and post-type information still survive in the metadata channel, but raw URL scaffolding no longer dominates the word or character audit.

## Output Location

Default output directory:

- `models/real-labels-precision-refresh/`

Override it:

```bash
make retrain MODEL_DIR=models/run-002
```

## After A Manual Retrain

After `make retrain`, run `make benchmark` if you want fresh suite metrics and a refreshed stacked decider artifact. Restart the bridge after retraining so it loads the new TF-IDF artifact and any updated suite models:

```bash
make bridge
```

The bridge only hot-reloads automatically when it was started with `RETRAIN_EVERY`.

## Auto-Retrain

Start the bridge with background retraining:

```bash
make bridge RETRAIN_EVERY=25
```

To keep the same target-domain evaluation policy during bridge auto-retrain:

```bash
make bridge RETRAIN_EVERY=25 EVAL_SUBREDDIT=seattle
```

That means:

- each saved label still only appends or updates the reviewed JSONL file
- the bridge recomputes the effective training-row count after normalization and dedupe
- once the effective row count grows by `25` since the last retrain trigger, the bridge retrains in the background and hot-reloads the model if the run succeeds

The threshold is based on effective training rows, not raw click count.

If an auto-retrain attempt fails, the bridge records the error and waits for another `RETRAIN_EVERY` effective rows before trying again. It does not immediately retry the same bad snapshot in a loop.

## Inspecting The Result

After `make retrain`, inspect:

- `models/real-labels-precision-refresh/training_summary.json`
- `models/benchmark-suite/suite_training_summary.json`

Those training-only summaries tell you:

- what data was prepared
- which split and manifest were used
- whether calibration and threshold selection succeeded
- whether `benchmark_status` is still `not_run`
- which artifacts were trained or reused

After `make benchmark`, inspect:

- `models/benchmark-suite/benchmark_suite_summary.json`
- per-model `training_summary.json` files under `models/benchmark-suite/*/`

Important benchmarked fields:

- `prepared_data`
- `split`
- `split.coverage`
- `calibration`
- `production_gate`
- `threshold_selection`
  - includes the review precision target used for the low threshold, the high precision target used for the strict bucket, the minimum calibration support required for the strict bucket, bootstrap strict-threshold diagnostics, and any strict-threshold fallback reason
- `metrics`
- `operating_metrics`
- `training_balance`
- `production_ready`
- `production_ready_blocked_reason`
- `benchmark_run`
  - run id, timestamp, optional notes, manifest fingerprint, and a short human-readable description of what this benchmark represents

The `operating_metrics` block is the ongoing comparison surface across model families once benchmarking has completed:

- `auto_band`
  - metrics for the strict `high` bucket only
- `review_queue`
  - metrics for everything at `low_threshold` or higher
- `queue_counts`
  - how many held-out posts land in `high`, `borderline`, and `low`
- `queue_rates`
  - what fraction of the held-out set falls into the auto band or review queue
- `positive_prevalence`
  - how common positives are in the held-out test set
- `ranking_metrics`
  - threshold-independent quality such as `pr_auc`
- `constraint_metrics`
  - fixed-constraint comparisons such as `auto_recall_at_precision_95` and `review_recall_at_precision_75`
- `slice_metrics`
  - the same operating metrics broken out by post type, low-text posts, and sparse-media posts
  - each slice now also records support counts and `support_status`

The `split.coverage` block tells you how many positives and negatives you actually have in those cohorts for train, calibration, and test.

The `training_balance` block tells you how the harness weighted underrepresented positive cohorts during fitting.

## Failure Modes

Retraining can still write artifacts even when benchmarking has not been run yet. In that case the summary shows:

- `benchmark_status = not_run`
- `production_ready = false`
- `production_ready_blocked_reason = benchmark_not_run`

After benchmarking, common reasons a model is still not production-ready include:

- the calibration slice does not contain both classes
- the held-out high-confidence test precision misses the target
- the held-out high-confidence test bucket contains too few predicted positives to trust the precision result
- you chose `SPLIT_STRATEGY=time` but there are not enough dated examples to build the chronological split

Next:

- [How to troubleshoot](troubleshoot.md)
- [Reviewed data and artifacts reference](../reference/data-format.md)
- [Model and thresholds](../explanation/model-and-thresholds.md)
