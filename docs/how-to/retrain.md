# How To Retrain From Reviewed Labels

Use this page when you want to retrain the local models or benchmark the trained suite.

If you want to run those same targets on remote hardware instead of on the MacBook, see [How to run training on RunPod](runpod-training.md) or [How to run training on a remote Windows WSL box](remote-wsl-training.md).

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
- all six suite models under `models/benchmark-suite/`

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
- a TF-IDF tuning grid over `C`, `char_weight`, `metadata_weight`, and `min_df`

All variants run on the exact same split.

## Compare The Full Six-Model Suite

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
- `semantic_minilm_tuned/training_summary.json`
- `semantic_qwen3_embedding_0_6b/training_summary.json`
- `transformer_deberta_v3_small/training_summary.json`
- `transformer_modernbert_base/training_summary.json`
- `causal_lm_qwen3_1_7b_lora/training_summary.json`
- `suite_training_summary.json`

On Apple Silicon, the transformer-backed semantic embedding path (`semantic_qwen3_embedding_0_6b`) now bypasses MPS and uses CPU during training. That is slower, but it avoids current Metal backend failures for that model family on the supported Mac baseline.

Benchmarking writes:

- `benchmark_suite_summary.json`
- `benchmark_history.json`
- `history/<run_id>/benchmark_suite_summary.json`

The default six-model suite is:

- TF-IDF baseline
- tuned MiniLM semantic model
- Qwen3 embedding semantic model
- DeBERTa-v3-small sequence classifier
- ModernBERT-base sequence classifier
- Qwen3-1.7B LoRA causal-language-model classifier

Important implementation details:

- every family consumes the same persisted `suite_input.json` manifest
- rerunning `make retrain` resumes from any compatible completed per-model artifact already on disk for that manifest
- `make benchmark` never retrains missing models; it only benchmarks the compatible trained artifacts already present
- the semantic family now encodes title and body separately, concatenates those embeddings with a metadata one-hot block, and then fits the calibrated logistic-regression head
- the encoder transformer family uses title/body pair encoding, compares plain vs balanced cross-entropy, and keeps the better candidate by calibration PR-AUC with early stopping
- the decoder-LLM family scores the two candidate label continuations directly instead of free-form generation
- the decoder-LLM prompt now uses a compact contextual template with only the structured fields that materially helped on current data, so prompt-template changes still force that family to retrain instead of silently reusing an older summary
- on Apple Silicon, the decoder-LLM family currently bypasses MPS and uses the CPU fallback profile by default because the Qwen3 fine-tuning path is not stable on the current MPS stack

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
- TF-IDF review-threshold selection now maximizes review recall subject to `review precision >= 0.70`, while the strict auto bucket still targets `high precision >= 0.95`
- slice-aware positive weighting during training now uses only `image` and `low_text` as active tuning levers

The shared post representation still includes normalized content metadata when available:

- `HAS_BODY`
- `POST_TYPE`
- `CONTENT_DOMAIN`
- `CROSSPOST`
- `TITLE_LEN_BUCKET`
- `BODY_LEN_BUCKET`
- `HAS_QUESTION_MARK`
- `LOW_TEXT`
- `SPARSE_MEDIA`

For the operational TF-IDF model specifically, those metadata tokens now live in a separate metadata feature channel instead of being mixed into the natural-language body and character channels.

The same TF-IDF path also normalizes visible URLs to a neutral `URL` token before vectorization. Domain and post-type information still survive in the metadata channel, but raw URL scaffolding no longer dominates the word or character audit.

## Output Location

Default output directory:

- `models/real-labels-precision-refresh/`

Override it:

```bash
make retrain MODEL_DIR=models/run-002
```

## After A Manual Retrain

After `make retrain`, run `make benchmark` if you want fresh suite metrics. Restart the bridge after retraining so it loads the new TF-IDF artifact:

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
  - includes the review precision target used for the low threshold and the high precision target used for the strict bucket
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
