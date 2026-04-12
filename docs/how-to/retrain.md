# How To Retrain From Reviewed Labels

Use this page when you want to rebuild the local model from the reviewed label file.

## Normal Retrain

```bash
make retrain
```

That expands to:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli train \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/real-labels-precision-refresh
```

If you want to train on mixed reviewed labels but calibrate and test only on one subreddit, add `EVAL_SUBREDDIT`:

```bash
make retrain EVAL_SUBREDDIT=seattle
```

## Run A Benchmark Without Replacing The Main Model

```bash
make benchmark
```

That expands to:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli train \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/benchmark
```

Use this when you want a fresh held-out evaluation run and `training_summary.json`, but do not want to overwrite the default bridge model artifact.

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
- extra stopwords only
- lower `char_wb` weight only
- the current recommended default

All variants run on the exact same split.

## Compare TF-IDF, Semantic, And Transformer Paths

Install the optional model dependencies first:

```bash
python -m pip install -e ".[dev,models]"
```

Then run:

```bash
make benchmark-suite EVAL_SUBREDDIT=seattle
```

That expands to:

```bash
PYTHONPATH=src python3 -m ask_seattle.cli benchmark-suite \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/benchmark-suite \
  --split-strategy random \
  --split-seed 13 \
  --eval-subreddit seattle
```

The suite uses one shared split across all model families and writes:

- `tfidf_recommended/training_summary.json`
- `semantic_embedding/training_summary.json`
- `transformer_sequence_classifier/training_summary.json`
- `benchmark_suite_summary.json`

The default semantic and transformer paths are:

- `sentence-transformers/all-MiniLM-L6-v2`
- `microsoft/deberta-v3-small`

The transformer benchmark currently uses:

- title/body pair encoding instead of one flattened text string
- `max_length=384`
- balanced class-weighted cross-entropy loss
- the same shared metadata tokens in the body sequence

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
9. writes:
   - `tfidf_logreg.joblib`
   - `training_summary.json`

If `EVAL_SUBREDDIT` is set, training still uses mixed reviewed data, but the calibration and test slices are restricted to the named subreddit.

Default split policy:

- `SPLIT_STRATEGY=random`
- `SPLIT_SEED=13`

Use `SPLIT_STRATEGY=time` when you intentionally want future-facing evaluation over a longer collection window:

```bash
make benchmark EVAL_SUBREDDIT=seattle SPLIT_STRATEGY=time
```

The current default model applies one conservative refinement relative to the legacy baseline:

- lower `char_wb` feature weight
- slice-aware positive weighting during training so underrepresented positive cohorts such as low-text or sparse-media posts count more

The shared text representation also includes normalized content metadata when available:

- `HAS_BODY`
- `POST_TYPE`
- `CONTENT_DOMAIN`
- `CROSSPOST`
- `TITLE_LEN_BUCKET`
- `BODY_LEN_BUCKET`
- `HAS_QUESTION_MARK`
- `LOW_TEXT`
- `SPARSE_MEDIA`

## Output Location

Default output directory:

- `models/real-labels-precision-refresh/`

Override it:

```bash
make retrain MODEL_DIR=models/run-002
```

## After A Manual Retrain

Restart the bridge so it loads the new model artifact:

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

Look at:

- `models/real-labels-precision-refresh/training_summary.json`

Important fields:

- `prepared_data`
- `split`
- `split.coverage`
- `calibration`
- `production_gate`
- `threshold_selection`
- `metrics`
- `operating_metrics`
- `training_balance`
- `production_ready`
- `production_ready_blocked_reason`

The `operating_metrics` block is the ongoing comparison surface across model families:

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
- `slice_metrics`
  - the same operating metrics broken out by post type, low-text posts, and sparse-media posts

The `split.coverage` block tells you how many positives and negatives you actually have in those cohorts for train, calibration, and test.

The `training_balance` block tells you how the harness weighted underrepresented positive cohorts during fitting.

One important interpretation detail: sparse image/link posts are intentionally treated more conservatively for the `high` bucket. They can still score positive overall, but they need a stronger score to count as high confidence.

## Failure Modes

Training can still write artifacts even when the run is not production-ready. Common reasons:

- the calibration slice does not contain both classes
- the held-out high-confidence test precision misses the target
- the held-out high-confidence test bucket contains too few predicted positives to trust the precision result
- you chose `SPLIT_STRATEGY=time` but there are not enough dated examples to build the chronological split

Next:

- [How to troubleshoot](troubleshoot.md)
- [Reviewed data and artifacts reference](../reference/data-format.md)
- [Model and thresholds](../explanation/model-and-thresholds.md)
