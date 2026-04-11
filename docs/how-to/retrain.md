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

## What Training Does

The training command:

1. reads the reviewed label JSONL file
2. normalizes labels
3. dedupes by identity and exact text hash
4. derives `time_key` and `time_source`
5. performs a chronological train/calibration/test split
6. fits the TF-IDF + logistic regression model
7. fits a sigmoid probability calibrator
8. selects low and high thresholds
9. writes:
   - `tfidf_logreg.joblib`
   - `training_summary.json`

If `EVAL_SUBREDDIT` is set, training still uses mixed reviewed data before the evaluation window, but the calibration and test slices are restricted to the named subreddit.

The current default model applies one conservative refinement relative to the legacy baseline:

- lower `char_wb` feature weight

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
- `calibration`
- `threshold_selection`
- `metrics`
- `production_ready`
- `production_ready_blocked_reason`

## Failure Modes

Training can still write artifacts even when the run is not production-ready. Common reasons:

- the calibration slice does not contain both classes
- the held-out high-confidence test precision misses the target
- there are not enough dated examples to make the chronological split

Next:

- [How to troubleshoot](troubleshoot.md)
- [Reviewed data and artifacts reference](../reference/data-format.md)
- [Model and thresholds](../explanation/model-and-thresholds.md)
