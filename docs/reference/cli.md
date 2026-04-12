# CLI Reference

Use this page when you need the exact command surface for the current implementation.

## Commands

The CLI entry point is:

```bash
ask-seattle
```

## `ask-seattle train`

Train the TF-IDF classifier bundle from the reviewed label JSONL file.

```bash
ask-seattle train --data PATH --output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle]
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--output-dir`
  - required
  - directory where training artifacts are written
- `--eval-subreddit`
  - optional
  - when set, training still uses mixed reviewed data but restricts calibration and test evaluation to the named subreddit
- `--split-strategy`
  - optional
  - defaults to `random`
  - `random` uses a deterministic seeded split across all reviewed posts
  - `time` uses a chronological split over dated examples only
- `--split-seed`
  - optional
  - defaults to `13`
  - only affects `--split-strategy random`

Writes:

- `tfidf_logreg.joblib`
- `training_summary.json`

## `ask-seattle benchmark-variants`

Compare a few lightweight TF-IDF variants on the same held-out split.

```bash
ask-seattle benchmark-variants --data PATH --output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle]
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--output-dir`
  - required
  - directory where variant benchmark artifacts are written
- `--eval-subreddit`
  - optional
  - when set, all variants train on mixed reviewed data but restrict calibration and test evaluation to the named subreddit
- `--split-strategy`
  - optional
  - defaults to `random`
- `--split-seed`
  - optional
  - defaults to `13`
  - only affects `--split-strategy random`

Writes:

- one subdirectory per variant, each with:
  - `tfidf_logreg.joblib`
  - `training_summary.json`
- `variant_benchmark_summary.json`

## `ask-seattle benchmark-suite`

Compare the recommended TF-IDF baseline against a semantic embedding path and a transformer path on the same held-out split.

```bash
ask-seattle benchmark-suite --data PATH --output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle] [--semantic-model-id sentence-transformers/all-MiniLM-L6-v2] [--transformer-model-id microsoft/deberta-v3-small]
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--output-dir`
  - required
  - directory where suite benchmark artifacts are written
- `--eval-subreddit`
  - optional
  - when set, all benchmark paths train on mixed reviewed data but restrict calibration and test evaluation to the named subreddit
- `--split-strategy`
  - optional
  - defaults to `random`
- `--split-seed`
  - optional
  - defaults to `13`
  - only affects `--split-strategy random`
- `--semantic-model-id`
  - optional
  - defaults to `sentence-transformers/all-MiniLM-L6-v2`
- `--transformer-model-id`
  - optional
  - defaults to `microsoft/deberta-v3-small`

Current suite details:

- the shared model text includes normalized content metadata when available
- the transformer path uses title/body pair encoding
- the transformer path uses balanced class-weighted cross-entropy loss

Writes:

- `tfidf_recommended/training_summary.json`
- `semantic_embedding/training_summary.json`
- `transformer_sequence_classifier/training_summary.json`
- `benchmark_suite_summary.json`

This command requires the optional model dependencies:

```bash
python -m pip install -e ".[dev,models]"
```

## `ask-seattle check`

Classify a single post without running the bridge.

```bash
ask-seattle check --model PATH --title "..." [--selftext "..."]
```

Arguments:

- `--model`
  - required
  - path to a trained `.joblib` model bundle
- `--title`
  - required
- `--selftext`
  - optional
  - defaults to an empty string

Prints a JSON object containing the same classification payload shape used by the bridge.

## `ask-seattle serve-bridge`

Run the localhost bridge used by the Tampermonkey helper.

```bash
ask-seattle serve-bridge \
  --model PATH \
  [--labels PATH] \
  [--comparison-suite PATH] \
  [--host 127.0.0.1] \
  [--port 8765] \
  [--log-level INFO] \
  [--retrain-every 0] \
  [--split-strategy random|time] \
  [--split-seed 13] \
  [--eval-subreddit seattle]
```

Arguments:

- `--model`
  - required
  - path to a trained TF-IDF `.joblib` model bundle
- `--labels`
  - optional
  - defaults to `data/processed/tampermonkey_labels.jsonl`
- `--host`
  - optional
  - defaults to `127.0.0.1`
- `--port`
  - optional
  - defaults to `8765`
- `--comparison-suite`
  - optional
  - defaults to `models/benchmark-suite/benchmark_suite_summary.json`
  - when the summary exists, the bridge loads the other benchmark models so `/check` can return side-by-side comparison results
- `--log-level`
  - optional
  - one of `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--retrain-every`
  - optional
  - defaults to `0`
  - when greater than zero, the bridge retrains after every N new effective training rows
- `--split-strategy`
  - optional
  - defaults to `random`
  - controls the split policy used by bridge auto-retrain
- `--split-seed`
  - optional
  - defaults to `13`
  - only affects `--split-strategy random`
- `--eval-subreddit`
  - optional
  - when set, bridge auto-retrain uses mixed reviewed data for training but restricts calibration and test evaluation to the named subreddit

## Make Targets

The canonical shortcuts are:

```bash
make retrain
make benchmark
make benchmark-variants
make benchmark-suite
make bridge
```

Variables:

- `LABELS`
- `MODEL_DIR`
- `MODEL_PATH`
- `BENCHMARK_DIR`
- `BENCHMARK_VARIANTS_DIR`
- `BENCHMARK_SUITE_DIR`
- `BENCHMARK_SUITE_SUMMARY`
- `EVAL_SUBREDDIT`
- `SPLIT_STRATEGY`
- `SPLIT_SEED`
- `SEMANTIC_MODEL_ID`
- `TRANSFORMER_MODEL_ID`
- `LOG_LEVEL`
- `RETRAIN_EVERY`

Examples:

```bash
make retrain MODEL_DIR=models/run-002
make benchmark BENCHMARK_DIR=models/benchmark-run-002
make benchmark-variants BENCHMARK_VARIANTS_DIR=models/benchmark-variants-run-002
make benchmark-suite BENCHMARK_SUITE_DIR=models/benchmark-suite-run-002
make benchmark EVAL_SUBREDDIT=seattle
make benchmark-suite EVAL_SUBREDDIT=seattle
make benchmark EVAL_SUBREDDIT=seattle SPLIT_STRATEGY=time
make benchmark SPLIT_SEED=21
make bridge MODEL_PATH=models/run-002/tfidf_logreg.joblib LOG_LEVEL=DEBUG
make bridge RETRAIN_EVERY=25 EVAL_SUBREDDIT=seattle
make bridge RETRAIN_EVERY=25
```

Next:

- [Bridge API reference](bridge-api.md)
- [Reviewed data and artifacts reference](data-format.md)
