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
ask-seattle train --data PATH --output-dir PATH [--eval-subreddit seattle]
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

Writes:

- `tfidf_logreg.joblib`
- `training_summary.json`

## `ask-seattle benchmark-variants`

Compare a few lightweight TF-IDF variants on the same held-out split.

```bash
ask-seattle benchmark-variants --data PATH --output-dir PATH [--eval-subreddit seattle]
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

Writes:

- one subdirectory per variant, each with:
  - `tfidf_logreg.joblib`
  - `training_summary.json`
- `variant_benchmark_summary.json`

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
  [--host 127.0.0.1] \
  [--port 8765] \
  [--log-level INFO] \
  [--retrain-every 0] \
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
- `--log-level`
  - optional
  - one of `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--retrain-every`
  - optional
  - defaults to `0`
  - when greater than zero, the bridge retrains after every N new effective training rows
- `--eval-subreddit`
  - optional
  - when set, bridge auto-retrain uses mixed reviewed data for training but restricts calibration and test evaluation to the named subreddit

## Make Targets

The canonical shortcuts are:

```bash
make retrain
make benchmark
make benchmark-variants
make bridge
```

Variables:

- `LABELS`
- `MODEL_DIR`
- `MODEL_PATH`
- `BENCHMARK_DIR`
- `BENCHMARK_VARIANTS_DIR`
- `EVAL_SUBREDDIT`
- `LOG_LEVEL`
- `RETRAIN_EVERY`

Examples:

```bash
make retrain MODEL_DIR=models/run-002
make benchmark BENCHMARK_DIR=models/benchmark-run-002
make benchmark-variants BENCHMARK_VARIANTS_DIR=models/benchmark-variants-run-002
make benchmark EVAL_SUBREDDIT=seattle
make bridge MODEL_PATH=models/run-002/tfidf_logreg.joblib LOG_LEVEL=DEBUG
make bridge RETRAIN_EVERY=25 EVAL_SUBREDDIT=seattle
make bridge RETRAIN_EVERY=25
```

Next:

- [Bridge API reference](bridge-api.md)
- [Reviewed data and artifacts reference](data-format.md)
