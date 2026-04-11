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
ask-seattle train --data PATH --output-dir PATH
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--output-dir`
  - required
  - directory where training artifacts are written

Writes:

- `tfidf_logreg.joblib`
- `training_summary.json`

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
  [--retrain-every 0]
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

## Make Targets

The canonical shortcuts are:

```bash
make retrain
make bridge
```

Variables:

- `LABELS`
- `MODEL_DIR`
- `MODEL_PATH`
- `LOG_LEVEL`
- `RETRAIN_EVERY`

Examples:

```bash
make retrain MODEL_DIR=models/run-002
make bridge MODEL_PATH=models/run-002/tfidf_logreg.joblib LOG_LEVEL=DEBUG
make bridge RETRAIN_EVERY=25
```

Next:

- [Bridge API reference](bridge-api.md)
- [Reviewed data and artifacts reference](data-format.md)
