# CLI Reference

Use this page when you need the exact command surface for the current implementation.

For the remote wrappers around the existing make targets, see [How to run training on RunPod](../how-to/runpod-training.md) and [How to run training on a remote Windows WSL box](../how-to/remote-wsl-training.md).

## Commands

The CLI entry point is:

```bash
ask-seattle
```

## Make Targets

The normal operator entry points are still the repository make targets:

- `make runpod-bootstrap`
- `make runpod-cleanup`
- `make retrain`
- `make benchmark`
- `make benchmark-variants`
- `make benchmark-suite`
- `make bridge`

Remote RunPod execution is selected by adding:

```bash
REMOTE=runpod
```

Remote Windows WSL execution is selected by adding:

```bash
REMOTE=wsl
```

Examples:

```bash
make retrain REMOTE=runpod EVAL_SUBREDDIT=seattle
make benchmark REMOTE=runpod EVAL_SUBREDDIT=seattle
make benchmark-variants REMOTE=runpod EVAL_SUBREDDIT=seattle
make runpod-cleanup
make retrain REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
make benchmark REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

Useful make variables for the WSL path:

- `REMOTE_WSL_HOST`
  - SSH target for the Windows machine
  - defaults to `gpu-win`
- `REMOTE_WSL_DISTRO`
  - WSL distro name
  - defaults to `Ubuntu`
- `REMOTE_WSL_DIR`
  - optional Linux repo path inside WSL
- `REMOTE_WSL_BOOTSTRAP`
  - set to `1` to install Ubuntu package prerequisites before the run
- `REMOTE_WSL_TORCH_INDEX_URL`
  - CUDA PyTorch wheel index URL used by the WSL helper
  - defaults to `https://download.pytorch.org/whl/cu128`
- `REMOTE_RUN_TIMEOUT`
  - max remote target runtime in seconds before it is terminated
  - defaults to `21600` (6 hours)

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

By default this command still evaluates the held-out test slice and writes benchmark metrics for the TF-IDF model only.

## `ask-seattle retrain-all`

Retrain the operational TF-IDF model plus the full six-model comparison suite without running held-out benchmarks.

```bash
ask-seattle retrain-all --data PATH --operational-output-dir PATH --benchmark-output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle] [--semantic-model-id sentence-transformers/all-MiniLM-L6-v2] [--semantic-secondary-model-id Qwen/Qwen3-Embedding-0.6B] [--transformer-model-id microsoft/deberta-v3-small] [--transformer-secondary-model-id answerdotai/ModernBERT-base] [--causal-lm-model-id Qwen/Qwen3-1.7B]
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--operational-output-dir`
  - required
  - directory where the operational TF-IDF artifacts are written
- `--benchmark-output-dir`
  - required
  - directory where the six suite model artifacts are written
- `--eval-subreddit`
  - optional
  - when set, training still uses mixed reviewed data but restricts later calibration/test evaluation to the named subreddit
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
- `--semantic-secondary-model-id`
  - optional
  - defaults to `Qwen/Qwen3-Embedding-0.6B`
- `--transformer-model-id`
  - optional
  - defaults to `microsoft/deberta-v3-small`
- `--transformer-secondary-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-base`
- `--causal-lm-model-id`
  - optional
  - defaults to `Qwen/Qwen3-1.7B`

Writes:

- operational TF-IDF artifacts under `--operational-output-dir`
- `suite_input.json`
- one per-model `training_summary.json` under `--benchmark-output-dir`
- `suite_training_summary.json`

## `ask-seattle benchmark-variants`

Compare the current TF-IDF default, the legacy baseline, and a small TF-IDF tuning grid on the same held-out split.

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

Benchmark the full six-model suite on one shared held-out split, using already-trained suite artifacts.

```bash
ask-seattle benchmark-suite --data PATH --output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle] [--semantic-model-id sentence-transformers/all-MiniLM-L6-v2] [--semantic-secondary-model-id Qwen/Qwen3-Embedding-0.6B] [--transformer-model-id microsoft/deberta-v3-small] [--transformer-secondary-model-id answerdotai/ModernBERT-base] [--causal-lm-model-id Qwen/Qwen3-1.7B] [--notes "free-form note"]
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
  - when set, all benchmark paths restrict calibration and test evaluation to the named subreddit while still using the same mixed-data training manifest
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
- `--semantic-secondary-model-id`
  - optional
  - defaults to `Qwen/Qwen3-Embedding-0.6B`
- `--transformer-model-id`
  - optional
  - defaults to `microsoft/deberta-v3-small`
- `--transformer-secondary-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-base`
- `--causal-lm-model-id`
  - optional
  - defaults to `Qwen/Qwen3-1.7B`
- `--notes`
  - optional
  - free-form text stored with the benchmark history record for this run

Current suite details:

- the command loads the shared `suite_input.json` manifest and benchmarks any compatible trained model artifacts already present for that manifest
- if a family is missing or incompatible, the command logs a warning and skips it instead of retraining it
- the semantic family includes a tuned MiniLM path and a Qwen3 embedding path
- the transformer family includes DeBERTa-v3-small and ModernBERT-base
- the decoder family includes a Qwen3-1.7B LoRA classifier scored via two candidate label continuations
- on Apple Silicon, the decoder family currently defaults to `cpu_fallback` instead of MPS because the Qwen3 fine-tuning path is not stable on the current MPS stack
- the shared model text includes normalized content metadata when available

Writes:

- updated per-model `training_summary.json` files for any model that was benchmarked successfully
- `benchmark_suite_summary.json`
- `benchmark_history.json`
- `history/<run_id>/benchmark_suite_summary.json`

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
- `BENCHMARK_VARIANTS_DIR`
- `BENCHMARK_SUITE_DIR`
- `BENCHMARK_SUITE_SUMMARY`
- `EVAL_SUBREDDIT`
- `SPLIT_STRATEGY`
- `SPLIT_SEED`
- `SEMANTIC_MODEL_ID`
- `SEMANTIC_SECONDARY_MODEL_ID`
- `TRANSFORMER_MODEL_ID`
- `TRANSFORMER_SECONDARY_MODEL_ID`
- `CAUSAL_LM_MODEL_ID`
- `BENCHMARK_NOTES`
- `LOG_LEVEL`
- `RETRAIN_EVERY`

Examples:

```bash
make retrain MODEL_DIR=models/run-002 BENCHMARK_SUITE_DIR=models/benchmark-suite-run-002
make benchmark BENCHMARK_SUITE_DIR=models/benchmark-suite-run-002
make benchmark-variants BENCHMARK_VARIANTS_DIR=models/benchmark-variants-run-002
make benchmark-suite BENCHMARK_SUITE_DIR=models/benchmark-suite-run-002
make benchmark EVAL_SUBREDDIT=seattle
make benchmark EVAL_SUBREDDIT=seattle BENCHMARK_NOTES="after adding april labels"
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
