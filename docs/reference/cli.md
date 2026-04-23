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
- `make runpod-prune-volumes`
- `make install-git-hooks`
- `make secret-scan`
- `make repair-crossposts`
- `make retrain`
- `make benchmark`
- `make benchmark-variants`
- `make benchmark-seed-sweep`
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
make install-git-hooks
make secret-scan
make repair-crossposts
make retrain REMOTE=runpod EVAL_SUBREDDIT=seattle
make benchmark REMOTE=runpod EVAL_SUBREDDIT=seattle
make benchmark-variants REMOTE=runpod EVAL_SUBREDDIT=seattle
make benchmark-seed-sweep REMOTE=runpod EVAL_SUBREDDIT=seattle
make retrain REMOTE=runpod RUNPOD_FALLBACK_GPU_TYPES="NVIDIA L4,NVIDIA RTX A4000" EVAL_SUBREDDIT=seattle
make runpod-cleanup
make runpod-prune-volumes
make retrain REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
make benchmark REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

Useful make variables for the RunPod path:

- `RUNPOD_GPU_TYPES`
  - primary GPU preference order used when choosing a new datacenter and volume
- `RUNPOD_FALLBACK_GPU_TYPES`
  - extra same-datacenter GPUs to try when reusing an existing retained volume
- `RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE`
  - set to `1` to relocate a retained cache volume when neither the preferred nor fallback GPU list can be allocated in its pinned datacenter
- `REMOTE_RUN_TIMEOUT`
  - max remote target runtime in seconds before it is terminated
  - defaults to `21600` (6 hours)

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

## `make install-git-hooks`

Install the repository-managed Git pre-commit hook path.

```bash
make install-git-hooks
```

That command sets:

- `git config core.hooksPath .githooks`

The current pre-commit hook runs the staged secret scan before commit.

## `make secret-scan`

Scan tracked repository files for likely secrets.

```bash
make secret-scan
```

The scanner:

- skips ignored local artifact areas such as `data/processed/` and `models/`
- is used by the repository pre-commit hook on staged files
- is also intended to run in CI

If you need to suppress a false positive on a specific line, add:

- `secret-scan: allow`

## `make repair-crossposts`

Repair the local reviewed corpus by hydrating crosspost rows from their paired originals and dropping safe duplicate original rows.

```bash
make repair-crossposts
```

This rewrites:

- `data/processed/tampermonkey_labels.jsonl`

using the same crosspost-repair logic that training now applies automatically before split and dedupe.

## `ask-seattle repair-crossposts`

Backfill crosspost bodies from paired original rows and optionally rewrite the reviewed JSONL file in place.

```bash
ask-seattle repair-crossposts --data PATH [--output PATH]
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--output`
  - optional
  - output path for the repaired JSONL
  - defaults to rewriting `--data` in place

The command prints a JSON summary including:

- input and output record counts
- hydrated crosspost-row count
- duplicate-original rows removed
- unmatched or conflicting crosspost rows left untouched

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

Retrain the operational TF-IDF model plus the five-model comparison suite without running held-out benchmarks.

```bash
ask-seattle retrain-all --data PATH --operational-output-dir PATH --benchmark-output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle] [--transformer-model-id answerdotai/ModernBERT-base] [--transformer-secondary-model-id chandar-lab/NeoBERT] [--transformer-tertiary-model-id answerdotai/ModernBERT-large]
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
  - directory where the five suite model artifacts are written
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
- `--transformer-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-base`
- `--transformer-secondary-model-id`
  - optional
  - defaults to `chandar-lab/NeoBERT`
- `--transformer-tertiary-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-large`

Writes:

- operational TF-IDF artifacts under `--operational-output-dir`
- `suite_input.json`
- one per-model `training_summary.json` under `--benchmark-output-dir`
- `suite_training_summary.json`

## `ask-seattle benchmark-variants`

Compare the current TF-IDF default, the legacy baseline, and a bounded TF-IDF tuning grid on the same held-out split.

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

Current behavior:

- if `models/benchmark-suite/suite_input.json` already exists next to the requested output directory, the command reuses that exact manifest instead of creating a new split
- the TF-IDF sweep varies `classifier_c`, `char_weight`, `metadata_weight`, `min_df`, and `max_slice_positive_weight`

## `ask-seattle benchmark-suite`

Benchmark the active five-model artifact-backed suite on one shared held-out split, using already-trained suite artifacts, and add a derived hybrid-policy row when enough benchmarked models are available for the routed bridge policy.

```bash
ask-seattle benchmark-suite --data PATH --output-dir PATH [--split-strategy random|time] [--split-seed 13] [--eval-subreddit seattle] [--transformer-model-id answerdotai/ModernBERT-base] [--transformer-secondary-model-id chandar-lab/NeoBERT] [--transformer-tertiary-model-id answerdotai/ModernBERT-large] [--notes "free-form note"]
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
- `--transformer-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-base`
- `--transformer-secondary-model-id`
  - optional
  - defaults to `chandar-lab/NeoBERT`
- `--transformer-tertiary-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-large`
- `--notes`
  - optional
  - free-form text stored with the benchmark history record for this run

Current suite details:

- the command loads the shared `suite_input.json` manifest and benchmarks any compatible trained model artifacts already present for that manifest
- if a family is missing or incompatible, the command logs a warning and skips it instead of retraining it
- the transformer family includes ModernBERT-base, NeoBERT, and ModernBERT-large
- the stacked transformer decider is trained from those three transformer bundles using out-of-fold component probabilities on the suite train split, then benchmarked like any other artifact-backed suite model
- the transformer family restores the best epoch checkpoint and ranks candidates with a precision-first calibration key
- the shared model text includes normalized content metadata when available
- when TF-IDF plus at least two comparison models benchmark successfully, the aggregate summary also includes `hybrid_consensus_policy`, a benchmarked policy row with `artifact_path: null` and `policy_metadata`

Writes:

- updated per-model `training_summary.json` files for any model that was benchmarked successfully
- `benchmark_suite_summary.json`
- `benchmark_history.json`
- `history/<run_id>/benchmark_suite_summary.json`

This command requires the optional model dependencies:

```bash
python -m pip install -e ".[dev,models]"
```

## `ask-seattle benchmark-seed-sweep`

Retrain and benchmark selected suite models across multiple deterministic split seeds.

```bash
ask-seattle benchmark-seed-sweep --data PATH --output-dir PATH [--split-strategy random|time] [--eval-subreddit seattle] [--benchmark-seeds 13,21,34] [--benchmark-seed-models transformer_modernbert_base,transformer_neobert,transformer_modernbert_large] [--transformer-model-id answerdotai/ModernBERT-base] [--transformer-secondary-model-id chandar-lab/NeoBERT] [--transformer-tertiary-model-id answerdotai/ModernBERT-large]
```

Arguments:

- `--data`
  - required
  - path to reviewed `.jsonl` label data
- `--output-dir`
  - required
  - directory where the seed-sweep artifacts are written
- `--eval-subreddit`
  - optional
  - when set, each seeded run restricts calibration and test evaluation to the named subreddit
- `--split-strategy`
  - optional
  - defaults to `random`
- `--benchmark-seeds`
  - optional
  - defaults to `13,21,34`
  - comma-separated deterministic split seeds
- `--benchmark-seed-models`
  - optional
  - defaults to `transformer_modernbert_base,transformer_neobert,transformer_modernbert_large`
  - comma-separated suite model names to retrain and benchmark across those seeds
- `--transformer-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-base`
- `--transformer-secondary-model-id`
  - optional
  - defaults to `chandar-lab/NeoBERT`
- `--transformer-tertiary-model-id`
  - optional
  - defaults to `answerdotai/ModernBERT-large`

Writes:

- one per-seed subdirectory under `seed_sweeps/seed_<seed>/`
- `seed_sweeps/seed_sweep_summary.json`

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
  [--decider-policy primary_only|hybrid_consensus|stacked_transformer_decider] \
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
  - path to the trained primary `.joblib` model bundle used for fallback, audit, and bridge auto-retrain
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
  - when the summary exists, the bridge loads the benchmark comparison models and the stacked transformer decider artifact when available
- `--decider-policy`
  - optional
  - defaults to `stacked_transformer_decider`
  - `primary_only` keeps `/check` anchored to the active bridge model only
  - `hybrid_consensus` keeps the primary TF-IDF result under `decision_context.primary_result` but can also return a routed decider verdict for borderline or hard-slice posts when enough comparison models are loaded
  - when benchmark-suite history exists, `hybrid_consensus` derives its per-model weights from comparable benchmark runs and exposes them in the bridge response metadata
  - `stacked_transformer_decider` returns the trained stacked transformer policy as the main `result` when its suite artifact exists and falls back to the primary model when it does not
- `--log-level`
  - optional
  - one of `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--retrain-every`
  - optional
  - defaults to `0`
  - when greater than zero, the bridge retrains the primary TF-IDF bundle after every N new effective training rows
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
- `TRANSFORMER_MODEL_ID`
- `TRANSFORMER_SECONDARY_MODEL_ID`
- `TRANSFORMER_TERTIARY_MODEL_ID`
- `BENCHMARK_SEEDS`
- `BENCHMARK_SEED_MODELS`
- `BENCHMARK_NOTES`
- `LOG_LEVEL`
- `RETRAIN_EVERY`
- `DECIDER_POLICY`

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
make bridge DECIDER_POLICY=primary_only
make bridge RETRAIN_EVERY=25 EVAL_SUBREDDIT=seattle
make bridge RETRAIN_EVERY=25
```

Next:

- [Bridge API reference](bridge-api.md)
- [Reviewed data and artifacts reference](data-format.md)
