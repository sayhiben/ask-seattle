# How To Run Training On RunPod

Use this page when you want to keep the normal local labeling workflow on your MacBook, but run retrains or benchmarks on an ephemeral RunPod Pod.

This path keeps the existing project boundary intact:

- the public GitHub repo contains code and docs only
- reviewed labels stay local to each contributor
- the label file is synced to the Pod for one run, then used as a normal local input there
- the Pod is ephemeral; persistence comes from a contributor-specific network volume

## What The RunPod Path Does

From your MacBook, the RunPod helper:

1. verifies GitHub, SSH, and RunPod CLI prerequisites
2. ensures the repo has a public `origin`
3. pushes the current `HEAD`
4. ensures your RunPod SSH key is registered
5. ensures your contributor-specific network volume exists
6. creates an ephemeral Secure Cloud Pod attached to that volume
7. syncs your local reviewed label file to the Pod for that run only
8. runs the existing `make` target remotely
9. pulls artifacts and logs back to your ignored local `models/` paths
10. deletes the Pod

## One-Time Bootstrap

Run this first:

```bash
make runpod-bootstrap
```

That command:

- verifies `gh`, `git`, `runpodctl`, `ssh`, and `rsync`
- checks that `runpodctl` exposes the modern Pod, datacenter, and network-volume commands
- creates `origin` as `sayhiben/ask-seattle` if missing
- pushes `main`
- registers your local public SSH key with RunPod

The bootstrap step requires a clean working tree.

## Day-To-Day Usage

Retrain remotely:

```bash
make retrain REMOTE=runpod EVAL_SUBREDDIT=seattle
```

Benchmark remotely:

```bash
make benchmark REMOTE=runpod EVAL_SUBREDDIT=seattle BENCHMARK_NOTES="runpod smoke"
```

Run TF-IDF variants remotely:

```bash
make benchmark-variants REMOTE=runpod EVAL_SUBREDDIT=seattle
```

## Corpus Handling

The reviewed label file is still local and ignored:

- default path: `data/processed/tampermonkey_labels.jsonl`

The RunPod helper:

- syncs that file to the Pod for the current run
- passes it through as `LABELS=...`
- never commits or pushes it
- never fetches any corpus from GitHub

Contributors should assume benchmark numbers are only comparable when they explicitly say:

- which local corpus they used
- which split settings they used

## Remote Persistence

By default the RunPod path uses:

- one persistent Secure Cloud network volume per contributor
- one ephemeral Pod per run

That means the expensive reusable state stays remote between runs:

- repo clone
- `.venv`
- Hugging Face cache
- pip cache

But the Pod itself is deleted after each run.

## Default RunPod Settings

Defaults are controlled through Make variables:

- `RUNPOD_REPO`
- `RUNPOD_VOLUME_NAME`
- `RUNPOD_VOLUME_SIZE_GB`
- `RUNPOD_GPU_TYPES`
- `RUNPOD_DATA_CENTER_IDS`
- `RUNPOD_SSH_KEY_PATH`
- `RUNPOD_IMAGE`

The default GPU preference order is:

1. `NVIDIA GeForce RTX 4090`
2. `NVIDIA RTX A5000`
3. `NVIDIA A40`

The default image is the official RunPod PyTorch image used by the remote helper.

## Artifacts And Logs

The remote helper pulls results back into the same ignored local paths used by the local workflow:

- `retrain`:
  - `models/real-labels-precision-refresh/`
  - `models/benchmark-suite/`
- `benchmark`:
  - `models/benchmark-suite/`
- `benchmark-variants`:
  - `models/benchmark-variants/`

It also pulls run metadata and logs into:

- `models/runpod-meta/<run_id>/`

## Troubleshooting

If `runpod-bootstrap` fails immediately on the CLI feature check, upgrade your local RunPod CLI first:

```bash
runpodctl update
```

If a run fails before the Pod comes up, no corpus material has been pushed to GitHub. The only remote copy is the per-run label file synced to the Pod or volume for that run.
