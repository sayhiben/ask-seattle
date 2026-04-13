# How To Run Training On RunPod

Use this page when you want to keep the normal local labeling workflow on your MacBook, but run retrains or benchmarks on an ephemeral RunPod Pod.

This is the preferred remote training path for the repository. If you want to avoid cloud spend entirely, use [How to run training on a remote Windows WSL box](remote-wsl-training.md) instead.

This path keeps the existing project boundary intact:

- the public GitHub repo contains code and docs only
- reviewed labels stay local to each contributor
- the label file is synced to the Pod for one run, then used as a normal local input there
- the Pod is ephemeral; persistence comes from a contributor-specific network volume
- successful cache volumes are retained for 3 days by default, then cleaned up on the next RunPod command

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

The helper does not keep Pods alive between runs. Only the retained cache volume can continue billing between runs, and that retention window is now bounded by default.

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

These commands now enforce a generous 6-hour remote target timeout by default. Override it with:

```bash
make retrain REMOTE=runpod EVAL_SUBREDDIT=seattle REMOTE_RUN_TIMEOUT=28800
```

## Corpus Handling

The reviewed label file is still local and ignored:

- default path: `data/processed/tampermonkey_labels.jsonl`

The RunPod helper:

- syncs that file to the Pod for the current run
- passes it through as `LABELS=...`
- never commits or pushes it
- never fetches any corpus from GitHub
- only syncs labels after the Pod passes a GPU smoke test
- removes the per-run remote label copy when the remote job exits

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

By default, a successful cache volume is retained for 72 hours so the next run can reuse:

- the repo checkout
- the virtualenv
- model downloads
- Hugging Face and pip caches

Expired cache volumes are deleted opportunistically at the start of the next RunPod command.

If a retained cache volume is pinned to a datacenter that no longer has the requested GPU capacity, the helper preserves that volume by default and fails clearly instead of silently discarding cached state. If you want to relocate the cache to another datacenter, opt in with:

```bash
make retrain REMOTE=runpod RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE=1 EVAL_SUBREDDIT=seattle
```

If the remote target exceeds its timeout, the helper terminates that target and the local orchestrator then tears the Pod down. The Pod-ready phase still has its own separate timeout.

## Default RunPod Settings

Defaults are controlled through Make variables:

- `RUNPOD_REPO`
- `RUNPOD_VOLUME_NAME`
- `RUNPOD_VOLUME_SIZE_GB`
- `RUNPOD_VOLUME_RETENTION_SECONDS`
- `RUNPOD_GPU_TYPES`
- `RUNPOD_DATA_CENTER_IDS`
- `RUNPOD_TEMPLATE_ID`
- `RUNPOD_SSH_KEY_PATH`
- `RUNPOD_IMAGE`

The default GPU preference order is:

1. `NVIDIA RTX A5000`
2. `NVIDIA GeForce RTX 4090`
3. `NVIDIA A40`

The default template is:

- `runpod-torch-v240`

The helper still accepts a raw image override, but the default is now template-first because that has been more reliable than direct image selection on RunPod.

The default cache retention is:

- `RUNPOD_VOLUME_RETENTION_SECONDS=259200`

Delete the retained cache volume immediately with:

```bash
make runpod-cleanup
```

The default datacenter preference order is:

1. `EU-RO-1`
2. `US-NC-1`
3. `US-KS-2`
4. `US-IL-1`
5. `US-GA-2`

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

If a run fails the GPU smoke test, the helper stops before syncing labels or starting training. That usually means the selected template/image or the provider runtime did not expose CUDA correctly inside the Pod.
