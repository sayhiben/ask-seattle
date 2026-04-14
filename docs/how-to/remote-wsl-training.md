# How To Run Training On A Remote Windows WSL Box

Use this page when you want to keep the normal local labeling workflow on your main machine, but run retrains or benchmarks on a separate Windows 11 PC over SSH.

The MacBook Pro M2 remains the primary supported machine. This guide is an optional speed-up path, not a required part of the normal workflow.

If you already have a Windows laptop with a suitable NVIDIA GPU, this is the no-cloud fallback remote training path. It avoids cloud spend and reuses the same local make targets.

This helper keeps the current project boundary intact:

- training still reads the browser-reviewed local JSONL label file
- the reviewed data still stays in local filesystem artifacts
- the remote machine just runs the same `make` targets inside WSL

The repository helper script is:

```bash
scripts/run_remote_training.sh
```

The same path is also available directly through the Makefile:

```bash
make retrain REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
make benchmark REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
make benchmark-variants REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

These commands now enforce a generous 6-hour remote runtime limit by default. Override it with:

```bash
make benchmark REMOTE=wsl REMOTE_WSL_HOST=gpu-win REMOTE_RUN_TIMEOUT=28800
```

## What The Helper Does

From your main laptop, the helper:

1. connects to the Windows machine over SSH
2. runs commands inside a named WSL distro with `wsl.exe`
3. syncs the current repo working tree, excluding local caches, `data/processed/`, and `models/`
4. syncs the reviewed label file separately
5. creates or refreshes `.venv` in WSL
6. runs one of the existing make targets there
7. optionally pulls the resulting artifact directory back to your local `models/`

It does not add a new training path. It is only a remote wrapper around the existing commands.

If your SSH session drops, the remote make target is still bounded by that timeout and will not run indefinitely.

## Recommended First-Time Remote Setup

On the Windows machine:

- install WSL 2 and Ubuntu
- install the latest NVIDIA Windows driver with WSL support if you want to run `benchmark-suite` on the GPU
- enable Windows OpenSSH Server

Then from your main machine run:

```bash
scripts/run_remote_training.sh \
  --host gpu-win \
  --bootstrap \
  --target benchmark-suite \
  --eval-subreddit seattle \
  --torch-index-url https://download.pytorch.org/whl/cu128
```

Notes:

- `--bootstrap` installs the required Ubuntu packages inside WSL:
  - `build-essential`
  - `git`
  - `make`
  - `python3`
  - `python3-venv`
  - `python3-pip`
  - `rsync`
- `--torch-index-url` is only for the CUDA-enabled PyTorch wheel. Use the current URL from the official PyTorch install selector.
- the helper defaults to `--target benchmark-suite`
- the helper defaults to `--eval-subreddit seattle`

## Day-To-Day Usage

Run the benchmark suite remotely through `make`:

```bash
make benchmark REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

Retrain remotely through `make`:

```bash
make retrain REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

Run TF-IDF variants remotely through `make`:

```bash
make benchmark-variants REMOTE=wsl REMOTE_WSL_HOST=gpu-win EVAL_SUBREDDIT=seattle
```

If you want the lower-level helper directly instead of the `make` wrapper:

Run the benchmark suite remotely:

```bash
scripts/run_remote_training.sh --host gpu-win --target benchmark
```

Retrain the operational model plus the full suite remotely:

```bash
scripts/run_remote_training.sh \
  --host gpu-win \
  --target retrain \
  --eval-subreddit seattle
```

Run the lightweight TF-IDF comparison remotely:

```bash
scripts/run_remote_training.sh \
  --host gpu-win \
  --target benchmark-variants \
  --eval-subreddit seattle
```

Use a non-default WSL distro or remote path:

```bash
scripts/run_remote_training.sh \
  --host gpu-win \
  --wsl-distro Ubuntu \
  --remote-dir /home/your-linux-user/ask-seattle
```

Pass through additional make overrides:

```bash
scripts/run_remote_training.sh \
  --host gpu-win \
  --target benchmark-suite \
  --make-arg SEMANTIC_MODEL_ID=sentence-transformers/all-mpnet-base-v2 \
  --make-arg TRANSFORMER_MODEL_ID=distilroberta-base
```

## Which Targets Benefit From The GPU

The current repository targets are:

- `make retrain`
- `make benchmark`
- `make benchmark-variants`
- `make benchmark-seed-sweep`
- `make benchmark-suite`

`make retrain` now retrains the operational TF-IDF model plus all nine suite models without held-out benchmarking.

`make benchmark` and `make benchmark-suite` are the same benchmark-only step.

`make benchmark-seed-sweep` retrains and benchmarks only the selected top neural models across multiple deterministic split seeds.

The GPU most strongly benefits:

- three semantic embedding paths
- four encoder transformer paths
- one decoder-LLM LoRA path

`make benchmark-variants` remains the lightweight CPU-oriented TF-IDF comparison path.

## Artifact Behavior

By default the helper pulls the target artifact directory back after the run:

- `retrain` -> `models/real-labels-precision-refresh/`
- `benchmark` -> `models/benchmark-suite/`
- `benchmark-variants` -> `models/benchmark-variants/`
- `benchmark-seed-sweep` -> `models/benchmark-suite/`
- `benchmark-suite` -> `models/benchmark-suite/`

If you want to leave the results on the remote machine only:

```bash
scripts/run_remote_training.sh --host gpu-win --no-pull-artifacts
```

## SSH Alias Example

Using an SSH alias keeps the command short:

```sshconfig
Host gpu-win
  HostName 192.168.1.50
  User your-windows-user
```

Then the helper can use:

```bash
scripts/run_remote_training.sh --host gpu-win
```

## Help

For the full option list:

```bash
scripts/run_remote_training.sh --help
```
