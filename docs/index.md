# Documentation Home

This page is the maintainer-facing entry point for the repository documentation.

Use it when you need to understand the current system, make changes safely, or find the right detailed document without scanning the whole repo.

## Start Here

- Want the system overview: [Architecture](architecture.md)
- Want to work on the code locally: [Development workflow](development.md)
- Want maintainer-specific guardrails and repo rules: [AGENTS.md](../AGENTS.md)
- Want to label posts in the browser: [How to label posts](how-to/label-posts.md)
- Want to retrain the model: [How to retrain](how-to/retrain.md)
- Want to compare the full six-model benchmark suite: [How to retrain](how-to/retrain.md)
- Want the preferred remote training path: [How to run training on RunPod](how-to/runpod-training.md)
- Want a no-cloud fallback on your own Windows GPU box: [How to run training on a remote Windows WSL box](how-to/remote-wsl-training.md)
- Need the exact CLI surface: [CLI reference](reference/cli.md)
- Need the bridge contract: [Bridge API reference](reference/bridge-api.md)
- Need the reviewed label schema or artifact layout: [Data format reference](reference/data-format.md)
- Need the model behavior explained in plain language: [Model and thresholds](explanation/model-and-thresholds.md)
- Need the moderation labeling rule itself: [Labeling policy](labeling_policy.md)

## Current System In One Paragraph

The project is a local-only review and classification loop. A Tampermonkey userscript reads the visible Reddit post in the browser, sends title/body text to a localhost bridge for `/check`, and sends reviewed labels to the same bridge for `/train`. The operational training path reads the reviewed JSONL file, normalizes and dedupes it, performs a deterministic random train/calibration/test split by default, fits a TF-IDF + logistic regression model, calibrates probabilities, and writes a `.joblib` model bundle plus `training_summary.json`. The benchmark-suite path reuses one persisted split manifest to compare six model families on the same examples.

## Documentation Map

The docs are organized by intent.

Repository-level maintainer rules live in [AGENTS.md](../AGENTS.md). Read that alongside these docs when changing public behavior, project scope, or model defaults.

### How-to

Task-oriented instructions for operators and maintainers:

- [Label posts in the browser](how-to/label-posts.md)
- [Retrain from reviewed labels](how-to/retrain.md)
- [Run training on RunPod](how-to/runpod-training.md)
- [Run training on a remote Windows WSL box](how-to/remote-wsl-training.md)
- [Troubleshoot common problems](how-to/troubleshoot.md)

### Reference

Exact interfaces and file contracts:

- [CLI reference](reference/cli.md)
- [Bridge API reference](reference/bridge-api.md)
- [Reviewed data and artifacts reference](reference/data-format.md)

### Explanation

Rationale and system behavior:

- [Architecture](architecture.md)
- [Model and thresholds](explanation/model-and-thresholds.md)
- [Roadmap](model_plan.md)

## Public Surfaces To Keep Accurate

When behavior changes, these are the docs that usually need review:

- root [README.md](../README.md) for the repo landing page
- [CLI reference](reference/cli.md) for command changes
- [Bridge API reference](reference/bridge-api.md) for request or response changes
- [How to label posts](how-to/label-posts.md) when the userscript UI changes
- [How to retrain](how-to/retrain.md) when the training loop changes
- [Reviewed data and artifacts reference](reference/data-format.md) when the local schema changes

## Core Invariants

These assumptions define the current project scope:

- no Reddit API reads
- no Reddit API writes
- browser-originated post text only
- local files only
- binary labels only: `askseattle` and `not_askseattle`
- one TF-IDF + logistic regression operational model path
- one six-model local benchmark suite for comparison work

If any of those change, treat it as a documentation-impacting change, not just an implementation detail.
