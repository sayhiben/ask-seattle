# AGENTS

This file is maintainer guidance for people and coding agents working in this repository.

Use it to stay inside the intended project boundary, avoid avoidable churn, and keep the public docs accurate.

## Project goals

The current project goal is narrow:

- classify Reddit posts as `askseattle` or `not_askseattle`
- keep the workflow local and cheap
- train only from browser-captured reviewed labels
- expose a localhost check path that other moderation tools can build on later

This repository is not trying to be a full moderation bot. It is building a reliable local classifier and review loop first.

## Current scope

The supported workflow is:

1. open a Reddit post in the browser
2. let the Tampermonkey helper capture visible title/body text
3. send `/check` and `/train` requests to the localhost bridge
4. retrain the local TF-IDF model from the reviewed JSONL file

Current architectural invariants:

- no Reddit API reads
- no Reddit API writes
- no server-side scraping
- browser-originated text only
- local filesystem artifacts only
- binary labels only: `askseattle` and `not_askseattle`
- one TF-IDF + logistic regression model path

If you change any of those, treat it as a major scope change and update the docs in the same change.

## Project-specific guidance

### Deployment target matters

The classifier is intended to help evaluate posts that appear in `/r/seattle`, even if some positive training data comes from `/r/askseattle`.

That means:

- never use `subreddit` as an input feature
- mixed-subreddit training is acceptable
- `/r/seattle` is the important evaluation domain when the user asks for realistic benchmark numbers
- prefer `EVAL_SUBREDDIT=seattle` when comparing or promoting model behavior

### Simplicity is a design constraint

Do not add infrastructure just because it is common elsewhere.

Prefer:

- one local bridge
- one local reviewed-label file
- one model family
- inspectable artifacts
- simple make targets

Avoid adding:

- hosted model services
- background workers
- databases
- Docker-only workflows
- server-side Reddit integrations

### The benchmark gate is precision-first

This project is moderation-adjacent. False positives are expensive.

When changing defaults, evaluate the held-out high-confidence metrics first. Do not promote a change just because it feels more semantically correct.

## Documentation rules

Documentation is part of the public surface here. Update it when behavior changes.

### Always update docs when changing:

- command names, flags, or make targets
- bridge endpoints or response fields
- userscript controls, labels, queue behavior, or hotkeys
- reviewed label schema or dedupe rules
- model artifact layout or output paths
- default model behavior or benchmark recommendations
- project scope or invariants

### Minimum doc mapping

If you change one of these areas, update these files in the same change:

- CLI or make targets:
  - `README.md`
  - `docs/reference/cli.md`
- Bridge request/response behavior:
  - `README.md` if the normal workflow changes
  - `docs/reference/bridge-api.md`
- Userscript UI or browser review flow:
  - `README.md` if visible behavior changes
  - `docs/how-to/label-posts.md`
- Training flow, thresholds, benchmark policy, or defaults:
  - `README.md`
  - `docs/how-to/retrain.md`
  - `docs/explanation/model-and-thresholds.md`
- Reviewed data format or local artifacts:
  - `docs/reference/data-format.md`
  - `docs/labeling_policy.md` if labeling expectations changed
- System boundary or major design choices:
  - `README.md`
  - `docs/architecture.md`
  - `docs/index.md`

### Documentation style rules

- keep README as a landing page, not a dumping ground
- put task instructions in `docs/how-to/*`
- put exact contracts in `docs/reference/*`
- put rationale in `docs/explanation/*`
- do not leave stale “recommended default” claims after experiments change direction

## Code change guidance

### Prefer the existing shape of the system

Most changes should land in one of these files:

- `userscripts/ask-seattle-reddit-helper.user.js`
- `src/ask_seattle/local_bridge.py`
- `src/ask_seattle/data.py`
- `src/ask_seattle/model.py`
- `src/ask_seattle/training.py`
- `src/ask_seattle/cli.py`

If a change seems to require a new subsystem, challenge that assumption first.

### Keep data flow auditable

The reviewed text used for training should still be traceable to the browser helper and the local reviewed-label file.

Do not add hidden collection paths or alternate training sources without explicit agreement and documentation.

### Do not let convenience features become product scope

Queueing, hotkeys, auto-retrain, and benchmark helpers exist to make review faster. They are not the core product.

Do not let helper features quietly reintroduce:

- moderation actions
- Reddit API dependencies
- opaque background behavior

## Verification expectations

For most code changes, run:

```bash
python3 -m ruff check src tests
PYTHONPATH=src python3 -m pytest
```

If the userscript changed, also run:

```bash
node --check userscripts/ask-seattle-reddit-helper.user.js
```

If model defaults, training policy, or lexical filtering changed, also run a benchmark that matches the actual deployment domain:

```bash
make benchmark EVAL_SUBREDDIT=seattle
```

If you are changing model defaults or recommending a new default, compare variants rather than relying on intuition:

```bash
make benchmark-variants EVAL_SUBREDDIT=seattle
```

## Three additional recommendations

### 1. Treat default-model changes as product changes

Changing stopwords, channel weights, threshold logic, or evaluation policy changes what the tool means in practice. Update docs and benchmark output together.

### 2. Prefer evidence over semantic intuition

Words that look unimportant can still carry useful ask-style structure. Conversely, words that feel obviously noisy can turn out harmless. Use the held-out benchmark and feature audit before promoting or removing defaults.

### 3. Keep `/check` as the stable foundation

Future moderation actions, dashboards, or workflows should build on top of `/check`. Do not push moderation-side effects down into the bridge unless the project goal changes explicitly.

## Things not to commit

Do not commit:

- reviewed label files under `data/processed/`
- model bundles under `models/`
- local environment files such as `.env`

Those are local artifacts, not source.
