# Development Workflow

Use this page when you need to run the project locally, change code safely, or know which files to update for a given type of change.

For repository-wide guardrails, scope limits, and documentation update rules, also read [AGENTS.md](../AGENTS.md).

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you need the full five-model benchmark suite:

```bash
python -m pip install -e ".[dev,models]"
```

That extra now covers:

- encoder transformer fine-tuning

## Common Commands

Run tests:

```bash
PYTHONPATH=src python3 -m pytest
```

Run lint:

```bash
python3 -m ruff check src tests
```

Run the repo secret scan:

```bash
make secret-scan
```

Check the userscript syntax:

```bash
node --check userscripts/ask-seattle-reddit-helper.user.js
```

Start the bridge:

```bash
make bridge
```

Retrain from reviewed labels:

```bash
make retrain
```

Benchmark the trained suite:

```bash
make benchmark EVAL_SUBREDDIT=seattle
```

## Public Entry Points

These are the user-facing surfaces that should stay stable and documented:

- `ask-seattle train`
- `ask-seattle retrain-all`
- `ask-seattle check`
- `ask-seattle benchmark-suite`
- `ask-seattle serve-bridge`
- `make retrain`
- `make bridge`
- the Tampermonkey userscript panel

## Where To Change Things

If you need to change label normalization or reviewed-data prep:

- `src/ask_seattle/data.py`
- `docs/reference/data-format.md`
- `docs/labeling_policy.md` if review rules changed

If you need to change model behavior or thresholds:

- `src/ask_seattle/model.py`
- `src/ask_seattle/training.py`
- `docs/explanation/model-and-thresholds.md`
- `docs/how-to/retrain.md`

If you need to change benchmark-suite model composition or family-specific training:

- `src/ask_seattle/training.py`
- `src/ask_seattle/model.py`
- `src/ask_seattle/local_bridge.py`
- `docs/reference/cli.md`
- `docs/reference/bridge-api.md`
- `docs/reference/data-format.md`

If you need to change bridge request or response behavior:

- `src/ask_seattle/local_bridge.py`
- `docs/reference/bridge-api.md`
- `README.md` if the normal workflow changes

If you need to change the browser review flow:

- `userscripts/ask-seattle-reddit-helper.user.js`
- `docs/how-to/label-posts.md`
- `README.md` if the visible controls change

## Documentation Expectations

Update docs when any of these change:

- command names or flags
- bridge endpoints or response fields
- userscript controls or hotkeys
- reviewed label schema
- model artifact layout
- current project scope or invariants

The docs are intentionally split by type. Avoid stuffing new operational instructions into architecture pages or new rationale into reference pages.

`AGENTS.md` is the stricter source for maintainer rules. If this page and `AGENTS.md` ever disagree, fix the docs in the same change rather than letting them drift.

## Repo Hygiene

Ignored local artifacts:

- `data/processed/`
- `models/`
- `.env`

Do not commit reviewed label files or local model bundles.

To block accidental secret commits locally, install the repo-managed pre-commit hook:

```bash
make install-git-hooks
```

## Before You Finish A Change

Minimum verification for most code changes:

```bash
python3 -m ruff check src tests
PYTHONPATH=src python3 -m pytest
make secret-scan
node --check userscripts/ask-seattle-reddit-helper.user.js
```

If the userscript did not change, the `node --check` step is optional.

Next:

- [Architecture](architecture.md)
- [CLI reference](reference/cli.md)
- [How to troubleshoot](how-to/troubleshoot.md)
