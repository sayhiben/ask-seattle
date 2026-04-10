# Ask Seattle Moderator Classifier

A small ML moderation project for classifying Reddit submissions as `askseattle` or `not_askseattle`.

The first implementation is intentionally cheap: a local TF-IDF + logistic regression classifier that can run on every new submission without paying for an API call. The project keeps training, inference, and moderation decisions separated so embeddings or fine-tuning can be added later if the baseline is not strong enough.

## Target Label

`askseattle` means a low-value, repeat question or recommendation request. Examples include visitor itineraries, where to stay, where to live, moving to or from Seattle, where to find a product or service, legal advice, vacation advice, pet or animal advice, basic city information, and similar recurring advice requests.

`not_askseattle` is for posts that should not be removed by this classifier, such as local news, original discussion, event information, alerts, policy discussion, moderation announcements, and specific posts with clear community value.

See [docs/labeling_policy.md](docs/labeling_policy.md) for labeling guidance.

For a developer-oriented architecture overview, see [docs/developer_notes.md](docs/developer_notes.md).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reddit]"
```

For local transformer fine-tuning and transformer inference, install the optional transformer dependencies:

```bash
python -m pip install -e ".[dev,reddit,transformer]"
```

Train the starter model:

```bash
ask-seattle train \
  --data data/seed/askseattle_seed.jsonl \
  --model models/askseattle_tfidf.joblib \
  --threshold 0.50
```

The seed file is only a smoke test for the pipeline. Use real subreddit labels before trusting any metric or removal threshold.

There is also a larger deterministic synthetic set for exercising model training:

```bash
python scripts/generate_synthetic_seed_data.py
ask-seattle train-all \
  --data data/seed/askseattle_synthetic.jsonl \
  --output-dir models/bert-tiny-synthetic-smoke \
  --min-precision 0.95 \
  --transformer-base-model google/bert_uncased_L-2_H-128_A-2 \
  --transformer-epochs 3 \
  --transformer-batch-size 8
```

Synthetic data is useful for proving the training pipeline can learn a signal, but it is still blocked from production-ready model selection.

Run a single prediction:

```bash
ask-seattle predict \
  --model models/askseattle_tfidf.joblib \
  --title "Where should I stay for a weekend visit?" \
  --selftext "First time in Seattle and looking for hotel and food recommendations."
```

Return the moderation decision payload:

```bash
ask-seattle decide \
  --model models/askseattle_tfidf.joblib \
  --title "Where should I stay for a weekend visit?" \
  --selftext "First time in Seattle and looking for hotel and food recommendations."
```

## Data Format

Training data can be JSONL or CSV. Each row needs at least:

```json
{"id":"abc123","title":"Where should I stay?","selftext":"Visiting next month.","label":"askseattle"}
```

Accepted positive labels: `1`, `true`, `askseattle`, `ask_seattle`, `ask`.

Accepted negative labels: `0`, `false`, `not_askseattle`, `not_ask_seattle`, `not`.

Put real exports in `data/raw/`; that folder is gitignored so moderation data does not get committed accidentally.

## Labeling Workflow

Collect recent posts:

```bash
ask-seattle collect \
  --subreddit Seattle \
  --output data/raw/submissions.jsonl \
  --limit 500
```

Continue collecting new posts after the recent fetch:

```bash
ask-seattle collect \
  --subreddit Seattle \
  --output data/raw/submissions.jsonl \
  --limit 500 \
  --stream
```

Export a review CSV:

```bash
ask-seattle export-labeling \
  --raw data/raw/submissions.jsonl \
  --output data/processed/labeling.csv
```

After manual review, import only rows with a filled label:

```bash
ask-seattle import-labels \
  --labels data/processed/labeling.csv \
  --output data/processed/training.jsonl
```

Refresh locally stored raw content for posts that were deleted or removed upstream:

```bash
ask-seattle refresh-deletions \
  --raw data/raw/submissions.jsonl
```

Full post text is stored locally for training by design, but `data/raw/` and `data/processed/` are gitignored.

## Full Model Training

Train and compare the TF-IDF baseline plus the local transformer classifier:

```bash
ask-seattle train-all \
  --data data/processed/training.jsonl \
  --output-dir models/run-001 \
  --min-precision 0.95
```

The default local transformer preset is `distilbert`, backed by `distilbert/distilbert-base-uncased`. List supported presets:

```bash
ask-seattle transformer-presets
```

Train a specific preset:

```bash
ask-seattle train-all \
  --data data/processed/training.jsonl \
  --output-dir models/run-001 \
  --transformer-preset deberta-v3-small
```

Train the core benchmark set:

```bash
ask-seattle train-all \
  --data data/processed/training.jsonl \
  --output-dir models/run-001 \
  --benchmark-transformers
```

If transformer dependencies are not installed or the machine is not ready for transformer training, run the baseline-only comparison:

```bash
ask-seattle train-all \
  --data data/processed/training.jsonl \
  --output-dir models/run-001 \
  --min-precision 0.95 \
  --skip-transformer
```

The active model is selected only if it reaches at least 95% `askseattle` precision on the held-out test set. If no candidate reaches that gate, artifacts are still written for review, but `training_summary.json` will not mark a production-ready model.

## Reddit Bot

Reddit's native AutoModerator YAML cannot run an ML classifier. This repo assumes an external bot process watches submissions and takes moderator actions through Reddit's API.

Copy `.env.example` to `.env` and add credentials for a moderator bot account. The stream runtime is no-write shadow mode only:

```bash
ASK_SEATTLE_NO_WRITE=1 ask-seattle stream
```

The bot writes JSONL decision logs under `reports/decisions/YYYY-MM-DD.jsonl`. This version does not remove, reply, report, approve, distinguish, lock, or send modmail. Any Reddit write behavior should be added only after a separate plan change.

Export shadow-mode decisions for moderator review:

```bash
ask-seattle export-review \
  --decisions reports/decisions/2026-04-09.jsonl \
  --output reports/review/2026-04-09.csv
```

Run with Docker:

```bash
docker compose up --build
```

The container mounts `data/raw/`, `data/processed/`, `models/`, and `reports/` as local volumes. Those paths are ignored by git.

## Model Plan

1. Start with the cheap local classifier in this repo.
2. Label real subreddit examples and tune for high precision before automatic removals.
3. If precision/recall is not good enough, add embeddings as features or a retrieval step for similar historical examples.
4. Consider fine-tuning only after the label policy is stable and the local baseline has been measured.

For removals, optimize for precision first. A false positive removes a valid community post, so automatic removal thresholds should be conservative.
