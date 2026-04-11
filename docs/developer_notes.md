# Developer Notes

## Project Overview

Ask Seattle is a local-first classifier for Reddit submissions. The repo currently does two things:

1. collect reviewed labels from the browser helper
2. train a local model and check posts against it

Anything beyond that, including moderator actions, should be layered on top later. The bridge itself does not fetch Reddit content and does not write anything back to Reddit.

## System Map

The main subsystems are:

- Data and labeling: `ask_seattle.data` handles label normalization, reviewed-label dedupe, exact text hashing, derived time keys, and JSONL helpers. Reviewed datasets live under ignored `data/processed/`.
- Model training and checking: `ask_seattle.model` owns the TF-IDF + logistic regression pipeline, split logic, calibration, threshold selection, and plain `/check` results.
- Training orchestration: `ask_seattle.training` trains the TF-IDF model from reviewed label data, writes artifacts under `models/`, and writes `training_summary.json`.
- Browser bridge: `ask_seattle.local_bridge` serves localhost-only endpoints for the Tampermonkey helper. It checks a scraped post with `/check`, appends a reviewed example with `/train`, and can optionally auto-retrain after every N new effective training rows.
- CLI: `ask_seattle.cli` exposes `train`, `check`, and `serve-bridge`.

## Data Flow

The intended workflow is:

1. Capture reviewed labels through the Tampermonkey helper into `data/processed/tampermonkey_labels.jsonl`.
2. Train the TF-IDF model with `ask-seattle train`, which normalizes labels, dedupes by identity and exact text hash, and derives `time_key` / `time_source` before fitting.
3. Run `ask-seattle serve-bridge` or `make bridge` using the selected model artifact.
4. Continue labeling and spot-checking in the browser, then retrain or let the bridge auto-retrain when enabled.

For the normal local loop, the repo also exposes `make retrain` and `make bridge`.

There is no separate server-side Reddit collection workflow in the supported path. If a record is used for training, its text and metadata should have come from the browser capture flow.

## Browser Workflow

The Tampermonkey helper is the primary operator interface for labeling and spot-checking.

On post pages it:

- runs an automatic `/check` when the page loads;
- shows a verdict block with either `Looks like askseattle (...)` or `Does not look like askseattle`;
- keeps `Re-check` for manual rescoring;
- supports keyboard shortcuts: `P` for `Train positive` and `N` for `Train negative`;
- optionally advances to the next queued post after a training click when `Auto next after training` is enabled.

The helper panel also shows whether the current post has already been recorded, so re-labeling stays explicit. Re-labeling is last-click-wins by post id or permalink.

## Bridge Auto-Retrain

Bridge-side auto-retrain is optional and exists to shorten the label -> model loop during active review sessions.

When the bridge starts with `--retrain-every N`:

- `/train` still only stores the browser-captured reviewed record;
- the bridge then normalizes and dedupes the reviewed label file in the background;
- it counts the resulting effective training rows, not the raw reviewed rows;
- once that effective training row count has grown by `N` since the last successful reload, the bridge retrains the local TF-IDF model in the background and hot-reloads `tfidf_logreg.joblib`.

The bridge does not block the browser while retraining, and it only swaps the in-memory model after a successful run.

## How Checking Works

The classifier gets the same text inputs every time: title, body, and a combined title+body string. The TF-IDF pipeline treats those channels separately so title wording can carry more weight than body wording.

`/check` returns:

- `label`: `askseattle` or `not_askseattle`
- `score_raw` and `score_calibrated`
- `low_threshold` and `high_threshold`
- `confidence_band`: `high`, `borderline`, or `low`
- model metadata and optional post metadata

The `confidence_band` is derived from the calibrated score:

- `high`: score is at or above `high_threshold`
- `borderline`: score is between `low_threshold` and `high_threshold`
- `low`: score is below `low_threshold`

That gives downstream tooling enough structure to act later without putting moderation behavior inside the bridge today.

## Baseline Model in Plain English

The baseline model is TF-IDF plus logistic regression.

TF-IDF turns text into numeric features. It looks at title words, body words, and character fragments separately. That lets the model learn that short templates like `where should I live` or `visiting this weekend` are strong signals without forcing long selftext to dominate the feature space.

Logistic regression then learns a weighted score from those features. Phrases like `where should I stay`, `itinerary`, `moving to`, or `recommendations` can push the score toward `askseattle`, while words common in alerts, news, events, photos, or local policy discussion can push it toward `not_askseattle`.

This is a good first model because it is:

- cheap to run on every post;
- fast enough for immediate browser feedback;
- easy to retrain as labels accumulate;
- inspectable through feature weights;
- strong for categories that reuse similar wording.

Its main weakness is that it understands wording patterns more than meaning. It can miss unusual phrasings, and it can overreact to common words if the training set is small or unbalanced. The fix is more reviewed data and tighter error analysis, not more runtime complexity.

## How Training Avoids Fooling Itself

`train` uses a chronological split by default: oldest posts for training, middle posts for calibration, newest posts for the final test.

The flow is:

1. fit the model on the training slice
2. fit a sigmoid calibrator on the calibration slice
3. pick `high_threshold` by maximizing recall subject to `askseattle` precision being at least 95%
4. pick `low_threshold` from the best-F1 calibration threshold, capped so it never exceeds `high_threshold`
5. evaluate once on the newest held-out test slice

The production gate is precision-first. A run is production-ready only if its held-out high-confidence band reaches at least 95% `askseattle` precision on the newest held-out test slice.

## Runtime and Safety Invariants

The current runtime must not call Reddit APIs. It should only accept browser-originated post content, score it locally, and store reviewed labels or local artifacts under ignored directories.

Private or sensitive artifacts should remain untracked:

- `data/processed/`
- `models/`
- `.env`

The Tampermonkey helper uses `http://127.0.0.1:8765` as a local bridge. The browser script sends the visible Reddit post title/body to the bridge; the bridge must not fetch post content through Reddit APIs. Training metadata also originates in the browser payload, including the browser-side capture timestamp and optional DOM-derived metadata such as post type, content URL/domain, created time, subreddit, and crosspost hints when Reddit exposes them.

## Where to Add Things

Use `ask_seattle.data` for file formats and label normalization. Use `ask_seattle.model` for vectorization, scoring, and threshold policy. Use `ask_seattle.training` for artifact-writing training runs. Keep browser-serving behavior in `ask_seattle.local_bridge`.

If downstream moderation actions are ever added later, they should consume the `/check` response rather than being built directly into the bridge.
