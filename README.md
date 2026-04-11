# Ask Seattle Classifier

A small local project for classifying Reddit submissions as `askseattle` or `not_askseattle`.

The current implementation is intentionally cheap and simple: one local TF-IDF + logistic regression classifier with calibrated thresholds. The server only talks to the browser helper and local files.

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
python -m pip install -e ".[dev]"
```

Start the local bridge:

```bash
make bridge
```

Then label real posts in the browser helper and retrain:

```bash
make retrain
```

Run a single check:

```bash
ask-seattle check \
  --model models/real-labels-precision-refresh/tfidf_logreg.joblib \
  --title "Where should I stay for a weekend visit?" \
  --selftext "First time in Seattle and looking for hotel and food recommendations."
```

## Data Format

Reviewed label data is JSONL from the Tampermonkey helper. Each row needs at least:

```json
{"id":"abc123","title":"Where should I stay?","selftext":"Visiting next month.","label":"askseattle"}
```

Accepted positive labels: `1`, `true`, `askseattle`, `ask_seattle`, `ask`.

Accepted negative labels: `0`, `false`, `not_askseattle`, `not_ask_seattle`, `not`.

Keep reviewed captures under `data/processed/`; that directory is gitignored so captured label data does not get committed accidentally.

## Labeling Workflow

The training workflow is Tampermonkey-only:

1. Label posts in Reddit with the Tampermonkey helper, which writes reviewed JSONL under `data/processed/`.
2. Retrain from that reviewed file.
3. The train command normalizes labels, dedupes by identity and text hash, derives time keys, and fits the model.
4. Run the local bridge with the selected model artifact and continue labeling or spot-checking in the browser.

There is no separate server-side collection or label-export path for training data. The reviewed post text that goes into training must originate in the browser helper.

Full post text is stored locally for training by design, and `data/processed/` is gitignored.

## Model Training

Train the calibrated TF-IDF model:

```bash
ask-seattle train \
  --data data/processed/tampermonkey_labels.jsonl \
  --output-dir models/run-001
```

`train` uses one built-in policy: chronological train/calibration/test splitting with a 95% high-confidence precision gate. If that gate is not met, artifacts are still written, but `training_summary.json` will show that the model is not production-ready.

## Make Targets

For the normal retraining loop, use the `Makefile`:

```bash
make retrain
```

That retrains the TF-IDF model from `data/processed/tampermonkey_labels.jsonl` into `models/real-labels-precision-refresh/`.

The targets are configurable through variables:

```bash
make retrain MODEL_DIR=models/run-002
make bridge MODEL_PATH=models/run-002/tfidf_logreg.joblib LOG_LEVEL=DEBUG
make bridge RETRAIN_EVERY=25
```

## Local Bridge Runtime

The runtime is bridge-only. The server does not fetch Reddit posts, stream Reddit traffic, or call Reddit moderator APIs. All text used for checking or training comes from the browser helper.

Run the bridge locally:

```bash
make bridge
```

## Tampermonkey Helper

The userscript at `userscripts/ask-seattle-reddit-helper.user.js` adds a small panel to Reddit listing and post pages with local helper buttons:

- `Seed queue`: records the currently visible listing order in browser storage.
- auto-check on post load: sends the visible post title/body to the local model and shows whether the post looks like `askseattle`.
- `Re-check`: runs the same check again on demand.
- `Train positive`: sends the visible post title/body to the local bridge and appends it as `askseattle`. Hotkey: `P`.
- `Train negative`: sends the visible post title/body to the local bridge and appends it as `not_askseattle`. Hotkey: `N`.
- `Auto next after training`: when enabled, moves to the next post from the visible listing queue after a train click.

Start the local bridge first:

```bash
ask-seattle serve-bridge \
  --model models/real-labels-precision-refresh/tfidf_logreg.joblib \
  --labels data/processed/tampermonkey_labels.jsonl
```

Then install `userscripts/ask-seattle-reddit-helper.user.js` in Tampermonkey and open a Reddit post page. The userscript calls `http://127.0.0.1:8765`; it does not call Reddit moderator write APIs.

The userscript scrapes the title/body already visible in your browser and sends that text to the local bridge; the bridge does not fetch Reddit content independently. It also sends the browser-side capture timestamp plus optional debugging metadata when visible in the page DOM, such as subreddit, post type, outbound content URL/domain, created time, and a crosspost hint. The train buttons collect reviewed labels for later retraining.

If you want the bridge to retrain itself after every N net-new effective training rows and hot-reload the TF-IDF model, start it with:

```bash
ask-seattle serve-bridge \
  --model models/real-labels-precision-refresh/tfidf_logreg.joblib \
  --labels data/processed/tampermonkey_labels.jsonl \
  --retrain-every 25
```

That path still runs only on browser-captured label data. It normalizes and dedupes labels in the background, retrains the TF-IDF model, and swaps the in-memory bundle only after a successful run.

To use auto-next, open a subreddit listing such as `/new` first so the userscript can record the visible post order. The panel appears on listing pages; click `Seed queue` after scrolling if the queue count is too low, then open a post from that listing. Auto-check and the training buttons only run on post/comment pages so a listing page is not accidentally labeled. Auto-next uses that browser-side queue; it does not call the Reddit API. The bridge also checks whether the current post is already recorded and displays the saved label. Re-labeling a post updates the local label file with last-click-wins behavior instead of creating duplicates. All training fields now originate in the browser payload; the bridge only stores and normalizes them.

Use `--log-level DEBUG` for request-level bridge diagnostics. Relative `models/...` and `data/...` paths are resolved from the current directory first, then from the project root, so the bridge still works if you start it from a subdirectory such as `scripts/`.

## Model Plan

1. Keep the current TF-IDF classifier small and inspectable.
2. Label real subreddit examples and tune for high-confidence precision before any downstream automation.
3. Use the browser loop and auto-retrain to tighten the model over time.

This repo currently stops at training and checking. Any later moderation actions should sit on top of the `/check` response rather than inside the bridge.
