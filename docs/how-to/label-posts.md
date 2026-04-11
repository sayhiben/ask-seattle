# How To Label Posts In The Browser

Use this page when you want to review Reddit posts with the Tampermonkey helper and save labels into the local training file.

## Prerequisites

- the bridge is running
- the userscript is installed in Tampermonkey
- the bridge has a trained `.joblib` model to load

Start the bridge:

```bash
make bridge
```

## Install The Userscript

Install:

- `userscripts/ask-seattle-reddit-helper.user.js`

The script expects the bridge at:

- `http://127.0.0.1:8765`

## Review Flow

### 1. Seed A Queue From A Listing

Open a listing page such as `/new`, scroll until the page contains the posts you want to review, then click `Seed queue`.

That stores the visible listing order in browser storage.

### 2. Open A Post

Open a post from that seeded listing.

On post pages, the helper automatically runs `/check` once after the page loads.

### 3. Read The Verdict

The verdict block shows either:

- `Looks like askseattle (...)`
- `Does not look like askseattle`

The lower status line also shows score and threshold details.

### 4. Take An Action

Available controls:

- `Skip (S)`: move to the next queued post without saving a label
- `Re-check`: run `/check` again
- `Train positive`: save the current post as `askseattle`
- `Train negative`: save the current post as `not_askseattle`
- `Auto next after training`: automatically move to the next queued post after saving a label

Hotkeys:

- `S` for `Skip`
- `P` for `Train positive`
- `N` for `Train negative`

The hotkeys are ignored while typing into inputs or editable elements.

### 5. Watch Recorded Status

The panel also shows whether the current post is already recorded.

Re-labeling is allowed. The saved dataset is last-write-wins by post identity.

## What Gets Sent To The Bridge

The userscript sends the text already visible in the browser, plus available page metadata such as:

- `id`
- `permalink`
- `title`
- `selftext`
- `collected_at`
- `created_utc` when visible in the DOM
- `subreddit`
- `post_type`
- `content_href`
- `content_domain`
- `is_crosspost`
- `capture_context`

The bridge does not fetch Reddit content separately.

## Notes

- Checking and training only run on post pages.
- Listing pages are for queue seeding only.
- If the queue count is low, scroll the listing further and seed again.

Next:

- [How to retrain](retrain.md)
- [How to troubleshoot](troubleshoot.md)
- [Bridge API reference](../reference/bridge-api.md)
