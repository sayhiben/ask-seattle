# How To Troubleshoot Common Problems

Use this page when the local bridge, userscript, or retraining loop is not behaving as expected.

## The Bridge Will Not Start

### Problem

`serve-bridge` fails because the model file does not exist.

### What To Check

- the `--model` path points to a real `.joblib` file
- or the default `models/real-labels-precision-refresh/tfidf_logreg.joblib` exists

### What To Do

- retrain from an existing reviewed label file, or
- point the bridge at an existing local model artifact

## The Userscript Times Out

### Problem

The panel says it timed out waiting for the local bridge.

### What To Check

- the bridge is running
- it is listening on `127.0.0.1:8765`
- Tampermonkey can reach localhost

### What To Do

Start the bridge again:

```bash
make bridge
```

If you need request-level diagnostics:

```bash
make bridge LOG_LEVEL=DEBUG
```

## Retraining Finished But The Browser Still Uses The Old Model

### Problem

`make retrain` completed, but checks still look unchanged.

### What To Do

- restart the bridge after a manual retrain
- or start the bridge with `RETRAIN_EVERY` so it can hot-reload after background retraining

## Training Fails Because Of Time Splitting

### Problem

Training reports that there are not enough dated examples for the chronological split.

### What To Check

- reviewed records include `created_utc` or `collected_at`
- the reviewed dataset contains enough dated examples overall

### What To Do

- keep labeling real posts through the browser helper
- make sure the labels come from the current userscript, which sends `collected_at`

## Training Writes Artifacts But `production_ready` Is False

### What It Means

This is not necessarily a crash. It usually means one of these:

- the calibration slice did not contain both classes
- the held-out test high-confidence precision missed the target

Check:

- `training_summary.json`

## The Queue Does Not Advance

### Problem

`Skip` or auto-next does not move to the next post.

### What To Check

- you seeded a queue from a listing page first
- the current post belongs to that queue
- the queue still contains a later post

### What To Do

- go back to the listing
- scroll until the desired posts are visible
- click `Seed queue` again

## The Userscript Cannot Find A Title Or Body

### Problem

The panel says it could not find the current post title.

### What To Check

- you are on an actual post page, not only a listing
- the Reddit page has finished rendering

### What To Do

- refresh the page
- wait for the post DOM to settle
- use `Re-check` after the page finishes loading

If Reddit changes its DOM structure, the userscript selectors may need an update.

Next:

- [Development workflow](../development.md)
- [Bridge API reference](../reference/bridge-api.md)
- [How to label posts](label-posts.md)
