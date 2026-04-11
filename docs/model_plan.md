# Roadmap

This page describes likely future directions. It is not a statement of current behavior.

For the current implementation, start with [Architecture](architecture.md) and [Model and thresholds](explanation/model-and-thresholds.md).

## Current State

The implemented system is intentionally narrow:

- browser-originated capture only
- local bridge only
- binary labels only
- TF-IDF + logistic regression only
- local training and local inference only
- no Reddit API reads or writes
- no moderation actions inside the bridge

## Near-Term Improvements

The most realistic next improvements are operational rather than architectural:

- better reviewed label coverage over time
- better held-out evaluation coverage
- cleaner false-positive and false-negative review loops
- clearer browser-side ergonomics for labeling sessions
- stricter documentation and artifact hygiene

## Deferred Work

These are explicitly out of scope in the current implementation, but remain plausible later:

- additional downstream moderation tooling on top of `/check`
- richer local review tooling
- more explicit offline evaluation dashboards
- broader label taxonomies beyond the current binary class
- model-family expansion beyond the current TF-IDF baseline

## Non-Goals For Now

- server-side Reddit collection
- server-side Reddit moderation actions
- hosted model inference
- large-model fine-tuning
- replacing the local browser capture loop

Any future work should preserve the current bias toward cheap inference, local control, and explicit operator review.
