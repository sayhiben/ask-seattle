# Model Plan

## Phase 1: Local Baseline

Use TF-IDF word and character n-grams with logistic regression. This is cheap, fast, and works well for moderation categories that have repeated wording.

Acceptance criteria before automatic removal:

- Evaluated on real held-out subreddit data.
- High precision at the removal threshold.
- Reviewed in dry-run or shadow mode for at least several days of real traffic.
- False positives reviewed by moderators and folded back into the training set.

## Phase 2: Local Transformer

Train a local open-source sequence classifier on the same deterministic split as the TF-IDF baseline. The default base model is `distilbert/distilbert-base-uncased`.

The active model must be selected by held-out `askseattle` precision first. Choose the candidate with the highest recall among models that reach at least 95% precision. If no model reaches that gate, do not mark a production-ready model.

The container runtime defaults to CPU. Local training uses CUDA if present, Apple MPS if available, otherwise CPU.

## Phase 3: Shadow Review

Run the selected model in no-write shadow mode. The bot writes decision logs only. It must not remove, reply, report, approve, distinguish, lock, or send modmail.

Before any future Reddit write action, shadow review should confirm at least 95% precision on 100 high-confidence flagged posts or 7 days of traffic, whichever comes first.

## Deployment Shape

Use a containerized external moderator bot:

1. Stream new submissions.
2. Build the same title/body text used in training.
3. Score the submission.
4. Write a structured JSONL decision event under `reports/decisions/`.
5. Export review CSVs for moderators and fold reviewed errors back into the training data.
