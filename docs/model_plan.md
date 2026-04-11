# Model Plan

## Phase 1: Local Baseline

Use TF-IDF word and character n-grams with logistic regression. This is cheap, fast, and works well for moderation categories that have repeated wording.

Acceptance criteria before automatic removal:

- Evaluated on real held-out subreddit data.
- High precision at the removal threshold.
- Reviewed in dry-run or shadow mode for at least several days of real traffic.
- False positives reviewed by moderators and folded back into the training set.

## Phase 2: Browser Review Loop

Run the selected model behind the local bridge and review it in the browser labeling workflow. The server should not fetch Reddit content or write anything back to Reddit.

Before any future Reddit-integrated action, browser review plus manually checked held-out evaluation should confirm at least 95% precision on 100 high-confidence flagged posts or an equivalent reviewed sample.

## Deployment Shape

Use a local bridge:

1. Accept title/body text from the browser helper.
2. Score the post locally.
3. Store reviewed labels locally.
4. Prepare labels and retrain locally.
5. Feed reviewed errors back into the next training run.
