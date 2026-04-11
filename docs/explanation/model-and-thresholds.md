# Model And Thresholds

Use this page when you want the model behavior explained in plain language rather than only as code or raw metrics.

## Why This Model

The current model is TF-IDF plus logistic regression.

That choice is deliberate:

- cheap to train
- cheap to run
- easy to inspect
- fast enough for immediate browser feedback
- good at repeated wording patterns, which is common in moderation-style text buckets

This project does not currently need a larger or more complex stack to prove the workflow.

## How The Model Sees A Post

Each post is represented through three text channels:

- title words
- body words
- character n-grams across combined text

That helps because:

- titles often carry the strongest intent signal
- bodies add context when present
- character features help with phrasing variants, spelling variation, and short recurring templates

## What TF-IDF Means Here

TF-IDF is a way to turn text into weighted numeric features.

In plain language:

- words and fragments that appear in a post become features
- terms that are common everywhere get less weight
- terms that are distinctive for one class get more weight

For this project, phrases like `where should I stay`, `moving to`, `recommendations`, or `visiting` can become strong positive signals. Words associated with news, alerts, or civic discussion can push the score the other direction.

## What Logistic Regression Does

Logistic regression takes those weighted text features and learns how much each one should push the final probability up or down.

That gives a model that is:

- linear and inspectable
- stable to retrain
- well-suited to sparse text features

It does not deeply understand meaning. It mostly learns wording patterns and their correlations with the label.

## Raw Score vs Calibrated Score

The model first produces a raw positive probability.

Then the training loop fits a sigmoid calibrator on a separate calibration slice.

Why:

- raw classifier probabilities are not always trustworthy as action thresholds
- calibration makes the score bands more interpretable
- the bridge and userscript should reason from the calibrated score, not just the raw margin

`/check` returns both values:

- `score_raw`
- `score_calibrated`

The main `score` field is the calibrated value.

## Low And High Thresholds

The system uses two thresholds.

### High threshold

This is the conservative threshold for the `high` confidence band.

Training chooses it by maximizing recall subject to meeting the precision target on the calibration slice.

### Low threshold

This is the lower threshold for the `borderline` band.

Training chooses it from the best-F1 calibration threshold, capped so it never exceeds the high threshold.

## Confidence Bands

The bridge maps the calibrated score into three bands:

- `high`
  - score is at or above `high_threshold`
- `borderline`
  - score is at or above `low_threshold` but below `high_threshold`
- `low`
  - score is below `low_threshold`

The predicted label is binary:

- `askseattle` when score is at or above `low_threshold`
- `not_askseattle` otherwise

That means the bridge can expose more structure than a single yes/no answer without embedding moderation actions into the bridge itself.

## Why Precision-First

For a moderation-adjacent classifier, false positives are more damaging than false negatives.

That is why the training loop optimizes for a strong high-confidence precision target instead of only chasing overall F1.

The current production gate is conservative:

- calibration must be available
- the threshold target must be achievable on the calibration slice
- the newest held-out test slice must still meet the high-confidence precision target

## Why Use A Chronological Split

The model is trained on older examples and evaluated on newer ones.

That matters because moderation language drifts over time. A random split can look better than the future-facing use case actually is.

The chronological split keeps the evaluation closer to the real review loop:

- train on older posts
- calibrate on newer posts
- test on the newest held-out posts

## What To Improve Before Adding Complexity

If the model underperforms, the highest-leverage fixes are usually:

- more reviewed labels
- better time coverage
- better negative coverage for near-miss posts
- tighter error analysis

Not:

- a more complex runtime
- server-side Reddit integration
- larger models by default

Next:

- [Architecture](../architecture.md)
- [How to retrain](../how-to/retrain.md)
- [Reviewed data and artifacts reference](../reference/data-format.md)
