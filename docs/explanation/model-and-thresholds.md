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

That said, the repository now also includes benchmark-only semantic and transformer paths so you can compare whether a denser model family improves the fuzzy edge of the decision boundary without changing the bridge runtime.

The transformer benchmark uses title/body pair encoding instead of one flattened text string. The body sequence carries the same shared metadata tokens used elsewhere, and the training loss is class-weighted so the minority positive class gets more influence during fine-tuning.

## How The Model Sees A Post

Each post is represented through three text channels:

- title words
- body words
- character n-grams across combined text

That helps because:

- titles often carry the strongest intent signal
- bodies add context when present
- character features help with phrasing variants, spelling variation, and short recurring templates

When available, the shared text also carries normalized content metadata:

- `HAS_BODY`
- `POST_TYPE`
- `CONTENT_DOMAIN`
- `CROSSPOST`
- `TITLE_LEN_BUCKET`
- `BODY_LEN_BUCKET`
- `HAS_QUESTION_MARK`
- `LOW_TEXT`
- `SPARSE_MEDIA` for link or image posts with very little text

That is especially useful for link, image, and crosspost submissions where the title alone often underspecifies the moderation intent.

The word channels use a small custom stopword list for obvious filler words such as `the`, `and`, `is`, and `was`.

That is intentionally conservative. Earlier experiments with adding `just`, `one`, and `some` as extra stopwords reduced recall on the `/r/seattle` benchmark, so they remain available as a benchmark variant rather than part of the default model.

The character channel is also intentionally downweighted relative to the original baseline. It still helps with phrasing variants and short templates, but it now has less influence when it starts overfitting generic fragments.

The training harness also applies conservative slice-aware positive weighting. If the train split contains very few positive low-text or sparse-media examples, those examples get extra weight during fitting so they are not drowned out by richer self-text positives.

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

The training summary includes a feature audit with:

- overall top positive and negative features
- top positive and negative features by channel
- the custom word stopword list used in training

That makes it easier to spot when the model is learning intent-bearing phrases versus low-value filler terms.

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

There is one extra policy layer on top of the raw thresholds: sparse media posts are treated more conservatively in the `high` bucket. If a post is an image or link post with little or no body text, it must clear a slightly higher effective high threshold to count as `high`. It can still land in `borderline` or `low` the normal way.

## Why Precision-First

For a moderation-adjacent classifier, false positives are more damaging than false negatives.

That is why the training loop optimizes for a strong high-confidence precision target instead of only chasing overall F1.

The current production gate is conservative:

- calibration must be available
- the threshold target must be achievable on the calibration slice
- the newest held-out test slice must still meet the high-confidence precision target
- the newest held-out test slice must also produce at least a minimum number of `high` predictions, so the gate is not cleared by one or two easy examples

## The Ongoing Metrics To Watch

Across model families, the most important metrics are:

- `auto_band.precision`
  - how often the strict `high` bucket is right
- `auto_band.recall`
  - how many true positives the strict bucket actually catches
- `review_queue.precision`
  - how clean the `low-or-higher` queue would be for human review
- `review_queue.recall`
  - how many true positives would at least make it into review
- `queue_rates.auto_rate`
  - how large the strict bucket is
- `queue_rates.review_rate`
  - how much moderator volume the broader queue would create
- `slice_metrics.post_type`
  - how those metrics change for self, link, image, and unknown/other posts
- `slice_metrics.low_text`
  - whether sparse-text posts behave differently from richer-text posts
- `slice_metrics.sparse_media`
  - whether link/image posts with little text should be trusted less aggressively

Those metrics are more useful for system design than a single raw accuracy number because they tell you:

- what could be automated safely
- what should stay in a review queue
- how much traffic each path would create

The `split.coverage` block complements those metrics. It tells you whether a weak slice is weak because the model struggles there or because you barely have any positive examples in that cohort to begin with.

## Why Use A Deterministic Random Split By Default

The current reviewed dataset is usually a short rolling window. In that regime, a strict chronological split can over-weight when you happened to label something instead of surfacing a stable domain pattern.

So the default policy is:

- use one deterministic random split
- reuse that exact split across TF-IDF, semantic, and transformer benchmarks
- keep `/r/seattle` as the evaluation domain when that is the deployment target

The seed makes the benchmark reproducible. The shared split makes cross-model comparisons fair.

The time-based split still exists, but it is now an explicit option for the point where the collection window is long enough that future-facing drift is the thing you actually want to measure.

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
