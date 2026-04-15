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

That said, the repository now also includes a five-model benchmark suite so you can compare whether encoder-transformer families improve the fuzzy edge of the decision boundary without changing the operational TF-IDF retrain path.

The comparison stack currently includes:

- TF-IDF baseline
- DeBERTa-v3-small encoder classifier
- ModernBERT-base encoder classifier
- NeoBERT encoder classifier
- ModernBERT-large encoder classifier

The encoder-transformer benchmarks use title/body pair encoding instead of one flattened text string. They use calibration PR-AUC for early stopping, restore the best epoch checkpoint, and keep the better candidate by a precision-first calibration ranking key. DeBERTa-v3-small, ModernBERT-base, NeoBERT, and ModernBERT-large all run small bounded config grids before final selection.

On CUDA runs, the neural training paths now also enable TF32 float32 matmul. That is a speed optimization for Ampere-and-newer NVIDIA GPUs; it lowers remote wall-clock cost without changing the product-level threshold policy.

## How The Model Sees A Post

Each post is represented through four TF-IDF-side channels:

- title words
- body words
- character n-grams across raw title/body text
- exact metadata tokens

That helps because:

- titles often carry the strongest intent signal
- bodies add context when present
- character features help with phrasing variants, spelling variation, and short recurring templates

When available, the shared representation also carries normalized content metadata:

- `HAS_BODY`
- `POST_TYPE`
- `CONTENT_DOMAIN`
- `CROSSPOST`
- `TITLE_LEN_BUCKET`
- `BODY_LEN_BUCKET`
- `HAS_QUESTION_MARK`
- `LOW_TEXT`
- `SPARSE_MEDIA` for link or image posts with very little text
- `IMAGE_NO_BODY` for image posts with no visible body text
- `LOW_TEXT_IMAGE` for image posts that are otherwise low-text

That is especially useful for link, image, and crosspost submissions where the title alone often underspecifies the moderation intent.

`SPARSE_MEDIA` is intentionally more conservative than the other markers. The system still reports sparse-media slice metrics at all times, but it only feeds that token into model inputs once the shared split has enough positive support to trust it.

For the operational TF-IDF model, those metadata tokens now live in a dedicated metadata feature channel rather than being mixed into the natural-language word and character channels. That keeps the feature audit cleaner and prevents the `char_wb` branch from overfitting our own synthetic marker syntax.

The lexical TF-IDF channels also normalize visible URLs to a neutral `URL` token before vectorization. That strips out brittle scaffolding such as `https`, `www`, and `://` from the word and character features while preserving cleaner structural signals through `CONTENT_DOMAIN` and `POST_TYPE`.

The word channels use a small custom stopword list for obvious filler words such as `the`, `and`, `is`, and `was`. The current default also excludes `just`, `one`, and `some`.

That remains intentionally conservative. Those three words were only promoted after the URL-normalization pass made the lexical audit cleaner and the held-out `/r/seattle` variant benchmark showed better strict-bucket and review recall with only a small review-precision tradeoff.

The character channel is still intentionally downweighted relative to the original baseline. It now only sees raw title/body text, which makes it much less likely to surface fragments from synthetic markers such as `HAS_BODY:no` or `SPARSE_MEDIA:yes`.

The default TF-IDF pipeline also raises `min_df` as the corpus grows:

- fewer than `50` posts: `min_df=1`
- `50` to `499` posts: `min_df=2`
- `500` to `1999` posts: `min_df=3`
- `2000+` posts: `min_df=5`

That change is there to suppress brittle low-support phrases once the reviewed label set is no longer tiny.

The training harness also applies conservative slice-aware positive weighting. Right now that weighting only uses `image` and `low_text` as active levers. `sparse_media` remains in the data model and benchmark output, but it is monitoring-only until the corpus has enough support to trust that slice.

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

Training chooses it by maximizing recall subject to meeting the precision target on the calibration slice and reaching a minimum number of predicted positives in that strict bucket.

The current minimum calibration support for the strict bucket is `5`. If no calibration threshold satisfies both precision and support, the training summary records a flagged fallback to the best precision-only threshold instead of pretending the stricter evidence existed.

### Low threshold

This is the lower threshold for the `borderline` band.

Training now chooses it by maximizing recall subject to a minimum review-queue precision target on the calibration slice, capped so it never exceeds the high threshold.

The TF-IDF review-threshold policy now uses a looser review precision target of `0.70`. That keeps the review queue recall-oriented without letting the threshold collapse into a pure catch-everything setting.

The broader five-model suite still reports fixed-constraint comparison metrics at stricter common bars:

- `auto_recall_at_precision_95`
- `review_recall_at_precision_75`

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
  - monitoring-only until the slice has enough positive support to be trustworthy

Each slice summary now also records:

- train positive counts
- test positive counts
- `support_status = active | observational`

Those metrics are more useful for system design than a single raw accuracy number because they tell you:

- what could be automated safely
- what should stay in a review queue
- how much traffic each path would create

The `split.coverage` block complements those metrics. It tells you whether a weak slice is weak because the model struggles there or because you barely have any positive examples in that cohort to begin with.

## Why Use A Deterministic Random Split By Default

The current reviewed dataset is usually a short rolling window. In that regime, a strict chronological split can over-weight when you happened to label something instead of surfacing a stable domain pattern.

So the default policy is:

- use one deterministic random split
- reuse that exact split across TF-IDF and transformer benchmarks
- keep `/r/seattle` as the evaluation domain when that is the deployment target

The seed makes the benchmark reproducible. The shared split makes cross-model comparisons fair.

The time-based split still exists, but it is now an explicit option for the point where the collection window is long enough that future-facing drift is the thing you actually want to measure.

## What The Benchmark Suite Is Actually Comparing

The benchmark suite keeps the evaluation contract aligned across all five models:

- one persisted `suite_input.json` manifest
- one split assignment reused by every family
- one calibration flow
- one threshold-selection policy
- one normalized operating-metrics surface

That matters because it lets you compare model families on the same deployment-relevant question instead of letting each path quietly define its own task.

The implementation details differ by family:

- TF-IDF uses sparse lexical features and a calibrated logistic-regression head
- encoder transformers use sequence classification heads

But all of them still end in the same bridge-facing concepts:

- calibrated probability
- low threshold
- high threshold
- `high`, `borderline`, and `low` confidence bands
- operating metrics for auto vs review behavior

Operationally, retraining and benchmarking are now separate steps:

- `make retrain` retrains the operational TF-IDF model plus all five suite models and writes training-only summaries
- `make benchmark` reads those trained suite artifacts later and computes held-out metrics only for the compatible models already on disk

That split is deliberate. It keeps training failures, resumability, and held-out evaluation easier to reason about than one giant command that mixes all three concerns.

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
