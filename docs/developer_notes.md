# Developer Notes

## Project Overview

Ask Seattle is a local-first moderation classifier for Reddit submissions. Its current goal is binary classification: decide whether a submission is `askseattle` content, meaning a recurring low-value advice, planning, or recommendation request, or `not_askseattle`, meaning it should not be flagged by this classifier.

The project is built around a conservative moderation posture. It can collect data, prepare labeling files, train models, score posts, and run a shadow-mode stream, but the runtime intentionally does not write to Reddit. It logs decisions locally so moderators can review precision before any future plan adds reports, removals, replies, or other moderator actions.

## System Map

The main subsystems are:

- Data and labeling: `ask_seattle.data` handles label normalization, JSONL/CSV loading, raw post records, labeling CSV export, reviewed label import, and JSONL helpers. Raw and processed real-world datasets live under ignored `data/raw/` and `data/processed/`.
- Reddit collection: `ask_seattle.reddit_data` creates the PRAW client, collects recent and streamed submissions, stores full post text without author fields, and refreshes local records for deleted or removed upstream content.
- Model training: `ask_seattle.model` owns the TF-IDF + logistic regression baseline, deterministic train/validation/test splitting, threshold sweeps, and 95% precision-gated model selection.
- Transformer training: `ask_seattle.transformer_model` provides the optional Hugging Face sequence-classifier path, defaulting to `distilbert/distilbert-base-uncased`. It selects CUDA, Apple MPS, or CPU at runtime, while the container defaults to CPU.
- Full training orchestration: `ask_seattle.training` trains and compares the baseline and optional transformer model on the same split, writes artifacts under `models/`, and writes `training_summary.json`.
- Moderation decisions: `ask_seattle.moderation` converts a model score into a stable decision payload with model metadata, threshold, score, predicted label, `should_flag`, and timestamp.
- Shadow logging: `ask_seattle.decision_log` writes decision JSONL files under `reports/decisions/YYYY-MM-DD.jsonl` and exports review CSVs for moderator adjudication.
- Runtime stream: `ask_seattle.reddit_stream` watches new submissions, scores them, and logs decisions. It is no-write by construction and exits unless `ASK_SEATTLE_NO_WRITE=1`.
- CLI: `ask_seattle.cli` exposes the system as the `ask-seattle` command.

## Data Flow

The intended workflow is:

1. Collect subreddit submissions with `ask-seattle collect` into `data/raw/submissions.jsonl`.
2. Export a manual review CSV with `ask-seattle export-labeling`.
3. Fill labels by hand using the labeling policy, leaving uncertain rows blank or adding notes.
4. Import reviewed labels with `ask-seattle import-labels` into `data/processed/training.jsonl`.
5. Train and compare models with `ask-seattle train-all`.
6. Run `ask-seattle stream` in no-write mode using the selected model artifact.
7. Export decision logs with `ask-seattle export-review`, review mistakes, and fold corrected labels back into training data.

The seed dataset under `data/seed/` is only a smoke test. `train-all` blocks seed data from being marked production-ready even if it hits the precision threshold.

## How the Classifier Works

Every classifier in this project gets the same input: a combined text string made from the Reddit submission title and body. The model does not look at the author, votes, comments, flair history, or moderator actions at inference time. That keeps the first version focused on the content of the post and avoids building policy around user identity or popularity signals.

The classifier returns a probability-like score for the positive class, `askseattle`. A score near `1.0` means the model thinks the post looks like recurring advice or recommendation content. A score near `0.0` means it thinks the post looks like something else. The moderation layer compares that score to a threshold and emits a decision payload:

- `predicted_label`: `askseattle` or `not_askseattle`.
- `score`: the model's positive-class score.
- `threshold`: the cutoff used for this decision.
- `should_flag`: whether the score met or exceeded the threshold.
- model metadata and post metadata for audit logs.

The threshold is a policy control, not just a model detail. A lower threshold catches more `askseattle` posts but creates more false positives. A higher threshold misses more low-value posts but is safer for legitimate submissions. Since false positives are costly in moderation, this project uses a high precision gate before a model can be considered production-ready.

## Baseline Model in Plain English

The baseline model is TF-IDF plus logistic regression.

TF-IDF turns text into numeric features. It looks for words and short word pairs such as `where stay`, plus character fragments that help with spelling variants and short phrases. Common terms that appear everywhere matter less, while terms that are distinctive in the training set matter more.

Logistic regression then learns a weighted score from those features. For example, phrases like `where should I live`, `visiting`, `itinerary`, `hotel`, or `recommendations` may push the score toward `askseattle`, while terms common in alerts, news, events, lost-and-found posts, or local policy discussions may push it toward `not_askseattle`.

This is a good first model because it is:

- cheap to run on every post;
- fast enough for a bot loop without special hardware;
- easy to retrain as the labeling policy changes;
- less opaque than a large language model;
- strong for moderation categories that reuse similar phrases.

Its main weakness is that it understands wording patterns more than meaning. It can miss unusual phrasings, and it can overreact to common words if the training set is small or unbalanced. That is why the project logs shadow decisions and expects moderators to fold mistakes back into the labeled data.

## Transformer Model in Plain English

The optional transformer path fine-tunes a small local text classifier, defaulting to `distilbert/distilbert-base-uncased`. A transformer reads the text with more context than TF-IDF. It can learn that two differently worded posts are asking for the same kind of advice even when they do not share many exact words.

This can help with borderline cases such as:

- posts that ask for recommendations indirectly;
- long posts where the core request is buried in context;
- wording that does not match the common phrases in the TF-IDF feature set.

The tradeoff is operational cost. Transformer training and inference are heavier, require extra dependencies, and may download model weights. That is why transformer support is optional and lives behind the `transformer` extra. The container defaults to CPU for portability, while local training can use CUDA or Apple MPS when available.

The transformer is not automatically better. `train-all` evaluates it on the same held-out data as the TF-IDF baseline and only selects it if it clears the same 95% precision gate and has better recall among eligible models.

The transformer path supports named presets so engineers do not need to memorize model IDs. The default preset is `distilbert`. The core benchmark set is `distilbert`, `deberta-v3-small`, `roberta-base`, and `electra-small`; these cover the main quality/cost tradeoff before trying more specialized models.

Social-text presets, `bertweet-base` and `twitter-roberta-base`, are available for experiments once real Reddit labels exist. They may help with informal phrasing, but they come from Twitter-like domains, so they should not be assumed better without evaluation.

Long-context presets, `modernbert-base` and `bigbird-roberta-base`, are available but should be used later. They are meant for the specific failure mode where truncating long selftexts hides the actual request. They are heavier and should not be part of the default local runtime until decision logs show that long-post truncation is a real problem.

## How Training Avoids Fooling Itself

The training pipeline separates labeled examples into train, validation, and test splits.

The training split is the data the model learns from. The validation split is reserved for future tuning and comparison work. The test split is held out so the project can estimate how the model behaves on examples it did not train on.

The important rule is that the production gate is measured on held-out examples. If a model is evaluated only on examples it already learned from, the metrics can look good while the model is not actually reliable.

`train-all` compares candidates on the same deterministic split so that the baseline and transformer are tested against the same posts. It then sweeps possible thresholds and applies this selection rule:

1. Keep only thresholds where `askseattle` precision is at least 95%.
2. Among those, choose the highest recall.
3. Use precision and F1 as tie-breakers.
4. If no threshold reaches 95% precision, do not mark any model production-ready.

Precision answers: of the posts the model flagged, how many were truly `askseattle`? Recall answers: of all true `askseattle` posts, how many did the model catch? This project prioritizes precision first because a false positive means a valid community post gets flagged.

## Model and Threshold Policy

The production gate is precision-first. A candidate model is production-ready only if its held-out `askseattle` precision is at least 95%. Among models that meet the gate, the active model is the one with the highest recall, then precision, then F1.

The baseline model is deliberately simple and cheap. It uses word and character TF-IDF features with logistic regression. The transformer path is optional and local; it is intended for comparison once enough real labels exist, not as a dependency for the whole project.

Transformer dependencies are in the `transformer` extra. The baseline path and most tests should remain usable without downloading transformer models.

## Why These Stack Choices

The stack is designed to keep the first useful system inexpensive, inspectable, and easy to operate.

Scikit-learn is a good fit for the baseline because it is mature, fast, and simple. TF-IDF plus logistic regression gives useful moderation performance without GPU work, hosted inference, vector databases, or a long-running ML service.

Hugging Face Transformers and PyTorch are used for the optional local fine-tuning path because they are the standard ecosystem for small open-source text classifiers. Keeping this path optional lets the project compare a stronger model without making every install or CI run depend on large model downloads.

Named transformer presets keep experiments reproducible. A training summary records the preset name, base model, tier, artifact path, threshold, and metrics, which makes it easier to compare model families without guessing which checkpoint produced a result.

PRAW is used for Reddit access because it is the conventional Python wrapper for Reddit's API and matches the bot-style workflow: collect submissions, stream new submissions, and read submission metadata.

JSONL and CSV are used deliberately. JSONL is easy to append, diff locally when needed, and process line by line. CSV keeps manual labeling accessible to moderators and non-ML contributors. More complex storage can be added later if the review workflow outgrows flat files.

Docker Compose is used for runtime packaging, not for model research. It gives the bot a repeatable environment and mounted local volumes for raw data, model artifacts, and reports while still keeping the actual development loop simple.

## Runtime and Safety Invariants

The current stream runtime must not call Reddit write APIs. That includes remove, reply, report, approve, distinguish, lock, and modmail. It should only read submissions and write local JSONL logs.

Private or moderation-sensitive artifacts should remain untracked:

- `data/raw/`
- `data/processed/`
- `models/`
- `reports/`
- `.env`

The Docker runtime mounts those directories as volumes and defaults to `ASK_SEATTLE_NO_WRITE=1` and `ASK_SEATTLE_TORCH_DEVICE=cpu`.

## Where to Add Things

Use `ask_seattle.data` for file formats and label normalization. Use `ask_seattle.model` for model-agnostic evaluation and threshold policy. Use `ask_seattle.transformer_model` for Hugging Face-specific training or inference details. Keep Reddit API access in `ask_seattle.reddit_data` and streaming behavior in `ask_seattle.reddit_stream`.

When adding any Reddit write capability in the future, treat it as a separate feature with explicit tests proving default no-write behavior remains intact.
