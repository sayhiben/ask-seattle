.PHONY: help retrain benchmark benchmark-variants benchmark-suite bridge

ASK_SEATTLE ?= PYTHONPATH=src python3 -m ask_seattle.cli
LABELS ?= data/processed/tampermonkey_labels.jsonl
MODEL_DIR ?= models/real-labels-precision-refresh
MODEL_PATH ?= $(MODEL_DIR)/tfidf_logreg.joblib
BENCHMARK_DIR ?= models/benchmark
BENCHMARK_VARIANTS_DIR ?= models/benchmark-variants
BENCHMARK_SUITE_DIR ?= models/benchmark-suite
BENCHMARK_SUITE_SUMMARY ?= $(BENCHMARK_SUITE_DIR)/benchmark_suite_summary.json
EVAL_SUBREDDIT ?=
EVAL_SUBREDDIT_ARG := $(if $(EVAL_SUBREDDIT), --eval-subreddit $(EVAL_SUBREDDIT))
SPLIT_STRATEGY ?= random
SPLIT_SEED ?= 13
SPLIT_ARGS := --split-strategy $(SPLIT_STRATEGY) --split-seed $(SPLIT_SEED)
SEMANTIC_MODEL_ID ?= sentence-transformers/all-MiniLM-L6-v2
TRANSFORMER_MODEL_ID ?= microsoft/deberta-v3-small
LOG_LEVEL ?= INFO
RETRAIN_EVERY ?= 0

help:
	@printf '%s\n' \
		'make retrain           Retrain the TF-IDF model from reviewed labels' \
		'make benchmark         Train/evaluate into a separate benchmark artifact directory' \
		'make benchmark-variants Compare lightweight TF-IDF variants on the same split' \
		'make benchmark-suite   Compare TF-IDF, semantic embedding, and transformer benchmarks' \
		'make bridge            Start the local Tampermonkey bridge with the current model and benchmark comparisons when available' \
		'' \
		'Useful overrides:' \
		'  EVAL_SUBREDDIT=seattle  Restrict calibration/test evaluation to /r/seattle' \
		'  SPLIT_STRATEGY=random|time  Control the train/calibration/test split policy' \
		'  SPLIT_SEED=13           Deterministic seed for random splits'

retrain:
	$(ASK_SEATTLE) train \
		--data $(LABELS) \
		--output-dir $(MODEL_DIR) $(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)

benchmark:
	$(ASK_SEATTLE) train \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_DIR) $(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)

benchmark-variants:
	$(ASK_SEATTLE) benchmark-variants \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_VARIANTS_DIR) $(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)

benchmark-suite:
	$(ASK_SEATTLE) benchmark-suite \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_SUITE_DIR) \
		$(SPLIT_ARGS) \
		--semantic-model-id $(SEMANTIC_MODEL_ID) \
		--transformer-model-id $(TRANSFORMER_MODEL_ID)$(EVAL_SUBREDDIT_ARG)

bridge:
	$(ASK_SEATTLE) serve-bridge \
		--model $(MODEL_PATH) \
		--labels $(LABELS) \
		--comparison-suite $(BENCHMARK_SUITE_SUMMARY) \
		--log-level $(LOG_LEVEL) \
		--retrain-every $(RETRAIN_EVERY) \
		$(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)
