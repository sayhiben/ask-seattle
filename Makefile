.PHONY: help retrain benchmark benchmark-variants bridge

ASK_SEATTLE ?= PYTHONPATH=src python3 -m ask_seattle.cli
LABELS ?= data/processed/tampermonkey_labels.jsonl
MODEL_DIR ?= models/real-labels-precision-refresh
MODEL_PATH ?= $(MODEL_DIR)/tfidf_logreg.joblib
BENCHMARK_DIR ?= models/benchmark
BENCHMARK_VARIANTS_DIR ?= models/benchmark-variants
EVAL_SUBREDDIT ?=
EVAL_SUBREDDIT_ARG := $(if $(EVAL_SUBREDDIT), --eval-subreddit $(EVAL_SUBREDDIT))
LOG_LEVEL ?= INFO
RETRAIN_EVERY ?= 0

help:
	@printf '%s\n' \
		'make retrain           Retrain the TF-IDF model from reviewed labels' \
		'make benchmark         Train/evaluate into a separate benchmark artifact directory' \
		'make benchmark-variants Compare lightweight TF-IDF variants on the same split' \
		'make bridge            Start the local Tampermonkey bridge with the current model'

retrain:
	$(ASK_SEATTLE) train \
		--data $(LABELS) \
		--output-dir $(MODEL_DIR)$(EVAL_SUBREDDIT_ARG)

benchmark:
	$(ASK_SEATTLE) train \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_DIR)$(EVAL_SUBREDDIT_ARG)

benchmark-variants:
	$(ASK_SEATTLE) benchmark-variants \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_VARIANTS_DIR)$(EVAL_SUBREDDIT_ARG)

bridge:
	$(ASK_SEATTLE) serve-bridge \
		--model $(MODEL_PATH) \
		--labels $(LABELS) \
		--log-level $(LOG_LEVEL) \
		--retrain-every $(RETRAIN_EVERY)$(EVAL_SUBREDDIT_ARG)
