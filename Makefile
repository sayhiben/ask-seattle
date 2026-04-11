.PHONY: help retrain bridge

ASK_SEATTLE ?= PYTHONPATH=src python3 -m ask_seattle.cli
LABELS ?= data/processed/tampermonkey_labels.jsonl
MODEL_DIR ?= models/real-labels-precision-refresh
MODEL_PATH ?= $(MODEL_DIR)/tfidf_logreg.joblib
LOG_LEVEL ?= INFO
RETRAIN_EVERY ?= 0

help:
	@printf '%s\n' \
		'make retrain           Retrain the TF-IDF model from reviewed labels' \
		'make bridge            Start the local Tampermonkey bridge with the current model'

retrain:
	$(ASK_SEATTLE) train \
		--data $(LABELS) \
		--output-dir $(MODEL_DIR)

bridge:
	$(ASK_SEATTLE) serve-bridge \
		--model $(MODEL_PATH) \
		--labels $(LABELS) \
		--log-level $(LOG_LEVEL) \
		--retrain-every $(RETRAIN_EVERY)
