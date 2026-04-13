.PHONY: help runpod-bootstrap retrain benchmark benchmark-variants benchmark-suite bridge

ASK_SEATTLE ?= PYTHONPATH=src python3 -m ask_seattle.cli
RUNPOD_TRAIN ?= PYTHONPATH=src python3 scripts/runpod_train.py
LABELS ?= data/processed/tampermonkey_labels.jsonl
MODEL_DIR ?= models/real-labels-precision-refresh
MODEL_PATH ?= $(MODEL_DIR)/tfidf_logreg.joblib
BENCHMARK_VARIANTS_DIR ?= models/benchmark-variants
BENCHMARK_SUITE_DIR ?= models/benchmark-suite
BENCHMARK_SUITE_SUMMARY ?= $(BENCHMARK_SUITE_DIR)/benchmark_suite_summary.json
EVAL_SUBREDDIT ?=
EVAL_SUBREDDIT_ARG := $(if $(EVAL_SUBREDDIT), --eval-subreddit $(EVAL_SUBREDDIT))
SPLIT_STRATEGY ?= random
SPLIT_SEED ?= 13
SPLIT_ARGS := --split-strategy $(SPLIT_STRATEGY) --split-seed $(SPLIT_SEED)
SEMANTIC_MODEL_ID ?= sentence-transformers/all-MiniLM-L6-v2
SEMANTIC_SECONDARY_MODEL_ID ?= Qwen/Qwen3-Embedding-0.6B
TRANSFORMER_MODEL_ID ?= microsoft/deberta-v3-small
TRANSFORMER_SECONDARY_MODEL_ID ?= answerdotai/ModernBERT-base
CAUSAL_LM_MODEL_ID ?= Qwen/Qwen3-1.7B
BENCHMARK_NOTES ?=
BENCHMARK_NOTES_ARG := $(if $(BENCHMARK_NOTES), --notes '$(BENCHMARK_NOTES)')
LOG_LEVEL ?= INFO
RETRAIN_EVERY ?= 0
REMOTE ?= local
RUNPOD_REPO ?= sayhiben/ask-seattle
RUNPOD_VOLUME_NAME ?= ask-seattle-train-$(shell gh api user -q .login 2>/dev/null || echo default)
RUNPOD_VOLUME_SIZE_GB ?= 100
RUNPOD_GPU_TYPES ?= NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA A40
RUNPOD_DATA_CENTER_IDS ?= US-KS-2,US-GA-1,US-IL-1,US-CA-1
RUNPOD_SSH_KEY_PATH ?= ~/.ssh/id_ed25519.pub
RUNPOD_IMAGE ?= runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
RUNPOD_REMOTE_DIR ?= /workspace/ask-seattle
RUNPOD_SSH_USER ?= root
RUNPOD_CONTAINER_DISK_GB ?= 50
RUNPOD_VOLUME_MOUNT_PATH ?= /workspace
RUNPOD_META_DIR ?= models/runpod-meta
RUNPOD_READY_TIMEOUT ?= 1800
RUNPOD_COMMON_ARGS := \
	--repo-root . \
	--repo $(RUNPOD_REPO) \
	--ssh-key-path $(RUNPOD_SSH_KEY_PATH) \
	--volume-name $(RUNPOD_VOLUME_NAME) \
	--volume-size-gb $(RUNPOD_VOLUME_SIZE_GB) \
	--gpu-types '$(RUNPOD_GPU_TYPES)' \
	--data-center-ids '$(RUNPOD_DATA_CENTER_IDS)' \
	--image $(RUNPOD_IMAGE) \
	--remote-dir $(RUNPOD_REMOTE_DIR) \
	--ssh-user $(RUNPOD_SSH_USER) \
	--container-disk-gb $(RUNPOD_CONTAINER_DISK_GB) \
	--volume-mount-path $(RUNPOD_VOLUME_MOUNT_PATH) \
	--labels $(LABELS) \
	--benchmark-meta-dir $(RUNPOD_META_DIR) \
	--split-strategy $(SPLIT_STRATEGY) \
	--split-seed $(SPLIT_SEED) \
	--semantic-model-id $(SEMANTIC_MODEL_ID) \
	--semantic-secondary-model-id $(SEMANTIC_SECONDARY_MODEL_ID) \
	--transformer-model-id $(TRANSFORMER_MODEL_ID) \
	--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
	--causal-lm-model-id $(CAUSAL_LM_MODEL_ID) \
	--pod-ready-timeout-seconds $(RUNPOD_READY_TIMEOUT)
RUNPOD_EVAL_ARG := $(if $(EVAL_SUBREDDIT), --eval-subreddit $(EVAL_SUBREDDIT))
RUNPOD_NOTES_ARG := $(if $(BENCHMARK_NOTES), --benchmark-notes '$(BENCHMARK_NOTES)')

help:
	@printf '%s\n' \
		'make runpod-bootstrap   Verify GitHub/RunPod prerequisites, create origin when missing, and register the local SSH key with RunPod' \
		'make retrain           Retrain the operational TF-IDF model and all suite models without benchmarking' \
		'make benchmark         Benchmark trained suite models only; warn and skip any untrained models' \
		'make benchmark-variants Compare lightweight TF-IDF variants on the same split' \
		'make benchmark-suite   Alias for make benchmark' \
		'make bridge            Start the local Tampermonkey bridge with the current model and benchmark comparisons when available' \
		'' \
		'Useful overrides:' \
		'  REMOTE=runpod           Run retrain or benchmark targets on an ephemeral RunPod Pod' \
		'  EVAL_SUBREDDIT=seattle  Restrict calibration/test evaluation to /r/seattle' \
		'  SPLIT_STRATEGY=random|time  Control the train/calibration/test split policy' \
		'  SPLIT_SEED=13           Deterministic seed for random splits' \
		'  BENCHMARK_NOTES="..."  Attach a note to the archived benchmark history entry'

runpod-bootstrap:
	$(RUNPOD_TRAIN) bootstrap $(RUNPOD_COMMON_ARGS)

ifeq ($(REMOTE),runpod)
retrain:
	$(RUNPOD_TRAIN) run --target retrain $(RUNPOD_COMMON_ARGS)$(RUNPOD_EVAL_ARG)

benchmark:
	$(RUNPOD_TRAIN) run --target benchmark $(RUNPOD_COMMON_ARGS)$(RUNPOD_EVAL_ARG)$(RUNPOD_NOTES_ARG)

benchmark-variants:
	$(RUNPOD_TRAIN) run --target benchmark-variants $(RUNPOD_COMMON_ARGS)$(RUNPOD_EVAL_ARG)
else
retrain:
	$(ASK_SEATTLE) retrain-all \
		--data $(LABELS) \
		--operational-output-dir $(MODEL_DIR) \
		--benchmark-output-dir $(BENCHMARK_SUITE_DIR) \
		$(SPLIT_ARGS) \
		--semantic-model-id $(SEMANTIC_MODEL_ID) \
		--semantic-secondary-model-id $(SEMANTIC_SECONDARY_MODEL_ID) \
		--transformer-model-id $(TRANSFORMER_MODEL_ID) \
		--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
		--causal-lm-model-id $(CAUSAL_LM_MODEL_ID)$(EVAL_SUBREDDIT_ARG)

benchmark:
	$(ASK_SEATTLE) benchmark-suite \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_SUITE_DIR) \
		$(SPLIT_ARGS) \
		--semantic-model-id $(SEMANTIC_MODEL_ID) \
		--semantic-secondary-model-id $(SEMANTIC_SECONDARY_MODEL_ID) \
		--transformer-model-id $(TRANSFORMER_MODEL_ID) \
		--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
		--causal-lm-model-id $(CAUSAL_LM_MODEL_ID)$(EVAL_SUBREDDIT_ARG)$(BENCHMARK_NOTES_ARG)

benchmark-variants:
	$(ASK_SEATTLE) benchmark-variants \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_VARIANTS_DIR) $(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)
endif

benchmark-suite:
	@$(MAKE) benchmark \
		LABELS='$(LABELS)' \
		BENCHMARK_SUITE_DIR='$(BENCHMARK_SUITE_DIR)' \
		REMOTE='$(REMOTE)' \
		RUNPOD_REPO='$(RUNPOD_REPO)' \
		RUNPOD_VOLUME_NAME='$(RUNPOD_VOLUME_NAME)' \
		RUNPOD_VOLUME_SIZE_GB='$(RUNPOD_VOLUME_SIZE_GB)' \
		RUNPOD_GPU_TYPES='$(RUNPOD_GPU_TYPES)' \
		RUNPOD_DATA_CENTER_IDS='$(RUNPOD_DATA_CENTER_IDS)' \
		RUNPOD_SSH_KEY_PATH='$(RUNPOD_SSH_KEY_PATH)' \
		RUNPOD_IMAGE='$(RUNPOD_IMAGE)' \
		RUNPOD_REMOTE_DIR='$(RUNPOD_REMOTE_DIR)' \
		RUNPOD_SSH_USER='$(RUNPOD_SSH_USER)' \
		RUNPOD_CONTAINER_DISK_GB='$(RUNPOD_CONTAINER_DISK_GB)' \
		RUNPOD_VOLUME_MOUNT_PATH='$(RUNPOD_VOLUME_MOUNT_PATH)' \
		RUNPOD_META_DIR='$(RUNPOD_META_DIR)' \
		RUNPOD_READY_TIMEOUT='$(RUNPOD_READY_TIMEOUT)' \
		EVAL_SUBREDDIT='$(EVAL_SUBREDDIT)' \
		SPLIT_STRATEGY='$(SPLIT_STRATEGY)' \
		SPLIT_SEED='$(SPLIT_SEED)' \
		SEMANTIC_MODEL_ID='$(SEMANTIC_MODEL_ID)' \
		SEMANTIC_SECONDARY_MODEL_ID='$(SEMANTIC_SECONDARY_MODEL_ID)' \
		TRANSFORMER_MODEL_ID='$(TRANSFORMER_MODEL_ID)' \
		TRANSFORMER_SECONDARY_MODEL_ID='$(TRANSFORMER_SECONDARY_MODEL_ID)' \
		CAUSAL_LM_MODEL_ID='$(CAUSAL_LM_MODEL_ID)' \
		BENCHMARK_NOTES='$(BENCHMARK_NOTES)'

bridge:
	$(ASK_SEATTLE) serve-bridge \
		--model $(MODEL_PATH) \
		--labels $(LABELS) \
		--comparison-suite $(BENCHMARK_SUITE_SUMMARY) \
		--log-level $(LOG_LEVEL) \
		--retrain-every $(RETRAIN_EVERY) \
		$(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)
