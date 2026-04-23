.PHONY: help runpod-bootstrap runpod-cleanup runpod-prune-volumes install-git-hooks secret-scan repair-crossposts retrain benchmark benchmark-variants benchmark-seed-sweep benchmark-suite bridge

ASK_SEATTLE ?= PYTHONPATH=src python3 -m ask_seattle.cli
RUNPOD_TRAIN ?= PYTHONPATH=src python3 scripts/runpod_train.py
WSL_TRAIN ?= scripts/run_remote_training.sh
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
TRANSFORMER_MODEL_ID ?= answerdotai/ModernBERT-base
TRANSFORMER_SECONDARY_MODEL_ID ?= chandar-lab/NeoBERT
TRANSFORMER_TERTIARY_MODEL_ID ?= answerdotai/ModernBERT-large
BENCHMARK_NOTES ?=
BENCHMARK_NOTES_ARG := $(if $(BENCHMARK_NOTES), --notes '$(BENCHMARK_NOTES)')
BENCHMARK_SEEDS ?= 13,21,34
BENCHMARK_SEED_MODELS ?= transformer_modernbert_base,transformer_neobert,transformer_modernbert_large
LOG_LEVEL ?= INFO
RETRAIN_EVERY ?= 0
DECIDER_POLICY ?= hybrid_consensus
REMOTE ?= local
REMOTE_RUN_TIMEOUT ?= 21600
REMOTE_WSL_HOST ?= gpu-win
REMOTE_WSL_DISTRO ?= Ubuntu
REMOTE_WSL_DIR ?=
REMOTE_WSL_BOOTSTRAP ?= 0
REMOTE_WSL_PULL_ARTIFACTS ?= 1
REMOTE_WSL_TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128
RUNPOD_REPO ?= sayhiben/ask-seattle
RUNPOD_VOLUME_NAME ?= ask-seattle-train-$(shell gh api user -q .login 2>/dev/null || echo default)
RUNPOD_VOLUME_SIZE_GB ?= 100
RUNPOD_VOLUME_RETENTION_SECONDS ?= 259200
RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE ?= 0
RUNPOD_GPU_TYPES ?= NVIDIA RTX 6000 Ada Generation,NVIDIA L40S,NVIDIA L40,NVIDIA GeForce RTX 5090
RUNPOD_FALLBACK_GPU_TYPES ?= NVIDIA RTX A6000,NVIDIA GeForce RTX 4090,NVIDIA A40,NVIDIA RTX A5000,NVIDIA RTX A4500,NVIDIA L4,NVIDIA RTX A4000,NVIDIA RTX 4000 Ada Generation
RUNPOD_DATA_CENTER_IDS ?= EU-RO-1,US-NC-1,US-KS-2,US-IL-1,US-GA-2
RUNPOD_SSH_KEY_PATH ?= ~/.ssh/id_ed25519.pub
RUNPOD_TEMPLATE_ID ?= runpod-torch-v240
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
	--volume-retention-seconds $(RUNPOD_VOLUME_RETENTION_SECONDS) \
	--gpu-types '$(RUNPOD_GPU_TYPES)' \
	--fallback-gpu-types '$(RUNPOD_FALLBACK_GPU_TYPES)' \
	--data-center-ids '$(RUNPOD_DATA_CENTER_IDS)' \
	--template-id '$(RUNPOD_TEMPLATE_ID)' \
	--image $(RUNPOD_IMAGE) \
	--remote-dir $(RUNPOD_REMOTE_DIR) \
	--ssh-user $(RUNPOD_SSH_USER) \
	--container-disk-gb $(RUNPOD_CONTAINER_DISK_GB) \
	--volume-mount-path $(RUNPOD_VOLUME_MOUNT_PATH) \
	--labels $(LABELS) \
	--benchmark-meta-dir $(RUNPOD_META_DIR) \
	--split-strategy $(SPLIT_STRATEGY) \
	--split-seed $(SPLIT_SEED) \
	--transformer-model-id $(TRANSFORMER_MODEL_ID) \
	--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
	--transformer-tertiary-model-id $(TRANSFORMER_TERTIARY_MODEL_ID) \
	--benchmark-seeds '$(BENCHMARK_SEEDS)' \
	--benchmark-seed-models '$(BENCHMARK_SEED_MODELS)' \
	--remote-run-timeout-seconds $(REMOTE_RUN_TIMEOUT) \
	--pod-ready-timeout-seconds $(RUNPOD_READY_TIMEOUT)
ifeq ($(filter 1 true yes,$(RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE)),)
RUNPOD_EVICT_ARG :=
else
RUNPOD_EVICT_ARG := --evict-volume-on-capacity-failure
endif
RUNPOD_EVAL_ARG := $(if $(EVAL_SUBREDDIT), --eval-subreddit $(EVAL_SUBREDDIT))
RUNPOD_NOTES_ARG := $(if $(BENCHMARK_NOTES), --benchmark-notes '$(BENCHMARK_NOTES)')
WSL_EVAL_ARG := $(if $(EVAL_SUBREDDIT), --eval-subreddit $(EVAL_SUBREDDIT))
WSL_REMOTE_DIR_ARG := $(if $(REMOTE_WSL_DIR), --remote-dir '$(REMOTE_WSL_DIR)')
WSL_BOOTSTRAP_ARG := $(if $(filter 1 true yes,$(REMOTE_WSL_BOOTSTRAP)), --bootstrap)
WSL_PULL_ARG := $(if $(filter 0 false no,$(REMOTE_WSL_PULL_ARTIFACTS)), --no-pull-artifacts, --pull-artifacts)
WSL_TORCH_INDEX_ARG := $(if $(REMOTE_WSL_TORCH_INDEX_URL), --torch-index-url '$(REMOTE_WSL_TORCH_INDEX_URL)')
WSL_COMMON_ARGS := \
	--host '$(REMOTE_WSL_HOST)' \
	--wsl-distro '$(REMOTE_WSL_DISTRO)' \
	$(WSL_REMOTE_DIR_ARG) \
	--labels '$(LABELS)' \
	--split-strategy '$(SPLIT_STRATEGY)' \
	--split-seed '$(SPLIT_SEED)' \
	$(WSL_BOOTSTRAP_ARG) \
	$(WSL_PULL_ARG) \
	$(WSL_TORCH_INDEX_ARG) \
	--run-timeout-seconds '$(REMOTE_RUN_TIMEOUT)' \
	--make-arg TRANSFORMER_MODEL_ID='$(TRANSFORMER_MODEL_ID)' \
	--make-arg TRANSFORMER_SECONDARY_MODEL_ID='$(TRANSFORMER_SECONDARY_MODEL_ID)' \
	--make-arg TRANSFORMER_TERTIARY_MODEL_ID='$(TRANSFORMER_TERTIARY_MODEL_ID)' \
	--make-arg BENCHMARK_SEEDS='$(BENCHMARK_SEEDS)' \
	--make-arg BENCHMARK_SEED_MODELS='$(BENCHMARK_SEED_MODELS)'
WSL_NOTES_ARG := $(if $(BENCHMARK_NOTES), --make-arg BENCHMARK_NOTES='$(BENCHMARK_NOTES)')

help:
	@printf '%s\n' \
		'make runpod-bootstrap   Verify GitHub/RunPod prerequisites, create origin when missing, and register the local SSH key with RunPod' \
		'make runpod-cleanup     Delete the retained RunPod cache volume for the current contributor settings' \
		'make runpod-prune-volumes Delete expired retained RunPod cache volumes recorded in local metadata' \
		'make install-git-hooks  Install the repo pre-commit hook that runs the secret scan on staged files' \
		'make secret-scan        Scan tracked repo files for likely secrets before commit or push' \
		'make repair-crossposts Backfill crosspost bodies from paired originals and rewrite the local reviewed corpus' \
		'make retrain           Retrain the operational TF-IDF model and all suite models without benchmarking' \
		'make benchmark         Benchmark trained suite models only; warn and skip any untrained models' \
		'make benchmark-variants Compare lightweight TF-IDF variants on the same split' \
		'make benchmark-seed-sweep Retrain and benchmark selected suite models across multiple split seeds' \
		'make benchmark-suite   Alias for make benchmark' \
		'make bridge            Start the local Tampermonkey bridge with the current model and benchmark comparisons when available' \
		'' \
		'Useful overrides:' \
		'  REMOTE=wsl              Run retrain or benchmark targets on a remote Windows WSL box over SSH' \
		'  REMOTE=runpod           Run retrain or benchmark targets on an ephemeral RunPod Pod' \
		'  RUNPOD_TEMPLATE_ID=runpod-torch-v240  Preferred official RunPod template for non-Blackwell remote GPU runs' \
		'  RUNPOD_VOLUME_RETENTION_SECONDS=259200  Keep the successful RunPod cache volume for 3 days by default' \
		'  RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE=1  Allow the helper to relocate a retained cache volume when its region has no capacity' \
		'  RUNPOD_FALLBACK_GPU_TYPES="..."  Extra same-datacenter fallback GPUs to try before giving up on a retained cache volume' \
		'  REMOTE_RUN_TIMEOUT=21600  Max remote target runtime in seconds before it is terminated' \
		'  EVAL_SUBREDDIT=seattle  Restrict calibration/test evaluation to /r/seattle' \
		'  SPLIT_STRATEGY=random|time  Control the train/calibration/test split policy' \
		'  SPLIT_SEED=13           Deterministic seed for random splits' \
		'  DECIDER_POLICY=primary_only|hybrid_consensus  Control how the bridge decides the main /check verdict' \
		'  BENCHMARK_NOTES="..."  Attach a note to the archived benchmark history entry'

runpod-bootstrap:
	$(RUNPOD_TRAIN) bootstrap $(RUNPOD_COMMON_ARGS)

runpod-cleanup:
	$(RUNPOD_TRAIN) cleanup $(RUNPOD_COMMON_ARGS)

runpod-prune-volumes:
	$(RUNPOD_TRAIN) prune-volumes --benchmark-meta-dir $(RUNPOD_META_DIR)

install-git-hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-commit

secret-scan:
	PYTHONPATH=src python3 -m ask_seattle.secret_scan --repo-root .

repair-crossposts:
	$(ASK_SEATTLE) repair-crossposts --data $(LABELS)

ifeq ($(REMOTE),runpod)
retrain:
	$(RUNPOD_TRAIN) run --target retrain $(RUNPOD_COMMON_ARGS) $(RUNPOD_EVICT_ARG)$(RUNPOD_EVAL_ARG)

benchmark:
	$(RUNPOD_TRAIN) run --target benchmark $(RUNPOD_COMMON_ARGS) $(RUNPOD_EVICT_ARG)$(RUNPOD_EVAL_ARG)$(RUNPOD_NOTES_ARG)

benchmark-variants:
	$(RUNPOD_TRAIN) run --target benchmark-variants $(RUNPOD_COMMON_ARGS) $(RUNPOD_EVICT_ARG)$(RUNPOD_EVAL_ARG)

benchmark-seed-sweep:
	$(RUNPOD_TRAIN) run --target benchmark-seed-sweep $(RUNPOD_COMMON_ARGS) $(RUNPOD_EVICT_ARG)$(RUNPOD_EVAL_ARG)
else ifeq ($(REMOTE),wsl)
retrain:
	$(WSL_TRAIN) $(WSL_COMMON_ARGS) $(WSL_EVAL_ARG) --target retrain

benchmark:
	$(WSL_TRAIN) $(WSL_COMMON_ARGS) $(WSL_EVAL_ARG) $(WSL_NOTES_ARG) --target benchmark

benchmark-variants:
	$(WSL_TRAIN) $(WSL_COMMON_ARGS) $(WSL_EVAL_ARG) --target benchmark-variants

benchmark-seed-sweep:
	$(WSL_TRAIN) $(WSL_COMMON_ARGS) $(WSL_EVAL_ARG) --target benchmark-seed-sweep
else
retrain:
	$(ASK_SEATTLE) retrain-all \
		--data $(LABELS) \
		--operational-output-dir $(MODEL_DIR) \
		--benchmark-output-dir $(BENCHMARK_SUITE_DIR) \
		$(SPLIT_ARGS) \
		--transformer-model-id $(TRANSFORMER_MODEL_ID) \
		--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
		--transformer-tertiary-model-id $(TRANSFORMER_TERTIARY_MODEL_ID)$(EVAL_SUBREDDIT_ARG)

benchmark:
	$(ASK_SEATTLE) benchmark-suite \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_SUITE_DIR) \
		$(SPLIT_ARGS) \
		--transformer-model-id $(TRANSFORMER_MODEL_ID) \
		--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
		--transformer-tertiary-model-id $(TRANSFORMER_TERTIARY_MODEL_ID)$(EVAL_SUBREDDIT_ARG)$(BENCHMARK_NOTES_ARG)

benchmark-variants:
	$(ASK_SEATTLE) benchmark-variants \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_VARIANTS_DIR) $(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)

benchmark-seed-sweep:
	$(ASK_SEATTLE) benchmark-seed-sweep \
		--data $(LABELS) \
		--output-dir $(BENCHMARK_SUITE_DIR) \
		--benchmark-seeds '$(BENCHMARK_SEEDS)' \
		--benchmark-seed-models '$(BENCHMARK_SEED_MODELS)' \
		--split-strategy $(SPLIT_STRATEGY) \
		--transformer-model-id $(TRANSFORMER_MODEL_ID) \
		--transformer-secondary-model-id $(TRANSFORMER_SECONDARY_MODEL_ID) \
		--transformer-tertiary-model-id $(TRANSFORMER_TERTIARY_MODEL_ID)$(EVAL_SUBREDDIT_ARG)
endif

benchmark-suite:
	@$(MAKE) benchmark \
		LABELS='$(LABELS)' \
		BENCHMARK_SUITE_DIR='$(BENCHMARK_SUITE_DIR)' \
		REMOTE='$(REMOTE)' \
		RUNPOD_REPO='$(RUNPOD_REPO)' \
		RUNPOD_VOLUME_NAME='$(RUNPOD_VOLUME_NAME)' \
		RUNPOD_VOLUME_SIZE_GB='$(RUNPOD_VOLUME_SIZE_GB)' \
		RUNPOD_VOLUME_RETENTION_SECONDS='$(RUNPOD_VOLUME_RETENTION_SECONDS)' \
		RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE='$(RUNPOD_EVICT_VOLUME_ON_CAPACITY_FAILURE)' \
		RUNPOD_GPU_TYPES='$(RUNPOD_GPU_TYPES)' \
		RUNPOD_FALLBACK_GPU_TYPES='$(RUNPOD_FALLBACK_GPU_TYPES)' \
		RUNPOD_DATA_CENTER_IDS='$(RUNPOD_DATA_CENTER_IDS)' \
		RUNPOD_TEMPLATE_ID='$(RUNPOD_TEMPLATE_ID)' \
		RUNPOD_SSH_KEY_PATH='$(RUNPOD_SSH_KEY_PATH)' \
		RUNPOD_IMAGE='$(RUNPOD_IMAGE)' \
		RUNPOD_REMOTE_DIR='$(RUNPOD_REMOTE_DIR)' \
		RUNPOD_SSH_USER='$(RUNPOD_SSH_USER)' \
		RUNPOD_CONTAINER_DISK_GB='$(RUNPOD_CONTAINER_DISK_GB)' \
		RUNPOD_VOLUME_MOUNT_PATH='$(RUNPOD_VOLUME_MOUNT_PATH)' \
		RUNPOD_META_DIR='$(RUNPOD_META_DIR)' \
		RUNPOD_READY_TIMEOUT='$(RUNPOD_READY_TIMEOUT)' \
		REMOTE_WSL_HOST='$(REMOTE_WSL_HOST)' \
		REMOTE_WSL_DISTRO='$(REMOTE_WSL_DISTRO)' \
		REMOTE_WSL_DIR='$(REMOTE_WSL_DIR)' \
		REMOTE_WSL_BOOTSTRAP='$(REMOTE_WSL_BOOTSTRAP)' \
		REMOTE_WSL_PULL_ARTIFACTS='$(REMOTE_WSL_PULL_ARTIFACTS)' \
		REMOTE_WSL_TORCH_INDEX_URL='$(REMOTE_WSL_TORCH_INDEX_URL)' \
		EVAL_SUBREDDIT='$(EVAL_SUBREDDIT)' \
		SPLIT_STRATEGY='$(SPLIT_STRATEGY)' \
		SPLIT_SEED='$(SPLIT_SEED)' \
		TRANSFORMER_MODEL_ID='$(TRANSFORMER_MODEL_ID)' \
		TRANSFORMER_SECONDARY_MODEL_ID='$(TRANSFORMER_SECONDARY_MODEL_ID)' \
		TRANSFORMER_TERTIARY_MODEL_ID='$(TRANSFORMER_TERTIARY_MODEL_ID)' \
		BENCHMARK_NOTES='$(BENCHMARK_NOTES)'

bridge:
	$(ASK_SEATTLE) serve-bridge \
		--model $(MODEL_PATH) \
		--labels $(LABELS) \
		--comparison-suite $(BENCHMARK_SUITE_SUMMARY) \
		--decider-policy $(DECIDER_POLICY) \
		--log-level $(LOG_LEVEL) \
		--retrain-every $(RETRAIN_EVERY) \
		$(SPLIT_ARGS)$(EVAL_SUBREDDIT_ARG)
