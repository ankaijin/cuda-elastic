# Cumulative optimization stages of loom-private/multisensor-attention
# (README.md).
#
#   make build   stage binaries 1-6 for this GPU, and the untimed host-check
#                reference (production attention_optimized.cu, unchanged,
#                built with its tuning hooks)
#   make test    the baseline's tests (run_tests.sh) and the skip-key mutant
#                on every stage, then tests/run_stage_tests.sh
#
# The baseline (stage 0) and the optimized variant (stage 7) are the suite's
# own binaries, built by applications/loom_private/attention/Makefile; run.py
# builds them. Stage binaries go to <repo>/<gpu>/loom_private_ablation/attention/build
# and use the suite's compiler flags (common.mk) and the application's extra
# flags. The test scripts take $(GPU_LOCK) around every GPU process
# themselves, so these targets must not wrap them in another flock on the
# same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
LP_ROOT := $(ROOT)/applications/loom_private
APP_DIR := $(LP_ROOT)/attention
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
LP_REQUIRED_NVCCFLAGS := -std=c++17 -arch=$(ARCH) --fmad=false \
                         -Xcompiler -ffp-contract=off
# The application's EXTRA_NVCCFLAGS (applications/loom_private/attention/Makefile).
EXTRA_NVCCFLAGS := -Xcompiler -pthread -lpthread
CONTRACT := $(APP_DIR)/contract.json
CONTRACT_SHA256 = $(shell sha256sum $(CONTRACT) | cut -d' ' -f1)
BUILD := $(ROOT)/$(GPU)/loom_private_ablation/attention/build
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock

STAGE_NAMES := 1 2 3 4 5 6
VARIANT_1 := stage1-tiles
VARIANT_2 := stage2-splits
VARIANT_3 := stage3-bound
VARIANT_4 := stage4-fma-score
VARIANT_5 := stage5-cheaper-math
VARIANT_6 := stage6-overlap
SRCS := $(HERE)/attention_stages.cu $(APP_DIR)/oracle.cpp
DEPS := $(SRCS) $(APP_DIR)/oracle.h $(CONTRACT) $(LP_ROOT)/common/lp_harness.cuh \
        $(HERE)/Makefile
common = $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common \
         -I$(APP_DIR) -DLP_APP=\"attention\" -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"
# flags <stage>: the suite's flags with this stage's variant name.
flags = $(common) -DLP_VARIANT=\"$(VARIANT_$(1))\" -DLP_STAGE=$(1)

STAGES := $(foreach s,$(STAGE_NAMES),$(BUILD)/attention_stage$(s))
HOOKS := $(foreach s,$(STAGE_NAMES),$(BUILD)/attention_stage$(s)_hooks)
MUTANTS := $(foreach s,$(STAGE_NAMES),$(BUILD)/attention_stage$(s)_skip_key_mutant)
# Untimed host-check reference: the unchanged production optimized source with
# its own tuning hooks (stages.json "host_equal").
REFERENCE := $(BUILD)/attention_hooks_reference
REFERENCE_SRCS := $(APP_DIR)/attention_optimized.cu $(APP_DIR)/oracle.cpp

.PHONY: build test clean
build: $(STAGES) $(REFERENCE)

$(BUILD)/attention_stage%: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) $(SRCS) -o $@

$(REFERENCE): $(REFERENCE_SRCS) $(APP_DIR)/oracle.h $(CONTRACT) \
              $(LP_ROOT)/common/lp_harness.cuh $(HERE)/Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(common) -DLP_VARIANT=\"optimized\" -DLP_ATTENTION_TUNING_HOOKS \
	  $(REFERENCE_SRCS) -o $@

# Test-only builds, never timed.
$(BUILD)/attention_stage%_hooks: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_ABLATION_TEST_HOOKS $(SRCS) -o $@

$(BUILD)/attention_stage%_skip_key_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_ATTENTION_MUTANT_SKIP_TILE_LAST_KEY $(SRCS) -o $@

test: $(STAGES) $(HOOKS) $(MUTANTS) $(REFERENCE)
	@for s in $(STAGE_NAMES); do \
	  echo "== stage $$s: baseline tests"; \
	  GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/run_tests.sh $(BUILD)/attention_stage$$s || exit 1; \
	done
	GPU_LOCK=$(GPU_LOCK) $(HERE)/tests/run_stage_tests.sh $(BUILD) \
	  $(BUILD)/test-output-stages

clean:
	rm -rf $(BUILD)
