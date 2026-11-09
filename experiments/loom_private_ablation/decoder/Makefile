# Cumulative optimization stages of loom-private/llama2c-decoder (README.md).
#
#   make build   stage binaries 1-4 for this GPU and the untimed host-check
#                reference decoder_optimized_hooks
#   make test    the baseline's tests on every stage, the optimized variant's
#                tests on stage 4, then tests/run_stage_tests.sh on stages 1-4
#
# The baseline (stage 0) and the optimized variant (stage 5) are the suite's
# own binaries, built by applications/loom_private/decoder/Makefile; run.py
# builds them. decoder_optimized_hooks is the unchanged production
# decoder_optimized.cu built with its test hooks (-DLP_DECODER_TEST_TUNING);
# run.py runs it with stages.json "env" as the host_equal reference and never
# times it. Stage binaries go to <repo>/<gpu>/loom_private_ablation/decoder/build
# and use the suite's compiler flags (common.mk) plus the decoder's
# -Xcompiler -pthread. Every GPU run takes $(GPU_LOCK); do not wrap these
# targets in another flock on the same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
LP_ROOT := $(ROOT)/applications/loom_private
APP_DIR := $(LP_ROOT)/decoder
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
EXTRA_NVCCFLAGS := -Xcompiler -pthread
LP_REQUIRED_NVCCFLAGS := -std=c++17 -arch=$(ARCH) --fmad=false \
                         -Xcompiler -ffp-contract=off
CONTRACT := $(APP_DIR)/contract.json
CONTRACT_SHA256 = $(shell sha256sum $(CONTRACT) | cut -d' ' -f1)
BUILD := $(ROOT)/$(GPU)/loom_private_ablation/decoder/build
BASELINE := $(ROOT)/$(GPU)/loom_private/decoder/build/decoder
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
LOCK := flock $(GPU_LOCK)
GPU_TEST_TIMEOUT := timeout 1800

STAGE_NAMES := 1 2 3 4
VARIANT_1 := stage1-regtile
VARIANT_2 := stage2-cpasync
VARIANT_3 := stage3-fragments
VARIANT_4 := stage4-streamk
SRCS := $(HERE)/decoder_stages.cu $(APP_DIR)/oracle.cpp
OPT_SRCS := $(APP_DIR)/decoder_optimized.cu $(APP_DIR)/oracle.cpp
COMMON_DEPS := $(APP_DIR)/oracle.cpp $(APP_DIR)/oracle.h $(CONTRACT) \
               $(LP_ROOT)/common/lp_harness.cuh $(HERE)/Makefile
DEPS := $(HERE)/decoder_stages.cu $(COMMON_DEPS)
# common <variant>: the suite's flags with this variant name.
common = $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common \
         -I$(APP_DIR) -DLP_APP=\"decoder\" -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\" \
         -DLP_VARIANT=\"$(1)\"
# flags <stage>: a stage's flags.
flags = $(call common,$(VARIANT_$(1))) -DLP_STAGE=$(1)

STAGES := $(foreach s,$(STAGE_NAMES),$(BUILD)/decoder_stage$(s))
REFERENCE := $(BUILD)/decoder_optimized_hooks
TUNING := $(foreach s,$(STAGE_NAMES),$(BUILD)/decoder_stage$(s)_tuning)
MUTANTS := $(foreach s,$(STAGE_NAMES),$(BUILD)/decoder_stage$(s)_mutant1 \
                                      $(BUILD)/decoder_stage$(s)_mutant2)

.PHONY: build test clean
build: $(STAGES) $(REFERENCE)

$(BUILD)/decoder_stage%: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) $(SRCS) -o $@

# Untimed host_equal reference: unchanged production source with its hooks.
$(REFERENCE): $(APP_DIR)/decoder_optimized.cu $(COMMON_DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call common,optimized) -DLP_DECODER_TEST_TUNING $(OPT_SRCS) -o $@

# Test-only builds, never timed: the production test hooks (DECODER_OPT_TILE,
# _SCHEDULE, _CTAS) and the production kernel mutants.
$(BUILD)/decoder_stage%_tuning: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_DECODER_TEST_TUNING $(SRCS) -o $@

$(BUILD)/decoder_stage%_mutant1: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_DECODER_TEST_TUNING -DDECODER_OPT_MUTANT=1 $(SRCS) -o $@

$(BUILD)/decoder_stage%_mutant2: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_DECODER_TEST_TUNING -DDECODER_OPT_MUTANT=2 $(SRCS) -o $@

$(BASELINE):
	$(MAKE) -C $(APP_DIR) build VARIANT=baseline

test: $(STAGES) $(TUNING) $(MUTANTS) $(BASELINE)
	@for s in 1 2 3; do \
	  echo "== stage $$s: the baseline's tests"; \
	  $(LOCK) $(GPU_TEST_TIMEOUT) bash $(APP_DIR)/tests/run_tests.sh \
	    $(BUILD)/decoder_stage$$s || exit 1; \
	done
	@echo "== stage 4: the optimized variant's tests (including the baseline's)"
	$(LOCK) $(GPU_TEST_TIMEOUT) bash $(APP_DIR)/tests/run_optimized_tests.sh \
	  $(BUILD)/decoder_stage4 $(BUILD)/decoder_stage4_tuning $(BASELINE) \
	  $(BUILD)/decoder_stage4_mutant1 $(BUILD)/decoder_stage4_mutant2
	@for s in $(STAGE_NAMES); do \
	  echo "== stage $$s: stage tests"; \
	  $(LOCK) $(GPU_TEST_TIMEOUT) bash $(HERE)/tests/run_stage_tests.sh $$s $(BUILD) \
	    $(BASELINE) $(BUILD)/test-output-stage$$s || exit 1; \
	done

clean:
	rm -rf $(BUILD)
