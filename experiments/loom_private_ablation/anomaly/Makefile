# Cumulative optimization stages of loom-private/mlperf-tiny-anomaly-model
# (README.md).
#
#   make build   stage binaries 1-4 for this GPU, and the untimed host-check
#                reference anomaly_optimized_chunks32 (production built from
#                its unchanged source with -DLP_ANOMALY_FORCE_CHUNKS=32)
#   make test    the production host test once; the production run_tests.sh
#                on every stage with its skip-last-sample mutant; the
#                production run_optimized_tests.sh on stage 4 (with the
#                exhaustive fast-path test); then tests/run_stage_tests.sh
#
# The baseline (stage 0) and the optimized variant (stage 5) are the suite's
# own binaries, built by applications/loom_private/anomaly/Makefile; run.py
# builds them. Stage binaries go to <repo>/<gpu>/loom_private_ablation/anomaly/build
# and use the suite's compiler flags (common.mk). The test scripts take
# $(GPU_LOCK) for each GPU run themselves; do not wrap these targets in
# another flock on the same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
LP_ROOT := $(ROOT)/applications/loom_private
APP_DIR := $(LP_ROOT)/anomaly
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
CXX := g++
LP_REQUIRED_NVCCFLAGS := -std=c++17 -arch=$(ARCH) --fmad=false \
                         -Xcompiler -ffp-contract=off
CONTRACT := $(APP_DIR)/contract.json
CONTRACT_SHA256 = $(shell sha256sum $(CONTRACT) | cut -d' ' -f1)
BUILD := $(ROOT)/$(GPU)/loom_private_ablation/anomaly/build
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
GPU_TEST_TIMEOUT := timeout 3600
ASSET_CACHE = $${LOOM_PRIVATE_ASSET_CACHE:-$$HOME/.cache/cuda-elastic/loom-private-assets}

STAGE_NAMES := 1 2 3 4
VARIANT_1 := stage1-tiles
VARIANT_2 := stage2-binary32
VARIANT_3 := stage3-tensor-cores
VARIANT_4 := stage4-pipeline
HOST_SRCS := $(APP_DIR)/model.cpp $(APP_DIR)/oracle.cpp $(APP_DIR)/assets.cpp
HOST_HEADERS := $(APP_DIR)/model.h $(APP_DIR)/oracle.h $(APP_DIR)/assets.h
SRCS := $(HERE)/anomaly_stages.cu $(HOST_SRCS)
DEPS := $(SRCS) $(HOST_HEADERS) $(APP_DIR)/requant.cuh $(CONTRACT) \
        $(LP_ROOT)/common/lp_harness.cuh $(HERE)/Makefile
# The production optimized sources (applications/loom_private/anomaly/Makefile
# OPT_SRCS), for the host-check reference and the stage-4 test directory.
OPT_SRCS := $(APP_DIR)/anomaly_optimized.cu $(HOST_SRCS)
OPT_DEPS := $(OPT_SRCS) $(HOST_HEADERS) $(APP_DIR)/requant.cuh $(CONTRACT) \
            $(LP_ROOT)/common/lp_harness.cuh $(HERE)/Makefile
common_flags = $(NVCCFLAGS) $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common -I$(APP_DIR) \
               -DLP_APP=\"anomaly\" -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"
# flags <stage>: the suite's flags with this stage's variant name.
flags = $(common_flags) -DLP_VARIANT=\"$(VARIANT_$(1))\" -DLP_STAGE=$(1)

STAGES := $(foreach s,$(STAGE_NAMES),$(BUILD)/anomaly_stage$(s))
REFERENCE := $(BUILD)/anomaly_optimized_chunks32

# Test-only builds, never timed.
SKIP_MUTANTS := $(foreach s,$(STAGE_NAMES),$(BUILD)/anomaly_stage$(s)_skip_last_mutant)
CORRECTION_MUTANTS := $(foreach s,$(STAGE_NAMES),$(BUILD)/anomaly_stage$(s)_no_correction_mutant)
ARITHMETIC := $(foreach s,2 3,$(BUILD)/anomaly_stage$(s)_baseline_arithmetic)
# Stage 4's directory for the production run_optimized_tests.sh, with its
# file names: stage-4 builds with the baseline arithmetic and with forced
# chunk counts, and the production fast-path test and its mutant.
STAGE4_TESTS := $(BUILD)/stage4-tests
STAGE4_CHUNKS := 1 3 16 32
STAGE4_TEST_BINS := $(STAGE4_TESTS)/anomaly_optimized_baseline_arithmetic \
                    $(foreach n,$(STAGE4_CHUNKS),$(STAGE4_TESTS)/anomaly_optimized_chunks$(n)) \
                    $(STAGE4_TESTS)/test_requant $(STAGE4_TESTS)/test_requant_no_margin_mutant
TEST_REQUANT_SRCS := $(APP_DIR)/tests/test_requant.cu $(APP_DIR)/model.cpp $(APP_DIR)/assets.cpp
HOST_TEST := $(BUILD)/test_host

.PHONY: build test clean
build: $(STAGES) $(REFERENCE)

$(BUILD)/anomaly_stage%: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) $(SRCS) -o $@

# Host-check reference: production, unchanged source, its existing test hook.
$(REFERENCE): $(OPT_DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(common_flags) -DLP_VARIANT=\"optimized\" -DLP_ANOMALY_FORCE_CHUNKS=32 \
	  $(OPT_SRCS) -o $@

$(BUILD)/anomaly_stage%_skip_last_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_ANOMALY_MUTANT_SKIP_LAST_SAMPLE $(SRCS) -o $@

$(BUILD)/anomaly_stage%_no_correction_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_ABLATION_MUTANT_NO_CORRECTION $(SRCS) -o $@

$(BUILD)/anomaly_stage%_baseline_arithmetic: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_ANOMALY_BASELINE_ARITHMETIC $(SRCS) -o $@

$(STAGE4_TESTS)/anomaly_optimized_baseline_arithmetic: $(DEPS)
	@mkdir -p $(STAGE4_TESTS)
	$(NVCC) $(call flags,4) -DLP_ANOMALY_BASELINE_ARITHMETIC $(SRCS) -o $@

$(STAGE4_TESTS)/anomaly_optimized_chunks%: $(DEPS)
	@mkdir -p $(STAGE4_TESTS)
	$(NVCC) $(call flags,4) -DLP_ANOMALY_FORCE_CHUNKS=$* $(SRCS) -o $@

$(STAGE4_TESTS)/test_requant: $(TEST_REQUANT_SRCS) $(OPT_DEPS)
	@mkdir -p $(STAGE4_TESTS)
	$(NVCC) $(common_flags) -DLP_VARIANT=\"optimized\" $(TEST_REQUANT_SRCS) -o $@

$(STAGE4_TESTS)/test_requant_no_margin_mutant: $(TEST_REQUANT_SRCS) $(OPT_DEPS)
	@mkdir -p $(STAGE4_TESTS)
	$(NVCC) $(common_flags) -DLP_VARIANT=\"optimized\" -DLP_ANOMALY_MUTANT_NO_MARGIN \
	  $(TEST_REQUANT_SRCS) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(HOST_SRCS) $(HOST_HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra $(APP_DIR)/tests/test_host.cpp \
	  $(HOST_SRCS) -o $@

test: $(STAGES) $(SKIP_MUTANTS) $(CORRECTION_MUTANTS) $(ARITHMETIC) $(STAGE4_TEST_BINS) \
      $(HOST_TEST)
	$(HOST_TEST) $(ASSET_CACHE)
	@for s in $(STAGE_NAMES); do \
	  echo "== stage $$s: production run_tests.sh"; \
	  GPU_LOCK=$(GPU_LOCK) $(GPU_TEST_TIMEOUT) $(APP_DIR)/tests/run_tests.sh \
	    $(BUILD)/anomaly_stage$$s $(BUILD)/anomaly_stage$${s}_skip_last_mutant \
	    $(BUILD)/test-output-stage$$s || exit 1; \
	done
	@echo "== stage 4: production run_optimized_tests.sh"
	GPU_LOCK=$(GPU_LOCK) $(GPU_TEST_TIMEOUT) $(APP_DIR)/tests/run_optimized_tests.sh \
	  $(BUILD)/anomaly_stage4 $(STAGE4_TESTS) "$(STAGE4_CHUNKS)" $(ASSET_CACHE) \
	  $(BUILD)/test-output-stage4
	GPU_LOCK=$(GPU_LOCK) $(GPU_TEST_TIMEOUT) $(HERE)/tests/run_stage_tests.sh $(BUILD) \
	  $(BUILD)/test-output-stages

clean:
	rm -rf $(BUILD)
