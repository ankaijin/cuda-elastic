# Cumulative optimization stages of loom-private/pagerank (README.md).
#
#   make build   stage binaries 1-4 for this GPU, and the untimed host_equal
#                reference pagerank_optimized_tunings
#   make test    production's host tests (graph and layout), the baseline's
#                GPU tests on every stage and its Jacobi mutant, then
#                tests/run_stage_tests.sh: shapes, sweep counts, and the
#                readiness-protocol mutants of the optimized variant's tests
#                (a stale-accepting tag in stages 2-4, a skipped barrier
#                reset in every stage) and stage 4's delayed map upload with
#                its no-wait mutant
#
# The baseline (stage 0) and the optimized variant (stage 5) are the suite's
# own binaries, built by applications/loom_private/pagerank/Makefile; run.py
# builds them. Stage binaries go to <repo>/<gpu>/loom_private_ablation/pagerank/build
# and use the suite's compiler flags (common.mk). Every GPU run takes
# $(GPU_LOCK); do not wrap these targets in another flock on the same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
LP_ROOT := $(ROOT)/applications/loom_private
APP_DIR := $(LP_ROOT)/pagerank
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
CXX := g++
LP_REQUIRED_NVCCFLAGS := -std=c++17 -arch=$(ARCH) --fmad=false \
                         -Xcompiler -ffp-contract=off
CONTRACT := $(APP_DIR)/contract.json
CONTRACT_SHA256 = $(shell sha256sum $(CONTRACT) | cut -d' ' -f1)
BUILD := $(ROOT)/$(GPU)/loom_private_ablation/pagerank/build
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
LOCK := flock $(GPU_LOCK)
GPU_TEST_TIMEOUT := timeout 1800

STAGE_NAMES := 1 2 3 4
VARIANT_1 := stage1-persistent
VARIANT_2 := stage2-tags
VARIANT_3 := stage3-layout
VARIANT_4 := stage4-map-overlap
SRCS := $(HERE)/pagerank_stages.cu $(APP_DIR)/layout.cpp $(APP_DIR)/graph.cpp \
        $(APP_DIR)/oracle.cpp
DEPS := $(SRCS) $(APP_DIR)/layout.h $(APP_DIR)/graph.h $(APP_DIR)/oracle.h $(CONTRACT) \
        $(LP_ROOT)/common/lp_harness.cuh $(HERE)/Makefile
# flags <stage>: the suite's flags with this stage's variant name.
flags = $(NVCCFLAGS) $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common -I$(APP_DIR) \
        -DLP_APP=\"pagerank\" -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\" \
        -DLP_VARIANT=\"$(VARIANT_$(1))\" -DLP_STAGE=$(1)

STAGES := $(foreach s,$(STAGE_NAMES),$(BUILD)/pagerank_stage$(s))
MUTANTS := $(foreach s,$(STAGE_NAMES),$(BUILD)/pagerank_stage$(s)_jacobi_mutant)
SHAPES := $(foreach s,$(STAGE_NAMES),$(BUILD)/pagerank_stage$(s)_shapes)
# Stages 2-4 have readiness tags; every stage resets its barrier.
TAG_MUTANTS := $(foreach s,2 3 4,$(BUILD)/pagerank_stage$(s)_tag_mutant)
RESET_MUTANTS := $(foreach s,$(STAGE_NAMES),$(BUILD)/pagerank_stage$(s)_no_reset_mutant)
# Stage 4 uploads the map during the sweeps: production's delayed-map build
# and its mutant without the wait for the map.
DELAYED := $(BUILD)/pagerank_stage4_delayed_map
NO_WAIT_MUTANT := $(BUILD)/pagerank_stage4_no_map_wait_mutant
# Production's host tests, built from its sources as its Makefile builds them.
HOST_TEST := $(BUILD)/test_host
LAYOUT_TEST := $(BUILD)/test_layout
REFERENCE := $(BUILD)/pagerank_optimized_tunings
OPT_SRCS := $(APP_DIR)/pagerank_optimized.cu $(APP_DIR)/layout.cpp $(APP_DIR)/graph.cpp \
            $(APP_DIR)/oracle.cpp

.PHONY: build test clean
build: $(STAGES) $(REFERENCE)

# Untimed host_equal reference: the unchanged production source with its
# test hook (-DLP_PAGERANK_TEST_TUNING), so LP_PAGERANK_TUNING=default runs
# production with stage 4's shape.
$(REFERENCE): $(OPT_SRCS) $(APP_DIR)/layout.h $(APP_DIR)/graph.h $(APP_DIR)/oracle.h \
              $(CONTRACT) $(LP_ROOT)/common/lp_harness.cuh $(HERE)/Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common \
	  -DLP_APP=\"pagerank\" -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\" \
	  -DLP_VARIANT=\"optimized\" -DLP_PAGERANK_TEST_TUNING $(OPT_SRCS) -o $@

$(BUILD)/pagerank_stage%: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) $(SRCS) -o $@

# Test-only builds, never timed.
$(BUILD)/pagerank_stage%_jacobi_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_PAGERANK_MUTANT_JACOBI $(SRCS) -o $@

$(BUILD)/pagerank_stage%_shapes: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_ABLATION_TEST_SHAPE $(SRCS) -o $@

$(BUILD)/pagerank_stage%_tag_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_PAGERANK_MUTANT_TAG $(SRCS) -o $@

$(BUILD)/pagerank_stage%_no_reset_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_PAGERANK_MUTANT_NO_RESET $(SRCS) -o $@

$(BUILD)/pagerank_stage%_delayed_map: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_PAGERANK_TEST_DELAY_NODES $(SRCS) -o $@

$(BUILD)/pagerank_stage%_no_map_wait_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_PAGERANK_TEST_DELAY_NODES -DLP_PAGERANK_MUTANT_NO_NODES_WAIT \
	  $(SRCS) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/graph.cpp $(APP_DIR)/oracle.cpp \
              $(APP_DIR)/graph.h $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/graph.cpp $(APP_DIR)/oracle.cpp -o $@

$(LAYOUT_TEST): $(APP_DIR)/tests/test_layout.cpp $(APP_DIR)/layout.cpp $(APP_DIR)/graph.cpp \
                $(APP_DIR)/oracle.cpp $(APP_DIR)/layout.h $(APP_DIR)/graph.h $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra \
	  $(APP_DIR)/tests/test_layout.cpp $(APP_DIR)/layout.cpp $(APP_DIR)/graph.cpp \
	  $(APP_DIR)/oracle.cpp -o $@

test: $(STAGES) $(MUTANTS) $(SHAPES) $(TAG_MUTANTS) $(RESET_MUTANTS) \
      $(DELAYED) $(NO_WAIT_MUTANT) $(HOST_TEST) $(LAYOUT_TEST)
	$(HOST_TEST)
	$(LAYOUT_TEST)
	@for s in $(STAGE_NAMES); do \
	  echo "== stage $$s: baseline GPU tests"; \
	  $(LOCK) $(GPU_TEST_TIMEOUT) $(APP_DIR)/tests/run_gpu_tests.sh \
	    $(BUILD)/pagerank_stage$$s $(BUILD)/pagerank_stage$${s}_jacobi_mutant \
	    $(BUILD)/test-output-stage$$s || exit 1; \
	done
	$(LOCK) $(GPU_TEST_TIMEOUT) $(HERE)/tests/run_stage_tests.sh $(BUILD) \
	  $(BUILD)/test-output-stages

clean:
	rm -rf $(BUILD)
