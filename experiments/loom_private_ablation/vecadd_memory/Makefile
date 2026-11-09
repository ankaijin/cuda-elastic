# Cumulative optimization stages of loom-private/vecadd-memory (README.md).
#
#   make build   stage binaries 1-4 for this GPU, and the host-check reference
#   make test    the application's host test once; the baseline's GPU tests on
#                every stage and its drop-last mutant; then
#                tests/run_stage_tests.sh on every stage (the optimized
#                variant's chunking, mutant, and memcheck tests)
#
# The baseline (stage 0) and the optimized variant (stage 5) are the suite's
# own binaries, built by applications/loom_private/vecadd_memory/Makefile;
# run.py builds them. Stage binaries go to
# <repo>/<gpu>/loom_private_ablation/vecadd_memory/build and use the suite's
# compiler flags (common.mk). Every GPU run takes $(GPU_LOCK); do not wrap
# these targets in another flock on the same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
LP_ROOT := $(ROOT)/applications/loom_private
APP_DIR := $(LP_ROOT)/vecadd_memory
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
CXX := g++
LP_REQUIRED_NVCCFLAGS := -std=c++17 -arch=$(ARCH) --fmad=false \
                         -Xcompiler -ffp-contract=off
CONTRACT := $(APP_DIR)/contract.json
CONTRACT_SHA256 = $(shell sha256sum $(CONTRACT) | cut -d' ' -f1)
BUILD := $(ROOT)/$(GPU)/loom_private_ablation/vecadd_memory/build
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
LOCK := flock $(GPU_LOCK)
GPU_TEST_TIMEOUT := timeout 1200

STAGE_NAMES := 1 2 3 4
VARIANT_1 := stage1-pipeline
VARIANT_2 := stage2-copy2d
VARIANT_3 := stage3-vector
VARIANT_4 := stage4-streaming
SRCS := $(HERE)/vecadd_memory_stages.cu $(APP_DIR)/oracle.cpp
DEPS := $(SRCS) $(APP_DIR)/oracle.h $(CONTRACT) $(LP_ROOT)/common/lp_harness.cuh \
        $(HERE)/Makefile
# common_flags <variant>: the suite's flags (common.mk's LP_FLAGS) with this
# variant name. vecadd_memory sets no EXTRA_NVCCFLAGS.
common_flags = $(NVCCFLAGS) $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common \
               -DLP_APP=\"vecadd_memory\" -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\" \
               -DLP_VARIANT=\"$(1)\"
flags = $(call common_flags,$(VARIANT_$(1))) -I$(APP_DIR) -DLP_STAGE=$(1)

STAGES := $(foreach s,$(STAGE_NAMES),$(BUILD)/vecadd_memory_stage$(s))
# host_equal reference (never timed): production vecadd_memory_optimized.cu,
# unchanged, with its existing test hook, as its own Makefile builds
# vecadd_memory_tuning_optimized. run.py runs it with LP_VECADD_CHUNKS=8
# LP_VECADD_KERNEL=0, the setting of stage 4.
REFERENCE := $(BUILD)/vecadd_memory_reference_tuning
OPT_SRCS := $(APP_DIR)/vecadd_memory_optimized.cu $(APP_DIR)/oracle.cpp
# Test-only builds, never timed: a tuning build (LP_VECADD_CHUNKS sets the
# chunk count) and the optimized variant's four mutants, all with
# -DLP_VECADD_TEST_TUNING as in the optimized variant's Makefile.
MUTANT_NAMES := drop_last skip_upload skip_download no_join
TEST_BINS := $(foreach s,$(STAGE_NAMES),$(BUILD)/vecadd_memory_stage$(s)_tuning \
               $(foreach m,$(MUTANT_NAMES),$(BUILD)/vecadd_memory_stage$(s)_$(m)_mutant))
HOST_TEST := $(BUILD)/test_host

.PHONY: build test clean
build: $(STAGES) $(REFERENCE)

$(BUILD)/vecadd_memory_stage%: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) $(SRCS) -o $@

$(REFERENCE): $(OPT_SRCS) $(APP_DIR)/oracle.h $(CONTRACT) $(LP_ROOT)/common/lp_harness.cuh \
              $(HERE)/Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(call common_flags,optimized) -DLP_VECADD_TEST_TUNING $(OPT_SRCS) -o $@

$(BUILD)/vecadd_memory_stage%_tuning: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$*) -DLP_VECADD_TEST_TUNING $(SRCS) -o $@

# The mutant's name, upper-cased, is the macro suffix: skip_upload is
# LP_VECADD_MUTANT_SKIP_UPLOAD.
define mutant_rule
$(BUILD)/vecadd_memory_stage$(1)_$(2)_mutant: $(DEPS)
	@mkdir -p $(BUILD)
	$(NVCC) $(call flags,$(1)) -DLP_VECADD_TEST_TUNING \
	  -DLP_VECADD_MUTANT_$(shell echo $(2) | tr a-z A-Z) $(SRCS) -o $$@
endef
$(foreach s,$(STAGE_NAMES),$(foreach m,$(MUTANT_NAMES),$(eval $(call mutant_rule,$(s),$(m)))))

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(STAGES) $(TEST_BINS) $(HOST_TEST)
	$(HOST_TEST)
	@for s in $(STAGE_NAMES); do \
	  echo "== stage $$s: baseline GPU tests"; \
	  $(LOCK) $(GPU_TEST_TIMEOUT) $(APP_DIR)/tests/run_gpu_tests.sh \
	    $(BUILD)/vecadd_memory_stage$$s $(BUILD)/vecadd_memory_stage$${s}_drop_last_mutant \
	    $(BUILD)/test-output-stage$$s || exit 1; \
	done
	@for s in $(STAGE_NAMES); do \
	  echo "== stage $$s: stage tests"; \
	  $(LOCK) $(GPU_TEST_TIMEOUT) $(HERE)/tests/run_stage_tests.sh $(BUILD) $$s \
	    $(BUILD)/test-output-stages || exit 1; \
	done

clean:
	rm -rf $(BUILD)
