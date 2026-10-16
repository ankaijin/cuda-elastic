# mlperf-tiny-anomaly-detection, kernel-level CUDA port.
#
#   build/loom_reference  Loom's build of the application: upstream mlperf-tiny
#                         at the commit Loom pins, compiled with the manifest's
#                         compiler_options around loom/mlperf-tiny-anomaly-smoke.h.
#                         Its stdout must equal the expected-smoke oracle and it
#                         exits 0 iff the smoke check passes.
#   build/anomaly_mse     CUDA port of mlperf_tiny_anomaly_mse_kernel.
#   build/anomaly_mse_sota  fastest version on the RTX 3090 (anomaly_mse_sota.cu):
#                         chunked copy/compute pipeline or zero-copy, one
#                         single-pass kernel, rotating over 3 input sets.
#
# Shapes run by `make check` (each with smoke and random inputs):
#   smoke  16 elements, Loom's smoke workload and oracle
#   model  640 elements, upstream's real kInputSize
#   large  64M elements
# Each shape pins its --replays so the timed graph replays (kernels-only plus
# end-to-end, excluding warm-up) total about 10 s on the RTX 3090; the
# binary reports the measured total as "timed total".
# `make check LAUNCH=direct` runs the same checks with the work launched
# directly instead of as CUDA graphs.
# `make sota` builds the SOTA binary; `make check-sota` runs it on the same
# shapes and inputs plus two odd counts (SOTA_SHAPES), with its default
# rotation over 3 input sets; `make check` runs both. compare_sota.sh compares
# the outputs of the baseline (graph and direct) and the SOTA.
SHELL := /bin/bash
# Binaries go to <repo>/<gpu>/mlperf-anomaly/build (see gpu.sh), so builds for
# different GPUs never overwrite each other. Override with GPU=... ARCH=...
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
LOOM_CXX ?= clang++
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --fmad=false \
             -Xcompiler -ffp-contract=off
BUILD := $(ROOT)/$(GPU)/mlperf-anomaly/build
LOOM := loom
LAUNCH ?= graph

MLPERF_TINY_URL := https://github.com/mlcommons/tiny.git
MLPERF_TINY_COMMIT := 4addd0fa08d216e20637637874e084895f289da4
UPSTREAM := $(BUILD)/mlperf-tiny/benchmark
APP := $(UPSTREAM)/reference_submissions/anomaly_detection

# Loom's manifest.json compiler_options. Two adjustments for a stock clang:
# Loom's -include=<file> is spelled -include <file>, and the Loom shim
# directory goes ahead of upstream's so its stub headers take precedence over
# upstream's util/tf_micro_model_runner.h.
LOOM_CXXFLAGS := -std=c++17 -O1 -ffp-contract=off -fno-exceptions -fno-rtti \
                 -Wno-unknown-pragmas
LOOM_INCLUDES := -I$(LOOM)/mlperf-tiny-anomaly-include -I$(UPSTREAM) -I$(APP) \
                 -include $(LOOM)/mlperf-tiny-anomaly-smoke.h

# name:count:replays
SHAPES := smoke:16:440000 model:640:358000 large:67108864:688
# The SOTA adds two odd counts (not multiples of 4 or of its 4096-element
# chunk alignment): one on its zero-copy path, one on its copy pipeline.
SOTA_SHAPES := $(SHAPES) odd:1000003:45000 odd-large:16777219:2800
SOTA_LAUNCH ?= $(LAUNCH)
# Optional command prefix for every binary run, e.g.
#   make check RUN="flock /tmp/cuda-elastic-gpu.lock"
RUN ?=
INPUTS := smoke random

all: $(BUILD)/loom_reference $(BUILD)/anomaly_mse $(BUILD)/anomaly_mse_sota

sota: $(BUILD)/anomaly_mse_sota

$(BUILD):
	mkdir -p $@

$(BUILD)/mlperf-tiny.stamp: | $(BUILD)
	rm -rf $(BUILD)/mlperf-tiny
	git init -q $(BUILD)/mlperf-tiny
	git -C $(BUILD)/mlperf-tiny fetch -q --depth 1 $(MLPERF_TINY_URL) \
	  $(MLPERF_TINY_COMMIT)
	git -C $(BUILD)/mlperf-tiny checkout -q FETCH_HEAD
	touch $@

$(BUILD)/loom_reference: $(BUILD)/mlperf-tiny.stamp \
                         $(LOOM)/mlperf-tiny-anomaly-smoke.h
	$(LOOM_CXX) $(LOOM_CXXFLAGS) $(LOOM_INCLUDES) -o $@ \
	  $(APP)/submitter_implemented.cpp

$(BUILD)/anomaly_mse: anomaly_mse.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD)/anomaly_mse_sota: anomaly_mse_sota.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

check: all check-baseline check-sota

check-baseline: $(BUILD)/loom_reference $(BUILD)/anomaly_mse
	$(BUILD)/loom_reference > $(BUILD)/loom_reference.out
	cmp $(BUILD)/loom_reference.out $(LOOM)/mlperf-tiny-anomaly-expected-smoke.txt
	@for shape in $(SHAPES); do \
	  IFS=: read name count replays <<< "$$shape"; \
	  for inputs in $(INPUTS); do \
	    $(RUN) $(BUILD)/anomaly_mse --count=$$count --inputs=$$inputs \
	      --launch=$(LAUNCH) --replays=$$replays || exit 1; \
	  done; \
	done

check-sota: $(BUILD)/anomaly_mse_sota
	@for shape in $(SOTA_SHAPES); do \
	  IFS=: read name count replays <<< "$$shape"; \
	  for inputs in $(INPUTS); do \
	    $(RUN) $(BUILD)/anomaly_mse_sota --count=$$count --inputs=$$inputs \
	      --launch=$(SOTA_LAUNCH) --replays=$$replays || exit 1; \
	  done; \
	done

clean:
	rm -rf $(BUILD)

.PHONY: all sota check check-baseline check-sota clean
