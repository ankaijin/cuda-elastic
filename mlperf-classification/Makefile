# mlperf-tiny keyword spotting and visual wake words, kernel-level CUDA ports.
# Both applications build against mlperf-tiny-classification-include (and
# mlperf-tiny-anomaly-include) in Loom's manifest.
#
#   build/loom_kws, build/loom_vww
#       Loom's builds of the applications: upstream mlperf-tiny at the commit
#       Loom pins, compiled with the manifest's compiler_options around
#       loom/mlperf-tiny-{kws,vww}-smoke.h. Stdout must equal the
#       expected-smoke oracle; each exits 0 iff its smoke check passes.
#   build/kws_postprocess  CUDA port of mlperf_tiny_kws_postprocess_kernel
#   build/vww_preprocess   CUDA port of mlperf_tiny_vww_preprocess_kernel
#   build/kws_postprocess_sota, build/vww_preprocess_sota
#       the fastest versions found for the RTX 3090 (`make sota`): zero-copy
#       KWS, chunked copy/compute pipeline VWW. Each rotates its timed
#       iterations over --rotate=3 input sets and checks them all.
#
# Shapes run by `make check`:
#   kws  smoke 12 x 1 (Loom's smoke workload; 12 is also upstream's
#        kCategoryCount), batch 12 x 4096, batch 12 x 4M;
#        smoke, random and ties inputs
#   vww  smoke 16 (Loom's smoke workload), model 27648 (96 x 96 x 3,
#        upstream's kVwwInputSize), large 256M; smoke and random inputs
# Each shape pins its --replays so the timed graph replays (kernel-only plus
# end-to-end, excluding warm-up) total about 10 s on the RTX 3090; the
# binaries report the measured total as "timed total".
# `make check LAUNCH=direct` runs the same checks with the work launched
# directly instead of as CUDA graphs.
# `make check-sota` (also run by `make check`) runs the SOTA binaries on the
# same shapes and inputs plus odd shapes (KWS 12 x 1000003 and a 7-category
# shape for the generic kernel, VWW 1000003 and 9999991 bytes), with the
# same pinned --replays (the odd shapes' take a few seconds); they launch
# directly unless SOTA_LAUNCH=graph. REPLAYS=N overrides every pinned replay
# count, for quick runs (check-baseline and check-sota alike).
# compare_sota.sh cmps the SOTA outputs against both baseline launch modes.
SHELL := /bin/bash
# Binaries go to <repo>/<gpu>/mlperf-classification/build (see gpu.sh), so builds for
# different GPUs never overwrite each other. Override with GPU=... ARCH=...
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
LOOM_CXX ?= clang++
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --fmad=false \
             -Xcompiler -ffp-contract=off
BUILD := $(ROOT)/$(GPU)/mlperf-classification/build
LOOM := loom
LAUNCH ?= graph
SOTA_LAUNCH ?=
REPLAYS ?=

MLPERF_TINY_URL := https://github.com/mlcommons/tiny.git
MLPERF_TINY_COMMIT := 4addd0fa08d216e20637637874e084895f289da4
UPSTREAM := $(BUILD)/mlperf-tiny/benchmark
APPS := $(UPSTREAM)/reference_submissions
APP_kws := $(APPS)/keyword_spotting
APP_vww := $(APPS)/person_detection

# Loom's manifest.json compiler_options. Two adjustments for a stock clang:
# Loom's -include=<file> is spelled -include <file>, and the Loom shim
# directories go ahead of upstream's so their stub headers take precedence
# over upstream's util/tf_micro_model_runner.h.
LOOM_CXXFLAGS := -std=c++17 -O1 -ffp-contract=off -fno-exceptions -fno-rtti \
                 -Wno-unknown-pragmas
LOOM_SHIMS := -I$(LOOM)/mlperf-tiny-anomaly-include \
              -I$(LOOM)/mlperf-tiny-classification-include

# count:batch:replays
KWS_SHAPES := 12:1:1185000 12:4096:348000 12:4194304:800
KWS_INPUTS := smoke random ties
# count:replays
VWW_SHAPES := 16:1265000 27648:522000 268435456:444
VWW_INPUTS := smoke random
# The SOTA checks add odd shapes: sizes that are not multiples of the tile,
# vector or chunk widths, and a count other than 12 (the generic kernel).
KWS_SOTA_SHAPES := $(KWS_SHAPES) 12:1000003:1000 7:4099:200000
VWW_SOTA_SHAPES := $(VWW_SHAPES) 1000003:50000 9999991:5000

CUDA_TARGETS := $(BUILD)/kws_postprocess $(BUILD)/vww_preprocess
SOTA_TARGETS := $(BUILD)/kws_postprocess_sota $(BUILD)/vww_preprocess_sota

all: $(BUILD)/loom_kws $(BUILD)/loom_vww $(CUDA_TARGETS) $(SOTA_TARGETS)

sota: $(SOTA_TARGETS)

$(BUILD):
	mkdir -p $@

$(BUILD)/mlperf-tiny.stamp: | $(BUILD)
	rm -rf $(BUILD)/mlperf-tiny
	git init -q $(BUILD)/mlperf-tiny
	git -C $(BUILD)/mlperf-tiny fetch -q --depth 1 $(MLPERF_TINY_URL) \
	  $(MLPERF_TINY_COMMIT)
	git -C $(BUILD)/mlperf-tiny checkout -q FETCH_HEAD
	touch $@

$(BUILD)/loom_%: $(BUILD)/mlperf-tiny.stamp $(LOOM)/mlperf-tiny-%-smoke.h \
                 $(LOOM)/mlperf-tiny-classification-stub.h
	$(LOOM_CXX) $(LOOM_CXXFLAGS) $(LOOM_SHIMS) -I$(UPSTREAM) -I$(APP_$*) \
	  -include $(LOOM)/mlperf-tiny-$*-smoke.h -o $@ \
	  $(APP_$*)/submitter_implemented.cpp

$(BUILD)/%: %.cu classification_harness.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

check: all check-baseline check-sota

check-baseline: all
	@for app in kws vww; do \
	  echo "$(BUILD)/loom_$$app"; \
	  $(BUILD)/loom_$$app > $(BUILD)/loom_$$app.out || exit 1; \
	  cmp $(BUILD)/loom_$$app.out \
	    $(LOOM)/mlperf-tiny-$$app-expected-smoke.txt || exit 1; \
	done
	@for shape in $(KWS_SHAPES); do \
	  IFS=: read count batch replays <<< "$$shape"; \
	  for inputs in $(KWS_INPUTS); do \
	    $(BUILD)/kws_postprocess --count=$$count --batch=$$batch \
	      --inputs=$$inputs --launch=$(LAUNCH) \
	      --replays=$(or $(REPLAYS),$$replays) || exit 1; \
	  done; \
	done
	@for shape in $(VWW_SHAPES); do \
	  IFS=: read count replays <<< "$$shape"; \
	  for inputs in $(VWW_INPUTS); do \
	    $(BUILD)/vww_preprocess --count=$$count --inputs=$$inputs \
	      --launch=$(LAUNCH) --replays=$(or $(REPLAYS),$$replays) || exit 1; \
	  done; \
	done

check-sota: sota
	@for shape in $(KWS_SOTA_SHAPES); do \
	  IFS=: read count batch replays <<< "$$shape"; \
	  for inputs in $(KWS_INPUTS); do \
	    $(BUILD)/kws_postprocess_sota --count=$$count --batch=$$batch \
	      --inputs=$$inputs $(if $(SOTA_LAUNCH),--launch=$(SOTA_LAUNCH)) \
	      --replays=$(or $(REPLAYS),$$replays) || exit 1; \
	  done; \
	done
	@for shape in $(VWW_SOTA_SHAPES); do \
	  IFS=: read count replays <<< "$$shape"; \
	  for inputs in $(VWW_INPUTS); do \
	    $(BUILD)/vww_preprocess_sota --count=$$count --inputs=$$inputs \
	      $(if $(SOTA_LAUNCH),--launch=$(SOTA_LAUNCH)) \
	      --replays=$(or $(REPLAYS),$$replays) || exit 1; \
	  done; \
	done

clean:
	rm -rf $(BUILD)

.PHONY: all sota check check-baseline check-sota clean
