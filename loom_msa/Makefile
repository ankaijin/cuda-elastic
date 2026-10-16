# Variants: main.c reference (4x4), frozen CUDA baseline, optimized CUDA,
# and the SOTA CUDA variant (msa_sota.cu; rotates over 3 input sets by
# default, see msa_harness.cuh --rotate).
#
# Every comparison build pins shape AND replay count explicitly, so neither
# binary depends on its source's defaults (msa_optimized.cu defaults to the
# large shape, the harness to the medium one):
#   *_4x4     main.c's shape; also checks main.c's hard-coded constants
#   *_medium  1000 x 50000, the original workload and tuning point
#   *_large   25000 x 250000, production scale
# Binaries go to <repo>/<gpu>/loom_msa/build (see gpu.sh), so builds for
# different GPUs never overwrite each other. Override with GPU=... ARCH=...
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
CC ?= cc
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -fopenmp
BUILD := $(ROOT)/$(GPU)/loom_msa/build
# `make check LAUNCH=direct` runs the checks with the pipeline launched
# directly instead of as a CUDA graph.
LAUNCH ?= graph

SHAPE_4x4 := -DTOKEN_COUNT=4 -DCONTEXT_COUNT=4 -DGRAPH_REPLAYS=5000
SHAPE_medium := -DTOKEN_COUNT=1000 -DCONTEXT_COUNT=50000 -DGRAPH_REPLAYS=5000
SHAPE_large := -DTOKEN_COUNT=25000 -DCONTEXT_COUNT=250000 -DGRAPH_REPLAYS=50

VARIANTS := baseline optimized sota
SHAPES := 4x4 medium large
CUDA_TARGETS := $(foreach v,$(VARIANTS),$(foreach s,$(SHAPES),$(BUILD)/msa_$(v)_$(s)))
SOTA_TARGETS := $(foreach s,$(SHAPES),$(BUILD)/msa_sota_$(s))

all: $(BUILD)/reference $(CUDA_TARGETS)

$(BUILD):
	mkdir -p $@

$(BUILD)/reference: main.c | $(BUILD)
	$(CC) -O2 -std=c11 -o $@ $< -lm

define shape_rule
$(BUILD)/msa_%_$(1): msa_%.cu msa_harness.cuh | $(BUILD)
	$$(NVCC) $$(NVCCFLAGS) $$(SHAPE_$(1)) -o $$@ $$<
endef
$(foreach s,$(SHAPES),$(eval $(call shape_rule,$(s))))

check: all
	$(BUILD)/reference
	@for bin in $(CUDA_TARGETS); do \
	  for inputs in repeat random; do \
	    $$bin --inputs=$$inputs --launch=$(LAUNCH) || exit 1; \
	  done; \
	done

sota: $(SOTA_TARGETS) $(BUILD)/msa_sota_large_1chunk

# msa_sota_large with the imu copied in one piece and the attention as one
# launch: the same kernel code, profiled by profile_cuda.sh's ncu step. The
# 13 per-chunk launches of msa_sota_large run concurrently, so ncu, which
# serializes launches, would time each of them on an otherwise empty GPU.
$(BUILD)/msa_sota_large_1chunk: msa_sota.cu msa_harness.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(SHAPE_large) -DSOTA_CHUNKS=1 -o $@ $<

# The SOTA binaries at every shape, both input modes, at their default
# rotation (3 input sets, each checked against its own CPU reference).
check-sota: $(SOTA_TARGETS)
	@for bin in $(SOTA_TARGETS); do \
	  for inputs in repeat random; do \
	    $$bin --inputs=$$inputs --launch=$(LAUNCH) || exit 1; \
	  done; \
	done

# Dumps of msa_optimized (graph and direct) and msa_sota compared within the
# harness tolerance, for several inputs and seeds (see compare_sota.sh).
compare-sota: $(CUDA_TARGETS)
	./compare_sota.sh $(BUILD)

compare-large: $(BUILD)/msa_baseline_large $(BUILD)/msa_optimized_large
	$(BUILD)/msa_baseline_large
	$(BUILD)/msa_optimized_large

# Repeated, interleaved runs of every shape and input set; median and spread.
benchmark: all
	python3 benchmark.py

clean:
	rm -rf $(BUILD)

.PHONY: all sota check check-sota compare-sota compare-large benchmark clean
