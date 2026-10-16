# Variants: main.c reference (4x4), frozen CUDA baseline, optimized CUDA.
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

VARIANTS := baseline optimized
SHAPES := 4x4 medium large
CUDA_TARGETS := $(foreach v,$(VARIANTS),$(foreach s,$(SHAPES),$(BUILD)/msa_$(v)_$(s)))

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

compare-large: $(BUILD)/msa_baseline_large $(BUILD)/msa_optimized_large
	$(BUILD)/msa_baseline_large
	$(BUILD)/msa_optimized_large

# Repeated, interleaved runs of every shape and input set; median and spread.
benchmark: all
	python3 benchmark.py

clean:
	rm -rf $(BUILD)

.PHONY: all check compare-large benchmark clean
