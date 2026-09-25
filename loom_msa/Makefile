# Variants: main.c reference (4x4), frozen CUDA baseline, optimized CUDA.
# The *_4x4 builds reproduce main.c's shape and check its hard-coded constants.
NVCC ?= nvcc
CC ?= cc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_86 -lineinfo
BUILD := build

CUDA_TARGETS := $(BUILD)/msa_baseline $(BUILD)/msa_optimized \
                $(BUILD)/msa_baseline_4x4 $(BUILD)/msa_optimized_4x4

all: $(BUILD)/reference $(CUDA_TARGETS)

$(BUILD):
	mkdir -p $@

$(BUILD)/reference: main.c | $(BUILD)
	$(CC) -O2 -std=c11 -o $@ $< -lm

$(BUILD)/msa_%: msa_%.cu msa_harness.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD)/msa_%_4x4: msa_%.cu msa_harness.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -DTOKEN_COUNT=4 -DCONTEXT_COUNT=4 -o $@ $<

check: all
	./$(BUILD)/reference
	@for bin in $(CUDA_TARGETS); do \
	  for inputs in repeat random; do \
	    ./$$bin --inputs=$$inputs || exit 1; \
	  done; \
	done

clean:
	rm -rf $(BUILD)

.PHONY: all check clean
