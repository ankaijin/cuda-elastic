# llama2c builds and checks. Binaries go to <repo>/<gpu>/llama2c/build (see
# gpu.sh), so builds for different GPUs never overwrite each other. Override
# with GPU=... ARCH=...
#   make sota          build the SOTA binary (llama2c_sota)
#   make baseline      build the baseline (llama2c_cuda_graph) into build/
#   make check-sota    validate the SOTA (deterministic + random inputs,
#                      nocopy + copy, graph + direct) at a small shape and the
#                      sheet shape
#   make compare       compare_sota.sh: baseline graph, baseline direct and
#                      SOTA dumps on identical inputs
# GPU commands run under flock $(GPU_LOCK) so they never share the GPU.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
BUILD := $(ROOT)/$(GPU)/llama2c/build
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
RUN := flock $(GPU_LOCK)

SOTA := $(BUILD)/llama2c_sota
BASELINE := $(BUILD)/llama2c_cuda_graph

# Sheet shape: batch size rows repeats warmup_ms.
SHEET := 1024 16384 8192 1 100
# Small shape: size not a multiple of 4 takes the generic path.
SMALL := 64 1000 300 1 20
SMALL_ODD := 33 1001 77 2 20

all: sota baseline

sota: $(SOTA)
baseline: $(BASELINE)

$(BUILD):
	mkdir -p $@

$(SOTA): llama2c_sota.cu llama2c_common.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BASELINE): llama2c_cuda_graph.cu llama2c_common.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

check-sota: $(SOTA)
	@set -e; \
	for shape in "$(SMALL)" "$(SMALL_ODD)" "$(SHEET)"; do \
	  for inputs in deterministic random=1 random=2; do \
	    for mode in graph direct; do \
	      for copies in nocopy copy; do \
	        line="$$($(RUN) $(SOTA) $$mode auto $$shape $$copies $$inputs)" || { echo "FAIL: $$mode $$shape $$copies $$inputs"; echo "$$line"; exit 1; }; \
	        echo "$$line" | tr ' ' '\n' | grep -E '^(mode|batch|size|rows|copies|inputs|seed|gpu_ms_per_iter|valid)=' | paste -sd' '; \
	      done; \
	    done; \
	  done; \
	done; \
	echo "check-sota: all runs valid"

compare: $(SOTA) $(BASELINE)
	./compare_sota.sh

clean:
	rm -rf $(BUILD)

.PHONY: all sota baseline check-sota compare clean
