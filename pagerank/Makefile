# PageRank builds: the baseline (gapbs_pagerank_cuda_graph.cu) and the SOTA
# version (pagerank_sota.cu), both built like profile_cuda.sh builds the
# baseline (nvcc -O3 -arch=<gpu arch>, no FMA restrictions).  Binaries go to
# <repo>/<gpu>/pagerank/build (see gpu.sh); override with GPU=... ARCH=...
#
#   make sota          build the SOTA binary
#   make check-sota    run the SOTA's six validation cases (both launch modes)
#   make bench-sota    validate, then time it (default: direct launches, 3 rotated inputs)
#   make baseline      build the baseline binary
#   make check         run the baseline's validation
#   make small         both, on a 2^14-vertex graph (*_small; time_all.sh's
#                      launch-overhead-sensitive shape)
#   make compare       compare_sota.sh: baseline (graph, direct) vs SOTA dumps
#
# LAUNCH=graph|direct overrides each binary's default launch mode (the
# baseline's is graph, the SOTA's direct).  GPU commands run under
# flock $(LOCK), shared with the other workloads.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)
BUILD := $(ROOT)/$(GPU)/pagerank/build
LOCK ?= /tmp/cuda-elastic-gpu.lock
LAUNCH ?=
LAUNCH_ENV := $(if $(LAUNCH),PAGERANK_LAUNCH=$(LAUNCH))
SRC_DIR := $(ROOT)/pagerank

SOTA := $(BUILD)/pagerank_sota
BASELINE := $(BUILD)/gapbs_pagerank_cuda_graph
SMALL := -DPAGERANK_LOG_VERTICES=14

all: sota baseline
sota: $(SOTA)
baseline: $(BASELINE)
small: $(SOTA)_small $(BASELINE)_small

$(BUILD):
	mkdir -p $@

$(SOTA): $(SRC_DIR)/pagerank_sota.cu $(SRC_DIR)/pagerank_common.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BASELINE): $(SRC_DIR)/gapbs_pagerank_cuda_graph.cu $(SRC_DIR)/pagerank_common.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(SOTA)_small: $(SRC_DIR)/pagerank_sota.cu $(SRC_DIR)/pagerank_common.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(SMALL) -o $@ $<

$(BASELINE)_small: $(SRC_DIR)/gapbs_pagerank_cuda_graph.cu $(SRC_DIR)/pagerank_common.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(SMALL) -o $@ $<

# Without LAUNCH, both launch modes are validated.
check-sota: $(SOTA)
	@for launch in $(or $(LAUNCH),direct graph); do \
	  PAGERANK_MODE=validate PAGERANK_LAUNCH=$$launch flock $(LOCK) $(SOTA) || exit 1; \
	done

bench-sota: $(SOTA)
	$(LAUNCH_ENV) GRAPH_REPLAYS=$(or $(GRAPH_REPLAYS),60) flock $(LOCK) $(SOTA)

check: $(BASELINE)
	PAGERANK_MODE=validate $(LAUNCH_ENV) flock $(LOCK) $(BASELINE)

compare: $(SOTA) $(BASELINE)
	BUILD=$(BUILD) LOCK=$(LOCK) $(SRC_DIR)/compare_sota.sh

clean:
	rm -f $(SOTA) $(BASELINE) $(SOTA)_small $(BASELINE)_small

.PHONY: all sota baseline small check-sota bench-sota check compare clean
