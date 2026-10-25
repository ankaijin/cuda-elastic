# Shared build rules for the loom_private applications.
#
# An application Makefile sets APP (its directory name), SRCS (CUDA and host
# sources linked into one binary) and optionally HEADERS, EXTRA_NVCCFLAGS, and
# TEST_COMMANDS, then includes this file:
#
#   APP := pagerank
#   SRCS := pagerank.cu oracle.cpp
#   include ../common.mk
#
# Targets:
#   build        builds $(BIN) for this GPU (the default)
#   check-large  runs the large correctness check in Direct and Graph modes
#   bench        times the large workload (LAUNCH=direct|graph required)
#   test         the application's own tests (never the large check)
#   clean        removes this GPU's build directory
#
# Binaries go to <repo>/<gpu>/loom_private/<app>/build. Every GPU run takes
# $(GPU_LOCK); do not wrap these targets in another flock on the same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

LP_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(LP_ROOT)/../..)
APP_DIR := $(LP_ROOT)/$(APP)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
ARCH := $(or $(ARCH),$(shell $(ROOT)/gpu.sh arch))
NVCC ?= nvcc
# NVCCFLAGS holds the tunable flags and may be overridden. The required set
# below is always added (through LP_FLAGS, which every compile rule, including
# the applications' test rules, passes): the language standard, this GPU's
# architecture, and --fmad=false with -ffp-contract=off, which keep
# floating-point arithmetic as written, matching private Loom's
# -ffp-contract=off builds and the CPU oracles.
NVCCFLAGS ?= -O3 -lineinfo
LP_REQUIRED_NVCCFLAGS := -std=c++17 -arch=$(ARCH) --fmad=false \
                         -Xcompiler -ffp-contract=off
OUT := $(ROOT)/$(GPU)/loom_private/$(APP)
BUILD := $(OUT)/build
BIN := $(BUILD)/$(APP)
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
LOCK := flock $(GPU_LOCK)
CONTRACT := $(APP_DIR)/contract.json
CONTRACT_SHA256 = $(shell sha256sum $(CONTRACT) | cut -d' ' -f1)
LP_FLAGS = $(LP_REQUIRED_NVCCFLAGS) -I$(LP_ROOT)/common -DLP_APP=\"$(APP)\" \
           -DLP_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"
# CHECK_JSON=path (BENCH_JSON=path) also writes the result record there.
CHECK_ARGS = $(if $(CHECK_JSON),--json-out $(CHECK_JSON))
# bench needs LAUNCH=direct or LAUNCH=graph; ITERATIONS and WARMUP are optional.
BENCH_ARGS = --launch $(LAUNCH) $(if $(ITERATIONS),--iterations $(ITERATIONS)) \
             $(if $(WARMUP),--warmup $(WARMUP)) $(if $(BENCH_JSON),--json-out $(BENCH_JSON))

.PHONY: build check-large bench test clean

build: $(BIN)

$(BIN): $(addprefix $(APP_DIR)/,$(SRCS) $(HEADERS)) $(CONTRACT) \
        $(LP_ROOT)/common/lp_harness.cuh $(LP_ROOT)/common.mk $(APP_DIR)/Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) \
	  $(addprefix $(APP_DIR)/,$(SRCS)) -o $@

check-large: $(BIN)
	$(LOCK) $(BIN) --preset large --action check --launch both $(CHECK_ARGS)

bench: $(BIN)
	@case "$(LAUNCH)" in direct|graph) ;; \
	  *) echo "bench needs LAUNCH=direct or LAUNCH=graph" >&2; exit 2;; esac
	$(LOCK) $(BIN) --preset large --action bench $(BENCH_ARGS)

clean:
	rm -rf $(BUILD)
