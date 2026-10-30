# Shared build rules for the RTX 5090 feature experiments.
#
# An experiment Makefile sets EXPERIMENT (its directory name) and CASES, and
# for each case <case>_SRCS (sources linked into one binary, absolute paths),
# <case>_ARCH, <case>_SPEC (the contract or experimental specification whose
# SHA-256 the binary embeds), <case>_APP (the harness's LP_APP), and
# optionally <case>_HEADERS, <case>_FLAGS, and <case>_ARCHFLAGS (default
# -arch=<case>_ARCH, as the production suite builds; a family target such as
# sm_120f needs -gencode arch=compute_120f,code=sm_120f, because -arch=sm_120f
# also embeds generic compute_120 PTX). Then it includes this file:
#
#   EXPERIMENT := packed
#   CASES := control add_u8x4
#   control_SRCS := $(LP_APP_DIR)/vecadd_product_optimized.cu ...
#   include ../common.mk
#
# Targets:
#   build   every case binary (the default); add -B for a fresh build
#   test    the experiment's own small-case tests (the experiment defines it)
#   clean   removes this GPU's build directory
#
# Binaries go to <repo>/<gpu>/blackwell_experiments/<experiment>/build/<case>.
# tools/build_info.py runs each compile and writes <case>.build.json beside the
# binary: the exact command, the compiler version, the architecture, the
# specification hash, and the SHA-256 of every source, header, and binary.
# The production suite's sources are compiled in place and never modified.
SHELL := /bin/bash
.DEFAULT_GOAL := build

BW_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(BW_ROOT)/../..)
LP_ROOT := $(ROOT)/applications/loom_private
EXP_DIR := $(BW_ROOT)/$(EXPERIMENT)
GPU := $(or $(GPU),$(shell $(ROOT)/gpu.sh name))
NVCC ?= nvcc
# The production suite's tunable flags and its required set (see
# applications/loom_private/common.mk): floating-point arithmetic as written.
NVCCFLAGS ?= -O3 -lineinfo
BW_REQUIRED_NVCCFLAGS := -std=c++17 --fmad=false -Xcompiler -ffp-contract=off
OUT := $(ROOT)/$(GPU)/blackwell_experiments/$(EXPERIMENT)
BUILD := $(OUT)/build
GPU_LOCK ?= /tmp/cuda-elastic-gpu.lock
LOCK := flock $(GPU_LOCK)
BUILD_INFO := python3 $(BW_ROOT)/tools/build_info.py
COMMON_DEPS := $(LP_ROOT)/common/lp_harness.cuh $(BW_ROOT)/common.mk \
               $(BW_ROOT)/tools/build_info.py $(EXP_DIR)/Makefile
spec_sha256 = $(shell sha256sum $(1) | cut -d' ' -f1)
arch_flags = $(or $($(1)_ARCHFLAGS),-arch=$($(1)_ARCH))

# compile_case <case> <output> <extra flags>: the compile command of a case,
# wrapped by build_info.py. Test builds of a case pass their extra defines.
define compile_case
mkdir -p $(dir $(2))
$(BUILD_INFO) --output $(2) --case $(1) --experiment $(EXPERIMENT) \
	  --arch $($(1)_ARCH) --spec $($(1)_SPEC) $(addprefix --source ,$($(1)_SRCS) $($(1)_HEADERS)) \
	  -- $(NVCC) $(NVCCFLAGS) $($(1)_FLAGS) $(3) $(BW_REQUIRED_NVCCFLAGS) $(call arch_flags,$(1)) \
	  -I$(LP_ROOT)/common -DLP_APP=\"$($(1)_APP)\" \
	  -DLP_CONTRACT_SHA256=\"$(call spec_sha256,$($(1)_SPEC))\" \
	  -DLP_VARIANT=\"$(EXPERIMENT)/$(1)\" $($(1)_SRCS) -o $(2)
endef

define case_rules
$(BUILD)/$(1): $$($(1)_SRCS) $$($(1)_HEADERS) $$($(1)_SPEC) $(COMMON_DEPS)
	$$(call compile_case,$(1),$$@,)
endef
$(foreach c,$(CASES),$(eval $(call case_rules,$(c))))

.PHONY: build clean print-cases
build: $(addprefix $(BUILD)/,$(CASES))

print-cases:
	@echo $(CASES)

clean:
	rm -rf $(BUILD)
