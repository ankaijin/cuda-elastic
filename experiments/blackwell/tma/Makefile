# Experiment B: TMA tile loads and warp specialization in
# loom-private/multisensor-attention.
#   make build | make test | make clean      (see ../README.md)
# Cases: control is the production optimized source, unchanged; tma and
# tma_producer build attention_tma.cu (attention_tma.diff is its exact
# difference) with LP_ATTENTION_TMA_MODE 1 and 2. All three use the production
# flags and -arch=sm_120.
HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
AT := $(abspath $(HERE)/../../../applications/loom_private/attention)
EXPERIMENT := tma
CASES := control tma tma_producer
control_SRCS := $(AT)/attention_optimized.cu $(AT)/oracle.cpp
tma_SRCS := $(HERE)/attention_tma.cu $(AT)/oracle.cpp
tma_producer_SRCS := $(tma_SRCS)
$(foreach c,$(CASES),$(eval $(c)_HEADERS := $(AT)/oracle.h))
$(foreach c,$(CASES),$(eval $(c)_ARCH := sm_120))
$(foreach c,$(CASES),$(eval $(c)_SPEC := $(AT)/contract.json))
$(foreach c,$(CASES),$(eval $(c)_APP := attention))
control_FLAGS := -Xcompiler -pthread -lpthread
tma_FLAGS := -Xcompiler -pthread -lpthread -I$(AT) -DLP_ATTENTION_TMA_MODE=1
tma_producer_FLAGS := -Xcompiler -pthread -lpthread -I$(AT) -DLP_ATTENTION_TMA_MODE=2
include ../common.mk

# Test builds of every case: the production tuning hooks (<case>_hooks), the
# production skip-key mutant (<case>_skip_key_mutant), and for the TMA cases a
# mutant whose compute warps never wait for a tile's copy
# (<case>_no_full_wait_mutant). Each must fail the check where the tests say.
TEST_BINS := $(foreach c,$(CASES),$(BUILD)/$(c)_hooks $(BUILD)/$(c)_skip_key_mutant) \
             $(foreach c,tma tma_producer,$(BUILD)/$(c)_no_full_wait_mutant)
define test_rules
$(BUILD)/$(1)_hooks: $$($(1)_SRCS) $$($(1)_HEADERS) $(COMMON_DEPS)
	$$(call compile_case,$(1),$$@,-DLP_ATTENTION_TUNING_HOOKS)
$(BUILD)/$(1)_skip_key_mutant: $$($(1)_SRCS) $$($(1)_HEADERS) $(COMMON_DEPS)
	$$(call compile_case,$(1),$$@,-DLP_ATTENTION_MUTANT_SKIP_TILE_LAST_KEY)
$(BUILD)/$(1)_no_full_wait_mutant: $$($(1)_SRCS) $$($(1)_HEADERS) $(COMMON_DEPS)
	$$(call compile_case,$(1),$$@,-DLP_ATTENTION_MUTANT_NO_FULL_WAIT)
endef
$(foreach c,$(CASES),$(eval $(call test_rules,$(c))))

.PHONY: test
test: build $(TEST_BINS)
	GPU_LOCK=$(GPU_LOCK) $(HERE)/tests/run_tests.sh $(BUILD) $(AT)
