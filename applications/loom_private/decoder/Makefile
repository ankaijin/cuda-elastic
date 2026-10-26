# loom-private/llama2c-decoder. See README.md.
APP := decoder
SRCS := decoder.cu oracle.cpp
HEADERS := oracle.h
# Optimized variant (README "Optimized variant"): same contract and oracle.
OPT_SRCS := decoder_optimized.cu oracle.cpp
OPT_HEADERS := oracle.h
EXTRA_NVCCFLAGS := -Xcompiler -pthread
include ../common.mk

.PHONY: test
# Small --set shapes, invalid dimensions, and injected faults; never the
# large check. The script runs inside one GPU lock and takes no lock itself.
ifeq ($(VARIANT),optimized)
# The optimized tests also compare digests with the baseline binary and run
# two kernel mutants (-DDECODER_OPT_MUTANT=1|2) that the check must reject.
MUTANTS := $(BUILD)/$(APP)_optimized_mutant1 $(BUILD)/$(APP)_optimized_mutant2
$(BUILD)/$(APP)_optimized_mutant%: $(addprefix $(APP_DIR)/,$(OPT_SRCS) $(OPT_HEADERS)) \
        $(CONTRACT) $(LP_ROOT)/common/lp_harness.cuh $(LP_ROOT)/common.mk $(APP_DIR)/Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DDECODER_OPT_MUTANT=$* \
	  $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

test: $(BIN) $(MUTANTS)
	$(MAKE) -C $(APP_DIR) build VARIANT=baseline
	$(LOCK) bash $(APP_DIR)/tests/run_optimized_tests.sh $(BIN) $(BUILD)/$(APP) $(MUTANTS)
else
test: $(BIN)
	$(LOCK) bash $(APP_DIR)/tests/run_tests.sh $(BIN)
endif
