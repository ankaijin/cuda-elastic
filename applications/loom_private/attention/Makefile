# loom-private/multisensor-attention. Targets come from ../common.mk:
#   make build | make check-large [CHECK_JSON=path] | make test | make clean
# VARIANT=optimized builds attention_optimized.cu (same contract).
APP := attention
SRCS := attention.cu oracle.cpp
HEADERS := oracle.h
OPT_SRCS := attention_optimized.cu oracle.cpp
OPT_HEADERS := oracle.h
EXTRA_NVCCFLAGS := -Xcompiler -pthread -lpthread
include ../common.mk

ifeq ($(VARIANT),optimized)
# Test-only builds next to the optimized binary: one with the tuning hooks
# (environment variables force each kernel shape, a split count, the exact
# row maxima, or the compute path in the checked invocation), and a mutant
# whose exponential pass skips the last key of every tile, so the check must
# fail with exit 1.
HOOKS := $(BUILD)/$(APP)_optimized_hooks
MUTANT := $(BUILD)/$(APP)_optimized_skip_key_mutant

$(HOOKS): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_ATTENTION_TUNING_HOOKS \
	  $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) \
	  -DLP_ATTENTION_MUTANT_SKIP_TILE_LAST_KEY $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

# The baseline's cases on the optimized binary, then the optimized variant's
# own cases. Each GPU run takes $(GPU_LOCK) inside the scripts.
test: $(BIN) $(HOOKS) $(MUTANT)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/run_tests.sh $(BIN)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/run_tests_optimized.sh $(HOOKS) $(MUTANT)
else
# Small --set cases against the oracle plus invalid-dimension exits. Each GPU
# run takes $(GPU_LOCK) inside the script.
test: $(BIN)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/run_tests.sh $(BIN)
endif
