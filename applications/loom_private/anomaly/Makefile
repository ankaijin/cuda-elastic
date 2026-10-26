# loom-private/mlperf-tiny-anomaly-model. Targets come from ../common.mk:
#   make build | make check-large [CHECK_JSON=path] | make test | make clean
APP := anomaly
SRCS := anomaly.cu model.cpp oracle.cpp assets.cpp
HEADERS := model.h oracle.h assets.h
# The optimized variant (VARIANT=optimized): same contract, tensor-core tiles
# and pipelined transfers; the host files are shared unchanged.
OPT_SRCS := anomaly_optimized.cu model.cpp oracle.cpp assets.cpp
OPT_HEADERS := $(HEADERS) requant.cuh
include ../common.mk

# Test-only mutant, next to the main binary: the kernel never writes the last
# sample, so its output stays poisoned and the check must fail with exit 1.
ifeq ($(VARIANT),baseline)
MUTANT := $(BUILD)/anomaly_skip_last_mutant
TEST_SCRATCH := $(BUILD)/test-output
else
MUTANT := $(BUILD)/anomaly_optimized_skip_last_mutant
TEST_SCRATCH := $(BUILD)/test-output-optimized
endif
# Host-only test of the SHA-256, the model reader and validation, and the
# oracle's input checks (no GPU).
HOST_TEST := $(BUILD)/test_host
CXX := g++

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) \
	  -DLP_ANOMALY_MUTANT_SKIP_LAST_SAMPLE $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(addprefix $(APP_DIR)/,model.cpp oracle.cpp \
              assets.cpp $(HEADERS))
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra $(APP_DIR)/tests/test_host.cpp \
	  $(addprefix $(APP_DIR)/,model.cpp oracle.cpp assets.cpp) -o $@

ASSET_CACHE = $${LOOM_PRIVATE_ASSET_CACHE:-$$HOME/.cache/cuda-elastic/loom-private-assets}

ifeq ($(VARIANT),baseline)
# Host tests, then small --set sample counts against the oracle, asset
# failures (exit 3), invalid dimensions (exit 2), and the mutant (exit 1).
# Each GPU run takes $(GPU_LOCK) inside the script.
test: $(BIN) $(MUTANT) $(HOST_TEST)
	$(HOST_TEST) $(ASSET_CACHE)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/tests/run_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH)
else
# The optimized variant: the same host tests and run_tests.sh cases, then
# tests/run_optimized_tests.sh: the exactness test of the binary32 fast paths
# (and its no-margin mutant, which it must reject), a build with the
# baseline's arithmetic for every code (the fast paths' fallback), and builds
# with forced pipeline chunk counts, including each GPU's choice (16, 32).
OPT_TEST_DIR := $(BUILD)/optimized-tests
OPT_CHUNKS := 1 3 16 32
OPT_CHUNK_BINS := $(foreach n,$(OPT_CHUNKS),$(OPT_TEST_DIR)/anomaly_optimized_chunks$(n))
OPT_TEST_BINS := $(OPT_TEST_DIR)/anomaly_optimized_baseline_arithmetic $(OPT_CHUNK_BINS) \
                 $(OPT_TEST_DIR)/test_requant $(OPT_TEST_DIR)/test_requant_no_margin_mutant
TEST_REQUANT_SRCS := $(addprefix $(APP_DIR)/,tests/test_requant.cu model.cpp assets.cpp)

$(OPT_TEST_DIR)/anomaly_optimized_baseline_arithmetic: $(BIN)
	@mkdir -p $(OPT_TEST_DIR)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_ANOMALY_BASELINE_ARITHMETIC \
	  $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

$(OPT_TEST_DIR)/anomaly_optimized_chunks%: $(BIN)
	@mkdir -p $(OPT_TEST_DIR)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_ANOMALY_FORCE_CHUNKS=$* \
	  $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

TEST_REQUANT_DEPS := $(TEST_REQUANT_SRCS) $(addprefix $(APP_DIR)/,$(OPT_HEADERS)) \
                     $(LP_ROOT)/common.mk $(APP_DIR)/Makefile

$(OPT_TEST_DIR)/test_requant: $(TEST_REQUANT_DEPS)
	@mkdir -p $(OPT_TEST_DIR)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) $(TEST_REQUANT_SRCS) -o $@

$(OPT_TEST_DIR)/test_requant_no_margin_mutant: $(TEST_REQUANT_DEPS)
	@mkdir -p $(OPT_TEST_DIR)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_ANOMALY_MUTANT_NO_MARGIN \
	  $(TEST_REQUANT_SRCS) -o $@

test: $(BIN) $(MUTANT) $(HOST_TEST) $(OPT_TEST_BINS)
	$(HOST_TEST) $(ASSET_CACHE)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/tests/run_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/tests/run_optimized_tests.sh $(BIN) $(OPT_TEST_DIR) \
	  "$(OPT_CHUNKS)" $(ASSET_CACHE) $(TEST_SCRATCH)
endif
