# loom-private/mlperf-tiny-anomaly-model. Targets come from ../common.mk:
#   make build | make check-large [CHECK_JSON=path] | make test | make clean
APP := anomaly
SRCS := anomaly.cu model.cpp oracle.cpp assets.cpp
HEADERS := model.h oracle.h assets.h
include ../common.mk

# Test-only mutant, next to the main binary: the kernel never writes the last
# sample, so its output stays poisoned and the check must fail with exit 1.
MUTANT := $(BUILD)/anomaly_skip_last_mutant
TEST_SCRATCH := $(BUILD)/test-output
# Host-only test of the SHA-256, the model reader and validation, and the
# oracle's input checks (no GPU).
HOST_TEST := $(BUILD)/test_host
CXX := g++

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) \
	  -DLP_ANOMALY_MUTANT_SKIP_LAST_SAMPLE $(addprefix $(APP_DIR)/,$(SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(addprefix $(APP_DIR)/,model.cpp oracle.cpp \
              assets.cpp $(HEADERS))
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra $(APP_DIR)/tests/test_host.cpp \
	  $(addprefix $(APP_DIR)/,model.cpp oracle.cpp assets.cpp) -o $@

# Host tests, then small --set sample counts against the oracle, asset
# failures (exit 3), invalid dimensions (exit 2), and the mutant (exit 1).
# Each GPU run takes $(GPU_LOCK) inside the script.
test: $(BIN) $(MUTANT) $(HOST_TEST)
	$(HOST_TEST) $${LOOM_PRIVATE_ASSET_CACHE:-$$HOME/.cache/cuda-elastic/loom-private-assets}
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/tests/run_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH)
