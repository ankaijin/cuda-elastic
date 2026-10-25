APP := vecadd_product
SRCS := vecadd_product.cu fixtures.cpp oracle.cpp
HEADERS := fixtures.h oracle.h
include ../common.mk

CXX := g++
# Test-only binaries next to the main one. The four mutants must fail the
# check; the slow-warmup binary shows bench warmups stay outside the timed
# interval; the host test covers the oracle, the fixtures, and SHA-256.
MUTANTS := $(foreach n,1 2 3 4,$(BUILD)/vecadd_product_mutant$(n))
SLOW_WARMUP := $(BUILD)/vecadd_product_slow_warmup
HOST_TEST := $(BUILD)/test_host
TEST_SCRATCH := $(BUILD)/test-output
SRC_PATHS := $(addprefix $(APP_DIR)/,$(SRCS) $(HEADERS))

.PHONY: test

$(BUILD)/vecadd_product_mutant%: $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT=$* \
	  $(addprefix $(APP_DIR)/,$(SRCS)) -o $@

$(SLOW_WARMUP): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_TEST_SLOW_WARMUP \
	  $(addprefix $(APP_DIR)/,$(SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/fixtures.cpp \
              $(APP_DIR)/oracle.cpp $(APP_DIR)/fixtures.h $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -pthread -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/fixtures.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(MUTANTS) $(SLOW_WARMUP) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(BUILD) $(TEST_SCRATCH)
