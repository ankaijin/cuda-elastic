APP := vecadd_memory
SRCS := vecadd_memory.cu oracle.cpp
HEADERS := oracle.h
include ../common.mk

CXX := g++
# Test-only binaries, next to the main one. The mutant drops the last element
# and must fail the oracle; the host test checks the oracle itself.
MUTANT := $(BUILD)/vecadd_memory_drop_last_mutant
HOST_TEST := $(BUILD)/test_host
TEST_SCRATCH := $(BUILD)/test-output

.PHONY: test

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT_DROP_LAST \
	  $(addprefix $(APP_DIR)/,$(SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(MUTANT) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH)
