APP := pagerank
SRCS := pagerank.cu graph.cpp oracle.cpp
HEADERS := graph.h oracle.h
include ../common.mk

CXX := g++
# Test-only binaries, next to the main one. The mutant runs Jacobi sweeps and
# must fail the oracle; the host test checks the generator and schedule.
MUTANT := $(BUILD)/pagerank_jacobi_mutant
HOST_TEST := $(BUILD)/test_host
TEST_SCRATCH := $(BUILD)/test-output

.PHONY: test

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_PAGERANK_MUTANT_JACOBI \
	  $(addprefix $(APP_DIR)/,$(SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/graph.cpp $(APP_DIR)/oracle.cpp \
              $(APP_DIR)/graph.h $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/graph.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(MUTANT) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH)
