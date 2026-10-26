APP := pagerank
SRCS := pagerank.cu graph.cpp oracle.cpp
HEADERS := graph.h oracle.h
# The optimized variant (README, "Optimized variant"): one persistent kernel
# over a level-ordered copy of the graph.
OPT_SRCS := pagerank_optimized.cu layout.cpp graph.cpp oracle.cpp
OPT_HEADERS := layout.h graph.h oracle.h
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

ifneq ($(VARIANT),optimized)
test: $(BIN) $(MUTANT) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH)
else
# The optimized variant runs the baseline's GPU cases on its own binary and
# Jacobi mutant, the layout's host test, and every tuning-table entry through
# a test-only build that may force one (tests/run_optimized_tests.sh).
OPT_MUTANT := $(BUILD)/pagerank_optimized_jacobi_mutant
OPT_TUNINGS := $(BUILD)/pagerank_optimized_tunings
LAYOUT_TEST := $(BUILD)/test_layout
OPT_TEST_SCRATCH := $(BUILD)/test-output-optimized

$(OPT_MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_PAGERANK_MUTANT_JACOBI \
	  $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

$(OPT_TUNINGS): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_PAGERANK_TEST_TUNING \
	  $(addprefix $(APP_DIR)/,$(OPT_SRCS)) -o $@

$(LAYOUT_TEST): $(APP_DIR)/tests/test_layout.cpp $(APP_DIR)/layout.cpp $(APP_DIR)/graph.cpp \
                $(APP_DIR)/oracle.cpp $(APP_DIR)/layout.h $(APP_DIR)/graph.h $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -Wall -Wextra \
	  $(APP_DIR)/tests/test_layout.cpp $(APP_DIR)/layout.cpp $(APP_DIR)/graph.cpp \
	  $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(OPT_MUTANT) $(OPT_TUNINGS) $(HOST_TEST) $(LAYOUT_TEST)
	$(HOST_TEST)
	$(LAYOUT_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(OPT_MUTANT) $(OPT_TEST_SCRATCH)
	$(LOCK) $(APP_DIR)/tests/run_optimized_tests.sh $(OPT_TUNINGS) $(OPT_TEST_SCRATCH)
endif
