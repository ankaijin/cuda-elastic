APP := vecadd_memory
SRCS := vecadd_memory.cu oracle.cpp
HEADERS := oracle.h
OPT_SRCS := vecadd_memory_optimized.cu oracle.cpp
OPT_HEADERS := oracle.h
include ../common.mk

CXX := g++
# Test-only binaries, next to the main one (the optimized variant's carry an
# _optimized suffix). The drop-last mutant never writes the last element and
# must fail the oracle; the optimized variant has two more, each skipping the
# last chunk's upload or download. The host test checks the oracle itself.
SUFFIX := $(if $(filter optimized,$(VARIANT)),_optimized)
MUTANT := $(BUILD)/vecadd_memory_drop_last_mutant$(SUFFIX)
OPT_MUTANTS := $(if $(filter optimized,$(VARIANT)),$(foreach n,skip_upload skip_download,$(BUILD)/vecadd_memory_$(n)_mutant_optimized))
HOST_TEST := $(BUILD)/test_host
TEST_SCRATCH := $(BUILD)/test-output$(SUFFIX)

.PHONY: test

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT_DROP_LAST \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(BUILD)/vecadd_memory_skip_upload_mutant_optimized: $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT_SKIP_UPLOAD \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(BUILD)/vecadd_memory_skip_download_mutant_optimized: $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT_SKIP_DOWNLOAD \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(MUTANT) $(OPT_MUTANTS) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH) $(VARIANT) \
	  $(BUILD)/vecadd_memory_skip_upload_mutant_optimized \
	  $(BUILD)/vecadd_memory_skip_download_mutant_optimized
