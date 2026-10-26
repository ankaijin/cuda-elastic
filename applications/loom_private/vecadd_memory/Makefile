APP := vecadd_memory
SRCS := vecadd_memory.cu oracle.cpp
HEADERS := oracle.h
OPT_SRCS := vecadd_memory_optimized.cu oracle.cpp
OPT_HEADERS := oracle.h
include ../common.mk

CXX := g++
# Test-only binaries, next to the main one (the optimized variant's carry an
# _optimized suffix). The drop-last mutant skips the last element and must fail
# the oracle. The optimized variant has three more mutants (each skips the last
# chunk's upload or download, or drops the harness stream's join of the
# chunks), and its mutants and a tuning binary are built with
# -DLP_VECADD_TEST_TUNING, which lets the tests set the chunk count and kernel
# shape through LP_VECADD_CHUNKS and LP_VECADD_KERNEL. The suite binary has no
# such override. The host test checks the oracle itself.
OPTIMIZED := $(filter optimized,$(VARIANT))
SUFFIX := $(if $(OPTIMIZED),_optimized)
TUNING := $(if $(OPTIMIZED),-DLP_VECADD_TEST_TUNING)
MUTANT := $(BUILD)/vecadd_memory_drop_last_mutant$(SUFFIX)
OPT_MUTANT_NAMES := skip_upload skip_download no_join
OPT_MUTANTS := $(if $(OPTIMIZED),$(foreach n,$(OPT_MUTANT_NAMES),$(BUILD)/vecadd_memory_$(n)_mutant_optimized))
TUNING_BIN := $(if $(OPTIMIZED),$(BUILD)/vecadd_memory_tuning_optimized)
HOST_TEST := $(BUILD)/test_host
TEST_SCRATCH := $(BUILD)/test-output$(SUFFIX)

.PHONY: test

$(MUTANT): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT_DROP_LAST $(TUNING) \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

# The mutant's name, upper-cased, is the macro suffix: skip_upload is
# LP_VECADD_MUTANT_SKIP_UPLOAD.
$(BUILD)/vecadd_memory_%_mutant_optimized: $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) \
	  -DLP_VECADD_MUTANT_$(shell echo $* | tr a-z A-Z) $(TUNING) \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(BUILD)/vecadd_memory_tuning_optimized: $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_TEST_TUNING \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(MUTANT) $(OPT_MUTANTS) $(TUNING_BIN) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(MUTANT) $(TEST_SCRATCH) $(VARIANT) $(BUILD)
