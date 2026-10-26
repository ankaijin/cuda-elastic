APP := vecadd_product
SRCS := vecadd_product.cu fixtures.cpp oracle.cpp
HEADERS := fixtures.h oracle.h
OPT_SRCS := vecadd_product_optimized.cu fixtures.cpp oracle.cpp
OPT_HEADERS := fixtures.h oracle.h
include ../common.mk

CXX := g++
# Test-only binaries next to the main one. The four mutants must fail the
# check; the slow-warmup binary shows bench warmups stay outside the timed
# interval; the host test covers the oracle, the fixtures, and SHA-256. The
# optimized variant's test binaries carry an _optimized suffix, and it has two
# more mutants (5: the last chunk is never uploaded, 6: never downloaded).
SUFFIX := $(if $(filter optimized,$(VARIANT)),_optimized)
MUTANT_IDS := 1 2 3 4 $(if $(filter optimized,$(VARIANT)),5 6)
MUTANTS := $(foreach n,$(MUTANT_IDS),$(BUILD)/vecadd_product_mutant$(n)$(SUFFIX))
SLOW_WARMUP := $(BUILD)/vecadd_product_slow_warmup$(SUFFIX)
HOST_TEST := $(BUILD)/test_host
TEST_SCRATCH := $(BUILD)/test-output$(SUFFIX)
SRC_PATHS := $(addprefix $(APP_DIR)/,$(VARIANT_SRCS) $(VARIANT_HEADERS))

.PHONY: test

$(BUILD)/vecadd_product_mutant%$(SUFFIX): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_MUTANT=$* \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(SLOW_WARMUP): $(BIN)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(LP_FLAGS) -DLP_VECADD_TEST_SLOW_WARMUP \
	  $(addprefix $(APP_DIR)/,$(VARIANT_SRCS)) -o $@

$(HOST_TEST): $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/fixtures.cpp \
              $(APP_DIR)/oracle.cpp $(APP_DIR)/fixtures.h $(APP_DIR)/oracle.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -pthread -Wall -Wextra \
	  $(APP_DIR)/tests/test_host.cpp $(APP_DIR)/fixtures.cpp $(APP_DIR)/oracle.cpp -o $@

test: $(BIN) $(MUTANTS) $(SLOW_WARMUP) $(HOST_TEST)
	$(HOST_TEST)
	$(LOCK) $(APP_DIR)/tests/run_gpu_tests.sh $(BIN) $(BUILD) $(TEST_SCRATCH) $(VARIANT)
