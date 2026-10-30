# Experiment A: packed byte arithmetic in loom-private/vecadd-product.
#   make build | make test | make clean      (see ../README.md)
# Cases: control is the production optimized source, unchanged; add_u8x4 is
# vecadd_product_packed.cu (add_u8x4.diff is its exact difference). Both are
# built for sm_120f (SASS only), the family target that add.u8x4 needs, with
# the production suite's other flags.
HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VP := $(abspath $(HERE)/../../../applications/loom_private/vecadd_product)
EXPERIMENT := packed
CASES := control add_u8x4
COMMON_SRCS := $(VP)/fixtures.cpp $(VP)/oracle.cpp
COMMON_HEADERS := $(VP)/fixtures.h $(VP)/oracle.h

control_SRCS := $(VP)/vecadd_product_optimized.cu $(COMMON_SRCS)
add_u8x4_SRCS := $(HERE)/vecadd_product_packed.cu $(COMMON_SRCS)
$(foreach c,$(CASES),$(eval $(c)_HEADERS := $(COMMON_HEADERS)))
$(foreach c,$(CASES),$(eval $(c)_ARCH := sm_120f))
$(foreach c,$(CASES),$(eval $(c)_ARCHFLAGS := -gencode arch=compute_120f,code=sm_120f))
$(foreach c,$(CASES),$(eval $(c)_SPEC := $(VP)/contract.json))
$(foreach c,$(CASES),$(eval $(c)_APP := vecadd_product))
$(foreach c,$(CASES),$(eval $(c)_FLAGS := -I$(VP)))
include ../common.mk

# Test builds: the exhaustive lane test of each case's add_bytes, the
# candidate's saturating-add mutant (its lane test and its check must fail),
# and nothing else; test binaries carry a test_ or mutant_ prefix.
TEST_BINS := $(BUILD)/test_lanes_control $(BUILD)/test_lanes_add_u8x4 \
             $(BUILD)/test_lanes_mutant_saturate $(BUILD)/mutant_saturate_add_u8x4
LANES = $(NVCC) $(NVCCFLAGS) $(BW_REQUIRED_NVCCFLAGS) \
        -gencode arch=compute_120f,code=sm_120f -I$(LP_ROOT)/common -I$(VP) \
        -DLP_APP=\"vecadd_product\" -DLP_CONTRACT_SHA256=\"test\" -DLP_VARIANT=\"test\"

$(BUILD)/test_lanes_control: $(HERE)/tests/test_lanes.cu $(control_SRCS) $(COMMON_DEPS)
	@mkdir -p $(BUILD)
	$(LANES) -DPACKED_SOURCE=\"$(VP)/vecadd_product_optimized.cu\" $< $(COMMON_SRCS) -o $@
$(BUILD)/test_lanes_add_u8x4: $(HERE)/tests/test_lanes.cu $(add_u8x4_SRCS) $(COMMON_DEPS)
	@mkdir -p $(BUILD)
	$(LANES) -DPACKED_SOURCE=\"$(HERE)/vecadd_product_packed.cu\" $< $(COMMON_SRCS) -o $@
$(BUILD)/test_lanes_mutant_saturate: $(HERE)/tests/test_lanes.cu $(add_u8x4_SRCS) $(COMMON_DEPS)
	@mkdir -p $(BUILD)
	$(LANES) -DLP_VECADD_MUTANT=1 -DPACKED_SOURCE=\"$(HERE)/vecadd_product_packed.cu\" $< \
	  $(COMMON_SRCS) -o $@
$(BUILD)/mutant_saturate_add_u8x4: $(add_u8x4_SRCS) $(COMMON_DEPS)
	$(call compile_case,add_u8x4,$@,-DLP_VECADD_MUTANT=1)

.PHONY: test
test: build $(TEST_BINS)
	GPU_LOCK=$(GPU_LOCK) $(HERE)/tests/run_tests.sh $(BUILD) $(BUILD)/test-output
