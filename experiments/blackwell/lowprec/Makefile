# Experiment C: lower precision classifiers in loom-private/llama2c-decoder.
#   make cutlass | make build | make test | make clean      (see README.md)
# Cases: fp32 (control: the production classifier), bf16, mxfp6, nvfp4, all
# built from decoder_lowprec.cu for sm_120f (SASS only) with the production
# flags. spec.json is the experimental specification whose hash the binaries
# embed. CUTLASS 4.8.0 is pinned by commit and cached outside the repository.
HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DEC := $(abspath $(HERE)/../../../applications/loom_private/decoder)
CUTLASS_DIR ?= $(HOME)/.cache/cuda-elastic/cutlass-4.8.0
CUTLASS_COMMIT := 098de2a652cf8f00fd70b2df54051c7eccbb855a
EXPERIMENT := lowprec
CASES := fp32 bf16 mxfp6 nvfp4
CASE_ID_fp32 := 0
CASE_ID_bf16 := 1
CASE_ID_mxfp6 := 2
CASE_ID_nvfp4 := 3
SRCS := $(HERE)/decoder_lowprec.cu $(HERE)/reference.cpp $(DEC)/oracle.cpp
HEADERS := $(HERE)/reference.h $(DEC)/decoder_optimized.cu $(DEC)/oracle.h
CUTLASS_FLAGS := -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include \
                 --expt-relaxed-constexpr
$(foreach c,$(CASES),$(eval $(c)_SRCS := $(SRCS)))
$(foreach c,$(CASES),$(eval $(c)_HEADERS := $(HEADERS)))
$(foreach c,$(CASES),$(eval $(c)_ARCH := sm_120f))
$(foreach c,$(CASES),$(eval $(c)_ARCHFLAGS := -gencode arch=compute_120f,code=sm_120f))
$(foreach c,$(CASES),$(eval $(c)_SPEC := $(HERE)/spec.json))
$(foreach c,$(CASES),$(eval $(c)_APP := decoder))
$(foreach c,$(CASES),$(eval $(c)_FLAGS := -Xcompiler -pthread -I$(DEC) -I$(HERE) \
  $(CUTLASS_FLAGS) -DLP_LOWPREC_CASE=$(CASE_ID_$(c))))
include ../common.mk

# Every case binary needs the pinned CUTLASS checkout (order-only).
$(addprefix $(BUILD)/,$(CASES)): | cutlass-pin
.PHONY: cutlass cutlass-pin
cutlass-pin:
	@test "$$(git -C $(CUTLASS_DIR) rev-parse HEAD 2>/dev/null)" = "$(CUTLASS_COMMIT)" || \
	  { echo "CUTLASS_DIR=$(CUTLASS_DIR) is not CUTLASS $(CUTLASS_COMMIT); run make cutlass" >&2; exit 2; }
	@test -z "$$(git -C $(CUTLASS_DIR) status --porcelain)" || \
	  { echo "CUTLASS_DIR=$(CUTLASS_DIR) has local changes" >&2; exit 2; }
cutlass:
	test -d $(CUTLASS_DIR) || git clone --no-checkout https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR)
	git -C $(CUTLASS_DIR) fetch --depth 1 origin $(CUTLASS_COMMIT)
	git -C $(CUTLASS_DIR) -c advice.detachedHead=false checkout $(CUTLASS_COMMIT)

# Test builds: the host format test (the reference against CUTLASS's own
# types and sub-byte packing) and the mutants, which the checks must reject:
# 1 rounding (every low precision case), 2 scale layout and 3 alpha (block
# scaled cases; 3 for NVFP4 only).
MUTANTS := bf16_mutant1 mxfp6_mutant1 mxfp6_mutant2 nvfp4_mutant1 nvfp4_mutant2 nvfp4_mutant3
TEST_BINS := $(BUILD)/test_formats $(addprefix $(BUILD)/,$(MUTANTS))
$(BUILD)/test_formats: $(HERE)/tests/test_formats.cu $(HERE)/reference.cpp $(HERE)/reference.h \
                       $(COMMON_DEPS) | cutlass-pin
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(BW_REQUIRED_NVCCFLAGS) -gencode arch=compute_120f,code=sm_120f \
	  -I$(HERE) $(CUTLASS_FLAGS) $< $(HERE)/reference.cpp -o $@
define mutant_rule
$(BUILD)/$(1)_mutant$(2): $(SRCS) $(HEADERS) $(COMMON_DEPS) | cutlass-pin
	$$(call compile_case,$(1),$$@,-DLP_LOWPREC_MUTANT=$(2))
endef
$(foreach m,$(MUTANTS),$(eval $(call mutant_rule,$(word 1,$(subst _mutant, ,$(m))),$(word 2,$(subst _mutant, ,$(m))))))

.PHONY: test
test: build $(TEST_BINS)
	GPU_LOCK=$(GPU_LOCK) $(HERE)/tests/run_tests.sh $(BUILD)
