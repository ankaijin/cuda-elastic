# loom-private/multisensor-attention. Targets come from ../common.mk:
#   make build | make check-large [CHECK_JSON=path] | make test | make clean
APP := attention
SRCS := attention.cu oracle.cpp
HEADERS := oracle.h
EXTRA_NVCCFLAGS := -Xcompiler -pthread -lpthread
include ../common.mk

# Small --set cases against the oracle plus invalid-dimension exits. Each GPU
# run takes $(GPU_LOCK) inside the script.
test: $(BIN)
	GPU_LOCK=$(GPU_LOCK) $(APP_DIR)/run_tests.sh $(BIN)
