# loom-private/llama2c-decoder. See README.md.
APP := decoder
SRCS := decoder.cu oracle.cpp
HEADERS := oracle.h
EXTRA_NVCCFLAGS := -Xcompiler -pthread
include ../common.mk

.PHONY: test
# Small --set shapes, invalid dimensions, and injected faults; never the
# large check. The script runs inside one GPU lock and takes no lock itself.
test: $(BIN)
	$(LOCK) bash $(APP_DIR)/tests/run_tests.sh $(BIN)
