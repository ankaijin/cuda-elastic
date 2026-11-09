# Optimization stages of loom-private/vecadd-product (README.md).
#
#   make build   nothing: there are no stage binaries
#   make test    the suite's own tests of the baseline and the optimized variant
#
# The optimized variant adds one optimization (the chunked multi-stream
# pipeline), so the only stages are the suite's baseline (stage 0) and
# optimized (stage 1) binaries, built by applications/loom_private/vecadd_product/Makefile.
# run.py builds them. The suite Makefile takes $(GPU_LOCK) around every GPU
# command; do not wrap these targets in another flock on the same file.
SHELL := /bin/bash
.DEFAULT_GOAL := build

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/../../..)
APP_DIR := $(ROOT)/applications/loom_private/vecadd_product

.PHONY: build test
build:
	@echo "vecadd_product: no stage binaries (stage 0 and stage 1 are the suite's)"

test:
	$(MAKE) -C $(APP_DIR) test VARIANT=baseline
	$(MAKE) -C $(APP_DIR) test VARIANT=optimized
