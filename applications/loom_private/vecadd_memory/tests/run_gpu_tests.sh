#!/usr/bin/env bash
# GPU tests for loom-private/vecadd-memory, run by `make test` under the GPU
# lock (this script does not lock). Small --set dimensions only; none of them is
# an upstream smoke or validation configuration (4096/2/0, 65536/3/1,
# 131072/5/7).
#   run_gpu_tests.sh <vecadd_memory binary> <drop-last mutant binary> <scratch dir>
#                    [optimized <build dir>]
# For the optimized variant the build dir holds its extra test binaries: the
# tuning binary and three more mutants. The mutants and the tuning binary are
# built with -DLP_VECADD_TEST_TUNING, so LP_VECADD_CHUNKS and LP_VECADD_KERNEL
# set the chunk count and kernel shape in them; the suite binary ignores both.
set -uo pipefail

bin=$1
mutant=$2
scratch=$3
variant=${4:-baseline}
build=${5:-}
tuning="$build/vecadd_memory_tuning_optimized"
skip_upload_mutant="$build/vecadd_memory_skip_upload_mutant_optimized"
skip_download_mutant="$build/vecadd_memory_skip_download_mutant_optimized"
no_join_mutant="$build/vecadd_memory_no_join_mutant_optimized"
mkdir -p "$scratch"
failures=0

# expect <name> <exit code> <command...>
expect() {
  local name=$1 want=$2
  shift 2
  local log="$scratch/$name.log"
  "$@" >"$log" 2>&1
  local got=$?
  if [[ $got -eq $want ]]; then
    echo "PASS $name (exit $got)"
  else
    echo "FAIL $name: exit $got, expected $want"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  fi
}

# Tail handling: lengths of 1, below, at, and above the 256-thread block size,
# and a large length that is a multiple of neither 256 nor any power of two.
expect length-1 0 "$bin" --set length=1
expect length-2 0 "$bin" --set length=2
expect length-255 0 "$bin" --set length=255
expect length-256 0 "$bin" --set length=256
expect length-257 0 "$bin" --set length=257
expect length-1000003 0 "$bin" --set length=1000003 --json-out "$scratch/length-1000003.json"
expect length-1000003-direct 0 "$bin" --set length=1000003 --launch direct
expect length-5000-graph 0 "$bin" --set length=5000 --launch graph

# Other multipliers and biases, including negative and zero ones.
expect mult-7-bias-neg5 0 "$bin" --set length=70001 --set multiplier=7 --set bias=-5
expect mult-neg3-bias-11 0 "$bin" --set length=8191 --set multiplier=-3 --set bias=11
expect mult-0-bias-neg100 0 "$bin" --set length=4097 --set multiplier=0 --set bias=-100
expect mult-neg1 0 "$bin" --set length=9999 --set multiplier=-1

# Values right at the int32 limits are accepted and computed exactly.
expect int32-max-result 0 "$bin" --set length=2 --set multiplier=1 --set bias=2147483645
expect int32-min-rhs 0 "$bin" --set length=2 --set multiplier=0 --set bias=-2147483648
expect int32-large-multiplier 0 "$bin" --set length=3 --set multiplier=1073741822
# Largest multiplier that fits at the large length; the 64-bit checksum is far
# beyond int32.
expect large-length-mult-2047 0 "$bin" --set multiplier=2047

if grep -q '"overrides": {"length": 1000003}' "$scratch/length-1000003.json" &&
   grep -q '"status": "pass"' "$scratch/length-1000003.json" &&
   grep -q '"dimensions": {"length": 1000003, "multiplier": 2, "bias": 0}' "$scratch/length-1000003.json"; then
  echo "PASS result-record"
else
  echo "FAIL result-record"; cat "$scratch/length-1000003.json"
  failures=$((failures + 1))
fi

# A kernel that skips the last element must fail the oracle, at a tail length
# and at length 1 (where nothing is written and the poison survives).
expect drop-last-mutant-1000003 1 "$mutant" --set length=1000003
expect drop-last-mutant-1 1 "$mutant" --set length=1
if grep -q 'mismatch at index 1000002: expected 3000006' "$scratch/drop-last-mutant-1000003.log" &&
   grep -q 'checksum mismatch' "$scratch/drop-last-mutant-1000003.log"; then
  echo "PASS drop-last-mutant-reports-index-and-checksum"
else
  echo "FAIL drop-last-mutant-reports-index-and-checksum"; failures=$((failures + 1))
fi

# Invalid dimensions and usage exit 2.
expect zero-length 2 "$bin" --set length=0
expect negative-length 2 "$bin" --set length=-4
expect too-long 2 "$bin" --set length=134217729
expect mult-overflows-result 2 "$bin" --set length=3 --set multiplier=1073741823
expect mult-overflows-large 2 "$bin" --set multiplier=2048
expect bias-overflows-result 2 "$bin" --set length=2 --set bias=2147483646
expect bias-overflows-rhs 2 "$bin" --set length=2 --set multiplier=2 --set bias=2147483647
expect negative-overflow 2 "$bin" --set length=2 --set multiplier=-2 --set bias=-2147483648
expect multiplier-not-int32 2 "$bin" --set multiplier=2147483648
expect bias-not-int32 2 "$bin" --set bias=-2147483649
expect non-integer 2 "$bin" --set length=abc
expect unknown-key 2 "$bin" --set length=16 --set width=3
expect unknown-preset 2 "$bin" --preset smoke
expect bench-with-set 2 "$bin" --action bench --launch direct --set length=1024
if grep -q 'length must be at least 1' "$scratch/zero-length.log" &&
   grep -q 'overflow int32' "$scratch/mult-overflows-result.log"; then
  echo "PASS validation-messages"
else
  echo "FAIL validation-messages"; failures=$((failures + 1))
fi

if [[ $variant == optimized ]]; then
  # The suite binary has no tuning overrides: it ignores the variables, so a
  # stray value cannot change a measured configuration.
  expect suite-binary-ignores-tuning-env 0 env LP_VECADD_CHUNKS=0 LP_VECADD_KERNEL=9 "$bin" \
    --set length=1000
  # Chunking: LP_VECADD_CHUNKS forces the chunk count, so small lengths cross
  # chunk boundaries, including more chunks than vectors, chunk starts that
  # are not multiples of 256 elements, and a last chunk with a scalar tail.
  # LP_VECADD_KERNEL forces each per-GPU kernel shape (1, 4 and 2 vectors per
  # thread, streaming hints or not) on this GPU. Every combination runs
  # Direct and Graph.
  for kernel in 0 1 2; do
    for chunks in 1 3 8 64; do
      for length in 1 5 257 5000 1000003 1000004; do
        expect "chunks-$chunks-kernel-$kernel-length-$length" 0 \
          env LP_VECADD_CHUNKS=$chunks LP_VECADD_KERNEL=$kernel "$tuning" --set length=$length
      done
    done
    expect "kernel-$kernel-large" 0 env LP_VECADD_KERNEL=$kernel "$tuning"
    expect "kernel-$kernel-mult-2047" 0 env LP_VECADD_KERNEL=$kernel "$tuning" --set multiplier=2047
  done
  # Out-of-range overrides are rejected, not ignored.
  expect tuning-chunks-0 2 env LP_VECADD_CHUNKS=0 "$tuning" --set length=16
  expect tuning-chunks-65 2 env LP_VECADD_CHUNKS=65 "$tuning" --set length=16
  expect tuning-chunks-text 2 env LP_VECADD_CHUNKS=abc "$tuning" --set length=16
  expect tuning-kernel-3 2 env LP_VECADD_KERNEL=3 "$tuning" --set length=16
  expect tuning-kernel-negative 2 env LP_VECADD_KERNEL=-1 "$tuning" --set length=16
  # The drop-last mutant fails in every kernel shape and chunk count, at a
  # length whose last element is in the 16-byte path and at one whose last
  # element is in the tail.
  for kernel in 0 1 2; do
    for chunks in 1 8 64; do
      expect "drop-last-mutant-kernel-$kernel-chunks-$chunks-1000000" 1 \
        env LP_VECADD_KERNEL=$kernel LP_VECADD_CHUNKS=$chunks "$mutant" --set length=1000000
      expect "drop-last-mutant-kernel-$kernel-chunks-$chunks-1000003" 1 \
        env LP_VECADD_KERNEL=$kernel LP_VECADD_CHUNKS=$chunks "$mutant" --set length=1000003
    done
  done
  # Only the last element is dropped, however many chunks there are.
  expect drop-last-mutant-64-chunks 1 env LP_VECADD_CHUNKS=64 "$mutant" --set length=1000003
  if grep -q 'mismatch at index 1000002: expected 3000006' "$scratch/drop-last-mutant-64-chunks.log" &&
     grep -q '1 of 1000003 results differ' "$scratch/drop-last-mutant-64-chunks.log"; then
    echo "PASS drop-last-mutant-drops-only-the-last-element"
  else
    echo "FAIL drop-last-mutant-drops-only-the-last-element"; failures=$((failures + 1))
  fi
  # A last chunk whose upload or download is skipped must fail the oracle, with
  # one chunk and with many (the poisoned device inputs and host outputs show).
  for chunks in 1 4 64; do
    expect "skip-upload-mutant-chunks-$chunks" 1 env LP_VECADD_CHUNKS=$chunks "$skip_upload_mutant" --set length=100003
    expect "skip-download-mutant-chunks-$chunks" 1 env LP_VECADD_CHUNKS=$chunks "$skip_download_mutant" --set length=100003
  done
  expect skip-upload-mutant-large 1 "$skip_upload_mutant"
  expect skip-download-mutant-large 1 "$skip_download_mutant"
  # Fork and join ordering. Without the join, the harness stream finishes
  # before the chunks: Direct reads unfinished results and fails the check, and
  # the Graph capture ends with unjoined work (CUDA error, exit 70).
  expect no-join-mutant-direct 1 "$no_join_mutant" --launch direct
  expect no-join-mutant-graph 70 "$no_join_mutant" --launch graph
  # Out-of-bounds accesses, under compute-sanitizer.
  sanitizer=$(command -v compute-sanitizer || true)
  if [[ -n $sanitizer ]]; then
    for chunks in 1 3 64; do
      expect "memcheck-chunks-$chunks" 0 env LP_VECADD_CHUNKS=$chunks "$sanitizer" \
        --error-exitcode 9 --tool memcheck "$tuning" --set length=100003 --launch direct
    done
    expect memcheck-length-5 0 "$sanitizer" --error-exitcode 9 --tool memcheck "$bin" \
      --set length=5 --launch direct
  else
    echo "SKIP memcheck (compute-sanitizer not found)"
  fi
  # Bench path works at the large size (it does not run the oracle).
  expect bench-direct 0 "$bin" --action bench --launch direct --iterations 2 --warmup 1
  expect bench-graph 0 "$bin" --action bench --launch graph --iterations 2 --warmup 1
  if grep -q '"variant": "optimized"' "$scratch/bench-graph.log"; then
    echo "PASS bench-record-names-variant"
  else
    echo "FAIL bench-record-names-variant"; failures=$((failures + 1))
  fi
fi

if [[ $failures -ne 0 ]]; then
  echo "GPU tests FAILED ($failures)"
  exit 1
fi
echo "GPU tests passed"
