#!/usr/bin/env bash
# Tests for experiment C (lower precision classifiers), run by `make test`.
# Each GPU command takes $GPU_LOCK itself.
#   run_tests.sh <build dir>
# 1. The host format test: the reference's decoding and sub-byte packing
#    against CUTLASS's types, ties, saturation, zero and tiny blocks.
# 2. Each case's device quantizer against the host reference on crafted rows
#    (--selftest), including 130 rows (a second scale-factor tile).
# 3. The implementation check of every case in Direct and Graph mode on the
#    three dense fixtures, a synthetic model with ragged tiles (batch 100,
#    vocab 200), and a one-sequence, two-step model.
# 4. Mutants (rounding, scale layout, alpha) must fail the self-test and the
#    check.
# 5. compute-sanitizer memcheck of each low precision case on a fixture.
set -uo pipefail
build=${1:?usage: run_tests.sh <build dir>}
LOCK=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
passed=0
failed=0

# expect <name> <exit code> <command...>
expect() {
  local name=$1 want=$2
  shift 2
  flock "$LOCK" "$@" >"$OUT/log" 2>&1
  local got=$?
  if [[ $got == "$want" ]]; then
    passed=$((passed + 1)); echo "PASS $name (exit $got)"
  else
    failed=$((failed + 1)); echo "FAIL $name (exit $got, want $want)"
    grep -E "FAIL|mismatch|failed|error" "$OUT/log" | head -n 6 | sed 's/^/    /'
  fi
}

expect host-formats 0 "$build/test_formats"
for case in fp32 bf16 mxfp6 nvfp4; do
  expect "$case-selftest" 0 "$build/$case" --selftest
  for fixture in 1 2 3; do
    expect "$case-fixture$fixture" 0 "$build/$case" --set fixture=$fixture
  done
  expect "$case-ragged" 0 "$build/$case" --set batch=100 --set width=1024 --set vocab=200
  expect "$case-one-sequence" 0 "$build/$case" --set batch=1 --set width=256 --set vocab=128 \
    --set steps=2
done
expect fp32-odd-width 0 "$build/fp32" --set batch=5 --set width=1001 --set vocab=999
expect bf16-unaligned-width 2 "$build/bf16" --set batch=5 --set width=1001 --set vocab=999
for mutant in bf16_mutant1 mxfp6_mutant1 mxfp6_mutant2 nvfp4_mutant1 nvfp4_mutant2 nvfp4_mutant3; do
  expect "$mutant-selftest" 1 "$build/$mutant" --selftest
  expect "$mutant-check" 1 "$build/$mutant" --set fixture=1
done
for case in bf16 mxfp6 nvfp4; do
  expect "$case-memcheck" 0 compute-sanitizer --tool memcheck --error-exitcode 9 \
    "$build/$case" --set fixture=2 --launch direct
done

echo "lowprec tests: $passed passed, $failed failed"
[[ $failed == 0 ]]
