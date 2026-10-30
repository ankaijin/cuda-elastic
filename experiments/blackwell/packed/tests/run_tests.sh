#!/usr/bin/env bash
# Tests for experiment A (packed byte arithmetic), run by `make test`. Each GPU
# command takes $GPU_LOCK itself.
#   run_tests.sh <build dir> <scratch dir>
# The exhaustive lane test of each case's add_bytes; the production tail
# cases (record counts around block, period and word boundaries) in Direct and
# Graph for both cases, with the canary check; the saturating mutant, which
# the lane test and the check must reject; and compute-sanitizer memcheck of
# the candidate.
set -uo pipefail
build=${1:?usage: run_tests.sh <build dir> <scratch dir>}
scratch=${2:?usage: run_tests.sh <build dir> <scratch dir>}
LOCK=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
mkdir -p "$scratch"
passed=0
failed=0

# expect <name> <exit code> <command...>
expect() {
  local name=$1 want=$2
  shift 2
  flock "$LOCK" "$@" >"$scratch/$name.log" 2>&1
  local got=$?
  if [[ $got == "$want" ]]; then
    passed=$((passed + 1))
    echo "PASS $name (exit $got)"
  else
    failed=$((failed + 1))
    echo "FAIL $name (exit $got, want $want)"
    tail -n 8 "$scratch/$name.log" | sed 's/^/    /'
  fi
}

for case in control add_u8x4; do
  expect "lanes-$case" 0 "$build/test_lanes_$case"
  for n in 1 2 3 7 63 64 255 256 257 1000 4097 65537 1000003; do
    expect "$case-records-$n" 0 "$build/$case" --set records=$n
  done
  expect "$case-records-257-direct" 0 "$build/$case" --set records=257 --launch direct
  expect "$case-records-257-graph" 0 "$build/$case" --set records=257 --launch graph
  if grep -q "0 canary bytes damaged" "$scratch/$case-records-1000.log"; then
    passed=$((passed + 1)); echo "PASS $case-canary-reported"
  else
    failed=$((failed + 1)); echo "FAIL $case-canary-reported"
  fi
done
expect lanes-mutant-saturate 1 "$build/test_lanes_mutant_saturate"
expect mutant-saturate-records-1000 1 "$build/mutant_saturate_add_u8x4" --set records=1000
expect memcheck-add_u8x4-1000 0 compute-sanitizer --tool memcheck --error-exitcode 9 \
  "$build/add_u8x4" --set records=1000
expect memcheck-add_u8x4-65537 0 compute-sanitizer --tool memcheck --error-exitcode 9 \
  "$build/add_u8x4" --set records=65537

echo "packed tests: $passed passed, $failed failed"
[[ $failed == 0 ]]
