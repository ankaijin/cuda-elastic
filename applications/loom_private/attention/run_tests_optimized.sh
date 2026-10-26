#!/usr/bin/env bash
# Tests for the optimized attention variant (run_tests.sh covers the
# baseline's cases on the optimized binary). Every kernel shape, split counts
# from 1 to many (IMU chunk and tile boundaries, both side streams), the exact
# row-maxima fallback, the compute path, the largest query extent (where the
# partial-buffer cap limits the splits), an invalid tuning (exit 2), and the
# skip-key mutant (exit 1).
# Usage: run_tests_optimized.sh <hooks binary> <mutant binary>; GPU_LOCK names
# the lock file.
set -u
HOOKS=${1:?usage: run_tests_optimized.sh <hooks binary> <mutant binary>}
MUTANT=${2:?usage: run_tests_optimized.sh <hooks binary> <mutant binary>}
LOCK=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
passed=0
failed=0

# expect_exit <name> <code> <environment> <binary> <args...>
expect_exit() {
  local name=$1 want=$2 environment=$3 binary=$4
  shift 4
  env $environment flock "$LOCK" "$binary" "$@" >"$OUT/stdout" 2>"$OUT/stderr"
  local got=$?
  local ok=1
  [[ $got == "$want" ]] || ok=0
  if [[ $want == 0 ]] && ! grep -q '"status": "pass"' "$OUT/stdout"; then ok=0; fi
  if [[ $want == 1 ]] && ! grep -q '"status": "fail"' "$OUT/stdout"; then ok=0; fi
  if [[ $want == 2 ]] && ! grep -q . "$OUT/stderr"; then ok=0; fi
  if [[ $ok == 1 ]]; then
    passed=$((passed + 1))
    echo "PASS $name (exit $got)"
  else
    failed=$((failed + 1))
    echo "FAIL $name (exit $got, want $want)"
    tail -n 8 "$OUT/stderr" | sed 's/^/    /'
  fi
}

# Every compiled kernel shape, with partial row groups, partial tiles, and
# one, a few, and many context splits.
for shape in 0 1 2 3; do
  expect_exit "shape${shape}_one_split"    0 "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=1" \
    "$HOOKS" --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
  expect_exit "shape${shape}_three_splits" 0 "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=3" \
    "$HOOKS" --set query_tokens=1025 --set context_tokens=2817
  expect_exit "shape${shape}_many_splits"  0 "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=64" \
    "$HOOKS" --set query_tokens=301 --set context_tokens=65537 --set input_variant=1
done
# The exact row-maxima fallback, in both launch paths.
expect_exit exact_maximum_application 0 "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=5" \
  "$HOOKS" --set query_tokens=2000 --set context_tokens=65537
expect_exit exact_maximum_compute     0 "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_CHECK_COMPUTE=1" \
  "$HOOKS" --set query_tokens=300 --set context_tokens=7 --set input_variant=2
expect_exit exact_maximum_small_rows  0 "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SHAPE=3" \
  "$HOOKS" --set query_tokens=37 --set context_tokens=1000
# The compute path (single stream, one launch over every split), checked.
expect_exit compute_path_default 0 "LP_ATTENTION_CHECK_COMPUTE=1" \
  "$HOOKS" --set query_tokens=2000 --set context_tokens=65537
expect_exit compute_path_splits  0 "LP_ATTENTION_CHECK_COMPUTE=1 LP_ATTENTION_SPLITS=7 LP_ATTENTION_SHAPE=2" \
  "$HOOKS" --set query_tokens=50 --set context_tokens=999
# The largest query extent: the partial-buffer cap allows two splits.
expect_exit largest_query_extent 0 "" \
  "$HOOKS" --set query_tokens=4194304 --set context_tokens=1024
# An out-of-range tuning hook is a usage error.
expect_exit invalid_shape 2 "LP_ATTENTION_SHAPE=9" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
# The mutant drops one key per tile; the check must reject it.
expect_exit mutant_skip_key       1 "" "$MUTANT" --set query_tokens=7 --set context_tokens=300
expect_exit mutant_skip_key_large 1 "" "$MUTANT" --set query_tokens=129 --set context_tokens=5000

echo "attention optimized tests: $passed passed, $failed failed"
[[ $failed == 0 ]]
