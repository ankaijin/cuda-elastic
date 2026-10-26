#!/usr/bin/env bash
# Tests for the optimized attention variant (run_tests.sh covers the
# baseline's cases on the optimized binary). Every kernel shape, split counts
# from 1 to many (IMU chunk and tile boundaries, both side streams), the exact
# row-maxima fallback, the compute path, the largest query extent (where the
# partial-buffer cap limits the splits), the largest context (where the
# per-split key bound raises them), invalid hook values (exit 2), and
# the skip-key mutant (exit 1). Each hook case also checks the setup line on
# stderr, so the hook is known to have applied.
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

# expect_exit <name> <code> <environment> <binary> <args...>; with
# EXPECT_SETUP set, the setup line on stderr must also contain it (so a hook
# is known to have applied).
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
  if [[ -n ${EXPECT_SETUP:-} ]] && ! grep -qF "$EXPECT_SETUP" "$OUT/stderr"; then ok=0; fi
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
shapes=(rows8_warps8_tile1024 rows8_warps8_tile512 rows8_warps4_tile512 rows4_warps8_tile256)
for shape in 0 1 2 3; do
  EXPECT_SETUP="${shapes[$shape]}, 1 context splits of 5120 keys" \
  expect_exit "shape${shape}_one_split"    0 "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=1" \
    "$HOOKS" --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
  EXPECT_SETUP="${shapes[$shape]}, 3 context splits of 1024 keys" \
  expect_exit "shape${shape}_three_splits" 0 "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=3" \
    "$HOOKS" --set query_tokens=1025 --set context_tokens=2817
  EXPECT_SETUP="${shapes[$shape]}, 52 context splits of 1280 keys" \
  expect_exit "shape${shape}_many_splits"  0 "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=64" \
    "$HOOKS" --set query_tokens=301 --set context_tokens=65537 --set input_variant=1
done
# The exact row-maxima fallback, in both launch paths.
EXPECT_SETUP="5 context splits of 13312 keys, exact row maxima" \
expect_exit exact_maximum_application 0 "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=5" \
  "$HOOKS" --set query_tokens=2000 --set context_tokens=65537
EXPECT_SETUP="1 context splits of 256 keys, exact row maxima, application runs the compute path" \
expect_exit exact_maximum_compute     0 "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_CHECK_COMPUTE=1" \
  "$HOOKS" --set query_tokens=300 --set context_tokens=7 --set input_variant=2
EXPECT_SETUP="rows4_warps8_tile256, 4 context splits of 256 keys, exact row maxima" \
expect_exit exact_maximum_small_rows  0 "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SHAPE=3" \
  "$HOOKS" --set query_tokens=37 --set context_tokens=1000
# The compute path (single stream, one launch over every split), checked.
EXPECT_SETUP="keys, application runs the compute path" \
expect_exit compute_path_default 0 "LP_ATTENTION_CHECK_COMPUTE=1" \
  "$HOOKS" --set query_tokens=2000 --set context_tokens=65537
EXPECT_SETUP="rows8_warps4_tile512, 4 context splits of 256 keys, application runs the compute path" \
expect_exit compute_path_splits  0 "LP_ATTENTION_CHECK_COMPUTE=1 LP_ATTENTION_SPLITS=7 LP_ATTENTION_SHAPE=2" \
  "$HOOKS" --set query_tokens=50 --set context_tokens=999
# The largest query extent: the partial-buffer cap allows eight splits.
EXPECT_SETUP="8 context splits of 256 keys" \
expect_exit largest_query_extent 0 "LP_ATTENTION_SPLITS=12" \
  "$HOOKS" --set query_tokens=4194304 --set context_tokens=2048
# The largest context: splits of at most 2^21 keys keep every float chain as
# short as the baseline's, even when a hook (or the cap) asks for fewer.
# Two splits of 2^23 keys failed the tolerance at input variant 2.
EXPECT_SETUP="8 context splits of 2097152 keys" \
expect_exit largest_context_extent 0 "LP_ATTENTION_SPLITS=2" \
  "$HOOKS" --set query_tokens=21 --set context_tokens=16777216 --set input_variant=2
# Tuning hook values outside their ranges, or not integers, are usage errors.
expect_exit invalid_shape          2 "LP_ATTENTION_SHAPE=9" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
expect_exit invalid_splits_zero    2 "LP_ATTENTION_SPLITS=0" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
expect_exit invalid_splits_large   2 "LP_ATTENTION_SPLITS=65536" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
expect_exit invalid_splits_text    2 "LP_ATTENTION_SPLITS=12x" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
expect_exit invalid_exact_maximum  2 "LP_ATTENTION_EXACT_MAXIMUM=2" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
expect_exit invalid_check_compute  2 "LP_ATTENTION_CHECK_COMPUTE=yes" \
  "$HOOKS" --set query_tokens=4 --set context_tokens=4
# The mutant drops one key per tile; the check must reject it.
expect_exit mutant_skip_key       1 "" "$MUTANT" --set query_tokens=7 --set context_tokens=300
expect_exit mutant_skip_key_large 1 "" "$MUTANT" --set query_tokens=129 --set context_tokens=5000

echo "attention optimized tests: $passed passed, $failed failed"
[[ $failed == 0 ]]
