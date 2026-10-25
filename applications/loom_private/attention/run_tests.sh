#!/usr/bin/env bash
# Small tests for the attention application: --set extents checked against the
# CPU oracle (Direct and Graph, two invocations each), and invalid dimensions
# or usage that must exit 2. None reproduces an upstream smoke or validation
# configuration (upstream uses equal extents 4/variant 0, 64/variant 1,
# 96/variant 2). Usage: run_tests.sh <binary>; GPU_LOCK names the lock file.
set -u
BIN=${1:?usage: run_tests.sh <binary>}
LOCK=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
passed=0
failed=0

# expect_exit <name> <code> <args...>
expect_exit() {
  local name=$1 want=$2
  shift 2
  flock "$LOCK" "$BIN" "$@" >"$OUT/stdout" 2>"$OUT/stderr"
  local got=$?
  local ok=1
  [[ $got == "$want" ]] || ok=0
  if [[ $want == 0 ]] && ! grep -q '"status": "pass"' "$OUT/stdout"; then ok=0; fi
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

# Oracle checks. q != c in most cases; context extents straddle the
# 256-thread attention stride and the 1024-thread statistics stride.
expect_exit single_token_each        0 --set query_tokens=1 --set context_tokens=1
expect_exit few_queries_long_context 0 --set query_tokens=7 --set context_tokens=300
expect_exit many_queries_short_ctx   0 --set query_tokens=300 --set context_tokens=7
expect_exit equal_extent_variant1    0 --set query_tokens=37 --set context_tokens=37 --set input_variant=1
expect_exit unequal_variant2         0 --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
expect_exit stride_edges             0 --set query_tokens=1025 --set context_tokens=257
expect_exit medium_unequal           0 --set query_tokens=2000 --set context_tokens=65537
expect_exit direct_only              0 --launch direct --set query_tokens=50 --set context_tokens=999
expect_exit graph_only               0 --launch graph --set query_tokens=50 --set context_tokens=999

# Invalid dimensions and usage: exit 2 before any GPU work.
expect_exit zero_queries             2 --set query_tokens=0
expect_exit negative_context         2 --set context_tokens=-5
expect_exit too_many_queries         2 --set query_tokens=4194305
expect_exit too_much_context         2 --set context_tokens=16777217
expect_exit bad_variant              2 --set input_variant=3
expect_exit non_integer              2 --set query_tokens=12x
expect_exit unknown_key              2 --set tokens=4
expect_exit unknown_preset           2 --preset smoke
expect_exit bench_with_set           2 --action bench --launch direct --set query_tokens=4

echo "attention tests: $passed passed, $failed failed"
[[ $failed == 0 ]]
