#!/usr/bin/env bash
# Tests for experiment B (TMA tile loads), run by `make test`. Each GPU command
# takes $GPU_LOCK itself.
#   run_tests.sh <build dir> <applications/loom_private/attention>
# 1. The production test scripts, unchanged, on every case: run_tests.sh (small
#    extents against the oracle, invalid dimensions) on the suite binary, and
#    run_tests_optimized.sh on the case's hooks and skip-key mutant builds
#    (every shape, 1 to 52 splits, the exact-maximum fallback in both launch
#    paths, the compute path, the largest query extent, the largest context
#    at input variant 2, invalid hooks, and the mutant).
# 2. Output digests: each TMA case must match the control bitwise on partial
#    tiles, partial query groups, one key, many splits, every shape, both
#    attention passes, the compute path, and the largest context.
# 3. The no-full-wait mutants, whose compute warps read a tile without waiting
#    for its copy, must be rejected: mode 1 fails the check (exit 1); in mode 2
#    the compute warps finish and the block exits while its copies are still
#    in flight, so the launch fails (exit 70).
# 4. compute-sanitizer memcheck, racecheck, and synccheck on the TMA cases.
set -uo pipefail
build=${1:?usage: run_tests.sh <build dir> <attention dir>}
at=${2:?usage: run_tests.sh <build dir> <attention dir>}
LOCK=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
passed=0
failed=0
pass() { passed=$((passed + 1)); echo "PASS $1"; }
fail() { failed=$((failed + 1)); echo "FAIL $1"; }

for case in control tma tma_producer; do
  if GPU_LOCK=$LOCK "$at/run_tests.sh" "$build/$case" >"$OUT/base.log" 2>&1; then
    pass "production run_tests.sh on $case: $(tail -n 1 "$OUT/base.log")"
  else
    fail "production run_tests.sh on $case"; grep FAIL "$OUT/base.log" | sed 's/^/    /'
  fi
  if GPU_LOCK=$LOCK "$at/run_tests_optimized.sh" "$build/${case}_hooks" \
       "$build/${case}_skip_key_mutant" >"$OUT/opt.log" 2>&1; then
    pass "production run_tests_optimized.sh on $case: $(tail -n 1 "$OUT/opt.log")"
  else
    fail "production run_tests_optimized.sh on $case"; grep FAIL "$OUT/opt.log" | sed 's/^/    /'
  fi
done

# digests <name> <environment> <args...>: every case passes, its four
# invocations agree, and each TMA case's digest equals the control's.
digest() {
  python3 -c 'import json, sys
r = json.load(open(sys.argv[1]))
d = {c["digest"] for c in r["checks"]}
print(d.pop() if r["status"] == "pass" and len(d) == 1 else "invalid")' "$1"
}
digests() {
  local name=$1 environment=$2
  shift 2
  local want="" ok=1 got
  for case in control tma tma_producer; do
    env $environment flock "$LOCK" "$build/${case}_hooks" "$@" --json-out "$OUT/$case.json" \
      >"$OUT/$case.log" 2>&1 || ok=0
    got=$(digest "$OUT/$case.json" 2>/dev/null || echo missing)
    [[ $got == invalid || $got == missing ]] && ok=0
    [[ -z $want ]] && want=$got
    [[ $got == "$want" ]] || ok=0
    name="$name $case=$got"
  done
  if [[ $ok == 1 ]]; then pass "$name"; else fail "$name"; fi
}
digests "partial-tile-and-group" "LP_ATTENTION_SHAPE=0 LP_ATTENTION_SPLITS=1" \
  --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
digests "three-splits" "LP_ATTENTION_SHAPE=0 LP_ATTENTION_SPLITS=3" \
  --set query_tokens=1025 --set context_tokens=2817
digests "many-splits" "LP_ATTENTION_SHAPE=0 LP_ATTENTION_SPLITS=64" \
  --set query_tokens=301 --set context_tokens=65537 --set input_variant=1
for shape in 1 2 3; do
  digests "shape$shape" "LP_ATTENTION_SHAPE=$shape LP_ATTENTION_SPLITS=3" \
    --set query_tokens=37 --set context_tokens=1000
done
digests "one-key" "" --set query_tokens=1 --set context_tokens=1
digests "both-passes" "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=5" \
  --set query_tokens=2000 --set context_tokens=65537
digests "both-passes-compute-path" "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_CHECK_COMPUTE=1" \
  --set query_tokens=300 --set context_tokens=7 --set input_variant=2
digests "compute-path" "LP_ATTENTION_CHECK_COMPUTE=1" \
  --set query_tokens=2000 --set context_tokens=65537
digests "largest-context" "LP_ATTENTION_SPLITS=2" \
  --set query_tokens=21 --set context_tokens=16777216 --set input_variant=2

for case in tma tma_producer; do
  flock "$LOCK" "$build/${case}_no_full_wait_mutant" --set query_tokens=2000 \
    --set context_tokens=65537 >"$OUT/mutant.log" 2>&1
  code=$?
  want=1
  [[ $case == tma_producer ]] && want=70
  if [[ $code == "$want" ]]; then pass "$case no-full-wait mutant is rejected (exit $code)"
  else fail "$case no-full-wait mutant: exit $code, want $want"; fi
  for tool in memcheck racecheck synccheck; do
    if flock "$LOCK" compute-sanitizer --tool "$tool" --error-exitcode 9 "$build/$case" \
         --set query_tokens=129 --set context_tokens=2817 >"$OUT/sanitizer.log" 2>&1; then
      pass "$case compute-sanitizer $tool"
    else
      fail "$case compute-sanitizer $tool"; tail -n 12 "$OUT/sanitizer.log" | sed 's/^/    /'
    fi
  done
done

echo "tma tests: $passed passed, $failed failed"
[[ $failed == 0 ]]
