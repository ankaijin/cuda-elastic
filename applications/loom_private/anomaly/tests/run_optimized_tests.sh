#!/usr/bin/env bash
# Extra tests of the optimized variant, run by `make test VARIANT=optimized`
# after run_tests.sh. Each GPU run takes $GPU_LOCK here.
#   run_optimized_tests.sh <optimized binary> <test binary dir> "<chunk counts>"
#                          <asset cache> <scratch dir>
set -uo pipefail

bin=${1:?usage: run_optimized_tests.sh <binary> <dir> <chunks> <cache> <scratch>}
dir=${2:?usage: run_optimized_tests.sh <binary> <dir> <chunks> <cache> <scratch>}
chunks=${3:?usage: run_optimized_tests.sh <binary> <dir> <chunks> <cache> <scratch>}
cache=${4:?usage: run_optimized_tests.sh <binary> <dir> <chunks> <cache> <scratch>}
scratch=${5:?usage: run_optimized_tests.sh <binary> <dir> <chunks> <cache> <scratch>}/optimized
lock=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
rm -rf "$scratch"
mkdir -p "$scratch"
passed=0
failed=0

# expect <name> <exit code> <output pattern or ''> <command...>: stdout and
# stderr are both searched; exit 0 also needs a passing result record or, for
# test_requant, no FAIL line.
expect() {
  local name=$1 want=$2 pattern=$3
  shift 3
  local log="$scratch/$name.log"
  flock "$lock" "$@" >"$log" 2>&1
  local got=$?
  local ok=1
  [[ $got -eq $want ]] || ok=0
  if [[ $want -eq 0 ]] && grep -q '^RESULT' "$log" && ! grep -q '"status": "pass"' "$log"; then
    ok=0
  fi
  if [[ -n $pattern ]] && ! grep -q -- "$pattern" "$log"; then ok=0; fi
  if [[ $ok -eq 1 ]]; then
    passed=$((passed + 1))
    echo "PASS $name (exit $got)"
  else
    failed=$((failed + 1))
    echo "FAIL $name: exit $got, expected $want${pattern:+, output matching '$pattern'}"
    tail -n 8 "$log" | sed 's/^/    /'
  fi
}

# --- The binary32 fast paths against the baseline arithmetic: every finite
# input value and every accumulator within the contract's bound, for every
# output of the pinned model. A mutant without the safety margin must fail.
expect fast-path-exactness 0 'PASS requantization' "$dir/test_requant" "$cache"
# Both halves must catch it: the requantization half shows that the
# half-integer margin matters (the input half also catches the overflow that
# the margin guards against).
expect fast-path-no-margin-mutant 1 'FAIL requantization' \
  "$dir/test_requant_no_margin_mutant" "$cache"
if grep -q 'FAIL input quantization' "$scratch/fast-path-no-margin-mutant.log"; then
  passed=$((passed + 1)); echo "PASS fast-path-no-margin-mutant-input"
else
  failed=$((failed + 1)); echo "FAIL fast-path-no-margin-mutant-input"
fi

# --- Tile edges with this GPU's configuration: one sample, one whole block of
# 16, a partial second block, and many blocks with a partial last one.
expect samples-1 0 'all 640 output bytes match' "$bin" --set samples=1
expect samples-16 0 'all 10240 output bytes match' "$bin" --set samples=16
expect samples-33 0 'all 21120 output bytes match' "$bin" --set samples=33
expect samples-4099-config 0 '16 samples per block' "$bin" --set samples=4099

# --- The baseline's arithmetic for every code (the fallback of both fast
# paths) gives the same bytes.
expect baseline-arithmetic-1003 0 'all 641920 output bytes match' \
  "$dir/anomaly_optimized_baseline_arithmetic" --set samples=1003

# --- Forced pipeline chunk counts (each GPU's choice among them): chunk
# boundaries on block boundaries, a short last chunk, fewer blocks than
# chunks, and Direct = Graph across the fork and join of the side streams.
for n in $chunks; do
  b="$dir/anomaly_optimized_chunks$n"
  expect "chunks$n-samples-4099" 0 'all 2623360 output bytes match' "$b" --set samples=4099
  expect "chunks$n-samples-17" 0 'all 10880 output bytes match' "$b" --set samples=17
done
expect chunks3-pipeline 0 ' 3 pipeline chunks' "$dir/anomaly_optimized_chunks3" \
  --set samples=4099
expect chunks32-pipeline 0 ' 32 pipeline chunks' "$dir/anomaly_optimized_chunks32" \
  --set samples=8180

echo "anomaly optimized tests: $passed passed, $failed failed"
[[ $failed -eq 0 ]]
