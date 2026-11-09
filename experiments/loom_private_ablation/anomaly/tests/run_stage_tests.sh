#!/usr/bin/env bash
# GPU tests of the anomaly stages beyond the production run_tests.sh and
# run_optimized_tests.sh, run by `make test`. Each GPU run takes $GPU_LOCK
# here, one invocation at a time.
#   run_stage_tests.sh <build dir> <scratch dir>
# <build dir> holds anomaly_stage{1,2,3,4} and their test-only builds:
# _no_correction_mutant (no zero-point correction) and, for stages 2 and 3,
# _baseline_arithmetic (the binary64 fallback for every code).
set -uo pipefail

build=${1:?usage: run_stage_tests.sh <build dir> <scratch dir>}
scratch=${2:?usage: run_stage_tests.sh <build dir> <scratch dir>}
here=$(cd "$(dirname "$0")" && pwd)
lock=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
rm -rf "$scratch"
mkdir -p "$scratch"
passed=0
failed=0
limit=600

# expect <name> <exit code> <output pattern or ''> <command...>: stdout and
# stderr are both searched; exit 0 also needs a passing result record.
expect() {
  local name=$1 want=$2 pattern=$3
  shift 3
  local log="$scratch/$name.log"
  flock "$lock" timeout $limit "$@" >"$log" 2>&1
  local got=$?
  local ok=1
  [[ $got -eq $want ]] || ok=0
  if [[ $want -eq 0 ]] && ! grep -q '"status": "pass"' "$log"; then ok=0; fi
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

# The timed source reads no environment variables (the shared harness reads
# only the asset cache location).
if grep -q 'getenv' "$here/../anomaly_stages.cu"; then
  failed=$((failed + 1)); echo "FAIL stage-source-reads-no-environment"
else
  passed=$((passed + 1)); echo "PASS stage-source-reads-no-environment"
fi

configs=(""
  "stage 1 configuration .*16 samples per block, dp4a dot products, binary64 requantization, unpipelined"
  "stage 2 configuration .*16 samples per block, dp4a dot products, binary32 fast-path requantization, unpipelined"
  "stage 3 configuration .*16 samples per block, int8 tensor-core dot products, binary32 fast-path requantization, unpipelined"
  "stage 4 configuration .*16 samples per block, 32 pipeline chunks")

for s in 1 2 3 4; do
  bin=$build/anomaly_stage$s
  # The timed binary names its stage and configuration; stage 4 uses the
  # generic 32 chunks on every GPU.
  expect stage$s-config 0 "${configs[$s]}" "$bin" --set samples=8180
  # Tile edges (16 samples per block): one sample, a partial tile, one whole
  # tile, one more, a partial third tile, and many tiles with a partial last
  # one; every check runs Direct and Graph twice each.
  for n in 1 15 16 17 33 4099; do
    expect stage$s-samples-$n 0 "all $((n * 640)) output bytes match" "$bin" --set samples=$n
  done
  # Without the zero-point correction the integer sums are wrong, and the
  # check must catch it.
  expect stage$s-no-correction-mutant 1 'output bytes mismatch' \
    "$build/anomaly_stage${s}_no_correction_mutant" --set samples=93
done

# The binary64 fallback of every code in stages 2 and 3 (stage 4's runs in
# run_optimized_tests.sh; stage 1 uses it in the timed binary).
for s in 2 3; do
  expect stage$s-baseline-arithmetic 0 'all 641920 output bytes match' \
    "$build/anomaly_stage${s}_baseline_arithmetic" --set samples=1003
done

# The largest sample count the binary accepts, for the dp4a kernel and for
# the pipeline (2,048 tiles per chunk).
for s in 1 4; do
  expect stage$s-samples-max 0 'all 671088640 output bytes match' \
    "$build/anomaly_stage$s" --set samples=1048576
done

echo "anomaly stage tests: $passed passed, $failed failed"
[[ $failed -eq 0 ]]
