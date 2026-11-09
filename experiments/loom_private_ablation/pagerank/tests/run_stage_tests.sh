#!/usr/bin/env bash
# GPU tests of the pagerank stages beyond the baseline's run_gpu_tests.sh,
# run by `make test` under the GPU lock (this script does not lock).
#   run_stage_tests.sh <build dir> <scratch dir>
# <build dir> holds pagerank_stage{1,2,3} and their test-only _shapes builds,
# in which LP_ABLATION_SHAPE=tiny forces a one-block, 64-thread grid.
set -uo pipefail

build=$1
scratch=$2
mkdir -p "$scratch"
failures=0
limit=120

# expect <name> <exit code> <log pattern> <command...>
expect() {
  local name=$1 want=$2 pattern=$3
  shift 3
  local log="$scratch/$name.log"
  timeout $limit "$@" >"$log" 2>&1
  local got=$?
  if [[ $got -ne $want ]]; then
    echo "FAIL $name: exit $got, expected $want"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  elif ! grep -q -- "$pattern" "$log"; then
    echo "FAIL $name: log lacks '$pattern'"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  else
    echo "PASS $name (exit $got)"
  fi
}

for s in 1 2 3; do
  bin=$build/pagerank_stage$s
  shapes=$build/pagerank_stage${s}_shapes
  # The timed binary uses the default shape and names its stage.
  expect stage$s-default 0 "stage $s: .*default shape: .* x 512 threads" \
    "$bin" --set nodes=65536 --set iterations=7
  # It ignores the test build's shape variable.
  LP_ABLATION_SHAPE=tiny expect stage$s-ignores-shape 0 "default shape" \
    "$bin" --set nodes=4096 --set iterations=3
  # One block of 64 threads: every thread visits many nodes per sweep and
  # crosses level boundaries; at 4, 16, 32 and 4096 nodes some updates wait
  # for a source stored by another lane of the same warp (stages 2 and 3).
  for size in "4 9" "16 3" "32 5" "4096 6" "65536 20"; do
    set -- $size
    LP_ABLATION_SHAPE=tiny expect stage$s-tiny-$1x$2 0 "tiny shape: 1 blocks x 64 threads" \
      "$shapes" --set nodes=$1 --set iterations=$2
  done
  LP_ABLATION_SHAPE=bogus expect stage$s-unknown-shape 2 "unknown shape" \
    "$shapes" --set nodes=1024
  # Sweep counts around the readiness tag's four-sweep cycle on the large graph.
  for n in 1 2 3 4 5; do
    expect stage$s-large-x$n 0 "1048576 nodes" \
      "$bin" --set nodes=1048576 --set iterations=$n --launch direct
  done
done

if [[ $failures -ne 0 ]]; then
  echo "stage tests FAILED ($failures)"
  exit 1
fi
echo "stage tests passed"
