#!/usr/bin/env bash
# GPU tests of the optimized variant's launch shapes, run by
# `make test VARIANT=optimized` under the GPU lock (this script does not
# lock). The binary is the optimized source built with
# -DLP_PAGERANK_TEST_TUNING, which lets LP_PAGERANK_TUNING force any entry of
# its tuning table, so every GPU's choice runs on whichever GPU runs the tests.
#   run_optimized_tests.sh <tuning test binary> <scratch dir>
set -uo pipefail

bin=$1
scratch=$2
mkdir -p "$scratch"
failures=0

# expect <name> <exit code> <tuning> <log pattern> <args...>
expect() {
  local name=$1 want=$2 tuning=$3 pattern=$4
  shift 4
  local log="$scratch/$name.log"
  LP_PAGERANK_TUNING=$tuning "$bin" "$@" >"$log" 2>&1
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

# Each GPU's launch shape (the 3090's with streamed graph loads) passes the
# oracle in Direct and Graph modes, bitwise identical across invocations.
expect tuning-sm_86 0 sm_86 "sm_86 tuning: .*streamed graph loads" \
  --set nodes=65536 --set iterations=20
expect tuning-sm_89 0 sm_89 "sm_89 tuning: " --set nodes=65536 --set iterations=20
expect tuning-sm_120 0 sm_120 "sm_120 tuning: " --set nodes=65536 --set iterations=20
expect tuning-default 0 default "default tuning: " --set nodes=65536 --set iterations=7

# One block of 64 threads: most waits are between lanes of one warp, and each
# thread visits many positions, crossing level boundaries, in every sweep.
expect tiny-4 0 tiny "tiny tuning: 1 blocks x 64 threads" --set nodes=4 --set iterations=9
expect tiny-4096x6 0 tiny "tiny tuning: 1 blocks x 64 threads" \
  --set nodes=4096 --set iterations=6
expect tiny-65536x20 0 tiny "tiny tuning: 1 blocks x 64 threads" \
  --set nodes=65536 --set iterations=20

# A grid capped by the node count: 32 nodes need one block of every shape.
expect sm_86-32 0 sm_86 "1 blocks x 768 threads" --set nodes=32 --set iterations=5

# Sweep counts around the tag cycle (the readiness tag repeats every four
# sweeps) on the large graph with the 3090 and 5090 shapes.
for n in 1 2 3 4 5; do
  expect large-x$n-sm_120 0 sm_120 "1048576 nodes" --set nodes=1048576 --set iterations=$n \
    --launch direct
done
expect large-x6-sm_86 0 sm_86 "1048576 nodes" --set nodes=1048576 --set iterations=6 \
  --launch graph

expect unknown-tuning 2 sm_00 "unknown tuning" --set nodes=1024

if [[ $failures -ne 0 ]]; then
  echo "optimized GPU tests FAILED ($failures)"
  exit 1
fi
echo "optimized GPU tests passed"
