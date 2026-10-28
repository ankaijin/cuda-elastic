#!/usr/bin/env bash
# GPU tests of the optimized variant, run by `make test VARIANT=optimized`
# under the GPU lock (this script does not lock). Every run has a timeout;
# the kernel's own spin limit traps long before it.
#   run_optimized_tests.sh <tuning test binary> <tag mutant> <no-reset mutant>
#                          <scratch dir>
# The tuning test binary is the optimized source built with
# -DLP_PAGERANK_TEST_TUNING, which lets LP_PAGERANK_TUNING force any entry of
# its tuning table, so every GPU's choice runs on whichever GPU runs the tests.
set -uo pipefail

bin=$1
tag_mutant=$2
reset_mutant=$3
scratch=$4
mkdir -p "$scratch"
failures=0
limit=120

# expect <name> <exit code> <tuning> <log pattern> <args...>
expect() {
  local name=$1 want=$2 tuning=$3 pattern=$4
  shift 4
  local log="$scratch/$name.log"
  LP_PAGERANK_TUNING=$tuning timeout $limit "$bin" "$@" >"$log" 2>&1
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

# expect_failure <name> <allowed exit codes> <command...>: the run must end,
# within the timeout, with one of the allowed codes.
expect_failure() {
  local name=$1 allowed=$2
  shift 2
  local log="$scratch/$name.log"
  timeout $limit "$@" >"$log" 2>&1
  local got=$?
  if [[ " $allowed " == *" $got "* ]]; then
    echo "PASS $name (exit $got)"
  else
    echo "FAIL $name: exit $got, expected one of: $allowed"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  fi
}

# Each GPU's launch shape (the 3090's with streamed graph loads) passes the
# oracle in Direct and Graph modes, bitwise identical across invocations.
expect tuning-sm_86 0 sm_86 "sm_86 tuning: .*streamed graph loads" \
  --set nodes=65536 --set iterations=20
expect tuning-sm_89 0 sm_89 "sm_89 tuning: " --set nodes=65536 --set iterations=20
expect tuning-sm_120 0 sm_120 "sm_120 tuning: " --set nodes=65536 --set iterations=20
expect tuning-default 0 default "default tuning: " --set nodes=65536 --set iterations=7

# One block of 64 threads: every thread visits many positions per sweep and
# crosses level boundaries. At 4, 16, 32 and 4096 nodes some updates wait for
# a source stored by another lane of the same warp in the same step, which
# relies on independent thread scheduling; from 8192 nodes up the generated
# graph has no such edge, so 65536x20 covers only the many-positions case.
expect tiny-4x9 0 tiny "tiny tuning: 1 blocks x 64 threads" --set nodes=4 --set iterations=9
expect tiny-16x3 0 tiny "tiny tuning: 1 blocks x 64 threads" --set nodes=16 --set iterations=3
expect tiny-32x5 0 tiny "tiny tuning: 1 blocks x 64 threads" --set nodes=32 --set iterations=5
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

# Mutants of the readiness protocol must fail. A tag that alternates every
# sweep accepts the value from two sweeps earlier: the check rejects it.
for size in "4096 6" "65536 20"; do
  set -- $size
  expect_failure tag-mutant-$1x$2 "1" "$tag_mutant" --set nodes=$1 --set iterations=$2
done
if grep -q 'mismatch at node' "$scratch/tag-mutant-4096x6.log"; then
  echo "PASS tag-mutant-reports-mismatches"
else
  echo "FAIL tag-mutant-reports-mismatches"; failures=$((failures + 1))
fi
# Without the barrier counter reset the sweeps overlap: the check rejects the
# scores (exit 1) or a wait exceeds the kernel's spin limit (exit 70), and
# the run never hangs (a timeout's 124 fails the test).
for size in "4096 6" "65536 20"; do
  set -- $size
  expect_failure no-reset-mutant-$1x$2 "1 70" "$reset_mutant" --set nodes=$1 --set iterations=$2
done

if [[ $failures -ne 0 ]]; then
  echo "optimized GPU tests FAILED ($failures)"
  exit 1
fi
echo "optimized GPU tests passed"
