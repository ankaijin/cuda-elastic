#!/usr/bin/env bash
# GPU tests for loom-private/pagerank, run by `make test` under the GPU lock
# (this script does not lock). Small --set dimensions only; none of them is an
# upstream smoke or validation configuration (4/8, 2048/12, 2048/16).
#   run_gpu_tests.sh <pagerank binary> <Jacobi mutant binary> <scratch dir>
set -uo pipefail

bin=$1
mutant=$2
scratch=$3
mkdir -p "$scratch"
failures=0

# expect <name> <exit code> <command...>
expect() {
  local name=$1 want=$2
  shift 2
  local log="$scratch/$name.log"
  "$@" >"$log" 2>&1
  local got=$?
  if [[ $got -eq $want ]]; then
    echo "PASS $name (exit $got)"
  else
    echo "FAIL $name: exit $got, expected $want"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  fi
}

# Ordered Gauss-Seidel on the GPU passes the oracle in Direct and Graph modes.
expect small-16x3 0 "$bin" --set nodes=16 --set iterations=3
expect small-4096x6 0 "$bin" --set nodes=4096 --set iterations=6 \
  --json-out "$scratch/small-4096x6.json"
expect small-65536x20 0 "$bin" --set nodes=65536 --set iterations=20
expect small-8192x1-direct 0 "$bin" --set nodes=8192 --set iterations=1 --launch direct

# The record names the overrides and the schedule depth.
if grep -q '"overrides": {"nodes": 4096, "iterations": 6}' "$scratch/small-4096x6.json" &&
   grep -q '"status": "pass"' "$scratch/small-4096x6.json" &&
   grep -q '"schedule_levels": ' "$scratch/small-4096x6.json"; then
  echo "PASS result-record"
else
  echo "FAIL result-record"; cat "$scratch/small-4096x6.json"
  failures=$((failures + 1))
fi

# A Jacobi kernel (every read from the previous sweep) must fail the oracle on
# the same dimensions that pass above: the check distinguishes the orders.
expect jacobi-mutant-4096x6 1 "$mutant" --set nodes=4096 --set iterations=6
expect jacobi-mutant-65536x20 1 "$mutant" --set nodes=65536 --set iterations=20
if grep -q 'mismatch at node' "$scratch/jacobi-mutant-4096x6.log"; then
  echo "PASS jacobi-mutant-reports-mismatches"
else
  echo "FAIL jacobi-mutant-reports-mismatches"; failures=$((failures + 1))
fi

# Invalid dimensions and usage exit 2.
expect non-power-of-two 2 "$bin" --set nodes=1000 --set iterations=2
expect too-small 2 "$bin" --set nodes=2 --set iterations=2
expect zero-nodes 2 "$bin" --set nodes=0
expect negative-nodes 2 "$bin" --set nodes=-8
expect too-large 2 "$bin" --set nodes=134217728
expect zero-iterations 2 "$bin" --set nodes=1024 --set iterations=0
expect non-integer 2 "$bin" --set nodes=abc
expect unknown-key 2 "$bin" --set nodes=1024 --set width=3
expect unknown-preset 2 "$bin" --preset small
expect bench-with-set 2 "$bin" --action bench --launch direct --set nodes=1024

if [[ $failures -ne 0 ]]; then
  echo "GPU tests FAILED ($failures)"
  exit 1
fi
echo "GPU tests passed"
