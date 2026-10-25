#!/usr/bin/env bash
# GPU tests for loom-private/vecadd-memory, run by `make test` under the GPU
# lock (this script does not lock). Small --set dimensions only; none of them is
# an upstream smoke or validation configuration (4096/2/0, 65536/3/1,
# 131072/5/7).
#   run_gpu_tests.sh <vecadd_memory binary> <drop-last mutant binary> <scratch dir>
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

# Tail handling: lengths of 1, below, at, and above the 256-thread block size,
# and a large length that is a multiple of neither 256 nor any power of two.
expect length-1 0 "$bin" --set length=1
expect length-2 0 "$bin" --set length=2
expect length-255 0 "$bin" --set length=255
expect length-256 0 "$bin" --set length=256
expect length-257 0 "$bin" --set length=257
expect length-1000003 0 "$bin" --set length=1000003 --json-out "$scratch/length-1000003.json"
expect length-1000003-direct 0 "$bin" --set length=1000003 --launch direct
expect length-5000-graph 0 "$bin" --set length=5000 --launch graph

# Other multipliers and biases, including negative and zero ones.
expect mult-7-bias-neg5 0 "$bin" --set length=70001 --set multiplier=7 --set bias=-5
expect mult-neg3-bias-11 0 "$bin" --set length=8191 --set multiplier=-3 --set bias=11
expect mult-0-bias-neg100 0 "$bin" --set length=4097 --set multiplier=0 --set bias=-100
expect mult-neg1 0 "$bin" --set length=9999 --set multiplier=-1

# Values right at the int32 limits are accepted and computed exactly.
expect int32-max-result 0 "$bin" --set length=2 --set multiplier=1 --set bias=2147483645
expect int32-min-rhs 0 "$bin" --set length=2 --set multiplier=0 --set bias=-2147483648
expect int32-large-multiplier 0 "$bin" --set length=3 --set multiplier=1073741822
# Largest multiplier that fits at the large length; the 64-bit checksum is far
# beyond int32.
expect large-length-mult-2047 0 "$bin" --set multiplier=2047

if grep -q '"overrides": {"length": 1000003}' "$scratch/length-1000003.json" &&
   grep -q '"status": "pass"' "$scratch/length-1000003.json" &&
   grep -q '"dimensions": {"length": 1000003, "multiplier": 2, "bias": 0}' "$scratch/length-1000003.json"; then
  echo "PASS result-record"
else
  echo "FAIL result-record"; cat "$scratch/length-1000003.json"
  failures=$((failures + 1))
fi

# A kernel that skips the last element must fail the oracle, at a tail length
# and at length 1 (where nothing is written and the poison survives).
expect drop-last-mutant-1000003 1 "$mutant" --set length=1000003
expect drop-last-mutant-1 1 "$mutant" --set length=1
if grep -q 'mismatch at index 1000002: expected 3000006' "$scratch/drop-last-mutant-1000003.log" &&
   grep -q 'checksum mismatch' "$scratch/drop-last-mutant-1000003.log"; then
  echo "PASS drop-last-mutant-reports-index-and-checksum"
else
  echo "FAIL drop-last-mutant-reports-index-and-checksum"; failures=$((failures + 1))
fi

# Invalid dimensions and usage exit 2.
expect zero-length 2 "$bin" --set length=0
expect negative-length 2 "$bin" --set length=-4
expect too-long 2 "$bin" --set length=134217729
expect mult-overflows-result 2 "$bin" --set length=3 --set multiplier=1073741823
expect mult-overflows-large 2 "$bin" --set multiplier=2048
expect bias-overflows-result 2 "$bin" --set length=2 --set bias=2147483646
expect bias-overflows-rhs 2 "$bin" --set length=2 --set multiplier=2 --set bias=2147483647
expect negative-overflow 2 "$bin" --set length=2 --set multiplier=-2 --set bias=-2147483648
expect multiplier-not-int32 2 "$bin" --set multiplier=2147483648
expect bias-not-int32 2 "$bin" --set bias=-2147483649
expect non-integer 2 "$bin" --set length=abc
expect unknown-key 2 "$bin" --set length=16 --set width=3
expect unknown-preset 2 "$bin" --preset smoke
expect bench-with-set 2 "$bin" --action bench --launch direct --set length=1024
if grep -q 'length must be at least 1' "$scratch/zero-length.log" &&
   grep -q 'overflow int32' "$scratch/mult-overflows-result.log"; then
  echo "PASS validation-messages"
else
  echo "FAIL validation-messages"; failures=$((failures + 1))
fi

if [[ $failures -ne 0 ]]; then
  echo "GPU tests FAILED ($failures)"
  exit 1
fi
echo "GPU tests passed"
