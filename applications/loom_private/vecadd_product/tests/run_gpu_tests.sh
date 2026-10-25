#!/usr/bin/env bash
# GPU tests for loom-private/vecadd-product, run by `make test` under the GPU
# lock (this script does not lock). Small --set records values only; upstream's
# smoke configuration (0 warmup, 1 measured sample) is not reproduced as a
# configuration here: records=1 is a one-record check against the oracle, and
# the bench test below runs the large workload only for its timing record.
#   run_gpu_tests.sh <binary> <build dir with mutant/slow-warmup binaries> <scratch dir>
set -uo pipefail

bin=$1
build=$2
scratch=$3
mkdir -p "$scratch"
failures=0
cache=${LOOM_PRIVATE_ASSET_CACHE:-$HOME/.cache/cuda-elastic/loom-private-assets}

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
# require_log <name> <log name> <pattern>: the named test's log has the pattern.
require_log() {
  if grep -q -- "$3" "$scratch/$2.log"; then echo "PASS $1"
  else echo "FAIL $1 (no '$3' in $2.log)"; failures=$((failures + 1)); fi
}

# Direct and Graph modes pass the oracle at record counts around block, period
# (256 records), and word-count boundaries, including odd counts.
for n in 1 2 3 7 63 64 255 256 257 1000 4097 65537 1000003; do
  expect "records-$n" 0 "$bin" --set records=$n
done
expect records-257-direct 0 "$bin" --set records=257 --launch direct
expect records-257-graph 0 "$bin" --set records=257 --launch graph
expect records-1000-json 0 "$bin" --set records=1000 --json-out "$scratch/records-1000.json"
if grep -q '"overrides": {"records": 1000}' "$scratch/records-1000.json" &&
   grep -q '"records": 1000, "operand_bytes": 64, "input_bytes": 128000, "output_bytes": 64000' "$scratch/records-1000.json" &&
   grep -q '"vecadd-product-operands": "89712b45' "$scratch/records-1000.json" &&
   grep -q '"status": "pass"' "$scratch/records-1000.json"; then
  echo "PASS result-record"
else
  echo "FAIL result-record"; cat "$scratch/records-1000.json"; failures=$((failures + 1))
fi
require_log check-reports-canary records-1000 "canary bytes damaged"

# Mutants of the same source must fail where the real binary passes.
#  1 saturating add (no wraparound)      2 one word past the output extent
#  3 sums land in the neighbouring slot  4 records with r mod 256 == 255 skipped
expect mutant1-saturating 1 "$build/vecadd_product_mutant1" --set records=1024
require_log mutant1-reports-mismatch mutant1-saturating "mismatch at output byte"
expect mutant2-overrun 1 "$build/vecadd_product_mutant2" --set records=1024
require_log mutant2-reports-canary mutant2-overrun "canary after the output overwritten"
expect mutant3-neighbour-slot 1 "$build/vecadd_product_mutant3" --set records=64
expect mutant4-skipped-slots 1 "$build/vecadd_product_mutant4" --set records=1024
require_log mutant4-skipped-slot-is-record-255 mutant4-skipped-slots "record 255, lane 0"
expect mutant4-below-first-skip 0 "$build/vecadd_product_mutant4" --set records=255

# Out-of-bounds device accesses (reads or writes) under compute-sanitizer.
sanitizer=$(command -v compute-sanitizer || true)
if [[ -n $sanitizer ]]; then
  expect memcheck-1000 0 "$sanitizer" --error-exitcode 9 --tool memcheck "$bin" \
    --set records=1000 --launch direct
  expect memcheck-65537 0 "$sanitizer" --error-exitcode 9 --tool memcheck "$bin" \
    --set records=65537 --launch direct
else
  echo "SKIP memcheck (compute-sanitizer not found)"
fi

# Assets: missing, truncated, extended, and corrupted files exit 3.
expect asset-missing 3 "$bin" --set records=4 --asset vecadd-product-operands="$scratch/none"
head -c 127 "$cache/vecadd-product-operands" >"$scratch/operands-short"
expect asset-truncated 3 "$bin" --set records=4 --asset vecadd-product-operands="$scratch/operands-short"
{ cat "$cache/vecadd-product-operands"; printf 'x'; } >"$scratch/operands-long"
expect asset-extended 3 "$bin" --set records=4 --asset vecadd-product-operands="$scratch/operands-long"
python3 -I -c '
import sys
data = bytearray(open(sys.argv[1], "rb").read()); data[5] ^= 1
open(sys.argv[2], "wb").write(data)' "$cache/vecadd-product-operands" "$scratch/operands-flipped"
expect asset-corrupt 3 "$bin" --set records=4 --asset vecadd-product-operands="$scratch/operands-flipped"
require_log asset-corrupt-message asset-corrupt "has sha256"
python3 -I -c '
import sys
data = bytearray(open(sys.argv[1], "rb").read()); data[30] ^= 1
open(sys.argv[2], "wb").write(data)' "$cache/vecadd-product-upstream-expected-smoke" "$scratch/reference-flipped"
expect reference-corrupt 3 "$bin" --set records=4 --asset vecadd-product-upstream-expected-smoke="$scratch/reference-flipped"

# Invalid dimensions and usage exit 2.
expect zero-records 2 "$bin" --set records=0
expect negative-records 2 "$bin" --set records=-5
expect too-many-records 2 "$bin" --set records=16777217
expect non-integer 2 "$bin" --set records=abc
expect unknown-key 2 "$bin" --set records=8 --set lanes=3
expect unknown-preset 2 "$bin" --preset small
expect bench-with-set 2 "$bin" --action bench --launch direct --set records=8

# Bench path works at the large size (it does not run the oracle).
expect bench-direct 0 "$bin" --action bench --launch direct --iterations 2 --warmup 1
expect bench-graph 0 "$bin" --action bench --launch graph --iterations 2 --warmup 1
require_log bench-record-unvalidated bench-graph '"status": "unvalidated_bench"'

# Warmup exclusion. The slow-warmup binary makes the first three application
# invocations (the default bench warmups) spin the GPU for 300 ms each. With
# 3 warmups the timed batch must not see them; with 0 warmups it must.
slow="$build/vecadd_product_slow_warmup"
batch_ms() { # batch_total_ms of the application scope from a RESULT line
  python3 -I -c '
import json, sys
for line in open(sys.argv[1]):
    if line.startswith("RESULT "):
        print(json.loads(line[7:])["timing"]["application"]["batch_total_ms"])' "$1"
}
expect slow-warmup-3 0 "$slow" --action bench --launch direct --iterations 3 --warmup 3
expect slow-warmup-0 0 "$slow" --action bench --launch direct --iterations 3 --warmup 0
excluded=$(batch_ms "$scratch/slow-warmup-3.log")
included=$(batch_ms "$scratch/slow-warmup-0.log")
if python3 -I -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 200 and float(sys.argv[2]) > 800 else 1)' \
     "${excluded:-1e9}" "${included:-0}"; then
  echo "PASS warmups-outside-timed-interval (batch ${excluded} ms with warmups, ${included} ms without)"
else
  echo "FAIL warmups-outside-timed-interval (batch ${excluded} ms with warmups, ${included} ms without)"
  failures=$((failures + 1))
fi

if [[ $failures -ne 0 ]]; then
  echo "GPU tests FAILED ($failures)"
  exit 1
fi
echo "GPU tests passed"
