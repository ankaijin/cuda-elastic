#!/usr/bin/env bash
# Tests for loom-private/mlperf-tiny-anomaly-model, run by `make test`.
# Each GPU run takes $GPU_LOCK here. Sample counts are small --set overrides;
# none is an upstream smoke (1 measured sample) or validation (2 warmup + 2
# measured) profile, and none is a multiple of 40 except where stated.
#   run_tests.sh <anomaly binary> <skip-last mutant binary> <scratch dir>
set -uo pipefail

bin=${1:?usage: run_tests.sh <binary> <mutant> <scratch>}
mutant=${2:?usage: run_tests.sh <binary> <mutant> <scratch>}
scratch=${3:?usage: run_tests.sh <binary> <mutant> <scratch>}
lock=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
cache=${LOOM_PRIVATE_ASSET_CACHE:-$HOME/.cache/cuda-elastic/loom-private-assets}
model=mlperf-tiny-ad01-int8-model
dataset=mlperf-tiny-ad01-dataset
smoke=anomaly-upstream-expected-smoke
rm -rf "$scratch"
mkdir -p "$scratch"
passed=0
failed=0

# expect <name> <exit code> <stderr pattern or ''> <command...>
expect() {
  local name=$1 want=$2 pattern=$3
  shift 3
  local log="$scratch/$name.log"
  flock "$lock" "$@" >"$log.out" 2>"$log"
  local got=$?
  local ok=1
  [[ $got -eq $want ]] || ok=0
  if [[ $want -eq 0 ]] && ! grep -q '"status": "pass"' "$log.out"; then ok=0; fi
  if [[ -n $pattern ]] && ! grep -q -- "$pattern" "$log"; then ok=0; fi
  if [[ $ok -eq 1 ]]; then
    passed=$((passed + 1))
    echo "PASS $name (exit $got)"
  else
    failed=$((failed + 1))
    echo "FAIL $name: exit $got, expected $want${pattern:+, stderr matching '$pattern'}"
    tail -n 8 "$log" | sed 's/^/    /'
  fi
}

# check <name> <condition command...>: a non-GPU assertion.
check() {
  local name=$1
  shift
  if "$@"; then
    passed=$((passed + 1)); echo "PASS $name"
  else
    failed=$((failed + 1)); echo "FAIL $name"
  fi
}

# --- GPU inference against the CPU oracle (Direct and Graph, 2 invocations each)
expect samples-7 0 'oracle matches upstream reference bytes' "$bin" --set samples=7
expect samples-93 0 'all 59520 output bytes match' "$bin" --set samples=93 \
  --json-out "$scratch/samples-93.json"
expect samples-41-direct 0 '' "$bin" --set samples=41 --launch direct
expect samples-1003-graph 0 '' "$bin" --set samples=1003 --launch graph
# The record names the override, the four verified assets, and the widths.
check result-record grep -q -e '"overrides": {"samples": 93}' "$scratch/samples-93.json"
check result-record-assets grep -q -e '"anomaly-upstream-expected-validation": "f1bc2c6d' \
  "$scratch/samples-93.json"
check result-record-widths grep -q -e '"layer_widths": \[640, 128, 128, 128, 128, 8, 128, 128, 128, 128, 640\]' \
  "$scratch/samples-93.json"

# --- The check catches an unwritten sample (poisoned output) with exit 1.
expect mutant-skip-last 1 'mismatch at sample 92 ' "$mutant" --set samples=93

# --- Asset failures exit 3 with a clear message.
cp "$cache/$model" "$scratch/model-corrupt"
python3 - "$scratch/model-corrupt" <<'EOF'
import sys
path = sys.argv[1]
data = bytearray(open(path, "rb").read())
data[100000] ^= 0x01  # inside the weight buffers
open(path, "wb").write(data)
EOF
expect corrupt-model 3 'has sha256' "$bin" --set samples=7 \
  --asset "$model=$scratch/model-corrupt"
expect missing-model 3 'is missing' "$bin" --set samples=7 \
  --asset "$model=$scratch/does-not-exist"
expect missing-dataset 3 'is missing' "$bin" --set samples=7 \
  --asset "$dataset=$scratch/does-not-exist"
head -c 102399 "$cache/$dataset" >"$scratch/dataset-truncated"
expect truncated-dataset 3 'has 102399 bytes' "$bin" --set samples=7 \
  --asset "$dataset=$scratch/dataset-truncated"
{ cat "$cache/$dataset"; printf 'x'; } >"$scratch/dataset-long"
expect oversized-dataset 3 'has more than 102400 bytes' "$bin" --set samples=7 \
  --asset "$dataset=$scratch/dataset-long"
cp "$cache/$smoke" "$scratch/smoke-corrupt"
python3 - "$scratch/smoke-corrupt" <<'EOF'
import sys
path = sys.argv[1]
data = bytearray(open(path, "rb").read())
data[len(data) // 2] ^= 0x01  # one bit inside the hex payload
open(path, "wb").write(data)
EOF
expect corrupt-reference 3 'has sha256' "$bin" --set samples=7 \
  --asset "$smoke=$scratch/smoke-corrupt"

# --- Invalid dimensions and usage exit 2.
expect zero-samples 2 'samples must be in' "$bin" --set samples=0
expect negative-samples 2 'samples must be in' "$bin" --set samples=-3
expect too-many-samples 2 'samples must be in' "$bin" --set samples=1048577
expect non-integer-samples 2 'not an integer' "$bin" --set samples=12x
expect unknown-key 2 'unsupported --set key' "$bin" --set batch=4
expect unknown-preset 2 'unknown preset' "$bin" --preset smoke
# A misspelled --asset name must not be ignored (the cached path would be used).
expect unknown-asset 2 "unsupported --asset name 'mlperf-tiny-ad01-datset'" "$bin" \
  --set samples=7 --asset "mlperf-tiny-ad01-datset=$cache/$dataset"
expect bench-with-set 2 'only valid with --action check' "$bin" --action bench \
  --launch direct --set samples=7

echo "anomaly tests: $passed passed, $failed failed"
[[ $failed -eq 0 ]]
