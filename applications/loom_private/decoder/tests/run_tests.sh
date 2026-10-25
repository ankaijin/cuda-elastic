#!/usr/bin/env bash
# Small tests for the decoder: run by `make test`, which already holds the GPU
# lock (this script must not take it again). Usage: run_tests.sh <binary>
set -uo pipefail

bin="${1:?usage: run_tests.sh <decoder binary>}"
log="$(mktemp -d)"
trap 'rm -rf "$log"' EXIT
passed=0
failed=0

# expect <name> <exit code> <stderr pattern or ''> -- <arguments...>
expect() {
  local name="$1" want="$2" pattern="$3"
  shift 4
  "$bin" "$@" >"$log/out" 2>"$log/err"
  local got=$?
  local ok=1
  [[ $got -eq $want ]] || ok=0
  if [[ $ok -eq 1 && $want -eq 0 ]]; then
    grep -q '"status": "pass"' "$log/out" || ok=0
  fi
  if [[ $ok -eq 1 && -n "$pattern" ]]; then
    grep -Eq "$pattern" "$log/err" || ok=0
  fi
  if [[ $ok -eq 1 ]]; then
    echo "PASS $name (exit $got)"
    passed=$((passed + 1))
  else
    echo "FAIL $name (exit $got, expected $want${pattern:+, stderr matching /$pattern/})"
    sed 's/^/    /' "$log/err" | tail -n 20
    failed=$((failed + 1))
  fi
}

# Correct runs (Direct and Graph, two invocations each). Shapes avoid
# upstream's smoke (128/16) and validation (1024/64) configurations.
# Prompts wrap (batch > vocab) and tokens wrap past vocab - 1; width and vocab
# are not multiples of the 64-wide tiles or the 16-column stages.
expect wraparound 0 'PASS: 0/200 tokens wrong' -- \
  --set batch=40 --set width=96 --set vocab=16 --set steps=5
# Smallest legal model: vocab = width = 2, two steps.
expect minimum 0 'PASS' -- --set batch=3 --set width=2 --set vocab=2 --set steps=2
# vocab == width, ragged tiles in both dimensions, prompts wrap at 1000.
expect square 0 'PASS' -- --set batch=1030 --set width=1000 --set vocab=1000 --set steps=3
# Long rows: width not a multiple of the 16-column stage, single sequence.
expect wide 0 'PASS' -- --set batch=1 --set width=4100 --set vocab=300 --set steps=2
# Graph-only launch path.
expect graph-only 0 'graph invocation 2: PASS' -- --launch graph \
  --set batch=70 --set width=200 --set vocab=33 --set steps=4

# Invalid dimensions and usage: exit 2 with a clear message.
expect vocab-1 2 'vocab must satisfy' -- --set vocab=1 --set width=8
expect vocab-gt-width 2 'vocab must satisfy' -- --set vocab=17 --set width=16
expect steps-1 2 'steps must be' -- --set steps=1 --set width=32 --set vocab=8
expect batch-0 2 'batch must be' -- --set batch=0
expect width-0 2 'width must be' -- --set width=0
# 2^24 - 15: the square sum would reach 2^24 (rejected before allocation).
expect width-over-exact 2 'width must be' -- --set batch=1 --set width=16777201 --set vocab=2 --set steps=2
expect too-large 2 'GiB' -- --set batch=1048576
expect not-integer 2 'not an integer' -- --set batch=abc
expect unknown-key 2 "unsupported --set key 'rows'" -- --set rows=4
expect unknown-preset 2 "unknown preset 'smoke'" -- --preset smoke --set width=32 --set vocab=8

# Injected faults must be reported with indices and expected/actual values.
expect fault-token 1 'token mismatch sequence 39 step 4: expected 12, actual 13' -- \
  --set batch=40 --set width=96 --set vocab=16 --set steps=5 --set fault=1
expect fault-activation 1 'invariant failure sequence 20 step 4' -- \
  --set batch=40 --set width=96 --set vocab=16 --set steps=5 --set fault=2
expect fault-probability 1 'probability mismatch sequence 0 step 4 index 8: expected' -- \
  --set batch=40 --set width=96 --set vocab=16 --set steps=5 --set fault=3

echo "decoder tests: $passed passed, $failed failed"
[[ $failed -eq 0 ]]
