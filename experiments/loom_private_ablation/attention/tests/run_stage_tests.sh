#!/usr/bin/env bash
# GPU tests of the attention stages beyond the baseline's run_tests.sh, run by
# `make test`. Every GPU process takes $GPU_LOCK itself (default
# /tmp/cuda-elastic-gpu.lock); do not run this script under that lock.
#   run_stage_tests.sh <build dir> <scratch dir>
# <build dir> holds attention_stage{1..6}, their test-only _hooks builds
# (LP_ATTENTION_SPLITS, LP_ATTENTION_EXACT_MAXIMUM from stage 3,
# LP_ATTENTION_CHECK_COMPUTE in stage 6), their _skip_key_mutant builds, and
# attention_hooks_reference (production attention_optimized.cu with its own
# tuning hooks).
set -uo pipefail

build=$1
scratch=$2
mkdir -p "$scratch"
LOCK=${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}
failures=0
passed=0
limit=600

record() {  # record <name> <ok> <log>
  if [[ $2 == 1 ]]; then
    passed=$((passed + 1))
    echo "PASS $1"
  else
    failures=$((failures + 1))
    echo "FAIL $1"
    tail -n 12 "$3" | sed 's/^/    /'
  fi
}

# expect <name> <exit code> <setup pattern> <environment> <binary> <args...>:
# the exit code, the check status for exits 0 and 1, and (when not empty) a
# fixed string the setup line on stderr must contain.
expect() {
  local name=$1 want=$2 pattern=$3 environment=$4 binary=$5
  shift 5
  local log="$scratch/$name.log" json="$scratch/$name.out"
  rm -f "$json"
  env $environment flock "$LOCK" timeout $limit "$binary" "$@" >"$json" 2>"$log"
  local got=$? ok=1
  [[ $got == "$want" ]] || ok=0
  if [[ $want == 0 ]] && ! grep -q '"status": "pass"' "$json"; then ok=0; fi
  if [[ $want == 1 ]] && ! grep -q '"status": "fail"' "$json"; then ok=0; fi
  if [[ $want == 2 ]] && ! grep -q . "$log"; then ok=0; fi
  if [[ -n $pattern ]] && ! grep -qF -- "$pattern" "$log"; then ok=0; fi
  echo "exit $got, expected $want" >>"$log"
  record "$name (exit $got)" $ok "$log"
}

# digest <name>: the one output digest of a passing expect run, or "none".
digest() {
  python3 -I - "$scratch/$1.out" <<'EOF'
import json, sys
try:
    lines = [l for l in open(sys.argv[1]) if l.startswith("RESULT ")]
    data = json.loads(lines[-1][len("RESULT "):])
    digests = {c["digest"] for c in data["checks"]}
    print(digests.pop() if data["status"] == "pass" and len(digests) == 1 else "none")
except Exception:
    print("none")
EOF
}

# same_digest <name> <run a> <run b>: two passing runs' digests are identical.
same_digest() {
  local a b
  a=$(digest "$2")
  b=$(digest "$3")
  local log="$scratch/$1.log"
  echo "$2: $a; $3: $b" >"$log"
  record "$1 ($a)" $([[ $a != none && $a == "$b" ]] && echo 1 || echo 0) "$log"
}

# Default split count of the timed binaries at 129 x 5000: one split in stage
# 1, the generic default of 12 (10 non-empty splits of 512 keys) after it.
default_split() { [[ $1 == 1 ]] && echo "1 context splits of 5120 keys" ||
                  echo "10 context splits of 512 keys"; }

for s in 1 2 3 4 5 6; do
  bin=$build/attention_stage$s
  hooks=$build/attention_stage${s}_hooks
  mutant=$build/attention_stage${s}_skip_key_mutant
  # The timed binary names its stage and uses the generic configuration ...
  expect s$s-default 0 "stage $s (" "" "$bin" \
    --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
  grep -qF "$(default_split $s)" "$scratch/s$s-default.log" ||
    record "s$s-default-splits" 0 "$scratch/s$s-default.log"
  # ... and ignores every hook variable.
  expect s$s-ignores-hooks 0 "$(default_split $s)" \
    "LP_ATTENTION_SPLITS=3 LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_CHECK_COMPUTE=1" "$bin" \
    --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
  if grep -q "exact row maxima\|compute path" "$scratch/s$s-ignores-hooks.log"; then
    record "s$s-ignores-hooks-line" 0 "$scratch/s$s-ignores-hooks.log"
  fi
  # One, a few, and many context splits: partial tiles, partial row groups,
  # and the merge over many splits.
  expect s$s-one-split 0 "1 context splits of 5120 keys" "LP_ATTENTION_SPLITS=1" "$hooks" \
    --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
  expect s$s-three-splits 0 "3 context splits of 1024 keys" "LP_ATTENTION_SPLITS=3" "$hooks" \
    --set query_tokens=1025 --set context_tokens=2817
  expect s$s-many-splits 0 "52 context splits of 1280 keys" "LP_ATTENTION_SPLITS=64" "$hooks" \
    --set query_tokens=301 --set context_tokens=65537 --set input_variant=1
  # The largest query extent: the partial-buffer cap allows eight splits.
  expect s$s-largest-query 0 "8 context splits of 256 keys" "LP_ATTENTION_SPLITS=12" "$hooks" \
    --set query_tokens=4194304 --set context_tokens=2048
  # The largest context: splits of at most 2^21 keys keep every float chain
  # as short as the baseline's, even when fewer are asked for.
  expect s$s-largest-context 0 "8 context splits of 2097152 keys" "LP_ATTENTION_SPLITS=2" "$hooks" \
    --set query_tokens=21 --set context_tokens=16777216 --set input_variant=2
  # Invalid hook values are usage errors.
  expect s$s-invalid-splits-zero 2 "" "LP_ATTENTION_SPLITS=0" "$hooks" \
    --set query_tokens=4 --set context_tokens=4
  expect s$s-invalid-splits-text 2 "" "LP_ATTENTION_SPLITS=12x" "$hooks" \
    --set query_tokens=4 --set context_tokens=4
  # The mutant drops the last key of every tile; the check must reject it.
  expect s$s-mutant-skip-key 1 "" "" "$mutant" --set query_tokens=7 --set context_tokens=300
  expect s$s-mutant-skip-key-large 1 "" "" "$mutant" \
    --set query_tokens=129 --set context_tokens=5000
done

# Stage 1 asks for one split; the key bound alone raises it at the largest context.
expect s1-largest-context-default 0 "8 context splits of 2097152 keys" "" \
  "$build/attention_stage1" --set query_tokens=21 --set context_tokens=16777216 --set input_variant=2

# The exact row-maxima fallback (stages 3-6, where the shift is a bound).
for s in 3 4 5 6; do
  hooks=$build/attention_stage${s}_hooks
  expect s$s-exact-splits5 0 "5 context splits of 13312 keys, exact row maxima" \
    "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=5" "$hooks" \
    --set query_tokens=2000 --set context_tokens=65537
  expect s$s-exact-short 0 "1 context splits of 256 keys, exact row maxima" \
    "LP_ATTENTION_EXACT_MAXIMUM=1" "$hooks" \
    --set query_tokens=300 --set context_tokens=7 --set input_variant=2
  expect s$s-exact-small 0 "4 context splits of 256 keys, exact row maxima" \
    "LP_ATTENTION_EXACT_MAXIMUM=1" "$hooks" --set query_tokens=37 --set context_tokens=1000
  expect s$s-invalid-exact 2 "" "LP_ATTENTION_EXACT_MAXIMUM=2" "$hooks" \
    --set query_tokens=4 --set context_tokens=4
done

# Stage 6's overlapped application path: one, two and three splits (both
# side streams), and the compute path in the checked invocation.
hooks6=$build/attention_stage6_hooks
expect s6-overlap-1 0 "1 context splits of 3072 keys" "LP_ATTENTION_SPLITS=1" "$hooks6" \
  --set query_tokens=1025 --set context_tokens=2817
expect s6-overlap-2 0 "2 context splits of 1536 keys" "LP_ATTENTION_SPLITS=2" "$hooks6" \
  --set query_tokens=1025 --set context_tokens=2817
expect s6-compute-path 0 "keys, application runs the compute path" \
  "LP_ATTENTION_CHECK_COMPUTE=1" "$hooks6" --set query_tokens=2000 --set context_tokens=65537
expect s6-compute-path-splits 0 "4 context splits of 256 keys, application runs the compute path" \
  "LP_ATTENTION_CHECK_COMPUTE=1 LP_ATTENTION_SPLITS=7" "$hooks6" \
  --set query_tokens=50 --set context_tokens=999
expect s6-invalid-check-compute 2 "" "LP_ATTENTION_CHECK_COMPUTE=yes" "$hooks6" \
  --set query_tokens=4 --set context_tokens=4

# Digests. Stages 4 and 6 declare no arithmetic change, so they must
# reproduce the previous stage's digest exactly; so must stage 6 and the
# production hooks build at the same settings (12 splits unless forced), and
# stage 6's two launch paths. The runs above supply most of the pairs.
for c in "2000 65537 0" "37 37 1" "1 1 0"; do
  set -- $c
  for s in 3 4 5 6; do
    expect s$s-timed-$1x$2v$3 0 "" "" "$build/attention_stage$s" \
      --set query_tokens=$1 --set context_tokens=$2 --set input_variant=$3
  done
done
reference=$build/attention_hooks_reference
expect ref-default 0 "10 context splits of 512 keys" "LP_ATTENTION_SPLITS=12" "$reference" \
  --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
expect ref-timed-2000x65537v0 0 "" "LP_ATTENTION_SPLITS=12" "$reference" \
  --set query_tokens=2000 --set context_tokens=65537
expect ref-timed-37x37v1 0 "" "LP_ATTENTION_SPLITS=12" "$reference" \
  --set query_tokens=37 --set context_tokens=37 --set input_variant=1
expect ref-timed-1x1v0 0 "" "LP_ATTENTION_SPLITS=12" "$reference" \
  --set query_tokens=1 --set context_tokens=1
expect ref-one-split 0 "" "LP_ATTENTION_SPLITS=1" "$reference" \
  --set query_tokens=129 --set context_tokens=5000 --set input_variant=2
expect ref-three-splits 0 "" "LP_ATTENTION_SPLITS=3" "$reference" \
  --set query_tokens=1025 --set context_tokens=2817
expect ref-many-splits 0 "" "LP_ATTENTION_SPLITS=64" "$reference" \
  --set query_tokens=301 --set context_tokens=65537 --set input_variant=1
expect ref-exact-splits5 0 "exact row maxima" "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=5" \
  "$reference" --set query_tokens=2000 --set context_tokens=65537
expect ref-exact-short 0 "exact row maxima" "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=12" \
  "$reference" --set query_tokens=300 --set context_tokens=7 --set input_variant=2
expect ref-exact-small 0 "exact row maxima" "LP_ATTENTION_EXACT_MAXIMUM=1 LP_ATTENTION_SPLITS=12" \
  "$reference" --set query_tokens=37 --set context_tokens=1000
for run in default timed-2000x65537v0 timed-37x37v1 timed-1x1v0 one-split three-splits \
           many-splits exact-splits5 exact-short exact-small; do
  same_digest digest-s4-eq-s3-$run s3-$run s4-$run
  same_digest digest-s6-eq-s5-$run s5-$run s6-$run
  same_digest digest-s6-eq-reference-$run s6-$run ref-$run
done
same_digest digest-s6-compute-path-eq-application s6-timed-2000x65537v0 s6-compute-path

echo "attention stage tests: $passed passed, $failures failed"
[[ $failures -eq 0 ]]
