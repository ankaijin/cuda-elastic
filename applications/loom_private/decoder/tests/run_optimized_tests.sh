#!/usr/bin/env bash
# Tests for the optimized decoder: run by `make test VARIANT=optimized`, which
# already holds the GPU lock (this script must not take it again).
# Usage: run_optimized_tests.sh <optimized> <baseline> <mutant 1> <mutant 2>
#
# 1. Every baseline test case (tests/run_tests.sh) against the optimized binary.
# 2. Small shapes with tails in every dimension, for every tile configuration
#    and schedule, including forced small stream-K grids (DECODER_OPT_CTAS):
#    each must pass the check and give exactly the baseline binary's digest.
# 3. Invalid test overrides exit 2.
# 4. Kernel mutants must be rejected by the check (exit 1).
set -uo pipefail

opt="${1:?usage: run_optimized_tests.sh <optimized> <baseline> <mutant1> <mutant2>}"
base="${2:?missing baseline binary}"
mutant1="${3:?missing mutant 1 binary}"
mutant2="${4:?missing mutant 2 binary}"
here="$(cd "$(dirname "$0")" && pwd)"
log="$(mktemp -d)"
trap 'rm -rf "$log"' EXIT
passed=0
failed=0

report() { # report <ok> <name> <detail>
  if [[ $1 -eq 1 ]]; then
    echo "PASS $2"
    passed=$((passed + 1))
  else
    echo "FAIL $2${3:+ ($3)}"
    sed 's/^/    /' "$log/err" | tail -n 20
    failed=$((failed + 1))
  fi
}

# 1. The baseline's cases.
if bash "$here/run_tests.sh" "$opt"; then
  report 1 "baseline test cases"
else
  : >"$log/err"
  report 0 "baseline test cases"
fi

# 2. Digest equality with the baseline. Shapes: odd width (4-byte tails and a
# partial last stage), ragged sequence and row tiles, prompt wraparound, a
# single tile, and the smallest legal model.
shapes=(
  "batch=300 width=1001 vocab=999 steps=3"
  "batch=1030 width=1000 vocab=1000 steps=3"
  "batch=40 width=96 vocab=16 steps=5"
  "batch=257 width=520 vocab=513 steps=2"
  "batch=3 width=2 vocab=2 steps=2"
)
digest_of() { grep -o 'direct invocation 1: PASS, digest [0-9a-f]*' "$1" | awk '{print $NF}'; }
sets() { local a; for a in $1; do printf -- '--set %s ' "$a"; done; }

for shape in "${shapes[@]}"; do
  # shellcheck disable=SC2046
  "$base" $(sets "$shape") >/dev/null 2>"$log/base"
  want="$(digest_of "$log/base")"
  for tile in 128x256x16 128x256x8 128x128x8; do
    for setting in "" DECODER_OPT_SCHEDULE=dp DECODER_OPT_CTAS=1 \
        DECODER_OPT_CTAS=5 DECODER_OPT_CTAS=7 DECODER_OPT_CTAS=11; do
      name="digest [$shape] tile=$tile ${setting:-default}"
      # shellcheck disable=SC2046
      env DECODER_OPT_TILE="$tile" DECODER_OPT_VERBOSE=1 $setting \
        "$opt" $(sets "$shape") >"$log/out" 2>"$log/err"
      code=$?
      got="$(digest_of "$log/err")"
      ok=1
      [[ $code -eq 0 && -n "$want" && "$got" == "$want" ]] || ok=0
      [[ $(grep -c ': PASS, digest' "$log/err") -eq 4 ]] || ok=0
      grep -q '"status": "pass"' "$log/out" || ok=0
      report $ok "$name" "exit $code, digest ${got:-none}, baseline ${want:-none}"
    done
  done
done

# The forced grids above must really split tiles between blocks.
expect_streamk() { # expect_streamk <ctas> <tile> <shape>
  # shellcheck disable=SC2046
  env DECODER_OPT_TILE="$2" DECODER_OPT_CTAS="$1" DECODER_OPT_VERBOSE=1 \
    "$opt" --launch direct $(sets "$3") >"$log/out" 2>"$log/err"
  local ok=1
  grep -Eq 'stream-K tiles [1-9]' "$log/err" || ok=0
  report $ok "stream-K split with $1 blocks, tile $2 [$3]"
}
expect_streamk 5 128x256x16 "batch=300 width=1001 vocab=999 steps=3"
expect_streamk 7 128x256x8 "batch=1030 width=1000 vocab=1000 steps=3"
expect_streamk 11 128x128x8 "batch=1030 width=1000 vocab=1000 steps=3"

# 3. Invalid test overrides.
expect_exit() { # expect_exit <name> <code> <stderr pattern> -- <env and args...>
  local name="$1" want="$2" pattern="$3"
  shift 4
  env "$@" >"$log/out" 2>"$log/err"
  local code=$? ok=1
  [[ $code -eq $want ]] || ok=0
  [[ -z "$pattern" ]] || grep -Eq "$pattern" "$log/err" || ok=0
  report $ok "$name" "exit $code, expected $want"
}
small=(--set batch=40 --set width=96 --set vocab=16 --set steps=5)
expect_exit unknown-tile 2 'DECODER_OPT_TILE' -- DECODER_OPT_TILE=64x64 "$opt" "${small[@]}"
expect_exit bad-schedule 2 'DECODER_OPT_SCHEDULE' -- DECODER_OPT_SCHEDULE=split "$opt" "${small[@]}"
expect_exit ctas-zero 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=0 "$opt" "${small[@]}"
expect_exit ctas-text 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=abc "$opt" "${small[@]}"
expect_exit ctas-too-many 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=1000000 "$opt" "${small[@]}"

# 4. Mutants: 1 drops the partial sums a stream-K block hands on, 2 drops the
# last stage of every tile.
expect_exit mutant-handoff 1 'FAIL' -- DECODER_OPT_TILE=128x256x16 DECODER_OPT_CTAS=5 \
  "$mutant1" --set batch=300 --set width=1001 --set vocab=999 --set steps=3
# vocab == width, so the last stage holds classifier columns that are not 0.
for tile in 128x256x16 128x256x8 128x128x8; do
  expect_exit "mutant-last-stage tile=$tile" 1 'FAIL' -- DECODER_OPT_TILE=$tile \
    "$mutant2" --set batch=300 --set width=1000 --set vocab=1000 --set steps=3
done

echo "decoder optimized tests: $passed passed, $failed failed"
[[ $failed -eq 0 ]]
