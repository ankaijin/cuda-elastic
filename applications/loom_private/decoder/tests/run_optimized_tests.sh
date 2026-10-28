#!/usr/bin/env bash
# Tests for the optimized decoder: run by `make test VARIANT=optimized`, which
# already holds the GPU lock (this script must not take it again).
# Usage: run_optimized_tests.sh <optimized> <tuning> <baseline> <mutant 1>
#                               <mutant 2>
# <tuning> and the mutants are test-only builds (-DLP_DECODER_TEST_TUNING)
# that honour the DECODER_OPT_TILE / _SCHEDULE / _CTAS hooks and print the
# applied plan ("decoder test tuning: ..."); <optimized> must ignore them.
#
# 1. Every baseline test case (tests/run_tests.sh) against the optimized binary,
#    and the optimized binary ignoring the hooks.
# 2. Small shapes with tails in every dimension, for every tile configuration,
#    both schedules and forced persistent grids: each must apply the forced
#    plan, pass the check, and give exactly the baseline binary's digest.
# 3. Invalid hook values exit 2.
# 4. Kernel mutants must be rejected by the check (exit 1).
set -uo pipefail

usage="usage: run_optimized_tests.sh <optimized> <tuning> <baseline> <mutant1> <mutant2>"
opt="${1:?$usage}"
tuning="${2:?$usage}"
base="${3:?$usage}"
mutant1="${4:?$usage}"
mutant2="${5:?$usage}"
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
sets() { local a; for a in $1; do printf -- '--set %s ' "$a"; done; }
all_pass() { # 4 passing invocations (Direct and Graph, twice) and status pass
  [[ $(grep -c ': PASS, digest' "$log/err") -eq 4 ]] && grep -q '"status": "pass"' "$log/out"
}

# 1. The baseline's cases, and the production binary ignoring the hooks.
if bash "$here/run_tests.sh" "$opt"; then
  report 1 "baseline test cases"
else
  : >"$log/err"
  report 0 "baseline test cases"
fi
env DECODER_OPT_TILE=bogus DECODER_OPT_SCHEDULE=bogus DECODER_OPT_CTAS=0 \
  "$opt" --set batch=300 --set width=1001 --set vocab=999 --set steps=3 >"$log/out" 2>"$log/err"
code=$?
ok=1
[[ $code -eq 0 ]] && all_pass || ok=0
grep -q 'decoder test tuning' "$log/err" && ok=0
report $ok "optimized binary ignores the tuning hooks" "exit $code"

# 2. Digest equality with the baseline. Shapes: odd width (4-byte tails and a
# partial last stage), ragged sequence and row tiles, prompt wraparound, and
# two single-tile shapes (where every grid runs one data-parallel tile).
shapes=(
  "batch=300 width=1001 vocab=999 steps=3"
  "batch=1030 width=1000 vocab=1000 steps=3"
  "batch=257 width=520 vocab=513 steps=2"
  "batch=40 width=96 vocab=16 steps=5"
  "batch=3 width=2 vocab=2 steps=2"
)
digest_of() { grep -o 'direct invocation 1: PASS, digest [0-9a-f]*' "$1" | awk '{print $NF}'; }

for shape in "${shapes[@]}"; do
  # shellcheck disable=SC2046
  "$base" $(sets "$shape") >/dev/null 2>"$log/base"
  want="$(digest_of "$log/base")"
  for tile in 128x256x16 128x256x8 128x128x8; do
    for setting in "" DECODER_OPT_SCHEDULE=dp DECODER_OPT_CTAS=1 \
        DECODER_OPT_CTAS=5 DECODER_OPT_CTAS=7 DECODER_OPT_CTAS=11; do
      name="digest [$shape] tile=$tile ${setting:-default}"
      # shellcheck disable=SC2046
      env DECODER_OPT_TILE="$tile" $setting "$tuning" $(sets "$shape") >"$log/out" 2>"$log/err"
      code=$?
      got="$(digest_of "$log/err")"
      plan="$(grep 'decoder test tuning' "$log/err")"
      ok=1
      [[ $code -eq 0 && -n "$want" && "$got" == "$want" ]] && all_pass || ok=0
      # The hooks were applied.
      [[ "$plan" == *"tile $tile,"* ]] || ok=0
      case "$setting" in
        DECODER_OPT_SCHEDULE=dp) [[ "$plan" == *"schedule dp,"* && "$plan" == *"stream-K tiles 0,"* ]] || ok=0 ;;
        DECODER_OPT_CTAS=*) [[ "$plan" == *"schedule streamk, ctas ${setting#*=},"* ]] || ok=0 ;;
        *) [[ "$plan" == *"schedule streamk,"* ]] || ok=0 ;;
      esac
      report $ok "$name" "exit $code, digest ${got:-none}, baseline ${want:-none}"
    done
  done
done

# The forced grids on multi-tile shapes really split tiles between blocks.
expect_split() { # expect_split <ctas> <tile> <shape>
  # shellcheck disable=SC2046
  env DECODER_OPT_TILE="$2" DECODER_OPT_CTAS="$1" \
    "$tuning" $(sets "$3") >"$log/out" 2>"$log/err"
  local code=$? ok=1
  [[ $code -eq 0 ]] && all_pass || ok=0
  grep -Eq 'split tiles [1-9]' "$log/err" || ok=0
  report $ok "stream-K split with $1 blocks, tile $2 [$3]" "exit $code"
}
expect_split 5 128x256x16 "batch=300 width=1001 vocab=999 steps=3"
expect_split 7 128x256x8 "batch=1030 width=1000 vocab=1000 steps=3"
expect_split 11 128x128x8 "batch=1030 width=1000 vocab=1000 steps=3"

# 3. Invalid hook values.
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
expect_exit unknown-tile 2 'DECODER_OPT_TILE' -- DECODER_OPT_TILE=64x64 "$tuning" "${small[@]}"
expect_exit empty-tile 2 'DECODER_OPT_TILE' -- DECODER_OPT_TILE= "$tuning" "${small[@]}"
expect_exit bad-schedule 2 'DECODER_OPT_SCHEDULE' -- DECODER_OPT_SCHEDULE=split "$tuning" "${small[@]}"
expect_exit ctas-zero 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=0 "$tuning" "${small[@]}"
expect_exit ctas-text 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=5x "$tuning" "${small[@]}"
expect_exit ctas-too-many 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=1000000 "$tuning" "${small[@]}"

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
