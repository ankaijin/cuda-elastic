#!/usr/bin/env bash
# GPU tests of one decoder stage beyond the application's own tests, run by
# `make test` under the GPU lock (this script does not lock).
#   run_stage_tests.sh <stage> <build dir> <baseline binary> <scratch dir>
# <build dir> holds decoder_stage<stage> (timed) and its test-only builds
# decoder_stage<stage>_tuning and _mutant1/_mutant2, which honour the
# production hooks DECODER_OPT_TILE, DECODER_OPT_SCHEDULE and DECODER_OPT_CTAS
# and print the applied plan ("decoder test tuning: ...").
#
# 1. The timed binary applies the stage's default plan and ignores the hooks.
# 2. Digest equality with the baseline binary on shapes with tails in every
#    dimension and on widths around the pipeline depth (1 to 4 k-stages, a
#    partial last stage), for every tile, the stage's own schedule, and
#    the other schedule with forced persistent grids of 1, 5, 7 and 11 blocks
#    (stages 1-3 then run their copy loop on partial k-ranges).
# 3. Forced grids really split tiles; invalid hook values exit 2.
# 4. The production kernel mutants built from this stage must fail the check.
set -uo pipefail

usage="usage: run_stage_tests.sh <stage> <build dir> <baseline> <scratch dir>"
stage="${1:?$usage}"
build="${2:?$usage}"
base="${3:?$usage}"
scratch="${4:?$usage}"
bin="$build/decoder_stage$stage"
tuning="$build/decoder_stage${stage}_tuning"
mkdir -p "$scratch"
passed=0
failed=0
limit=300

own=dp
other=streamk
if [[ $stage -ge 4 ]]; then own=streamk; other=dp; fi

report() { # report <ok> <name> <detail>
  if [[ $1 -eq 1 ]]; then
    echo "PASS $2"
    passed=$((passed + 1))
  else
    echo "FAIL $2${3:+ ($3)}"
    sed 's/^/    /' "$scratch/err" | tail -n 20
    failed=$((failed + 1))
  fi
}
sets() { local a; for a in $1; do printf -- '--set %s ' "$a"; done; }
all_pass() { # 4 passing invocations (Direct and Graph, twice) and status pass
  [[ $(grep -c ': PASS, digest' "$scratch/err") -eq 4 ]] && grep -q '"status": "pass"' "$scratch/out"
}
digest_of() { grep -o 'direct invocation 1: PASS, digest [0-9a-f]*' "$1" | awk '{print $NF}'; }

# 1. The timed binary: the stage's default plan, hooks ignored.
small="batch=300 width=1001 vocab=999 steps=3"
# shellcheck disable=SC2046
env DECODER_OPT_TILE=bogus DECODER_OPT_SCHEDULE=bogus DECODER_OPT_CTAS=0 \
  timeout $limit "$bin" $(sets "$small") >"$scratch/out" 2>"$scratch/err"
code=$?
ok=1
[[ $code -eq 0 ]] && all_pass || ok=0
grep -q "decoder stage $stage: tile 128x256x16, schedule $own," "$scratch/err" || ok=0
grep -q 'decoder test tuning' "$scratch/err" && ok=0
report $ok "stage $stage: timed binary uses the default plan and ignores the hooks" "exit $code"

# 2. Digest equality with the baseline.
shapes=(
  "batch=300 width=1001 vocab=999 steps=3"
  "batch=1030 width=1000 vocab=1000 steps=3"
  "batch=257 width=520 vocab=513 steps=2"
  "batch=40 width=96 vocab=16 steps=5"
  "batch=3 width=2 vocab=2 steps=2"
  "batch=130 width=8 vocab=8 steps=2"
  "batch=130 width=9 vocab=9 steps=2"
  "batch=130 width=16 vocab=16 steps=2"
  "batch=130 width=17 vocab=17 steps=2"
  "batch=130 width=24 vocab=24 steps=2"
  "batch=130 width=40 vocab=40 steps=2"
  "batch=130 width=48 vocab=48 steps=2"
  "batch=130 width=49 vocab=49 steps=2"
)
for shape in "${shapes[@]}"; do
  # shellcheck disable=SC2046
  timeout $limit "$base" $(sets "$shape") >/dev/null 2>"$scratch/base"
  want="$(digest_of "$scratch/base")"
  for tile in 128x256x16 128x256x8 128x128x8; do
    for setting in "" "DECODER_OPT_SCHEDULE=$other" \
        "DECODER_OPT_SCHEDULE=streamk DECODER_OPT_CTAS=1" \
        "DECODER_OPT_SCHEDULE=streamk DECODER_OPT_CTAS=5" \
        "DECODER_OPT_SCHEDULE=streamk DECODER_OPT_CTAS=7" \
        "DECODER_OPT_SCHEDULE=streamk DECODER_OPT_CTAS=11"; do
      name="stage $stage: digest [$shape] tile=$tile ${setting:-default}"
      # shellcheck disable=SC2046,SC2086
      env DECODER_OPT_TILE="$tile" $setting timeout $limit "$tuning" $(sets "$shape") \
        >"$scratch/out" 2>"$scratch/err"
      code=$?
      got="$(digest_of "$scratch/err")"
      plan="$(grep 'decoder test tuning' "$scratch/err")"
      schedule=$own
      [[ "$setting" == *SCHEDULE=* ]] && schedule="${setting#DECODER_OPT_SCHEDULE=}" && schedule="${schedule%% *}"
      ok=1
      [[ $code -eq 0 && -n "$want" && "$got" == "$want" ]] && all_pass || ok=0
      [[ "$plan" == *"tile $tile, schedule $schedule,"* ]] || ok=0
      [[ "$setting" == *CTAS=* ]] && { [[ "$plan" == *"ctas ${setting##*=},"* ]] || ok=0; }
      report $ok "$name" "exit $code, digest ${got:-none}, baseline ${want:-none}"
    done
  done
done

# 3. Forced stream-K grids split tiles between blocks; invalid hooks exit 2.
for spec in "5 128x256x16 batch=300 width=1001 vocab=999 steps=3" \
    "7 128x256x8 batch=1030 width=1000 vocab=1000 steps=3" \
    "11 128x128x8 batch=1030 width=1000 vocab=1000 steps=3"; do
  set -- $spec
  ctas=$1 tile=$2
  shift 2
  # shellcheck disable=SC2046
  env DECODER_OPT_TILE=$tile DECODER_OPT_SCHEDULE=streamk DECODER_OPT_CTAS=$ctas \
    timeout $limit "$tuning" $(sets "$*") >"$scratch/out" 2>"$scratch/err"
  code=$?
  ok=1
  [[ $code -eq 0 ]] && all_pass || ok=0
  grep -Eq 'split tiles [1-9]' "$scratch/err" || ok=0
  report $ok "stage $stage: stream-K split with $ctas blocks, tile $tile [$*]" "exit $code"
done

expect_exit() { # expect_exit <name> <code> <stderr pattern> -- <env and args...>
  local name="$1" want="$2" pattern="$3"
  shift 4
  env "$@" >"$scratch/out" 2>"$scratch/err"
  local code=$? ok=1
  [[ $code -eq $want ]] || ok=0
  [[ -z "$pattern" ]] || grep -Eq "$pattern" "$scratch/err" || ok=0
  report $ok "stage $stage: $name" "exit $code, expected $want"
}
tiny=(--set batch=40 --set width=96 --set vocab=16 --set steps=5)
expect_exit unknown-tile 2 'DECODER_OPT_TILE' -- DECODER_OPT_TILE=64x64 "$tuning" "${tiny[@]}"
expect_exit bad-schedule 2 'DECODER_OPT_SCHEDULE' -- DECODER_OPT_SCHEDULE=split "$tuning" "${tiny[@]}"
expect_exit ctas-zero 2 'DECODER_OPT_CTAS' -- DECODER_OPT_CTAS=0 "$tuning" "${tiny[@]}"

# 4. Mutants: 1 drops the partial sums a stream-K block hands on, 2 drops the
# last k-stage of every tile (vocab == width, so it holds nonzero columns).
expect_exit mutant-handoff 1 'FAIL' -- DECODER_OPT_TILE=128x256x16 DECODER_OPT_SCHEDULE=streamk \
  DECODER_OPT_CTAS=5 "$build/decoder_stage${stage}_mutant1" \
  --set batch=300 --set width=1001 --set vocab=999 --set steps=3
for tile in 128x256x16 128x256x8 128x128x8; do
  expect_exit "mutant-last-stage tile=$tile" 1 'FAIL' -- DECODER_OPT_TILE=$tile \
    "$build/decoder_stage${stage}_mutant2" --set batch=300 --set width=1000 --set vocab=1000 --set steps=3
done

echo "decoder stage $stage tests: $passed passed, $failed failed"
[[ $failed -eq 0 ]]
