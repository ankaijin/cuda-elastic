#!/usr/bin/env bash
set -euo pipefail

# Runs the baseline in graph mode, the baseline in direct mode and the SOTA
# (default launch and --rotate) on identical inputs, and checks that their
# --dump results agree within the oracle tolerance |a - b| <= 1e-5*|b| + 1e-7
# (b = baseline graph). Inputs: smoke, and random with seeds 1, 2, 3; shapes
# 16, 640, 64M and two odd counts (1000003 on the SOTA's zero-copy path,
# 16777219 on its copy pipeline). Every binary must also pass its own check
# against the double-precision reference (exit status 0); the SOTA checks all
# of its rotated input sets. The SOTA's set 0 is the baseline's input.
#
# Run from this directory after `make`. Prefix every GPU run with RUN, e.g.
#   RUN="flock /tmp/cuda-elastic-gpu.lock" ./compare_sota.sh
# Override the shapes or seeds with COUNTS="16 640" SEEDS="1 2 3".
BUILD="${BUILD:-../$(../gpu.sh name)/mlperf-anomaly/build}"
BASE="$BUILD/anomaly_mse"
SOTA="$BUILD/anomaly_mse_sota"
COUNTS="${COUNTS:-16 640 1000003 16777219 67108864}"
SEEDS="${SEEDS:-1 2 3}"
REPLAYS="${REPLAYS:-3}"
RUN="${RUN:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for bin in "$BASE" "$SOTA"; do
  [[ -x "$bin" ]] || { echo "missing $bin (run make)" >&2; exit 2; }
done

result() {  # result <dump file>: set 0's result
  sed -n 's/^count=[0-9]* set=0 result=\([^ ]*\) .*/\1/p' "$1"
}

failures=0
runs=0
printf '%-10s %-12s %-16s %-16s %-16s %s\n' count inputs base-graph \
  base-direct sota status
for count in $COUNTS; do
  cases=("smoke 1")
  for seed in $SEEDS; do cases+=("random $seed"); done
  for entry in "${cases[@]}"; do
    read -r inputs seed <<< "$entry"
    args=(--count="$count" --inputs="$inputs" --seed="$seed"
          --replays="$REPLAYS")
    status=ok
    $RUN "$BASE" "${args[@]}" --launch=graph --dump="$TMP/graph" \
      > "$TMP/graph.log" || status="FAIL(base-graph check)"
    $RUN "$BASE" "${args[@]}" --launch=direct --dump="$TMP/direct" \
      > "$TMP/direct.log" || status="FAIL(base-direct check)"
    $RUN "$SOTA" "${args[@]}" --dump="$TMP/sota" \
      > "$TMP/sota.log" || status="FAIL(sota check)"
    graph=$(result "$TMP/graph")
    direct=$(result "$TMP/direct")
    sota=$(result "$TMP/sota")
    if [[ $status == ok ]]; then
      for other in "$direct" "$sota"; do
        awk -v a="$other" -v b="$graph" 'BEGIN {
              d = a - b; if (d < 0) d = -d; m = b < 0 ? -b : b;
              exit !(a == a && d <= 1e-5 * m + 1e-7) }' ||
          status="FAIL(mismatch)"
      done
    fi
    label="$inputs"
    [[ $inputs == random ]] && label="random/$seed"
    printf '%-10s %-12s %-16s %-16s %-16s %s\n' "$count" "$label" "$graph" \
      "$direct" "$sota" "$status"
    runs=$((runs + 1))
    [[ $status == ok ]] || failures=$((failures + 1))
  done
done
echo "$((runs - failures))/$runs cases match"
[[ $failures -eq 0 ]]
