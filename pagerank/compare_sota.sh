#!/usr/bin/env bash
# Checks that the SOTA PageRank (pagerank_sota.cu) produces the same scores
# as the baseline in graph mode and in direct mode.  For each input below,
# the baseline (graph and direct launches) and the SOTA (direct, its
# default, and graph) each run one short benchmark (PAGERANK_MODE=bench, the
# default rotation of each binary: 1 for the baseline, 3 for the SOTA) and dumps the
# benchmark input's final float32 scores (PAGERANK_DUMP); the SOTA also
# dumps the double-precision CPU oracle for it (PAGERANK_DUMP_REF).  Every
# dump must match the oracle within the validation tolerance
# (|x - ref| <= 1e-6/V + 1e-4 |ref|, score sum within 1e-4 relative), and
# the SOTA must match each baseline dump within the same tolerance.
#
# Inputs: the benchmark fixture with a uniform start and three skewed starts,
# the irregular validation graph (uniform and skewed), and two random graphs
# of the benchmark's shape (uniform and skewed).  The fixture's skewed starts
# converge to its uniform stationary scores within the twenty iterations
# (spread 0), so the irregular and random cases are the discriminating ones.
#
# Usage: pagerank/compare_sota.sh   (or: make -C pagerank compare)
# Environment: BUILD (binary dir, default <repo>/<gpu>/pagerank/build),
# LOCK (GPU lock file), GRAPH_REPLAYS (default 3), OUT (dump dir, default a
# temporary directory).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD="${BUILD:-$ROOT/$("$ROOT/gpu.sh" name)/pagerank/build}"
LOCK="${LOCK:-/tmp/cuda-elastic-gpu.lock}"
REPLAYS="${GRAPH_REPLAYS:-3}"
SOTA="$BUILD/pagerank_sota"
BASELINE="$BUILD/gapbs_pagerank_cuda_graph"
make -C "$SCRIPT_DIR" --no-print-directory sota baseline BUILD="$BUILD" >/dev/null

OUT="${OUT:-$(mktemp -d)}"
mkdir -p "$OUT"
echo "dumps in $OUT"

CASES=(
  "fixture uniform"
  "fixture skewed:1"
  "fixture skewed:2"
  "fixture skewed:3"
  "irregular uniform"
  "irregular skewed:7"
  "random:11 uniform"
  "random:12 skewed:5"
)

run() {  # run <name> <binary> <launch> <graph> <start> [extra env...]
  local name="$1" bin="$2" launch="$3" graph="$4" start="$5"; shift 5
  local log="$OUT/$name.log"
  if ! env PAGERANK_MODE=bench GRAPH_REPLAYS="$REPLAYS" PAGERANK_LAUNCH="$launch" \
      PAGERANK_GRAPH="$graph" PAGERANK_START="$start" PAGERANK_DUMP="$OUT/$name.f32" "$@" \
      flock "$LOCK" "$bin" >"$log" 2>&1; then
    echo "FAIL: $name exited with an error; see $log"
    tail -n 5 "$log"
    exit 1
  fi
}

status=0
for case in "${CASES[@]}"; do
  read -r graph start <<<"$case"
  tag="${graph//:/-}_${start//:/-}"
  run "$tag.baseline-graph" "$BASELINE" graph "$graph" "$start"
  run "$tag.baseline-direct" "$BASELINE" direct "$graph" "$start"
  run "$tag.sota" "$SOTA" direct "$graph" "$start" PAGERANK_DUMP_REF="$OUT/$tag.ref.f32"
  run "$tag.sota-graph" "$SOTA" graph "$graph" "$start"
  python3 -I - "$OUT" "$tag" "$graph/$start" <<'EOF' || status=1
import sys
import numpy as np
out, tag, label = sys.argv[1:4]
ref = np.fromfile(f"{out}/{tag}.ref.f32", dtype=np.float32).astype(np.float64)
dumps = {k: np.fromfile(f"{out}/{tag}.{k}.f32", dtype=np.float32).astype(np.float64)
         for k in ("baseline-graph", "baseline-direct", "sota", "sota-graph")}
V = ref.size
ok = True

def check(name, x, want):
    global ok
    err = np.abs(x - want)
    tol = 1e-6 / V + 1e-4 * np.abs(want)
    bad = int(np.count_nonzero(~(err <= tol)))
    sum_ok = abs(x.sum() - want.sum()) <= 1e-4 * abs(want.sum())
    rel = float(np.max(err / np.where(want != 0, np.abs(want), np.inf)))
    passed = bad == 0 and sum_ok and x.size == want.size
    ok &= passed
    return f"{name} {'ok' if passed else 'FAIL'} (max_rel {rel:.2e}{'' if bad == 0 else f', {bad} outside'}{'' if sum_ok else ', sum off'})"

parts = [check(f"{k} vs oracle", x, ref) for k, x in dumps.items()]
parts.append(check("sota vs baseline-graph", dumps["sota"], dumps["baseline-graph"]))
parts.append(check("sota vs baseline-direct", dumps["sota"], dumps["baseline-direct"]))
spread = (ref.max() - ref.min()) / ref.mean()
print(f"{'PASS' if ok else 'FAIL'}  {label:<22} V={V:<8} spread={spread:.3g}: " + "; ".join(parts))
sys.exit(0 if ok else 1)
EOF
done
if [[ $status -eq 0 ]]; then echo "compare_sota: all cases match"; else echo "compare_sota: MISMATCH"; fi
exit $status
