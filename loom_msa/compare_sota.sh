#!/usr/bin/env bash
# Output matching for the SOTA variant: msa_optimized in graph mode, msa_optimized
# in direct mode and msa_sota, run on the same inputs, must agree within the
# harness tolerance (|d| <= 1e-4 + 1e-4 * |optimized graph|) on every output
# element and statistic of every input set.
#
#   compare_sota.sh [BUILD_DIR]        (default: <repo>/<gpu>/loom_msa/build)
#   SHAPES="4x4 medium" compare_sota.sh   (default: 4x4 medium large)
#   SEEDS="..." compare_sota.sh           (default: 3 seeds, the first being the
#                                          harness default)
#
# Inputs: repeat, and random with every seed. msa_sota runs at its default
# rotation (3 input sets per run, see msa_harness.cuh --rotate); msa_optimized
# is given the same --rotate so all of its input sets are dumped and compared
# set by set. Every binary also runs its own check against the CPU reference
# (PASSED required). Each run holds the GPU lock on its own.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:-$HERE/../$("$HERE/../gpu.sh" name)/loom_msa/build}"
SHAPES="${SHAPES:-4x4 medium large}"
SEEDS="${SEEDS:-0x9e3779b9 0x2545f491 0x6c8e9cf5}"
LOCK="${LOCK:-/tmp/cuda-elastic-gpu.lock}"
ROTATE="${ROTATE:-3}"  # msa_sota's default rotation (SOTA_DEFAULT_ROTATION)
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

run() {  # run <log> <binary> <args...>
  local log="$1"
  shift
  if ! flock "$LOCK" "$@" > "$log" 2>&1 || ! grep -q '^PASSED$' "$log"; then
    echo "FAILED: $*" >&2
    cat "$log" >&2
    exit 1
  fi
}

compare() {  # compare <reference dump> <dump>...
  python3 -I - "$@" <<'EOF'
import struct, sys
def load(path):
    data = open(path, "rb").read()
    return struct.unpack("<%df" % (len(data) // 4), data)
reference = load(sys.argv[1])
ok = True
for path in sys.argv[2:]:
    values = load(path)
    if len(values) != len(reference):
        print("  size mismatch: %s" % path)
        ok = False
        continue
    worst, used, bad = 0.0, 0.0, 0
    for r, v in zip(reference, values):
        d = abs(v - r)
        tolerance = 1e-4 + 1e-4 * abs(r)
        if not d <= tolerance:  # NaN fails
            bad += 1
        if not d <= worst:
            worst = d
        if not d / tolerance <= used:
            used = d / tolerance
    print("  %-34s max |diff| %.3g, max diff/tolerance %.3g, %d outside"
          % (path.rsplit(".", 1)[-1], worst, used, bad))
    ok = ok and bad == 0
sys.exit(0 if ok else 1)
EOF
}

status=0
for shape in $SHAPES; do
  configs=("--inputs=repeat")
  for seed in $SEEDS; do
    configs+=("--inputs=random --seed=$seed")
  done
  for config in "${configs[@]}"; do
    tag="$shape $(echo "$config" | tr -d '-' | tr ' =' '_')"
    name="${tag// /_}"
    echo "$shape $config:"
    # shellcheck disable=SC2086
    run "$WORK/$name.og.log" "$BUILD/msa_optimized_$shape" $config \
      --launch=graph --rotate="$ROTATE" --dump="$WORK/$name.optimized_graph"
    # shellcheck disable=SC2086
    run "$WORK/$name.od.log" "$BUILD/msa_optimized_$shape" $config \
      --launch=direct --rotate="$ROTATE" --dump="$WORK/$name.optimized_direct"
    # shellcheck disable=SC2086
    run "$WORK/$name.s.log" "$BUILD/msa_sota_$shape" $config \
      --dump="$WORK/$name.sota"
    if ! compare "$WORK/$name.optimized_graph" "$WORK/$name.optimized_direct" \
        "$WORK/$name.sota"; then
      status=1
    fi
  done
done
if [[ $status -eq 0 ]]; then
  echo "compare_sota: all dumps match"
else
  echo "compare_sota: MISMATCH" >&2
fi
exit $status
