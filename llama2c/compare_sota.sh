#!/usr/bin/env bash
# Runs the graph-mode baseline, the direct-mode baseline and the SOTA on
# identical inputs, each writing --dump, and compares the dumps within the
# oracle tolerances (compare_dumps.py). Every binary also validates its own
# output against the CPU oracle and must exit 0.
#
# Inputs: the deterministic inputs plus random inputs with several seeds.
# Shapes: the sheet shape plus small shapes (one with size % 4 != 0, which
# takes the SOTA's generic path). Each shape is "batch size rows".
#
# Environment:
#   SHAPES   default "1024 16384 8192|64 1000 300|33 1001 77"
#   INPUTS   default "deterministic random=1 random=2 random=3"
#   COPIES   default "nocopy" (also try "copy")
#   GPU_LOCK default /tmp/cuda-elastic-gpu.lock; every GPU run holds it.
#   DUMP_DIR default a fresh temporary directory, removed afterwards.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD:-$SCRIPT_DIR/../$("$SCRIPT_DIR/../gpu.sh" name)/llama2c/build}"
BASELINE="${BASELINE:-$BUILD/llama2c_cuda_graph}"
SOTA="${SOTA:-$BUILD/llama2c_sota}"
GPU_LOCK="${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}"
SHAPES="${SHAPES:-1024 16384 8192|64 1000 300|33 1001 77}"
INPUTS="${INPUTS:-deterministic random=1 random=2 random=3}"
COPIES="${COPIES:-nocopy}"
# Few timed launches: this checks outputs, it does not measure.
LAUNCHES="${LAUNCHES:-3}"
WARMUP_MS="${WARMUP_MS:-1}"

make -C "$SCRIPT_DIR" --no-print-directory sota baseline >/dev/null

if [[ -n "${DUMP_DIR:-}" ]]; then
  mkdir -p "$DUMP_DIR"
  out="$DUMP_DIR"
else
  out="$(mktemp -d "${TMPDIR:-/tmp}/llama2c-compare.XXXXXX")"
  trap 'rm -rf "$out"' EXIT
fi

run() {  # run <label> <binary> <mode> <batch> <size> <rows> <copies> <inputs> <dump>
  local label="$1" bin="$2" mode="$3" batch="$4" size="$5" rows="$6" copies="$7" inputs="$8" dump="$9"
  local line
  if ! line="$(flock "$GPU_LOCK" "$bin" "$mode" "$LAUNCHES" "$batch" "$size" "$rows" 1 "$WARMUP_MS" \
               "$copies" coalesced "$inputs" "--dump=$dump" 2>&1)"; then
    printf '  %s: FAILED (exit code or validation)\n%s\n' "$label" "$line"
    return 1
  fi
  printf '  %-15s %s\n' "$label" "$(printf '%s\n' "$line" | tr ' ' '\n' |
    grep -E '^(mode|matmul|gpu_ms_per_iter|valid)=' | paste -sd' ')"
}

failures=0
IFS='|' read -r -a shape_list <<< "$SHAPES"
for shape in "${shape_list[@]}"; do
  read -r batch size rows <<< "$shape"
  for copies in $COPIES; do
    for inputs in $INPUTS; do
      printf '== batch=%s size=%s rows=%s %s %s\n' "$batch" "$size" "$rows" "$copies" "$inputs"
      ok=1
      run baseline-graph "$BASELINE" graph "$batch" "$size" "$rows" "$copies" "$inputs" "$out/baseline-graph.bin" || ok=0
      run baseline-direct "$BASELINE" direct "$batch" "$size" "$rows" "$copies" "$inputs" "$out/baseline-direct.bin" || ok=0
      run sota "$SOTA" graph "$batch" "$size" "$rows" "$copies" "$inputs" "$out/sota.bin" || ok=0
      if ((ok)); then
        python3 "$SCRIPT_DIR/compare_dumps.py" "$out/baseline-graph.bin" "$out/baseline-direct.bin" "$out/sota.bin" || ok=0
      fi
      ((ok)) || failures=$((failures + 1))
      rm -f "$out"/*.bin
    done
  done
done
if ((failures)); then
  printf 'compare_sota: %d configuration(s) FAILED\n' "$failures"
  exit 1
fi
printf 'compare_sota: all configurations match\n'
