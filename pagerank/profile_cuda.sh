#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/gapbs_pagerank_cuda_graph}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/profiles}"
STAMP="$(date +%Y%m%d-%H%M%S)"
NVCC="${NVCC:-nvcc}"
read -r -a NVCC_FLAGS <<< "${NVCC_FLAGS:-}"

usage() { printf 'Usage: %s [build|run|nsys|ncu|all]\n' "$0"; }
require_tool() { command -v "$1" >/dev/null || { printf 'error: %s is not installed or not in PATH\n' "$1" >&2; exit 127; }; }
build() { require_tool "$NVCC"; "$NVCC" "${NVCC_FLAGS[@]}" -O3 -o "$BIN" "$SCRIPT_DIR/gapbs_pagerank_cuda_graph.cu"; }
ensure_bin() { [[ -x "$BIN" ]] || build; }
run() { ensure_bin; "$BIN"; }
profile_nsys() {
  require_tool nsys; ensure_bin; mkdir -p "$OUT_DIR"
  nsys profile --trace=cuda --cuda-graph-trace=graph --sample=none \
    --cpuctxsw=none --stats=true --force-overwrite=true \
    --output="$OUT_DIR/nsys-$STAMP" "$BIN"
}
profile_ncu() {
  require_tool ncu; ensure_bin; mkdir -p "$OUT_DIR"
  ncu --set basic --launch-skip 0 --launch-count 9 --replay-mode kernel \
    --graph-profiling node --target-processes application-only \
    --export "$OUT_DIR/ncu-pagerank-$STAMP" --force-overwrite "$BIN"
}

case "${1:-all}" in
  build) build ;;
  run) run ;;
  nsys) profile_nsys ;;
  ncu) profile_ncu ;;
  all) run; profile_nsys; profile_ncu ;;
  *) usage >&2; exit 2 ;;
esac
