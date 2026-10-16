#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# VARIANT=sota profiles pagerank_sota.cu instead of the baseline; its
# binary goes to <gpu>/pagerank/build/ like the Makefile's, and its output
# names get a sota- prefix.
VARIANT="${VARIANT:-baseline}"
# Binaries and profiles go to <repo>/<gpu>/pagerank/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-$SCRIPT_DIR/../$("$SCRIPT_DIR/../gpu.sh" name)/pagerank}"
if [[ "$VARIANT" == sota ]]; then
  SRC="$SCRIPT_DIR/pagerank_sota.cu"
  BIN="${BIN:-$GPU_DIR/build/pagerank_sota}"
  PREFIX=sota-
else
  SRC="$SCRIPT_DIR/gapbs_pagerank_cuda_graph.cu"
  BIN="${BIN:-$GPU_DIR/gapbs_pagerank_cuda_graph}"
  PREFIX=
fi
OUT_DIR="${OUT_DIR:-$GPU_DIR/profiles}"
# PAGERANK_LAUNCH=direct profiles the direct-launch mode; output names get
# a -graph or -direct suffix. Set TIMESTAMP to give both the same suffix.
# The default is each binary's own: graph for the baseline, direct for the
# SOTA.
if [[ "$VARIANT" == sota ]]; then LAUNCH="${PAGERANK_LAUNCH:-direct}"; else LAUNCH="${PAGERANK_LAUNCH:-graph}"; fi
STAMP="$PREFIX$LAUNCH-${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
NVCC="${NVCC:-nvcc}"
read -r -a NVCC_FLAGS <<< "${NVCC_FLAGS:--arch=$("$SCRIPT_DIR/../gpu.sh" arch)}"
# graph: one range per graph launch.  node: one range per kernel/memcpy node.
NSYS_GRAPH_TRACE="${NSYS_GRAPH_TRACE:-graph}"
# Baseline: one initialize_scores kernel, one first-iteration pagerank_update
# per upload chunk, and nineteen full pagerank_update kernels (graph nodes in
# graph mode).  SOTA: plan_kernel, init_kernel, five kernels per upload chunk
# (csr_update, count, prefix, scatter, fine) and twenty binned_update_kernel
# (iterations 1-18, then iteration 19 as two waves); with the defaults (8
# chunks, the last one split three more times) that is 2 + 5 * 11 + 20 = 77,
# which the binary also prints as "kernel launches per full run".
if [[ "$VARIANT" == sota ]]; then
  NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-$((2 + 5 * (${PAGERANK_UPLOAD_CHUNKS:-8} + ${PAGERANK_TAIL_SPLITS:-3}) + 20))}"
else
  NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-$((1 + ${PAGERANK_UPLOAD_CHUNKS:-8} + 19))}"
fi
# Memory metrics that --set basic omits: DRAM traffic, L1/L2 hit rates, and
# global-load sectors per request (coalescing).
NCU_METRICS="${NCU_METRICS:-dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,l1tex__t_sector_hit_rate.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread}"

usage() { printf 'Usage: %s [build|check|run|sweep|nsys|ncu|all]\n' "$0"; }
require_tool() { command -v "$1" >/dev/null || { printf 'error: %s is not installed or not in PATH\n' "$1" >&2; exit 127; }; }
build_id() { printf '%s %s\n' "$NVCC" "${NVCC_FLAGS[*]}"; }
build() {
  require_tool "$NVCC"
  mkdir -p "$(dirname "$BIN")"
  "$NVCC" "${NVCC_FLAGS[@]}" -O3 -o "$BIN" "$SRC"
  build_id > "$BIN.build"
}
# Rebuild when the binary is missing, older than the source, or was built
# with a different compiler or flags.
ensure_bin() {
  if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" || "$SCRIPT_DIR/pagerank_common.h" -nt "$BIN" ||
        "$(cat "$BIN.build" 2>/dev/null)" != "$(build_id)" ]]; then
    build
  fi
}
check() { ensure_bin; PAGERANK_MODE=validate "$BIN"; }
run() { ensure_bin; "$BIN"; }
# Validates, then times, each block size in BLOCK_SIZES.
sweep() {
  ensure_bin
  local block_size out
  for block_size in ${BLOCK_SIZES:-32 64 128 256 512 1024}; do
    out="$(PAGERANK_BLOCK_SIZE="$block_size" PAGERANK_MODE=validate "$BIN")" || { printf '%s\n' "$out"; exit 1; }
    PAGERANK_BLOCK_SIZE="$block_size" PAGERANK_MODE=bench GRAPH_REPLAYS="${GRAPH_REPLAYS:-100}" "$BIN" | tail -n 1
  done
}
# Profiles skip validation so that the profiled launches are the benchmark's.
profile_nsys() {
  require_tool nsys; ensure_bin; mkdir -p "$OUT_DIR"
  PAGERANK_MODE=bench PAGERANK_LAUNCH="$LAUNCH" nsys profile --trace=cuda,nvtx --cuda-graph-trace="$NSYS_GRAPH_TRACE" --sample=none \
    --cpuctxsw=none --stats=true --force-overwrite=true \
    --output="$OUT_DIR/nsys-$STAMP" "$BIN"
}
profile_ncu() {
  require_tool ncu; ensure_bin; mkdir -p "$OUT_DIR"
  PAGERANK_MODE=bench PAGERANK_LAUNCH="$LAUNCH" GRAPH_REPLAYS="${GRAPH_REPLAYS:-1}" \
    ncu --set basic --metrics "$NCU_METRICS" \
    --launch-skip 0 --launch-count "$NCU_LAUNCH_COUNT" --replay-mode kernel \
    --graph-profiling node --target-processes application-only \
    --export "$OUT_DIR/ncu-pagerank-$STAMP" --force-overwrite "$BIN"
}

case "${1:-all}" in
  build) build ;;
  check) check ;;
  run) run ;;
  sweep) sweep ;;
  nsys) profile_nsys ;;
  ncu) profile_ncu ;;
  all) run; profile_nsys; profile_ncu ;;
  *) usage >&2; exit 2 ;;
esac
