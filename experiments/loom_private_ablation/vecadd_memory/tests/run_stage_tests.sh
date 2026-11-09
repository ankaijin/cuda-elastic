#!/usr/bin/env bash
# GPU tests of one vecadd_memory stage beyond the baseline's run_gpu_tests.sh,
# run by `make test` under the GPU lock (this script does not lock).
#   run_stage_tests.sh <build dir> <stage> <scratch dir>
# <build dir> holds vecadd_memory_stage<N>, its test-only tuning build (in which
# LP_VECADD_CHUNKS sets the chunk count), and its drop_last, skip_upload,
# skip_download, and no_join mutants. These are the optimized variant's
# variant-specific tests (applications/loom_private/vecadd_memory/tests/
# run_gpu_tests.sh, its `optimized` block) with the stage's binaries and
# variant name, without the kernel-shape loop (each stage has one shape), plus
# pitch and vector-tail lengths. Small --set dimensions only; none is an
# upstream smoke or validation configuration (4096/2/0, 65536/3/1, 131072/5/7).
set -uo pipefail

build=$1
s=$2
scratch=$3/stage$s
bin=$build/vecadd_memory_stage$s
tuning=$build/vecadd_memory_stage${s}_tuning
drop_last_mutant=$build/vecadd_memory_stage${s}_drop_last_mutant
skip_upload_mutant=$build/vecadd_memory_stage${s}_skip_upload_mutant
skip_download_mutant=$build/vecadd_memory_stage${s}_skip_download_mutant
no_join_mutant=$build/vecadd_memory_stage${s}_no_join_mutant
mkdir -p "$scratch"
failures=0
limit=120

# expect <name> <exit code> <log pattern or ""> <command...>
expect() {
  local name=$1 want=$2 pattern=$3
  shift 3
  local log="$scratch/$name.log"
  timeout $limit "$@" >"$log" 2>&1
  local got=$?
  if [[ $got -ne $want ]]; then
    echo "FAIL stage$s $name: exit $got, expected $want"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  elif [[ -n $pattern ]] && ! grep -q -- "$pattern" "$log"; then
    echo "FAIL stage$s $name: log lacks '$pattern'"
    sed 's/^/    /' "$log"
    failures=$((failures + 1))
  else
    echo "PASS stage$s $name (exit $got)"
  fi
}

# The timed binary uses the default 8 chunks and names its stage; it ignores
# the test builds' chunk variable.
expect default-large 0 "stage $s: 8 chunks of 131072 elements" "$bin"
LP_VECADD_CHUNKS=0 expect ignores-chunks-env 0 "stage $s: 8 chunks of 128 elements" \
  "$bin" --set length=1000
LP_VECADD_CHUNKS=3 expect tuning-names-chunks 0 "stage $s: 3 chunks" \
  "$tuning" --set length=5000

# Chunking: small lengths cross chunk boundaries, including more chunks than
# vectors, chunk starts that are not multiples of 256 elements, and a last chunk
# with a tail. Every combination runs Direct and Graph.
for chunks in 1 3 8 64; do
  for length in 1 5 257 5000 1000003 1000004; do
    LP_VECADD_CHUNKS=$chunks expect "chunks-$chunks-length-$length" 0 "" \
      "$tuning" --set length=$length
  done
done
# Pitch padding (rhs starts at the length rounded up to 64 elements) and
# vector tails of 1 to 3 elements, at 3 chunks.
for length in 2 3 4 7 63 64 65 4097; do
  LP_VECADD_CHUNKS=3 expect "chunks-3-length-$length" 0 "" "$tuning" --set length=$length
done
expect large-mult-2047 0 "" "$tuning" --set multiplier=2047

# Out-of-range overrides are rejected, not ignored.
LP_VECADD_CHUNKS=0 expect tuning-chunks-0 2 "" "$tuning" --set length=16
LP_VECADD_CHUNKS=65 expect tuning-chunks-65 2 "" "$tuning" --set length=16
LP_VECADD_CHUNKS=abc expect tuning-chunks-text 2 "" "$tuning" --set length=16

# The drop-last mutant fails at every chunk count, at a length whose last
# element is in the main path and at one whose last element is in a tail.
for chunks in 1 8 64; do
  for length in 1000000 1000003; do
    LP_VECADD_CHUNKS=$chunks expect "drop-last-mutant-chunks-$chunks-$length" 1 "" \
      "$drop_last_mutant" --set length=$length
  done
done
# Only the last element is dropped, however many chunks there are.
LP_VECADD_CHUNKS=64 expect drop-last-mutant-drops-only-the-last-element 1 \
  "1 of 1000003 results differ" "$drop_last_mutant" --set length=1000003
if ! grep -q 'mismatch at index 1000002: expected 3000006' \
     "$scratch/drop-last-mutant-drops-only-the-last-element.log"; then
  echo "FAIL stage$s drop-last-mutant-index"; failures=$((failures + 1))
fi

# A last chunk whose upload or download is skipped must fail the oracle, with
# one chunk and with many.
for chunks in 1 4 64; do
  LP_VECADD_CHUNKS=$chunks expect "skip-upload-mutant-chunks-$chunks" 1 "" \
    "$skip_upload_mutant" --set length=100003
  LP_VECADD_CHUNKS=$chunks expect "skip-download-mutant-chunks-$chunks" 1 "" \
    "$skip_download_mutant" --set length=100003
done
expect skip-upload-mutant-large 1 "" "$skip_upload_mutant"
expect skip-download-mutant-large 1 "" "$skip_download_mutant"

# Fork and join ordering. Without the join, the harness stream finishes before
# the chunks: Direct reads unfinished results and fails the check, and the
# Graph capture ends with unjoined work (CUDA error, exit 70).
expect no-join-mutant-direct 1 "" "$no_join_mutant" --launch direct
expect no-join-mutant-graph 70 "" "$no_join_mutant" --launch graph

# Out-of-bounds accesses, under compute-sanitizer.
sanitizer=$(command -v compute-sanitizer || true)
if [[ -n $sanitizer ]]; then
  for chunks in 1 3 64; do
    LP_VECADD_CHUNKS=$chunks expect "memcheck-chunks-$chunks" 0 "" "$sanitizer" \
      --error-exitcode 9 --tool memcheck "$tuning" --set length=100003 --launch direct
  done
  expect memcheck-length-5 0 "" "$sanitizer" --error-exitcode 9 --tool memcheck "$bin" \
    --set length=5 --launch direct
else
  echo "SKIP stage$s memcheck (compute-sanitizer not found)"
fi

# Bench path works at the large size, and its record names the stage.
expect bench-direct 0 "" "$bin" --action bench --launch direct --iterations 2 --warmup 1
expect bench-graph 0 "\"variant\": \"stage$s-" "$bin" --action bench --launch graph \
  --iterations 2 --warmup 1

if [[ $failures -ne 0 ]]; then
  echo "stage $s tests FAILED ($failures)"
  exit 1
fi
echo "stage $s tests passed"
