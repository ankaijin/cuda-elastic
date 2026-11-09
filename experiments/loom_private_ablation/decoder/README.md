# llama2c decoder optimization stages

The optimized decoder
([`decoder_optimized.cu`](../../../applications/loom_private/decoder/decoder_optimized.cu))
replaces only the baseline's classifier matmul
([`decoder.cu`](../../../applications/loom_private/decoder/decoder.cu)), which
takes 98.7% of the baseline's GPU time. The replacement combines five
optimizations. Each depends on the one before it, which fixes the order:

| Stage | Binary | Adds |
|--:|---|---|
| 0 | suite `decoder` | Baseline: 64 × 64 classifier tiles, 4 × 4 outputs per thread (Direct, or one CUDA Graph) |
| 1 | `decoder_stage1` | **Register-tiled SGEMM.** 128 × 256 block tiles, 16 × 8 outputs per thread from float4 shared loads. Synchronous copies into one buffer. |
| 2 | `decoder_stage2` | **cp.async multi-stage pipeline.** Copies run two 16-column stages ahead of the arithmetic. |
| 3 | `decoder_stage3` | **Register fragment double-buffering.** Column k + 1 loads from shared memory while column k is multiplied. The production kernel, data parallel. |
| 4 | `decoder_stage4` | **Stream-K tail.** A persistent grid; the ragged last wave's k-stages are split evenly, and each split sum is handed on in order. |
| 5 | suite `decoder_optimized` | **Per-GPU tuning.** The tile. Only the RTX 5090 differs from the default: 8-column stages, four deep. |

The step columns use the compute scope (`stages.json` `scope`): the token
reset and all steps, with the model resident.

Stages 1–4 are built from [`decoder_stages.cu`](decoder_stages.cu) with
`-DLP_STAGE=1|2|3|4`. It is a copy of `decoder_optimized.cu`; the stage
differences are marked `LP_STAGE`. rmsnorm, softmax+argmax, the workload
class, the oracle call, and the stream-K branches of the classifier kernel
are production code in every stage. All stages use production's generic
default tile, `128x256x16` (128 sequences × 256 classifier rows × 16 columns
per stage, three stages deep), which production applies on every GPU except
sm_120. Stage 5 changes only the tile, so it is a no-op on the RTX 3090 and
4090.

Details:

- **Stage 1** is production's tile and thread mapping without the pipeline.
  Each k-stage is loaded with plain loads (the same lanes, line clamping,
  and zero fill past `width` as production's copies) and stored into one
  shared buffer; then a barrier, 16 columns, and a barrier. Grid = tiles
  (256 at the large shape), one tile per block, sequence tile fastest.
- **Stage 2** uses production's `cp.async` ring, prologue, wait counts, and
  copy placement (sequence lines before the columns, classifier lines before
  the last two). It reads one fragment per column and multiplies it; the
  barrier for the next stage follows the last column. `cp.async` and the
  multi-stage ring are one step: a one-slot `cp.async` overlaps nothing, and a
  multi-stage ring without it needs register staging, which production does
  not use.
- **Stage 3** is production's `accumulate_tile` unchanged. Its fragments are
  double-buffered in registers, and the barrier moves before the last column
  so that the next stage's first fragment loads during it. `run.py` checks
  that its classifier machine code equals production's
  `classifier_kernel<TileConfig<2, 4, 16, 16, 3, 1>>`, and that its launches
  equal production run with `DECODER_OPT_SCHEDULE=dp
  DECODER_OPT_TILE=128x256x16`.
- **Stage 4** adds production's stream-K plan unchanged: a grid of resident
  blocks (occupancy × SMs), whole waves data parallel, the last one to two
  waves' k-stages split evenly, and a per-step memset of the ready flags.
  RTX 3090: 2 waves plus 92 split tiles on 82 blocks. RTX 4090: 256 tiles
  are exactly 2 waves of 128 blocks, so nothing is split (the grid becomes
  128 persistent blocks of 2 tiles, with no memset). RTX 5090: all 256 tiles
  are split over 170 blocks. Its machine code equals production's, and its
  launches equal production run with `DECODER_OPT_TILE=128x256x16`, and the
  plain production binary on the RTX 3090 and 4090.
- **Stage 5** is the production binary. The RTX 5090 runs
  `TileConfig<2, 4, 16, 8, 4, 1>` (50,176 bytes of shared memory instead of
  75,264); the grid, block, and memsets are stage 4's.

Every logit is accumulated by one thread, from `0.0f`, in increasing column
order, with a separate multiply and add, also across a stream-K hand-off.
This is the baseline's and llama2.c's order, so every stage's outputs are
bitwise the baseline's (large digest `4fcc3e102976547b`; `arithmetic_changes`
is empty). As the production README notes, the synthetic model gives each
logit one nonzero product, so the digest tests cover the column coverage of
every split, not the summation order; the order rests on the code.

Registers (`-Xptxas -v`, sm_86): stage 1 uses 212 and stage 2 uses 214
registers, with no spills. Stages 3 and 4 are the production kernel: 255
registers and 16 bytes of spill stores and loads, as in production. Every
stage runs one block per SM.

The untimed `decoder_optimized_hooks` is the unchanged production source
built with its test hooks (`-DLP_DECODER_TEST_TUNING`). It is the
`host_equal` reference, run with the `env` in `stages.json`.

## Tests (`make test`)

- The baseline's [`run_tests.sh`](../../../applications/loom_private/decoder/tests/run_tests.sh)
  on stages 1–3: five shapes, invalid dimensions, and injected faults.
- The optimized variant's
  [`run_optimized_tests.sh`](../../../applications/loom_private/decoder/tests/run_optimized_tests.sh)
  on stage 4 with its test-only builds: the baseline's cases, every tile with
  both schedules and forced grids, digests against the baseline binary,
  invalid hooks, and both kernel mutants.
- [`tests/run_stage_tests.sh`](tests/run_stage_tests.sh), on every stage, with
  test-only builds (`-DLP_DECODER_TEST_TUNING`) that honour production's
  `DECODER_OPT_TILE`, `DECODER_OPT_SCHEDULE`, and `DECODER_OPT_CTAS`:
  - the timed binary applies the stage's default plan and ignores the hooks;
  - digest equality with the baseline binary on the five tail shapes and on
    widths 8, 9, 16, 17, 24, 40, 48, and 49 (1 to 4 k-stages, partial last
    stages), for every tile, the stage's schedule, and stream-K with forced
    grids of 1, 5, 7, and 11 blocks (stages 1–3 then run their copy loops on
    partial k-ranges);
  - forced grids really split tiles, and invalid hooks exit 2;
  - the production mutants built from the stage (hand-off dropped, last
    k-stage dropped) fail the check.
