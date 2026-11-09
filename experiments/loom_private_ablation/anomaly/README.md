# Anomaly detection optimization stages

The optimized anomaly variant
([`anomaly_optimized.cu`](../../../applications/loom_private/anomaly/anomaly_optimized.cu))
differs from the baseline
([`anomaly.cu`](../../../applications/loom_private/anomaly/anomaly.cu)) in five
ways. The step columns use the application scope (with transfers), because
the pipeline and the per-GPU chunk count change only the transfers.

| Stage | Binary | Adds |
|--:|---|---|
| 0 | suite `anomaly` | Baseline: upload, one block per sample, download (Direct, or one CUDA Graph) |
| 1 | `anomaly_stage1` | **Tiles of 16 samples (dp4a).** All 32 lanes of a warp requantize, so each FP64 instruction yields 32 results instead of 1. |
| 2 | `anomaly_stage2` | **Proven binary32 requantization.** Binary64 runs only where the binary32 code is not proven equal. |
| 3 | `anomaly_stage3` | **Int8 tensor cores.** `mma.sync.m16n8k32.s8` replaces dp4a. This is the production kernel. |
| 4 | `anomaly_stage4` | **Chunked multi-stream pipeline.** Uploads, kernels, and downloads of 32 chunks overlap. |
| 5 | suite `anomaly_optimized` | **Per-GPU tuning.** 32 chunks on the RTX 3090 and 4090 (no change), 16 on the RTX 5090. |

Stages 1–4 are built from [`anomaly_stages.cu`](anomaly_stages.cu) with
`-DLP_STAGE=1|2|3|4`. It is a copy of the production variant. The tile kernel,
`store_unit`, `InputChunk`, the host setup, the oracle and the check are
unchanged. [`requant.cuh`](../../../applications/loom_private/anomaly/requant.cuh)
is included unchanged.

## Why this order

The baseline is bound by the FP64 pipe, not by memory. In the RTX 3090
profile, the FP64 pipe is 85% busy, and the LSU and L1 are 8% busy. Lane 0
alone requantizes each output row (`anomaly.cu` lines 135–136). So each FP64
warp instruction yields one result.

- **Stage 1** fixes that with the tile. Its kernel is the same as the
  production kernel, except the dot products. The tensor-core version needs
  the tile and its accumulator layout, so the tile comes first.
- **Stage 2** comes before the tensor cores. Neither needs the other. With
  binary64 requantization, a dp4a tile and an mma tile are both FP64 bound,
  so the tensor-core step would show almost nothing. After stage 2, stage 3
  measures the GEMM itself. This differs from the order in which the variant
  was developed (tensor cores first).
- **Stage 4** changes only host code. **Stage 5** is the production binary.

## Details

- **Stage 1** keeps the production tile: 256 threads per 16 samples (6,554
  blocks at the large size), padded weights, two activation buffers and two
  input staging buffers in shared memory, and the layer-1 input stream. The
  zero point is applied after the dot product,
  $\sum_i (a_i - z)\,w_i = \sum_i a_i w_i - z \sum_i w_i$, with the row
  sums packed at setup. Lane (group, quad) of a unit's warp computes rows
  `group` and `group + 8` and columns `2 quad` and `2 quad + 1` of each
  n-tile with `__dp4a`. These are the eight accumulators of the mma C layout,
  so the epilogue is shared code. The dp4a kernel thus inherits a geometry
  tuned for the tensor cores. Its launch bounds `(256, 4)` keep the production
  residency: 64 registers, 4 blocks per SM, no spills. Without them, the
  unrolled K step takes 114 registers (2 blocks per SM). Every code uses the
  baseline's binary64 `requantize()` and input quantization.
- **Stage 2** switches `quantize_code` and `requantize_code` to the binary32
  fast paths. Nothing else changes.
- **Stage 3** compiles the production `mma_s8`, `load_b`, `unit_k_step`,
  `tile_layer`, and `tiled_inference_kernel` unchanged. `run.py` checks that
  the machine code of `tiled_inference_kernel<16>` equals the optimized
  binary's.
- **Stages 1–3** use the baseline's host sequence: one upload, one launch, one
  download, all on the harness stream.
- **Stage 4** adds the production pipeline with its generic default of 32
  chunks on every GPU. The production code selects 16 chunks for cc 12 and
  later and 32 otherwise. Its kernel's machine code also equals production's.
  Its launch structure (kernels with grid and block, copies with kind and
  bytes) must equal two references:
  - `anomaly_optimized_chunks32`, which is production built from its
    unchanged source with its test hook `-DLP_ANOMALY_FORCE_CHUNKS=32`. This
    binary is never timed. The comparison runs on every GPU.
  - The plain optimized binary, on the RTX 3090 and 4090, where production
    also uses 32 chunks.

All stages produce the same integer accumulators, in a different order of
summation. Integer addition is exact, because
$|\sum_i a_i w_i| \le 640 \cdot 128 \cdot 128 < 2^{31}$. Every binary32
code is proven equal to the binary64 one. So every stage's output is bitwise
identical to the baseline's and the oracle's, and `arithmetic_changes` is
empty.

## Tests (`make test`)

- The production host test (`tests/test_host.cpp`), once. The stages share
  the host code unchanged.
- The production
  [`run_tests.sh`](../../../applications/loom_private/anomaly/tests/run_tests.sh)
  on every stage, with that stage's skip-last-sample mutant
  (`-DLP_ANOMALY_MUTANT_SKIP_LAST_SAMPLE`). The mutant must fail the check.
- The production
  [`run_optimized_tests.sh`](../../../applications/loom_private/anomaly/tests/run_optimized_tests.sh)
  on stage 4, with its test directory built from the stage source:
  - the exhaustive fast-path test and its no-margin mutant (production
    `tests/test_requant.cu`);
  - tile edges;
  - the binary64 fallback for every code;
  - forced chunk counts 1, 3, 16, and 32 at 17 and 4,099 samples.
- [`tests/run_stage_tests.sh`](tests/run_stage_tests.sh), on every stage:
  - the timed binary logs its stage and configuration; stage 4 uses 32
    chunks on every GPU;
  - 1, 15, 16, 17, 33, and 4,099 samples, in Direct and Graph modes;
  - a test-only mutant without the zero-point correction
    (`-DLP_ABLATION_MUTANT_NO_CORRECTION`), which must fail the check;
  - the binary64 fallback build on stages 2 and 3;
  - 1,048,576 samples, the largest accepted, on stages 1 and 4;
  - a check that the stage source reads no environment variable.
