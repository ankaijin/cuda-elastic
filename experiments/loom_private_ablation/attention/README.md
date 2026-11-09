# Multisensor attention optimization stages

The optimized attention variant
([`attention_optimized.cu`](../../../applications/loom_private/attention/attention_optimized.cu))
differs from the baseline
([`attention.cu`](../../../applications/loom_private/attention/attention.cu)) in six
ways. Their dependencies fix the order:

| Stage | Binary | Adds |
|--:|---|---|
| 0 | suite `attention` | Baseline: one block per query row, two passes over the context (Direct, or one CUDA Graph) |
| 1 | `attention_stage1` | **64-row blocks with shared K/V tiles.** Each 1,024-key tile leaves L2 once per block and serves 64 rows. One context split. |
| 2 | `attention_stage2` | **Context splits.** 12 splits per row group, merged in split order: 4,692 blocks instead of 391. |
| 3 | `attention_stage3` | **One-pass bound shift.** Each row shifts by an integer upper bound of its scores. The exact-maximum pass runs only above 20. |
| 4 | `attention_stage4` | **Three-FMA score.** Exact weights and the shift as the bias give the same bits with fewer instructions. |
| 5 | `attention_stage5` | **Cheaper math.** `ex2.approx` with $\log_2 e$-scaled weights, fused value sums, and the fourth lane as $\sum pa - \sum pb$. |
| 6 | `attention_stage6` | **Overlapped chunked transfers.** One IMU chunk per split. Each split computes as soon as its chunk arrives, on two side streams. |
| 7 | suite `attention_optimized` | **Per-GPU tuning.** RTX 3090: 8 splits. RTX 4090 and 5090: 12, the default, so this stage changes nothing there. |

The report's step columns use the application scope (`stages.json`
`"scope"`), because stage 6 changes only the transfers.

Stages 1–6 are built from [`attention_stages.cu`](attention_stages.cu) with
`-DLP_STAGE=1..6`. They use the optimized variant's kernel shape on every GPU
(8 warps × 8 rows, 1,024-key tiles) and, from stage 2, its generic default of
12 splits. Stage 7 changes only the split count.

Order:

- Every later stage lives inside stage 1's tiled kernel.
- Splits come second for two reasons. Stage 1 already needs the split and
  merge path, because a lane's float chain may hold at most $2^{16}$ keys,
  which forces splits at the largest contexts. And with one split the large
  preset has only 391 blocks: 2.4, 1.5 and 1.15 waves on the RTX 3090, 4090
  and 5090. Splitting first means the later arithmetic stages are measured at
  the production geometry.
- The bound shift needs per-split key maxima. Its fallback is the tiled
  exact pass.
- The three-FMA score folds in the shift. Cheaper math folds $\log_2 e$ into
  that three-term chain.
- The overlapped transfers send one chunk per split, and fold each chunk's
  key bounds as it lands.

Details:

- **Stage 1** uses the optimized variant's `stream_keys`, its split rule
  (asked for one split), and its merge and statistics kernels. The residual
  moves into the merge, with the baseline's expression. Its attention kernel
  makes the baseline's two passes with the baseline's `half_score`, `expf`,
  and five unfused accumulators per row. Lane $i$ sums keys
  $i, i + 32, \ldots$ in order, then an xor butterfly combines the lanes.
- **Stage 2** asks for 12 splits. Each split shifts by its own exact maximum.
- **Stage 3** adds the production `project_imu_kernel`, which folds the
  per-split $\max |k_l|$ with integer `atomicMax`, and the memset before it.
  It also adds the production bound
  $\lceil \sum_l |h_l| \max |k_l| \rceil$. The score stays `half_score`.
- **Stage 4** evaluates $s - \text{shift}$ as
  `fmaf(h.x, a, fmaf(h.y, b, fmaf(h.z, c, -shift)))` with
  $h = (x + y/2, -x/2, z/2)$, in both passes. Every product and partial sum
  is exact for the contract's inputs, so its digest equals stage 3's.
- **Stage 5** is the production attention kernel, copied unchanged.
  `run.py` checks that its machine code equals the production
  `attention_partial_kernel<8, 8, 1024>`. The application path is still the
  baseline's: whole uploads, the compute path on one stream, then the
  downloads.
- **Stage 6** copies the production `enqueue_application`. It is the
  optimized variant with its generic 12 splits:
  - Every kernel's machine code equals production's (`sass_equal`).
  - Its launch structure (`host_equal`) and digest equal the production
    hooks build run with `LP_ATTENTION_SPLITS=12`
    (`attention_hooks_reference`). That build comes from the unchanged
    `attention_optimized.cu` with `-DLP_ATTENTION_TUNING_HOOKS` and is never
    timed.
  - On the RTX 4090 and 5090 it also equals the plain production binary.

Registers per thread of the attention kernel (`cuobjdump -res-usage`, RTX
3090, nvcc 13.3): 128 in stages 1–3, 126 in stage 4, 99 in stages 5–6. None
uses local memory (no spills). At 128 or fewer, every stage keeps two
256-thread blocks per SM.

## Numerics

No stage loosens a tolerance. Every stage passes the contract's check, with
the production oracle and tolerance. `stages.json` `arithmetic_changes`
declares where the output digest may change:

- **Stage 1:** the summation order. 32 per-lane chains and a butterfly
  replace 256 strided partials and a tree.
- **Stage 2:** the split boundaries and the rescaled merge.
- **Stage 3:** the bound shift.
- **Stage 5:** the rounded $\log_2 e$ weights, `ex2.approx`, and the fused
  products.
- **Stage 7, RTX 3090 only:** 8 splits instead of 12.

Stages 4 and 6 must reproduce the previous stage's digest exactly. So must
stage 7 on the RTX 4090 and 5090.

## Tests (`make test`)

- The baseline's [`run_tests.sh`](../../../applications/loom_private/attention/run_tests.sh)
  (nine oracle cases and nine usage errors) on every stage.
- [`tests/run_stage_tests.sh`](tests/run_stage_tests.sh), with test-only
  `_hooks` builds (`-DLP_ABLATION_TEST_HOOKS`). They read
  `LP_ATTENTION_SPLITS`, `LP_ATTENTION_EXACT_MAXIMUM` (stages 3–6), and
  `LP_ATTENTION_CHECK_COMPUTE` (stage 6). On every stage it runs:
  - the timed binary's generic configuration, and a check that it ignores
    every hook variable;
  - 1, 3 and 52 splits (partial tiles, partial row groups, a long merge);
  - the largest query extent, where the partial-buffer cap allows 8 splits;
  - the largest context ($2^{24}$ keys, variant 2), where the key bound
    raises the splits to 8. Stage 1 gets there from its default of one;
  - invalid hook values, which must exit 2;
  - the skip-key mutant (`-DLP_ATTENTION_MUTANT_SKIP_TILE_LAST_KEY`), which
    must fail the check with exit 1.
- On stages 3–6 only: the exact-maximum fallback, with 5 splits, with one
  short split, and at 37 rows.
- On stage 6 only: 1 and 2 splits, and the compute path inside the checked
  invocation.
- Digest equality at 10 small configurations, including forced splits and
  the forced fallback:
  - stage 4 equals stage 3;
  - stage 6 equals stage 5;
  - stage 6 equals the production hooks build at the same settings;
  - stage 6's compute path equals its overlapped application path.
