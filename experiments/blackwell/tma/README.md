# Experiment B: TMA tile loads and warp specialization

**Question.** Can changing tile loading reduce instruction overhead without
changing attention arithmetic?

**Workload.** `loom-private/multisensor-attention`, optimized variant, at the
large size: 25,000 queries over 250,000 context tokens, 12 context splits on
the RTX 5090. The kernel shape is the production one: 64 query rows per block
(8 per warp), 1,024-key tiles, and two shared buffers. Only the tile loading
changes. Every score, exponential, and sum is the production code, and so are
the split-length bound and the exact-maximum fallback.

| Case | Source | Tile loading |
|---|---|---|
| `control` | `applications/loom_private/attention/attention_optimized.cu`, unchanged | All 256 threads issue 16-byte `cp.async` copies (4 per thread per tile), then `cp.async.wait_group` and a block barrier before each tile, and a block barrier after it. |
| `tma` | [`attention_tma.cu`](attention_tma.cu), `LP_ATTENTION_TMA_MODE=1` | Thread 0 issues one `cp.async.bulk` per tile into the same `float4` tile, with `mbarrier.arrive.expect_tx`. Every thread waits on the tile's mbarrier. A block barrier after each tile guards buffer reuse. |
| `tma_producer` | the same, `LP_ATTENTION_TMA_MODE=2` | A ninth warp only issues the bulk copies. The eight compute warps wait on a full mbarrier per buffer and release it through an empty mbarrier (one arrival per warp). No block barrier remains inside the stream. |

[`attention_tma.diff`](attention_tma.diff) is the exact difference from
production. All cases build with `-arch=sm_120` and the production flags.

Synchronization details:

- A block's tiles are numbered across both attention passes, so tile $s$ uses
  buffer $s \bmod 2$ and waits for phase parity $\lfloor s/2 \rfloor \bmod 2$.
  The exact-maximum pass and the exponential pass share one barrier sequence.
- The producer refills buffer $b$ for fill $f$ only after the empty barrier
  completes fill $f - 1$. That needs one arrival from each of the eight compute
  warps.
- Warps without query rows (the last row group) still wait for every tile and
  still release it, so the arrival counts hold. The producer warp owns no rows
  and does not vote in the exact-maximum decision.
- The issuing thread executes `fence.proxy.async.shared::cta` before each bulk
  copy. This orders the compute warps' earlier reads, which the block barrier
  or empty mbarrier ordered before it, ahead of the async-proxy write.
- With 288 threads, the producer case takes 115 registers and fits only one
  block per SM. `__launch_bounds__(288, 2)` keeps the production's two blocks
  per SM at 96 registers without spills. The `tma` case keeps the production
  bounds and uses 100 registers; the control uses 99.

## Tests (`make test`)

[`tests/run_tests.sh`](tests/run_tests.sh), under the GPU lock:

1. The production test scripts, unchanged, on every case.
   `run_tests.sh` covers small extents against the oracle and invalid
   dimensions. `run_tests_optimized.sh` runs on each case's hooks and
   skip-key mutant builds: every compiled shape with 1, 3, and 52 splits,
   partial tiles and row groups, the exact-maximum fallback in both launch
   paths, the compute path, the largest query extent, the largest context at
   input variant 2 (the regression that bounds a split to $2^{21}$ keys),
   invalid hooks, and the mutant.
2. Output digests of both TMA cases must equal the control's, bit for bit. The
   configurations cover partial tiles and query groups, one key, 3 and 52
   splits, every shape, both attention passes, the compute path, and the
   largest context.
3. A mutant whose compute warps skip the wait for a tile's copy must be
   rejected. Mode 1 fails the check. In mode 2 the block exits while its copies
   are in flight, so the launch fails.
4. `compute-sanitizer` memcheck, racecheck, and synccheck on both TMA cases.

The runner's large check then requires identical digests across all cases,
and each bench's digest must equal its check's, after more than 40 Graph
replays.
