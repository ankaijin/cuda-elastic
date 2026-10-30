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

## Results on kilia's RTX 5090

Runs at `840b7b5` (the commit that added this experiment), `ede7016`, and
`dc6a739`; the later two add only recorded evidence. CUDA 13.4, driver 615.71.10, GPU `GPU-947b8c66-8235-01b9-4934-9a9da6d6c54b`.
Runs live under `rtx5090/blackwell_experiments/tma/`.

**Correctness.** All three cases pass every test, the large Direct/Graph
check, and the sanitizers. Their output digests are identical, and every
bench's digest equals its check's.

**Feature verification.** In SASS, the production kernel shape
(`attention_partial_kernel<8, 8, 1024>`) shows:

| | `LDGSTS` (cp.async) | `UBLKCP` (bulk copy) | `SYNCS` (mbarrier) | `BAR.SYNC` | `MUFU.EX2` | Registers |
|---|--:|--:|--:|--:|--:|--:|
| `control` | 20 | 0 | 0 | 4 | 264 | 99 |
| `tma` | 0 | 4 | 12 | 2 | 264 | 100 |
| `tma_producer` | 0 | 2 | 16 | 0 | 264 | 96 |

Nsight Compute, summed over the 12 attention launches of one Graph invocation
(profiled launches run one at a time):

| | Warp instructions | Kernel time (µs) | Issue active | Stall: wait | Stall: short scoreboard |
|---|--:|--:|--:|--:|--:|
| `control` | 1,684,334,227 | 2,214.6 | 70.2% | 0.47 | 0.54 |
| `tma` | 1,657,016,752 (−1.6%) | 2,156.1 (−2.6%) | 70.6% | 0.23 | 0.52 |
| `tma_producer` | 1,679,060,395 (−0.3%) | 2,608.4 (+17.8%) | 61.2% | 1.14 | 2.03 |

Stalls are median warps stalled per issue-active cycle. The remaining limits
match the production analysis: the MUFU pipe (`ex2`) and the FMA pipe are
about 65% busy, and L2 throughput is about 11%.

**Performance.** Median batch time per iteration (µs). The ratio is the median
of candidate/control within each pair, with the range of pair ratios and the
number of pairs in which the candidate was faster.

| Run | Scope | Launch | Control (range) | `tma` | Ratio | `tma_producer` | Ratio |
|---|---|---|--:|--:|---|--:|---|
| 5 pairs `a2a2dcdd` | compute | Direct | 1584.6 (1479.1–1626.7) | 1428.2 | 0.9361 (0.8746–0.9658), 5/5 | 1625.8 | 1.0619 (0.9707–1.1741), 2/5 |
| | application | Direct | 1375.0 (1368.1–1391.4) | 1344.7 | 0.9773 (0.9696–0.9866), 5/5 | 1515.8 | 1.1040 (1.0913–1.1117), 0/5 |
| | compute | Graph | 1478.0 (1471.6–1631.3) | 1422.6 | 0.9592 (0.8708–0.9661), 5/5 | 1590.7 | 1.0726 (0.9742–1.1762), 2/5 |
| | application | Graph | 1368.7 (1359.3–1374.7) | 1340.7 | 0.9795 (0.9785–0.9826), 5/5 | 1505.5 | 1.1055 (1.0881–1.1076), 0/5 |
| 10 pairs `920ee789` | compute | Direct | 1500.4 (1457.9–1583.0) | 1438.5 | 0.9592 (0.8903–1.0464), 8/10 | 1605.5 | 1.0738 (1.0335–1.1745), 0/10 |
| | application | Direct | 1374.4 (1370.6–1433.4) | 1346.7 | 0.9775 (0.9492–0.9872), 10/10 | 1507.8 | 1.0966 (1.0630–1.1130), 0/10 |
| | compute | Graph | 1500.5 (1456.5–1582.5) | 1434.7 | 0.9583 (0.9092–1.0766), 9/10 | 1601.8 | 1.0812 (1.0004–1.1495), 0/10 |
| | application | Graph | 1365.0 (1361.0–1379.2) | 1338.5 | 0.9802 (0.9687–0.9861), 10/10 | 1499.6 | 1.0994 (1.0832–1.1119), 0/10 |

Run ids are `20261030-183830-kilia-a2a2dcdd` and
`20261030-184041-kilia-920ee789`. The application scope is faster than the
compute scope because it runs the context splits on two streams, so
neighbouring splits fill each other's tails. The compute scope runs one launch.

Applying the decision rule from [`../README.md`](../README.md):

- `tma`, Graph application scope: 2.0% faster in 10 of 10 confirmation pairs.
  This exceeds the control's half range (0.67%), so the improvement is
  **confirmed**.
- `tma`, Direct application scope: 2.25% faster in 10 of 10 pairs. One slow
  control run widened the control's half range to 2.28%, so this is not
  confirmed by the rule.
- `tma`, compute scope: about 4% faster, in 8 and 9 of 10 pairs. The control's
  compute-scope times vary by about ±4%, so this is not confirmed either.
- `tma_producer`: 7–10% slower in every scope, with no confirmation pair
  faster at application scope: a **regression**.

**Conclusions.**

- `tma`: **improved**, by about 2% at Graph application scope, with identical
  outputs. Bulk copies remove the 16 `cp.async` instructions per thread per
  pass and the barrier before each tile. Nsight Compute shows 1.6% fewer
  instructions and half the fixed-latency waits. The kernel remains bound by
  instruction issue and the exponential unit, which limits the gain.
- `tma_producer`: **regressed**. The producer warp saves almost no compute-warp
  instructions, because one thread already issued each copy in `tma`. Its
  per-warp full and empty barriers add stalls: short-scoreboard and wait
  stalls rise 4x and 2.5x, and issue activity falls from 70% to 61%. Holding
  two blocks per SM also limits the compute warps to 96 registers.
