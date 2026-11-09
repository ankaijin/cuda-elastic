# PageRank optimization stages

The optimized PageRank variant
([`pagerank_optimized.cu`](../../../applications/loom_private/pagerank/pagerank_optimized.cu))
differs from the baseline
([`pagerank.cu`](../../../applications/loom_private/pagerank/pagerank.cu)) in five
ways. Each depends on the one before it, which fixes the order:

| Stage | Binary | Adds |
|--:|---|---|
| 0 | suite `pagerank` | Baseline: 321 launches per invocation (Direct, or one CUDA Graph) |
| 1 | `pagerank_stage1` | **Persistent cooperative kernel.** One launch runs all sweeps. A grid barrier after every dependency level replaces each level's launch (320 barriers). |
| 2 | `pagerank_stage2` | **Per-node readiness tags.** Within a sweep a node waits only for the sources it reads. Only the 19 barriers between sweeps remain. |
| 3 | `pagerank_stage3` | **Level-ordered renumbering.** Graph arrays are read sequentially and contributions are stored coalesced. |
| 4 | `pagerank_stage4` | **Overlapped map upload.** The position-to-node map, which only the last sweep reads, uploads on a side stream during the sweeps. |
| 5 | suite `pagerank_optimized` | **Per-GPU tuning.** On the RTX 3090: 768 × 2 blocks per SM and evict-first streaming loads of the graph arrays. |

Stages 1–4 are built from [`pagerank_stages.cu`](pagerank_stages.cu) with
`-DLP_STAGE=1|2|3|4`. They all use the optimized variant's generic "default"
shape: 512 threads, at most 2 resident blocks per SM, and graph loads through
the read-only cache. Stage 5 changes only that shape.

The step columns use the application scope, because stage 4 saves time only
there: it moves an upload, and the compute scope has no transfers.

Details:

- **Stage 1** keeps the baseline's arrays, node numbering, and level schedule.
  Contributions are read through L2 (`__ldcg`), because other blocks write
  them during the launch. The level bounds travel as a kernel argument, as in
  the baseline's launches.
- **Stage 2** uses the optimized variant's readiness protocol: sweep tags in
  the sign bit, relaxed loads and stores, and three source loads issued before
  any wait. It still uses the original numbering, so it forms each source
  reference `2v + (v < u)` from the node ids at run time.
- **Stage 3** is the production kernel as it was before the map overlap,
  with the four uploads in series. It takes over the production score write:
  once, in the last sweep. Stages 1 and 2 write scores like the baseline,
  every sweep.
- **Stage 4** uploads offsets, sources and out-degrees, then forks a side
  stream that uploads the map and a 4-byte ready flag while the sweeps run.
  Before the last sweep each block waits for the flag. The side stream joins
  before the score download, and the same bytes are uploaded. `run.py` checks
  that its machine code equals the production binary's
  `gauss_seidel<false>`, and that its launches, copies and memsets equal
  production's run with the default tuning entry (`host_equal`; the reference
  is `pagerank_optimized_tunings` with `LP_PAGERANK_TUNING=default`, built by
  `make build` and never timed). On the RTX 5090 the tuning entry equals the
  default, so stage 4 is also compared with the production binary itself.

All stages compute the contract's ordered Gauss–Seidel updates. One thread
sums each node's incoming list in order, in float. So every stage's scores are
bitwise identical to the oracle and to the baseline (digest
`97eb70ad937006ab` at the large size).

## Tests (`make test`)

- Production's host tests, [`test_host.cpp`](../../../applications/loom_private/pagerank/tests/test_host.cpp)
  (graph generator and level schedule) and [`test_layout.cpp`](../../../applications/loom_private/pagerank/tests/test_layout.cpp)
  (the level-ordered layout stage 3 uses), built as production's Makefile
  builds them.
- The baseline's [`run_gpu_tests.sh`](../../../applications/loom_private/pagerank/tests/run_gpu_tests.sh)
  on every stage and its Jacobi mutant (`-DLP_PAGERANK_MUTANT_JACOBI`), which
  must fail the check.
- [`tests/run_stage_tests.sh`](tests/run_stage_tests.sh), on every stage:
  - the default shape;
  - a test-only build (`-DLP_ABLATION_TEST_SHAPE`) that forces one block of 64
    threads, at 4, 16, 32, 4,096 and 65,536 nodes, which includes waits on
    another lane of the same warp;
  - 1 to 5 sweeps on the large graph, which covers the readiness tag's cycle;
  - a check that the timed binary ignores the shape variable;
  - the readiness-protocol mutants of production's
    [`run_optimized_tests.sh`](../../../applications/loom_private/pagerank/tests/run_optimized_tests.sh),
    on the same sizes (4,096 x 6 and 65,536 x 20): a tag that alternates every
    sweep (`-DLP_PAGERANK_MUTANT_TAG`, stages 2-4), which must fail the
    check with mismatches, and a skipped barrier reset
    (`-DLP_PAGERANK_MUTANT_NO_RESET`, every stage), which must fail the check
    or trap, never hang;
  - stage 4 only: production's delayed map upload (`-DLP_PAGERANK_TEST_DELAY_NODES`,
    200 ms), which must pass at 65,536 x 20, 65,536 x 1 and the large size,
    and its mutant without the last sweep's wait
    (`-DLP_PAGERANK_MUTANT_NO_NODES_WAIT`), which must fail the check.
