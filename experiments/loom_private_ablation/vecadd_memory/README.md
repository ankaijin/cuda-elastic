# Vecadd memory optimization stages

The optimized vecadd-memory variant
([`vecadd_memory_optimized.cu`](../../../applications/loom_private/vecadd_memory/vecadd_memory_optimized.cu))
differs from the baseline
([`vecadd_memory.cu`](../../../applications/loom_private/vecadd_memory/vecadd_memory.cu))
in four ways, plus its per-GPU tuning. The application scope is dominated by
PCIe copies (two 4 MiB uploads and a 4 MiB download), so the report's step
columns use the application scope (`stages.json` `"scope": "application"`).

| Stage | Binary | Adds |
|--:|---|---|
| 0 | suite `vecadd_memory` | Baseline: two uploads, one scalar kernel, one download, in order on one stream (Direct, or one CUDA Graph) |
| 1 | `vecadd_memory_stage1` | **Chunked multi-stream pipeline.** 8 chunks, each on its own stream: upload, kernel, download. Later uploads overlap earlier downloads. |
| 2 | `vecadd_memory_stage2` | **One 2D copy for both inputs.** lhs and rhs share one pitched buffer, so one copy uploads a chunk of both. |
| 3 | `vecadd_memory_stage3` | **16-byte vector loads and stores.** Each thread adds four elements per access. |
| 4 | `vecadd_memory_stage4` | **Streaming cache hints.** Evict-first loads and stores (`ld.global.cs`, `st.global.cs`). |
| 5 | suite `vecadd_memory_optimized` | **Per-GPU tuning.** On the RTX 3090 the tuning entry equals the default, so this step changes nothing. |

Stages 1–4 are built from [`vecadd_memory_stages.cu`](vecadd_memory_stages.cu)
with `-DLP_STAGE=1|2|3|4`. They all use the optimized variant's generic
default chunk count, 8. The default kernel shape (`tuning_for`'s default,
kernel 0) is 1 vector per thread with streaming hints, which stage 4 reaches.

## Order

None of the optimizations strictly requires another. The pipeline comes first
because the scope is dominated by the copies, and overlapping the two copy
directions is the main saving. The 2D copy changes the copies, which are the
dominant cost, and saves one command per chunk, so it is measured in the
pipeline. The kernel changes come last: in the pipeline, the kernel of one
chunk overlaps the copies of the others. Streaming hints are a property of the
vector kernel, so they follow it. Splitting them keeps one optimization per
stage, and the default includes them, so they are not part of the tuning step.

## Details

- **Stage 1** keeps the baseline's three separate buffers and its kernel
  (one thread per element, 256-thread blocks), launched once per chunk over
  that chunk's range. The harness stream forks to the chunk streams through
  an event and joins on one event per chunk, so Direct and the Graph hold the
  same pipeline. Chunks are multiples of four elements, as in the optimized
  variant; at the large size every chunk is 131,072 elements either way. Per
  invocation: 16 upload copies of 512 KiB, 8 kernels of 512 blocks, and 8
  downloads of 512 KiB. The compute scope is the baseline's launch, unchanged.
  `run.py` checks that the kernel's machine code equals the baseline's.
- **Stage 2** allocates lhs and rhs together on each side, rhs one pitch after
  lhs (the length rounded up to 64 elements). One `cudaMemcpy2DAsync` of height
  2 uploads a chunk of both: 8 upload copies of 1 MiB. The kernel is still the
  baseline's (machine code checked again).
- **Stage 3** uses the optimized variant's kernel template with one vector per
  thread and plain loads and stores (`vecadd_kernel<1, false>`, which the
  production binary does not contain). 128 blocks per chunk. The final
  (length mod 4) elements take a scalar loop in the first thread.
- **Stage 4** adds `__ldcs`/`__stcs`: `vecadd_kernel<1, true>`, the production
  default. `run.py` checks that its machine code equals the production
  binary's. The profile action checks that its launch structure (kernels with
  grid and block, copies with kind and bytes) equals that of production built
  with its test hook (`-DLP_VECADD_TEST_TUNING`, binary
  `vecadd_memory_reference_tuning`, never timed) run with `LP_VECADD_CHUNKS=8
  LP_VECADD_KERNEL=0`. On the RTX 3090 it also equals the plain production
  binary's.
- **Stage 5** is the production binary. Its tuning entries:

  | GPU | Chunks | Kernel | Versus the default |
  |---|--:|---|---|
  | RTX 3090 (sm_86) | 8 | 1 vector per thread, streaming hints | equal (no change) |
  | RTX 4090 (sm_89) | 8 | 4 vectors per thread, plain | kernel shape |
  | RTX 5090 (sm_120) | 4 | 2 vectors per thread, streaming hints | chunks and kernel shape |

All stages compute the baseline's int32 adds, so every output equals the
oracle and the baseline bitwise (`arithmetic_changes` is empty).

## Tests (`make test`)

- The application's host test (`tests/test_host.cpp`, the oracle) once.
- The baseline's
  [`run_gpu_tests.sh`](../../../applications/loom_private/vecadd_memory/tests/run_gpu_tests.sh)
  on every stage and its drop-last mutant (`-DLP_VECADD_MUTANT_DROP_LAST`),
  which must fail the check.
- [`tests/run_stage_tests.sh`](tests/run_stage_tests.sh), on every stage: the
  optimized variant's own tests with the stage's binaries. Its test-only builds
  (`-DLP_VECADD_TEST_TUNING`) read `LP_VECADD_CHUNKS`; each stage has one kernel
  shape, so there is no `LP_VECADD_KERNEL`.
  - the timed binary's default (8 chunks) and that it ignores the variable;
  - chunk counts 1, 3, 8, and 64 at lengths 1, 5, 257, 5,000, 1,000,003, and
    1,000,004, in Direct and Graph;
  - pitch and vector-tail lengths 2, 3, 4, 7, 63, 64, 65, and 4,097 at 3 chunks;
  - out-of-range chunk counts exit 2;
  - the drop-last mutant at 1, 8, and 64 chunks, and that it drops only the
    last element;
  - mutants that skip the last chunk's upload or download (1, 4, and 64 chunks,
    and the large size) fail the check;
  - a mutant without the join: Direct fails the check, and the Graph capture
    fails with unjoined work (exit 70);
  - `compute-sanitizer --tool memcheck` at 1, 3, and 64 chunks and at length 5;
  - a short bench in both launch modes, whose record names the stage.
