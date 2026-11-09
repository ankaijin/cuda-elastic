# Vecadd product optimization stages

The optimized vecadd product variant
([`vecadd_product_optimized.cu`](../../../applications/loom_private/vecadd_product/vecadd_product_optimized.cu))
differs from the baseline
([`vecadd_product.cu`](../../../applications/loom_private/vecadd_product/vecadd_product.cu))
in one way, so there is one stage after the baseline:

| Stage | Binary | Adds |
|--:|---|---|
| 0 | suite `vecadd_product` | Baseline: one 128 MiB upload, one kernel, one 64 MiB download, in order on one stream (Direct, or one CUDA Graph) |
| 1 | suite `vecadd_product_optimized` | **Chunked multi-stream pipeline.** 16 chunks, each on its own stream, so later uploads overlap earlier downloads. |

The step columns use the application scope (`"scope": "application"`): the
change moves transfers, and the compute scope (one launch of the same kernel
over all records) is unchanged.

Why one stage:

- **Pinned memory** is not a step. The baseline already allocates its host
  buffers with `cudaMallocHost`.
- **The kernel** is the baseline's with a `first` offset, so each chunk keeps
  absolute word indices. That only serves the chunking.
- **The fork and join events** are needed for correctness and for the Graph
  capture, not for speed.
- **Chunking on one stream** would overlap nothing, so it is not a step of its
  own. The kernel is about 3% of the application scope, so almost all of the
  gain is the two copy directions running at once.
- **Per-GPU tuning** is a no-op. The chunk count is 16 on the RTX 3090, 4090
  and 5090 (`kChunks`; see the variant's README), so the stage 1 binary already
  runs every GPU's tuning.

The bytes moved are the baseline's: 128 bytes up and 64 bytes down per
record. The arithmetic is exact integer arithmetic, so both stages produce the
same output digest.

## Tests (`make test`)

There are no stage sources, so `make test` runs the suite's own tests on both
binaries: `make test VARIANT=baseline` and `make test VARIANT=optimized` in
[`applications/loom_private/vecadd_product`](../../../applications/loom_private/vecadd_product).
They cover the oracle at many record counts, mutants, memcheck, assets, invalid
input, the bench path, and warmup exclusion, plus, for the optimized variant,
chunk counts 1 to 64, skipped chunk copies, and a missing join.
