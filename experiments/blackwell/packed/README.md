# Experiment A: packed byte arithmetic

**Question.** Does native packed byte addition reduce instructions or kernel
time while preserving exact results?

**Workload.** `loom-private/vecadd-product`, optimized variant, at the large
size: 1,048,576 records, 256-thread blocks, and 16 transfer chunks. Nothing
else changes between the cases.

| Case | Source | `add_bytes` |
|---|---|---|
| `control` | `applications/loom_private/vecadd_product/vecadd_product_optimized.cu`, unchanged | $((a \mathbin{\&} \text{0x7f7f7f7f}) + (b \mathbin{\&} \text{0x7f7f7f7f})) \oplus ((a \oplus b) \mathbin{\&} \text{0x80808080})$ |
| `add_u8x4` | [`vecadd_product_packed.cu`](vecadd_product_packed.cu), [`add_u8x4.diff`](add_u8x4.diff) | one PTX `add.u8x4` |

Both cases are built with `-gencode arch=compute_120f,code=sm_120f` and the
production suite's other flags. `add.u8x4` needs a family or
architecture-specific target: ptxas rejects it for plain `sm_120`, and
`-arch=sm_120f` fails too because it also embeds generic `compute_120` PTX.

## Machine code

The SASS inspection runs before timing. The compiler does not recognize the
control's expression as a packed add. Per word, it emits four `LOP3.LUT` and
one `IADD`. The candidate emits one `VIADD.U8x4` instead. Everything else in
`add_records` (index arithmetic, two loads, one store) is identical. So the
experiment continues past the plan's stopping point for equivalent code.

## Tests (`make test`)

[`tests/run_tests.sh`](tests/run_tests.sh), under the GPU lock:

- **Exhaustive lanes.** [`tests/test_lanes.cu`](tests/test_lanes.cu) includes
  each case's source, so it tests that case's own `add_bytes`. Every lane sees
  all 65,536 byte pairs. Every combination of the edge bytes `00 01 7f 80 81 fe
  ff` across the four lanes of both operands appears (5,764,801 words, covering
  every carry and overflow pattern across neighbouring lanes). Another 2^20
  random words follow. Every byte must equal $(a + b) \bmod 256$.
- **Production tail cases** for both cases in Direct and Graph mode: 1, 2, 3,
  7, 63, 64, 255, 256, 257, 1,000, 4,097, 65,537, and 1,000,003 records. Each
  check includes the 256-byte canaries on both sides of the output.
- **Mutant.** The candidate built with the production saturating-add mutant
  must fail both the lane test and the check.
- `compute-sanitizer --tool memcheck` on the candidate at 1,000 and 65,537
  records.

The runner's check then runs the large Direct/Graph check of both cases and
requires every output digest to equal the control's.

## Results on kilia's RTX 5090

Source `b2f0601` (benches) and `353359f` (profile; it changed only the
runner), CUDA 13.4, driver 615.71.10, GPU
`GPU-947b8c66-8235-01b9-4934-9a9da6d6c54b`. Runs live under
`rtx5090/blackwell_experiments/packed/`.

**Correctness.** Both cases pass every test and the large Direct/Graph check.
Their output digests are identical, and every bench's digest equals its
check's.

**Feature verification.** `add_records` has 31 non-`NOP` SASS instructions in
the control and 27 in the candidate. One `VIADD.U8x4` replaces four `LOP3.LUT`
and one `IADD`. Nsight Compute counts 15,728,640 warp instructions in the
control's 16 chunk launches and 13,631,488 in the candidate's. The difference
is 4 per warp over 524,288 warps, or 13.3%.

**Performance.** Median batch time per iteration (µs). The ratio is the median
of candidate/control within each pair, with the range of pair ratios.

| Run | Scope | Launch | Control | `add_u8x4` | Ratio (range) | Pairs faster |
|---|---|---|--:|--:|---|--:|
| 5 pairs (`20261030-182653-kilia-455ace66`) | kernel | Direct | 117.01 | 117.28 | 1.0027 (1.0011–1.0063) | 0/5 |
| | kernel | Graph | 117.44 | 117.57 | 1.0000 (0.9986–1.0028) | 3/5 |
| | application | Direct | 4723.2 | 4637.8 | 0.9943 | 4/5 |
| | application | Graph | 4617.1 | 4732.8 | 1.0045 | 2/5 |
| 10 pairs (`20261030-182825-kilia-e20c6a4c`) | kernel | Direct | 117.26 | 117.24 | 0.9993 (0.9959–1.0071) | 6/10 |
| | kernel | Graph | 117.12 | 117.30 | 1.0009 (0.9978–1.0046) | 3/10 |
| | application | Direct | 4644.8 | 4665.5 | 1.0065 (0.9825–1.0431) | 3/10 |
| | application | Graph | 4731.4 | 4637.5 | 1.0020 (0.9676–1.0238) | 4/10 |

The five-pair Direct kernel time looked 0.27% slower, but ten more pairs did
not reproduce it. Kernel time stays at 117 µs because the kernel is limited by
memory traffic. In the profiled application scope, each of the 16 chunk
launches takes about 7.3 µs with DRAM about 65% active. The application scope
is dominated by the 192 MiB of PCIe transfers (about 4.6 ms).

**Conclusion: no measurable benefit.** `add.u8x4` executes and removes 13% of
the kernel's instructions with exact results. Neither kernel nor application
time changes beyond run-to-run variation. Promotion would also need the
production 5090 build to target `sm_120f`.
