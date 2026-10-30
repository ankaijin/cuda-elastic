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
