// Fastest PageRank we know how to build for the RTX 3090 (sm_86) on the GAPBS
// workload of gapbs_pagerank_cuda_graph.cu: the same input (an incoming-edge
// CSR uploaded from pinned host memory on every run), the same twenty
// Jacobi iterations with damping 0.85 and no dangling redistribution, the
// same scores written back to the host, and the same validation.
//
// Why the baseline is slow: each iteration gathers 16M random 4-byte
// contributions from a 4 MB array.  The array stays in L2, but every gather
// costs its own 32-byte L2 sector, and the update runs at the L2 sector rate
// (~18M sectors, ~0.23-0.26 ms per iteration).  Restructuring the per-row
// kernel (warp-per-row, int4 column loads, cache hints, a persisting-L2
// window) does not change the sector count, so it does not help.
//
// What this does instead: inside every run, while the 64 MB column array is
// still uploading, the GPU converts each upload chunk into a source-sorted,
// destination-partitioned layout:
//   * destinations are split into P "parts" of A rows (41 parts of 25,576
//     rows on the benchmark: about the most fp32 accumulators one SM's
//     shared memory holds; the few rows past it use global atomics);
//   * within a part, the edges are counting-sorted by source sector (a
//     256-bin coarse pass, then a per-bin sort by sector), so a warp's 32
//     lanes gather a few 32-byte sectors instead of 32, and each part reads
//     each contribution sector about once: ~5.4M gathered sectors per
//     iteration instead of 16.8M on the benchmark graph;
//   * each edge becomes one 32-bit record (source low bits << DB | row in
//     part); the source's high bits come from the "superbin" the record is
//     grouped in.
// An iteration is one cooperative launch of 82 blocks, two per part on two
// SMs.  The part's blocks claim 512-record chunks from a shared counter (so
// they finish together) and accumulate in shared memory; then each block
// writes the half of its sums the other block finalizes, waits for it at a
// per-part barrier (safe: all blocks are co-resident), and finalizes its half
// of the rows.  Iteration 0 runs on the CSR as each upload chunk lands, so it
// and the conversion hide behind the PCIe upload, which is the floor of the
// run (72 MB at ~24 GB/s); the last upload chunk is split into smaller ones
// so little is left to do after the final copy.  Scores are written only by
// the final iteration, which runs as two waves of 82 blocks over half the
// parts each, so the first half of the scores downloads while the second
// half is computed.  Graphs this layout cannot encode (more than 2^25
// vertices) fall back to a plain CSR pull kernel.  The layout depends only on
// V, E and the device; everything data-dependent is computed on the GPU in
// the timed run, and nothing is cached between runs.
//
// Environment: PAGERANK_MODE (all|validate|bench), PAGERANK_LAUNCH
// (graph|direct), PAGERANK_BLOCK_SIZE (threads of the CSR kernels),
// PAGERANK_PIN_HOST, PAGERANK_UPLOAD_CHUNKS (column upload chunks, default
// 8), PAGERANK_TAIL_SPLITS (times the last chunk is halved, default 3),
// PAGERANK_SPLIT_FINAL (0 disables the two-wave final iteration),
// GRAPH_REPLAYS, and PAGERANK_GRAPH / PAGERANK_START / PAGERANK_ROTATE /
// PAGERANK_DUMP / PAGERANK_DUMP_REF (pagerank_common.h).  The timed replays
// rotate between three distinct inputs by default.  Building with -DTIMING=1
// prints a per-phase breakdown of the last update launch.

#include "pagerank_common.h"

namespace {

// ---------------------------------------------------------------------------
// Layout parameters, chosen on the host from V, E and the device only.

#ifndef ITER_THREADS
#define ITER_THREADS 1024
#endif
constexpr int kIterThreads = ITER_THREADS;  // threads of the binned update
constexpr int kSliceEdges = 4096;      // edges per conversion slice
constexpr int kCoarseBits = 8;         // 256 coarse source bins per part
constexpr int kCoarseBins = 1 << kCoarseBits;
constexpr int kFineBinsMax = 512;      // fine (sector) bins per coarse bin
constexpr int kSliceThreads = 512;     // threads of the count and scatter passes
constexpr int kEdgesPerThread = kSliceEdges / kSliceThreads;
constexpr int kFineThreads = 512;      // threads of the fine sort (>= kFineBinsMax)
constexpr int kRecordsPerThread = 8;   // records a fine-sort thread holds
constexpr int kMaxPlanParts = 1024;    // parts the plan balances items over
constexpr int kWarpChunk = 512;        // records a warp claims at a time in the update
constexpr int kFinalizeBatch = 13;     // rows per thread loaded together when finalizing
constexpr int kMaxSuperbins = 256;     // V <= 2^25 and >= 17 source bits per record
constexpr int kMaxItemsPerPart = 8;
constexpr int kDefaultBlockSize = 256;
constexpr int kDefaultUploadChunks = 8;
constexpr int kDefaultRotate = 3;
constexpr int kDefaultTailSplits = 3;

struct Layout {
  int sms = 0;
  int vertices = 0;
  int edges = 0;
  bool binned = false;
  int cap = 0;        // accumulators in shared memory (rows beyond use global atomics)
  int rows = 0;       // A: rows per part
  int parts = 0;      // P
  int items = 0;      // blocks of the binned update (all co-resident when cooperative)
  bool cooperative = false;
  int split_parts = 0;  // first wave of the split final iteration (0: no split)
  int dst_bits = 0;   // DB: bits of the row-in-part field
  int src_bits = 0;   // SB = 32 - DB: source bits stored in a record
  int superbins = 0;  // NS = ceil(V / 2^SB)
  int coarse_shift = 0;  // CS: coarse bin = source >> CS
  int coarse_bins = 0;   // NCB = ceil(V / 2^CS) <= 256
  int fine_shift = 0;    // fine bin = (source & (2^CS - 1)) >> fine_shift
  int fine_bins = 0;
  int slices_max = 0;    // upper bound on conversion slices
};

int ceil_log2(long long x) {
  int bits = 0;
  while ((1ll << bits) < x) ++bits;
  return bits;
}

Layout make_layout(int vertices, int edges, const cudaDeviceProp& prop) {
  Layout L;
  L.vertices = vertices;
  L.edges = edges;
  const int sms = prop.multiProcessorCount;
  L.sms = sms;
  L.cap = static_cast<int>((prop.sharedMemPerBlockOptin - 64) / sizeof(float));
  const int vbits = std::max(1, ceil_log2(vertices));
  // Allow ~3% of a part's rows to overflow shared memory into global atomics
  // so that the parts split evenly over the SMs (41 parts x 2 on 82 SMs).
  const long long cap_soft = static_cast<long long>(L.cap) * 103 / 100;
  int parts = static_cast<int>((vertices + cap_soft - 1) / cap_soft);
  int per_part = 1;
  if (parts <= sms) {
    per_part = std::min(kMaxItemsPerPart, sms / parts);
    parts = std::max(parts, sms / per_part);
  }
  L.parts = parts;
  L.rows = (vertices + parts - 1) / parts;
  L.parts = (vertices + L.rows - 1) / L.rows;
  L.items = parts <= sms ? std::min(sms, L.parts * per_part) : L.parts;
  L.cooperative = L.parts <= sms;
  L.dst_bits = std::max(1, ceil_log2(L.rows));
  L.src_bits = 32 - L.dst_bits;
  L.coarse_shift = std::max(0, vbits - kCoarseBits);
  L.binned = edges > 0 && L.rows <= (1 << 15) && L.coarse_shift <= L.src_bits &&
             vbits <= 25 && prop.cooperativeLaunch;
  if (L.binned && ((static_cast<long long>(vertices) + (1ll << L.src_bits) - 1) >> L.src_bits) > kMaxSuperbins)
    L.binned = false;
  L.superbins = static_cast<int>((static_cast<long long>(vertices) + (1ll << L.src_bits) - 1) >> L.src_bits);
  L.coarse_bins = static_cast<int>((static_cast<long long>(vertices) + (1ll << L.coarse_shift) - 1) >> L.coarse_shift);
  L.fine_shift = std::max(3, L.coarse_shift - 9);
  L.fine_bins = L.coarse_shift > L.fine_shift ? 1 << (L.coarse_shift - L.fine_shift) : 1;
  L.slices_max = (edges + kSliceEdges - 1) / kSliceEdges + L.parts;
  L.split_parts = L.binned && L.cooperative && L.parts >= 2 ? L.parts / 2 : 0;
  return L;
}

// Which blocks of a binned update work on which part.  The whole schedule
// covers every part with `items` blocks in one launch; the split schedule
// runs the update as two launches (waves) of `items` blocks, the first over
// parts [0, split_parts) and the second over the rest, so the first half of
// the final scores can be downloaded while the second half is computed.
struct Schedule {
  int* item_part;   // [waves * items]: part of each block
  int* part_items;  // [parts]: blocks per part
  int* part_slot;   // [parts]: first block of the part within its wave
  int* arrived;     // [parts]: barrier counter, + blocks of the part per launch
  int* cursor;      // [2 * parts]: chunks claimed, alternating slots per launch
};

// Kernel parameters shared by every kernel of a run (passed by value).
struct Params {
  // Inputs.
  const int* row_offsets;
  const int* columns;
  const int* out_degrees;
  // Layout.
  int vertices, edges, rows, parts, items, cap;
  int dst_bits, src_bits, superbins, coarse_shift, coarse_bins, fine_shift, fine_bins;
  float base, damping;
  // Conversion state (all rebuilt every run).
  int* part_edge;     // [parts + 1]: first edge of each part
  int* part_slice;    // [parts + 1]: first conversion slice of each part
  int* slice_part;    // [slices]: part of each slice
  int* slice_rows;    // [2 * slices]: rows each slice spans
  int* bucket_start;  // [parts * (coarse_bins + 1)]: first record of each coarse bin
  int* chunk_parts;   // [chunks + 1]: parts complete after each chunk
  const int* chunk_end;  // [chunks]: edge count uploaded after each chunk (host-known)
  int chunks;
  int* coarse_count;  // [parts * coarse_bins]
  int* coarse_fill;   // [parts * coarse_bins]
  uint32_t* staged;   // [edges]: records grouped by coarse bin
  uint32_t* records;  // [edges]: records sorted by fine bin
  int* superbin_start;  // [parts * (superbins + 1)]
  // Update state.
  float* partial;     // [items * cap]: partial sums swapped between a part's blocks
  Schedule whole, split;
  int split_parts;    // parts in the split schedule's first wave
  float* overflow;    // [vertices]: rows past `cap`, accumulated with atomics
};

__device__ __forceinline__ float contribution(float score, int out_degree) {
  return out_degree > 0 ? score / out_degree : 0.0f;
}

// Exclusive scan of n <= 1024 ints in shared memory, in place; returns the total.
__device__ int block_exclusive_scan(int* data, int n, int* scratch /* >= 32 ints */) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int value = threadIdx.x < n ? data[threadIdx.x] : 0;
  int inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) scratch[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    int w = lane < warps ? scratch[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, w, offset);
      if (lane >= offset) w += other;
    }
    scratch[lane] = w;  // inclusive warp totals
  }
  __syncthreads();
  const int total = scratch[((blockDim.x + 31) >> 5) - 1];
  const int exclusive = inclusive - value + (warp > 0 ? scratch[warp - 1] : 0);
  __syncthreads();
  if (threadIdx.x < n) data[threadIdx.x] = exclusive;
  __syncthreads();
  return total;
}

// ---------------------------------------------------------------------------
// Largest i in [lo, hi) with a[i] <= x, for nondecreasing a with a[lo] <= x.
// Run by a whole warp: each step probes 32 positions, so a search over 2^20
// entries takes four dependent loads instead of twenty.
__device__ int warp_last_le(const int* a, int lo, int hi, int x) {
  const int lane = threadIdx.x & 31;
  while (hi - lo > 32) {
    const int step = (hi - lo + 31) / 32;
    const int i = lo + lane * step;
    const unsigned ok = __ballot_sync(0xffffffffu, i < hi && __ldg(a + i) <= x);
    lo += (31 - __clz(ok)) * step;
    hi = min(hi, lo + step);
  }
  const int i = lo + lane;
  const unsigned ok = __ballot_sync(0xffffffffu, i < hi && __ldg(a + i) <= x);
  return lo + 31 - __clz(ok);
}

// Items of parts [q0, q1) in schedule `sch`, as blocks [base, base + items)
// of one launch: one block per part, then each spare block goes to the part
// with the most edges per block (an argmax by warp 0).  With more parts
// than kMaxPlanParts (or than blocks), every part has one block.
__device__ void allocate_items(const Params& p, const Schedule& sch, int q0, int q1, int base,
                               const int* edges, int* items_of, int* scan, int* scratch) {
  const int n = q1 - q0;
  for (int q = q0 + threadIdx.x; q < q1; q += blockDim.x) {
    sch.arrived[q] = 0;
    sch.cursor[2 * q] = 0;
    sch.cursor[2 * q + 1] = 0;
  }
  if (p.parts > kMaxPlanParts || n >= p.items) {
    for (int q = q0 + threadIdx.x; q < q1; q += blockDim.x) {
      sch.part_items[q] = 1;
      sch.part_slot[q] = q - q0;
      sch.item_part[base + q - q0] = q;
    }
    __syncthreads();
    return;
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) items_of[i] = 1;
  __syncthreads();
  if (threadIdx.x < 32) {
    for (int extra = p.items - n; extra > 0; --extra) {
      float best = -1.0f;
      int arg = 0;
      for (int i = threadIdx.x; i < n; i += 32) {
        const float per = static_cast<float>(edges[q0 + i + 1] - edges[q0 + i]) / items_of[i];
        if (per > best) { best = per; arg = i; }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, best, offset);
        const int other_arg = __shfl_xor_sync(0xffffffffu, arg, offset);
        if (other > best || (other == best && other_arg < arg)) { best = other; arg = other_arg; }
      }
      if (threadIdx.x == 0) items_of[arg]++;
      __syncwarp();
    }
  }
  __syncthreads();
  // Block prefixes (n <= kMaxPlanParts == blockDim).
  scan[threadIdx.x] = threadIdx.x < n ? items_of[threadIdx.x] : 0;
  __syncthreads();
  block_exclusive_scan(scan, blockDim.x, scratch);
  if (threadIdx.x < n) {
    const int q = q0 + threadIdx.x, g = items_of[threadIdx.x], first = scan[threadIdx.x];
    sch.part_items[q] = g;
    sch.part_slot[q] = first;
    for (int k = 0; k < g; ++k) sch.item_part[base + first + k] = q;
  }
  __syncthreads();
}

// Plan: after the row offsets land.  Part edge ranges, conversion slices,
// the parts each upload chunk completes, the update's items (blocks spread
// over parts in proportion to their edges), and zeroed counters.
__global__ void __launch_bounds__(1024) plan_kernel(Params p) {
  __shared__ int scratch[32];
  __shared__ int scan[1024];
  __shared__ int edges[kMaxPlanParts + 1];
  __shared__ int items_of[kMaxPlanParts];
  const int P = p.parts;
  for (int q = threadIdx.x; q <= P; q += blockDim.x) {
    const int e = p.row_offsets[min(q * p.rows, p.vertices)];
    p.part_edge[q] = e;
    if (q <= kMaxPlanParts) edges[q] = e;
  }
  for (int i = threadIdx.x; i < P * p.coarse_bins; i += blockDim.x) p.coarse_fill[i] = 0;
  for (int i = threadIdx.x; i < P * p.coarse_bins; i += blockDim.x) p.coarse_count[i] = 0;
  __syncthreads();
  // part_slice: prefix of ceil(E_q / slice) over parts, 1024 at a time; and
  // slice -> part.
  int carry = 0;
  for (int q0 = 0; q0 < P; q0 += blockDim.x) {
    const int q = q0 + threadIdx.x;
    const int n = q < P ? (p.part_edge[q + 1] - p.part_edge[q] + kSliceEdges - 1) / kSliceEdges : 0;
    scan[threadIdx.x] = n;
    __syncthreads();
    const int total = block_exclusive_scan(scan, blockDim.x, scratch);
    if (q < P) {
      const int first = carry + scan[threadIdx.x];
      p.part_slice[q] = first;
      for (int k = 0; k < n; ++k) p.slice_part[first + k] = q;
    }
    carry += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) p.part_slice[P] = carry;
  // chunk_parts[c + 1] = parts whose edges all arrived by the end of chunk c.
  for (int c = threadIdx.x; c < p.chunks; c += blockDim.x) {
    const int end = p.chunk_end[c];
    int lo = 0, hi = P;  // largest q in [0, P] with part_edge[q] <= end
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (p.part_edge[mid] <= end) lo = mid; else hi = mid - 1;
    }
    p.chunk_parts[c + 1] = c + 1 == p.chunks ? P : lo;
  }
  if (threadIdx.x == 0) p.chunk_parts[0] = 0;
  allocate_items(p, p.whole, 0, P, 0, edges, items_of, scan, scratch);
  if (p.split_parts > 0) {
    allocate_items(p, p.split, 0, p.split_parts, 0, edges, items_of, scan, scratch);
    allocate_items(p, p.split, p.split_parts, P, p.items, edges, items_of, scan, scratch);
  }
}

// Initial contributions (and the uniform start's scores, for zero
// iterations), and zeroed overflow accumulators.
__global__ void init_kernel(Params p, float* scores, float* contributions, bool uniform) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= p.vertices) return;
  const float score = uniform ? 1.0f / p.vertices : scores[v];
  if (uniform) scores[v] = score;
  contributions[v] = contribution(score, p.out_degrees[v]);
  p.overflow[v] = 0.0f;
}

// The conversion slices of the parts completed by chunk `chunk`.
struct SliceRange { int begin, end; };
__device__ __forceinline__ SliceRange chunk_slices(const Params& p, int chunk) {
  const int q0 = p.chunk_parts[chunk], q1 = p.chunk_parts[chunk + 1];
  return {p.part_slice[q0], p.part_slice[q1]};
}

__device__ __forceinline__ void slice_edges(const Params& p, int slice, int& part, int& e0, int& e1) {
  part = p.slice_part[slice];
  e0 = p.part_edge[part] + (slice - p.part_slice[part]) * kSliceEdges;
  e1 = min(e0 + kSliceEdges, p.part_edge[part + 1]);
}

// Pass 1 (per completed part): coarse-bin histogram, and the rows each
// slice spans (found by warp 0 while the block's loads are in flight).
__global__ void __launch_bounds__(kSliceThreads) count_kernel(Params p, int chunk) {
  __shared__ int hist[kCoarseBins];
  const SliceRange range = chunk_slices(p, chunk);
  for (int slice = range.begin + blockIdx.x; slice < range.end; slice += gridDim.x) {
    int part, e0, e1;
    slice_edges(p, slice, part, e0, e1);
    for (int i = threadIdx.x; i < kCoarseBins; i += blockDim.x) hist[i] = 0;
    int src[kEdgesPerThread];
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k) {
      const int e = e0 + threadIdx.x + k * kSliceThreads;
      src[k] = e < e1 ? __ldg(p.columns + e) : -1;
    }
    if (threadIdx.x < 32) {
      const int row0 = part * p.rows, row1 = min(row0 + p.rows, p.vertices);
      const int ra = warp_last_le(p.row_offsets, row0, row1, e0);
      const int rb = warp_last_le(p.row_offsets, row0, row1 + 1, e1 - 1) + 1;
      if (threadIdx.x == 0) {
        p.slice_rows[2 * slice] = ra;
        p.slice_rows[2 * slice + 1] = rb;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k)
      if (src[k] >= 0) atomicAdd(&hist[src[k] >> p.coarse_shift], 1);
    __syncthreads();
    int* count = p.coarse_count + part * p.coarse_bins;
    for (int i = threadIdx.x; i < p.coarse_bins; i += blockDim.x)
      if (hist[i]) atomicAdd(&count[i], hist[i]);
    __syncthreads();
  }
}

// Bucket (coarse bin) starts and superbin starts of every completed part.
__global__ void prefix_kernel(Params p, int chunk) {
  __shared__ int start[kCoarseBins];
  __shared__ int scratch[32];
  const int q0 = p.chunk_parts[chunk], q1 = p.chunk_parts[chunk + 1];
  const int bins_per_superbin = 1 << min(30, p.src_bits - p.coarse_shift);
  for (int part = q0 + blockIdx.x; part < q1; part += gridDim.x) {
    const int* count = p.coarse_count + part * p.coarse_bins;
    start[threadIdx.x] = threadIdx.x < p.coarse_bins ? count[threadIdx.x] : 0;
    __syncthreads();
    block_exclusive_scan(start, kCoarseBins, scratch);
    const int base = p.part_edge[part];
    int* bucket = p.bucket_start + part * (p.coarse_bins + 1);
    if (threadIdx.x < p.coarse_bins) bucket[threadIdx.x] = base + start[threadIdx.x];
    if (threadIdx.x == 0) bucket[p.coarse_bins] = p.part_edge[part + 1];
    int* sb = p.superbin_start + part * (p.superbins + 1);
    if (threadIdx.x < p.coarse_bins && threadIdx.x % bins_per_superbin == 0)
      sb[threadIdx.x / bins_per_superbin] = base + start[threadIdx.x];
    if (threadIdx.x == 0) sb[p.superbins] = p.part_edge[part + 1];
    __syncthreads();
  }
}

// Pass 2: scatter each completed part's edges into its coarse bins as
// 32-bit records (source low bits << DB | row in part).  A slice's records
// are first grouped by bin in shared memory, so each bin's run is written
// with consecutive threads (coalesced) even when sources are random.
__global__ void __launch_bounds__(kSliceThreads) scatter_kernel(Params p, int chunk) {
  __shared__ int start[kCoarseBins];
  __shared__ int local[kCoarseBins];
  __shared__ int offset[kCoarseBins];
  __shared__ int scratch[32];
  __shared__ uint16_t row_of[kSliceEdges];
  __shared__ uint32_t grouped[kSliceEdges];
  __shared__ uint8_t bin_of[kSliceEdges];
  const SliceRange range = chunk_slices(p, chunk);
  const uint32_t src_mask = (1u << p.src_bits) - 1u;
  for (int slice = range.begin + blockIdx.x; slice < range.end; slice += gridDim.x) {
    int part, e0, e1;
    slice_edges(p, slice, part, e0, e1);
    const int row0 = part * p.rows;
    int src[kEdgesPerThread];
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k) {
      const int e = e0 + threadIdx.x + k * kSliceThreads;
      src[k] = e < e1 ? __ldg(p.columns + e) : -1;
    }
    for (int i = threadIdx.x; i < kCoarseBins; i += blockDim.x) local[i] = 0;
    const int ra = p.slice_rows[2 * slice], rb = p.slice_rows[2 * slice + 1];
    for (int r = ra + threadIdx.x; r < rb; r += blockDim.x) {
      const int lo = max(__ldg(p.row_offsets + r), e0), hi = min(__ldg(p.row_offsets + r + 1), e1);
      for (int e = lo; e < hi; ++e) row_of[e - e0] = static_cast<uint16_t>(r - row0);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k)
      if (src[k] >= 0) atomicAdd(&local[src[k] >> p.coarse_shift], 1);
    __syncthreads();
    // Reserve this slice's range in every bin of the part, and lay the bins
    // out back to back in shared memory.
    const int* bucket = p.bucket_start + part * (p.coarse_bins + 1);
    int* fill = p.coarse_fill + part * p.coarse_bins;
    for (int i = threadIdx.x; i < kCoarseBins; i += blockDim.x) {
      const int n = i < p.coarse_bins ? local[i] : 0;
      start[i] = n ? bucket[i] + atomicAdd(&fill[i], n) : 0;
      offset[i] = n;
      local[i] = 0;
    }
    __syncthreads();
    block_exclusive_scan(offset, kCoarseBins, scratch);
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k) {
      if (src[k] < 0) continue;
      const int bin = src[k] >> p.coarse_shift;
      const int slot = offset[bin] + atomicAdd(&local[bin], 1);
      const int e = threadIdx.x + k * kSliceThreads;
      grouped[slot] = ((static_cast<uint32_t>(src[k]) & src_mask) << p.dst_bits) | row_of[e];
      bin_of[slot] = static_cast<uint8_t>(bin);
    }
    __syncthreads();
    for (int slot = threadIdx.x; slot < e1 - e0; slot += blockDim.x) {
      const int bin = bin_of[slot];
      p.staged[start[bin] + slot - offset[bin]] = grouped[slot];
    }
    __syncthreads();
  }
}

// Pass 3: sort each coarse bin by source sector (fine bin) into `records`.
// A bin larger than one block's share is sorted in pieces (still correct,
// just less coalesced).
__global__ void __launch_bounds__(kFineThreads) fine_kernel(Params p, int chunk) {
  __shared__ int fine[kFineBinsMax];
  __shared__ int scratch[32];
  __shared__ uint32_t sorted[kFineThreads * kRecordsPerThread];
  const int q0 = p.chunk_parts[chunk], q1 = p.chunk_parts[chunk + 1];
  const int buckets = (q1 - q0) * p.coarse_bins;
  const uint32_t coarse_mask = (1u << p.coarse_shift) - 1u;
  const int db = p.dst_bits, fs = p.fine_shift;
  for (int bucket = blockIdx.x; bucket < buckets; bucket += gridDim.x) {
    const int part = q0 + bucket / p.coarse_bins, bin = bucket % p.coarse_bins;
    const int* starts = p.bucket_start + part * (p.coarse_bins + 1);
    const int b0 = starts[bin], b1 = starts[bin + 1];
    for (int c0 = b0; c0 < b1; c0 += kFineThreads * kRecordsPerThread) {
      const int n = min(kFineThreads * kRecordsPerThread, b1 - c0);
      for (int i = threadIdx.x; i < p.fine_bins; i += blockDim.x) fine[i] = 0;
      uint32_t r[kRecordsPerThread];
#pragma unroll
      for (int k = 0; k < kRecordsPerThread; ++k) {
        const int i = threadIdx.x + k * kFineThreads;
        r[k] = i < n ? __ldcs(p.staged + c0 + i) : 0xffffffffu;
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < kRecordsPerThread; ++k)
        if (threadIdx.x + k * kFineThreads < n) atomicAdd(&fine[((r[k] >> db) & coarse_mask) >> fs], 1);
      __syncthreads();
      block_exclusive_scan(fine, p.fine_bins, scratch);  // fine_bins <= blockDim
#pragma unroll
      for (int k = 0; k < kRecordsPerThread; ++k)
        if (threadIdx.x + k * kFineThreads < n)
          sorted[atomicAdd(&fine[((r[k] >> db) & coarse_mask) >> fs], 1)] = r[k];
      __syncthreads();
      // Sorted in shared memory; written out coalesced.
      for (int i = threadIdx.x; i < n; i += blockDim.x) p.records[c0 + i] = sorted[i];
      __syncthreads();
    }
  }
}

// CSR pull update for rows [row_begin, row_end) (or, with chunk >= 0, the
// rows of the parts that chunk completed): four lanes per row, each loading
// up to four columns before gathering.  Writes the scores on the last
// iteration and the next contributions otherwise.
__global__ void csr_update_kernel(Params p, int row_begin_arg, int row_end_arg,
                                  const float* __restrict__ contributions, float* next,
                                  float* scores, int chunk) {
  int row_begin = row_begin_arg, row_end = row_end_arg;
  if (chunk >= 0) {
    row_begin = min(p.chunk_parts[chunk] * p.rows, p.vertices);
    row_end = min(p.chunk_parts[chunk + 1] * p.rows, p.vertices);
  }
  const int lane = threadIdx.x & 3;
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 2;
  const int groups = (gridDim.x * blockDim.x) >> 2;
  for (int row0 = row_begin; row0 < row_end; row0 += groups) {
    const int row = row0 + group;
    float sum = 0.0f;
    if (row < row_end) {
      const int e1 = __ldg(p.row_offsets + row + 1);
      int e = __ldg(p.row_offsets + row) + lane;
      for (; e + 12 < e1; e += 16) {
        const int s0 = __ldg(p.columns + e), s1 = __ldg(p.columns + e + 4);
        const int s2 = __ldg(p.columns + e + 8), s3 = __ldg(p.columns + e + 12);
        sum += (__ldg(contributions + s0) + __ldg(contributions + s1)) +
               (__ldg(contributions + s2) + __ldg(contributions + s3));
      }
      for (; e < e1; e += 4) sum += __ldg(contributions + __ldg(p.columns + e));
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    if (row < row_end && lane == 0) {
      const float score = p.base + p.damping * sum;
      if (scores) scores[row] = score;
      else next[row] = contribution(score, p.out_degrees[row]);
    }
  }
}

// One binned iteration.  Each block works on one part (sch.item_part); the G
// blocks of a part claim its records in chunks, accumulate them in shared
// memory, then swap partial sums through global memory (each finalizes one
// G-th of the part's rows) after a per-part barrier, which is safe because
// the launch is cooperative (all blocks co-resident).  `epoch` counts the
// launches of this schedule before this one in the run, so the barrier
// counters only ever grow within a run.  Building with -DTIMING=1 records
// per-block phase times (printed by main) for tuning.
#ifndef TIMING
#define TIMING 0
#endif
#if TIMING
__device__ unsigned long long g_times[1024][5];
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define STAMP(k) do { if (threadIdx.x == 0) g_times[blockIdx.x][k] = now_ns(); } while (0)
#else
#define STAMP(k) do {} while (0)
#endif
__global__ void __launch_bounds__(kIterThreads, 1)
binned_update_kernel(Params p, Schedule sch, int item_base, const float* contributions, float* next,
                     float* scores, int epoch) {
  extern __shared__ float acc[];
  const int item = blockIdx.x;  // block within this launch; also its partial-sum slot
  const int part = sch.item_part[item_base + item];
  const int g = sch.part_items[part];
  const int first = sch.part_slot[part];  // first block of the part in this launch
  const int rank = item - first;
  const int row0 = part * p.rows;
  const int rows = min(p.rows, p.vertices - row0);
  const int cap = p.cap;
  const int srows = min(rows, cap);
  const int seg0 = rows * rank / g, seg1 = rows * (rank + 1) / g;  // rows this block finalizes
  float* overflow = p.overflow + row0;
  const int bd = blockDim.x;
  STAMP(0);
  // Work counters alternate between two slots per part; the slot the next
  // launch uses is cleared now (the previous launch that used it is done).
  int* cursor = sch.cursor + 2 * part;
  if (rank == 0 && threadIdx.x == 0) cursor[(epoch + 1) & 1] = 0;
  cursor += epoch & 1;
  const int* __restrict__ sb = p.superbin_start + part * (p.superbins + 1);
  // Warm L2 with the out-degrees the finalize step reads.
  for (int i = seg0 + threadIdx.x * 32; i < seg1; i += blockDim.x * 32)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p.out_degrees + row0 + i));
  for (int i = threadIdx.x; i < srows; i += blockDim.x) acc[i] = 0.0f;
  __syncthreads();
  STAMP(1);

  const uint32_t dst_mask = (1u << p.dst_bits) - 1u;
  const int db = p.dst_bits;
  const uint32_t* records = p.records;
  // Adds one record's contribution: shared memory for the first `cap` rows of
  // the part, global atomics for the few rows past it.
  const auto add = [&](uint32_t r, float v) {
    const uint32_t d = r & dst_mask;
    if (d < static_cast<uint32_t>(cap)) atomicAdd(&acc[d], v);
    else atomicAdd(&overflow[d], v);
  };
  // Every warp of the part's blocks claims chunks of kWarpChunk consecutive
  // records until the part is done, so its blocks finish together.  The
  // claim for the next chunk is issued before the current one is processed,
  // which hides its latency.
  const int lane = threadIdx.x & 31;
  const int pe0 = sb[0], pe1 = sb[p.superbins];
  const int chunks = (pe1 - pe0 + kWarpChunk - 1) / kWarpChunk;
  int s = 0;
  int claim = lane == 0 ? atomicAdd(cursor, 1) : 0;
  for (;;) {
    const int chunk = __shfl_sync(0xffffffffu, claim, 0);
    if (chunk >= chunks) break;
    if (lane == 0) claim = atomicAdd(cursor, 1);
    const int c0 = pe0 + chunk * kWarpChunk, c1 = min(c0 + kWarpChunk, pe1);
    while (sb[s + 1] <= c0) ++s;
    for (int t = s; t < p.superbins && sb[t] < c1; ++t) {
      const int a = max(sb[t], c0), b = min(sb[t + 1], c1);
      const uint32_t high = static_cast<uint32_t>(t) << p.src_bits;
      int e = a + lane;
      for (; e + 96 < b; e += 128) {
        const uint32_t r0 = __ldcs(records + e), r1 = __ldcs(records + e + 32);
        const uint32_t r2 = __ldcs(records + e + 64), r3 = __ldcs(records + e + 96);
        const float v0 = contributions[high | (r0 >> db)], v1 = contributions[high | (r1 >> db)];
        const float v2 = contributions[high | (r2 >> db)], v3 = contributions[high | (r3 >> db)];
        add(r0, v0);
        add(r1, v1);
        add(r2, v2);
        add(r3, v3);
      }
      for (; e < b; e += 32) {
        const uint32_t r = __ldcs(records + e);
        add(r, contributions[high | (r >> db)]);
      }
    }
  }
  __syncthreads();
  STAMP(2);

  if (g > 1) {
    float* mine = p.partial + static_cast<size_t>(item) * cap;
    for (int i = threadIdx.x; i < min(seg0, srows); i += bd) __stcg(mine + i, acc[i]);
    for (int i = seg1 + threadIdx.x; i < srows; i += bd) __stcg(mine + i, acc[i]);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&sch.arrived[part], 1);
      const int target = g * (epoch + 1);
      volatile int* arrived = sch.arrived + part;
      while (*arrived < target) __nanosleep(32);
    }
    __syncthreads();
    __threadfence();
  }
  STAMP(3);
  // Finalize rows [seg0, seg1): this block's sums plus the other blocks'
  // partial sums (rows below cap), or the global overflow sums (rows past it).
  const float* __restrict__ partial = p.partial + static_cast<size_t>(first) * cap;
  const int* __restrict__ degrees = p.out_degrees + row0;
  const float base = p.base, damping = p.damping;
  // The other blocks' sums for row i.  Two blocks per part is the common
  // case; straight-line code there lets the loads of a batch overlap.
  const float* __restrict__ pair = partial + static_cast<size_t>(1 - rank) * cap;
  const auto others = [&](int i) {
    if (g == 2) return __ldcg(pair + i);
    float sum = 0.0f;
    for (int j = 0; j < g; ++j)
      if (j != rank) sum += __ldcg(partial + static_cast<size_t>(j) * cap + i);
    return sum;
  };
  const auto emit = [&](int i, float sum, int degree) {
    const float score = base + damping * sum;
    if (scores) scores[row0 + i] = score;
    else next[row0 + i] = contribution(score, degree);
  };
  // All of a thread's rows are loaded in one batch (a block finalizes at
  // most ~13K rows), so the batch costs one round trip to L2.
  const int shared_end = min(seg1, srows);
  for (int i0 = seg0 + threadIdx.x; i0 < shared_end; i0 += kFinalizeBatch * bd) {
    int degree[kFinalizeBatch];
    float other[kFinalizeBatch];
#pragma unroll
    for (int k = 0; k < kFinalizeBatch; ++k) {
      const int i = i0 + k * bd;
      degree[k] = i < shared_end ? __ldg(degrees + i) : 0;
      other[k] = i < shared_end ? others(i) : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < kFinalizeBatch; ++k) {
      const int i = i0 + k * bd;
      if (i < shared_end) emit(i, acc[i] + other[k], degree[k]);
    }
  }
  for (int i = max(seg0, srows) + threadIdx.x; i < seg1; i += bd) {
    const float sum = __ldcg(overflow + i);
    overflow[i] = 0.0f;
    emit(i, sum, __ldg(degrees + i));
  }
  __syncthreads();
  STAMP(4);
}

// ---------------------------------------------------------------------------
// Host side.

struct Settings {
  // PAGERANK_LAUNCH: direct (the default) enqueues every run; graph replays
  // captured graphs.  Direct measured ~0.17 ms faster per full run with the
  // default three rotated inputs (graph-mode uploads ran slower), and the
  // same with one input.
  bool graph = false;
  int block_size = kDefaultBlockSize;  // PAGERANK_BLOCK_SIZE: CSR kernels
  bool pin_host = true;                // PAGERANK_PIN_HOST
  int upload_chunks = kDefaultUploadChunks;  // PAGERANK_UPLOAD_CHUNKS
  int tail_splits = kDefaultTailSplits;      // PAGERANK_TAIL_SPLITS
};

// compute: every kernel and the result download; upload: every H2D copy,
// in order.  (Running the first iteration on a third stream beside the
// conversion measured ~150 us slower: it slowed the upload itself.)
struct Streams {
  cudaStream_t compute = nullptr, upload = nullptr;
};

// Device buffers for one graph shape, reused by every run of that shape.
// Every run rebuilds all of their contents.
struct Device {
  Layout L;
  int* row_offsets = nullptr;
  int* columns = nullptr;
  int* out_degrees = nullptr;
  float* scores = nullptr;
  float* contributions_a = nullptr;
  float* contributions_b = nullptr;
  Params params{};
  std::vector<void*> allocations;
};

template <typename T>
T* device_alloc(Device& d, size_t count) {
  void* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, std::max<size_t>(1, count) * sizeof(T)));
  d.allocations.push_back(pointer);
  return static_cast<T*>(pointer);
}

Device allocate_device(const HostGraph& host, const cudaDeviceProp& prop) {
  Device d;
  const int V = host.vertices;
  const int E = static_cast<int>(host.columns.size());
  d.L = make_layout(V, E, prop);
  const Layout& L = d.L;
  d.row_offsets = device_alloc<int>(d, V + 1);
  d.columns = device_alloc<int>(d, E);
  d.out_degrees = device_alloc<int>(d, V);
  d.scores = device_alloc<float>(d, V);
  d.contributions_a = device_alloc<float>(d, V);
  d.contributions_b = device_alloc<float>(d, V);
  Params& p = d.params;
  p.row_offsets = d.row_offsets;
  p.columns = d.columns;
  p.out_degrees = d.out_degrees;
  p.vertices = V;
  p.edges = E;
  p.rows = L.rows;
  p.parts = L.parts;
  p.items = L.items;
  p.cap = L.cap;
  p.dst_bits = L.dst_bits;
  p.src_bits = L.src_bits;
  p.superbins = L.superbins;
  p.coarse_shift = L.coarse_shift;
  p.coarse_bins = L.coarse_bins;
  p.fine_shift = L.fine_shift;
  p.fine_bins = L.fine_bins;
  p.base = (1.0f - kDamping) / V;
  p.damping = kDamping;
  p.part_edge = device_alloc<int>(d, L.parts + 1);
  p.part_slice = device_alloc<int>(d, L.parts + 1);
  p.slice_part = device_alloc<int>(d, L.slices_max);
  p.slice_rows = device_alloc<int>(d, 2 * static_cast<size_t>(L.slices_max));
  p.bucket_start = device_alloc<int>(d, static_cast<size_t>(L.parts) * (L.coarse_bins + 1));
  p.coarse_count = device_alloc<int>(d, static_cast<size_t>(L.parts) * L.coarse_bins);
  p.coarse_fill = device_alloc<int>(d, static_cast<size_t>(L.parts) * L.coarse_bins);
  p.superbin_start = device_alloc<int>(d, static_cast<size_t>(L.parts) * (L.superbins + 1));
  for (Schedule* sch : {&p.whole, &p.split}) {
    sch->item_part = device_alloc<int>(d, 2 * static_cast<size_t>(L.items));
    sch->part_items = device_alloc<int>(d, L.parts);
    sch->part_slot = device_alloc<int>(d, L.parts);
    sch->arrived = device_alloc<int>(d, L.parts);
    sch->cursor = device_alloc<int>(d, 2 * static_cast<size_t>(L.parts));
  }
  p.split_parts = L.split_parts;
  p.overflow = device_alloc<float>(d, V);
  if (L.binned) {
    p.staged = device_alloc<uint32_t>(d, E);
    p.records = device_alloc<uint32_t>(d, E);
    p.partial = device_alloc<float>(d, static_cast<size_t>(L.items) * L.cap);
  }
  return d;
}

void free_device(Device& d) {
  for (void* pointer : d.allocations) CUDA_CHECK(cudaFree(pointer));
  d = Device{};
}

struct RunOptions {
  const float* initial_scores = nullptr;  // null: uniform start
  int iterations = kIterations;
  bool include_transfers = true;
  int upload_chunks = kDefaultUploadChunks;
  int tail_splits = kDefaultTailSplits;
  int block_size = kDefaultBlockSize;
};

// One PageRank run on one input: graph-captured or enqueued directly.
struct Run {
  const Streams* streams = nullptr;
  const HostGraph* host = nullptr;
  Device* device = nullptr;
  RunOptions options;
  float* result = nullptr;
  std::vector<int> chunk_end;  // edges uploaded after each chunk
  int* chunk_end_device = nullptr;
  int* chunk_parts_device = nullptr;
  // 0: fork, 1: row offsets landed, 2: metadata landed, 3: first wave of the
  // final iteration done, 4: its scores downloaded, 5..: column chunks landed.
  std::vector<cudaEvent_t> events;
  cudaGraphExec_t executable = nullptr;
};

int grid_for(int threads_total, int block) { return std::max(1, (threads_total + block - 1) / block); }

void launch_binned_update(const Run& run, const Schedule& sch, int item_base, const float* in,
                          float* out, float* scores, int epoch) {
  const Device& d = *run.device;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(d.L.items);
  config.blockDim = dim3(kIterThreads);
  config.dynamicSmemBytes = static_cast<size_t>(d.L.cap) * sizeof(float);
  config.stream = run.streams->compute;
  cudaLaunchAttribute attribute[1];
  int n = 0;
  if (d.L.cooperative) {
    attribute[n].id = cudaLaunchAttributeCooperative;
    attribute[n].val.cooperative = 1;
    ++n;
  }
  config.attrs = attribute;
  config.numAttrs = n;
  CUDA_CHECK(cudaLaunchKernelEx(&config, binned_update_kernel, d.params, sch, item_base, in, out,
                                scores, epoch));
}

void enqueue_pagerank(const Run& run) {
  const Streams& s = *run.streams;
  const HostGraph& host = *run.host;
  const Device& d = *run.device;
  const Layout& L = d.L;
  const RunOptions& o = run.options;
  const int V = host.vertices;
  Params p = d.params;
  p.chunk_end = run.chunk_end_device;
  p.chunk_parts = run.chunk_parts_device;
  p.chunks = static_cast<int>(run.chunk_end.size());
  const int sms = L.sms;
  const bool convert = L.binned && o.iterations > 1;
  const int bs = o.block_size;

  if (o.include_transfers) {
    CUDA_CHECK(cudaEventRecord(run.events[0], s.compute));
    CUDA_CHECK(cudaStreamWaitEvent(s.upload, run.events[0], 0));
    CUDA_CHECK(cudaMemcpyAsync(d.row_offsets, host.row_offsets.data(), sizeof(int) * (V + 1),
                               cudaMemcpyHostToDevice, s.upload));
    CUDA_CHECK(cudaEventRecord(run.events[1], s.upload));
    CUDA_CHECK(cudaMemcpyAsync(d.out_degrees, host.out_degrees.data(), sizeof(int) * V,
                               cudaMemcpyHostToDevice, s.upload));
    if (o.initial_scores)
      CUDA_CHECK(cudaMemcpyAsync(d.scores, o.initial_scores, sizeof(float) * V,
                                 cudaMemcpyHostToDevice, s.upload));
    CUDA_CHECK(cudaEventRecord(run.events[2], s.upload));
    int begin = 0;
    for (size_t c = 0; c < run.chunk_end.size(); ++c) {
      const int end = run.chunk_end[c];
      if (end > begin)
        CUDA_CHECK(cudaMemcpyAsync(d.columns + begin, host.columns.data() + begin,
                                   sizeof(int) * (end - begin), cudaMemcpyHostToDevice, s.upload));
      CUDA_CHECK(cudaEventRecord(run.events[5 + c], s.upload));
      begin = end;
    }
    CUDA_CHECK(cudaStreamWaitEvent(s.compute, run.events[1], 0));
  }
  plan_kernel<<<1, 1024, 0, s.compute>>>(p);
  CUDA_CHECK(cudaGetLastError());
  if (o.include_transfers) CUDA_CHECK(cudaStreamWaitEvent(s.compute, run.events[2], 0));
  init_kernel<<<grid_for(V, 256), 256, 0, s.compute>>>(p, d.scores, d.contributions_a,
                                                       o.initial_scores == nullptr);
  CUDA_CHECK(cudaGetLastError());

  float* in = d.contributions_a;
  float* out = d.contributions_b;
  const auto swap = [&] { std::swap(in, out); };
  // The first iteration runs on the CSR, chunk by chunk as the columns land,
  // unless the whole graph is already on the device and converted (compute
  // only), in which case it is a binned iteration like the rest.
  const bool chunked_first = o.iterations > 0 && (o.include_transfers || !convert);
  int iteration = 0;
  for (size_t c = 0; c < run.chunk_end.size(); ++c) {
    if (o.include_transfers) CUDA_CHECK(cudaStreamWaitEvent(s.compute, run.events[5 + c], 0));
    if (chunked_first) {
      // Iteration 0 on the CSR rows of the parts this chunk completed.
      csr_update_kernel<<<sms * 8, bs, 0, s.compute>>>(p, 0, 0, in, out,
                                                   o.iterations == 1 ? d.scores : nullptr,
                                                   static_cast<int>(c));
      CUDA_CHECK(cudaGetLastError());
    }
    if (convert) {
      count_kernel<<<sms * 4, kSliceThreads, 0, s.compute>>>(p, static_cast<int>(c));
      CUDA_CHECK(cudaGetLastError());
      prefix_kernel<<<sms, kCoarseBins, 0, s.compute>>>(p, static_cast<int>(c));
      CUDA_CHECK(cudaGetLastError());
      scatter_kernel<<<sms * 4, kSliceThreads, 0, s.compute>>>(p, static_cast<int>(c));
      CUDA_CHECK(cudaGetLastError());
      fine_kernel<<<sms * 8, kFineThreads, 0, s.compute>>>(p, static_cast<int>(c));
      CUDA_CHECK(cudaGetLastError());
    }
  }
  if (chunked_first) {
    swap();
    iteration = 1;
  }
  // With transfers, the final iteration runs in two waves so the first half
  // of the scores downloads (on the idle upload stream) while the second half
  // is computed.
  static const bool split_env = [] {
    const char* v = std::getenv("PAGERANK_SPLIT_FINAL");
    return !v || std::atoi(v) != 0;
  }();
  const bool split_final = o.include_transfers && L.split_parts > 0 && split_env;
  bool downloaded = false;
  for (int epoch = 0; iteration < o.iterations; ++iteration) {
    const bool last = iteration + 1 == o.iterations;
    float* scores = last ? d.scores : nullptr;
    if (L.binned && last && split_final) {
      const int split_rows = L.split_parts * L.rows;
      launch_binned_update(run, d.params.split, 0, in, out, scores, 0);
      CUDA_CHECK(cudaEventRecord(run.events[3], s.compute));
      CUDA_CHECK(cudaStreamWaitEvent(s.upload, run.events[3], 0));
      CUDA_CHECK(cudaMemcpyAsync(run.result, d.scores, sizeof(float) * split_rows,
                                 cudaMemcpyDeviceToHost, s.upload));
      CUDA_CHECK(cudaEventRecord(run.events[4], s.upload));
      launch_binned_update(run, d.params.split, L.items, in, out, scores, 0);
      CUDA_CHECK(cudaMemcpyAsync(run.result + split_rows, d.scores + split_rows,
                                 sizeof(float) * (V - split_rows), cudaMemcpyDeviceToHost, s.compute));
      CUDA_CHECK(cudaStreamWaitEvent(s.compute, run.events[4], 0));
      downloaded = true;
    } else if (L.binned) {
      launch_binned_update(run, d.params.whole, 0, in, out, scores, epoch++);
    } else {
      csr_update_kernel<<<sms * 8, bs, 0, s.compute>>>(p, 0, V, in, out, scores, -1);
      CUDA_CHECK(cudaGetLastError());
    }
    swap();
  }
  if (o.include_transfers && !downloaded)
    CUDA_CHECK(cudaMemcpyAsync(run.result, d.scores, sizeof(float) * V, cudaMemcpyDeviceToHost,
                               s.compute));
}

// Chunk boundaries depend only on E and the chunk count: upload chunks are
// equal slices of the column array (whole 4 KB pages), the last one ending
// at E, except that the last slice is split again into halves, quarters ...
// (`tail_splits` times), so the work left after the final copy is small.
std::vector<int> make_chunks(int edges, int chunks, int tail_splits) {
  std::vector<int> ends;
  const long long per = std::max<long long>(1024, ((long long)edges / chunks + 1023) / 1024 * 1024);
  long long end = per;
  for (; end < edges; end += per) ends.push_back(static_cast<int>(end));
  // Split [last uniform end, edges) geometrically.
  long long begin = ends.empty() ? 0 : ends.back();
  for (int k = 0; k < tail_splits; ++k) {
    const long long half = ((edges - begin) / 2 + 1023) / 1024 * 1024;
    if (half < 4096 || begin + half >= edges) break;
    begin += half;
    ends.push_back(static_cast<int>(begin));
  }
  ends.push_back(edges);
  return ends;
}

Run prepare_pagerank(const Streams& s, const HostGraph& host, Device& device,
                     const RunOptions& options, float* result, bool graph) {
  Run run;
  run.streams = &s;
  run.host = &host;
  run.device = &device;
  run.options = options;
  run.result = result;
  run.chunk_end = options.include_transfers
                      ? make_chunks(device.L.edges, options.upload_chunks, options.tail_splits)
                      : std::vector<int>{device.L.edges};
  // The chunk boundaries are constants of the run's shape, like the kernel
  // grids; they are uploaded once when the run is prepared.
  CUDA_CHECK(cudaMalloc(&run.chunk_end_device, sizeof(int) * run.chunk_end.size()));
  CUDA_CHECK(cudaMalloc(&run.chunk_parts_device, sizeof(int) * (run.chunk_end.size() + 1)));
  CUDA_CHECK(cudaMemcpy(run.chunk_end_device, run.chunk_end.data(),
                        sizeof(int) * run.chunk_end.size(), cudaMemcpyHostToDevice));
  run.events.resize(5 + run.chunk_end.size());
  for (cudaEvent_t& event : run.events)
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  if (graph) {
    cudaGraph_t captured = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s.compute, cudaStreamCaptureModeGlobal));
    enqueue_pagerank(run);
    CUDA_CHECK(cudaStreamEndCapture(s.compute, &captured));
    CUDA_CHECK(cudaGraphInstantiate(&run.executable, captured, 0));
    CUDA_CHECK(cudaGraphDestroy(captured));
  }
  return run;
}

void launch_pagerank(const Run& run) {
  if (run.executable)
    CUDA_CHECK(cudaGraphLaunch(run.executable, run.streams->compute));
  else
    enqueue_pagerank(run);
}

void destroy_pagerank(Run& run) {
  if (run.executable) CUDA_CHECK(cudaGraphExecDestroy(run.executable));
  for (cudaEvent_t event : run.events) CUDA_CHECK(cudaEventDestroy(event));
  CUDA_CHECK(cudaFree(run.chunk_end_device));
  CUDA_CHECK(cudaFree(run.chunk_parts_device));
  run = Run{};
}

cudaDeviceProp device_properties() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop;
}

void configure_kernels(const cudaDeviceProp& prop) {
  CUDA_CHECK(cudaFuncSetAttribute(binned_update_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(prop.sharedMemPerBlockOptin - 64)));
}

bool validate_case(const Streams& streams, const Settings& settings, const cudaDeviceProp& prop,
                   const char* name, const HostGraph& graph, const std::vector<float>* initial,
                   int iterations) {
  if (!check_case_graph(name, graph)) return false;
  const int vertices = graph.vertices;
  std::vector<float> actual(vertices);
  std::vector<void*> pinned = pin_host_buffers(settings.pin_host, graph, initial, actual);
  Device device = allocate_device(graph, prop);
  RunOptions options;
  options.initial_scores = initial ? initial->data() : nullptr;
  options.iterations = iterations;
  options.block_size = settings.block_size;
  options.upload_chunks = settings.upload_chunks;
  options.tail_splits = settings.tail_splits;
  Run run = prepare_pagerank(streams, graph, device, options, actual.data(), settings.graph);
  launch_pagerank(run);
  CUDA_CHECK(cudaStreamSynchronize(streams.compute));
  destroy_pagerank(run);
  free_device(device);
  unpin_host_buffers(pinned);
  const std::vector<double> expected = reference_pagerank(
      graph, initial ? *initial : std::vector<float>(vertices, 1.0f / vertices), iterations);
  return compare_scores(name, graph, iterations, actual, expected);
}

bool run_validation(const Streams& streams, const Settings& settings, const cudaDeviceProp& prop,
                    const HostGraph& fixture) {
  const bool ok = run_validation_cases(
      fixture, [&](const char* name, const HostGraph& graph, const std::vector<float>* initial,
                   int iterations) {
        return validate_case(streams, settings, prop, name, graph, initial, iterations);
      });
  std::printf("validation %s (launch=%s block_size=%d pin_host=%d upload_chunks=%d)\n",
              ok ? "passed" : "FAILED", settings.graph ? "graph" : "direct",
              settings.block_size, settings.pin_host, settings.upload_chunks);
  return ok;
}

float time_replays(const Streams& streams, const std::vector<Run>& runs, int replays,
                   const char* range) {
  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, streams.compute));
  for (int replay = 0; replay < replays; ++replay) {
    nvtx_push(range);
    launch_pagerank(runs[replay % runs.size()]);
    nvtx_pop();
  }
  CUDA_CHECK(cudaEventRecord(stop, streams.compute));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms / replays;
}

bool run_benchmark(const Streams& streams, const Settings& settings, const cudaDeviceProp& prop,
                   const HostGraph& fixture, const InputOptions& input_options) {
  std::vector<BenchInput> inputs = make_bench_inputs(input_options, fixture);
  std::vector<std::vector<void*>> pinned;
  for (BenchInput& input : inputs) {
    std::fill(input.result.begin(), input.result.end(), std::nanf(""));
    pinned.push_back(pin_host_buffers(settings.pin_host, input.graph, input.initial_or_null(),
                                      input.result));
  }
  for (const BenchInput& input : inputs)
    if (input.graph.vertices != inputs[0].graph.vertices ||
        input.graph.columns.size() != inputs[0].graph.columns.size()) {
      std::fprintf(stderr, "rotated inputs must share one shape\n");
      return false;
    }
  Device device = allocate_device(inputs[0].graph, prop);
  RunOptions options;
  options.block_size = settings.block_size;
  options.upload_chunks = settings.upload_chunks;
  options.tail_splits = settings.tail_splits;
  std::vector<Run> full;
  for (BenchInput& input : inputs) {
    options.initial_scores = input.initial.empty() ? nullptr : input.initial.data();
    full.push_back(prepare_pagerank(streams, input.graph, device, options, input.result.data(),
                                    settings.graph));
  }
  options.initial_scores = inputs[0].initial.empty() ? nullptr : inputs[0].initial.data();
  options.include_transfers = false;
  // Compute only: the raw CSR is already on the device (input 0's, uploaded by
  // the full run just before); the conversion and all twenty iterations run.
  std::vector<Run> compute_only{
      prepare_pagerank(streams, inputs[0].graph, device, options, nullptr, settings.graph)};
  if (settings.graph) {
    for (Run& run : full) CUDA_CHECK(cudaGraphUpload(run.executable, streams.compute));
    CUDA_CHECK(cudaGraphUpload(compute_only[0].executable, streams.compute));
  }

  // Warm-up (and the calibration when GRAPH_REPLAYS is unset): one run of
  // each input, so first-launch costs stay out of the timed replays.
  const float replay_ms =
      time_replays(streams, full, static_cast<int>(full.size()), "pagerank.calibrate");
  int graph_replays = 0;
  if (const char* value = std::getenv("GRAPH_REPLAYS")) graph_replays = std::atoi(value);
  if (graph_replays <= 0)
    graph_replays = replay_ms > 0.0f ? static_cast<int>(10000.0f / replay_ms) + 1 : 1;
  graph_replays = std::max<int>(graph_replays, static_cast<int>(inputs.size()));
  const float full_ms = time_replays(streams, full, graph_replays, "pagerank.full");
  const float single_ms = time_single_ms(
      streams.compute, [&](int replay) { launch_pagerank(full[replay % full.size()]); },
      "pagerank.full.single");
  if (inputs.size() > 1) {
    launch_pagerank(full[0]);
    CUDA_CHECK(cudaStreamSynchronize(streams.compute));
  }
  const float compute_ms = time_replays(streams, compute_only, graph_replays, "pagerank.compute");

  const std::vector<float>& h_result = inputs[0].result;
  const char* launch = settings.graph ? "graph" : "direct";
  std::printf("PageRank scores after %d iterations (launch=%s replays=%d rotate=%zu): %.6f %.6f %.6f %.6f\n",
              kIterations, launch, graph_replays, inputs.size(), h_result[0], h_result[1],
              h_result[2], h_result[3]);
  const bool ok = check_bench_inputs(input_options, inputs, true);
  const Layout& L = device.L;
  const int chunks = static_cast<int>(full[0].chunk_end.size());
  std::printf("layout: %s, %d parts of %d rows, %d blocks per update (%d shared-memory rows per block)\n",
              L.binned ? "binned" : "CSR fallback", L.parts, L.rows, L.items, L.cap);
  if (L.binned) {
    const bool split = L.split_parts > 0 && !(std::getenv("PAGERANK_SPLIT_FINAL") &&
                                              std::atoi(std::getenv("PAGERANK_SPLIT_FINAL")) == 0);
    const int updates = kIterations - 1 + (split ? 1 : 0);
    std::printf("kernel launches per full run: plan_kernel, init_kernel, then for each of %d upload chunks "
                "csr_update_kernel, count_kernel, prefix_kernel, scatter_kernel, fine_kernel, then %d "
                "binned_update_kernel (%s) = %d\n",
                chunks, updates, split ? "the last two are the final iteration's two waves" : "one per iteration",
                2 + 5 * chunks + updates);
  }
  // Before the per-replay summary, which `profile_cuda.sh sweep` reads as the last line.
  std::printf("batch: %.3f ms for %d back-to-back replays with transfers; single: %.3f ms "
              "per replay with transfers (median of %d replays, each launched alone)\n",
              full_ms * graph_replays, graph_replays, single_ms, kSingleReplays);
  std::printf("launch=%s block_size=%d pin_host=%d upload_chunks=%d: %.3f ms per replay with "
              "transfers, %.3f ms compute only (%.4f ms per iteration)\n",
              launch, settings.block_size, settings.pin_host, settings.upload_chunks, full_ms,
              compute_ms, compute_ms / kIterations);

  destroy_pagerank(compute_only[0]);
  for (Run& run : full) destroy_pagerank(run);
  free_device(device);
  for (std::vector<void*>& buffers : pinned) unpin_host_buffers(buffers);
  return ok;
}

}  // namespace

int main() {
  const char* mode_value = std::getenv("PAGERANK_MODE");
  const std::string mode = mode_value ? mode_value : "all";
  if (mode != "all" && mode != "validate" && mode != "bench") {
    std::fprintf(stderr, "PAGERANK_MODE must be all, validate or bench (got \"%s\")\n", mode.c_str());
    return 2;
  }
  Settings settings;
  if (const char* value = std::getenv("PAGERANK_LAUNCH")) {
    const std::string launch = value;
    if (launch != "graph" && launch != "direct") {
      std::fprintf(stderr, "PAGERANK_LAUNCH must be graph or direct (got \"%s\")\n", value);
      return 2;
    }
    settings.graph = launch == "graph";
  }
  const char* block_size_value = std::getenv("PAGERANK_BLOCK_SIZE");
  if (block_size_value) settings.block_size = std::atoi(block_size_value);
  if (settings.block_size < 32 || settings.block_size > 1024 || settings.block_size % 32 != 0) {
    std::fprintf(stderr, "PAGERANK_BLOCK_SIZE must be a multiple of 32 from 32 to 1024 (got \"%s\")\n",
                 block_size_value);
    return 2;
  }
  if (const char* value = std::getenv("PAGERANK_PIN_HOST")) {
    const std::string pin_host = value;
    if (pin_host != "0" && pin_host != "1") {
      std::fprintf(stderr, "PAGERANK_PIN_HOST must be 0 or 1 (got \"%s\")\n", value);
      return 2;
    }
    settings.pin_host = pin_host == "1";
  }
  if (const char* value = std::getenv("PAGERANK_TAIL_SPLITS")) {
    settings.tail_splits = std::atoi(value);
    if (settings.tail_splits < 0 || settings.tail_splits > 16) {
      std::fprintf(stderr, "PAGERANK_TAIL_SPLITS must be from 0 to 16 (got \"%s\")\n", value);
      return 2;
    }
  }
  if (const char* value = std::getenv("PAGERANK_UPLOAD_CHUNKS")) {
    settings.upload_chunks = std::atoi(value);
    if (settings.upload_chunks < 1 || settings.upload_chunks > 1024) {
      std::fprintf(stderr, "PAGERANK_UPLOAD_CHUNKS must be from 1 to 1024 (got \"%s\")\n", value);
      return 2;
    }
  }
  const InputOptions input_options = read_input_options(kDefaultRotate);

  const cudaDeviceProp prop = device_properties();
  configure_kernels(prop);
  Streams streams;
  CUDA_CHECK(cudaStreamCreateWithFlags(&streams.compute, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&streams.upload, cudaStreamNonBlocking));
  const HostGraph fixture = make_benchmark_fixture();
  bool ok = true;
  if (mode != "bench") ok = run_validation(streams, settings, prop, fixture);
  if (ok && mode != "validate") ok = run_benchmark(streams, settings, prop, fixture, input_options);
#if TIMING
  {
    static unsigned long long times[1024][5];
    CUDA_CHECK(cudaMemcpyFromSymbol(times, g_times, sizeof(times)));
    unsigned long long t0 = ~0ull, tend = 0;
    for (int b = 0; b < 82; ++b) { t0 = std::min(t0, times[b][0]); tend = std::max(tend, times[b][4]); }
    double sum[4] = {0, 0, 0, 0};
    for (int b = 0; b < 82; ++b) {
      for (int k = 0; k < 4; ++k) sum[k] += (times[b][k + 1] - times[b][k]) / 1e3;
      if (b < 6 || b > 78)
        std::printf("block %2d start %+7.1f zero %6.1f proc %6.1f barrier %6.1f final %6.1f end %+7.1f us\n", b,
                    (times[b][0] - t0) / 1e3, (times[b][1] - times[b][0]) / 1e3, (times[b][2] - times[b][1]) / 1e3,
                    (times[b][3] - times[b][2]) / 1e3, (times[b][4] - times[b][3]) / 1e3, (times[b][4] - t0) / 1e3);
    }
    std::printf("proc-end per part:");
    for (int b = 0; b < 82; b += 2) std::printf(" %.0f", (std::max(times[b][2], times[b + 1][2]) - t0) / 1e3);
    std::printf("\nmean zero %.1f proc %.1f barrier %.1f final %.1f; span %.1f us\n", sum[0] / 82, sum[1] / 82,
                sum[2] / 82, sum[3] / 82, (tend - t0) / 1e3);
  }
#endif
  CUDA_CHECK(cudaStreamDestroy(streams.upload));
  CUDA_CHECK(cudaStreamDestroy(streams.compute));
  return ok ? 0 : 1;
}
