// Cumulative optimization stages of loom-private/pagerank, between the
// baseline (applications/loom_private/pagerank/pagerank.cu) and the optimized
// variant (pagerank_optimized.cu in the same directory). See README.md.
//
//   -DLP_STAGE=1  one persistent cooperative kernel per invocation; a grid
//                 barrier after every dependency level replaces the
//                 baseline's launch per level; original node numbering.
//   -DLP_STAGE=2  stage 1 with per-node readiness tags: inside a sweep a node
//                 waits only for the sources it reads, so the only barriers
//                 left separate the sweeps; original node numbering.
//   -DLP_STAGE=3  stage 2 over the level-ordered layout (layout.cpp). This is
//                 pagerank_optimized.cu's kernel, unchanged, with the
//                 optimized variant's generic "default" launch shape instead
//                 of the per-GPU tuning table, and its serial uploads, as
//                 the optimized variant had them before the map overlap.
//   -DLP_STAGE=4  stage 3 with the map upload overlapped: the application
//                 scope uploads the position-to-node map, which only the
//                 last sweep reads, on a side stream during the sweeps, and
//                 the last sweep waits for a flag copied after it. This is
//                 pagerank_optimized.cu, unchanged, with the "default" shape.
//
// Stage 5, the per-GPU tuning, is the production optimized binary itself.
// Every stage uses the "default" shape: 512 threads per block, at most 2
// resident blocks per SM, graph arrays through the read-only cache (__ldg).
//
// Every stage keeps the contract's ordered Gauss-Seidel updates: one thread
// sums each node's incoming list in order, in float, with the baseline's
// operations, so every stage's scores equal the sequential loop bitwise.
// Stages 1 and 2 write the scores as the baseline does (the 1/N initial
// scores, then every sweep); stages 3 and 4 write them only in the last
// sweep, as the optimized variant does.
#include "lp_harness.cuh"

#include "graph.h"
#include "layout.h"
#include "oracle.h"

#include <cmath>

#if !defined(LP_STAGE) || LP_STAGE < 1 || LP_STAGE > 4
#error "build with -DLP_STAGE=1, 2, 3 or 4"
#endif

namespace {

constexpr int kDefaultNodes = 1 << 20;
constexpr int kDefaultIterations = 20;
constexpr long long kLargeEdges = 2097151;
constexpr long long kMaxNodes = 1LL << 26;
constexpr int kMaxIterations = 1000;
constexpr float kDamping = 0.85f;
constexpr float kRelativeTolerance = 1.0e-5f;
#if LP_STAGE >= 2
// Readiness protocol helpers (stages 2 and 3).
constexpr unsigned kSign = 0x80000000u;
// Never a contribution; its tag is not sweep 0's, so sweep 0 waits on it.
constexpr unsigned kNotReady = 0xFFFFFFFFu;

__device__ __forceinline__ unsigned load_relaxed(const unsigned *address) {
  unsigned value;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void store_relaxed(unsigned *address, unsigned value) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(address), "r"(value)
               : "memory");
}
#endif

// Every spin wait gives up after this many SM clock cycles and traps, so a
// broken protocol ends as a CUDA error (exit 70) instead of a hang.
constexpr long long kSpinLimit = 1LL << 34;

__device__ __forceinline__ void check_spin(long long start) {
  if (clock64() - start > kSpinLimit) __trap();
}

// Grid-wide barrier over co-resident blocks; `epoch` counts the barriers this
// invocation has passed, and the counter is zeroed before each launch.
__device__ __forceinline__ void grid_barrier(unsigned *counter, unsigned epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned target = epoch * gridDim.x;
    __threadfence();
    atomicAdd(counter, 1u);
    const long long start = clock64();
    unsigned seen;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                   : "=r"(seen)
                   : "l"(counter)
                   : "memory");
      check_spin(start);
    } while (seen < target);
    __threadfence();
  }
  __syncthreads();
}

#if LP_STAGE == 1
// ---------------------------------------------------------------------------
// Stage 1: the baseline's level schedule inside one persistent kernel.

// The level bounds travel as a kernel argument, as the baseline passes each
// level's range in its launch arguments.
constexpr int kMaxLevels = 512;
struct LevelBounds {
  int count;
  int offsets[kMaxLevels + 1];
};

// The baseline's initialize and update_level kernels in one launch. Sweep k
// writes buffer k % 2 and reads the previous sweep from the other one; the
// grid barrier after each level makes the level's contributions visible to
// the next. Contributions are read from L2 (__ldcg): other blocks wrote them
// during this launch, so a line cached in this SM's L1 may be stale.
__global__ void __launch_bounds__(1024)
persistent_levels(const int *in_offsets, const int *in_sources, const int *out_degrees,
                  const int *level_nodes, LevelBounds levels, float *contrib,
                  float *scores, unsigned *barrier, int count, int iterations,
                  float initial, float base, float damping) {
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int n = first; n < count; n += stride) {
    scores[n] = initial;
    contrib[count + n] = initial / (float)__ldg(out_degrees + n);
  }
  unsigned epoch = 0;
  grid_barrier(barrier, ++epoch);
  for (int sweep = 0; sweep < iterations; ++sweep) {
    float *current = contrib + (size_t)(sweep & 1) * count;
    const float *previous = contrib + (size_t)((sweep & 1) ^ 1) * count;
    for (int level = 0; level < levels.count; ++level) {
      const int end = levels.offsets[level + 1];
      for (int t = levels.offsets[level] + first; t < end; t += stride) {
        const int u = __ldg(level_nodes + t);
        float total = 0.0f;
        const int edges_end = __ldg(in_offsets + u + 1);
        for (int e = __ldg(in_offsets + u); e < edges_end; ++e) {
          const int v = __ldg(in_sources + e);
#if defined(LP_PAGERANK_MUTANT_JACOBI)
          // Test-only mutant: every read from the previous sweep.
          total += __ldcg(previous + v);
#else
          total += v < u ? __ldcg(current + v) : __ldcg(previous + v);
#endif
        }
        const float score = base + damping * total;
        scores[u] = score;
        current[u] = score / (float)__ldg(out_degrees + u);
      }
      if (sweep + 1 < iterations || level + 1 < levels.count)
        grid_barrier(barrier, ++epoch);
    }
  }
}

#else
// ---------------------------------------------------------------------------
// Stages 2 and 3: per-node readiness (pagerank_optimized.cu's protocol).

// Readiness tag of the contributions a sweep stores: the sign bit, set for
// sweeps 2, 3, 6, 7, ... (see pagerank_optimized.cu).
__device__ __forceinline__ unsigned sweep_tag(int sweep) {
  return (unsigned)(sweep >> 1 & 1) << 31;
}

constexpr int kAhead = 3;

// Issues the read of one source contribution: this sweep's value for v < u
// (which may not be stored yet), the previous sweep's for v >= u. A source
// reference is 2 * index + (v < u).
__device__ __forceinline__ unsigned load_source(int source, const unsigned *current,
                                                const unsigned *previous) {
  return (source & 1) ? load_relaxed(current + (source >> 1))
                      : __ldcg(previous + (source >> 1));
}

__device__ __forceinline__ float ready_value(unsigned bits, int source,
                                             const unsigned *current, unsigned tag) {
  if ((source & 1) && (bits & kSign) != tag) {
    const long long start = clock64();
    do {
      bits = load_relaxed(current + (source >> 1));
      check_spin(start);
    } while ((bits & kSign) != tag);
  }
  return __uint_as_float(bits & ~kSign);
}

#if LP_STAGE == 2
// Stage 2 reads the original incoming CSR, so the source reference is formed
// here from the node ids instead of being stored by the layout.
__device__ __forceinline__ int source_reference(const int *in_sources, int e, int u) {
  const int v = __ldg(in_sources + e);
#if defined(LP_PAGERANK_MUTANT_JACOBI)
  return 2 * v; // test-only mutant: every read from the previous sweep
#else
  return 2 * v + (v < u);
#endif
}

// u's sum over its in-neighbours in list order, in float; the first kAhead
// loads are issued before any of them is waited on, as in the optimized
// variant.
__device__ __forceinline__ float in_sum(int u, int begin, int end, const int *in_sources,
                                        const unsigned *current, const unsigned *previous,
                                        unsigned tag) {
  int source[kAhead];
  unsigned bits[kAhead];
#pragma unroll
  for (int i = 0; i < kAhead; ++i)
    if (begin + i < end) source[i] = source_reference(in_sources, begin + i, u);
#pragma unroll
  for (int i = 0; i < kAhead; ++i)
    if (begin + i < end) bits[i] = load_source(source[i], current, previous);
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < kAhead; ++i)
    if (begin + i < end) total += ready_value(bits[i], source[i], current, tag);
  for (int e = begin + kAhead; e < end; ++e) {
    const int s = source_reference(in_sources, e, u);
    total += ready_value(load_source(s, current, previous), s, current, tag);
  }
  return total;
}

// Every thread visits positions first, first + stride, ... of the baseline's
// level list, so in level order: every wait is for a node of a lower level,
// which an earlier position holds. Contributions, scores and the graph stay
// indexed by node id.
__global__ void __launch_bounds__(1024)
persistent_tags(const int *in_offsets, const int *in_sources, const int *out_degrees,
                const int *level_nodes, unsigned *contrib, float *scores,
                unsigned *barrier, int count, int iterations, float initial, float base,
                float damping) {
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int n = first; n < count; n += stride) {
    scores[n] = initial;
    contrib[count + n] = sweep_tag(-1) | __float_as_uint(initial / (float)__ldg(out_degrees + n));
    contrib[n] = kNotReady;
  }
  unsigned epoch = 0;
  grid_barrier(barrier, ++epoch);
  for (int sweep = 0; sweep < iterations; ++sweep) {
    unsigned *current = contrib + (size_t)(sweep & 1) * count;
    const unsigned *previous = contrib + (size_t)((sweep & 1) ^ 1) * count;
    const unsigned tag = sweep_tag(sweep);
    for (int p = first; p < count; p += stride) {
      const int u = __ldg(level_nodes + p);
      const int degree = __ldg(out_degrees + u);
      const float total = in_sum(u, __ldg(in_offsets + u), __ldg(in_offsets + u + 1),
                                 in_sources, current, previous, tag);
      const float score = base + damping * total;
      store_relaxed(current + u, tag | __float_as_uint(score / (float)degree));
      scores[u] = score;
    }
    if (sweep + 1 < iterations) grid_barrier(barrier, ++epoch);
  }
}

#else // LP_STAGE 3 and 4
// Stage 4: pagerank_optimized.cu's kernel, copied unchanged (the run.py check
// compares its machine code with the production binary's gauss_seidel<false>).
// Stage 3: the same kernel as it was before the map overlap: no flag, and the
// map read through the read-only cache.

template <bool kStream>
__device__ __forceinline__ int load_graph(const int *address) {
  return kStream ? __ldcs(address) : __ldg(address);
}

template <bool kStream>
__device__ __forceinline__ int source_entry(const int *sources, int e) {
#if defined(LP_PAGERANK_MUTANT_JACOBI)
  return load_graph<kStream>(sources + e) & ~1;
#else
  return load_graph<kStream>(sources + e);
#endif
}

template <bool kStream>
__device__ __forceinline__ float in_sum(int begin, int end, const int *sources,
                                        const unsigned *current,
                                        const unsigned *previous, unsigned tag) {
  int source[kAhead];
  unsigned bits[kAhead];
#pragma unroll
  for (int i = 0; i < kAhead; ++i)
    if (begin + i < end) source[i] = source_entry<kStream>(sources, begin + i);
#pragma unroll
  for (int i = 0; i < kAhead; ++i)
    if (begin + i < end) bits[i] = load_source(source[i], current, previous);
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < kAhead; ++i)
    if (begin + i < end) total += ready_value(bits[i], source[i], current, tag);
  for (int e = begin + kAhead; e < end; ++e) {
    const int s = source_entry<kStream>(sources, e);
    total += ready_value(load_source(s, current, previous), s, current, tag);
  }
  return total;
}

template <bool kStream>
__global__ void __launch_bounds__(1024)
#if LP_STAGE >= 4
gauss_seidel(const int *offsets, const int *sources, const int *out_degrees,
             const int *nodes, const unsigned *nodes_ready, unsigned *contrib,
             float *scores, unsigned *barrier, int count, int iterations,
             float initial, float base, float damping) {
#else
gauss_seidel(const int *offsets, const int *sources, const int *out_degrees,
             const int *nodes, unsigned *contrib, float *scores, unsigned *barrier,
             int count, int iterations, float initial, float base, float damping) {
#endif
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int p = first; p < count; p += stride) {
    contrib[count + p] = sweep_tag(-1) | __float_as_uint(initial / (float)out_degrees[p]);
    contrib[p] = kNotReady;
  }
  unsigned epoch = 0;
  grid_barrier(barrier, ++epoch);
  for (int sweep = 0; sweep < iterations; ++sweep) {
    unsigned *current = contrib + (size_t)(sweep & 1) * count;
    const unsigned *previous = contrib + (size_t)((sweep & 1) ^ 1) * count;
    const unsigned tag = sweep_tag(sweep);
    const bool last = sweep + 1 == iterations;
#if LP_STAGE >= 4
    if (last && nodes_ready) {
      if (threadIdx.x == 0) {
        const long long start = clock64();
        unsigned ready;
        do {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                       : "=r"(ready)
                       : "l"(nodes_ready)
                       : "memory");
          check_spin(start);
        } while (ready == 0);
      }
      __syncthreads();
    }
#endif
    for (int p = first; p < count; p += stride) {
      const int degree = load_graph<kStream>(out_degrees + p);
      const float total =
          in_sum<kStream>(load_graph<kStream>(offsets + p),
                          load_graph<kStream>(offsets + p + 1), sources, current,
                          previous, tag);
      const float score = base + damping * total;
      store_relaxed(current + p, tag | __float_as_uint(score / (float)degree));
#if LP_STAGE >= 4
      if (last) scores[__ldcg(nodes + p)] = score;
#else
      if (last) scores[__ldg(nodes + p)] = score;
#endif
    }
    if (!last) grid_barrier(barrier, ++epoch);
  }
}
#endif
#endif

// The optimized variant's generic launch shape ("default" in its tuning
// table). A test build may force a one-block, 64-thread grid instead.
struct Shape {
  const char *name;
  int threads, blocks_per_sm;
};

Shape select_shape() {
#if defined(LP_ABLATION_TEST_SHAPE)
  if (const char *forced = std::getenv("LP_ABLATION_SHAPE")) {
    if (std::string(forced) == "tiny") return {"tiny", 64, 0};
    if (*forced && std::string(forced) != "default")
      lp::fail(lp::kExitUsage, "unknown shape %s", forced);
  }
#endif
  return {"default", 512, 2};
}

class Pagerank final : public lp::Workload {
public:
  Pagerank(int nodes, int iterations)
      : nodes_(nodes), iterations_(iterations),
        expected_edges_(pagerank::expected_edge_count(nodes)) {}
  ~Pagerank() override {
    cudaFreeHost(h_offsets_);
    cudaFreeHost(h_sources_);
    cudaFreeHost(h_out_degrees_);
    cudaFreeHost(h_nodes_);
    cudaFreeHost(h_scores_);
    cudaFree(d_offsets_);
    cudaFree(d_sources_);
    cudaFree(d_out_degrees_);
    cudaFree(d_nodes_);
    cudaFree(d_scores_);
    cudaFree(d_contrib_);
    cudaFree(d_barrier_);
#if LP_STAGE >= 4
    cudaFree(d_nodes_ready_);
    cudaFreeHost(h_ready_);
    if (side_) cudaStreamDestroy(side_);
    if (fork_) cudaEventDestroy(fork_);
    if (join_) cudaEventDestroy(join_);
#endif
  }

  std::string identity() const override { return "loom-private/pagerank"; }

  lp::JsonObject dimensions() const override {
    return {{"nodes", lp::Json::integer(nodes_)},
            {"iterations", lp::Json::integer(iterations_)},
            {"edges", lp::Json::integer(expected_edges_)},
            {"damping", lp::Json::number(kDamping)},
            {"epsilon", lp::Json::number(0.0)},
            {"schedule_levels", lp::Json::integer(levels_)}};
  }

  // The contract's scope strings, word for word, as in the optimized variant.
  std::string application_scope() const override {
    return "upload of the incoming CSR (offsets and sources), out-degrees and "
           "level schedule from pinned host memory, score and contribution "
           "initialization, all ordered Gauss-Seidel sweeps (one launch per "
           "dependency level per sweep), and the score download to pinned "
           "host memory";
  }
  std::string compute_scope() const override {
    return "score and contribution initialization and all ordered "
           "Gauss-Seidel sweeps with the graph and schedule resident; no "
           "transfers";
  }

  void setup(cudaStream_t) override {
    const pagerank::HostGraph g = pagerank::build_graph(nodes_);
    if (g.edges != expected_edges_ ||
        (nodes_ == kDefaultNodes && g.edges != kLargeEdges))
      lp::fail(lp::kExitMismatch, "generated %lld edges, expected %lld",
               g.edges, expected_edges_);
    edges_ = (int)g.edges;
    levels_ = g.levels();
#if LP_STAGE == 1
    if (levels_ > kMaxLevels)
      lp::fail(lp::kExitUsage, "%d dependency levels exceed stage 1's %d", levels_,
               kMaxLevels);
    bounds_.count = levels_;
    std::copy(g.level_offsets.begin(), g.level_offsets.end(), bounds_.offsets);
#endif
#if LP_STAGE >= 3
    // The four arrays and byte counts of the baseline's upload, renumbered.
    const pagerank::LevelLayout layout = pagerank::build_level_layout(g);
    const std::vector<int> &offsets = layout.offsets, &sources = layout.sources,
                           &out_degrees = layout.out_degrees, &nodes = layout.nodes;
#else
    // The baseline's arrays: incoming CSR, out-degrees and level list.
    const std::vector<int> &offsets = g.in_offsets, &sources = g.in_sources,
                           &out_degrees = g.out_degrees, &nodes = g.level_nodes;
#endif

#if LP_STAGE == 1
    kernel_ = (const void *)persistent_levels;
#elif LP_STAGE == 2
    kernel_ = (const void *)persistent_tags;
#else
    kernel_ = (const void *)gauss_seidel<false>;
#endif
    int device = 0;
    cudaDeviceProp properties{};
    LP_CUDA(cudaGetDevice(&device));
    LP_CUDA(cudaGetDeviceProperties(&properties, device));
    const Shape shape = select_shape();
    threads_ = shape.threads;
    int resident = 0;
    LP_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel_, threads_, 0));
    if (resident < 1) lp::fail(lp::kExitCuda, "persistent kernel does not fit an SM");
    const long long wanted =
        shape.blocks_per_sm == 0
            ? 1
            : (long long)std::min(resident, shape.blocks_per_sm) * properties.multiProcessorCount;
    const long long needed = (nodes_ + threads_ - 1) / threads_;
    blocks_ = (int)std::max(1LL, std::min(wanted, needed));
    std::fprintf(stderr,
                 "%s: stage %d: %d nodes, %d edges, %d dependency levels; %s shape: "
                 "%d blocks x %d threads\n",
                 LP_APP, LP_STAGE, nodes_, edges_, levels_, shape.name, blocks_, threads_);

    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    const size_t edge_bytes = sizeof(int) * (size_t)edges_;
    LP_CUDA(cudaMallocHost(&h_offsets_, node_bytes + sizeof(int)));
    LP_CUDA(cudaMallocHost(&h_sources_, edge_bytes));
    LP_CUDA(cudaMallocHost(&h_out_degrees_, node_bytes));
    LP_CUDA(cudaMallocHost(&h_nodes_, node_bytes));
    LP_CUDA(cudaMallocHost(&h_scores_, sizeof(float) * (size_t)nodes_));
    std::copy(offsets.begin(), offsets.end(), h_offsets_);
    std::copy(sources.begin(), sources.end(), h_sources_);
    std::copy(out_degrees.begin(), out_degrees.end(), h_out_degrees_);
    std::copy(nodes.begin(), nodes.end(), h_nodes_);

    LP_CUDA(cudaMalloc(&d_offsets_, node_bytes + sizeof(int)));
    LP_CUDA(cudaMalloc(&d_sources_, edge_bytes));
    LP_CUDA(cudaMalloc(&d_out_degrees_, node_bytes));
    LP_CUDA(cudaMalloc(&d_nodes_, node_bytes));
    LP_CUDA(cudaMalloc(&d_scores_, sizeof(float) * (size_t)nodes_));
    LP_CUDA(cudaMalloc(&d_contrib_, 2 * sizeof(unsigned) * (size_t)nodes_));
    LP_CUDA(cudaMalloc(&d_barrier_, sizeof(unsigned)));
#if LP_STAGE >= 4
    LP_CUDA(cudaMalloc(&d_nodes_ready_, sizeof(unsigned)));
    LP_CUDA(cudaMallocHost(&h_ready_, sizeof(unsigned)));
    *h_ready_ = 1;
    LP_CUDA(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
    LP_CUDA(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
    LP_CUDA(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
#endif
  }

#if LP_STAGE >= 4
  // As in the optimized variant: the map and its flag upload on a side stream
  // that forks after the arrays the first sweep needs and joins before the
  // score download.
  void enqueue_application(cudaStream_t stream) override {
    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    LP_CUDA(cudaMemsetAsync(d_nodes_ready_, 0, sizeof(unsigned), stream));
    LP_CUDA(cudaMemcpyAsync(d_offsets_, h_offsets_, node_bytes + sizeof(int),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_sources_, h_sources_, sizeof(int) * (size_t)edges_,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_out_degrees_, h_out_degrees_, node_bytes,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaEventRecord(fork_, stream));
    LP_CUDA(cudaStreamWaitEvent(side_, fork_, 0));
    LP_CUDA(cudaMemcpyAsync(d_nodes_, h_nodes_, node_bytes, cudaMemcpyHostToDevice,
                            side_));
    LP_CUDA(cudaMemcpyAsync(d_nodes_ready_, h_ready_, sizeof(unsigned),
                            cudaMemcpyHostToDevice, side_));
    LP_CUDA(cudaEventRecord(join_, side_));
    launch(stream, d_nodes_ready_);
    LP_CUDA(cudaStreamWaitEvent(stream, join_, 0));
    LP_CUDA(cudaMemcpyAsync(h_scores_, d_scores_, sizeof(float) * (size_t)nodes_,
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override { launch(stream, nullptr); }
#else
  void enqueue_application(cudaStream_t stream) override {
    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    LP_CUDA(cudaMemcpyAsync(d_offsets_, h_offsets_, node_bytes + sizeof(int),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_sources_, h_sources_, sizeof(int) * (size_t)edges_,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_out_degrees_, h_out_degrees_, node_bytes,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_nodes_, h_nodes_, node_bytes, cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_scores_, d_scores_, sizeof(float) * (size_t)nodes_,
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override { launch(stream, nullptr); }
#endif

  // nodes_ready (stage 4): the flag the last sweep waits for, or null when the
  // map is resident. Earlier stages take no flag.
  void launch(cudaStream_t stream, [[maybe_unused]] const unsigned *nodes_ready) {
    const float initial = 1.0f / (float)nodes_;
    const float base = (1.0f - kDamping) / (float)nodes_;
    LP_CUDA(cudaMemsetAsync(d_barrier_, 0, sizeof(unsigned), stream));
    cudaLaunchAttribute cooperative{};
    cooperative.id = cudaLaunchAttributeCooperative;
    cooperative.val.cooperative = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks_);
    config.blockDim = dim3(threads_);
    config.stream = stream;
    config.attrs = &cooperative;
    config.numAttrs = 1;
    const int *offsets = d_offsets_, *sources = d_sources_, *degrees = d_out_degrees_,
              *nodes = d_nodes_;
#if LP_STAGE == 1
    LP_CUDA(cudaLaunchKernelEx(&config, persistent_levels, offsets, sources, degrees, nodes,
                               bounds_, reinterpret_cast<float *>(d_contrib_), d_scores_,
                               d_barrier_, nodes_, iterations_, initial, base, kDamping));
#elif LP_STAGE == 2
    LP_CUDA(cudaLaunchKernelEx(&config, persistent_tags, offsets, sources, degrees, nodes,
                               d_contrib_, d_scores_, d_barrier_, nodes_, iterations_,
                               initial, base, kDamping));
#elif LP_STAGE == 3
    LP_CUDA(cudaLaunchKernelEx(&config, gauss_seidel<false>, offsets, sources, degrees, nodes,
                               d_contrib_, d_scores_, d_barrier_, nodes_, iterations_,
                               initial, base, kDamping));
#else
    LP_CUDA(cudaLaunchKernelEx(&config, gauss_seidel<false>, offsets, sources, degrees, nodes,
                               nodes_ready, d_contrib_, d_scores_, d_barrier_, nodes_,
                               iterations_, initial, base, kDamping));
#endif
  }

  void poison_outputs(cudaStream_t stream) override {
    // As in the optimized variant: NaN scores and contributions, zeroed
    // inputs (an edgeless graph whose level list names only node 0), and a
    // barrier counter above every target, so a skipped reset fails.
    const size_t bytes = sizeof(float) * (size_t)nodes_;
    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    LP_CUDA(cudaMemsetAsync(d_offsets_, 0, node_bytes + sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_sources_, 0, sizeof(int) * (size_t)edges_, stream));
    LP_CUDA(cudaMemsetAsync(d_out_degrees_, 0, node_bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_nodes_, 0, node_bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_scores_, 0xFF, bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_contrib_, 0xFF, 2 * bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_barrier_, 0x40, sizeof(unsigned), stream));
#if LP_STAGE >= 4
    LP_CUDA(cudaMemsetAsync(d_nodes_ready_, 0xFF, sizeof(unsigned), stream));
#endif
    std::memset(h_scores_, 0xFF, bytes);
  }

  void compute_reference() override {
    expected_ = pagerank_oracle::gauss_seidel_scores(nodes_, iterations_);
  }

  bool check_outputs() override {
    long long mismatches = 0, identical = 0;
    double worst = 0.0;
    for (int n = 0; n < nodes_; ++n) {
      const float e = expected_[n], a = h_scores_[n];
      const bool ok = std::isfinite(a) && std::isfinite(e) && e > 0.0f &&
                      std::fabs(a - e) <= kRelativeTolerance * e;
      if (std::memcmp(&a, &e, sizeof a) == 0) ++identical;
      if (std::isfinite(a) && e > 0.0f)
        worst = std::max(worst, std::fabs((double)a - (double)e) / (double)e);
      if (!ok) {
        if (mismatches < 8)
          std::fprintf(stderr, "%s: mismatch at node %d: expected %.9g, actual %.9g\n",
                       LP_APP, n, (double)e, (double)a);
        ++mismatches;
      }
    }
    std::fprintf(stderr,
                 "%s: %lld of %d scores outside |a-e| <= 1e-5*e; %lld bitwise "
                 "identical to the oracle; max relative error %.3g\n",
                 LP_APP, mismatches, nodes_, identical, worst);
    return mismatches == 0;
  }

  uint64_t output_digest() const override {
    return lp::fnv1a64(h_scores_, sizeof(float) * (size_t)nodes_);
  }

private:
  int nodes_, iterations_;
  long long expected_edges_;
  int edges_ = 0, levels_ = 0;
  int blocks_ = 1, threads_ = 1;
  const void *kernel_ = nullptr;
#if LP_STAGE == 1
  LevelBounds bounds_{};
#endif
  std::vector<float> expected_;
  int *h_offsets_ = nullptr, *h_sources_ = nullptr, *h_out_degrees_ = nullptr,
      *h_nodes_ = nullptr;
  float *h_scores_ = nullptr;
  int *d_offsets_ = nullptr, *d_sources_ = nullptr, *d_out_degrees_ = nullptr,
      *d_nodes_ = nullptr;
  float *d_scores_ = nullptr;
  unsigned *d_contrib_ = nullptr, *d_barrier_ = nullptr;
#if LP_STAGE >= 4
  unsigned *d_nodes_ready_ = nullptr, *h_ready_ = nullptr;
  cudaStream_t side_ = nullptr;
  cudaEvent_t fork_ = nullptr, join_ = nullptr;
#endif
};

std::unique_ptr<lp::Workload> make(lp::Options &options) {
  if (options.preset != "large") return nullptr;
  const long long nodes = options.take_int("nodes", kDefaultNodes);
  const long long iterations = options.take_int("iterations", kDefaultIterations);
  if (nodes < 3 || nodes > kMaxNodes || (nodes & (nodes - 1)) != 0)
    lp::fail(lp::kExitUsage,
             "nodes must be a power of two from 4 to %lld (upstream requires a "
             "power of two of at least 3), got %lld",
             kMaxNodes, nodes);
  if (iterations < 1 || iterations > kMaxIterations)
    lp::fail(lp::kExitUsage, "iterations must be from 1 to %d, got %lld",
             kMaxIterations, iterations);
  return std::make_unique<Pagerank>((int)nodes, (int)iterations);
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make); }
