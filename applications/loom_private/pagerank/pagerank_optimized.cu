// loom-private/pagerank, optimized variant: the same ordered Gauss-Seidel
// PageRank contract as pagerank.cu, run as one persistent kernel per
// invocation instead of one launch per dependency level per sweep.
//
// Layout. Setup renumbers the nodes to positions in level order (layout.cpp):
// each level's nodes are contiguous, and the incoming CSR, out-degrees and
// source references use positions. Every source entry also carries whether u
// reads v from the current sweep (v < u) or the previous one.
//
// Order. A sweep keeps the sequential semantics with per-node readiness
// instead of per-level launches. Two contribution buffers hold the current
// and the previous sweep, as in the baseline: sweep k writes buffer k % 2 and
// reads the previous sweep from the other one. Every stored contribution
// carries a tag of its sweep in the sign bit (contributions are positive, so
// the magnitude is the exact value). Updating u, a read of v < u waits until
// v's slot in the current buffer carries this sweep's tag, i.e. until v's
// update of this sweep stored it; a read of v >= u takes the previous buffer.
// Sweeps are separated by a grid-wide barrier, so the previous buffer is
// complete and no value is overwritten while another node still has to read
// it. Each node's sum runs sequentially in list order in one thread, in
// float, with the baseline's operations, so the scores equal the sequential
// loop bitwise.
//
// Progress. Every thread visits its positions (first + i * stride) in
// increasing order, so in level order, and every wait is for a position of a
// lower level. The kernel is a cooperative launch, so all blocks are
// co-resident, and the earliest unfinished position can always complete.
#include "lp_harness.cuh"

#include "graph.h"
#include "layout.h"
#include "oracle.h"

#include <cmath>

namespace {

constexpr int kDefaultNodes = 1 << 20;
constexpr int kDefaultIterations = 20;
constexpr long long kLargeEdges = 2097151;
constexpr long long kMaxNodes = 1LL << 26;
constexpr int kMaxIterations = 1000;
constexpr float kDamping = 0.85f;
constexpr float kRelativeTolerance = 1.0e-5f;
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

// Grid-wide barrier over co-resident blocks; `epoch` counts the barriers this
// invocation has passed, and the counter is zeroed before each launch.
__device__ __forceinline__ void grid_barrier(unsigned *counter, unsigned epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned target = epoch * gridDim.x;
    __threadfence();
    atomicAdd(counter, 1u);
    unsigned seen;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                   : "=r"(seen)
                   : "l"(counter)
                   : "memory");
    } while (seen < target);
    __threadfence();
  }
  __syncthreads();
}

// Readiness tag of the contributions a sweep stores: the sign bit, set for
// sweeps 2, 3, 6, 7, ... Contributions are positive, so the stored value is
// +c or -c and |stored| == c exactly. A buffer is rewritten every second
// sweep and the tag alternates between those writes, so the value left from
// two sweeps earlier never carries the current sweep's tag.
__device__ __forceinline__ unsigned sweep_tag(int sweep) {
  return (unsigned)(sweep >> 1 & 1) << 31;
}

// Sources of one node loaded ahead of its waits; a longer incoming list (the
// generated graph has none) continues in a sequential loop.
constexpr int kAhead = 3;

// Read-only graph arrays, read once per sweep: with the evict-first
// (streaming) hint when kStream, else through the read-only cache.
template <bool kStream>
__device__ __forceinline__ int load_graph(const int *address) {
  return kStream ? __ldcs(address) : __ldg(address);
}

template <bool kStream>
__device__ __forceinline__ int source_entry(const int *sources, int e) {
#if defined(LP_PAGERANK_MUTANT_JACOBI)
  // Test-only mutant (tests/): a Jacobi sweep that ignores this sweep's
  // updates. The tests require it to fail the oracle.
  return load_graph<kStream>(sources + e) & ~1;
#else
  return load_graph<kStream>(sources + e);
#endif
}

// Issues the read of one source contribution: this sweep's value for v < u
// (which may not be stored yet), the previous sweep's for v >= u.
__device__ __forceinline__ unsigned load_source(int source, const unsigned *current,
                                                const unsigned *previous) {
  return (source & 1) ? load_relaxed(current + (source >> 1))
                      : __ldcg(previous + (source >> 1));
}

// A current-sweep value is used once it carries this sweep's tag; until then
// the update that produces it has not stored it, and the read is repeated.
__device__ __forceinline__ float ready_value(unsigned bits, int source,
                                             const unsigned *current, unsigned tag) {
  if (source & 1)
    while ((bits & kSign) != tag) bits = load_relaxed(current + (source >> 1));
  return __uint_as_float(bits & ~kSign);
}

// u's sum over its in-neighbours in list order, in float. The first kAhead
// contribution loads are all issued before any of them is waited on.
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

// Initialization and every sweep of one invocation. Sweep k writes buffer
// k % 2 and reads the previous sweep from the other one. Each thread visits
// its positions in increasing order, so it has stored every earlier position
// of its own before it waits for anything.
template <bool kStream>
__global__ void __launch_bounds__(1024)
gauss_seidel(const int *offsets, const int *sources, const int *out_degrees,
             const int *nodes, unsigned *contrib, float *scores, unsigned *barrier,
             int count, int iterations, float initial, float base, float damping) {
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  // The initial contributions, in buffer 1, carry the tag of sweep -1; buffer
  // 0 gets a value whose tag is not sweep 0's.
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
    for (int p = first; p < count; p += stride) {
      // The out-degree is loaded before the waits, not after them.
      const int degree = load_graph<kStream>(out_degrees + p);
      const float total =
          in_sum<kStream>(load_graph<kStream>(offsets + p),
                          load_graph<kStream>(offsets + p + 1), sources, current,
                          previous, tag);
      const float score = base + damping * total;
      store_relaxed(current + p, tag | __float_as_uint(score / (float)degree));
      // Only the last sweep's scores are output; earlier ones are overwritten
      // without being read, as in the sequential loop.
      if (last) scores[__ldg(nodes + p)] = score;
    }
    if (!last) grid_barrier(barrier, ++epoch);
  }
}

using Kernel = void (*)(const int *, const int *, const int *, const int *,
                       unsigned *, float *, unsigned *, int, int, float, float, float);

// Launch shape per GPU (see README): threads per block, resident blocks per
// SM of the persistent grid, and whether the graph arrays are streamed.
struct Tuning {
  const char *name;
  int threads, blocks_per_sm;
  bool stream_graph;
};

constexpr Tuning kTunings[] = {
    {"sm_86", 768, 2, true},   // RTX 3090
    {"sm_89", 512, 3, false},  // RTX 4090
    {"sm_120", 512, 2, false}, // RTX 5090
    {"default", 512, 2, false},
    // Test only: a grid of one block of 64 threads (blocks_per_sm 0), so most
    // waits are between lanes of a warp and every thread visits many
    // positions per sweep.
    {"tiny", 64, 0, false},
};

const Tuning &find_tuning(const std::string &name) {
  for (const Tuning &t : kTunings)
    if (name == t.name) return t;
  lp::fail(lp::kExitUsage, "unknown tuning %s", name.c_str());
}

const Tuning &select_tuning(const cudaDeviceProp &p) {
#if defined(LP_PAGERANK_TEST_TUNING)
  // Test builds (tests/) may force any entry, so every GPU's choice runs on
  // whichever GPU runs the tests.
  if (const char *forced = std::getenv("LP_PAGERANK_TUNING"))
    if (*forced) return find_tuning(forced);
#endif
  const std::string arch = "sm_" + std::to_string(p.major * 10 + p.minor);
  for (const Tuning &t : kTunings)
    if (arch == t.name) return t;
  return find_tuning("default");
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
    const pagerank::LevelLayout layout = pagerank::build_level_layout(g);

    int device = 0;
    cudaDeviceProp properties{};
    LP_CUDA(cudaGetDevice(&device));
    LP_CUDA(cudaGetDeviceProperties(&properties, device));
    const Tuning &tuning = select_tuning(properties);
    threads_ = tuning.threads;
    kernel_ = tuning.stream_graph ? gauss_seidel<true> : gauss_seidel<false>;
    int resident = 0;
    LP_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel_,
                                                          threads_, 0));
    if (resident < 1) lp::fail(lp::kExitCuda, "persistent kernel does not fit an SM");
    const long long wanted =
        tuning.blocks_per_sm == 0 ? 1
                                  : (long long)std::min(resident, tuning.blocks_per_sm) *
                                        properties.multiProcessorCount;
    const long long needed = (nodes_ + threads_ - 1) / threads_;
    blocks_ = (int)std::max(1LL, std::min(wanted, needed));
    std::fprintf(stderr,
                 "%s: %d nodes, %d edges, %d dependency levels; %s tuning: "
                 "%d blocks x %d threads%s\n",
                 LP_APP, nodes_, edges_, levels_, tuning.name, blocks_, threads_,
                 tuning.stream_graph ? ", streamed graph loads" : "");

    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    const size_t edge_bytes = sizeof(int) * (size_t)edges_;
    LP_CUDA(cudaMallocHost(&h_offsets_, node_bytes + sizeof(int)));
    LP_CUDA(cudaMallocHost(&h_sources_, edge_bytes));
    LP_CUDA(cudaMallocHost(&h_out_degrees_, node_bytes));
    LP_CUDA(cudaMallocHost(&h_nodes_, node_bytes));
    LP_CUDA(cudaMallocHost(&h_scores_, sizeof(float) * (size_t)nodes_));
    std::copy(layout.offsets.begin(), layout.offsets.end(), h_offsets_);
    std::copy(layout.sources.begin(), layout.sources.end(), h_sources_);
    std::copy(layout.out_degrees.begin(), layout.out_degrees.end(), h_out_degrees_);
    std::copy(layout.nodes.begin(), layout.nodes.end(), h_nodes_);

    LP_CUDA(cudaMalloc(&d_offsets_, node_bytes + sizeof(int)));
    LP_CUDA(cudaMalloc(&d_sources_, edge_bytes));
    LP_CUDA(cudaMalloc(&d_out_degrees_, node_bytes));
    LP_CUDA(cudaMalloc(&d_nodes_, node_bytes));
    LP_CUDA(cudaMalloc(&d_scores_, sizeof(float) * (size_t)nodes_));
    LP_CUDA(cudaMalloc(&d_contrib_, 2 * sizeof(unsigned) * (size_t)nodes_));
    LP_CUDA(cudaMalloc(&d_barrier_, sizeof(unsigned)));
  }

  void enqueue_application(cudaStream_t stream) override {
    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    LP_CUDA(cudaMemcpyAsync(d_offsets_, h_offsets_, node_bytes + sizeof(int),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_sources_, h_sources_, sizeof(int) * (size_t)edges_,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_out_degrees_, h_out_degrees_, node_bytes,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_nodes_, h_nodes_, node_bytes,
                            cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_scores_, d_scores_, sizeof(float) * (size_t)nodes_,
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    // Float constants exactly as the sequential definition forms them.
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
    LP_CUDA(cudaLaunchKernelEx(&config, kernel_, (const int *)d_offsets_,
                               (const int *)d_sources_, (const int *)d_out_degrees_,
                               (const int *)d_nodes_, d_contrib_, d_scores_,
                               d_barrier_, nodes_, iterations_, initial, base,
                               kDamping));
  }

  void poison_outputs(cudaStream_t stream) override {
    // 0xFF bytes are a NaN float: any score left unwritten fails the
    // finiteness check. The contribution buffers hold NaN, and the barrier
    // counter a value past every barrier, so the initialization and the
    // counter reset must run in each invocation.
    const size_t bytes = sizeof(float) * (size_t)nodes_;
    // The device inputs are zeroed: an edgeless graph with out-degree 0 whose
    // positions all name node 0. An invocation that skips their upload then
    // leaves every other score NaN, without any out-of-bounds index.
    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    LP_CUDA(cudaMemsetAsync(d_offsets_, 0, node_bytes + sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_sources_, 0, sizeof(int) * (size_t)edges_, stream));
    LP_CUDA(cudaMemsetAsync(d_out_degrees_, 0, node_bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_nodes_, 0, node_bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_scores_, 0xFF, bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_contrib_, 0xFF, 2 * bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_barrier_, 0xFF, sizeof(unsigned), stream));
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
      // Private Loom's bound: |a - e| <= 1e-5 * e (every score exceeds the
      // positive base score), evaluated in float; non-finite values fail.
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
  Kernel kernel_ = nullptr;
  std::vector<float> expected_;
  int *h_offsets_ = nullptr, *h_sources_ = nullptr, *h_out_degrees_ = nullptr,
      *h_nodes_ = nullptr;
  float *h_scores_ = nullptr;
  int *d_offsets_ = nullptr, *d_sources_ = nullptr, *d_out_degrees_ = nullptr,
      *d_nodes_ = nullptr;
  float *d_scores_ = nullptr;
  unsigned *d_contrib_ = nullptr, *d_barrier_ = nullptr;
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
