// loom-private/pagerank: ordered Gauss-Seidel PageRank (GAPBS PageRankPullGS
// as wrapped by private Loom's gapbs-pagerank harness) on the generated graph.
//
// One sweep visits nodes 0..N-1 in order. Node u sums the contributions of
// its in-neighbours in list order, sets s_u = base + 0.85 * sum and
// immediately c_u = s_u / outdeg(u), so later nodes in the same sweep see the
// new c_u while in-neighbours v >= u still contribute their previous-sweep
// value.
//
// The GPU keeps that order exactly with level scheduling. Within a sweep, u
// depends only on its in-neighbours v < u, so nodes are grouped by dependency
// level on the host (graph.cpp) and each level is one kernel launch. Two
// contribution buffers hold the previous and the current sweep: reads of
// v < u come from the current one (already written by an earlier level) and
// reads of v >= u from the previous one. Each node's sum runs over its
// in-neighbours in order in a single thread, in float, so the result matches
// the sequential loop bitwise.
#include "lp_harness.cuh"

#include "graph.h"
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
constexpr int kBlock = 256;

// Scores 1/N and contributions (1/N) / outdeg for every node: the state at
// the start of the first sweep, written into the "previous" buffer.
__global__ void initialize(const int *out_degrees, float *scores,
                           float *previous, int nodes, float initial) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= nodes) return;
  scores[n] = initial;
  previous[n] = initial / (float)out_degrees[n];
}

// Updates every node of one dependency level. All in-neighbours v < u sit in
// earlier levels, so current[v] already holds this sweep's contribution.
__global__ void update_level(const int *in_offsets, const int *in_sources,
                             const int *out_degrees, const int *level_nodes,
                             int level_begin, int level_size,
                             const float *previous, float *current,
                             float *scores, float base, float damping) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= level_size) return;
  const int u = level_nodes[level_begin + t];
  float total = 0.0f;
  for (int e = in_offsets[u]; e < in_offsets[u + 1]; ++e) {
    const int v = in_sources[e];
#if defined(LP_PAGERANK_MUTANT_JACOBI)
    // Test-only mutant (tests/): a Jacobi sweep that ignores this sweep's
    // updates. The tests require it to fail the oracle.
    total += previous[v];
#else
    total += v < u ? current[v] : previous[v];
#endif
  }
  const float score = base + damping * total;
  scores[u] = score;
  current[u] = score / (float)out_degrees[u];
}

class Pagerank final : public lp::Workload {
public:
  Pagerank(int nodes, int iterations)
      : nodes_(nodes), iterations_(iterations),
        expected_edges_(pagerank::expected_edge_count(nodes)) {}
  ~Pagerank() override {
    cudaFreeHost(h_in_offsets_);
    cudaFreeHost(h_in_sources_);
    cudaFreeHost(h_out_degrees_);
    cudaFreeHost(h_level_nodes_);
    cudaFreeHost(h_scores_);
    cudaFree(d_in_offsets_);
    cudaFree(d_in_sources_);
    cudaFree(d_out_degrees_);
    cudaFree(d_level_nodes_);
    cudaFree(d_scores_);
    cudaFree(d_contrib_[0]);
    cudaFree(d_contrib_[1]);
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
    level_offsets_ = g.level_offsets;
    std::fprintf(stderr, "%s: %d nodes, %d edges, %d dependency levels\n",
                 LP_APP, nodes_, edges_, levels_);

    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    const size_t edge_bytes = sizeof(int) * (size_t)edges_;
    LP_CUDA(cudaMallocHost(&h_in_offsets_, node_bytes + sizeof(int)));
    LP_CUDA(cudaMallocHost(&h_in_sources_, edge_bytes));
    LP_CUDA(cudaMallocHost(&h_out_degrees_, node_bytes));
    LP_CUDA(cudaMallocHost(&h_level_nodes_, node_bytes));
    LP_CUDA(cudaMallocHost(&h_scores_, sizeof(float) * (size_t)nodes_));
    std::copy(g.in_offsets.begin(), g.in_offsets.end(), h_in_offsets_);
    std::copy(g.in_sources.begin(), g.in_sources.end(), h_in_sources_);
    std::copy(g.out_degrees.begin(), g.out_degrees.end(), h_out_degrees_);
    std::copy(g.level_nodes.begin(), g.level_nodes.end(), h_level_nodes_);

    LP_CUDA(cudaMalloc(&d_in_offsets_, node_bytes + sizeof(int)));
    LP_CUDA(cudaMalloc(&d_in_sources_, edge_bytes));
    LP_CUDA(cudaMalloc(&d_out_degrees_, node_bytes));
    LP_CUDA(cudaMalloc(&d_level_nodes_, node_bytes));
    LP_CUDA(cudaMalloc(&d_scores_, sizeof(float) * (size_t)nodes_));
    LP_CUDA(cudaMalloc(&d_contrib_[0], sizeof(float) * (size_t)nodes_));
    LP_CUDA(cudaMalloc(&d_contrib_[1], sizeof(float) * (size_t)nodes_));
  }

  void enqueue_application(cudaStream_t stream) override {
    const size_t node_bytes = sizeof(int) * (size_t)nodes_;
    LP_CUDA(cudaMemcpyAsync(d_in_offsets_, h_in_offsets_, node_bytes + sizeof(int),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_in_sources_, h_in_sources_, sizeof(int) * (size_t)edges_,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_out_degrees_, h_out_degrees_, node_bytes,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_level_nodes_, h_level_nodes_, node_bytes,
                            cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_scores_, d_scores_, sizeof(float) * (size_t)nodes_,
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    // Float constants exactly as the sequential definition forms them.
    const float initial = 1.0f / (float)nodes_;
    const float base = (1.0f - kDamping) / (float)nodes_;
    // Sweep k writes buffer k % 2 and reads the previous sweep from the other
    // one; initialization fills the buffer sweep 0 treats as previous.
    initialize<<<(nodes_ + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        d_out_degrees_, d_scores_, d_contrib_[1], nodes_, initial);
    LP_CUDA_LAUNCH();
    for (int sweep = 0; sweep < iterations_; ++sweep) {
      float *current = d_contrib_[sweep % 2];
      const float *previous = d_contrib_[(sweep + 1) % 2];
      for (int level = 0; level < levels_; ++level) {
        const int begin = level_offsets_[level];
        const int size = level_offsets_[level + 1] - begin;
        update_level<<<(size + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
            d_in_offsets_, d_in_sources_, d_out_degrees_, d_level_nodes_, begin,
            size, previous, current, d_scores_, base, kDamping);
        LP_CUDA_LAUNCH();
      }
    }
  }

  void poison_outputs(cudaStream_t stream) override {
    // 0xFF bytes are a NaN float: any score or contribution left unwritten
    // fails the finiteness check or propagates into one.
    const size_t bytes = sizeof(float) * (size_t)nodes_;
    LP_CUDA(cudaMemsetAsync(d_scores_, 0xFF, bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_contrib_[0], 0xFF, bytes, stream));
    LP_CUDA(cudaMemsetAsync(d_contrib_[1], 0xFF, bytes, stream));
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
  std::vector<int> level_offsets_;
  std::vector<float> expected_;
  int *h_in_offsets_ = nullptr, *h_in_sources_ = nullptr,
      *h_out_degrees_ = nullptr, *h_level_nodes_ = nullptr;
  float *h_scores_ = nullptr;
  int *d_in_offsets_ = nullptr, *d_in_sources_ = nullptr,
      *d_out_degrees_ = nullptr, *d_level_nodes_ = nullptr;
  float *d_scores_ = nullptr;
  float *d_contrib_[2] = {nullptr, nullptr};
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
