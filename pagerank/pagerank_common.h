// Host-side code shared by the PageRank harnesses (gapbs_pagerank_cuda_graph.cu
// and pagerank_sota.cu): the workload constants, the benchmark and validation
// graphs, the double-precision CPU oracle, the score comparison, host-buffer
// pinning, NVTX ranges, and the environment options that select the
// benchmark input, rotate between inputs and dump scores.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define PAGERANK_HAS_NVTX 1
#  endif
#endif
#ifndef PAGERANK_HAS_NVTX
#  define PAGERANK_HAS_NVTX 0
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err__)); \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

constexpr int kVertices = 1 << 20;
constexpr int kEdgesPerVertex = 16;
constexpr int kEdges = kVertices * kEdgesPerVertex;
constexpr int kIterations = 20;
constexpr float kDamping = 0.85f;

// A score passes when |gpu - reference| <= abs + rel * |reference|.  Scores
// are O(1/V), so the absolute floor is scaled by 1/V.
constexpr double kRelTolerance = 1e-4;
constexpr double kAbsToleranceScale = 1e-6;

struct SplitMix64 {
  uint64_t state;
  uint64_t next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
};

// CSR over incoming edges: row_offsets/columns are indexed by destination and
// columns holds each edge's source vertex.
struct HostGraph {
  int vertices = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<int> out_degrees;
};

// Generated regular CSR fixture: every destination has sixteen incoming
// edges.  17 is odd, so each edge slot maps destinations to sources one to
// one and every vertex also has exactly sixteen outgoing edges.  That makes
// the uniform start the stationary distribution, so a uniform-start run
// cannot tell a correct update from one that reads the wrong edges.
inline HostGraph make_benchmark_fixture() {
  HostGraph graph;
  graph.vertices = kVertices;
  graph.row_offsets.resize(kVertices + 1);
  graph.columns.resize(kEdges);
  graph.out_degrees.assign(kVertices, kEdgesPerVertex);
  for (int vertex = 0; vertex <= kVertices; ++vertex)
    graph.row_offsets[vertex] = vertex * kEdgesPerVertex;
  for (int dst = 0; dst < kVertices; ++dst)
    for (int edge = 0; edge < kEdgesPerVertex; ++edge)
      graph.columns[dst * kEdgesPerVertex + edge] =
          (dst * 17 + edge * 104729) & (kVertices - 1);
  return graph;
}

// Small irregular graph for validation: skewed in-degrees including a few
// heavy rows, vertices with no incoming edges, dangling vertices with no
// outgoing edges, and a vertex count that is not a multiple of any block
// size.  Duplicate edges and self loops are allowed; both count towards
// degree like any other edge.
inline HostGraph make_irregular_graph(int vertices, uint64_t seed) {
  HostGraph graph;
  graph.vertices = vertices;
  graph.row_offsets.assign(vertices + 1, 0);
  SplitMix64 rng{seed};
  for (int dst = 0; dst < vertices; ++dst) {
    int degree;
    if (dst % 11 == 0)
      degree = 0;
    else if (dst % 997 == 1)
      degree = 500 + static_cast<int>(rng.next() % 2500);
    else if (rng.next() % 32 == 0)
      degree = 32 + static_cast<int>(rng.next() % 200);
    else
      degree = static_cast<int>(rng.next() % 24);
    graph.row_offsets[dst + 1] = graph.row_offsets[dst] + degree;
  }
  graph.columns.resize(graph.row_offsets[vertices]);
  graph.out_degrees.assign(vertices, 0);
  for (int& source : graph.columns) {
    // Vertices with source % 7 == 3 are never sources, so they dangle.
    source = static_cast<int>(rng.next() % vertices);
    if (source % 7 == 3) source = (source + 1) % vertices;
    ++graph.out_degrees[source];
  }
  return graph;
}

// Random graph with the benchmark's shape (2^20 vertices, exactly 2^24
// edges) for rotating between distinct inputs: destinations come in pairs
// whose in-degrees are 16 + k and 16 - k for a random k in [-8, 8], and every
// source is uniformly random, so out-degrees vary too.  Unlike the fixture,
// its uniform start is not stationary, so the result depends on every array.
inline HostGraph make_random_graph(uint64_t seed) {
  HostGraph graph;
  graph.vertices = kVertices;
  graph.row_offsets.assign(kVertices + 1, 0);
  SplitMix64 rng{seed};
  for (int dst = 0; dst < kVertices; dst += 2) {
    const int skew = static_cast<int>(rng.next() % 17) - 8;
    graph.row_offsets[dst + 1] = graph.row_offsets[dst] + kEdgesPerVertex + skew;
    graph.row_offsets[dst + 2] = graph.row_offsets[dst + 1] + kEdgesPerVertex - skew;
  }
  graph.columns.resize(kEdges);
  graph.out_degrees.assign(kVertices, 0);
  for (int& source : graph.columns) {
    source = static_cast<int>(rng.next() & (kVertices - 1));
    ++graph.out_degrees[source];
  }
  return graph;
}

// Deterministic, uneven starting scores normalized to sum to one.
inline std::vector<float> make_skewed_scores(int vertices, uint64_t seed) {
  SplitMix64 rng{seed};
  std::vector<double> weights(vertices);
  double total = 0.0;
  for (double& weight : weights) {
    weight = 1.0 + static_cast<double>(rng.next() % 1000);
    total += weight;
  }
  std::vector<float> scores(vertices);
  for (int v = 0; v < vertices; ++v)
    scores[v] = static_cast<float>(weights[v] / total);
  return scores;
}

// Returns an empty string when the CSR arrays are well formed and every
// stored out-degree matches the number of edges that vertex is a source of.
inline std::string check_graph(const HostGraph& graph) {
  const int vertices = graph.vertices;
  if (graph.row_offsets.size() != static_cast<size_t>(vertices) + 1 ||
      graph.out_degrees.size() != static_cast<size_t>(vertices))
    return "array sizes do not match the vertex count";
  if (graph.row_offsets[0] != 0 ||
      graph.row_offsets[vertices] != static_cast<int>(graph.columns.size()))
    return "row offsets do not span the column array";
  for (int v = 0; v < vertices; ++v)
    if (graph.row_offsets[v + 1] < graph.row_offsets[v])
      return "row offsets decrease at vertex " + std::to_string(v);
  std::vector<int> counted(vertices, 0);
  for (int source : graph.columns) {
    if (source < 0 || source >= vertices)
      return "column " + std::to_string(source) + " is out of range";
    ++counted[source];
  }
  for (int v = 0; v < vertices; ++v)
    if (counted[v] != graph.out_degrees[v])
      return "out_degrees[" + std::to_string(v) + "] is " +
             std::to_string(graph.out_degrees[v]) + " but the vertex is the source of " +
             std::to_string(counted[v]) + " edges";
  return {};
}

// Plain double-precision Jacobi PageRank with the same semantics as
// pagerank_update.  As in GAPBS, rank held by dangling vertices is not
// redistributed.
inline std::vector<double> reference_pagerank(const HostGraph& graph,
                                              const std::vector<float>& initial,
                                              int iterations) {
  const int vertices = graph.vertices;
  std::vector<double> previous(initial.begin(), initial.end());
  std::vector<double> next(vertices);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (int dst = 0; dst < vertices; ++dst) {
      double incoming = 0.0;
      for (int edge = graph.row_offsets[dst]; edge < graph.row_offsets[dst + 1]; ++edge) {
        const int source = graph.columns[edge];
        incoming += previous[source] / graph.out_degrees[source];
      }
      next[dst] = (1.0 - kDamping) / vertices + kDamping * incoming;
    }
    previous.swap(next);
  }
  return previous;
}

// Compares GPU scores with the CPU reference and prints one PASS/FAIL line
// (plus details on failure).  The score sum is a secondary check.  "spread"
// is (max - min) / mean of the expected scores; near zero means the case
// cannot detect wrong edges.
inline bool compare_scores(const char* name, const HostGraph& graph, int iterations,
                           const std::vector<float>& actual,
                           const std::vector<double>& expected) {
  const int vertices = graph.vertices;
  const double abs_tolerance = kAbsToleranceScale / vertices;
  double max_abs = 0.0, max_rel = 0.0, sum_actual = 0.0, sum_expected = 0.0;
  double min_expected = expected[0], max_expected = expected[0];
  int failures = 0, first_failure = -1;
  for (int v = 0; v < vertices; ++v) {
    const double want = expected[v];
    const double error = std::fabs(static_cast<double>(actual[v]) - want);
    // Written as !(<=) so NaN scores fail.
    if (!(error <= abs_tolerance + kRelTolerance * std::fabs(want)) && failures++ == 0)
      first_failure = v;
    max_abs = std::max(max_abs, error);
    if (want != 0.0) max_rel = std::max(max_rel, error / std::fabs(want));
    sum_actual += actual[v];
    sum_expected += want;
    min_expected = std::min(min_expected, want);
    max_expected = std::max(max_expected, want);
  }
  const bool sum_ok =
      std::fabs(sum_actual - sum_expected) <= kRelTolerance * std::fabs(sum_expected);
  const bool pass = failures == 0 && sum_ok;
  const double spread = (max_expected - min_expected) / (sum_expected / vertices);

  std::printf("%s  %-34s V=%-8d E=%-9zu iters=%-2d max_abs=%.2e max_rel=%.2e spread=%.3g\n",
              pass ? "PASS" : "FAIL", name, vertices, graph.columns.size(), iterations,
              max_abs, max_rel, spread);
  if (failures > 0)
    std::printf("      %d of %d scores outside tolerance; first at vertex %d: got %.9g, expected %.9g\n",
                failures, vertices, first_failure, actual[first_failure],
                expected[first_failure]);
  if (!sum_ok)
    std::printf("      score sum %.9g, expected %.9g\n", sum_actual, sum_expected);
  return pass;
}

// Runs one validation case on the GPU: (name, graph, initial scores or null
// for a uniform start, iterations) -> pass.  Each harness supplies its own.
using ValidateCase = std::function<bool(const char*, const HostGraph&,
                                        const std::vector<float>*, int)>;

// Malformed graphs fail before anything runs on the GPU.
inline bool check_case_graph(const char* name, const HostGraph& graph) {
  const std::string graph_error = check_graph(graph);
  if (graph_error.empty()) return true;
  std::printf("FAIL %s: malformed graph: %s\n", name, graph_error.c_str());
  return false;
}

// The six validation cases every harness runs, against the CPU reference.
inline bool run_validation_cases(const HostGraph& fixture, const ValidateCase& validate_case) {
  const HostGraph irregular = make_irregular_graph(100003, 0x5eed);
  const std::vector<float> fixture_skewed = make_skewed_scores(fixture.vertices, 1);
  const std::vector<float> irregular_skewed = make_skewed_scores(irregular.vertices, 2);

  bool ok = true;
  // The benchmark graph itself.  Its uniform start is already the fixture's
  // stationary distribution, so this checks plumbing, not the update.
  ok &= validate_case("fixture, uniform start (benchmark)", fixture, nullptr, kIterations);
  // Skewed starts make every edge matter, but the fixture mixes quickly: the
  // spread of a skewed start falls below kRelTolerance after about six
  // iterations, so longer fixture runs are no stronger than the uniform one.
  // One and two iterations check a single update and a chained update that
  // reads the contributions the first one wrote.
  ok &= validate_case("fixture, skewed start", fixture, &fixture_skewed, 1);
  ok &= validate_case("fixture, skewed start", fixture, &fixture_skewed, 2);
  // The irregular graph has a nonuniform stationary distribution, so its
  // full-length runs stay discriminating.
  ok &= validate_case("irregular, uniform start", irregular, nullptr, kIterations);
  ok &= validate_case("irregular, skewed start", irregular, &irregular_skewed, 1);
  ok &= validate_case("irregular, skewed start", irregular, &irregular_skewed, kIterations);
  return ok;
}

// Page-locks the host buffers that a PageRank run copies from and to.
// `mapped` also maps them into the device address space for zero-copy access.
// Returns the registered pointers for unpinning.
inline std::vector<void*> pin_host_buffers(bool pin_host, const HostGraph& graph,
                                           const std::vector<float>* initial,
                                           std::vector<float>& result, bool mapped = false) {
  std::vector<void*> pinned;
  if (!pin_host) return pinned;
  const auto pin = [&](const void* data, size_t bytes) {
    if (bytes == 0) return;
    void* pointer = const_cast<void*>(data);
    CUDA_CHECK(cudaHostRegister(pointer, bytes,
                                mapped ? cudaHostRegisterMapped : cudaHostRegisterDefault));
    pinned.push_back(pointer);
  };
  pin(graph.row_offsets.data(), graph.row_offsets.size() * sizeof(int));
  pin(graph.columns.data(), graph.columns.size() * sizeof(int));
  pin(graph.out_degrees.data(), graph.out_degrees.size() * sizeof(int));
  if (initial) pin(initial->data(), initial->size() * sizeof(float));
  pin(result.data(), result.size() * sizeof(float));
  return pinned;
}

inline void unpin_host_buffers(std::vector<void*>& pinned) {
  for (void* pointer : pinned) CUDA_CHECK(cudaHostUnregister(pointer));
  pinned.clear();
}

inline void nvtx_push(const char* label) {
#if PAGERANK_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

inline void nvtx_pop() {
#if PAGERANK_HAS_NVTX
  nvtxRangePop();
#endif
}

// One benchmark input: a graph, its starting scores (empty for a uniform
// start) and the pinned buffer its run writes the final scores into.
struct BenchInput {
  std::string label;
  HostGraph graph;
  std::vector<float> initial;
  std::vector<float> result;
  const std::vector<float>* initial_or_null() const { return initial.empty() ? nullptr : &initial; }
};

// Benchmark-input options, read from the environment:
//   PAGERANK_GRAPH=fixture|irregular|random:<seed>  the benchmark graph (fixture)
//   PAGERANK_START=uniform|skewed:<seed>            its starting scores (uniform)
//   PAGERANK_ROTATE=K   timed replays cycle through K distinct inputs: the
//                       benchmark input, then random graphs of the same shape
//   PAGERANK_DUMP=file  writes the benchmark input's final scores (float32)
//   PAGERANK_DUMP_REF=file  writes the CPU reference for it (float32)
struct InputOptions {
  std::string graph = "fixture";
  std::string start = "uniform";
  int rotate = 1;
  std::string dump, dump_ref;
};

// Parses "name" or "name:<seed>"; returns false when the name does not match.
inline bool parse_seeded(const std::string& value, const char* name, uint64_t& seed) {
  const std::string prefix = std::string(name) + ":";
  if (value == name) return true;
  if (value.compare(0, prefix.size(), prefix) != 0) return false;
  char* end = nullptr;
  seed = std::strtoull(value.c_str() + prefix.size(), &end, 0);
  return end && *end == '\0' && end != value.c_str() + prefix.size();
}

// Reads InputOptions from the environment; `default_rotate` is the
// harness's default rotation.  Exits with status 2 on a malformed value.
inline InputOptions read_input_options(int default_rotate) {
  InputOptions options;
  options.rotate = default_rotate;
  uint64_t seed = 0;
  if (const char* value = std::getenv("PAGERANK_GRAPH")) {
    options.graph = value;
    if (options.graph != "fixture" && options.graph != "irregular" &&
        !parse_seeded(options.graph, "random", seed)) {
      std::fprintf(stderr, "PAGERANK_GRAPH must be fixture, irregular or random[:seed] (got \"%s\")\n", value);
      std::exit(2);
    }
  }
  if (const char* value = std::getenv("PAGERANK_START")) {
    options.start = value;
    if (options.start != "uniform" && !parse_seeded(options.start, "skewed", seed)) {
      std::fprintf(stderr, "PAGERANK_START must be uniform or skewed[:seed] (got \"%s\")\n", value);
      std::exit(2);
    }
  }
  if (const char* value = std::getenv("PAGERANK_ROTATE")) {
    options.rotate = std::atoi(value);
    if (options.rotate < 1 || options.rotate > 16) {
      std::fprintf(stderr, "PAGERANK_ROTATE must be from 1 to 16 (got \"%s\")\n", value);
      std::exit(2);
    }
  }
  if (const char* value = std::getenv("PAGERANK_DUMP")) options.dump = value;
  if (const char* value = std::getenv("PAGERANK_DUMP_REF")) options.dump_ref = value;
  return options;
}

// Builds the K inputs that timed replays rotate through.  Input 0 is the
// benchmark input (PAGERANK_GRAPH / PAGERANK_START; by default the fixture
// with a uniform start, so K = 1 is exactly the original benchmark).  Inputs
// 1..K-1 are distinct random graphs of the same shape with uniform starts,
// so every replay moves the same bytes and does the same work.  (When input
// 0 is the small irregular graph, which no random graph matches in shape,
// inputs 1..K-1 are that graph with distinct skewed starts instead.)
inline std::vector<BenchInput> make_bench_inputs(const InputOptions& options,
                                                 const HostGraph& fixture) {
  std::vector<BenchInput> inputs(options.rotate);
  uint64_t seed = 1;
  BenchInput& first = inputs[0];
  if (options.graph == "fixture") {
    first.graph = fixture;
  } else if (options.graph == "irregular") {
    first.graph = make_irregular_graph(100003, 0x5eed);
  } else {
    seed = 1;
    parse_seeded(options.graph, "random", seed);
    first.graph = make_random_graph(seed);
  }
  first.label = options.graph + "/" + options.start;
  if (options.start != "uniform") {
    seed = 1;
    parse_seeded(options.start, "skewed", seed);
    first.initial = make_skewed_scores(first.graph.vertices, seed);
  }
  for (int k = 1; k < options.rotate; ++k) {
    if (options.graph == "irregular") {
      inputs[k].graph = first.graph;
      inputs[k].initial = make_skewed_scores(first.graph.vertices, 100 + k);
      inputs[k].label = "irregular/skewed:" + std::to_string(100 + k);
    } else {
      inputs[k].graph = make_random_graph(0x7a7e0000ull + k);
      inputs[k].label = "random:" + std::to_string(0x7a7e0000ull + k) + "/uniform";
    }
  }
  for (BenchInput& input : inputs) input.result.assign(input.graph.vertices, 0.0f);
  return inputs;
}

// Writes `count` floats to `path`; exits on failure.
inline void write_floats(const std::string& path, const float* data, size_t count) {
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file || std::fwrite(data, sizeof(float), count, file) != count || std::fclose(file) != 0) {
    std::fprintf(stderr, "could not write %s\n", path.c_str());
    std::exit(1);
  }
}

// After the timed replays: checks every rotated input's last result against
// its own CPU reference (so a run that reused data from an earlier replay
// fails), and writes the dumps.  Returns false when any input fails.
inline bool check_bench_inputs(const InputOptions& options,
                               const std::vector<BenchInput>& inputs, bool check) {
  bool ok = true;
  for (size_t k = 0; k < inputs.size(); ++k) {
    const BenchInput& input = inputs[k];
    const bool need_ref = check || (k == 0 && !options.dump_ref.empty());
    std::vector<double> expected;
    if (need_ref) {
      const std::vector<float> start =
          input.initial.empty() ? std::vector<float>(input.graph.vertices, 1.0f / input.graph.vertices)
                                : input.initial;
      expected = reference_pagerank(input.graph, start, kIterations);
    }
    if (check) {
      const std::string name = "rotated input " + std::to_string(k) + " " + input.label;
      ok &= compare_scores(name.c_str(), input.graph, kIterations, input.result, expected);
    }
    if (k == 0 && !options.dump.empty())
      write_floats(options.dump, input.result.data(), input.result.size());
    if (k == 0 && !options.dump_ref.empty()) {
      std::vector<float> reference(expected.begin(), expected.end());
      write_floats(options.dump_ref, reference.data(), reference.size());
    }
  }
  return ok;
}
