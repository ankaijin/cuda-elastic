// Host-only tests for loom-private/pagerank: graph generation, the level
// schedule, and the Gauss-Seidel versus Jacobi distinction the GPU tests rely
// on. Built and run by `make test`; never part of the large check.
#include "../graph.h"
#include "../oracle.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

#define EXPECT(condition, ...)                                                 \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                         \
      std::printf(__VA_ARGS__);                                                \
      std::printf("\n");                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// Jacobi sweeps (every read from the previous sweep), as the legacy
// cuda-elastic pagerank computes. Used only to show the tests can tell the
// two orders apart.
static std::vector<float> jacobi_scores(const pagerank::HostGraph &g, int iterations) {
  const int n_nodes = g.nodes;
  const float damping = 0.85f, initial = 1.0f / n_nodes,
              base = (1.0f - damping) / n_nodes;
  std::vector<float> score(n_nodes, initial), previous(n_nodes), next(n_nodes);
  for (int n = 0; n < n_nodes; ++n) previous[n] = initial / g.out_degrees[n];
  for (int sweep = 0; sweep < iterations; ++sweep) {
    for (int u = 0; u < n_nodes; ++u) {
      float total = 0.0f;
      for (int e = g.in_offsets[u]; e < g.in_offsets[u + 1]; ++e)
        total += previous[g.in_sources[e]];
      score[u] = base + damping * total;
      next[u] = score[u] / g.out_degrees[u];
    }
    previous.swap(next);
  }
  return score;
}

// Ordered Gauss-Seidel on graph.cpp's CSR, executed level by level with the
// same double-buffer rule as the GPU kernel. Matching the oracle bitwise
// checks both the CSR order and the schedule rule on the host.
static std::vector<float> leveled_scores(const pagerank::HostGraph &g, int iterations) {
  const int n_nodes = g.nodes;
  const float damping = 0.85f, initial = 1.0f / n_nodes,
              base = (1.0f - damping) / n_nodes;
  std::vector<float> score(n_nodes, initial), buffer[2];
  buffer[0].assign(n_nodes, std::nanf(""));
  buffer[1].resize(n_nodes);
  for (int n = 0; n < n_nodes; ++n) buffer[1][n] = initial / g.out_degrees[n];
  for (int sweep = 0; sweep < iterations; ++sweep) {
    std::vector<float> &current = buffer[sweep % 2];
    const std::vector<float> &previous = buffer[(sweep + 1) % 2];
    for (int level = 0; level < g.levels(); ++level)
      for (int t = g.level_offsets[level]; t < g.level_offsets[level + 1]; ++t) {
        const int u = g.level_nodes[t];
        float total = 0.0f;
        for (int e = g.in_offsets[u]; e < g.in_offsets[u + 1]; ++e) {
          const int v = g.in_sources[e];
          total += v < u ? current[v] : previous[v];
        }
        score[u] = base + damping * total;
        current[u] = score[u] / g.out_degrees[u];
      }
  }
  return score;
}

static bool within(float actual, float expected) {
  return std::isfinite(actual) && std::fabs(actual - expected) <= 1.0e-5f * expected;
}

static void test_graph(int nodes) {
  const pagerank::HostGraph g = pagerank::build_graph(nodes);
  EXPECT(g.edges == pagerank::expected_edge_count(nodes), "nodes %d: %lld edges, expected %lld",
         nodes, g.edges, pagerank::expected_edge_count(nodes));
  EXPECT((long long)g.in_sources.size() == g.edges && g.in_offsets[nodes] == g.edges,
         "nodes %d: CSR size", nodes);
  bool ascending = true, ordered = true;
  std::vector<int> level(nodes, -1);
  for (int l = 0; l < g.levels(); ++l)
    for (int t = g.level_offsets[l]; t < g.level_offsets[l + 1]; ++t) {
      if (level[g.level_nodes[t]] != -1) ordered = false;
      level[g.level_nodes[t]] = l;
    }
  for (int u = 0; u < nodes; ++u) {
    if (level[u] < 0) ordered = false;
    for (int e = g.in_offsets[u]; e < g.in_offsets[u + 1]; ++e) {
      if (e > g.in_offsets[u] && g.in_sources[e] < g.in_sources[e - 1]) ascending = false;
      const int v = g.in_sources[e];
      if (v < u && level[v] >= level[u]) ordered = false;
    }
  }
  EXPECT(ascending, "nodes %d: incoming lists not in ascending source order", nodes);
  EXPECT(ordered, "nodes %d: level schedule does not cover each node once after its "
         "in-neighbours v < u", nodes);
  std::printf("graph nodes=%d edges=%lld levels=%d\n", nodes, g.edges, g.levels());
}

int main() {
  EXPECT(pagerank::expected_edge_count(1 << 20) == 2097151, "large edge count");
  for (int nodes : {8, 1024, 1 << 16}) test_graph(nodes);
  {
    const pagerank::HostGraph large = pagerank::build_graph(1 << 20);
    EXPECT(large.edges == 2097151, "large graph has %lld edges", large.edges);
    EXPECT(large.levels() == 16, "large schedule has %d levels (README records 16)",
           large.levels());
  }

  // Small configurations distinct from every upstream smoke and validation
  // configuration (4/8, 2048/12, 2048/16).
  struct Case { int nodes, iterations; };
  for (Case c : {Case{16, 3}, Case{4096, 6}, Case{8192, 20}}) {
    const pagerank::HostGraph g = pagerank::build_graph(c.nodes);
    const std::vector<float> oracle = pagerank_oracle::gauss_seidel_scores(c.nodes, c.iterations);
    const std::vector<float> leveled = leveled_scores(g, c.iterations);
    const std::vector<float> jacobi = jacobi_scores(g, c.iterations);
    EXPECT(std::memcmp(oracle.data(), leveled.data(), sizeof(float) * c.nodes) == 0,
           "nodes %d iterations %d: leveled schedule differs bitwise from the oracle",
           c.nodes, c.iterations);
    int jacobi_outside = 0, first = -1;
    for (int n = 0; n < c.nodes; ++n)
      if (!within(jacobi[n], oracle[n])) {
        if (first < 0) first = n;
        ++jacobi_outside;
      }
    EXPECT(jacobi_outside > 0,
           "nodes %d iterations %d: Jacobi agrees with Gauss-Seidel within tolerance",
           c.nodes, c.iterations);
    if (first >= 0)
      std::printf("nodes=%d iterations=%d: Jacobi outside tolerance on %d nodes "
                  "(first %d: Gauss-Seidel %.9g, Jacobi %.9g)\n",
                  c.nodes, c.iterations, jacobi_outside, first, oracle[first], jacobi[first]);
  }

  std::printf(failures ? "host tests FAILED (%d)\n" : "host tests passed\n", failures);
  return failures ? 1 : 0;
}
