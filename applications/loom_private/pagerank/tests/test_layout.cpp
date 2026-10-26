// Host-only tests for the optimized variant's level-ordered layout
// (layout.cpp). Built and run by `make test VARIANT=optimized`.
#include "../graph.h"
#include "../layout.h"
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

// The layout describes the same graph: every position's sources decode, in
// list order, to the node's incoming list with the right current/previous
// bit, out-degrees match, and the arrays have the baseline's sizes. Every
// current-sweep source sits at an earlier position, which is what lets each
// GPU thread visit its positions in increasing order without deadlock.
static void test_layout(int nodes) {
  const pagerank::HostGraph g = pagerank::build_graph(nodes);
  const pagerank::LevelLayout l = pagerank::build_level_layout(g);
  EXPECT(l.offsets.size() == g.in_offsets.size() && l.sources.size() == g.in_sources.size() &&
             l.out_degrees.size() == g.out_degrees.size() &&
             l.nodes.size() == g.level_nodes.size(),
         "nodes %d: layout array sizes differ from the baseline's", nodes);
  std::vector<int> position(nodes, -1);
  bool permutation = true;
  for (int p = 0; p < nodes; ++p) {
    const int u = l.nodes[p];
    if (u < 0 || u >= nodes || position[u] != -1) permutation = false;
    else position[u] = p;
  }
  EXPECT(permutation, "nodes %d: positions are not a permutation of the nodes", nodes);
  if (!permutation) return;
  bool same = l.offsets[0] == 0, earlier = true;
  for (int p = 0; p < nodes && same; ++p) {
    const int u = l.nodes[p];
    const int begin = g.in_offsets[u], degree = g.in_offsets[u + 1] - begin;
    if (l.offsets[p + 1] - l.offsets[p] != degree || l.out_degrees[p] != g.out_degrees[u]) {
      same = false;
      break;
    }
    for (int i = 0; i < degree; ++i) {
      const int s = l.sources[l.offsets[p] + i], v = g.in_sources[begin + i];
      if (s >> 1 != position[v] || (s & 1) != (v < u ? 1 : 0)) same = false;
      if ((s & 1) && s >> 1 >= p) earlier = false;
    }
  }
  EXPECT(same, "nodes %d: layout does not describe the incoming CSR", nodes);
  EXPECT(earlier, "nodes %d: a current-sweep source is not at an earlier position", nodes);
}

// The GPU's computation in position order, with the two contribution buffers
// indexed by position. Matching the oracle bitwise checks the layout and the
// current/previous rule together.
static std::vector<float> layout_scores(int nodes, int iterations) {
  const pagerank::LevelLayout l = pagerank::build_level_layout(pagerank::build_graph(nodes));
  const float damping = 0.85f, initial = 1.0f / nodes, base = (1.0f - damping) / nodes;
  std::vector<float> score(nodes, std::nanf("")), buffer[2];
  buffer[0].assign(nodes, std::nanf(""));
  buffer[1].resize(nodes);
  for (int p = 0; p < nodes; ++p) buffer[1][p] = initial / (float)l.out_degrees[p];
  for (int sweep = 0; sweep < iterations; ++sweep) {
    std::vector<float> &current = buffer[sweep % 2];
    const std::vector<float> &previous = buffer[(sweep + 1) % 2];
    for (int p = 0; p < nodes; ++p) {
      float total = 0.0f;
      for (int e = l.offsets[p]; e < l.offsets[p + 1]; ++e) {
        const int s = l.sources[e];
        total += (s & 1) ? current[s >> 1] : previous[s >> 1];
      }
      score[l.nodes[p]] = base + damping * total;
      current[p] = score[l.nodes[p]] / (float)l.out_degrees[p];
    }
  }
  return score;
}

int main() {
  for (int nodes : {4, 8, 1024, 1 << 16, 1 << 20}) test_layout(nodes);
  struct Case { int nodes, iterations; };
  for (Case c : {Case{16, 3}, Case{4096, 6}, Case{8192, 20}}) {
    const std::vector<float> oracle = pagerank_oracle::gauss_seidel_scores(c.nodes, c.iterations);
    const std::vector<float> layout = layout_scores(c.nodes, c.iterations);
    EXPECT(std::memcmp(oracle.data(), layout.data(), sizeof(float) * c.nodes) == 0,
           "nodes %d iterations %d: layout computation differs bitwise from the oracle",
           c.nodes, c.iterations);
  }
  std::printf(failures ? "layout tests FAILED (%d)\n" : "layout tests passed\n", failures);
  return failures ? 1 : 0;
}
