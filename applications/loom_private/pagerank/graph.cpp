#include "graph.h"

#include <algorithm>

namespace pagerank {

long long expected_edge_count(long long nodes) {
  // Each full cycle of three nodes has out-degrees 1, 2, 3; a remainder of
  // two nodes adds degrees 1 and 2, a remainder of one adds degree 1.
  const long long cycles = nodes / 3, rest = nodes % 3;
  return cycles * 6 + (rest == 2 ? 3 : rest == 1 ? 1 : 0);
}

HostGraph build_graph(int nodes) {
  HostGraph g;
  g.nodes = nodes;
  const long long n_nodes = nodes;

  g.out_degrees.resize(nodes);
  std::vector<long long> out_offsets(n_nodes + 1);
  for (long long n = 0; n < n_nodes; ++n) {
    g.out_degrees[n] = 1 + (int)(n % kMaxOutDegree);
    out_offsets[n + 1] = out_offsets[n] + g.out_degrees[n];
  }
  g.edges = out_offsets[n_nodes];
  std::vector<int> out_targets(g.edges);
  for (long long n = 0; n < n_nodes; ++n)
    for (int i = 0; i < g.out_degrees[n]; ++i)
      out_targets[out_offsets[n] + i] =
          (int)((n * kSourceStride + i * kNeighborStride + 1) % n_nodes);

  // Stable counting sort of the out-edges by target: sources visited in
  // ascending order, so every incoming list is in ascending source order.
  g.in_offsets.assign(n_nodes + 1, 0);
  for (long long e = 0; e < g.edges; ++e) ++g.in_offsets[out_targets[e] + 1];
  for (long long u = 0; u < n_nodes; ++u) g.in_offsets[u + 1] += g.in_offsets[u];
  std::vector<int> cursor(g.in_offsets.begin(), g.in_offsets.end() - 1);
  g.in_sources.resize(g.edges);
  for (long long n = 0; n < n_nodes; ++n)
    for (long long e = out_offsets[n]; e < out_offsets[n + 1]; ++e)
      g.in_sources[cursor[out_targets[e]]++] = (int)n;

  // Dependency levels in node order: every in-neighbour v < u already has
  // its level when u is visited.
  std::vector<int> level(nodes, 0);
  int max_level = 0;
  for (int u = 0; u < nodes; ++u) {
    int l = 0;
    for (int e = g.in_offsets[u]; e < g.in_offsets[u + 1]; ++e) {
      const int v = g.in_sources[e];
      if (v < u) l = std::max(l, level[v] + 1);
    }
    level[u] = l;
    max_level = std::max(max_level, l);
  }
  g.level_offsets.assign(max_level + 2, 0);
  for (int u = 0; u < nodes; ++u) ++g.level_offsets[level[u] + 1];
  for (int l = 0; l <= max_level; ++l) g.level_offsets[l + 1] += g.level_offsets[l];
  std::vector<int> fill(g.level_offsets.begin(), g.level_offsets.end() - 1);
  g.level_nodes.resize(nodes);
  for (int u = 0; u < nodes; ++u) g.level_nodes[fill[level[u]]++] = u;
  return g;
}

} // namespace pagerank
