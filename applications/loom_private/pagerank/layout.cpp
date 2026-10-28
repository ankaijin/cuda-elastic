#include "layout.h"

namespace pagerank {

LevelLayout build_level_layout(const HostGraph &g) {
  const int nodes = g.nodes;
  LevelLayout l;
  l.nodes = g.level_nodes;
  std::vector<int> position(nodes);
  for (int p = 0; p < nodes; ++p) position[l.nodes[p]] = p;

  l.offsets.resize(nodes + 1);
  l.out_degrees.resize(nodes);
  l.sources.resize(g.in_sources.size());
  int edge = 0;
  for (int p = 0; p < nodes; ++p) {
    const int u = l.nodes[p];
    l.offsets[p] = edge;
    l.out_degrees[p] = g.out_degrees[u];
    for (int e = g.in_offsets[u]; e < g.in_offsets[u + 1]; ++e) {
      const int v = g.in_sources[e];
      l.sources[edge++] = 2 * position[v] + (v < u ? 1 : 0);
    }
  }
  l.offsets[nodes] = edge;
  return l;
}

} // namespace pagerank
