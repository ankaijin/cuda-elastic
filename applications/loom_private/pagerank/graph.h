// Host-side graph preparation for the loom-private/pagerank workload: the
// generated graph as an incoming CSR plus out-degrees, and the dependency-level
// schedule the GPU uses to run ordered Gauss-Seidel sweeps. Nothing here is
// timed; the arrays it produces are uploaded inside the application scope.
#pragma once

#include <cstdint>
#include <vector>

namespace pagerank {

// Out-degree of node n is 1 + (n mod 3); out-neighbour i of n is
// (7n + 13i + 1) mod N.
constexpr int kMaxOutDegree = 3;
constexpr long long kSourceStride = 7;
constexpr long long kNeighborStride = 13;

// Edge count the generator must produce for `nodes` vertices:
// N + sum_{n<N} (n mod 3).
long long expected_edge_count(long long nodes);

struct HostGraph {
  int nodes = 0;
  long long edges = 0;
  // Incoming CSR: the sources of node u are in_sources[in_offsets[u] ..
  // in_offsets[u+1]) in ascending source order (stable in generation order).
  std::vector<int> in_offsets;
  std::vector<int> in_sources;
  std::vector<int> out_degrees;
  // Gauss-Seidel dependency levels. Node u depends, within one sweep, on its
  // in-neighbours v < u; level(u) = 0 when it has none, else 1 + the largest
  // level among them. level_nodes lists the nodes grouped by level (ascending
  // node order within a level); level k holds level_nodes[level_offsets[k] ..
  // level_offsets[k+1]).
  std::vector<int> level_offsets;
  std::vector<int> level_nodes;
  int levels() const { return (int)level_offsets.size() - 1; }
};

// Builds the generated graph and its level schedule. `nodes` must already be
// validated (a power of two, at least 4).
HostGraph build_graph(int nodes);

} // namespace pagerank
