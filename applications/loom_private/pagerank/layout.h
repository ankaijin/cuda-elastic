// Level-ordered copy of the pagerank graph for the optimized variant
// (pagerank_optimized.cu). Built on the host at setup from graph.cpp's CSR and
// level schedule, untimed like them; the arrays it produces are uploaded
// inside the application scope instead of the baseline's.
#pragma once

#include "graph.h"

#include <vector>

namespace pagerank {

// Nodes renumbered to positions in level order: position p holds node
// nodes[p] (graph.cpp's level_nodes, so each level's nodes are contiguous and
// ascending). The incoming CSR, out-degrees and every source reference use
// positions, so one sweep streams the arrays in order.
struct LevelLayout {
  // Incoming edges of position p are sources[offsets[p] .. offsets[p+1]),
  // in the original list order (ascending source node).
  std::vector<int> offsets;
  // 2 * position(v) + (v < u): the low bit says whether the update of u reads
  // v's contribution from the current sweep (v < u) or the previous one.
  std::vector<int> sources;
  std::vector<int> out_degrees; // by position
  std::vector<int> nodes;       // node id of each position
};

// Same edge count and the same byte size per array as the baseline's upload
// (offsets N+1, sources E, out-degrees N, nodes N).
LevelLayout build_level_layout(const HostGraph &g);

} // namespace pagerank
