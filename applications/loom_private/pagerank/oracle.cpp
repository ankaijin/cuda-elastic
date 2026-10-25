#include "oracle.h"

namespace pagerank_oracle {

std::vector<float> gauss_seidel_scores(int nodes, int iterations) {
  // Per-node incoming lists, filled by walking sources in ascending order
  // (and each source's edges in generation order), which is the order the
  // sums must follow.
  std::vector<std::vector<int>> incoming(nodes);
  std::vector<int> degree(nodes);
  for (int source = 0; source < nodes; ++source) {
    degree[source] = 1 + source % 3;
    for (int i = 0; i < degree[source]; ++i) {
      const long long target = (7LL * source + 13LL * i + 1) % nodes;
      incoming[target].push_back(source);
    }
  }

  const float damping = 0.85f;
  const float initial = 1.0f / nodes;
  const float base = (1.0f - damping) / nodes;
  std::vector<float> score(nodes), contribution(nodes);
  for (int n = 0; n < nodes; ++n) {
    score[n] = initial;
    contribution[n] = initial / degree[n];
  }
  // One array of contributions updated in place: when node u is visited,
  // sources v < u already hold this sweep's value and v >= u the previous
  // sweep's, which is exactly the ordered Gauss-Seidel rule.
  for (int sweep = 0; sweep < iterations; ++sweep) {
    for (int u = 0; u < nodes; ++u) {
      float total = 0.0f;
      for (int v : incoming[u]) total += contribution[v];
      score[u] = base + damping * total;
      contribution[u] = score[u] / degree[u];
    }
  }
  return score;
}

} // namespace pagerank_oracle
