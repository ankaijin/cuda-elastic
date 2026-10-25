// Independent CPU oracle for loom-private/pagerank. It regenerates the graph
// from the out-edge rule on its own and runs the sequential ordered
// Gauss-Seidel pull loop in float, node by node. It shares no code with the
// GPU path (graph.cpp and pagerank.cu).
#pragma once

#include <vector>

namespace pagerank_oracle {

// Scores after `iterations` ordered Gauss-Seidel sweeps over the generated
// graph with `nodes` vertices.
std::vector<float> gauss_seidel_scores(int nodes, int iterations);

} // namespace pagerank_oracle
