// Independent CPU oracle for the multisensor attention workload.
//
// It regenerates the sensor inputs from the documented formula, projects them,
// and evaluates the attention, residual, and statistics in double precision.
// It shares no computation code with the CUDA kernels in attention.cu. Memory
// stays bounded: each query row streams over the context twice (maximum, then
// exponentials and weighted sums); no query x context matrix is formed.
#pragma once

#include <vector>

namespace attention_oracle {

struct Reference {
  // Residual output [query_tokens][4], row-major.
  std::vector<double> output;
  // projection energy, attention sum, weighted sum, maximum |attention|.
  double statistics[4] = {0.0, 0.0, 0.0, 0.0};
  // statistics[2] + 3 statistics[0] + 7 statistics[1] + 11 statistics[3].
  double combined = 0.0;
};

// scale is the input variant's scale (1.0, 0.5, or 1.5). Rows are split over
// worker threads; the result does not depend on the thread count.
void compute(long long query_tokens, long long context_tokens, double scale,
             Reference &reference);

} // namespace attention_oracle
