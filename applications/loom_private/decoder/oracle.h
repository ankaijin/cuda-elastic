// Independent CPU oracle for the llama2c decoder (host code only).
//
// Nothing here is shared with the CUDA kernels. The oracle regenerates the
// synthetic model from its own definition, predicts every generated token from
// the model's analytic progression, evaluates the dense
// rmsnorm -> classifier matmul -> softmax -> argmax path in double precision
// for a deterministic subset of sequences, and checks the GPU's outputs.
#pragma once

#include <cstdint>
#include <vector>

namespace decoder_oracle {

struct Shape {
  long long batch;  // independent sequences
  long long width;  // model dimension (embedding and classifier columns)
  long long vocab;  // classifier rows
  long long steps;  // dependent decoding steps per sequence
};

struct Reference {
  // Expected token of every sequence after every step, [batch][steps].
  std::vector<int32_t> tokens;
  // Sequences evaluated densely in double, in increasing order.
  std::vector<long long> sampled;
  // Dense double results for sampled sequence i at step s:
  //   normalized[(i * steps + s) * width + j]
  //   logits / probabilities[(i * steps + s) * vocab + r]
  std::vector<double> normalized;
  std::vector<double> logits;
  std::vector<double> probabilities;
  // False if the dense evaluation disagreed with the analytic progression,
  // which would mean the oracle itself is wrong.
  bool self_consistent = true;
};

// Prompt of sequence b (identical definition to the contract).
long long prompt_of(long long sequence, long long vocab);

// Runs the oracle (multithreaded over the sampled sequences).
Reference compute(const Shape &shape);

// GPU outputs as copied to the host.
struct Observed {
  const int32_t *tokens;      // [batch][steps]
  const float *normalized;    // [steps][batch][width]
  const float *logits;        // [steps][batch][vocab]
  const float *probabilities; // [steps][batch][vocab]
};

// Compares observed outputs with the reference. Prints the first mismatches
// of each kind and a summary to stderr. Returns true when everything passes.
bool check(const Shape &shape, const Reference &reference,
           const Observed &observed);

} // namespace decoder_oracle
