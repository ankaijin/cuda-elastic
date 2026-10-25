// Independent host oracle for loom-private/vecadd-memory. It shares no code
// with the CUDA kernel or the workload's input generation.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vecadd_oracle {

struct Result {
  std::vector<int32_t> result; // lhs[i] + rhs[i] for i = 0 .. length-1
  int64_t checksum = 0;        // 64-bit sum of every result element
};

// Valid when length >= 1 and every lhs, rhs, and result value fits int32.
// Returns an empty string when valid, otherwise a message naming the problem.
// The inputs are lhs[i] = i and rhs[i] = i * multiplier + bias.
std::string validate(long long length, long long multiplier, long long bias);

// Builds both inputs in 64-bit arithmetic and adds them element by element.
Result compute(long long length, long long multiplier, long long bias);

} // namespace vecadd_oracle
