// Independent CPU oracle for loom-private/mlperf-tiny-anomaly-model. Shares
// no arithmetic with the CUDA kernels: it uses only the parsed model data.
#pragma once

#include "model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace anomaly_oracle {

// Runs the 10-layer int8 model on every dataset sample (40 x 640 float32,
// little-endian) and returns the int8 outputs, [40][640]. Throws
// std::runtime_error if a sample holds a non-finite value.
std::vector<int8_t> infer_dataset(const anomaly_model::Model &model,
                                  const std::vector<uint8_t> &dataset);

// Parses an upstream expected-output file: lines of
// "measured_sample=K output=<1280 lowercase hex digits>" with K = 0, 1, ...
// Returns false and sets `error` unless it holds exactly `rows` such lines.
bool parse_expected(const std::vector<uint8_t> &text, size_t rows,
                    std::vector<std::vector<uint8_t>> &outputs,
                    std::string &error);

} // namespace anomaly_oracle
