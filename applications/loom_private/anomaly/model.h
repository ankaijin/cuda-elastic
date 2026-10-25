// TFLite model reader for the MLPerf Tiny ad01 int8 autoencoder.
//
// A bounds-checked FlatBuffer reader written from the public TFLite schema
// (schema.fbs, version 3), plus the validation private Loom's runner applies
// before it will execute the model. Parsing produces plain data (scales, zero
// points, int8 weights, int32 biases); it contains no inference arithmetic.
// Host only; no CUDA.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace anomaly_model {

constexpr size_t kModelBytes = 276976;
constexpr size_t kSampleElements = 640;
constexpr size_t kDatasetSamples = 40;
constexpr size_t kDatasetBytes = kDatasetSamples * kSampleElements * sizeof(float);
constexpr size_t kLayerCount = 10;
constexpr uint32_t kBufferCount = 33;
constexpr uint32_t kTensorCount = 31;
constexpr int32_t kInputTensor = 0;
constexpr int32_t kOutputTensor = 30;

// Thrown for any malformed or unexpected model content.
struct ModelError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Per-tensor quantization of one tensor.
struct Quantization {
  float scale = 0.0f;
  int64_t zero_point = 0;
};

// One fully connected layer: out[o] = requant(sum_i (in[i] - in.zp) * W[o][i]
// + bias), ReLU on every layer but the last.
struct Layer {
  int32_t input_tensor = -1, weight_tensor = -1, bias_tensor = -1,
          output_tensor = -1;
  uint32_t input_count = 0, output_count = 0;
  Quantization input, weight, bias, output;
  bool relu = false;
  std::vector<int8_t> weights; // [output_count][input_count], row-major
  std::vector<int32_t> biases; // [output_count]
};

struct Model {
  Quantization input; // tensor 0, used to quantize the float32 sample
  std::vector<Layer> layers;
};

// Parses and validates the model bytes; throws ModelError with a clear
// message on any deviation from the expected ad01 int8 structure.
Model parse(const std::vector<uint8_t> &bytes);

// Little-endian float32 at `data`.
float load_le_float(const uint8_t *data);

} // namespace anomaly_model
