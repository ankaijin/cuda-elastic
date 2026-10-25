// CPU oracle: a direct scalar evaluation of the quantized autoencoder.
//
// Arithmetic follows the workload contract (contract.json "data_generation"):
//   input:   q = clamp(trunc_away(double(x /f scale)) + zp)  (x /f scale is a
//            float32 division; trunc_away(v) = int64(v - 0.5) for v < 0 and
//            int64(v + 0.5) otherwise; |v| >= 1024 saturates first)
//   layer:   acc = sum_i (in_i - in_zp) * w_oi                (int64)
//            real = (double(acc) * in_scale) * w_scale + double(b_o) * b_scale
//            q = clamp(trunc_away(real / out_scale) + out_zp), max(q, out_zp)
//            on layers 1-9
// Built with -ffp-contract=off on x86-64 (SSE2 binary64), so every operation
// is a single IEEE-754 round-to-nearest-even step.
#include "oracle.h"

#include <cmath>
#include <stdexcept>

namespace anomaly_oracle {
namespace {

using anomaly_model::kDatasetSamples;
using anomaly_model::kSampleElements;

// Saturating conversion of a scaled real value to an int8 code.
int64_t to_code(double value, int64_t zero_point) {
  int64_t code;
  if (value <= -1024.0) {
    code = -128;
  } else if (value >= 1024.0) {
    code = 127;
  } else {
    const double biased = value < 0.0 ? value - 0.5 : value + 0.5;
    code = (int64_t)biased + zero_point;
  }
  if (code < -128) code = -128;
  if (code > 127) code = 127;
  return code;
}

void infer_one(const anomaly_model::Model &model, const uint8_t *sample,
               int8_t *result) {
  std::vector<int64_t> activations(kSampleElements);
  for (size_t i = 0; i < kSampleElements; ++i) {
    const float x = anomaly_model::load_le_float(sample + 4 * i);
    if (!std::isfinite(x))
      throw std::runtime_error("dataset holds a non-finite input value");
    const float ratio = x / model.input.scale; // float32 division
    activations[i] = to_code((double)ratio, model.input.zero_point);
  }
  std::vector<int64_t> next;
  for (const anomaly_model::Layer &layer : model.layers) {
    next.assign(layer.output_count, 0);
    for (uint32_t o = 0; o < layer.output_count; ++o) {
      int64_t acc = 0;
      const int8_t *row = layer.weights.data() + (size_t)o * layer.input_count;
      for (uint32_t i = 0; i < layer.input_count; ++i)
        acc += (activations[i] - layer.input.zero_point) * (int64_t)row[i];
      const double scaled_acc =
          ((double)acc * (double)layer.input.scale) * (double)layer.weight.scale;
      const double scaled_bias = (double)layer.biases[o] * (double)layer.bias.scale;
      const double real = scaled_acc + scaled_bias;
      int64_t code = to_code(real / (double)layer.output.scale,
                             layer.output.zero_point);
      if (layer.relu && code < layer.output.zero_point)
        code = layer.output.zero_point;
      next[o] = code;
    }
    activations.swap(next);
  }
  if (activations.size() != kSampleElements)
    throw std::runtime_error("model output is not 640 elements");
  for (size_t i = 0; i < kSampleElements; ++i) result[i] = (int8_t)activations[i];
}

int hex_value(uint8_t c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

} // namespace

std::vector<int8_t> infer_dataset(const anomaly_model::Model &model,
                                  const std::vector<uint8_t> &dataset) {
  if (dataset.size() != anomaly_model::kDatasetBytes)
    throw std::runtime_error("dataset size is not 102400 bytes");
  std::vector<int8_t> outputs(kDatasetSamples * kSampleElements);
  for (size_t s = 0; s < kDatasetSamples; ++s)
    infer_one(model, dataset.data() + s * kSampleElements * 4,
              outputs.data() + s * kSampleElements);
  return outputs;
}

bool parse_expected(const std::vector<uint8_t> &text, size_t rows,
                    std::vector<std::vector<uint8_t>> &outputs,
                    std::string &error) {
  outputs.clear();
  size_t at = 0;
  while (at < text.size()) {
    size_t end = at;
    while (end < text.size() && text[end] != '\n') ++end;
    const std::string line(text.begin() + at, text.begin() + end);
    at = end + 1;
    const std::string prefix =
        "measured_sample=" + std::to_string(outputs.size()) + " output=";
    if (line.compare(0, prefix.size(), prefix) != 0 ||
        line.size() != prefix.size() + 2 * kSampleElements) {
      error = "line " + std::to_string(outputs.size() + 1) +
              " is not 'measured_sample=" + std::to_string(outputs.size()) +
              " output=<1280 hex digits>'";
      return false;
    }
    std::vector<uint8_t> row(kSampleElements);
    for (size_t i = 0; i < kSampleElements; ++i) {
      const int hi = hex_value((uint8_t)line[prefix.size() + 2 * i]);
      const int lo = hex_value((uint8_t)line[prefix.size() + 2 * i + 1]);
      if (hi < 0 || lo < 0) {
        error = "non-hex digit in line " + std::to_string(outputs.size() + 1);
        return false;
      }
      row[i] = (uint8_t)(hi << 4 | lo);
    }
    outputs.push_back(std::move(row));
  }
  if (outputs.size() != rows) {
    error = "expected " + std::to_string(rows) + " output lines, found " +
            std::to_string(outputs.size());
    return false;
  }
  return true;
}

} // namespace anomaly_oracle
