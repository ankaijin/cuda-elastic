// FlatBuffer reader and ad01 model validation. Field numbers follow the public
// TFLite schema (tensorflow/lite/schema/schema.fbs):
//   Model:          0 version, 1 operator_codes, 2 subgraphs, 4 buffers
//   OperatorCode:   0 deprecated_builtin_code (byte), 2 version (int, default 1),
//                   3 builtin_code (int)
//   SubGraph:       0 tensors, 1 inputs, 2 outputs, 3 operators
//   Tensor:         0 shape, 1 type, 2 buffer, 4 quantization
//   Quantization:   2 scale [float], 3 zero_point [long]
//   Operator:       0 opcode_index, 1 inputs, 2 outputs,
//                   3 builtin_options_type, 4 builtin_options
//   FullyConnectedOptions: 0 fused_activation_function, 1 weights_format,
//                   2 keep_num_dims, 3 asymmetric_quantize_inputs
//   Buffer:         0 data [ubyte]
#include "model.h"

#include <cmath>
#include <cstring>

namespace anomaly_model {
namespace {

constexpr int kTypeInt32 = 2;
constexpr int kTypeInt8 = 9;
constexpr int kOpFullyConnected = 9;
constexpr int kOpExtendedCodes = 127; // builtin_code holds codes >= this
constexpr int kOptionsFullyConnected = 8;
constexpr int kActivationNone = 0;
constexpr int kActivationRelu = 1;

[[noreturn]] void bad(const std::string &what) { throw ModelError(what); }

// Bounds-checked little-endian view of the whole FlatBuffer.
class Bytes {
public:
  explicit Bytes(const std::vector<uint8_t> &bytes) : bytes_(bytes) {}
  size_t size() const { return bytes_.size(); }
  void need(uint64_t at, uint64_t length, const char *what) const {
    if (at > bytes_.size() || length > bytes_.size() - at)
      bad(std::string("FlatBuffer read past the end (") + what + ")");
  }
  uint64_t le(uint64_t at, int width, const char *what) const {
    need(at, (uint64_t)width, what);
    uint64_t value = 0;
    for (int i = width - 1; i >= 0; --i) value = value << 8 | bytes_[at + i];
    return value;
  }
  // Follows a uoffset_t stored at `at`; the target must lie inside the buffer.
  uint64_t follow(uint64_t at, const char *what) const {
    const uint64_t relative = le(at, 4, what);
    if (relative == 0 || at + relative >= bytes_.size())
      bad(std::string("FlatBuffer offset out of range (") + what + ")");
    return at + relative;
  }
  const uint8_t *data(uint64_t at) const { return bytes_.data() + at; }

private:
  const std::vector<uint8_t> &bytes_;
};

class Vector;

// A table: an soffset_t to its vtable, then inline fields.
class Table {
public:
  Table(const Bytes &bytes, uint64_t at, const char *what)
      : bytes_(bytes), at_(at), what_(what) {
    const int64_t to_vtable = (int32_t)(uint32_t)bytes_.le(at_, 4, what_);
    const int64_t vtable = (int64_t)at_ - to_vtable;
    if (to_vtable == 0 || vtable < 0)
      bad(std::string("FlatBuffer vtable out of range (") + what_ + ")");
    vtable_ = (uint64_t)vtable;
    vtable_bytes_ = (uint32_t)bytes_.le(vtable_, 2, what_);
    if (vtable_bytes_ < 4 || vtable_bytes_ % 2)
      bad(std::string("FlatBuffer vtable malformed (") + what_ + ")");
    bytes_.need(vtable_, vtable_bytes_, what_);
  }
  // Absolute position of field `id`, or 0 when the field is absent.
  uint64_t field(unsigned id) const {
    const uint64_t slot = 4 + 2ull * id;
    if (slot + 2 > vtable_bytes_) return 0;
    const uint64_t offset = bytes_.le(vtable_ + slot, 2, what_);
    if (offset == 0) return 0;
    bytes_.need(at_ + offset, 1, what_);
    return at_ + offset;
  }
  bool has(unsigned id) const { return field(id) != 0; }
  int64_t scalar(unsigned id, int width, bool is_signed, int64_t fallback) const {
    const uint64_t at = field(id);
    if (!at) return fallback;
    const uint64_t raw = bytes_.le(at, width, what_);
    if (!is_signed) return (int64_t)raw;
    const int shift = 64 - 8 * width;
    return (int64_t)(raw << shift) >> shift;
  }
  Table table(unsigned id, const char *what) const {
    const uint64_t at = field(id);
    if (!at) bad(std::string("required table missing: ") + what);
    return Table(bytes_, bytes_.follow(at, what), what);
  }
  Vector vector(unsigned id, unsigned element_bytes, const char *what) const;

private:
  const Bytes &bytes_;
  uint64_t at_;
  const char *what_;
  uint64_t vtable_ = 0;
  uint32_t vtable_bytes_ = 0;
};

// A vector: a uint32 length, then elements of a fixed width.
class Vector {
public:
  Vector(const Bytes &bytes, uint64_t at, unsigned element_bytes, const char *what)
      : bytes_(bytes), element_bytes_(element_bytes), what_(what) {
    length_ = (uint32_t)bytes_.le(at, 4, what);
    start_ = at + 4;
    bytes_.need(start_, (uint64_t)length_ * element_bytes_, what);
  }
  uint32_t size() const { return length_; }
  uint64_t raw(uint32_t index) const {
    return bytes_.le(position(index), (int)element_bytes_, what_);
  }
  int32_t i32(uint32_t index) const { return (int32_t)(uint32_t)raw(index); }
  int64_t i64(uint32_t index) const { return (int64_t)raw(index); }
  float f32(uint32_t index) const { return load_le_float(bytes_.data(position(index))); }
  Table table(uint32_t index, const char *what) const {
    return Table(bytes_, bytes_.follow(position(index), what), what);
  }
  const uint8_t *data() const { return bytes_.data(start_); }

private:
  uint64_t position(uint32_t index) const {
    if (index >= length_)
      bad(std::string("FlatBuffer vector index out of range (") + what_ + ")");
    return start_ + (uint64_t)index * element_bytes_;
  }
  const Bytes &bytes_;
  unsigned element_bytes_;
  const char *what_;
  uint32_t length_ = 0;
  uint64_t start_ = 0;
};

Vector Table::vector(unsigned id, unsigned element_bytes, const char *what) const {
  const uint64_t at = field(id);
  if (!at) bad(std::string("required vector missing: ") + what);
  return Vector(bytes_, bytes_.follow(at, what), element_bytes, what);
}

struct TensorInfo {
  std::vector<int32_t> shape;
  int type = -1;
  Quantization quantization;
  const uint8_t *data = nullptr; // buffer contents, or nullptr when empty
  uint64_t data_bytes = 0;
  uint64_t elements = 1;
};

struct BufferInfo {
  const uint8_t *data = nullptr;
  uint64_t bytes = 0;
};

bool finite_positive(float value) { return std::isfinite(value) && value > 0.0f; }

int32_t le_i32(const uint8_t *p) {
  const uint32_t bits = (uint32_t)p[0] | (uint32_t)p[1] << 8 |
                        (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
  int32_t value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

} // namespace

float load_le_float(const uint8_t *data) {
  const uint32_t bits = (uint32_t)data[0] | (uint32_t)data[1] << 8 |
                        (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

Model parse(const std::vector<uint8_t> &raw) {
  if (raw.size() != kModelBytes)
    bad("model has " + std::to_string(raw.size()) + " bytes; expected " +
        std::to_string(kModelBytes));
  if (std::memcmp(raw.data() + 4, "TFL3", 4) != 0)
    bad("model lacks the TFL3 file identifier");
  const Bytes bytes(raw);
  const Table model(bytes, bytes.follow(0, "model root"), "model");
  if (model.scalar(0, 4, false, 0) != 3)
    bad("model schema version is not 3");

  // Operator catalog: exactly one code, FULLY_CONNECTED version 4.
  const Vector codes = model.vector(1, 4, "operator_codes");
  if (codes.size() != 1) bad("model must have exactly one operator code");
  const Table code = codes.table(0, "operator code");
  const int64_t deprecated = code.scalar(0, 1, true, 0);
  const int64_t builtin = code.scalar(3, 4, true, 0);
  const int64_t effective = builtin < kOpExtendedCodes ? deprecated : builtin;
  if (effective != kOpFullyConnected || code.scalar(2, 4, true, 1) != 4)
    bad("model operator code is not FULLY_CONNECTED version 4");

  // Buffers: 33 entries; a buffer without data is empty.
  const Vector buffer_list = model.vector(4, 4, "buffers");
  if (buffer_list.size() != kBufferCount)
    bad("model must have 33 buffers, found " + std::to_string(buffer_list.size()));
  std::vector<BufferInfo> buffers(kBufferCount);
  for (uint32_t b = 0; b < kBufferCount; ++b) {
    const Table buffer = buffer_list.table(b, "buffer");
    if (buffer.has(0)) {
      const Vector data = buffer.vector(0, 1, "buffer data");
      buffers[b] = {data.data(), data.size()};
    }
  }

  const Vector subgraphs = model.vector(2, 4, "subgraphs");
  if (subgraphs.size() != 1) bad("model must have exactly one subgraph");
  const Table subgraph = subgraphs.table(0, "subgraph");

  // Tensors: 31, rank 1 or 2, per-tensor scale and zero point.
  const Vector tensor_list = subgraph.vector(0, 4, "tensors");
  if (tensor_list.size() != kTensorCount)
    bad("model must have 31 tensors, found " + std::to_string(tensor_list.size()));
  std::vector<TensorInfo> tensors(kTensorCount);
  for (uint32_t t = 0; t < kTensorCount; ++t) {
    TensorInfo &info = tensors[t];
    const Table tensor = tensor_list.table(t, "tensor");
    const Vector shape = tensor.vector(0, 4, "tensor shape");
    if (shape.size() < 1 || shape.size() > 2)
      bad("tensor " + std::to_string(t) + " rank is not 1 or 2");
    for (uint32_t d = 0; d < shape.size(); ++d) {
      const int32_t extent = shape.i32(d);
      if (extent <= 0 || extent > 1 << 20)
        bad("tensor " + std::to_string(t) + " has an invalid extent");
      info.shape.push_back(extent);
      info.elements *= (uint64_t)extent;
    }
    info.type = (int)tensor.scalar(1, 1, true, 0);
    const int64_t buffer = tensor.scalar(2, 4, false, 0);
    if (buffer >= (int64_t)kBufferCount)
      bad("tensor " + std::to_string(t) + " names a buffer out of range");
    info.data = buffers[buffer].data;
    info.data_bytes = buffers[buffer].bytes;
    const Table quantization = tensor.table(4, "tensor quantization");
    const Vector scales = quantization.vector(2, 4, "quantization scale");
    const Vector zero_points = quantization.vector(3, 8, "quantization zero_point");
    if (scales.size() != 1 || zero_points.size() != 1)
      bad("tensor " + std::to_string(t) + " is not per-tensor quantized");
    info.quantization.scale = scales.f32(0);
    info.quantization.zero_point = zero_points.i64(0);
    if (!finite_positive(info.quantization.scale))
      bad("tensor " + std::to_string(t) + " scale is not finite and positive");
  }

  // Graph endpoints: input tensor 0, output tensor 30, both int8 [.., 640].
  const Vector inputs = subgraph.vector(1, 4, "subgraph inputs");
  const Vector outputs = subgraph.vector(2, 4, "subgraph outputs");
  if (inputs.size() != 1 || inputs.i32(0) != kInputTensor ||
      outputs.size() != 1 || outputs.i32(0) != kOutputTensor)
    bad("model input must be tensor 0 and output tensor 30");
  for (int32_t endpoint : {kInputTensor, kOutputTensor})
    if (tensors[endpoint].type != kTypeInt8 ||
        tensors[endpoint].elements != kSampleElements)
      bad("model endpoints must be int8 tensors of 640 elements");

  // Ten chained FULLY_CONNECTED operators.
  const Vector operators = subgraph.vector(3, 4, "operators");
  if (operators.size() != kLayerCount)
    bad("model must have 10 operators, found " + std::to_string(operators.size()));
  Model result;
  result.input = tensors[kInputTensor].quantization;
  int32_t previous = kInputTensor;
  for (uint32_t index = 0; index < kLayerCount; ++index) {
    const std::string where = "layer " + std::to_string(index + 1) + ": ";
    const Table op = operators.table(index, "operator");
    if (op.scalar(0, 4, false, 0) != 0) bad(where + "opcode index is not 0");
    const Vector op_inputs = op.vector(1, 4, "operator inputs");
    const Vector op_outputs = op.vector(2, 4, "operator outputs");
    if (op_inputs.size() != 3 || op_outputs.size() != 1)
      bad(where + "expected 3 inputs and 1 output");
    Layer layer;
    layer.input_tensor = op_inputs.i32(0);
    layer.weight_tensor = op_inputs.i32(1);
    layer.bias_tensor = op_inputs.i32(2);
    layer.output_tensor = op_outputs.i32(0);
    for (int32_t t : {layer.input_tensor, layer.weight_tensor, layer.bias_tensor,
                      layer.output_tensor})
      if (t < 0 || t >= (int32_t)kTensorCount) bad(where + "tensor index out of range");
    if (op.scalar(3, 1, false, 0) != kOptionsFullyConnected)
      bad(where + "options are not FullyConnectedOptions");
    const Table options = op.table(4, "FullyConnectedOptions");
    const int64_t activation = options.scalar(0, 1, true, kActivationNone);
    const int64_t weights_format = options.scalar(1, 1, true, 0);
    const int64_t keep_dims = options.scalar(2, 1, false, 0);
    const int64_t asymmetric = options.scalar(3, 1, false, 0);
    if (keep_dims > 1 || asymmetric > 1) bad(where + "malformed boolean option");
    if (weights_format != 0 || keep_dims != 0 || asymmetric != 0)
      bad(where + "unsupported FullyConnectedOptions");
    layer.relu = index + 1 < kLayerCount;
    if (activation != (layer.relu ? kActivationRelu : kActivationNone))
      bad(where + "activation must be RELU on layers 1-9 and NONE on layer 10");
    if (layer.input_tensor != previous)
      bad(where + "input is not the previous layer's output");

    const TensorInfo &in = tensors[layer.input_tensor];
    const TensorInfo &w = tensors[layer.weight_tensor];
    const TensorInfo &bias = tensors[layer.bias_tensor];
    const TensorInfo &out = tensors[layer.output_tensor];
    if (in.type != kTypeInt8 || w.type != kTypeInt8 || bias.type != kTypeInt32 ||
        out.type != kTypeInt8)
      bad(where + "tensor types must be int8 input/weights/output, int32 bias");
    if (in.elements > kSampleElements || out.elements > kSampleElements)
      bad(where + "layer is wider than 640");
    layer.input_count = (uint32_t)in.elements;
    layer.output_count = (uint32_t)out.elements;
    if (w.shape.size() != 2 || (uint32_t)w.shape[0] != layer.output_count ||
        (uint32_t)w.shape[1] != layer.input_count || bias.shape.size() != 1 ||
        bias.elements != layer.output_count)
      bad(where + "weight or bias shape does not match the layer extents");
    if (!w.data || w.data_bytes != (uint64_t)layer.input_count * layer.output_count ||
        !bias.data || bias.data_bytes != 4ull * layer.output_count)
      bad(where + "weight or bias buffer size is wrong");
    if (w.quantization.zero_point != 0 || bias.quantization.zero_point != 0)
      bad(where + "weight and bias zero points must be 0");
    if (in.quantization.zero_point < -128 || in.quantization.zero_point > 127 ||
        out.quantization.zero_point < -128 || out.quantization.zero_point > 127)
      bad(where + "activation zero point outside int8");
    layer.input = in.quantization;
    layer.weight = w.quantization;
    layer.bias = bias.quantization;
    layer.output = out.quantization;
    layer.weights.resize(w.data_bytes);
    std::memcpy(layer.weights.data(), w.data, w.data_bytes);
    layer.biases.resize(layer.output_count);
    for (uint32_t o = 0; o < layer.output_count; ++o)
      layer.biases[o] = le_i32(bias.data + 4ull * o);
    previous = layer.output_tensor;
    result.layers.push_back(std::move(layer));
  }
  if (previous != kOutputTensor) bad("the layer chain does not end at tensor 30");
  return result;
}

} // namespace anomaly_model
