// Independent host reference for experiment C (experiments/blackwell/lowprec,
// spec.json). Plain C++ with no CUDA or CUTLASS dependency: the format tables,
// the quantizers (nearest representable value found by search, not by the
// device's bit manipulation), dequantization, the dense fixtures, and double
// precision rmsnorm, classifier, softmax, and decoding.
#pragma once

#include <cstdint>
#include <vector>

namespace lowprec_ref {

enum class Format { E2M1, E3M2, UE4M3, UE8M0 };

// The value of a code (magnitude and sign as the format defines them).
double decode(Format format, unsigned code);
// The code nearest to v: ties to the code whose least significant magnitude
// bit is 0, saturation at the largest magnitude, +0 for a zero magnitude.
// UE4M3 and UE8M0 take non-negative v only.
unsigned encode(Format format, double v);
// BF16 bits of a float, round to nearest even.
uint16_t to_bf16(float v);
double bf16_value(uint16_t bits);

// One row quantized: element codes (one per value, unpacked) and one scale
// code per block.
struct QuantizedRow {
  std::vector<uint8_t> codes;
  std::vector<uint8_t> scales;
};

// MXFP6 E3M2 with UE8M0 scales per 32 values (spec.json "mxfp6").
void quantize_mxfp6(const float *row, long long width, uint8_t *codes, uint8_t *scales);
// NVFP4 E2M1 with UE4M3 scales per 16 values and the tensor encode scale s
// (spec.json "nvfp4").
void quantize_nvfp4(const float *row, long long width, float encode_scale, uint8_t *codes,
                    uint8_t *scales);
// Tensor scales of NVFP4 from the largest magnitude of the tensor.
float nvfp4_encode_scale(float tensor_amax);
float nvfp4_decode_scale(float encode_scale);
// Dequantized value of element k of a row; scale is the block's scale code.
double dequantize_mxfp6(uint8_t code, uint8_t scale);
double dequantize_nvfp4_unscaled(uint8_t code, uint8_t scale); // without d

// Bit unpacking of the device layouts: value i of a packed E2M1 / E3M2 row.
unsigned unpack_e2m1(const uint8_t *row, long long i);
unsigned unpack_e3m2(const uint8_t *row, long long i);

// Dense fixture (spec.json "fixtures"): vocab x width embedding and classifier.
struct Fixture {
  std::vector<float> embedding, classifier;
};
Fixture make_fixture(int seed, long long width, long long vocab);

// rmsnorm in double of x (weight 1), as the contract defines it.
void rmsnorm_double(const float *x, long long width, double *out);
// rmsnorm in float with a sequential square sum (the fixed FP32 reference
// activations of the accuracy measurement).
void rmsnorm_float(const float *x, long long width, float *out);
// Softmax in double of a row of logits.
void softmax_double(const double *logits, long long count, double *out);
long long argmax_first(const double *values, long long count);

// Precision metrics of case logits against reference logits ([rows][vocab]).
struct ErrorMetrics {
  double max_abs_error = 0, normalized_rms_error = 0, max_abs_probability_error = 0,
         mean_total_variation = 0;
  long long rows = 0, top1_agree = 0;
};
ErrorMetrics compare_logits(const std::vector<double> &reference,
                            const std::vector<double> &actual, long long rows,
                            long long vocab);

} // namespace lowprec_ref
