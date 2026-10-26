// Device quantization arithmetic of the optimized anomaly variant, shared by
// anomaly_optimized.cu and the exactness test tests/test_requant.cu.
//
// saturate_int8(), requantize(), and quantize_input() are the baseline's
// arithmetic (anomaly.cu), unchanged: the exact sequence of the contract. The
// optimized kernel computes almost every code with the binary32 fast paths
// below, each of which either proves that its code equals the baseline's or
// falls back to the baseline function. tests/test_requant.cu checks both fast
// paths against the baseline functions for every binary32 input and every
// accumulator within the contract's bound.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace anomaly_requant {

// Kernel-side layer description (as in anomaly.cu).
struct LayerParams {
  int input_count, output_count;
  int weight_offset, bias_offset; // into the packed weight / bias arrays
  float input_scale, weight_scale, bias_scale, output_scale;
  int input_zero_point, output_zero_point;
  int relu;
};

// Upstream's saturating int8 conversion of a scaled value: |v| >= 1024
// saturates; otherwise add +-0.5 in double, truncate toward zero, add the
// zero point, and clip to [-128, 127]. (Unchanged from anomaly.cu.)
__device__ __forceinline__ int saturate_int8(double value, int zero_point) {
  if (value <= -1024.0) return -128;
  if (value >= 1024.0) return 127;
  const double biased = value < 0.0 ? __dsub_rn(value, 0.5) : __dadd_rn(value, 0.5);
  const long long shifted = (long long)biased + zero_point; // cvt.rzi
  return shifted <= -128 ? -128 : shifted >= 127 ? 127 : (int)shifted;
}

// real = (double(acc) * in_scale) * w_scale + double(bias) * bias_scale,
// then real / out_scale; every step one IEEE-754 binary64 round-to-nearest.
// (Unchanged from anomaly.cu.)
__device__ __forceinline__ int requantize(int acc, int bias, const LayerParams &layer) {
  const double scaled_acc =
      __dmul_rn(__dmul_rn((double)acc, (double)layer.input_scale),
                (double)layer.weight_scale);
  const double scaled_bias = __dmul_rn((double)bias, (double)layer.bias_scale);
  const double real = __dadd_rn(scaled_acc, scaled_bias);
  int code = saturate_int8(__ddiv_rn(real, (double)layer.output_scale),
                           layer.output_zero_point);
  if (layer.relu && code < layer.output_zero_point) code = layer.output_zero_point;
  return code;
}

// The baseline's input quantization of one element: binary32 division, then
// saturate_int8 of the quotient. (As in anomaly.cu.)
__device__ __forceinline__ int quantize_input(float x, float scale, int zero_point) {
  const float ratio = __fdiv_rn(x, scale); // float32 division
  return saturate_int8((double)ratio, zero_point);
}

// Host constants of the fast paths, from binary64:
// - input_reciprocal = fl32(1 / input scale), or 0 when it is not a normal
//   number (never for the pinned model; 0 selects quantize_input()).
// - per layer kappa = fl32(in_scale * w_scale / out_scale), or 0 when it is
//   not finite or below 2^-100, so that kappa * 2^-20 stays normal (0
//   selects requantize()); and per output beta = fl32(bias * bias_scale /
//   out_scale); a non-finite beta also sets kappa to 0.
inline float fast_input_reciprocal(float scale) {
  const float reciprocal = (float)(1.0 / (double)scale);
  return std::isnormal(reciprocal) && reciprocal > 0.0f ? reciprocal : 0.0f;
}
inline float fast_constants(const LayerParams &layer, const int32_t *biases,
                            float *betas) {
  float kappa = (float)((double)layer.input_scale * (double)layer.weight_scale /
                        (double)layer.output_scale);
  if (!std::isfinite(kappa) || !(kappa >= 0x1p-100f)) kappa = 0.0f;
  for (int o = 0; o < layer.output_count; ++o) {
    betas[o] = (float)((double)biases[o] * (double)layer.bias_scale /
                       (double)layer.output_scale);
    if (!std::isfinite(betas[o])) {
      betas[o] = 0.0f;
      kappa = 0.0f;
    }
  }
  return kappa;
}

// quantize_input_try_fast() computes quantize_input(x, scale, zero_point)
// with one binary32 multiply and returns true when that code is proven equal;
// quantize_input_fast() falls back to quantize_input() otherwise. reciprocal
// comes from fast_input_reciprocal(scale).
//
// The baseline code is sat(ratio) with ratio = fl32(x / scale). For
// |ratio| < 1024, ratio +- 0.5 is exact in binary64 when |ratio| >= 2^-29 (at
// most 34 significant bits) and stays within 2^-29 of +-0.5 otherwise, so
// saturate_int8 gives clip(round_half_away(ratio) + zero point); for
// |ratio| >= 1024 it saturates, as does the clip of any |n| >= 1023 plus a
// zero point in [-128, 127].
// Let p = x / scale exactly. q = fl32(x * reciprocal) is within
// (2^-23 + 2^-47) |p| of p, and ratio within 2^-24 |p|. r = rint(q), and
// d = |q - r| is exact (r = 0 for |q| < 0.5, else Sterbenz). The test
// fl32(d + |q| 2^-20) < 0.5 implies that q is more than |q| 2^-20 from every
// half-integer (rounding is monotone and 0.5 is representable), so p is more
// than 0.87 * 2^-20 |p| > 2^-24 |p| from every half-integer, and q, p, and
// ratio all lie strictly between the same two half-integers: r is
// round_half_away(ratio). |q| >= 2^19 (or an overflowed q) fails the test, so
// the conversion of r cannot overflow. The fallback fraction is about
// 2^-19 |p|.
__device__ __forceinline__ bool quantize_input_try_fast(float x, float reciprocal,
                                                       int zero_point, int &code) {
  const float q = __fmul_rn(x, reciprocal);
  const float r = rintf(q);
#ifdef LP_ANOMALY_MUTANT_NO_MARGIN
  const float margin = 0.0f; // test-only mutant: accepts unproven codes
#else
  const float margin = __fmul_rn(fabsf(q), 0x1p-20f);
#endif
  if (!(reciprocal > 0.0f && __fadd_rn(fabsf(__fsub_rn(q, r)), margin) < 0.5f)) return false;
  code = min(max((int)r + zero_point, -128), 127);
  return true;
}

__device__ __forceinline__ int quantize_input_fast(float x, float reciprocal, float scale,
                                                   int zero_point) {
  int code;
  return quantize_input_try_fast(x, reciprocal, zero_point, code)
             ? code
             : quantize_input(x, scale, zero_point);
}

// The per-output part of requantize_fast()'s margin: 2^-20 |beta| + 2^-22.
__device__ __forceinline__ float fast_margin_base(float beta) {
  return __fmaf_rn(fabsf(beta), 0x1p-20f, 0x1p-22f);
}

// requantize_try_fast() computes requantize(acc, bias, layer) in binary32 and
// returns true when that code is proven equal; requantize_fast() falls back to
// requantize() otherwise. kappa and beta come from
// fast_constants(), lambda = kappa * 2^-20 (exact), margin_base from
// fast_margin_base(beta), and code_min is the layer's zero point with ReLU,
// else -128.
//
// Let v = acc * kappa* + beta* be the exact value of real / out_scale
// (kappa*, beta* the exact quotients) and S = |acc| kappa* + |beta*| >= |v|.
// - The binary64 path rounds five times, so its quotient V is within
//   6 * 2^-53 S of v, and its code is v rounded half away from zero (then
//   zero point, clip, ReLU) whenever v is farther than
//   E64 = 6 * 2^-53 S + 2^-52 (S + 1) from every half-integer (the second
//   term covers the rounding of V +- 0.5).
// - f = fma(fl32(acc), kappa, beta) rounds acc (2^-24 relative, exact below
//   2^24), kappa and beta (2^-24 + 2^-51 each), and the fma (2^-24), so
//   |f - v| <= E32 = 2^-21.9 S.
// - r = rint(f), and d = |f - r| is exact (r = 0 for |f| < 0.5, else
//   Sterbenz). margin = fma(|fl32(acc)|, lambda, margin_base) is at least
//   (1 - 2^-23) (2^-20 S + 2^-22). The test fl32(d + margin) < 0.5 implies
//   d + margin < 0.5 (rounding is monotone and 0.5 is representable), so f is
//   more than margin from every half-integer, and v more than
//   margin - E32 >= 0.73 * 2^-20 S + 0.99 * 2^-22 > E64.
// So no half-integer lies between f and v, r is v rounded to nearest (no tie
// is possible), which is the binary64 path's rounding, and r + zero point
// clipped to [code_min, 127] is the binary64 path's clip and ReLU (the zero
// point is in [-128, 127]). Saturation agrees too: |v| >= 1024 gives -128 or
// 127 in the binary64 path, and |r| >= 1023 does after the clip. A huge f
// cannot overflow the conversion: |f| >= 2^19 makes margin >= 0.5 and takes
// the fallback. The fallback fraction is about 2 * margin per output.
__device__ __forceinline__ bool requantize_try_fast(int acc, float beta, float margin_base,
                                                   float kappa, float lambda, int zero_point,
                                                   int code_min, int &code) {
  const float a = (float)acc;
  const float f = __fmaf_rn(a, kappa, beta);
#ifdef LP_ANOMALY_MUTANT_NO_MARGIN
  const float margin = 0.0f * margin_base * lambda; // test-only mutant
#else
  const float margin = __fmaf_rn(fabsf(a), lambda, margin_base);
#endif
  const float r = rintf(f);
  if (!(kappa > 0.0f && __fadd_rn(fabsf(__fsub_rn(f, r)), margin) < 0.5f)) return false;
  code = min(max((int)r + zero_point, code_min), 127);
  return true;
}

__device__ __forceinline__ int requantize_fast(int acc, const int32_t *bias, float beta,
                                               float margin_base, float kappa,
                                               float lambda, int code_min,
                                               const LayerParams &layer) {
  int code;
  return requantize_try_fast(acc, beta, margin_base, kappa, lambda, layer.output_zero_point,
                             code_min, code)
             ? code
             : requantize(acc, *bias, layer);
}

} // namespace anomaly_requant
