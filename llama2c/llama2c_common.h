// Host-side code shared by llama2c_cuda_graph.cu (the baseline) and
// llama2c_sota.cu: input generation, the CPU oracles with their tolerances,
// and the --dump writer. Both binaries include this file so they generate
// identical inputs and judge their outputs by identical rules.
//
// Input modes (an extra command-line token, accepted anywhere):
//   deterministic   (default) the original inputs: input and probabilities
//                   are 1,2,3,4 repeating, weight is 1, the matrix is a
//                   shifted identity (row r has a 1 in column r % size).
//   random[=SEED]   every input element is uniform in [-1, 1) from a seeded
//                   counter-based RNG (SEED defaults to 1): input,
//                   probabilities, weight and matrix are independent streams.
// Dump (an extra token, or the LLAMA2C_DUMP environment variable):
//   --dump=FILE     after validation, write the host output buffers to FILE.
//
// Dump layout (little-endian, no padding):
//   offset 0   char[8]  magic "L2CDUMP1"
//   offset 8   int32    batch
//   offset 12  int32    size
//   offset 16  int32    rows
//   offset 20  int32    inputs (0 = deterministic, 1 = random)
//   offset 24  uint64   seed (0 for deterministic inputs)
//   offset 32  float32  normalized[batch * size]   rmsnorm output, row-major
//              float32  softmax[batch * size]      softmax output, row-major
//              float32  product[rows]              matmul output
//              float32  product_tolerance[rows]    absolute tolerance the
//                       oracle applied to each product element (see
//                       validate_outputs_reference)
//              float64  product_reference[rows]    the oracle's double
//                       reference for each product element
// The last two let a dump comparer check products without regenerating the
// matrix.
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

constexpr float kRmsEpsilon = 1.0e-5f;

struct InputOptions {
  bool random = false;
  uint64_t seed = 0;
  std::string dump_path;
};

// Consumes `deterministic`, `random`, `random=SEED` and `--dump=FILE`.
// Returns false if `arg` is none of them (the caller parses it), and exits on
// a malformed seed.
static bool parse_input_option(const char *arg, InputOptions *options) {
  if (std::strcmp(arg, "deterministic") == 0) {
    options->random = false;
    options->seed = 0;
    return true;
  }
  if (std::strcmp(arg, "random") == 0) {
    options->random = true;
    options->seed = 1;
    return true;
  }
  if (std::strncmp(arg, "random=", 7) == 0) {
    char *end = nullptr;
    const unsigned long long seed = std::strtoull(arg + 7, &end, 10);
    if (arg[7] == '\0' || *end != '\0') {
      std::fprintf(stderr, "bad seed in '%s'\n", arg);
      std::exit(EXIT_FAILURE);
    }
    options->random = true;
    options->seed = seed;
    return true;
  }
  if (std::strncmp(arg, "--dump=", 7) == 0) {
    options->dump_path = arg + 7;
    return true;
  }
  return false;
}

static void input_options_from_env(InputOptions *options) {
  const char *dump = std::getenv("LLAMA2C_DUMP");
  if (dump != nullptr && *dump != '\0' && options->dump_path.empty())
    options->dump_path = dump;
}

// splitmix64 of (seed, stream, index): a counter-based generator, so every
// element is independent of how the buffers are filled.
static inline uint64_t llama2c_mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Uniform in [-1, 1) on a 2^-23 grid; every value is exact in fp32.
static inline float llama2c_uniform(uint64_t seed, uint64_t stream, uint64_t index) {
  const uint64_t bits = llama2c_mix(llama2c_mix(seed * 4 + stream) ^ index);
  return static_cast<float>(static_cast<int64_t>(bits >> 40)) * (1.0f / 8388608.0f) - 1.0f;
}

static void fill_inputs(const InputOptions &options, int batch, int size, int rows,
                        std::vector<float> *input, std::vector<float> *probabilities,
                        std::vector<float> *weight, std::vector<float> *matrix) {
  const size_t vector_elements = static_cast<size_t>(batch) * size;
  const size_t matrix_elements = static_cast<size_t>(rows) * size;
  input->assign(vector_elements, 0.0f);
  probabilities->assign(vector_elements, 0.0f);
  weight->assign(size, 1.0f);
  matrix->assign(matrix_elements, 0.0f);
  if (!options.random) {
    for (size_t i = 0; i < vector_elements; ++i) {
      (*input)[i] = 1.0f + (i % 4);
      (*probabilities)[i] = 1.0f + (i % 4);
    }
    for (int row = 0; row < rows; ++row)
      for (int col = 0; col < size; ++col)
        (*matrix)[static_cast<size_t>(row) * size + col] = (col == (row % size)) ? 1.0f : 0.0f;
    return;
  }
  for (size_t i = 0; i < vector_elements; ++i) {
    (*input)[i] = llama2c_uniform(options.seed, 0, i);
    (*probabilities)[i] = llama2c_uniform(options.seed, 1, i);
  }
  for (int i = 0; i < size; ++i)
    (*weight)[i] = llama2c_uniform(options.seed, 2, i);
  for (size_t i = 0; i < matrix_elements; ++i)
    (*matrix)[i] = llama2c_uniform(options.seed, 3, i);
}

// ---------------------------------------------------------------------------
// The baseline's original oracle, moved here verbatim. It assumes the
// deterministic inputs: softmax is checked against `input` (equal to the
// probabilities buffer there) and the matmul against the shifted identity.

static bool approximately_equal(float actual, float expected, float tolerance) {
  return std::fabs(actual - expected) <= tolerance;
}

static bool approximately_equal_relative(float actual, float expected,
                                         float relative_tolerance,
                                         float absolute_tolerance) {
  return std::fabs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::fabs(expected);
}

static bool validate_outputs(const std::vector<float> &input,
                             const std::vector<float> &normalized,
                             const std::vector<float> &softmax,
                             const std::vector<float> &product,
                             const std::vector<float> &weight,
                             int batch, int size, int rows) {
  bool valid = true;
  const float rms_tolerance = 3.0e-4f;
  const float softmax_relative_tolerance = 1.0e-3f;
  const float softmax_absolute_tolerance = 1.0e-9f;
  for (int vector = 0; vector < batch; ++vector) {
    const size_t offset = static_cast<size_t>(vector) * size;
    double square_sum = 0.0;
    for (int index = 0; index < size; ++index) {
      const float value = input[offset + index];
      square_sum += static_cast<double>(value) * value;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(square_sum / size) + kRmsEpsilon);
    double probability_sum = 0.0;
    float maximum = -FLT_MAX;
    for (int index = 0; index < size; ++index)
      maximum = std::fmax(maximum, input[offset + index]);
    double exp_sum = 0.0;
    for (int index = 0; index < size; ++index)
      exp_sum += std::exp(static_cast<double>(input[offset + index] - maximum));
    for (int index = 0; index < size; ++index) {
      const float expected = input[offset + index] * scale * weight[index];
      valid = approximately_equal(normalized[offset + index], expected, rms_tolerance) && valid;
      const float expected_probability = static_cast<float>(
          std::exp(static_cast<double>(input[offset + index] - maximum)) / exp_sum);
      valid = approximately_equal_relative(softmax[offset + index], expected_probability,
                                           softmax_relative_tolerance,
                                           softmax_absolute_tolerance) && valid;
      probability_sum += softmax[offset + index];
    }
    valid = std::isfinite(probability_sum) &&
            std::fabs(probability_sum - 1.0) <= 3.0e-4 && valid;
    for (int index = 0; index < size; ++index) {
      // Every output must remain finite and positive, including vectors beyond
      // vector zero. The per-element reference comparison above catches value
      // and ordering errors.
      if (!std::isfinite(softmax[offset + index]) || softmax[offset + index] <= 0.0f)
        valid = false;
    }
  }
  for (int row = 0; row < rows; ++row)
    valid = approximately_equal(product[row], input[static_cast<size_t>(row) % size], 1.0e-4f) && valid;
  return valid;
}

// ---------------------------------------------------------------------------
// General oracle for any inputs: a double-precision CPU reference with the
// original rmsnorm and softmax tolerances (3e-4 absolute; 1e-3 relative plus
// 1e-9 absolute; probabilities sum to 1 within 3e-4).
//
// The matmul tolerance is an fp32 rounding-error bound relative to
// S_r = sum_c |M[r,c] x[c]|, the condition-number-free scale of a dot product:
//   tolerance_r = gamma~_n(lambda) * S_r + FLT_MIN,
//   gamma~_n(lambda) = exp(lambda sqrt(n) u + n u^2 / (1 - u)) - 1,
//   u = 2^-24, n = size, lambda = 9.
// This is the probabilistic bound of Higham & Mary ("A New Approach to
// Probabilistic Rounding Error Analysis", SIAM J. Sci. Comput. 41(5), 2019,
// inner-product theorem): for any summation order, with rounding errors
// modelled as independent and mean-zero, it holds with probability at least
// 1 - 2 n exp(-lambda^2 (1 - u)^2 / 2) (failure odds ~1e-13 per row at
// n = 16384), and FMA contraction only removes roundings.
// The worst-case bound gamma_n * S_r (gamma_n = n u / (1 - n u), Higham,
// "Accuracy and Stability of Numerical Algorithms", 2002, eq. 3.5) is
// ~1e-3 S_r at n = 16384, about 4 for random [-1,1) inputs, which would hide a
// dropped or misindexed term (|M x| < 1); gamma~ is ~7e-5 S_r, about 0.28.
// (A SOTA build that drops the last 4 terms of every row fails the random
// check and passes the deterministic one.)
// Implementations with a known accumulation depth d also satisfy the
// rigorous bound gamma_d * S_r: d ~ 140 for the baseline warp-per-row matmul
// (gamma_d ~ 8e-6, ~0.03 at S_r ~ 4096) and d ~ 25 for the SOTA (~1.5e-6),
// both inside the tolerance, so neither can fail it by a correct
// computation. The double reference is exact to ~n 2^-53. FLT_MIN keeps an
// all-zero row from being judged at zero tolerance.
//
// When product_tolerance / product_reference are non-null they receive the
// tolerance applied to, and the double reference for, each product element.
static double llama2c_gamma_probabilistic(int n) {
  const double u = std::ldexp(1.0, -24);
  const double lambda = 9.0;
  return std::expm1(lambda * std::sqrt(static_cast<double>(n)) * u + n * u * u / (1.0 - u));
}

static bool validate_outputs_reference(const std::vector<float> &input,
                                       const std::vector<float> &probabilities,
                                       const std::vector<float> &weight,
                                       const std::vector<float> &matrix,
                                       const std::vector<float> &normalized,
                                       const std::vector<float> &softmax,
                                       const std::vector<float> &product,
                                       int batch, int size, int rows,
                                       std::vector<float> *product_tolerance,
                                       std::vector<double> *product_reference) {
  bool valid = true;
  const double rms_tolerance = 3.0e-4;
  const double softmax_relative_tolerance = 1.0e-3;
  const double softmax_absolute_tolerance = 1.0e-9;
  for (int vector = 0; vector < batch; ++vector) {
    const size_t offset = static_cast<size_t>(vector) * size;
    double square_sum = 0.0;
    for (int index = 0; index < size; ++index)
      square_sum += static_cast<double>(input[offset + index]) * input[offset + index];
    const double scale = 1.0 / std::sqrt(square_sum / size + static_cast<double>(kRmsEpsilon));
    double maximum = -DBL_MAX;
    for (int index = 0; index < size; ++index)
      maximum = std::fmax(maximum, static_cast<double>(probabilities[offset + index]));
    double exp_sum = 0.0;
    for (int index = 0; index < size; ++index)
      exp_sum += std::exp(probabilities[offset + index] - maximum);
    double probability_sum = 0.0;
    for (int index = 0; index < size; ++index) {
      const double expected = input[offset + index] * scale * weight[index];
      if (!(std::fabs(normalized[offset + index] - expected) <= rms_tolerance)) valid = false;
      const double expected_probability = std::exp(probabilities[offset + index] - maximum) / exp_sum;
      const double actual = softmax[offset + index];
      if (!(std::fabs(actual - expected_probability) <=
            softmax_absolute_tolerance + softmax_relative_tolerance * expected_probability))
        valid = false;
      // Probabilities may underflow to zero for general logits, so only
      // finiteness and sign are required beyond the reference comparison.
      if (!std::isfinite(actual) || actual < 0.0) valid = false;
      probability_sum += actual;
    }
    if (!(std::isfinite(probability_sum) && std::fabs(probability_sum - 1.0) <= 3.0e-4)) valid = false;
  }
  const double gamma = llama2c_gamma_probabilistic(size);
  if (product_tolerance != nullptr) product_tolerance->assign(rows, 0.0f);
  if (product_reference != nullptr) product_reference->assign(rows, 0.0);
  for (int row = 0; row < rows; ++row) {
    const float *m = matrix.data() + static_cast<size_t>(row) * size;
    double sum = 0.0, magnitude = 0.0;
    for (int col = 0; col < size; ++col) {
      const double term = static_cast<double>(m[col]) * input[col];
      sum += term;
      magnitude += std::fabs(term);
    }
    const double tolerance = gamma * magnitude + FLT_MIN;
    if (product_tolerance != nullptr) (*product_tolerance)[row] = static_cast<float>(tolerance);
    if (product_reference != nullptr) (*product_reference)[row] = sum;
    if (!(std::fabs(product[row] - sum) <= tolerance)) valid = false;
  }
  return valid;
}

// Runs the oracles that apply to the input mode: the original one for
// deterministic inputs (unchanged tolerances) and the general reference for
// every mode. For deterministic inputs the reported product tolerance is the
// stricter of the two (the original 1e-4 or the fp32 bound); the identity
// matrix makes the product exact there.
static bool validate_all(const InputOptions &options,
                         const std::vector<float> &input, const std::vector<float> &probabilities,
                         const std::vector<float> &weight, const std::vector<float> &matrix,
                         const std::vector<float> &normalized, const std::vector<float> &softmax,
                         const std::vector<float> &product, int batch, int size, int rows,
                         std::vector<float> *product_tolerance,
                         std::vector<double> *product_reference) {
  bool valid = validate_outputs_reference(input, probabilities, weight, matrix, normalized,
                                          softmax, product, batch, size, rows, product_tolerance,
                                          product_reference);
  if (!options.random) {
    valid = validate_outputs(input, normalized, softmax, product, weight, batch, size, rows) && valid;
    if (product_tolerance != nullptr)
      for (float &tolerance : *product_tolerance) tolerance = std::fmin(tolerance, 1.0e-4f);
  }
  return valid;
}

static bool write_dump(const std::string &path, const InputOptions &options,
                       int batch, int size, int rows,
                       const float *normalized, const float *softmax, const float *product,
                       const std::vector<float> &product_tolerance,
                       const std::vector<double> &product_reference) {
  std::FILE *file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot open dump file %s\n", path.c_str());
    return false;
  }
  const size_t vector_elements = static_cast<size_t>(batch) * size;
  const int32_t header[4] = {batch, size, rows, options.random ? 1 : 0};
  const uint64_t seed = options.random ? options.seed : 0;
  bool ok = std::fwrite("L2CDUMP1", 1, 8, file) == 8 &&
            std::fwrite(header, sizeof(header), 1, file) == 1 &&
            std::fwrite(&seed, sizeof(seed), 1, file) == 1 &&
            std::fwrite(normalized, sizeof(float), vector_elements, file) == vector_elements &&
            std::fwrite(softmax, sizeof(float), vector_elements, file) == vector_elements &&
            std::fwrite(product, sizeof(float), rows, file) == static_cast<size_t>(rows) &&
            std::fwrite(product_tolerance.data(), sizeof(float), rows, file) == static_cast<size_t>(rows) &&
            std::fwrite(product_reference.data(), sizeof(double), rows, file) == static_cast<size_t>(rows);
  ok = std::fclose(file) == 0 && ok;
  if (!ok) std::fprintf(stderr, "failed to write dump file %s\n", path.c_str());
  return ok;
}
