// Host tests (no GPU): SHA-256 vectors, the model reader and its validation
// on the cached model and on corrupted copies (which the binary never parses,
// because their hash fails first), and the oracle's non-finite rejection.
//   test_host <asset cache directory>
#include "../assets.h"
#include "../model.h"
#include "../oracle.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

int passed = 0, failed = 0;

void expect(bool ok, const std::string &name) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
  (ok ? passed : failed)++;
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::vector<uint8_t> bytes;
  if (FILE *file = std::fopen(path.c_str(), "rb")) {
    uint8_t chunk[65536];
    size_t got;
    while ((got = std::fread(chunk, 1, sizeof chunk, file)) > 0)
      bytes.insert(bytes.end(), chunk, chunk + got);
    std::fclose(file);
  }
  return bytes;
}

// True when parse throws ModelError (and nothing else).
bool rejects(const std::vector<uint8_t> &bytes) {
  try {
    anomaly_model::parse(bytes);
    return false;
  } catch (const anomaly_model::ModelError &) {
    return true;
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_host <asset cache directory>\n");
    return 2;
  }
  const std::string cache = argv[1];
  using anomaly_assets::sha256_hex;

  // FIPS 180-4 example vectors, including a two-block padding case.
  const std::string abc = "abc";
  const std::string two_block =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  expect(sha256_hex(nullptr, 0) ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "sha256-empty");
  expect(sha256_hex((const uint8_t *)abc.data(), abc.size()) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "sha256-abc");
  expect(sha256_hex((const uint8_t *)two_block.data(), two_block.size()) ==
             "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
         "sha256-448-bit");

  const std::vector<uint8_t> model_bytes =
      read_file(cache + "/mlperf-tiny-ad01-int8-model");
  const std::vector<uint8_t> dataset = read_file(cache + "/mlperf-tiny-ad01-dataset");
  expect(model_bytes.size() == anomaly_model::kModelBytes, "model-asset-present");
  expect(dataset.size() == anomaly_model::kDatasetBytes, "dataset-asset-present");
  if (failed) return 1;

  // The pinned model parses into the expected structure.
  anomaly_model::Model model;
  bool parsed = false;
  try {
    model = anomaly_model::parse(model_bytes);
    parsed = true;
  } catch (const anomaly_model::ModelError &error) {
    std::printf("    %s\n", error.what());
  }
  expect(parsed, "parse-pinned-model");
  if (!parsed) return 1;
  const uint32_t widths[] = {640, 128, 128, 128, 128, 8, 128, 128, 128, 128, 640};
  bool shape_ok = model.layers.size() == 10;
  for (size_t l = 0; shape_ok && l < 10; ++l) {
    const auto &layer = model.layers[l];
    shape_ok = layer.input_count == widths[l] && layer.output_count == widths[l + 1] &&
               layer.relu == (l < 9) && layer.weights.size() == widths[l] * widths[l + 1] &&
               layer.biases.size() == widths[l + 1] && layer.weight.zero_point == 0 &&
               layer.input_tensor == (l == 0 ? 0 : model.layers[l - 1].output_tensor);
  }
  expect(shape_ok && model.layers[9].output_tensor == 30, "model-structure");

  // Corrupted copies are rejected with ModelError.
  auto mutated = [&](const std::function<void(std::vector<uint8_t> &)> &edit) {
    std::vector<uint8_t> copy = model_bytes;
    edit(copy);
    return copy;
  };
  expect(rejects(mutated([](auto &b) { b[4] = 'X'; })), "reject-identifier");
  expect(rejects(mutated([](auto &b) { b.pop_back(); })), "reject-size");
  expect(rejects(mutated([](auto &b) { b[0] = b[1] = b[2] = 0xff; })),
         "reject-root-offset");
  expect(rejects(mutated([](auto &b) { std::memset(b.data(), 0, 4); })),
         "reject-zero-root");
  // The version field: the root table's first field (version 3 -> 2).
  {
    std::vector<uint8_t> copy = model_bytes;
    const uint32_t root = copy[0] | copy[1] << 8 | copy[2] << 16 | (uint32_t)copy[3] << 24;
    const int32_t to_vtable = (int32_t)(copy[root] | copy[root + 1] << 8 |
                                        copy[root + 2] << 16 | (uint32_t)copy[root + 3] << 24);
    const uint32_t vtable = root - to_vtable;
    const uint32_t version_at = root + (copy[vtable + 4] | copy[vtable + 5] << 8);
    const bool was_three = copy[version_at] == 3;
    copy[version_at] = 2;
    expect(was_three && rejects(copy), "reject-schema-version");
  }

  // Robustness: single-byte corruptions anywhere either parse or raise
  // ModelError; none crashes or escapes the buffer.
  size_t tried = 0, rejected = 0;
  bool robust = true;
  for (size_t at = 0; at < model_bytes.size(); at += (at < 8192 || at + 8192 > model_bytes.size()) ? 1 : 61) {
    std::vector<uint8_t> copy = model_bytes;
    copy[at] ^= 0x5b;
    try {
      anomaly_model::parse(copy);
    } catch (const anomaly_model::ModelError &) {
      ++rejected;
    } catch (...) {
      robust = false;
    }
    ++tried;
  }
  std::printf("    %zu single-byte corruptions, %zu rejected\n", tried, rejected);
  expect(robust && rejected > 0, "corruption-robustness");

  // The oracle reproduces upstream's cached smoke reference (dataset sample 0)
  // byte for byte, and rejects NaN.
  const std::vector<int8_t> outputs = anomaly_oracle::infer_dataset(model, dataset);
  {
    std::vector<std::vector<uint8_t>> expected;
    std::string error;
    const bool parsed_reference = anomaly_oracle::parse_expected(
        read_file(cache + "/anomaly-upstream-expected-smoke"), 1, expected, error);
    if (!parsed_reference) std::printf("    %s\n", error.c_str());
    expect(parsed_reference && expected[0].size() == anomaly_model::kSampleElements &&
               std::memcmp(expected[0].data(), outputs.data(),
                           anomaly_model::kSampleElements) == 0,
           "oracle-sample-0-matches-smoke-reference");
  }
  {
    std::vector<uint8_t> copy = dataset;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(copy.data() + 4 * 1000, &nan, 4);
    bool threw = false;
    try {
      anomaly_oracle::infer_dataset(model, copy);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    expect(threw, "oracle-rejects-non-finite");
  }

  // Reference-file parser: format errors are reported.
  std::vector<std::vector<uint8_t>> rows;
  std::string error;
  const std::string bad = "measured_sample=1 output=00\n";
  expect(!anomaly_oracle::parse_expected(std::vector<uint8_t>(bad.begin(), bad.end()),
                                         1, rows, error),
         "reference-parser-rejects-bad-line");

  std::printf("anomaly host tests: %d passed, %d failed\n", passed, failed);
  return failed ? 1 : 0;
}
