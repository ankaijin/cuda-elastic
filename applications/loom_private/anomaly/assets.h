// Asset loading for loom-private/mlperf-tiny-anomaly-model: exact-size reads
// and SHA-256 verification. Host only; no CUDA.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace anomaly_assets {

// Lowercase hex SHA-256 (FIPS 180-4) of a byte range.
std::string sha256_hex(const uint8_t *data, size_t size);

// Reads `path` and checks that it has exactly `expected_size` bytes whose
// SHA-256 is `expected_sha256`. On failure returns false and sets `error` to a
// message naming the asset, the path, and what was wrong (missing, wrong size,
// unreadable, or wrong hash).
bool load_verified(const std::string &name, const std::string &path,
                   size_t expected_size, const std::string &expected_sha256,
                   std::vector<uint8_t> &bytes, std::string &error);

} // namespace anomaly_assets
