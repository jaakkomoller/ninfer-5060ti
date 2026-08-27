#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer {

enum class DType : std::uint8_t {
    BF16       = 0,
    FP32       = 1,
    I32        = 2,
    U8         = 3,
    I64        = 4,
    I8         = 5,
    FP16       = 6,
    FP8_E4M3FN = 7,
    // Logical KV-format marker only: a paged-KV cache view whose code planes are packed 4-bit
    // codes (two per byte) with per-group FP16 scales. Code planes are stored as U8 with a
    // leading extent of head_dim/2; I4 is never a storage element type, so dtype_size(I4)
    // throws to keep it out of byte-sized tensor math.
    I4 = 8,
};

std::size_t dtype_size(DType dtype);

} // namespace ninfer
