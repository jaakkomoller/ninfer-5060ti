#pragma once

// Q3G64 RowSplit storage atoms.
//
// One group holds 64 signed codes q in [-4, 3] packed little-endian at 3 bits
// per code: code i occupies bit range [3i, 3i + 3) of the 192-bit (24-byte)
// group word. The stored unsigned code is the offset encoding u = (q + 4) & 7,
// so decode is q = ((word >> (3i)) & 7) - 4. One FP16 scale follows every group.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q3RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 24;
    static constexpr int kScaleBytesPerGroup = 2;
    static constexpr int kCodesPerWord       = 8;
    static constexpr int kWordsPerGroup      = kGroupK / kCodesPerWord;
};

// Loads the 24-bit word containing codes 8*word .. 8*word + 7 from the packed
// group plane. The word is unaligned within the group, so the three bytes are
// read individually.
__device__ __forceinline__ std::uint32_t q3_load_word24(const std::uint8_t* group_base, int word) {
    const std::uint8_t* p = group_base + 3 * word;
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16);
}

struct Q3SimtDecodeAtom {
    // Decodes one 24-bit word (eight consecutive codes) into scaled FP32 weights.
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t word, std::uint16_t scale_bits, float (&weights)[8]) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int code = static_cast<int>(word >> (3 * i)) & 0x7u;
            weights[i] = static_cast<float>(code - 4) * scale;
        }
    }
};

struct Q3MmaDecodeAtom {
    // Lane `lane` decodes codes 2*lane and 2*lane + 1 of one 24-byte group word
    // (bit ranges [6*lane, 6*lane + 2) and [6*lane + 3, 6*lane + 5)) and returns
    // the scaled BF16 pair. A 3-bit code straddles two bytes only when its bit
    // offset within the byte is 6 or 7.
    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                const std::uint8_t* scale_ptr,
                                                                std::int64_t group_index,
                                                                int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t* group_base =
            codes + group_index * Q3RowSplitStorage::kCodeBytesPerGroup;

        const int bit0   = 6 * lane;
        const int byte0  = bit0 >> 3;
        const int shift0 = bit0 & 7;
        std::uint32_t word0 = static_cast<std::uint32_t>(group_base[byte0]);
        if (shift0 >= 6) { word0 |= static_cast<std::uint32_t>(group_base[byte0 + 1]) << 8; }
        const int code0 = static_cast<int>(word0 >> shift0) & 0x7u;

        const int bit1   = bit0 + 3;
        const int byte1  = bit1 >> 3;
        const int shift1 = bit1 & 7;
        std::uint32_t word1 = static_cast<std::uint32_t>(group_base[byte1]);
        if (shift1 >= 6) { word1 |= static_cast<std::uint32_t>(group_base[byte1 + 1]) << 8; }
        const int code1 = static_cast<int>(word1 >> shift1) & 0x7u;

        return __floats2bfloat162_rn(static_cast<float>(code0 - 4) * scale,
                                     static_cast<float>(code1 - 4) * scale);
    }
};

} // namespace ninfer::ops::detail