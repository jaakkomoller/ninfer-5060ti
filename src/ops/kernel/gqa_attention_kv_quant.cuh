#pragma once

// ninfer::ops - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// INT4-G64 code-plane index. The code plane is a U8 tensor whose leading extent is head_dim/2
// (two signed 4-bit codes per byte), so the byte holding dimension d_even sits at leading index
// d_even/2. d_even must be even; the odd partner (d_even+1) shares the same byte.
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index_i4(int physical_page, int kv_head,
                                                                   int d_even, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim / 2, Geometry::KVHeads>(physical_page, kv_head,
                                                                              page_offset, d_even / 2);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Quantize one FP32 value to a signed 4-bit code (stored as 0..15 two's complement) given a
// precomputed 1/scale (scale is the FP16-rounded per-group absmax/7). Round-to-nearest-even +
// symmetric clamp to [-8, 7] keeps codes bit-identical to the CPU oracle and to the I8 path's
// quantize convention.
__device__ __forceinline__ unsigned gqa_kv_quant_code_i4(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0u; }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-8, min(7, q));
    return static_cast<unsigned>(q) & 0xFu;
}

// Pack one even-dimension code (low nibble) and one odd-dimension code (high nibble) into one
// byte. Both inputs are 4-bit codes in 0..15.
__device__ __forceinline__ unsigned char gqa_kv_pack_i4(unsigned code_even, unsigned code_odd) {
    return static_cast<unsigned char>(((code_odd & 0xFu) << 4) | (code_even & 0xFu));
}

// Sign-extend one 4-bit code (0..15) to its signed two's-complement value (-8..7).
__device__ __forceinline__ int gqa_kv_i4_sign_extend(unsigned nibble) {
    return static_cast<int>((nibble & 0xFu) ^ 8u) - 8;
}

// Unpack 8 packed bytes (16 signed 4-bit codes, dims [d, d+16), d a multiple of 16 and even) into
// 16 sign-extended int8 codes in dimension order, returned as an int4. Low nibble = even dim, high
// nibble = odd dim. This feeds the s8 QK MMA tile, which expects one signed int8 per dimension.
__device__ __forceinline__ int4 gqa_kv_unpack_i4x16_from(const unsigned char* packed8) {
    const std::uint64_t raw = load_vec<std::uint64_t>(packed8);
    const unsigned char* p  = reinterpret_cast<const unsigned char*>(&raw);
    int8_t codes[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        codes[2 * j]     = static_cast<int8_t>(gqa_kv_i4_sign_extend(p[j] & 0x0Fu));
        codes[2 * j + 1] = static_cast<int8_t>(gqa_kv_i4_sign_extend(p[j] >> 4));
    }
    auto word = [&codes](int a, int b, int c, int d) {
        return static_cast<int>(static_cast<unsigned char>(codes[a]) |
                                (static_cast<unsigned>(static_cast<unsigned char>(codes[b])) << 8) |
                                (static_cast<unsigned>(static_cast<unsigned char>(codes[c])) << 16) |
                                (static_cast<unsigned>(static_cast<unsigned char>(codes[d])) << 24));
    };
    return make_int4(word(0, 1, 2, 3), word(4, 5, 6, 7), word(8, 9, 10, 11), word(12, 13, 14, 15));
}

// Dequantize 8 consecutive signed 4-bit codes (dims [d, d+8), a multiple of 8 and even) given a
// pointer to their 4 packed bytes and the group's dequant scale. Returns 8 bf16 packed as an int4.
// The pointer may be in global or shared memory: the packed bytes are streamed via cp.async into
// smem (decode) or read directly from the cache (prefill), so the ALU is identical either way.
__device__ __forceinline__ int4 gqa_kv_dequant_i4x8_from(const unsigned char* packed4, float s) {
    const unsigned raw = load_vec<unsigned>(packed4);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c0 = gqa_kv_i4_sign_extend(p[i] & 0x0Fu);
        const int c1 = gqa_kv_i4_sign_extend(p[i] >> 4);
        const float x0 = static_cast<float>(c0) * s;
        const float x1 = static_cast<float>(c1) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops
