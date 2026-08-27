#pragma once

// INT4-native GQA prompt kernel for the registered Qwen3.6 head geometries. Mirrors the INT8
// prompt kernel (gqa_attention_prefill_i8.cuh) with packed 4-bit K/V codes:
//   * QK stays INT8 through m16n8k32.s8 Tensor Cores. Packed K is staged via cp.async into a
//     packed arena, unpacked to a sign-extended int8 tile, then the s8 QK + per-64-group rescale
//     run unchanged.
//   * V is dequantized with packed FP16 arithmetic (unpack + scale) to an FP16 tile while producer
//     warps execute QK. Sixteen warps split each 16-row FP16 PV output across four 64-dimension
//     slices, exactly as INT8.
//   * The smem arena is the same size as INT8: k_packed (Bc*D/2) + k_i8 (Bc*D) + v_packed (Bc*D/2)
//     + v_f16 (2*Bc*D) = 4*Bc*D for the K/V staging. DRAM traffic for K and V is halved.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillI4Warps      = 16;
inline constexpr int kGqaPrefillI4Threads    = kGqaPrefillI4Warps * 32;
inline constexpr int kGqaPrefillI4Br         = 64;
inline constexpr int kGqaPrefillI4Bc         = 64;
inline constexpr int kGqaPrefillI4Groups     = kGqaPrefillHeadDim / kGqaKvQuantGroup;
inline constexpr int kGqaPrefillI4DB16       = kGqaPrefillHeadDim / 2;
inline constexpr int kGqaPrefillI4RowTiles   = kGqaPrefillI4Br / 16;
inline constexpr int kGqaPrefillI4DConsumers = kGqaPrefillI4Warps / kGqaPrefillI4RowTiles;

inline constexpr int kGqaPrefillI4QBytes = kGqaPrefillI4Br * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI4QScaleBytes =
    kGqaPrefillI4Br * kGqaPrefillI4Groups * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillI4KPackedBytes = kGqaPrefillI4Bc * kGqaPrefillI4DB16;
inline constexpr int kGqaPrefillI4KBytes = kGqaPrefillI4Bc * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI4VPackedBytes = kGqaPrefillI4Bc * kGqaPrefillI4DB16;
inline constexpr int kGqaPrefillI4VStageBytes =
    kGqaPrefillI4Bc * kGqaPrefillHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI4PBytes =
    kGqaPrefillI4Br * kGqaPrefillI4Bc * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI4ScaleBytes =
    2 * kGqaPrefillI4Bc * kGqaPrefillI4Groups * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI4StatsBytes = 2 * kGqaPrefillI4Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillI4SmemBytes = kGqaPrefillI4QBytes + kGqaPrefillI4QScaleBytes +
                                              kGqaPrefillI4KPackedBytes + kGqaPrefillI4KBytes +
                                              kGqaPrefillI4VPackedBytes + kGqaPrefillI4VStageBytes +
                                              kGqaPrefillI4PBytes + kGqaPrefillI4ScaleBytes +
                                              kGqaPrefillI4StatsBytes;

static_assert(kGqaPrefillI4Groups == 4);
static_assert(kGqaPrefillI4DConsumers == 4);
static_assert(kGqaPrefillI4SmemBytes == 92672);

__device__ __forceinline__ void gqa_prefill_i4_store_swz(std::int8_t* tile, int row, int d,
                                                         std::int8_t code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * kGqaPrefillI4DB16 + gqa_prefill_swz(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

__device__ __forceinline__ int gqa_prefill_i4_p_swz(int row, int col) {
    if constexpr (kGqaPrefillI4Bc == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return gqa_prefill_swz(row, col);
}

// Dequantize 8 consecutive signed 4-bit codes (dims [d, d+8), even and a multiple of 8) given a
// pointer to their 4 packed bytes and the group's FP16 dequant scale. Returns 8 fp16 packed as an
// int4. Mirrors the INT8 prefill dequant but reads packed codes.
__device__ __forceinline__ int4 gqa_prefill_i4_dequant_f16x8(const unsigned char* packed4,
                                                             __half scale) {
    const unsigned raw  = load_vec<unsigned>(packed4);
    const unsigned char* c = reinterpret_cast<const unsigned char*>(&raw);
    const __half2 s2     = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int code0 = gqa_kv_i4_sign_extend(c[i] & 0x0Fu);
        const int code1 = gqa_kv_i4_sign_extend(c[i] >> 4);
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(code0), static_cast<float>(code1));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Eight independent quantization units per CTA; one warp owns one
// (token, kv_head, 64-d group), with two adjacent dimensions per lane (packed into one byte).
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_i4_kernel(const __nv_bfloat16* __restrict__ k,
                                              const __nv_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, unsigned char* __restrict__ cache_k,
                                              unsigned char* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaPrefillI4Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaPrefillI4Groups;
    const int tmp                   = unit / kGqaPrefillI4Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    const int d0                    = group * kGqaKvQuantGroup + 2 * lane;
    const int d1                    = d0 + 1;

    const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
    const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, d1, token);
    const float k0          = __bfloat162float(k[src0]);
    const float k1          = __bfloat162float(k[src1]);
    const float v0          = __bfloat162float(v[src0]);
    const float v1          = __bfloat162float(v[src1]);

    float k_abs = fmaxf(fabsf(k0), fabsf(k1));
    float v_abs = fmaxf(fabsf(v0), fabsf(v1));
    k_abs       = warp_max(k_abs, FullMask);
    v_abs       = warp_max(v_abs, FullMask);

    const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 7.0f : 0.0f);
    const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / 7.0f : 0.0f);
    const float ks   = __half2float(ksh);
    const float vs   = __half2float(vsh);
    const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
    page             = __shfl_sync(FullMask, page, 0);

    const std::int64_t code_byte =
        gqa_kv_quant_code_index_i4<Geometry>(page, kv_head, d0, page_off);
    cache_k[code_byte] =
        gqa_kv_pack_i4(gqa_kv_quant_code_i4(k0, kinv), gqa_kv_quant_code_i4(k1, kinv));
    cache_v[code_byte] =
        gqa_kv_pack_i4(gqa_kv_quant_code_i4(v0, vinv), gqa_kv_quant_code_i4(v1, vinv));
    if (lane == 0) {
        const std::int64_t scale_off =
            gqa_kv_quant_scale_index<Geometry>(page, kv_head, group, page_off);
        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

// Large appends are scheduled in absolute eight-token tiles. Eight divides P=64, so each CTA is
// page-local while an unknown base offset costs at most one empty tail CTA in the launch envelope.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_i4_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    unsigned char* __restrict__ cache_k, unsigned char* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int group             = static_cast<int>(blockIdx.z);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) { return; }

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;

    const int token  = token_begin + warp;
    const bool valid = token < token_end;
    const int d0     = group * kGqaKvQuantGroup + 2 * lane;
    const int d1     = d0 + 1;
    float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, d1, token);
        k0                      = __bfloat162float(k[src0]);
        k1                      = __bfloat162float(k[src1]);
        v0                      = __bfloat162float(v[src0]);
        v1                      = __bfloat162float(v[src1]);
    }
    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    const __half ksh  = __float2half_rn(k_abs > 0.0f ? k_abs / 7.0f : 0.0f);
    const __half vsh  = __float2half_rn(v_abs > 0.0f ? v_abs / 7.0f : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv  = vs > 0.0f ? 1.0f / vs : 0.0f;
    physical_page     = __shfl_sync(FullMask, physical_page, 0);
    if (!valid) { return; }

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;
    const std::int64_t code_byte =
        gqa_kv_quant_code_index_i4<Geometry>(physical_page, kv_head, d0, page_off);
    cache_k[code_byte] =
        gqa_kv_pack_i4(gqa_kv_quant_code_i4(k0, kinv), gqa_kv_quant_code_i4(k1, kinv));
    cache_v[code_byte] =
        gqa_kv_pack_i4(gqa_kv_quant_code_i4(v0, vinv), gqa_kv_quant_code_i4(v1, vinv));
    if (lane == 0) {
        const std::int64_t scale_offset =
            gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, group, page_off);
        scale_k[scale_offset] = ksh;
        scale_v[scale_offset] = vsh;
    }
}

template <typename Geometry, typename Metadata>
__global__ __maxnreg__(120) void gqa_attention_prefill_i4_kernel(
    const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ cache_k,
    const unsigned char* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillI4Br;
    constexpr int Bc            = kGqaPrefillI4Bc;
    constexpr int DB16          = kGqaPrefillI4DB16;
    constexpr int Groups        = kGqaPrefillI4Groups;
    constexpr int GroupKc       = kGqaKvQuantGroup / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kGqaPrefillI4DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kGqaPrefillI4RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillI4Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kGqaPrefillI4QBytes);
    std::int8_t* k_packed_p =
        reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                       kGqaPrefillI4QScaleBytes);
    std::int8_t* k_i8 = k_packed_p + kGqaPrefillI4KPackedBytes;
    std::int8_t* v_packed_p = k_i8 + kGqaPrefillI4KBytes;
    __half* v_f16           = reinterpret_cast<__half*>(v_packed_p + kGqaPrefillI4VPackedBytes);
    __half* p_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) + kGqaPrefillI4VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kGqaPrefillI4PBytes);
    __half* v_scale_s   = k_scale_s + Bc * Groups;
    float* alpha_s      = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s    = alpha_s + Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kGqaPrefillI4Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One warp owns one (row, 64-d group) at a time.
    for (int unit = warp; unit < Br * Groups; unit += kGqaPrefillI4Warps) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroup + lane;
        const int d1  = d0 + 32;
        float x0      = 0.0f;
        float x1      = 0.0f;
        if (row < tile_rows) {
            x0 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row)]);
            x1 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d1, q0 + row)]);
        }
        float absmax    = fmaxf(fabsf(x0), fabsf(x1));
        absmax          = warp_max(absmax, FullMask);
        const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_prefill_i4_store_swz(q_i8, row, d0, gqa_kv_quant_code(x0, inv));
        gqa_prefill_i4_store_swz(q_i8, row, d1, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale[row * Groups + grp] = qs; }
    }
    __syncthreads();

    // Stream one 64-key tile of packed K/V codes and scales into the staging arena. Packed bytes
    // are half the width of INT8, so each 16-dimension chunk is an 8-byte cp.async.
    auto issue_kv_packed = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kGqaPrefillI4Threads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * Groups];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
                store_vec(vd, make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += kGqaPrefillI4Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            std::int8_t* kd = &k_packed_p[key_l * DB16 + dc * 8];
            std::int8_t* vd = &v_packed_p[key_l * DB16 + dc * 8];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_kv_quant_code_index_i4<Geometry>(physical_page, kv_head, d, key_l);
                cp_async<8>(kd, &cache_k[off]);
                cp_async<8>(vd, &cache_v[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
                store_vec(vd, make_int2(0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    // Unpack one staged tile of packed K (8 packed bytes -> 16 sign-extended int8) into the
    // swizzled s8 MMA tile. Runs on every warp before the producer's QK reads k_i8.
    auto unpack_k_packed = [&]() {
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += kGqaPrefillI4Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const std::int8_t* src = &k_packed_p[key_l * DB16 + dc * 8];
            std::int8_t* dst         = &k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, dc * 8)) * 2];
            store_vec(dst, gqa_kv_unpack_i4x16_from(reinterpret_cast<const unsigned char*>(src)));
        }
    };

    issue_kv_packed(0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Keeping exactly two group scales live is the spill-free 120-register point on SM120.
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
    if (warp < ProducerWarps) {
        const int scale_row0 = warp * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
#pragma unroll
        for (int grp = 0; grp < Groups - 2; ++grp) {
            float qs0       = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
            float qs1       = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
            q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;

        unpack_k_packed();
        __syncthreads();

        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    const int scale_row0 = row_base + gid;
                    const int scale_row1 = scale_row0 + 8;
                    qs0                  = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
                    qs1                  = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
                    qs0                  = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1                  = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 gqa_prefill_swz(row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + gqa_prefill_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ks0      = 0.0f;
                    float ks1      = 0.0f;
                    if (gid == 0) {
                        ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                        ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    }
                    ks0          = __shfl_sync(FullMask, ks0, lid);
                    ks1          = __shfl_sync(FullMask, ks1, lid);
                    score[nt][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                    score[nt][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                    score[nt][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                    score[nt][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + gqa_prefill_i4_p_swz(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + gqa_prefill_i4_p_swz(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + gqa_prefill_i4_p_swz(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + gqa_prefill_i4_p_swz(row1, col1)] = __float2half_rn(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + gqa_prefill_swz(key_l, d)];
                if (key <= max_query_abs) {
                    const int grp = d >> 6;
                    __half vs     = __float2half_rn(0.0f);
                    if ((lane & 7) == 0) { vs = v_scale_s[key_l * Groups + grp]; }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    store_vec(dst, gqa_prefill_i4_dequant_f16x8(
                                        reinterpret_cast<const unsigned char*>(
                                            &v_packed_p[key_l * DB16 + d / 2]),
                                        vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_packed((kb + 1) * Bc); }

        const int row_tile = warp % kGqaPrefillI4RowTiles;
        const int d_slice  = warp / kGqaPrefillI4RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       gqa_prefill_i4_p_swz(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + gqa_prefill_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kGqaPrefillI4RowTiles;
    const int d_slice  = warp / kGqaPrefillI4RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           kGqaPrefillI4Threads);
}

} // namespace ninfer::ops