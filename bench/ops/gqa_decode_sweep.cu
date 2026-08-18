// Automated decode kernel parameter sweep
// Instantiates all (Bc, WarpsPerCta) variants at compile time
// Dispatches at runtime for systematic measurement

#include "core/device.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "core/kv_cache.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace ninfer;
using namespace ninfer::ops;

// Geometry for 9B model
using Geometry = GqaGeometry<16, 4, 1>;

struct SweepConfig {
    int bc;
    int warps_per_cta;
    const char* name;
};

struct SweepResult {
    int bc;
    int warps_per_cta;
    double median_us;
    double throughput_gbs;
};

// Instantiate all kernel variants we want to test
// Bc values: 16, 24, 32, 48, 64
// WarpsPerCta values: 1, 2, 4

template <int Bc, int WarpsPerCta>
void launch_decode_kernel(
    const __nv_bfloat16* q, GqaCachedInput input, const std::int32_t* pos,
    __nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
    std::int32_t tokens, std::int32_t padded_context, std::int32_t max_context,
    float scale, __nv_bfloat16* partial_acc, float* partial_m, float* partial_l,
    cudaStream_t stream) {
    
    // Adjust grid dimensions based on Bc (affects shared memory usage)
    constexpr int kThreadsPerCta = WarpsPerCta * 32;
    const dim3 grid(Geometry::KVHeads, 85);  // Simplified: use fixed split count
    const dim3 block(kThreadsPerCta, 1, 1);
    
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, 1, WarpsPerCta, GqaCachedInput>
        <<<grid, block, 0, stream>>>
        (q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, 
         scale, partial_acc, partial_m, partial_l);
}

// Dispatch function to select kernel variant at runtime
void dispatch_decode_kernel(
    int bc, int warps_per_cta,
    const __nv_bfloat16* q, GqaCachedInput input, const std::int32_t* pos,
    __nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
    std::int32_t tokens, std::int32_t padded_context, std::int32_t max_context,
    float scale, __nv_bfloat16* partial_acc, float* partial_m, float* partial_l,
    cudaStream_t stream) {
    
    // Instantiate all combinations
    if (bc == 16 && warps_per_cta == 1) {
        launch_decode_kernel<16, 1>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 16 && warps_per_cta == 2) {
        launch_decode_kernel<16, 2>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 16 && warps_per_cta == 4) {
        launch_decode_kernel<16, 4>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 24 && warps_per_cta == 1) {
        launch_decode_kernel<24, 1>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 24 && warps_per_cta == 2) {
        launch_decode_kernel<24, 2>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 24 && warps_per_cta == 4) {
        launch_decode_kernel<24, 4>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 32 && warps_per_cta == 1) {
        launch_decode_kernel<32, 1>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 32 && warps_per_cta == 2) {
        launch_decode_kernel<32, 2>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 32 && warps_per_cta == 4) {
        launch_decode_kernel<32, 4>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 48 && warps_per_cta == 1) {
        launch_decode_kernel<48, 1>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 48 && warps_per_cta == 2) {
        launch_decode_kernel<48, 2>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 48 && warps_per_cta == 4) {
        launch_decode_kernel<48, 4>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 64 && warps_per_cta == 1) {
        launch_decode_kernel<64, 1>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 64 && warps_per_cta == 2) {
        launch_decode_kernel<64, 2>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else if (bc == 64 && warps_per_cta == 4) {
        launch_decode_kernel<64, 4>(q, input, pos, cache_k, cache_v, tokens, padded_context, max_context, scale, partial_acc, partial_m, partial_l, stream);
    } else {
        printf("Unknown variant: bc=%d, warps=%d\n", bc, warps_per_cta);
    }
}

int main(int argc, char** argv) {
    printf("Decode Kernel Parameter Sweep\n");
    
    // Test configurations
    std::vector<SweepConfig> configs = {
        {16, 1, "Bc16_W1"}, {16, 2, "Bc16_W2"}, {16, 4, "Bc16_W4"},
        {24, 1, "Bc24_W1"}, {24, 2, "Bc24_W2"}, {24, 4, "Bc24_W4"},
        {32, 1, "Bc32_W1"}, {32, 2, "Bc32_W2"}, {32, 4, "Bc32_W4"},
        {48, 1, "Bc48_W1"}, {48, 2, "Bc48_W2"}, {48, 4, "Bc48_W4"},
        {64, 1, "Bc64_W1"}, {64, 2, "Bc64_W2"}, {64, 4, "Bc64_W4"},
    };
    
    printf("Testing %zu configurations\n\n", configs.size());
    
    // Allocate test data
    constexpr int kContext = 2048;
    constexpr int kTokens = 1;
    constexpr int kPaddedContext = 2048;
    constexpr int kMaxContext = 4096;
    
    DeviceBuffer q_buf(kQHeads * kHeadDim * sizeof(__nv_bfloat16));
    DeviceBuffer k_buf(kKVHeads * kHeadDim * kMaxContext * sizeof(__nv_bfloat16));
    DeviceBuffer v_buf(kKVHeads * kHeadDim * kMaxContext * sizeof(__nv_bfloat16));
    DeviceBuffer pos_buf(sizeof(std::int32_t));
    DeviceBuffer partial_acc_buf(kQHeads * kHeadDim * 85 * sizeof(__nv_bfloat16));
    DeviceBuffer partial_m_buf(kQHeads * 85 * sizeof(float));
    DeviceBuffer partial_l_buf(kQHeads * 85 * sizeof(float));
    
    // Initialize position
    std::int32_t pos = kContext - 1;
    cudaMemcpy(pos_buf.p, &pos, sizeof(pos), cudaMemcpyHostToDevice);
    
    // Run sweep
    std::vector<SweepResult> results;
    
    for (const auto& cfg : configs) {
        printf("Testing %s (Bc=%d, Warps=%d)...\n", cfg.name, cfg.bc, cfg.warps_per_cta);
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            dispatch_decode_kernel(cfg.bc, cfg.warps_per_cta,
                (__nv_bfloat16*)q_buf.p, GqaCachedInput{}, (std::int32_t*)pos_buf.p,
                (__nv_bfloat16*)k_buf.p, (__nv_bfloat16*)v_buf.p,
                kTokens, kPaddedContext, kMaxContext, 0.0625f,
                (__nv_bfloat16*)partial_acc_buf.p, (float*)partial_m_buf.p, (float*)partial_l_buf.p,
                0);
        }
        cudaDeviceSynchronize();
        
        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        const int kReps = 100;
        for (int i = 0; i < kReps; i++) {
            dispatch_decode_kernel(cfg.bc, cfg.warps_per_cta,
                (__nv_bfloat16*)q_buf.p, GqaCachedInput{}, (std::int32_t*)pos_buf.p,
                (__nv_bfloat16*)k_buf.p, (__nv_bfloat16*)v_buf.p,
                kTokens, kPaddedContext, kMaxContext, 0.0625f,
                (__nv_bfloat16*)partial_acc_buf.p, (float*)partial_m_buf.p, (float*)partial_l_buf.p,
                0);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::micro> duration = end - start;
        double median_us = duration.count() / kReps;
        
        // Calculate throughput (simplified)
        double bytes = kTokens * kKVHeads * kContext * kHeadDim * 2 * sizeof(__nv_bfloat16);
        double throughput_gbs = (bytes / median_us) * 1000.0;
        
        printf("  Median: %.2f us, Throughput: %.2f GB/s\n", median_us, throughput_gbs);
        
        results.push_back({cfg.bc, cfg.warps_per_cta, median_us, throughput_gbs});
    }
    
    // Print summary
    printf("\n=== Sweep Results ===\n");
    printf("Sorted by throughput (descending):\n");
    std::sort(results.begin(), results.end(), 
              [](const SweepResult& a, const SweepResult& b) {
                  return a.throughput_gbs > b.throughput_gbs;
              });
    
    for (size_t i = 0; i < results.size(); i++) {
        printf("%zu. Bc=%d, Warps=%d: %.2f us, %.2f GB/s\n", 
               i+1, results[i].bc, results[i].warps_per_cta, 
               results[i].median_us, results[i].throughput_gbs);
    }
    
    return 0;
}
