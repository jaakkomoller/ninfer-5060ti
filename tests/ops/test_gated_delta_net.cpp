#include "ninfer/ops/gated_delta_net.h"

#include "ops/gdn_ref.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim = 128;

constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

// I8 state quantization budget. The pool carries the recurrence state through per-row
// 128-level quantization and re-pays that round trip at every kernel stage boundary (full
// chunk or tail). The per-token decay damps every earlier boundary, so the budget against the
// staged FP64 reference is flat in T and is dominated by the final round trip plus one
// intermediate. The absolute term is disabled: state magnitude is not bounded, so the gross
// error is judged against the reference max.
constexpr ReductionCriterion gated_delta_net_i8_state_criterion() {
    return {/*relative_l2=*/2.0e-2, /*gross_absolute=*/0.0,
            /*gross_relative_to_max_reference=*/2.0e-2};
}

// Pins kernel-vs-oracle codec agreement against the staged ideal pool: the kernel pool must
// equal the staged reference round-tripped through the same FP32 row-max / FP16 scale / RNE
// code quantizer, up to the FP32 recurrence profile and isolated code/scale boundary flips
// (one step = row_max/127). A wrong rounding mode or a codec drift between kernel and oracle
// exceeds relative_l2 by an order of magnitude. Valid only where the kernel carries the
// FP64-oracle recurrence profile at ~1e-6 (recurrent route); the chunked route's staging
// profile sets a ~0.3% floor and uses the chunked criterion below.
constexpr ReductionCriterion gated_delta_net_i8_state_codec_criterion() {
    return {/*relative_l2=*/1.0e-3, /*gross_absolute=*/0.0,
            /*gross_relative_to_max_reference=*/1.5e-2};
}

// Chunked-route codec agreement. The chunked kernel stages its state matmuls through TF32/BF16
// (measured ~0.3% relative against the FP64 oracle on the FP32-state route, which the FP32
// state criterion carries at ratio ~0.95), so the I8 pool inherits that floor. This criterion
// catches codec drift an order of magnitude above the floor; the single-chunk paired-FP32
// check pins the Phase-Z codec exactly where no intermediate pool round trip exists.
constexpr ReductionCriterion gated_delta_net_i8_state_codec_chunked_criterion() {
    return {/*relative_l2=*/1.5e-2, /*gross_absolute=*/0.0,
            /*gross_relative_to_max_reference=*/3.0e-2};
}

// I8 out: the staged oracle removes the pool round trip from the stage seeds, so the error is
// the kernel's own profile plus the quantized stage seeds' effect on later-stage outputs.
constexpr ReductionCriterion gated_delta_net_i8_output_criterion() {
    return {/*relative_l2=*/4.5e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/7.0e-3};
}

// Paired-check budget: the dequantized I8 pool versus the kernel's own FP32 state from a
// paired run on identical inputs (the final quantize/dequantize round trip, no kernel profile).
constexpr ReductionCriterion gated_delta_net_i8_paired_state_criterion() {
    return {/*relative_l2=*/1.0e-2, /*gross_absolute=*/0.0,
            /*gross_relative_to_max_reference=*/1.5e-2};
}

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

// Ideal I8 pool content for an FP64 reference state: quantize the FP32 view with the same
// per-row codec, dequantize. The kernel pool must match this up to its FP32 recurrence error.
std::vector<double> ideal_i8_round_trip(const std::vector<double>& reference,
                                        std::int64_t value_heads) {
    const std::vector<float> f32(reference.begin(), reference.end());
    const gdn_ref::I8StateQuantization pool =
        gdn_ref::i8_quantize_state(f32, kStateDim, value_heads);
    return doubles(gdn_ref::i8_dequantize_state(pool.codes, pool.scale, kStateDim, value_heads));
}

// Mirrors the two-stage `gated_delta_net` dispatch: full kChunkSize stages followed by a tail
// stage, with the pool round trip the kernel pays at every stage boundary applied through the
// host codec. `seeded` is the dequantized initial pool the kernel loads.
struct I8StagedReference {
    std::vector<double> out;
    std::vector<double> final_state;  // FP64 state after the last stage (no final quantization)
    std::vector<double> pool;         // dequantized ideal final pool
};

I8StagedReference staged_i8_reference(const gdn_ref::Inputs& in, double scale, bool normalize_qk,
                                      const std::vector<float>& seeded) {
    I8StagedReference result;
    result.out.resize(in.v.size());
    const std::int64_t qk_stride    = static_cast<std::int64_t>(kStateDim) * in.qk_heads;
    const std::int64_t value_stride = static_cast<std::int64_t>(kStateDim) * in.value_heads;
    const std::int64_t gate_stride  = in.value_heads;
    const std::int64_t chunk        = ops::detail::gated_delta_net::kChunkSize;
    std::vector<float> cur          = seeded;
    for (std::int64_t pos = 0; pos < in.tokens; pos += chunk) {
        const std::int64_t n = std::min<std::int64_t>(chunk, in.tokens - pos);
        gdn_ref::Inputs stage    = in;
        stage.tokens             = n;
        stage.q.assign(in.q.begin() + pos * qk_stride, in.q.begin() + (pos + n) * qk_stride);
        stage.k.assign(in.k.begin() + pos * qk_stride, in.k.begin() + (pos + n) * qk_stride);
        stage.v.assign(in.v.begin() + pos * value_stride, in.v.begin() + (pos + n) * value_stride);
        stage.g.assign(in.g.begin() + pos * gate_stride, in.g.begin() + (pos + n) * gate_stride);
        stage.beta.assign(in.beta.begin() + pos * gate_stride,
                          in.beta.begin() + (pos + n) * gate_stride);
        stage.state = cur;
        const gdn_ref::Result res = gdn_ref::evaluate(stage, scale, normalize_qk);
        std::copy(res.out.begin(), res.out.end(), result.out.begin() + pos * value_stride);
        result.final_state = res.final_state;
        const std::vector<float> stage_f32(res.final_state.begin(), res.final_state.end());
        const gdn_ref::I8StateQuantization pool =
            gdn_ref::i8_quantize_state(stage_f32, kStateDim, in.value_heads);
        const std::vector<float> pool_dequant =
            gdn_ref::i8_dequantize_state(pool.codes, pool.scale, kStateDim, in.value_heads);
        if (pos + n < in.tokens) {
            cur = pool_dequant;
        } else {
            result.pool = doubles(pool_dequant);
        }
    }
    return result;
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

int verify_recurrence(const std::string& label, const std::vector<double>& got,
                      const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), got, expected, criterion);
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state.data(), DType::FP32, {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, state_tensor,
                         Tensor{}, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += state.verify_guards((label + " state").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int distinct_state_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_in_tensor(state_in.data(), DType::FP32,
                           {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_out_tensor(state_out.data(), DType::FP32,
                            {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                         state_in_tensor, Tensor{}, state_out_tensor, Tensor{}, out_tensor,
                         nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct-state";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state_out.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += verify_exact(label + " state-in unchanged",
                             from_device<float>(state_in.data(), in.state.size()), in.state);
    failures += state_in.verify_guards((label + " state-in").c_str());
    failures += state_out.verify_guards((label + " state-out").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

// I8 persistent-state parity: the pool holds int8 codes plus a per-(value_head, dv-row) FP16
// scale. The staged FP64 oracle is seeded with the dequantized pool value (what the kernel
// loads) and re-pays the pool round trip at every kernel stage boundary; the kernel pool is
// checked against the staged FP64 state (quantization budget) and against the dequantized
// staged ideal pool (codec agreement).
int i8_inplace_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::I8StateQuantization initial_pool =
        gdn_ref::i8_quantize_state(in.state, kStateDim, test_case.value_heads);
    const std::vector<float> pool_state =
        gdn_ref::i8_dequantize_state(initial_pool.codes, initial_pool.scale, kStateDim,
                                     test_case.value_heads);
    const I8StagedReference ref =
        staged_i8_reference(in, static_cast<double>(scale), test_case.normalize_qk, pool_state);
    DeviceInputs device(in);
    GuardedDeviceBuffer state_codes(initial_pool.codes.size() * sizeof(std::int8_t));
    GuardedDeviceBuffer state_scale(initial_pool.scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_codes.copy_from_host(initial_pool.codes.data(), state_codes.bytes());
    state_scale.copy_from_host(initial_pool.scale.data(), state_scale.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state_codes.data(), DType::I8,
                        {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_scale_tensor(state_scale.data(), DType::FP16,
                              {kStateDim, test_case.value_heads, 1, 1});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, state_tensor,
                         state_scale_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = 0;
    const std::vector<std::int8_t> codes_after =
        from_device<std::int8_t>(state_codes.data(), in.state.size());
    const std::vector<std::uint16_t> scale_after =
        from_device<std::uint16_t>(state_scale.data(), initial_pool.scale.size());
    const std::vector<float> state_after =
        gdn_ref::i8_dequantize_state(codes_after, scale_after, kStateDim, test_case.value_heads);
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_i8_output_criterion());
    failures += verify_recurrence(label + " state", doubles(state_after), ref.final_state,
                                  gated_delta_net_i8_state_criterion());
    const ReductionCriterion codec_criterion =
        (test_case.tokens >= ops::detail::gated_delta_net::kChunkSize)
            ? gated_delta_net_i8_state_codec_chunked_criterion()
            : gated_delta_net_i8_state_codec_criterion();
    failures += verify_recurrence(label + " state codec", doubles(state_after), ref.pool,
                                  codec_criterion);
    // A single full-chunk run has no intermediate pool round trip, so the kernel's own FP32
    // state (paired run, identical inputs and seed) is the exact codec reference: the I8 pool
    // must be that state's quantization, up to codegen-level state differences (every code
    // within one step, every scale within one ulp) and the final round trip.
    if (test_case.tokens == static_cast<int>(ops::detail::gated_delta_net::kChunkSize)) {
        GuardedDeviceBuffer fp32_state(in.state.size() * sizeof(float));
        GuardedDeviceBuffer paired_out(in.v.size() * sizeof(std::uint16_t));
        fp32_state.copy_from_host(pool_state.data(), fp32_state.bytes());
        paired_out.fill(0xff);
        Tensor fp32_state_tensor(fp32_state.data(), DType::FP32,
                                 {kStateDim, kStateDim, test_case.value_heads});
        Tensor paired_out_tensor(paired_out.data(), DType::BF16,
                                 {kStateDim, test_case.value_heads, test_case.tokens});
        ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                             fp32_state_tensor, Tensor{}, fp32_state_tensor, Tensor{},
                             paired_out_tensor, nullptr);
        cuda_synchronize();
        const std::vector<float> fp32_state_after =
            from_device<float>(fp32_state.data(), in.state.size());
        const gdn_ref::I8StateQuantization ideal_pool =
            gdn_ref::i8_quantize_state(fp32_state_after, kStateDim, test_case.value_heads);
        std::size_t code_bad = 0;
        for (std::size_t i = 0; i < codes_after.size(); ++i) {
            const int diff = codes_after[i] - ideal_pool.codes[i];
            if (diff < -1 || diff > 1) { ++code_bad; }
        }
        std::size_t scale_bad = 0;
        for (std::size_t i = 0; i < scale_after.size(); ++i) {
            const int sa = static_cast<int>(scale_after[i]);
            const int sb = static_cast<int>(ideal_pool.scale[i]);
            const int diff = sa > sb ? sa - sb : sb - sa;
            if (diff > 1) { ++scale_bad; }
        }
        if (code_bad != 0 || scale_bad != 0) {
            std::cerr << label << " paired codec: " << code_bad << " codes off by more than one "
                      << "step, " << scale_bad << " scales off by more than one ulp\n";
            ++failures;
        }
        failures += verify_recurrence(label + " state vs paired fp32", doubles(state_after),
                                      doubles(fp32_state_after),
                                      gated_delta_net_i8_paired_state_criterion());
        failures += fp32_state.verify_guards((label + " paired fp32 state").c_str());
        failures += paired_out.verify_guards((label + " paired out").c_str());
    }
    failures += state_codes.verify_guards((label + " state codes").c_str());
    failures += state_scale.verify_guards((label + " state scale").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int i8_distinct_state_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::I8StateQuantization initial_pool =
        gdn_ref::i8_quantize_state(in.state, kStateDim, test_case.value_heads);
    const std::vector<float> pool_state =
        gdn_ref::i8_dequantize_state(initial_pool.codes, initial_pool.scale, kStateDim,
                                     test_case.value_heads);
    const I8StagedReference ref =
        staged_i8_reference(in, static_cast<double>(scale), test_case.normalize_qk, pool_state);
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in_codes(initial_pool.codes.size() * sizeof(std::int8_t));
    GuardedDeviceBuffer state_out_codes(initial_pool.codes.size() * sizeof(std::int8_t));
    GuardedDeviceBuffer state_in_scale(initial_pool.scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_out_scale(initial_pool.scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in_codes.copy_from_host(initial_pool.codes.data(), state_in_codes.bytes());
    state_out_codes.copy_from_host(initial_pool.codes.data(), state_out_codes.bytes());
    state_in_scale.copy_from_host(initial_pool.scale.data(), state_in_scale.bytes());
    state_out_scale.copy_from_host(initial_pool.scale.data(), state_out_scale.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_in_tensor(state_in_codes.data(), DType::I8,
                           {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_in_scale_tensor(state_in_scale.data(), DType::FP16,
                                 {kStateDim, test_case.value_heads, 1, 1});
    Tensor state_out_tensor(state_out_codes.data(), DType::I8,
                            {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_out_scale_tensor(state_out_scale.data(), DType::FP16,
                                  {kStateDim, test_case.value_heads, 1, 1});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                         state_in_tensor, state_in_scale_tensor, state_out_tensor,
                         state_out_scale_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct-state";
    int failures            = 0;
    const std::vector<std::int8_t> out_codes_after =
        from_device<std::int8_t>(state_out_codes.data(), in.state.size());
    const std::vector<std::uint16_t> out_scale_after =
        from_device<std::uint16_t>(state_out_scale.data(), initial_pool.scale.size());
    const std::vector<float> state_after =
        gdn_ref::i8_dequantize_state(out_codes_after, out_scale_after, kStateDim,
                                     test_case.value_heads);
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_i8_output_criterion());
    failures += verify_recurrence(label + " state", doubles(state_after), ref.final_state,
                                  gated_delta_net_i8_state_criterion());
    const ReductionCriterion codec_criterion =
        (test_case.tokens >= ops::detail::gated_delta_net::kChunkSize)
            ? gated_delta_net_i8_state_codec_chunked_criterion()
            : gated_delta_net_i8_state_codec_criterion();
    failures += verify_recurrence(label + " state codec", doubles(state_after), ref.pool,
                                  codec_criterion);
    failures += verify_exact(label + " state-in codes unchanged",
                             from_device<std::int8_t>(state_in_codes.data(), in.state.size()),
                             initial_pool.codes);
    failures += verify_exact(label + " state-in scale unchanged",
                             from_device<std::uint16_t>(state_in_scale.data(),
                                                        initial_pool.scale.size()),
                             initial_pool.scale);
    failures += state_in_codes.verify_guards((label + " state-in codes").c_str());
    failures += state_out_codes.verify_guards((label + " state-out codes").c_str());
    failures += state_in_scale.verify_guards((label + " state-in scale").c_str());
    failures += state_out_scale.verify_guards((label + " state-out scale").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int i8_snapshot_case(const Case& test_case, int slots, int initial_slot, int snapshot_base_slot,
                     std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t state_size = in.state.size();
    const gdn_ref::I8StateQuantization initial_pool =
        gdn_ref::i8_quantize_state(in.state, kStateDim, test_case.value_heads);
    const std::vector<float> pool_state =
        gdn_ref::i8_dequantize_state(initial_pool.codes, initial_pool.scale, kStateDim,
                                     test_case.value_heads);
    gdn_ref::Inputs oracle_input = in;
    oracle_input.state           = pool_state;
    const gdn_ref::Result ref =
        gdn_ref::evaluate(oracle_input, static_cast<double>(scale), test_case.normalize_qk, true);

    // Untouched slots hold a host-quantized nonzero pattern so byte-exactness is meaningful.
    const std::vector<float> pattern(state_size, 17.0f);
    const gdn_ref::I8StateQuantization pattern_pool =
        gdn_ref::i8_quantize_state(pattern, kStateDim, test_case.value_heads);

    std::vector<std::int8_t> pool_codes(state_size * static_cast<std::size_t>(slots));
    std::vector<std::uint16_t> pool_scale(kStateDim * test_case.value_heads * slots);
    const std::size_t scale_slot_elements =
        static_cast<std::size_t>(kStateDim * test_case.value_heads);
    for (int slot = 0; slot < slots; ++slot) {
        const auto& source = (slot == initial_slot) ? initial_pool : pattern_pool;
        std::copy(source.codes.begin(), source.codes.end(),
                  pool_codes.begin() + static_cast<std::size_t>(slot) * state_size);
        std::copy(source.scale.begin(), source.scale.end(),
                  pool_scale.begin() + static_cast<std::size_t>(slot) * scale_slot_elements);
    }

    DeviceInputs device(in);
    GuardedDeviceBuffer states_codes(pool_codes.size() * sizeof(std::int8_t));
    GuardedDeviceBuffer states_scale(pool_scale.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    states_codes.copy_from_host(pool_codes.data(), states_codes.bytes());
    states_scale.copy_from_host(pool_scale.data(), states_scale.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slot       = to_device_i32({initial_slot});
    DeviceBuffer device_snapshot_base_slot = to_device_i32({snapshot_base_slot});

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor states_tensor(states_codes.data(), DType::I8,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor states_scale_tensor(states_scale.data(), DType::FP16,
                               {kStateDim, test_case.value_heads, slots});
    Tensor initial_slot_tensor(device_initial_slot.p, DType::I32, {1});
    Tensor snapshot_base_slot_tensor(device_snapshot_base_slot.p, DType::I32, {1});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  states_scale_tensor, Tensor{}, initial_slot_tensor,
                                  snapshot_base_slot_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label             = std::string(test_case.name) + " snapshot";
    const std::vector<std::int8_t> codes_after =
        from_device<std::int8_t>(states_codes.data(), pool_codes.size());
    const std::vector<std::uint16_t> scale_after =
        from_device<std::uint16_t>(states_scale.data(), pool_scale.size());
    int failures = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_i8_output_criterion());
    for (int token = 0; token < test_case.tokens; ++token) {
        const int slot = snapshot_base_slot + token;
        std::vector<std::int8_t> slot_codes(codes_after.begin() +
                                            static_cast<std::size_t>(slot) * state_size,
                                            codes_after.begin() +
                                                static_cast<std::size_t>(slot + 1) * state_size);
        std::vector<std::uint16_t> slot_scale(
            scale_after.begin() + static_cast<std::size_t>(slot) * scale_slot_elements,
            scale_after.begin() + static_cast<std::size_t>(slot + 1) * scale_slot_elements);
        const std::vector<float> slot_state =
            gdn_ref::i8_dequantize_state(slot_codes, slot_scale, kStateDim,
                                         test_case.value_heads);
        const auto expected_begin =
            ref.snapshots.begin() + static_cast<std::size_t>(token) * state_size;
        const std::vector<double> expected_slice(expected_begin,
                                                 expected_begin +
                                                     static_cast<std::ptrdiff_t>(state_size));
        failures +=
            verify_recurrence(label + " snapshot slot " + std::to_string(slot),
                              doubles(slot_state), expected_slice,
                              gated_delta_net_i8_state_criterion());
        failures += verify_recurrence(label + " snapshot slot " + std::to_string(slot) + " codec",
                                      doubles(slot_state),
                                      ideal_i8_round_trip(expected_slice, test_case.value_heads),
                                      gated_delta_net_i8_state_codec_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (slot >= snapshot_base_slot && slot < snapshot_base_slot + test_case.tokens) {
            continue;
        }
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot) + " codes",
            std::vector<std::int8_t>(codes_after.begin() +
                                        static_cast<std::size_t>(slot) * state_size,
                                     codes_after.begin() + static_cast<std::size_t>(slot + 1) *
                                                             state_size),
            std::vector<std::int8_t>(
                pool_codes.begin() + static_cast<std::size_t>(slot) * state_size,
                pool_codes.begin() + static_cast<std::size_t>(slot + 1) * state_size));
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot) + " scale",
            std::vector<std::uint16_t>(
                scale_after.begin() + static_cast<std::size_t>(slot) * scale_slot_elements,
                scale_after.begin() + static_cast<std::size_t>(slot + 1) * scale_slot_elements),
            std::vector<std::uint16_t>(
                pool_scale.begin() + static_cast<std::size_t>(slot) * scale_slot_elements,
                pool_scale.begin() + static_cast<std::size_t>(slot + 1) * scale_slot_elements));
    }
    failures +=
        verify_exact(label + " initial-slot scalar unchanged",
                     from_device_i32(device_initial_slot, 1), std::vector<int>{initial_slot});
    failures += verify_exact(label + " snapshot-base scalar unchanged",
                             from_device_i32(device_snapshot_base_slot, 1),
                             std::vector<int>{snapshot_base_slot});
    failures += states_codes.verify_guards((label + " states codes").c_str());
    failures += states_scale.verify_guards((label + " states scale").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    return failures;
}

int snapshot_case(const Case& test_case, int slots, int initial_slot, int snapshot_base_slot,
                  std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk, true);
    const std::size_t state_size = in.state.size();
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 17.0f);
    std::copy(in.state.begin(), in.state.end(),
              initial_states.begin() + static_cast<std::size_t>(initial_slot) * state_size);

    DeviceInputs device(in);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slot       = to_device_i32({initial_slot});
    DeviceBuffer device_snapshot_base_slot = to_device_i32({snapshot_base_slot});

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor initial_slot_tensor(device_initial_slot.p, DType::I32, {1});
    Tensor snapshot_base_slot_tensor(device_snapshot_base_slot.p, DType::I32, {1});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  Tensor{}, Tensor{}, initial_slot_tensor, snapshot_base_slot_tensor,
                                  out_tensor, nullptr);
    cuda_synchronize();

    const std::string label             = std::string(test_case.name) + " snapshot";
    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    const auto got_updated_begin =
        got_states.begin() + static_cast<std::size_t>(snapshot_base_slot) * state_size;
    const auto got_updated_end =
        got_updated_begin + static_cast<std::size_t>(test_case.tokens) * state_size;
    int failures = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " updated state slots",
                                  doubles(std::vector<float>(got_updated_begin, got_updated_end)),
                                  ref.snapshots, gated_delta_net_state_fp32_criterion());
    const auto initial_updated_begin =
        initial_states.begin() + static_cast<std::size_t>(snapshot_base_slot) * state_size;
    const auto initial_updated_end =
        initial_updated_begin + static_cast<std::size_t>(test_case.tokens) * state_size;
    failures += verify_exact(label + " slots before destination unchanged",
                             std::vector<float>(got_states.begin(), got_updated_begin),
                             std::vector<float>(initial_states.begin(), initial_updated_begin));
    failures += verify_exact(label + " slots after destination unchanged",
                             std::vector<float>(got_updated_end, got_states.end()),
                             std::vector<float>(initial_updated_end, initial_states.end()));
    failures +=
        verify_exact(label + " initial-slot scalar unchanged",
                     from_device_i32(device_initial_slot, 1), std::vector<int>{initial_slot});
    failures += verify_exact(label + " snapshot-base scalar unchanged",
                             from_device_i32(device_snapshot_base_slot, 1),
                             std::vector<int>{snapshot_base_slot});
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    return failures;
}

int batched_snapshot_case(const Case& test_case, const std::vector<int>& initial_slots,
                          const std::vector<int>& snapshot_bases,
                          const std::vector<int>& valid_columns, int slots, std::uint32_t seed) {
    const int batch   = static_cast<int>(initial_slots.size());
    const int width   = test_case.tokens;
    const bool masked = !valid_columns.empty();
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }

    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(value_row_size * static_cast<std::size_t>(batch), 0.0);
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        const int valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        gdn_ref::Inputs oracle_input = rows[static_cast<std::size_t>(row)];
        oracle_input.tokens          = valid;
        oracle_input.q.resize(static_cast<std::size_t>(kStateDim * test_case.qk_heads * valid));
        oracle_input.k.resize(static_cast<std::size_t>(kStateDim * test_case.qk_heads * valid));
        oracle_input.v.resize(static_cast<std::size_t>(kStateDim * test_case.value_heads * valid));
        oracle_input.g.resize(static_cast<std::size_t>(test_case.value_heads * valid));
        oracle_input.beta.resize(static_cast<std::size_t>(test_case.value_heads * valid));
        gdn_ref::Result reference = gdn_ref::evaluate(oracle_input, static_cast<double>(scale),
                                                      test_case.normalize_qk, true);
        std::copy(reference.out.begin(), reference.out.end(),
                  expected_output.begin() + static_cast<std::size_t>(row) * value_row_size);
        for (int column = 0; column < valid; ++column) {
            written_slots[static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)] +
                                                   column)] = true;
        }
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slots  = to_device(initial_slots);
    DeviceBuffer device_snapshot_bases = to_device(snapshot_bases);
    DeviceBuffer device_valid_columns;
    if (masked) { device_valid_columns = to_device(valid_columns); }

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor valid_tensor;
    if (masked) { valid_tensor = Tensor(device_valid_columns.p, DType::I32, {batch}); }
    Tensor initial_tensor(device_initial_slots.p, DType::I32, {batch});
    Tensor bases_tensor(device_snapshot_bases.p, DType::I32, {batch});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  Tensor{}, valid_tensor, initial_tensor, bases_tensor, out_tensor,
                                  nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) +
                              " batched snapshot B=" + std::to_string(batch) +
                              (masked ? " masked" : " dense");
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    failures += verify_recurrence(label + " out", got_output, expected_output,
                                  gated_delta_net_output_bf16_criterion());
    if (masked) {
        const std::vector<std::uint16_t> output_bits =
            from_device<std::uint16_t>(out.data(), aggregate.v.size());
        for (int row = 0; row < batch; ++row) {
            for (int column = valid_columns[static_cast<std::size_t>(row)]; column < width;
                 ++column) {
                const std::size_t begin =
                    static_cast<std::size_t>(row) * value_row_size +
                    static_cast<std::size_t>(column) * kStateDim * test_case.value_heads;
                const std::size_t end =
                    begin + static_cast<std::size_t>(kStateDim * test_case.value_heads);
                if (!std::all_of(output_bits.begin() + begin, output_bits.begin() + end,
                                 [](std::uint16_t value) { return value == 0; })) {
                    std::cerr << label << ": invalid output tail is not exact zero\n";
                    ++failures;
                    row = batch;
                    break;
                }
            }
        }
    }

    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    for (int row = 0; row < batch; ++row) {
        const int valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        const std::size_t begin =
            static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)]) * state_size;
        failures += verify_recurrence(
            label + " row " + std::to_string(row) + " snapshots",
            doubles(std::vector<float>(got_states.begin() + begin,
                                       got_states.begin() + begin +
                                           static_cast<std::size_t>(valid) * state_size)),
            references[static_cast<std::size_t>(row)].snapshots,
            gated_delta_net_state_fp32_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<float>(got_states.begin() + begin, got_states.begin() + begin + state_size),
            std::vector<float>(initial_states.begin() + begin,
                               initial_states.begin() + begin + state_size));
    }
    failures +=
        verify_exact(label + " initial selectors unchanged",
                     from_device_i32(device_initial_slots, initial_slots.size()), initial_slots);
    failures +=
        verify_exact(label + " snapshot bases unchanged",
                     from_device_i32(device_snapshot_bases, snapshot_bases.size()), snapshot_bases);
    if (masked) {
        failures += verify_exact(label + " valid columns unchanged",
                                 from_device_i32(device_valid_columns, valid_columns.size()),
                                 valid_columns);
    }
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(8 * sizeof(float));
    DeviceBuffer beta_buffer(8 * sizeof(float));
    DeviceBuffer state_buffer(kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    WorkspaceArena workspace(256);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.p, DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.p, DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.p, DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, Tensor{}, out,
                                 nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    for (const bool normalize_qk : {false, true}) {
        const std::size_t interval =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 63, 65);
        const std::size_t witness =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 65, 65);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
    }
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 0, 65);
        std::cerr << "gated_delta_net accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, true, 1, 65);
        std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += contract_rejection_cases();

    // Registered 27B/35B-A3B geometries, public state forms, and the recurrent/chunk/tail route
    // boundary are all qualified directly against the same complete FP64 recurrence.
    failures += inplace_case({"27b decode fused-qk-norm", 16, 48, 1, true}, 12001u);
    failures += distinct_state_case({"27b raw-qk small-T", 16, 48, 7, false}, 12007u);
    failures += distinct_state_case({"35b pre-chunk fused-qk-norm", 16, 32, 63, true}, 12063u);
    failures += distinct_state_case({"27b exact chunk fused-qk-norm", 16, 48, 64, true}, 12064u);
    failures += distinct_state_case({"27b exact chunk raw-qk", 16, 48, 64, false}, 12164u);
    failures += inplace_case({"35b chunk-tail fused-qk-norm", 16, 32, 65, true}, 12065u);
    failures += distinct_state_case({"generic grouped-map chunk-tail", 3, 12, 65, true}, 12365u);
    failures += distinct_state_case({"27b two-chunk fused-qk-norm", 16, 48, 128, true}, 12128u);
    failures += inplace_case({"35b two-chunk raw-qk", 16, 32, 128, false}, 12228u);

    // Snapshot is a separate public state transition. Nonzero source slots also prove that the
    // selected initial state, not slot zero, seeds the complete recurrence.
    failures += snapshot_case({"27b verify fused-qk-norm", 16, 48, 4, true}, 8, 7, 1, 12104u);
    failures += snapshot_case({"35b verify fused-qk-norm near-zero", 16, 32, 4, true, true}, 8, 6,
                              1, 12204u);
    failures +=
        batched_snapshot_case({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
                              {0, 1, 2, 3, 4, 5, 6, 7}, {}, 16, 13001u);
    failures += batched_snapshot_case({"27b MTP", 16, 48, 6, true}, {18, 19, 20}, {0, 6, 12},
                                      {6, 3, 1}, 21, 13006u);
    // Row 0's initial state is its final destination; every state tile must load it before write.
failures +=
        batched_snapshot_case({"35b DFlash", 16, 32, 16, true}, {15, 33}, {0, 16}, {16, 7},
                              34, 13016u);

    // I8 persistent state: same public contracts through the int8 + per-row FP16-scale pool.
    // The direct (T=1) and chunked (T=65/128) routes plus the snapshot transition each qualify
    // their quantize/dequantize round trips against the same complete FP64 recurrence.
    failures += i8_inplace_case({"27b i8 decode fused-qk-norm", 16, 48, 1, true}, 12501u);
    failures += i8_distinct_state_case({"27b i8 raw-qk small-T", 16, 48, 7, false}, 12507u);
    failures += i8_inplace_case({"27b i8 chunk-tail fused-qk-norm", 16, 48, 65, true}, 12565u);
    failures += i8_inplace_case({"27b i8 exact chunk fused-qk-norm", 16, 48, 64, true}, 12664u);
    failures += i8_inplace_case({"27b i8 two-chunk fused-qk-norm", 16, 48, 128, true}, 12628u);
    failures += i8_snapshot_case({"27b i8 verify fused-qk-norm", 16, 48, 4, true}, 8, 7, 1, 12704u);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
