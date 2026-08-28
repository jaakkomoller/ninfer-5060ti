#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
constexpr ReductionCriterion kFp8GdnInputProjConvRecordA16Tolerance{1.0 / 256.0, 1.0 / 256.0,
                                                                    2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjConvRecordA8Tolerance{0.04, 1.0 / 256.0, 0.06};

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<std::uint16_t> make_bf16_bits(std::size_t elements, std::uint32_t seed, float low,
                                          float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

int verify_equal(std::string_view label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin())) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_zero_tail(std::string_view label, const std::vector<std::uint16_t>& values,
                     std::int32_t rows, std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = valid_columns[static_cast<std::size_t>(batch_row)]; token < width;
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (values[base + row] != 0) {
                    std::cerr << label << ": invalid tail is not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_record_prefix_written(std::string_view label, const std::vector<std::uint16_t>& values,
                                 std::int32_t channels, std::int32_t width, std::int32_t batch,
                                 const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        const std::int32_t valid = valid_columns[static_cast<std::size_t>(batch_row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * channels;
            for (std::int32_t row = 0; row < channels; ++row) {
                if (!std::isfinite(bf16_to_f32(values[base + row]))) {
                    std::cerr << label << ": valid record prefix was not fully written\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_conv_record(std::string_view label, const std::vector<std::uint16_t>& snapshot_state,
                       const std::vector<std::uint16_t>& record, std::int32_t channels,
                       std::int32_t width, std::int32_t batch,
                       const std::vector<std::int32_t>& valid_columns,
                       const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t snapshot =
                static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                         token) *
                    slot_stride +
                2ULL * channels;
            const std::size_t record_column =
                static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(snapshot_state.begin() + static_cast<std::ptrdiff_t>(snapshot),
                            snapshot_state.begin() +
                                static_cast<std::ptrdiff_t>(snapshot + channels),
                            record.begin() + static_cast<std::ptrdiff_t>(record_column))) {
                std::cerr << label << ": conv record differs from snapshot newest column\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class SnapshotLaunch, class RecordLaunch>
int run_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
             std::int32_t z_rows, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::size_t snapshot_workspace_bytes,
             std::size_t record_workspace_bytes, SnapshotLaunch&& snapshot_launch,
             RecordLaunch&& record_launch, std::uint32_t seed) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t slots             = aggregate_columns + batch + 1;
    const bool dense                     = valid_columns.empty();
    if (valid_columns.empty()) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation = make_bf16_activation(hidden, aggregate_columns, seed);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(channels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(channels) * 3 * slots, seed + 2, -0.05F, 0.05F);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        initial_slots[static_cast<std::size_t>(batch_row)]  = aggregate_columns + batch_row;
    }

    DeviceBuffer device_x           = to_device_bf16(activation);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state     = to_device(state_before);
    DeviceBuffer record_state       = to_device(state_before);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer device_initial  = to_device(initial_slots);
    DeviceBuffer device_snapshot = to_device(snapshot_bases);

    GuardedBf16Tensor snapshot_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor snapshot_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor snapshot_value(value_rows, aggregate_columns);
    GuardedBf16Tensor snapshot_z(z_rows, aggregate_columns);
    GuardedBf16Tensor record_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor record_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor record_value(value_rows, aggregate_columns);
    GuardedBf16Tensor record_z(z_rows, aggregate_columns);
    GuardedBf16Tensor conv_record(channels, aggregate_columns);

    Tensor x(device_x.p, DType::BF16, {hidden, width, batch});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {channels, 4});
    Tensor snapshot_state_view(snapshot_state.p, DType::BF16, {channels, 3, slots});
    Tensor record_state_view(record_state.p, DType::BF16, {channels, 3, slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor snapshot_base(device_snapshot.p, DType::I32, {batch});
    Tensor snapshot_q(snapshot_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor snapshot_k(snapshot_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor snapshot_v(snapshot_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor snapshot_z_view(snapshot_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor record_q(record_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor record_k(record_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor record_v(record_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor record_z_view(record_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor conv_record_view(conv_record.data(), DType::BF16, {channels, width, batch});

    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_workspace_bytes));
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_workspace_bytes));
    snapshot_launch(x, conv_weight, snapshot_state_view, valid, initial, snapshot_base, snapshot_q,
                    snapshot_k, snapshot_v, snapshot_z_view, snapshot_workspace);
    record_launch(x, conv_weight, record_state_view, valid, initial, conv_record_view, record_q,
                  record_k, record_v, record_z_view, record_workspace);
    cuda_synchronize();

    int failures = 0;
    failures +=
        verify_equal(std::string(label) + " query", snapshot_query.bits(), record_query.bits());
    failures += verify_equal(std::string(label) + " key", snapshot_key.bits(), record_key.bits());
    failures +=
        verify_equal(std::string(label) + " value", snapshot_value.bits(), record_value.bits());
    failures += verify_equal(std::string(label) + " z", snapshot_z.bits(), record_z.bits());
    failures += verify_zero_tail(std::string(label) + " query", record_query.bits(), kQueryRows,
                                 width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " key", record_key.bits(), kKeyRows, width,
                                 batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", record_value.bits(), value_rows,
                                 width, batch, valid_columns);

    const std::vector<std::uint16_t> snapshot_state_after =
        from_device<std::uint16_t>(snapshot_state, state_before.size());
    const std::vector<std::uint16_t> record_state_after =
        from_device<std::uint16_t>(record_state, state_before.size());
    failures += verify_conv_record(label, snapshot_state_after, conv_record.bits(), channels, width,
                                   batch, valid_columns, snapshot_bases);
    failures +=
        verify_equal(std::string(label) + " source state", state_before, record_state_after);

    failures += snapshot_query.verify_guards(std::string(label) + " snapshot query");
    failures += snapshot_key.verify_guards(std::string(label) + " snapshot key");
    failures += snapshot_value.verify_guards(std::string(label) + " snapshot value");
    failures += snapshot_z.verify_guards(std::string(label) + " snapshot z");
    failures += record_query.verify_guards(std::string(label) + " record query");
    failures += record_key.verify_guards(std::string(label) + " record key");
    failures += record_value.verify_guards(std::string(label) + " record value");
    failures += record_z.verify_guards(std::string(label) + " record z");
    failures += conv_record.verify_guards(std::string(label) + " conv record");
    if (snapshot_workspace.used() != 0 ||
        snapshot_workspace.peak_used() != snapshot_workspace_bytes) {
        std::cerr << label << ": snapshot workspace query/execution mismatch\n";
        ++failures;
    }
    if (record_workspace.used() != 0 || record_workspace.peak_used() != record_workspace_bytes) {
        std::cerr << label << ": record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 1401U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 1403U));
    DevicePackedWeight value_z_q4(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 12288, kHidden, 1407U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed, DevicePackedWeight& vz, std::string_view pair =
                             "Q4/Q5") {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            std::string(pair) + " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_snapshot(
                    x, qk.view(), vz.view(), conv, state, Tensor{}, valid_columns, initial,
                    snapshot_base, q, k, v, z, workspace, nullptr);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_record(x, qk.view(), vz.view(), conv, state,
                                                Tensor{}, valid_columns, initial, record, q, k, v,
                                                z, workspace, nullptr);
            },
            seed);
    };
    failures += run(2, 1, {}, 1411U, value_z);
    failures += run(4, 1, {3}, 1421U, value_z);
    failures += run(7, 1, {5}, 1431U, value_z);
    failures += run(16, 1, {}, 1441U, value_z);
    failures += run(6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1451U, value_z);
    failures += run(2, 1, {}, 1461U, value_z_q4, "Q4/Q4");
    failures += run(4, 1, {3}, 1471U, value_z_q4, "Q4/Q4");
    failures += run(6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1481U, value_z_q4, "Q4/Q4");
    failures += qk.verify_preserved("Q4 record qk weight");
    failures += value_z.verify_preserved("Q5 record value/z weight");
    failures += value_z_q4.verify_preserved("Q4 record value/z weight");
    return failures;
}

// ===== I8 conv state form (two-parent Q4/Q5 record) ==============================
// conv_states is a read-only I8 code plane plus a sticky FP16 scale per (128-channel group,
// slot): the convolution dequantizes history exactly (code * scale), the record column keeps
// the raw BF16 projected value, and the source codes and scale are bit-exact unchanged.
// The same A16 route profile qualifies the Q4/Q5 conv output, z, and record against the
// complete FP64 oracle.

std::uint16_t f32_to_fp16_rne(float f) { return __half_as_ushort(__float2half_rn(f)); }

float fp16_bits_to_f32(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

struct I8InitialWindow {
    std::vector<std::int8_t> codes;        // tap-major [3, C]
    std::vector<std::uint16_t> scale_bits; // [C/128]
    std::vector<float> decoded;            // exact dequantization, tap-major [3, C]
};

// The sticky slot scale is derived from the stored window, so the initial window sits at the
// projected values' magnitude (the patterned projections here reach ~9).
I8InitialWindow make_i8_initial(std::int32_t channels, std::uint32_t seed) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<float> logical(slot_stride);
    fill_uniform(logical, seed, -12.0F, 12.0F);
    round_to_bf16(logical);

    I8InitialWindow result;
    result.codes.assign(slot_stride, 0);
    const std::int32_t groups = channels / 128;
    result.scale_bits.assign(static_cast<std::size_t>(groups), 0U);
    result.decoded.assign(slot_stride, 0.0F);
    for (std::int32_t g = 0; g < groups; ++g) {
        const std::int32_t c_begin = g * 128;
        double max_abs             = 0.0;
        for (std::int32_t c = c_begin; c < c_begin + 128; ++c) {
            for (std::int32_t s = 0; s < 3; ++s) {
                max_abs = std::max(
                    max_abs, static_cast<double>(std::fabs(
                                 logical[static_cast<std::size_t>(s) * channels + c])));
            }
        }
        const std::uint16_t scale_bits = f32_to_fp16_rne(static_cast<float>(max_abs / 127.0));
        const float scale              = fp16_bits_to_f32(scale_bits);
        result.scale_bits[static_cast<std::size_t>(g)] = scale_bits;
        for (std::int32_t c = c_begin; c < c_begin + 128; ++c) {
            for (std::int32_t s = 0; s < 3; ++s) {
                const std::size_t index = static_cast<std::size_t>(s) * channels + c;
                const float quotient    = (scale == 0.0F) ? 0.0F
                                                          : static_cast<float>(logical[index] /
                                                                               scale);
                double code             = std::nearbyint(static_cast<double>(quotient));
                if (code > 127.0) { code = 127.0; }
                if (code < -127.0) { code = -127.0; }
                result.codes[index]     = static_cast<std::int8_t>(code);
                result.decoded[index]   = static_cast<float>(result.codes[index]) * scale;
            }
        }
    }
    return result;
}

int run_q4_q5_i8_record_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                             std::int32_t width, std::uint32_t seed) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kChannels    = 10240;
    constexpr std::int32_t kGroups      = kChannels / 128;
    constexpr std::int32_t kSlots       = 2;
    constexpr std::int32_t kInitialSlot = 1;
    constexpr ReductionCriterion kA16Tolerance{3.15e-3, 4.0e-3, 3.2e-3};
    const std::size_t slot_stride       = static_cast<std::size_t>(kChannels) * 3;

    const std::vector<float> activation = make_bf16_activation(kHidden, width, seed);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    std::vector<float> conv_weight(static_cast<std::size_t>(kChannels) * 4);
    fill_uniform(conv_weight, seed + 1U, -0.02F, 0.02F);
    round_to_bf16(conv_weight);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const I8InitialWindow initial = make_i8_initial(kChannels, seed + 2U);

    std::vector<std::int8_t> codes(slot_stride * kSlots, 0x7e);
    std::vector<std::uint16_t> scale_bits(static_cast<std::size_t>(kGroups) * kSlots, 0x0001U);
    for (std::int32_t s = 0; s < 3; ++s) {
        std::copy(initial.codes.begin() + s * kChannels, initial.codes.begin() + (s + 1) * kChannels,
                  codes.begin() + static_cast<std::ptrdiff_t>(kInitialSlot) * slot_stride +
                      s * kChannels);
    }
    // The scale plane is slot-major: slot s owns the contiguous [groups] block
    // [s * groups, (s + 1) * groups), element (group, slot) at slot * groups + group.
    for (std::int32_t g = 0; g < kGroups; ++g) {
        scale_bits[static_cast<std::size_t>(kInitialSlot) * kGroups + g] =
            initial.scale_bits[static_cast<std::size_t>(g)];
    }
    const std::vector<std::int32_t> initial_value{kInitialSlot};

    DeviceBuffer device_x           = to_device(activation_bits);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer device_codes(codes.size());
    DeviceBuffer device_scale(scale_bits.size() * sizeof(std::uint16_t));
    DeviceBuffer device_initial = to_device(initial_value);
    device_codes.copy_from_host(codes.data(), codes.size());
    device_scale.copy_from_host(scale_bits.data(), device_scale.bytes);
    GuardedBf16Tensor query(kQueryRows, width);
    GuardedBf16Tensor key(kKeyRows, width);
    GuardedBf16Tensor value(kValueRows, width);
    GuardedBf16Tensor z(kZRows, width);
    GuardedBf16Tensor conv_record(kChannels, width);

    Tensor x(device_x.p, DType::BF16, {kHidden, width});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(device_codes.p, DType::I8, {kChannels, 3, kSlots});
    Tensor conv_scale(device_scale.p, DType::FP16, {kGroups, 1, kSlots});
    Tensor initial_tensor(device_initial.p, DType::I32, {1});
    Tensor q        = query.tensor();
    Tensor k        = key.tensor();
    Tensor v        = value.tensor();
    Tensor z_output = z.tensor();
    Tensor record_view = conv_record.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, width, width);
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    ops::gdn_input_proj_conv_record(x, query_key.view(), value_z_weight.view(), conv, conv_state,
                                    conv_scale, Tensor{}, initial_tensor, record_view, q, k, v,
                                    z_output, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q4/Q5 I8 T=" + std::to_string(width);
    int failures             = 0;
    const auto project = [&](std::int32_t global_row, std::int32_t token) {
        const float* token_activation =
            activation.data() + static_cast<std::size_t>(token) * kHidden;
        if (global_row < kQueryRows + kKeyRows) {
            return quantized_weight::dot_fp64(query_key.host, global_row, token_activation,
                                              kHidden);
        }
        return quantized_weight::dot_fp64(
            value_z_weight.host, global_row - kQueryRows - kKeyRows, token_activation, kHidden);
    };

    const std::int32_t sample_count = width <= 4 ? 32 : 7;
    const std::vector<std::int32_t> query_rows = sampled_rows(kQueryRows, sample_count);
    const std::vector<std::int32_t> key_rows   = sampled_rows(kKeyRows, sample_count);
    const std::vector<std::int32_t> value_rows = sampled_rows(kValueRows, sample_count);
    std::vector<double> ref_query, ref_key, ref_value;
    const auto run_row = [&](std::int32_t global_row, std::vector<double>& output) {
        double state0       = initial.decoded[global_row];
        double state1       = initial.decoded[static_cast<std::size_t>(kChannels) + global_row];
        double state2       = initial.decoded[2ULL * kChannels + global_row];
        const double w0     = conv_weight[global_row];
        const double w1     = conv_weight[static_cast<std::size_t>(kChannels) + global_row];
        const double w2     = conv_weight[2ULL * kChannels + global_row];
        const double w3     = conv_weight[3ULL * kChannels + global_row];
        for (std::int32_t token = 0; token < width; ++token) {
            const double projected = project(global_row, token);
            output.push_back(
                silu_fp64(w0 * state0 + w1 * state1 + w2 * state2 + w3 * projected));
            state0  = state1;
            state1  = state2;
            state2  = projected;
        }
    };
    for (const std::int32_t row : query_rows) { run_row(row, ref_query); }
    for (const std::int32_t row : key_rows) { run_row(kQueryRows + row, ref_key); }
    for (const std::int32_t row : value_rows) {
        run_row(kQueryRows + kKeyRows + row, ref_value);
    }
    failures += compare("record I8 query" + suffix,
                        gather_rows(query.values(), kQueryRows, 0, kQueryRows, width,
                                    sample_count),
                        ref_query, kA16Tolerance);
    failures += compare("record I8 key" + suffix,
                        gather_rows(key.values(), kKeyRows, 0, kKeyRows, width, sample_count),
                        ref_key, kA16Tolerance);
    failures += compare("record I8 value" + suffix,
                        gather_rows(value.values(), kValueRows, 0, kValueRows, width,
                                    sample_count),
                        ref_value, kA16Tolerance);
    failures += z.verify_guards("record I8 z" + suffix);
    failures += z.verify_fully_written("record I8 z" + suffix);
    failures += compare("record I8 z" + suffix,
                        gather_rows(z.values(), kZRows, 0, kZRows, width, sample_count),
                        projection_oracle(value_z_weight.host, kValueRows, kZRows, activation,
                                          kHidden, width, sample_count),
                        kA16Tolerance);
    // The record column keeps the raw BF16 projected value.
    failures += conv_record.verify_guards("record I8 conv record" + suffix);
    failures += conv_record.verify_fully_written("record I8 conv record" + suffix);
    failures += compare(
        "record I8 conv record" + suffix,
        gather_rows(conv_record.values(), kChannels, kQueryRows + kKeyRows, kValueRows, width,
                    sample_count),
        projection_oracle(value_z_weight.host, 0, kValueRows, activation, kHidden, width,
                          sample_count),
        kA16Tolerance);

    // The source state plane is read-only: codes and sticky scale are bit-exact unchanged.
    const std::vector<std::int8_t> codes_after =
        from_device<std::int8_t>(device_codes, codes.size());
    const std::vector<std::uint16_t> scale_after =
        from_device<std::uint16_t>(device_scale, scale_bits.size());
    if (codes_after != codes) {
        std::cerr << "record I8" << suffix << ": source conv codes were modified\n";
        ++failures;
    }
    if (scale_after != scale_bits) {
        std::cerr << "record I8" << suffix << ": source conv scale plane was modified\n";
        ++failures;
    }
    failures += verify_preserved("record I8 x" + suffix, device_x, activation_bits);
    failures += verify_preserved("record I8 conv weight" + suffix, device_conv_weight,
                                 conv_weight_bits);
    failures += verify_preserved("record I8 initial slot" + suffix, device_initial,
                                 initial_value);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "record I8" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5_i8_record() {
    constexpr std::int32_t kHidden = 5120;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 1701U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 1703U));
    int failures = 0;
    // Fused Small-T record routes plus the T=4 composed MTP verify route.
    for (const std::int32_t width : {2, 3, 4, 6}) {
        failures += run_q4_q5_i8_record_case(query_key, value_z_weight, width,
                                             1711U + static_cast<std::uint32_t>(width));
    }
    failures += query_key.verify_preserved("I8 Q4 record qk weight");
    failures += value_z_weight.verify_preserved("I8 Q5 record value/z weight");
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 1501U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "W8 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, Tensor{}, valid_columns,
                                                   initial, snapshot_base, q, k, v, z, workspace,
                                                   nullptr);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, Tensor{},
                                                valid_columns, initial, record, q, k, v, z,
                                                workspace, nullptr);
            },
            seed);
    };
    failures += run(2, 1, {1}, 1511U);
    failures += run(16, 1, {}, 1521U);
    failures += run(16, 8, {16, 13, 9, 7, 5, 3, 2, 1}, 1531U);
    failures += parent.verify_preserved("W8 record parent weight");
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1601U, options));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         ops::LinearPolicy policy, std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                       policy, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, policy, batch, width, width);
        return run_case(
            std::string("NVFP4 ") + (policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16") +
                " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, Tensor{}, valid_columns,
                                                   initial, snapshot_base, q, k, v, z, policy,
                                                   workspace, nullptr);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, Tensor{},
                                                valid_columns, initial, record, q, k, v, z, policy,
                                                workspace, nullptr);
            },
            seed);
    };
    failures += run(2, 1, {}, ops::LinearPolicy::A16Only, 1611U);
    failures += run(16, 1, {11}, ops::LinearPolicy::A16Only, 1621U);
    failures += run(3, 1, {2}, ops::LinearPolicy::AllowA4, 1631U);
    failures += run(4, 1, {}, ops::LinearPolicy::AllowA4, 1641U);
    failures += run(16, 1, {13}, ops::LinearPolicy::AllowA4, 1651U);
    failures += run(6, 3, {6, 4, 1}, ops::LinearPolicy::AllowA4, 1661U);
    failures += parent.verify_preserved("NVFP4 record parent weight");
    return failures;
}

int run_fp8_oracle_case(DevicePackedWeight& parent, std::int32_t width, std::int32_t batch,
                        std::vector<std::int32_t> valid_columns, ops::LinearPolicy policy,
                        std::uint32_t seed) {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows      = kChannels + kZRows;
    const std::int32_t columns        = width * batch;
    const std::int32_t slots          = batch + 2;
    const bool dense                  = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation              = make_bf16_activation(kHidden, columns, seed);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * slots, seed + 2, -0.05F, 0.05F);
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        initial_slots[static_cast<std::size_t>(batch_row)] = batch_row;
    }

    DeviceBuffer device_x           = to_device(activation_bits);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer device_state       = to_device(state_before);
    DeviceBuffer device_initial     = to_device(initial_slots);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    GuardedBf16Tensor conv_record(kChannels, columns);
    GuardedBf16Tensor query(kQueryRows, columns);
    GuardedBf16Tensor key(kKeyRows, columns);
    GuardedBf16Tensor value(kValueRows, columns);
    GuardedBf16Tensor z(kZRows, columns);

    Tensor x(device_x.p, DType::BF16, {kHidden, width, batch});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor state(device_state.p, DType::BF16, {kChannels, 3, slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor record_view(conv_record.data(), DType::BF16, {kChannels, width, batch});
    Tensor query_view(query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor key_view(key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor value_view(value.data(), DType::BF16, {kValueRows, width, batch});
    Tensor z_view(z.data(), DType::BF16, {kZRows, width, batch});
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, policy, batch, width, width);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    ops::gdn_input_proj_conv_record(x, parent.view(), conv_weight, state, Tensor{}, valid, initial,
                                    record_view, query_view, key_view, value_view, z_view, policy,
                                    workspace, nullptr);
    cuda_synchronize();

    const std::vector<double> record_values = conv_record.values();
    const std::vector<double> query_values  = query.values();
    const std::vector<double> key_values    = key.values();
    const std::vector<double> value_values  = value.values();
    const std::vector<double> z_values      = z.values();
    std::vector<double> record_actual;
    std::vector<double> record_expected;
    std::vector<double> output_actual;
    std::vector<double> output_expected;
    std::vector<double> z_actual;
    std::vector<double> z_expected;
    const auto append_channel = [&](std::int32_t global_row, std::int32_t output_rows,
                                    std::int32_t output_row, const std::vector<double>& output) {
        const double w0 = bf16_to_f32(conv_weight_bits[global_row]);
        const double w1 = bf16_to_f32(conv_weight_bits[kChannels + global_row]);
        const double w2 = bf16_to_f32(conv_weight_bits[2LL * kChannels + global_row]);
        const double w3 = bf16_to_f32(conv_weight_bits[3LL * kChannels + global_row]);
        for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
            const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(batch_row)];
            const std::size_t initial_base =
                static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(batch_row)]) * 3 *
                kChannels;
            double s0 = bf16_to_f32(state_before[initial_base + global_row]);
            double s1 = bf16_to_f32(state_before[initial_base + kChannels + global_row]);
            double s2 = bf16_to_f32(state_before[initial_base + 2LL * kChannels + global_row]);
            for (std::int32_t token = 0; token < valid_extent; ++token) {
                const std::int32_t column = batch_row * width + token;
                const double projected    = quantized_weight::dot_fp64(
                    parent.host, global_row,
                    activation.data() + static_cast<std::size_t>(column) * kHidden, kHidden);
                record_actual.push_back(
                    record_values[static_cast<std::size_t>(column) * kChannels + global_row]);
                record_expected.push_back(projected);
                output_actual.push_back(
                    output[static_cast<std::size_t>(column) * output_rows + output_row]);
                output_expected.push_back(silu_fp64(w0 * s0 + w1 * s1 + w2 * s2 + w3 * projected));
                s0 = s1;
                s1 = s2;
                s2 = projected;
            }
        }
    };

    constexpr std::int32_t kSampleRows = 7;
    for (const std::int32_t row : sampled_rows(kQueryRows, kSampleRows)) {
        append_channel(row, kQueryRows, row, query_values);
    }
    for (const std::int32_t row : sampled_rows(kKeyRows, kSampleRows)) {
        append_channel(kQueryRows + row, kKeyRows, row, key_values);
    }
    for (const std::int32_t row : sampled_rows(kValueRows, kSampleRows)) {
        append_channel(kQueryRows + kKeyRows + row, kValueRows, row, value_values);
    }
    for (const std::int32_t row : sampled_rows(kZRows, kSampleRows)) {
        for (std::int32_t column = 0; column < columns; ++column) {
            z_actual.push_back(z_values[static_cast<std::size_t>(column) * kZRows + row]);
            z_expected.push_back(quantized_weight::dot_fp64(
                parent.host, kChannels + row,
                activation.data() + static_cast<std::size_t>(column) * kHidden, kHidden));
        }
    }

    const bool uses_a8 =
        policy == ops::LinearPolicy::AllowA8 && (batch == 1 ? width >= 10 : width * batch >= 8);
    const ReductionCriterion& criterion =
        uses_a8 ? kFp8GdnInputProjConvRecordA8Tolerance : kFp8GdnInputProjConvRecordA16Tolerance;
    const std::string label = std::string("FP8 ") + (uses_a8 ? "A8" : "A16") +
                              " B=" + std::to_string(batch) + " W=" + std::to_string(width);
    int failures = compare(label + " record", record_actual, record_expected, criterion);
    failures += compare(label + " query/key/value", output_actual, output_expected, criterion);
    failures += compare(label + " z", z_actual, z_expected, criterion);
    failures +=
        verify_zero_tail(label + " query", query.bits(), kQueryRows, width, batch, valid_columns);
    failures += verify_zero_tail(label + " key", key.bits(), kKeyRows, width, batch, valid_columns);
    failures +=
        verify_zero_tail(label + " value", value.bits(), kValueRows, width, batch, valid_columns);
    failures += verify_record_prefix_written(label + " record", conv_record.bits(), kChannels,
                                             width, batch, valid_columns);
    failures += query.verify_fully_written(label + " query");
    failures += key.verify_fully_written(label + " key");
    failures += value.verify_fully_written(label + " value");
    failures += z.verify_fully_written(label + " z");
    failures += conv_record.verify_guards(label + " record");
    failures += query.verify_guards(label + " query");
    failures += key.verify_guards(label + " key");
    failures += value.verify_guards(label + " value");
    failures += z.verify_guards(label + " z");
    failures += verify_preserved(label + " x", device_x, activation_bits);
    failures += verify_preserved(label + " conv weight", device_conv_weight, conv_weight_bits);
    failures += verify_preserved(label + " initial", device_initial, initial_slots);
    if (!dense) { failures += verify_preserved(label + " valid", device_valid, valid_columns); }
    failures += verify_equal(label + " source state", state_before,
                             from_device<std::uint16_t>(device_state, state_before.size()));
    failures += parent.verify_preserved(label + " parent weight");
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S,
                                                                      kRows, kHidden, 1701U));
    int failures = 0;
    failures += run_fp8_oracle_case(parent, 2, 1, {}, ops::LinearPolicy::A16Only, 1711U);
    failures += run_fp8_oracle_case(parent, 3, 1, {2}, ops::LinearPolicy::A16Only, 1721U);
    failures += run_fp8_oracle_case(parent, 4, 1, {}, ops::LinearPolicy::A16Only, 1731U);
    failures += run_fp8_oracle_case(parent, 6, 1, {5}, ops::LinearPolicy::A16Only, 1741U);
    failures += run_fp8_oracle_case(parent, 7, 1, {}, ops::LinearPolicy::AllowA8, 1751U);
    failures += run_fp8_oracle_case(parent, 9, 1, {7}, ops::LinearPolicy::AllowA8, 1761U);
    failures += run_fp8_oracle_case(parent, 10, 1, {}, ops::LinearPolicy::AllowA8, 1771U);
    failures += run_fp8_oracle_case(parent, 10, 1, {8}, ops::LinearPolicy::A16Only, 1781U);
    failures += run_fp8_oracle_case(parent, 11, 1, {9}, ops::LinearPolicy::A16Only, 1791U);
    failures += run_fp8_oracle_case(parent, 3, 2, {3, 1}, ops::LinearPolicy::AllowA8, 1801U);
    failures += run_fp8_oracle_case(parent, 4, 2, {4, 2}, ops::LinearPolicy::AllowA8, 1811U);
    failures += run_fp8_oracle_case(parent, 16, 8, {16, 13, 11, 7, 5, 3, 2, 1},
                                    ops::LinearPolicy::AllowA8, 1821U);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures                   = 0;
    const auto fp8_record_capacity = [](ops::LinearPolicy policy, std::int32_t batch,
                                        std::int32_t min_width, std::int32_t max_width) {
        return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 16384, 5120, policy, batch, min_width, max_width);
    };
    const std::size_t fp8_b1_w10 = fp8_record_capacity(ops::LinearPolicy::AllowA8, 1, 10, 10);
    const std::size_t fp8_b2_w4  = fp8_record_capacity(ops::LinearPolicy::AllowA8, 2, 4, 4);
    if (fp8_record_capacity(ops::LinearPolicy::A16Only, 1, 2, 16) != 0 ||
        fp8_record_capacity(ops::LinearPolicy::AllowA8, 1, 2, 9) != 0 || fp8_b1_w10 == 0 ||
        fp8_record_capacity(ops::LinearPolicy::AllowA8, 1, 2, 10) != fp8_b1_w10 ||
        fp8_record_capacity(ops::LinearPolicy::AllowA8, 2, 2, 3) != 0 || fp8_b2_w4 == 0 ||
        fp8_record_capacity(ops::LinearPolicy::AllowA8, 2, 2, 4) != fp8_b2_w4) {
        std::cerr << "FP8 record capacity did not preserve measured route witnesses\n";
        ++failures;
    }
    failures += run_q4_q5();
    failures += run_q4_q5_i8_record();
    failures += run_w8();
    failures += run_nvfp4();
    failures += run_fp8();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
