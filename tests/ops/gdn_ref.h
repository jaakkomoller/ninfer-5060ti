#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::gdn_ref {

struct Inputs {
    std::int64_t head_dim    = 0;
    std::int64_t qk_heads    = 0;
    std::int64_t value_heads = 0;
    std::int64_t tokens      = 0;

    // q/k/v contain the exact FP32 values represented by their public BF16 tensors.
    // g/beta/state contain the exact public FP32 values.
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
};

struct Result {
    std::vector<double> out;
    std::vector<double> final_state;
    // When requested, token t occupies the contiguous state slot t.
    std::vector<double> snapshots;
};

inline std::int64_t qk_head(std::int64_t value_head, std::int64_t qk_heads,
                            std::int64_t value_heads) {
    return value_head / (value_heads / qk_heads);
}

inline Result evaluate(const Inputs& in, double scale, bool normalize_qk,
                       bool record_snapshots = false) {
    const std::int64_t S    = in.head_dim;
    const std::int64_t H_qk = in.qk_heads;
    const std::int64_t H_v  = in.value_heads;
    const std::int64_t T    = in.tokens;
    if (S <= 0 || H_qk <= 0 || H_v < H_qk || H_v % H_qk != 0 || T <= 0) {
        throw std::invalid_argument("gdn_ref: invalid geometry");
    }

    const std::size_t qk_size    = static_cast<std::size_t>(S * H_qk * T);
    const std::size_t value_size = static_cast<std::size_t>(S * H_v * T);
    const std::size_t state_size = static_cast<std::size_t>(S * S * H_v);
    if (in.q.size() != qk_size || in.k.size() != qk_size || in.v.size() != value_size ||
        in.g.size() != static_cast<std::size_t>(H_v * T) ||
        in.beta.size() != static_cast<std::size_t>(H_v * T) || in.state.size() != state_size) {
        throw std::invalid_argument("gdn_ref: input size does not match geometry");
    }

    // These are logical Q/K values, not a model of any production staging tensor.
    std::vector<double> q_logical(qk_size);
    std::vector<double> k_logical(qk_size);
    for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t h = 0; h < H_qk; ++h) {
            const std::size_t base = static_cast<std::size_t>((t * H_qk + h) * S);
            double q_sumsq         = 0.0;
            double k_sumsq         = 0.0;
            for (std::int64_t d = 0; d < S; ++d) {
                const double q_value = static_cast<double>(in.q[base + d]);
                const double k_value = static_cast<double>(in.k[base + d]);
                q_logical[base + d]  = q_value;
                k_logical[base + d]  = k_value;
                q_sumsq += q_value * q_value;
                k_sumsq += k_value * k_value;
            }
            if (normalize_qk) {
                constexpr double kNormEps = 1.0e-6;
                const double q_inv        = 1.0 / std::sqrt(q_sumsq + kNormEps);
                const double k_inv        = 1.0 / std::sqrt(k_sumsq + kNormEps);
                for (std::int64_t d = 0; d < S; ++d) {
                    q_logical[base + d] *= q_inv;
                    k_logical[base + d] *= k_inv;
                }
            }
        }
    }

    Result result;
    result.out.resize(value_size);
    result.final_state.resize(state_size);
    if (record_snapshots) { result.snapshots.resize(state_size * static_cast<std::size_t>(T)); }

    std::vector<double> state(static_cast<std::size_t>(S * S));
    std::vector<double> delta(static_cast<std::size_t>(S));
    for (std::int64_t h = 0; h < H_v; ++h) {
        const std::size_t state_base = static_cast<std::size_t>(h * S * S);
        for (std::int64_t i = 0; i < S * S; ++i) {
            state[static_cast<std::size_t>(i)] =
                static_cast<double>(in.state[state_base + static_cast<std::size_t>(i)]);
        }

        const std::int64_t qh = qk_head(h, H_qk, H_v);
        for (std::int64_t t = 0; t < T; ++t) {
            const std::size_t qk_base = static_cast<std::size_t>((t * H_qk + qh) * S);
            const std::size_t v_base  = static_cast<std::size_t>((t * H_v + h) * S);
            const std::size_t gb      = static_cast<std::size_t>(t * H_v + h);
            const double alpha        = std::exp(static_cast<double>(in.g[gb]));
            const double beta         = static_cast<double>(in.beta[gb]);

            for (std::int64_t row = 0; row < S; ++row) {
                double state_dot_k         = 0.0;
                const std::size_t row_base = static_cast<std::size_t>(row * S);
                for (std::int64_t col = 0; col < S; ++col) {
                    state_dot_k += state[row_base + static_cast<std::size_t>(col)] *
                                   k_logical[qk_base + static_cast<std::size_t>(col)];
                }
                delta[static_cast<std::size_t>(row)] =
                    beta * (static_cast<double>(in.v[v_base + static_cast<std::size_t>(row)]) -
                            alpha * state_dot_k);
            }

            for (std::int64_t row = 0; row < S; ++row) {
                const std::size_t row_base = static_cast<std::size_t>(row * S);
                const double row_delta     = delta[static_cast<std::size_t>(row)];
                for (std::int64_t col = 0; col < S; ++col) {
                    const std::size_t index = row_base + static_cast<std::size_t>(col);
                    state[index]            = alpha * state[index] +
                                   row_delta * k_logical[qk_base + static_cast<std::size_t>(col)];
                }
            }

            for (std::int64_t row = 0; row < S; ++row) {
                double state_dot_q         = 0.0;
                const std::size_t row_base = static_cast<std::size_t>(row * S);
                for (std::int64_t col = 0; col < S; ++col) {
                    state_dot_q += state[row_base + static_cast<std::size_t>(col)] *
                                   q_logical[qk_base + static_cast<std::size_t>(col)];
                }
                result.out[v_base + static_cast<std::size_t>(row)] = scale * state_dot_q;
            }

            if (record_snapshots) {
                const std::size_t snapshot_base =
                    static_cast<std::size_t>(t) * state_size + state_base;
                for (std::int64_t i = 0; i < S * S; ++i) {
                    result.snapshots[snapshot_base + static_cast<std::size_t>(i)] =
                        state[static_cast<std::size_t>(i)];
                }
            }
        }

        for (std::int64_t i = 0; i < S * S; ++i) {
            result.final_state[state_base + static_cast<std::size_t>(i)] =
                state[static_cast<std::size_t>(i)];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// I8 persistent-state codec. Mirrors the production kernel exactly at the
// codec level: per (value_head, dv row) of `dim` dk values,
//   row_max = max |x|  (FP32, exact maximum)
//   scale16 = __float2half_rn(row_max / 127.0f)
//   scale   = FP32(scale16)
//   code    = scale == 0 ? 0 : clamp(__float2int_rn(x / scale), -127, 127)
//   stored  = code * scale
// The host reference seeds its FP64 recurrence from the dequantized pool
// values and compares pool contents through the same round trip.
// ---------------------------------------------------------------------------

// Round-to-nearest-even FP32 -> FP16, matching __float2half_rn.
inline std::uint16_t i8_fp32_to_fp16_rn(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const float magnitude    = std::fabsf(value);
    if (std::isnan(value)) { return static_cast<std::uint16_t>(sign | 0x7E00U); }
    if (std::isinf(value) || magnitude >= 65520.0f) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    if (magnitude >= 65504.0f) {
        return static_cast<std::uint16_t>(sign | 0x7BFFU);
    }
    if (magnitude < std::ldexp(1.0f, -25)) {
        return sign;
    }
    if (magnitude < 6.103515625e-5f) {
        const std::uint32_t rounded =
            static_cast<std::uint32_t>(std::nearbyint(static_cast<double>(magnitude) * 16777216.0));
        if (rounded >= 1024U) { return static_cast<std::uint16_t>(sign | 0x0401U); }
        return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(rounded));
    }
    int exponent   = 0;
    const float mantissa = std::frexpf(magnitude, &exponent);
    const std::uint32_t rounded =
        static_cast<std::uint32_t>(std::nearbyint(static_cast<double>(mantissa) * 2048.0));
    // magnitude = mantissa * 2^exponent = (2 * mantissa) * 2^(exponent - 1); the half exponent
    // field is (exponent - 1) - 15 + 15 = exponent - 1 relative, i.e. exponent + 14.
    std::uint16_t half_exponent = static_cast<std::uint16_t>(exponent + 14);
    std::uint16_t fraction      = static_cast<std::uint16_t>(rounded - 1024U);
    if (rounded == 2048U) {
        fraction      = 0;
        ++half_exponent;
    }
    return static_cast<std::uint16_t>(sign | (half_exponent << 10) | fraction);
}

// Exact FP16 bits -> FP32, matching __half2float.
inline float i8_fp16_bits_to_fp32(std::uint16_t bits) {
    const std::uint32_t sign     = (bits >> 15) & 1U;
    const std::uint32_t exponent = (bits >> 10) & 0x1FU;
    const std::uint32_t fraction = bits & 0x3FFU;
    double value;
    if (exponent == 0U) {
        value = static_cast<double>(fraction) * std::ldexp(1.0, -24);
    } else if (exponent == 31U) {
        value = (fraction == 0U) ? INFINITY : NAN;
    } else {
        value = (1.0 + static_cast<double>(fraction) / 1024.0) *
                std::ldexp(1.0, static_cast<int>(exponent) - 15);
    }
    const float result = static_cast<float>(value);
    return sign != 0U ? -result : result;
}

// Quantize one dv row (dim dk values); dim is kStateDim (128) in every production use.
inline void i8_quantize_row(const float* values, std::int64_t dim, std::int8_t* codes,
                            std::uint16_t* scale_bits) {
    float row_max = 0.0f;
    for (std::int64_t index = 0; index < dim; ++index) {
        row_max = std::fmaxf(row_max, std::fabsf(values[static_cast<std::size_t>(index)]));
    }
    const std::uint16_t scale16 = i8_fp32_to_fp16_rn(row_max / 127.0f);
    const float scale           = i8_fp16_bits_to_fp32(scale16);
    for (std::int64_t index = 0; index < dim; ++index) {
        int code = 0;
        if (scale != 0.0f) {
            code = static_cast<int>(std::nearbyint(
                values[static_cast<std::size_t>(index)] / scale));
            if (code > 127) { code = 127; }
            if (code < -127) { code = -127; }
        }
        codes[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(code);
    }
    *scale_bits = scale16;
}

inline void i8_dequantize_row(const std::int8_t* codes, std::uint16_t scale_bits,
                              std::int64_t dim, float* values) {
    const float scale = i8_fp16_bits_to_fp32(scale_bits);
    for (std::int64_t index = 0; index < dim; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>(codes[static_cast<std::size_t>(index)]) * scale;
    }
}

struct I8StateQuantization {
    // Same element layout as the FP32 state (ne fastest-first {dim, dim, value_heads}):
    // (h, d, c) at h * dim * dim + d * dim + c.
    std::vector<std::int8_t> codes;
    // Single-slot scale plane, ne fastest-first {dim, value_heads}: (h, d) at h * dim + d.
    std::vector<std::uint16_t> scale;
};

inline I8StateQuantization i8_quantize_state(const std::vector<float>& state, std::int64_t dim,
                                             std::int64_t value_heads) {
    I8StateQuantization result;
    result.codes.resize(state.size());
    result.scale.resize(static_cast<std::size_t>(dim * value_heads));
    std::vector<std::int8_t> row_codes(static_cast<std::size_t>(dim));
    for (std::int64_t h = 0; h < value_heads; ++h) {
        for (std::int64_t d = 0; d < dim; ++d) {
            const float* row =
                state.data() + static_cast<std::size_t>(h * dim * dim + d * dim);
            i8_quantize_row(row, dim, row_codes.data(),
                            result.scale.data() + static_cast<std::size_t>(h * dim + d));
            std::copy(row_codes.begin(), row_codes.end(),
                      result.codes.begin() + static_cast<std::size_t>(h * dim * dim + d * dim));
        }
    }
    return result;
}

inline std::vector<float> i8_dequantize_state(const std::vector<std::int8_t>& codes,
                                              const std::vector<std::uint16_t>& scale,
                                              std::int64_t dim, std::int64_t value_heads) {
    std::vector<float> state(codes.size());
    for (std::int64_t h = 0; h < value_heads; ++h) {
        for (std::int64_t d = 0; d < dim; ++d) {
            const float row_scale =
                i8_fp16_bits_to_fp32(scale[static_cast<std::size_t>(h * dim + d)]);
            float* out = state.data() + static_cast<std::size_t>(h * dim * dim + d * dim);
            for (std::int64_t c = 0; c < dim; ++c) {
                out[static_cast<std::size_t>(c)] =
                    static_cast<float>(codes[static_cast<std::size_t>(h * dim * dim + d * dim + c)]) *
                    row_scale;
            }
        }
    }
    return state;
}

} // namespace ninfer::test::gdn_ref
