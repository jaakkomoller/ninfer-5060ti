#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "product/prompt_input/prompt_input.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include "targets/qwen3_5_9b/impl/diagnostic/activation_dump_access.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace target = ninfer::targets::qwen3_5_9b;
namespace detail = target::detail;
namespace schedule = detail::schedule;

std::uint64_t parse_u64(std::string_view text, std::string_view label) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(label) + ": invalid number");
    }
    return value;
}

struct TopLogit {
    ninfer::TokenId token;
    float logit;
};

class LogitsCollector {
public:
    void set_decode_context(std::uint32_t step, std::uint32_t position) {
        decode_step_ = step;
        decode_position_ = position;
    }

    static void text_callback(void* opaque, schedule::TapId id, int layer, schedule::Phase phase,
                              const ninfer::Tensor& tensor, cudaStream_t stream) {
        static_cast<LogitsCollector*>(opaque)->capture(id, layer, phase, tensor, stream);
    }

    static void vision_callback(void*, std::uint32_t, schedule::VisionTapId, int,
                                const ninfer::Tensor&, cudaStream_t) {}

    [[nodiscard]] const std::vector<TopLogit>& top_logits() const { return top_; }

private:
    void capture(schedule::TapId id, int layer, schedule::Phase phase,
                 const ninfer::Tensor& tensor, cudaStream_t stream) {
        if (id == schedule::TapId::AfterMixer) {
            if (tensor.dtype != ninfer::DType::BF16) { return; }
            const std::size_t elements = static_cast<std::size_t>(tensor.numel());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::uint16_t> bits(elements);
            CUDA_CHECK(cudaMemcpy(bits.data(), tensor.data, elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            bool nan = false, inf = false;
            float maxv = 0.0f, minv = 0.0f, sum = 0.0f;
            for (std::size_t i = 0; i < elements; ++i) {
                const float value = std::bit_cast<float>(static_cast<std::uint32_t>(bits[i]) << 16U);
                if (std::isnan(value)) { nan = true; }
                if (std::isinf(value)) { inf = true; }
                if (i == 0) { maxv = minv = value; }
                maxv = std::max(maxv, value);
                minv = std::min(minv, value);
                sum += std::isfinite(value) ? value : 0.0f;
            }
            const float mean = sum / static_cast<float>(elements);
            std::cout << "MIXER phase="
                      << (phase == schedule::Phase::Prefill ? "prefill" : "decode")
                      << " step=" << decode_step_ << " layer=" << layer << " T=" << tensor.ne[1]
                      << " nan=" << nan << " inf=" << inf << " min=" << minv << " max=" << maxv
                      << " mean=" << mean << "\n";
            return;
        }
        if (id != schedule::TapId::AfterLogits) { return; }
        if (tensor.dtype != ninfer::DType::BF16) { return; }
        const std::size_t elements = static_cast<std::size_t>(tensor.numel());
        if (elements == 0) { return; }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> bits(elements);
        CUDA_CHECK(cudaMemcpy(bits.data(), tensor.data, elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        const int T = tensor.ne[1];
        std::vector<std::pair<float, ninfer::TokenId>> ranked;
        ranked.reserve(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            const float value = std::bit_cast<float>(static_cast<std::uint32_t>(bits[i]) << 16U);
            ranked.emplace_back(value, static_cast<ninfer::TokenId>(i % tensor.ne[0]));
        }
        std::partial_sort(ranked.begin(), ranked.begin() + std::min<std::size_t>(10, ranked.size()),
                          ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "LOGITS phase="
                  << (phase == schedule::Phase::Prefill ? "prefill" : "decode")
                  << " step=" << decode_step_ << " pos=" << decode_position_ << " T=" << T << "\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, ranked.size()); ++i) {
            std::cout << "  top[" << i << "] token=" << ranked[i].second
                      << " logit=" << ranked[i].first << "\n";
        }
    }

    std::uint32_t decode_step_ = 0;
    std::uint32_t decode_position_ = 0;
    std::vector<TopLogit> top_;
};

struct Options {
    std::filesystem::path weights;
    std::filesystem::path messages;
    std::uint32_t decode = 8;
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t max_context = 0;
    int device = 0;
    bool enable_thinking = true;
};

std::string usage_text() {
    return "usage: ninfer-qwen3_5_9b-logits --weights MODEL.ninfer --messages FILE.json\n"
           "       [--decode N] [--prefill-chunk N] [--max-context N] [--device N] [--no-thinking]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](std::string_view flag) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string(flag) + " requires value"); }
            return argv[index];
        };
        if (argument == "--weights") {
            options.weights = value(argument);
        } else if (argument == "--messages") {
            options.messages = value(argument);
        } else if (argument == "--decode") {
            options.decode = static_cast<std::uint32_t>(parse_u64(value(argument), "decode"));
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(parse_u64(value(argument), "prefill-chunk"));
        } else if (argument == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(parse_u64(value(argument), "max-context"));
        } else if (argument == "--device") {
            options.device = static_cast<int>(parse_u64(value(argument), "device"));
        } else if (argument == "--no-thinking") {
            options.enable_thinking = false;
        } else if (argument == "--help" || argument == "-h") {
            std::cout << usage_text();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.weights.empty() || options.messages.empty()) {
        throw std::invalid_argument("--weights and --messages are required");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        ninfer::EngineOptions engine;
        engine.artifact_path = options.weights;
        engine.device = options.device;
        engine.prefill_chunk = options.prefill_chunk;
        engine.kv_cache = ninfer::KvCacheStorage::BFloat16;
        engine.speculative.backend = ninfer::SpeculativeBackend::None;
        engine.speculative.draft_tokens = 0;
        engine.use_cuda_graph = false;

        ninfer::DeviceContext device(options.device);
        ninfer::artifact::Reader reader(options.weights);
        const auto weights_profile = target::Package::resolve_weights(reader.identity());
        ninfer::artifact::Binder binder(reader);
        auto load_plan = target::Package::plan_load(binder, engine, weights_profile);
        auto materialized = ninfer::artifact::materialize(reader, load_plan.materialization(), device, nullptr);
        auto model = target::Package::construct_loaded_model(std::move(load_plan), std::move(materialized));
        auto frontend = target::Package::make_frontend(*model);
        auto prompt = frontend.prepare(
            ninfer::product::prompt_from_messages(options.messages, options.enable_thinking, true));
        const ninfer::PromptSummary prompt_summary = prompt.summary();
        std::cout << "prompt tokens: " << prompt_summary.prompt_tokens << "\n";

        const std::uint64_t required =
            static_cast<std::uint64_t>(prompt_summary.prompt_tokens) + options.decode;
        engine.max_context = options.max_context != 0
                                 ? options.max_context
                                 : static_cast<std::uint32_t>(std::max<std::uint64_t>(2048, required));

        auto sequence_plan = target::Package::plan_sequence(device, engine, weights_profile);
        const std::size_t request_transient_capacity = sequence_plan.request_transient_capacity_bytes();
        auto program = target::Package::create_program(*model, std::move(sequence_plan), device);
        ninfer::runtime::RequestMemory transient(device, request_transient_capacity);
        ninfer::ExecutionOptions execution;
        execution.requested_output_tokens = options.decode;
        execution.sampling = {};
        execution.allow_prefix_reuse = false;
        auto request_plan = program->plan_request(prompt, execution);
        const auto summary = request_plan.summary();

        LogitsCollector collector;
        detail::ActivationDumpAccess::attach(*program, &collector, &LogitsCollector::text_callback,
                                             &LogitsCollector::vision_callback);
        std::vector<ninfer::TokenId> generated;
        try {
            transient.activate(summary.transient_bytes, summary.transient_alignment);
            auto first = program->begin(std::move(prompt), std::move(request_plan), transient.region());
            transient.deactivate();
            const auto tokens = first.round.tokens;
            generated.insert(generated.end(), tokens.begin(), tokens.end());
            std::cout << "prefill round tokens:";
            for (const auto t : tokens) { std::cout << " " << t; }
            std::cout << "\n";
            program->resolve_pending(static_cast<std::uint32_t>(generated.size()), false);

            while (generated.size() < options.decode) {
                collector.set_decode_context(static_cast<std::uint32_t>(generated.size() - 1),
                                             program->materialized_tokens());
                auto round = program->decode_round(ninfer::runtime::RoundBudget{
                    .generated_tokens_remaining = options.decode - static_cast<std::uint32_t>(generated.size())});
                const auto tokens = round.tokens;
                generated.insert(generated.end(), tokens.begin(), tokens.end());
                std::cout << "decode round tokens:";
                for (const auto t : tokens) { std::cout << " " << t; }
                std::cout << "\n";
                program->resolve_pending(static_cast<std::uint32_t>(tokens.size()), false);
            }
        } catch (...) {
            transient.deactivate();
            program->abort_request();
            detail::ActivationDumpAccess::detach(*program);
            throw;
        }
        detail::ActivationDumpAccess::detach(*program);

        std::cout << "generated ids:";
        for (const auto t : generated) { std::cout << " " << t; }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
