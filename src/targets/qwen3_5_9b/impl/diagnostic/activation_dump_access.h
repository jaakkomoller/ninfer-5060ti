#pragma once

#include <ninfer/targets/qwen3_5_9b/package.h>
#include <ninfer/targets/qwen3_6/diagnostics.h>

namespace ninfer::targets::qwen3_5_9b::detail {

namespace schedule {
using Phase             = qwen3_6::TextPhase;
using TapId             = qwen3_6::TextTapId;
using VisionTapId       = qwen3_6::VisionTapId;
using TextTapCallback   = qwen3_6::TextTapCallback;
using VisionTapCallback = qwen3_6::VisionTapCallback;
} // namespace schedule

class ActivationDumpAccess {
public:
    static void attach(Package::Program& program, void* context, schedule::TextTapCallback text,
                       schedule::VisionTapCallback vision = nullptr);
    static void detach(Package::Program& program) noexcept;
};

} // namespace ninfer::targets::qwen3_5_9b::detail