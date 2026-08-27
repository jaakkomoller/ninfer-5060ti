#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                    1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // Exact per-page table (driver-granularity rounding makes the reservation non-affine).
    // Steps are 1900, 200, 200, 200: the affine closed form with the first-page stride
    // undershoots, so resolution must search the table.
    const ninfer::runtime::SequenceCapacityCurve exact_curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 1900,
        .exact_reservations                   = {1000, 2900, 3100, 3300, 3500},
    };

    const auto exact_pick =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(0), exact_curve,
                                             3450);
    failures += check(exact_pick.main_page_groups == 5 && exact_pick.resolved_tokens == 320 &&
                          exact_pick.runtime_reservation_bytes == 3300 &&
                          exact_pick.planned_slack_bytes == 150,
                      "automatic KV capacity missed the largest fitting page count on the exact table");

    const auto exact_minimum =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(0), exact_curve,
                                             1000);
    failures += check(exact_minimum.main_page_groups == 2 &&
                          exact_minimum.planned_slack_bytes == 0,
                      "automatic KV capacity did not land on the exact minimum");

    const auto exact_explicit = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), exact_curve, 3450);
    failures += check(exact_explicit.main_page_groups == 3 &&
                          exact_explicit.runtime_reservation_bytes == 2900,
                      "explicit KV capacity ignored the exact per-page reservation");

    failures += check(exact_curve.reservation_bytes(5) == 3300,
                      "reservation_bytes did not prefer the exact table");

    bool exact_insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(0),
                                                   exact_curve, 999);
    } catch (const std::invalid_argument&) { exact_insufficient_rejected = true; }
    failures +=
        check(exact_insufficient_rejected, "automatic KV capacity accepted below the exact minimum");

    const auto bad_span = [&] { auto copy = exact_curve; copy.exact_reservations.pop_back(); return copy; }();
    bool bad_span_rejected = false;
    try { (void)bad_span.reservation_bytes(2); } catch (const std::invalid_argument&) {
        bad_span_rejected = true;
    }
    failures += check(bad_span_rejected, "capacity curve accepted a truncated exact table");

    const auto bad_monotone = [&] { auto copy = exact_curve; copy.exact_reservations[3] = 3000; return copy; }();
    bool bad_monotone_rejected = false;
    try { (void)bad_monotone.reservation_bytes(3); } catch (const std::invalid_argument&) {
        bad_monotone_rejected = true;
    }
    failures += check(bad_monotone_rejected, "capacity curve accepted a non-increasing exact table");

    const auto bad_base = [&] { auto copy = exact_curve; copy.minimum_device_reservation_bytes = 999; return copy; }();
    bool bad_base_rejected = false;
    try { (void)bad_base.reservation_bytes(2); } catch (const std::invalid_argument&) {
        bad_base_rejected = true;
    }
    failures += check(bad_base_rejected,
                      "capacity curve accepted an exact table whose base differs from the minimum");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
