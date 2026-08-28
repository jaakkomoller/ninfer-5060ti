#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q3_a16_conformance() {
    // The Q3 profile quantizes only mlp/down (N=5120, K=17408), so the conformance set is that
    // single registered shape through every closed schedule. run_shape checks b-1/b/b+1 for every
    // route start plus one interior point for every region, through the public Op.
    constexpr std::array<std::int32_t, 5> kRouteStarts{2, 17, 33, 49, 129};
    constexpr std::array<std::int32_t, 6> kRouteInteriors{1, 8, 24, 40, 96, 256};
    return ninfer::test::linear_add::run_shape(
        "Q3_A16 LinearAdd", WeightFormat::Q3G64F16S,
        ShapeCase{5120, 17408, 419U, kRouteStarts, kRouteInteriors});
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q3_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q3_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}