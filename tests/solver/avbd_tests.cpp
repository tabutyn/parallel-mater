// SPDX-License-Identifier: MIT
#include "avbd_cases.hpp"
#include "../../src/convex_plane_dedup.hpp"
#include <cmath>
#include <cstdio>

static bool convex_plane_upload_regression() {
    struct Normal { float x, y, z; };
    struct Plane { Normal normal; float offset; };
    const Plane faces[6]{{{1,0,0},1},{{-1,0,0},1},{{0,1,0},1},
                         {{0,-1,0},1},{{0,0,1},1},{{0,0,-1},1}};
    std::vector<Plane> tessellated;
    // Interleave the six exact support planes: their repetitions need not
    // remain adjacent after BVH triangle reordering.
    for (unsigned subdivision = 0; subdivision < 1024; ++subdivision)
        for (const auto &face : faces) tessellated.push_back(face);
    const auto unique = parallel_mater::detail::unique_convex_planes(tessellated);
    if (unique.size() != 6 || tessellated.size() != 6144) return false;
    for (unsigned index = 0; index < tessellated.size(); ++index) {
        const auto &expected = faces[index % 6];
        const auto &actual = tessellated[index];
        if (actual.normal.x != expected.normal.x || actual.normal.y != expected.normal.y ||
            actual.normal.z != expected.normal.z || actual.offset != expected.offset) return false;
    }
    for (unsigned index = 0; index < 6; ++index) {
        const auto &expected = faces[index], &actual = unique[index];
        if (actual.normal.x != expected.normal.x || actual.normal.y != expected.normal.y ||
            actual.normal.z != expected.normal.z || actual.offset != expected.offset) return false;
    }
    // No tolerance merge: even a one-ULP offset difference remains a plane.
    tessellated.push_back({faces[0].normal,std::nextafter(1.0F,2.0F)});
    const auto distinct = parallel_mater::detail::unique_convex_planes(tessellated);
    return distinct.size() == 7 && distinct.back().offset == tessellated.back().offset;
}

int main() {
    if (!convex_plane_upload_regression()) {
        std::fprintf(stderr, "Exact convex-plane upload deduplication regression failed\n");
        return 1;
    }
    std::printf("1 host convex-plane upload deduplication regression passed\n");
    for (unsigned index = 0; index < avbd_cases::count; ++index) {
        const auto result = avbd_cases::evaluate(index);
        if (!avbd_cases::validate(index, result)) {
            std::fprintf(stderr, "AVBD fixture %u failed (%s)\n", index, avbd_cases::name(index));
            return 1;
        }
    }
    std::printf("%u portable AVBD block, dual, material and friction fixtures passed\n", avbd_cases::count);
}
