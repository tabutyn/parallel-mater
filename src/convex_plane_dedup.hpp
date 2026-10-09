// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <map>
#include <vector>

namespace parallel_mater::detail {

// Host upload helper. Preserve the first occurrence of every exact plane;
// never merge nearby planes or modify the original triangle-indexed array.
template<class Plane>
std::vector<Plane> unique_convex_planes(const std::vector<Plane> &planes) {
    std::map<std::array<float, 4>, bool> seen;
    std::vector<Plane> unique;
    unique.reserve(planes.size());
    for (const auto &plane : planes) {
        const std::array<float, 4> key{
            plane.normal.x, plane.normal.y, plane.normal.z, plane.offset};
        if (seen.emplace(key, true).second) unique.push_back(plane);
    }
    return unique;
}

} // namespace parallel_mater::detail
