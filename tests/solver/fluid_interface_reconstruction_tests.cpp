// SPDX-License-Identifier: MIT
#include "simulation/fluid_dispatch.hpp"
#include "simulation/primitives.hpp"
#include "simulation/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

using namespace parallel_mater;
using namespace parallel_mater::simulation;

namespace {
struct EndpointReconstruction {
    std::vector<std::uint32_t> total_neighbors;
    std::vector<std::uint32_t> cross_neighbors;
    std::vector<float> phase_labels;
    std::vector<Vec3> normals;
    std::vector<float> curvatures;
    std::vector<std::uint32_t> flags;
    std::uint32_t interface_samples{};
    std::uint32_t under_resolved{};
    std::uint32_t degenerate{};
    std::uint32_t curvature_clamps{};
};

struct Reconstruction {
    EndpointReconstruction first;
    EndpointReconstruction second;
};

float dot(Vec3 first, Vec3 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

float length(Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 scaled(Vec3 value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

bool finite(Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

Reconstruction reconstruct(const std::vector<Vec3> &first_positions,
    std::uint32_t first_count, const std::vector<Vec3> &second_positions,
    std::uint32_t second_count, FluidDispatchConstants first_constants,
    FluidDispatchConstants second_constants) {
    const auto first_capacity =
        static_cast<std::uint32_t>(first_positions.size());
    const auto second_capacity =
        static_cast<std::uint32_t>(second_positions.size());
    const auto total_capacity = first_capacity + second_capacity;
    CHECK(first_capacity != 0U && second_capacity != 0U);
    CHECK(first_count <= first_capacity && second_count <= second_capacity);

    const float pair_support = std::max(
        first_constants.support_radius, second_constants.support_radius);
    std::vector<std::uint64_t> keys[2];
    std::vector<std::uint32_t> identities[2];
    for (unsigned slot = 0U; slot < 2U; ++slot) {
        keys[slot].resize(total_capacity);
        identities[slot].resize(total_capacity);
    }
    fluid_pair_cell_keys((total_capacity + 127U) / 128U, 128U, {},
        first_positions.data(), &first_count, first_capacity,
        second_positions.data(), &second_count, second_capacity,
        1.0F / pair_support, keys[0].data(), identities[0].data());
    std::vector<std::byte> workspace(
        radix_sort_scratch_bytes(total_capacity));
    CHECK(radix_sort_pairs_u64_u32({}, keys[0].data(), keys[1].data(),
        identities[0].data(), identities[1].data(), total_capacity,
        workspace.data(), workspace.size()) == PM_SUCCESS);

    Reconstruction result;
    auto prepare = [](EndpointReconstruction &endpoint,
                       std::uint32_t capacity) {
        endpoint.total_neighbors.assign(capacity, 0U);
        endpoint.cross_neighbors.assign(capacity, 0U);
        endpoint.phase_labels.assign(capacity,
            std::numeric_limits<float>::quiet_NaN());
        const auto infinite = std::numeric_limits<float>::infinity();
        endpoint.normals.assign(capacity, {infinite, infinite, infinite});
        endpoint.curvatures.assign(capacity, infinite);
        endpoint.flags.assign(capacity,
            std::numeric_limits<std::uint32_t>::max());
    };
    prepare(result.first, first_capacity);
    prepare(result.second, second_capacity);
    std::vector<float> first_density(first_capacity);
    std::vector<float> second_density(second_capacity);
    std::uint32_t first_neighbor_overflow{};
    std::uint32_t second_neighbor_overflow{};
    std::uint32_t first_maximum_neighbors{};
    std::uint32_t second_maximum_neighbors{};
    fluid_pair_neighbor_counts((total_capacity + 127U) / 128U, 128U, {},
        first_positions.data(), &first_count, first_capacity,
        first_constants.maximum_neighbors, second_positions.data(),
        &second_count, second_capacity, second_constants.maximum_neighbors,
        keys[1].data(), identities[1].data(), total_capacity,
        first_constants, second_constants,
        result.first.total_neighbors.data(),
        result.first.cross_neighbors.data(), first_density.data(),
        result.second.total_neighbors.data(),
        result.second.cross_neighbors.data(), second_density.data(),
        &first_neighbor_overflow, &second_neighbor_overflow,
        &first_maximum_neighbors, &second_maximum_neighbors);

    fluid_immiscible_interface_normals(
        (total_capacity + 127U) / 128U, 128U, {}, first_positions.data(),
        &first_count, first_capacity, result.first.total_neighbors.data(),
        result.first.cross_neighbors.data(), result.first.phase_labels.data(),
        result.first.normals.data(), result.first.flags.data(),
        second_positions.data(), &second_count, second_capacity,
        result.second.total_neighbors.data(),
        result.second.cross_neighbors.data(),
        result.second.phase_labels.data(), result.second.normals.data(),
        result.second.flags.data(), keys[1].data(), identities[1].data(),
        total_capacity, first_constants, second_constants,
        &result.first.interface_samples, &result.second.interface_samples,
        &result.first.under_resolved, &result.second.under_resolved,
        &result.first.degenerate, &result.second.degenerate);
    fluid_immiscible_interface_curvature(
        (total_capacity + 127U) / 128U, 128U, {}, first_positions.data(),
        &first_count, first_capacity, result.first.normals.data(),
        result.first.curvatures.data(), result.first.flags.data(),
        second_positions.data(), &second_count, second_capacity,
        result.second.normals.data(), result.second.curvatures.data(),
        result.second.flags.data(), keys[1].data(), identities[1].data(),
        total_capacity, first_constants, second_constants,
        &result.first.under_resolved, &result.second.under_resolved,
        &result.first.degenerate, &result.second.degenerate,
        &result.first.curvature_clamps, &result.second.curvature_clamps);
    return result;
}

FluidDispatchConstants material(float spacing) {
    FluidDispatchConstants result{};
    result.particle_radius = 0.45F * spacing;
    result.support_radius = 2.5F * spacing;
    result.rest_density = 1'000.0F;
    result.rest_particle_volume = spacing * spacing * spacing;
    result.maximum_neighbors = 512U;
    return result;
}

void check_bounded_and_finite(const EndpointReconstruction &endpoint,
    std::uint32_t count, float curvature_limit) {
    CHECK(endpoint.interface_samples <= count);
    CHECK(endpoint.under_resolved <= count);
    CHECK(endpoint.degenerate <= count);
    CHECK(endpoint.curvature_clamps <= count);
    for (std::size_t index = 0U; index < endpoint.normals.size(); ++index) {
        CHECK(finite(endpoint.normals[index]));
        CHECK(std::isfinite(endpoint.phase_labels[index]));
        CHECK(std::isfinite(endpoint.curvatures[index]));
        CHECK(std::fabs(endpoint.curvatures[index]) <= curvature_limit);
    }
}

void test_planar_orientation_and_endpoint_reversal() {
    constexpr float spacing = 0.2F;
    std::vector<Vec3> first;
    std::vector<Vec3> second;
    for (int z = -5; z <= 5; ++z) {
        for (int y = -5; y <= 5; ++y) {
            for (int x = -5; x <= 4; ++x) {
                const Vec3 position{
                    (static_cast<float>(x) + 0.5F) * spacing,
                    static_cast<float>(y) * spacing,
                    static_cast<float>(z) * spacing};
                (position.x < 0.0F ? first : second).push_back(position);
            }
        }
    }
    const auto constants = material(spacing);
    const auto result = reconstruct(first,
        static_cast<std::uint32_t>(first.size()), second,
        static_cast<std::uint32_t>(second.size()), constants, constants);
    CHECK(result.first.degenerate == 0U);
    CHECK(result.second.degenerate == 0U);
    CHECK(result.first.interface_samples != 0U);
    CHECK(result.second.interface_samples != 0U);
    const Vec3 positive_x{1.0F, 0.0F, 0.0F};
    std::uint32_t inspected = 0U;
    const auto inspect = [&](const std::vector<Vec3> &positions,
                             const EndpointReconstruction &endpoint,
                             float phase) {
        for (std::size_t index = 0U; index < positions.size(); ++index) {
            CHECK(endpoint.phase_labels[index] == phase);
            if (std::fabs(positions[index].y) > spacing ||
                std::fabs(positions[index].z) > spacing ||
                (endpoint.flags[index] & k_fluid_interface_sample) == 0U)
                continue;
            CHECK(dot(endpoint.normals[index], positive_x) > 0.999F);
            if (std::fabs(endpoint.curvatures[index]) >= 1.0e-4F)
                std::fprintf(stderr,
                    "planar sample (%g, %g, %g): curvature=%g normal=(%g,%g,%g) flags=%u\n",
                    positions[index].x, positions[index].y,
                    positions[index].z, endpoint.curvatures[index],
                    endpoint.normals[index].x, endpoint.normals[index].y,
                    endpoint.normals[index].z, endpoint.flags[index]);
            CHECK(std::fabs(endpoint.curvatures[index]) < 1.0e-4F);
            ++inspected;
        }
    };
    inspect(first, result.first, -1.0F);
    inspect(second, result.second, 1.0F);
    CHECK(inspected >= 18U);

    const auto reversed = reconstruct(second,
        static_cast<std::uint32_t>(second.size()), first,
        static_cast<std::uint32_t>(first.size()), constants, constants);
    for (std::size_t index = 0U; index < first.size(); ++index) {
        if ((result.first.flags[index] & k_fluid_interface_sample) == 0U)
            continue;
        CHECK((reversed.second.flags[index] & k_fluid_interface_sample) != 0U);
        CHECK(dot(result.first.normals[index],
                  reversed.second.normals[index]) < -0.999F);
    }
    for (std::size_t index = 0U; index < second.size(); ++index) {
        if ((result.second.flags[index] & k_fluid_interface_sample) == 0U)
            continue;
        CHECK((reversed.first.flags[index] & k_fluid_interface_sample) != 0U);
        CHECK(dot(result.second.normals[index],
                  reversed.first.normals[index]) < -0.999F);
    }
}

struct SphereQuality {
    float mean_normal_alignment{};
    float mean_curvature{};
    float curvature_error{};
    std::uint32_t samples{};
};

SphereQuality sphere_quality(float spacing) {
    constexpr float radius = 1.0F;
    const auto constants = material(spacing);
    const float extent = radius + constants.support_radius;
    const int limit = static_cast<int>(std::ceil(extent / spacing)) + 1;
    std::vector<Vec3> first;
    std::vector<Vec3> second;
    for (int z = -limit; z <= limit; ++z) {
        for (int y = -limit; y <= limit; ++y) {
            for (int x = -limit; x <= limit; ++x) {
                const Vec3 position{
                    (static_cast<float>(x) + 0.13F) * spacing,
                    (static_cast<float>(y) + 0.27F) * spacing,
                    (static_cast<float>(z) + 0.39F) * spacing};
                if (std::fabs(position.x) > extent ||
                    std::fabs(position.y) > extent ||
                    std::fabs(position.z) > extent)
                    continue;
                (length(position) <= radius ? first : second)
                    .push_back(position);
            }
        }
    }
    const auto result = reconstruct(first,
        static_cast<std::uint32_t>(first.size()), second,
        static_cast<std::uint32_t>(second.size()), constants, constants);
    check_bounded_and_finite(result.first,
        static_cast<std::uint32_t>(first.size()),
        2.0F / constants.particle_radius);
    check_bounded_and_finite(result.second,
        static_cast<std::uint32_t>(second.size()),
        2.0F / constants.particle_radius);

    SphereQuality quality;
    const auto accumulate = [&](const std::vector<Vec3> &positions,
                                const EndpointReconstruction &endpoint) {
        for (std::size_t index = 0U; index < positions.size(); ++index) {
            const float radial_distance = length(positions[index]);
            if (std::fabs(radial_distance - radius) > 0.6F * spacing ||
                (endpoint.flags[index] & k_fluid_interface_sample) == 0U ||
                (endpoint.flags[index] & k_fluid_interface_degenerate) != 0U)
                continue;
            const Vec3 radial = scaled(
                positions[index], 1.0F / radial_distance);
            quality.mean_normal_alignment +=
                dot(endpoint.normals[index], radial);
            quality.mean_curvature += endpoint.curvatures[index];
            ++quality.samples;
        }
    };
    accumulate(first, result.first);
    accumulate(second, result.second);
    CHECK(quality.samples >= 80U);
    quality.mean_normal_alignment /= static_cast<float>(quality.samples);
    quality.mean_curvature /= static_cast<float>(quality.samples);
    quality.curvature_error = std::fabs(quality.mean_curvature - 2.0F);
    CHECK(quality.mean_normal_alignment > 0.97F);
    return quality;
}

void test_spherical_curvature_converges() {
    const SphereQuality coarse = sphere_quality(0.4F);
    const SphereQuality medium = sphere_quality(0.25F);
    const SphereQuality fine = sphere_quality(0.16F);
    CHECK(medium.curvature_error < coarse.curvature_error);
    CHECK(fine.curvature_error < medium.curvature_error);
    CHECK(fine.curvature_error < 0.05F);
    std::printf("sphere mean curvature: %.6f -> %.6f -> %.6f\n",
        coarse.mean_curvature, medium.mean_curvature, fine.mean_curvature);
}

void test_empty_and_degenerate_neighborhoods() {
    constexpr float spacing = 0.2F;
    const auto constants = material(spacing);
    std::vector<Vec3> first{{0.0F, 0.0F, 0.0F}};
    std::vector<Vec3> second{{0.1F, 0.0F, 0.0F}};
    auto result = reconstruct(first, 1U, second, 1U, constants, constants);
    for (const auto *endpoint : {&result.first, &result.second}) {
        CHECK(endpoint->interface_samples == 1U);
        CHECK(endpoint->under_resolved == 1U);
        CHECK(endpoint->degenerate == 1U);
        CHECK(endpoint->curvature_clamps == 0U);
        CHECK((endpoint->flags[0] & k_fluid_interface_sample) != 0U);
        CHECK((endpoint->flags[0] & k_fluid_interface_under_resolved) != 0U);
        CHECK((endpoint->flags[0] & k_fluid_interface_degenerate) != 0U);
        CHECK(length(endpoint->normals[0]) == 0.0F);
        CHECK(endpoint->curvatures[0] == 0.0F);
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    first = {{nan, nan, nan}, {nan, nan, nan}};
    second = {{nan, nan, nan}, {nan, nan, nan}, {nan, nan, nan}};
    result = reconstruct(first, 0U, second, 0U, constants, constants);
    for (const auto *endpoint : {&result.first, &result.second}) {
        CHECK(endpoint->interface_samples == 0U);
        CHECK(endpoint->under_resolved == 0U);
        CHECK(endpoint->degenerate == 0U);
        CHECK(endpoint->curvature_clamps == 0U);
        for (std::size_t index = 0U; index < endpoint->normals.size(); ++index) {
            CHECK(endpoint->phase_labels[index] == 0.0F);
            CHECK(length(endpoint->normals[index]) == 0.0F);
            CHECK(endpoint->curvatures[index] == 0.0F);
            CHECK(endpoint->flags[index] == 0U);
        }
    }
}
} // namespace

int main() {
    test_planar_orientation_and_endpoint_reversal();
    test_spherical_curvature_converges();
    test_empty_and_degenerate_neighborhoods();
    std::puts("PASS: bounded immiscible interface reconstruction");
}
