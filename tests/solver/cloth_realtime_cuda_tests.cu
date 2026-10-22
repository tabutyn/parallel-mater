// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "simulation/cloth_realtime.hpp"
#include "simulation/types.hpp"
#include "parallel_mater_cloth_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

struct HostDeformableNeighbor {
    unsigned index{};
    float rest_length{};
    float compliance{};
    unsigned bond{};
};

static_assert(sizeof(Vec3) == sizeof(Vec3_0));
static_assert(sizeof(HostDeformableNeighbor) ==
              sizeof(DeformableNeighbor_0));

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    count = std::max<std::size_t>(count, 1U);
    if (cudaMallocManaged(&result, count * sizeof(T)) != cudaSuccess)
        return nullptr;
    std::fill(result, result + count, T{});
    return result;
}

float length(const Vec3 &value) {
    return std::sqrt(value.x * value.x + value.y * value.y +
        value.z * value.z);
}

Vec3 subtract_host(const Vec3 &left, const Vec3 &right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

float dot_host(const Vec3 &left, const Vec3 &right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3 cross_host(const Vec3 &left, const Vec3 &right) {
    return {left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x};
}

struct Fixture {
    unsigned capacity{};
    unsigned neighbor_capacity{};
    unsigned triangle_capacity{};
    Vec3 *positions{};
    Vec3 *velocities{};
    Vec3 *previous{};
    Vec3 *position_deltas{};
    Vec3 *gradients{};
    Vec3 *surface_positions{};
    float *correction_used{};
    float *inverse_masses{};
    float *characteristic_lengths{};
    float *contact_offsets{};
    float *lambda{};
    unsigned *offsets{};
    unsigned *indices{};
    unsigned *corner_offsets{};
    unsigned *corner_indices{};
    unsigned *time_advances{};
    unsigned *status{};
    HostDeformableNeighbor *neighbors{};
    std::uint8_t *bond_active{};
    std::uint8_t *contact_active{};
    Vec3 *contact_normals{};
    unsigned count{};
    unsigned triangle_count{};
    unsigned neighbor_count{};

    Fixture(unsigned vertex_capacity, unsigned neighbor_slots,
            unsigned triangle_slots)
        : capacity(vertex_capacity), neighbor_capacity(neighbor_slots),
          triangle_capacity(triangle_slots),
          positions(managed<Vec3>(capacity)),
          velocities(managed<Vec3>(capacity)),
          previous(managed<Vec3>(capacity)),
          position_deltas(managed<Vec3>(capacity)),
          gradients(managed<Vec3>(capacity)),
          surface_positions(managed<Vec3>(3U * triangle_capacity)),
          correction_used(managed<float>(capacity)),
          inverse_masses(managed<float>(capacity)),
          characteristic_lengths(managed<float>(capacity)),
          contact_offsets(managed<float>(capacity)),
          lambda(managed<float>(1U)), offsets(managed<unsigned>(capacity + 1U)),
          indices(managed<unsigned>(3U * triangle_capacity)),
          corner_offsets(managed<unsigned>(capacity + 1U)),
          corner_indices(managed<unsigned>(3U * triangle_capacity)),
          time_advances(managed<unsigned>(1U)), status(managed<unsigned>(1U)),
          neighbors(managed<HostDeformableNeighbor>(neighbor_capacity)),
          bond_active(managed<std::uint8_t>(neighbor_capacity)),
          contact_active(managed<std::uint8_t>(capacity)),
          contact_normals(managed<Vec3>(capacity)) {
        count = capacity;
        for (unsigned vertex = 0U; vertex < capacity; ++vertex) {
            inverse_masses[vertex] = 50.0F;
            characteristic_lengths[vertex] = 0.1F;
            contact_normals[vertex] = {0.0F, 1.0F, 0.0F};
        }
        std::fill(bond_active, bond_active + neighbor_capacity, 1U);
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 21>{positions, velocities,
                 previous, position_deltas, gradients, surface_positions,
                 correction_used, inverse_masses, characteristic_lengths,
                 contact_offsets, lambda, offsets, indices, corner_offsets,
                 corner_indices, time_advances, status, neighbors,
                 bond_active, contact_active, contact_normals})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return positions && velocities && previous && position_deltas &&
            gradients && surface_positions && correction_used &&
            inverse_masses && characteristic_lengths && contact_offsets &&
            lambda && offsets && indices && corner_offsets && corner_indices &&
            time_advances && status && neighbors && bond_active &&
            contact_active && contact_normals;
    }

    void set_adjacency(
        const std::vector<std::vector<HostDeformableNeighbor>> &adjacency) {
        CHECK(adjacency.size() == count);
        unsigned cursor{};
        for (unsigned vertex = 0U; vertex < count; ++vertex) {
            offsets[vertex] = cursor;
            for (const auto &neighbor : adjacency[vertex]) {
                CHECK(cursor < neighbor_capacity);
                if (cursor < neighbor_capacity) neighbors[cursor++] = neighbor;
            }
        }
        offsets[count] = cursor;
        neighbor_count = cursor;
    }

    void set_triangles(const std::vector<unsigned> &source) {
        CHECK(source.size() % 3U == 0U);
        CHECK(source.size() <= 3U * triangle_capacity);
        std::copy(source.begin(), source.end(), indices);
        triangle_count = static_cast<unsigned>(source.size() / 3U);
        std::vector<std::vector<unsigned>> corners(count);
        for (unsigned corner = 0U; corner < source.size(); ++corner) {
            CHECK(source[corner] < count);
            if (source[corner] < count) corners[source[corner]].push_back(corner);
        }
        unsigned cursor{};
        for (unsigned vertex = 0U; vertex < count; ++vertex) {
            corner_offsets[vertex] = cursor;
            for (const unsigned corner : corners[vertex])
                corner_indices[cursor++] = corner;
        }
        corner_offsets[count] = cursor;
    }

    void run_tick(bool preserve_volume = false, float target_volume = 1.0F,
                  float orientation = 1.0F, float volume_compliance = 0.0F,
                  Vec3 gravity = {0.0F, -9.81F, 0.0F},
                  float velocity_damping = 5.0F,
                  float maximum_speed = 2.3F, float thickness = 0.01F,
                  float friction = 0.4F,
                  unsigned profile = cloth_realtime_profile_version,
                  float dt = cloth_realtime_timestep) {
        const unsigned blocks = (capacity + threads - 1U) / threads;
        cloth_realtime_predict<<<blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(velocities),
            reinterpret_cast<Vec3_0 *>(previous), correction_used,
            inverse_masses, characteristic_lengths, count, capacity,
            *reinterpret_cast<Vec3_0 *>(&gravity), velocity_damping,
            maximum_speed, dt, profile, time_advances, status);
        for (unsigned sweep = 0U;
             sweep < cloth_realtime_constraint_sweeps; ++sweep) {
            cloth_realtime_project_links<<<blocks, threads>>>(
                reinterpret_cast<Vec3_0 *>(positions),
                reinterpret_cast<Vec3_0 *>(position_deltas), correction_used,
                inverse_masses, characteristic_lengths, offsets,
                reinterpret_cast<DeformableNeighbor_0 *>(neighbors),
                bond_active, count, capacity, dt, profile, status);
            cloth_realtime_apply_position_delta<<<blocks, threads>>>(
                reinterpret_cast<Vec3_0 *>(positions),
                reinterpret_cast<Vec3_0 *>(position_deltas), correction_used,
                inverse_masses, characteristic_lengths, count, capacity, dt,
                profile, status);
            cloth_realtime_project_volume<<<1, threads>>>(
                reinterpret_cast<Vec3_0 *>(positions), inverse_masses,
                characteristic_lengths, correction_used, indices, count,
                triangle_count, corner_offsets, corner_indices,
                reinterpret_cast<Vec3_0 *>(gradients), target_volume,
                orientation, volume_compliance, preserve_volume, dt, profile,
                lambda, sweep == 0U, status);
            if ((sweep & 1U) != 0U)
                cloth_realtime_project_contacts<<<blocks, threads>>>(
                    reinterpret_cast<Vec3_0 *>(positions),
                    reinterpret_cast<Vec3_0 *>(previous), inverse_masses,
                    characteristic_lengths, correction_used,
                    reinterpret_cast<Vec3_0 *>(contact_normals),
                    contact_offsets, contact_active, count, capacity,
                    thickness, friction, dt, profile, status);
        }
        cloth_realtime_commit<<<blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(previous),
            reinterpret_cast<Vec3_0 *>(velocities), inverse_masses,
            characteristic_lengths,
            reinterpret_cast<Vec3_0 *>(contact_normals), contact_offsets,
            contact_active, count, capacity, thickness, friction,
            maximum_speed, dt, profile, status);
        const unsigned triangle_blocks =
            std::max(1U, (triangle_count + threads - 1U) / threads);
        cloth_update_surface<<<triangle_blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions), indices,
            reinterpret_cast<Vec3_0 *>(surface_positions), triangle_count);
    }
};

std::vector<std::vector<HostDeformableNeighbor>> make_links(
    const Fixture &fixture,
    const std::vector<std::pair<unsigned, unsigned>> &edges,
    float compliance = 0.0F) {
    std::vector<std::vector<HostDeformableNeighbor>> result(fixture.count);
    unsigned bond{};
    for (const auto [first, second] : edges) {
        const float rest = length(subtract_host(
            fixture.positions[first], fixture.positions[second]));
        result[first].push_back({second, rest, compliance, bond});
        result[second].push_back({first, rest, compliance, bond});
        ++bond;
    }
    return result;
}

void test_profile_stiff_topology_and_impact() {
    Fixture stiff(2U, 4U, 1U);
    CHECK(stiff.allocated());
    stiff.positions[0] = {0.0F, 0.0F, 0.0F};
    stiff.positions[1] = {0.12F, 0.0F, 0.0F};
    stiff.inverse_masses[0] = 0.0F;
    stiff.characteristic_lengths[0] = 0.1F;
    stiff.characteristic_lengths[1] = 0.1F;
    auto adjacency = make_links(stiff, {{0U, 1U}});
    // The authored rest state is 0.1 m, while current state starts at 20%.
    adjacency[0][0].rest_length = 0.1F;
    adjacency[1][0].rest_length = 0.1F;
    stiff.set_adjacency(adjacency);
    stiff.set_triangles({});
    stiff.run_tick(false, 1.0F, 1.0F, 0.0F, Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const float restored = length(subtract_host(
        stiff.positions[0], stiff.positions[1]));
    CHECK(*stiff.status == 0U);
    CHECK(*stiff.time_advances == 1U);
    CHECK(std::abs(restored - 0.1F) / 0.1F <= 0.10F);
    CHECK(length(stiff.positions[0]) <= 1.0e-7F);
    CHECK(stiff.correction_used[1] <= 0.025001F);

    stiff.positions[1] = {0.12F, 0.0F, 0.0F};
    stiff.velocities[1] = {};
    stiff.bond_active[0] = 0U;
    stiff.run_tick(false, 1.0F, 1.0F, 0.0F, Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(std::abs(stiff.positions[1].x - 0.12F) < 1.0e-6F);
    stiff.bond_active[0] = 1U;
    stiff.velocities[1] = {};
    stiff.run_tick(false, 1.0F, 1.0F, 0.0F, Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(std::abs(length(subtract_host(
        stiff.positions[0], stiff.positions[1])) - 0.1F) <= 0.01F);

    Fixture impact(1U, 1U, 1U);
    CHECK(impact.allocated());
    impact.positions[0] = {0.0F, 0.020F, 0.0F};
    impact.velocities[0] = {0.0F, -0.8F, 0.0F};
    impact.characteristic_lengths[0] = 0.1F;
    impact.contact_active[0] = 1U;
    impact.set_adjacency({{}});
    impact.set_triangles({});
    impact.run_tick(false, 1.0F, 1.0F, 0.0F, Vec3{}, 0.0F, 2.3F,
        0.01F, 0.4F);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*impact.status == 0U);
    CHECK(*impact.time_advances == 1U);
    CHECK(impact.positions[0].y >= 0.009999F);
    CHECK(impact.velocities[0].y >= -1.0e-5F);
    CHECK(impact.correction_used[0] <= 0.025001F);

    const Vec3 rejected_position = impact.positions[0];
    impact.run_tick(false, 1.0F, 1.0F, 0.0F, Vec3{}, 0.0F, 2.3F,
        0.01F, 0.4F, 99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*impact.status & cloth_realtime_status_invalid_profile) != 0U);
    CHECK(*impact.time_advances == 1U);
    CHECK(length(subtract_host(impact.positions[0], rejected_position)) == 0.0F);
    std::printf(
        "F8_CLOTH_60HZ_STIFF stretch_error=%.8f pinned_drift=%.8f "
        "impact_height=%.8f impact_normal_speed=%.8f topology_reset=pass\n",
        std::abs(restored - 0.1F) / 0.1F, length(stiff.positions[0]),
        impact.positions[0].y, impact.velocities[0].y);
}

float signed_volume(const Fixture &fixture) {
    float result{};
    for (unsigned triangle = 0U; triangle < fixture.triangle_count; ++triangle) {
        const Vec3 &a = fixture.positions[fixture.indices[3U * triangle]];
        const Vec3 &b = fixture.positions[fixture.indices[3U * triangle + 1U]];
        const Vec3 &c = fixture.positions[fixture.indices[3U * triangle + 2U]];
        result += dot_host(a, cross_host(b, c)) / 6.0F;
    }
    return result;
}

void test_closed_volume() {
    Fixture volume(4U, 1U, 4U);
    CHECK(volume.allocated());
    volume.positions[0] = {0.0F, 0.0F, 0.0F};
    volume.positions[1] = {0.4F, 0.0F, 0.0F};
    volume.positions[2] = {0.0F, 0.4F, 0.0F};
    volume.positions[3] = {0.0F, 0.0F, 0.4F};
    volume.inverse_masses[0] = 0.0F;
    for (unsigned vertex = 0U; vertex < volume.count; ++vertex)
        volume.characteristic_lengths[vertex] = 0.4F;
    volume.set_adjacency({{}, {}, {}, {}});
    volume.set_triangles({0U, 2U, 1U, 0U, 1U, 3U,
        0U, 3U, 2U, 1U, 2U, 3U});
    const float target = std::abs(signed_volume(volume));
    const float orientation = signed_volume(volume) < 0.0F ? -1.0F : 1.0F;
    volume.positions[3].z = 0.2F;
    const float before = std::abs(std::abs(signed_volume(volume)) - target) /
        target;
    volume.run_tick(true, target, orientation, 0.0F, Vec3{}, 0.0F,
        2.3F, 0.0F, 0.0F);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const float after = std::abs(std::abs(signed_volume(volume)) - target) /
        target;
    CHECK(*volume.status == 0U);
    CHECK(before > 0.25F);
    CHECK(after <= 0.15F);
    for (unsigned vertex = 0U; vertex < volume.count; ++vertex)
        CHECK(volume.correction_used[vertex] <= 0.100001F);
    std::printf(
        "F8_CLOTH_60HZ_VOLUME before=%.8f after=%.8f target=%.8f\n",
        before, after, target);
}

struct GridMetrics {
    float stretch_rms{};
    float stretch_max{};
    float pinned_drift{};
    float bend_rms_degrees{};
    float final_energy{};
    unsigned tunneling{};
};

GridMetrics grid_metrics(const Fixture &fixture,
                         const std::vector<std::pair<unsigned, unsigned>> &edges,
                         const std::vector<Vec3> &rest_positions,
                         unsigned side) {
    GridMetrics result{};
    double squared_error{};
    for (const auto [first, second] : edges) {
        const float rest = length(subtract_host(
            rest_positions[first], rest_positions[second]));
        const float current = length(subtract_host(
            fixture.positions[first], fixture.positions[second]));
        const float error = std::abs(current - rest) / rest;
        squared_error += static_cast<double>(error) * error;
        result.stretch_max = std::max(result.stretch_max, error);
    }
    result.stretch_rms = static_cast<float>(std::sqrt(
        squared_error / std::max<std::size_t>(edges.size(), 1U)));
    for (unsigned column = 0U; column < side; ++column)
        result.pinned_drift = std::max(result.pinned_drift,
            length(subtract_host(fixture.positions[column],
                rest_positions[column])));
    // The fixture starts planar and has no out-of-plane load. Any bend is a
    // numerical artifact, so estimate the adjacent-normal RMS explicitly.
    std::vector<Vec3> normals(fixture.triangle_count);
    for (unsigned triangle = 0U; triangle < fixture.triangle_count; ++triangle) {
        const Vec3 a = fixture.positions[fixture.indices[3U * triangle]];
        const Vec3 b = fixture.positions[fixture.indices[3U * triangle + 1U]];
        const Vec3 c = fixture.positions[fixture.indices[3U * triangle + 2U]];
        Vec3 normal = cross_host(subtract_host(b, a), subtract_host(c, a));
        const float magnitude = length(normal);
        if (magnitude > 0.0F) {
            normal.x /= magnitude;
            normal.y /= magnitude;
            normal.z /= magnitude;
        }
        normals[triangle] = normal;
    }
    double bend_squared{};
    unsigned bend_count{};
    for (unsigned cell = 0U; cell < fixture.triangle_count / 2U; ++cell) {
        const float cosine = std::clamp(
            dot_host(normals[2U * cell], normals[2U * cell + 1U]),
            -1.0F, 1.0F);
        const float angle = std::acos(cosine);
        bend_squared += static_cast<double>(angle) * angle;
        ++bend_count;
    }
    result.bend_rms_degrees = static_cast<float>(
        std::sqrt(bend_squared / std::max(bend_count, 1U)) *
        180.0 / 3.14159265358979323846);
    for (unsigned vertex = 0U; vertex < fixture.count; ++vertex) {
        if (fixture.positions[vertex].y < 0.0099F) ++result.tunneling;
        if (fixture.inverse_masses[vertex] > 0.0F) {
            const float mass = 1.0F / fixture.inverse_masses[vertex];
            const float speed = length(fixture.velocities[vertex]);
            result.final_energy += 0.5F * mass * speed * speed;
        }
    }
    return result;
}

void test_hanging_resting_long_run() {
    constexpr unsigned side = 8U;
    constexpr unsigned count = side * side;
    constexpr float spacing = 0.10F;
    Fixture cloth(count, 12U * count, 2U * (side - 1U) * (side - 1U));
    CHECK(cloth.allocated());
    cloth.count = count;
    std::vector<Vec3> rest_positions(count);
    for (unsigned row = 0U; row < side; ++row)
        for (unsigned column = 0U; column < side; ++column) {
            const unsigned vertex = row * side + column;
            const Vec3 position{(float(column) - 3.5F) * spacing,
                0.71F - float(row) * spacing, 0.0F};
            cloth.positions[vertex] = position;
            rest_positions[vertex] = position;
            cloth.inverse_masses[vertex] = row == 0U ? 0.0F : 50.0F;
            cloth.characteristic_lengths[vertex] = spacing;
            cloth.contact_active[vertex] = 1U;
        }

    std::vector<unsigned> triangles;
    std::set<std::pair<unsigned, unsigned>> unique_edges;
    const auto edge = [&](unsigned first, unsigned second) {
        if (first > second) std::swap(first, second);
        unique_edges.emplace(first, second);
    };
    for (unsigned row = 0U; row + 1U < side; ++row)
        for (unsigned column = 0U; column + 1U < side; ++column) {
            const unsigned a = row * side + column;
            const unsigned b = a + 1U;
            const unsigned c = a + side;
            const unsigned d = c + 1U;
            triangles.insert(triangles.end(), {a, c, b, b, c, d});
            edge(a, b); edge(a, c); edge(b, c);
            edge(b, d); edge(c, d);
            // Opposite-vertex distance is the legacy bending constraint.
            edge(a, d);
        }
    std::vector<std::pair<unsigned, unsigned>> edges(
        unique_edges.begin(), unique_edges.end());
    cloth.set_adjacency(make_links(cloth, edges));
    cloth.set_triangles(triangles);

    for (unsigned tick = 0U; tick < 120U; ++tick) cloth.run_tick();
    constexpr unsigned measured_ticks = 600U;
    std::vector<cudaEvent_t> starts(measured_ticks), ends(measured_ticks);
    for (unsigned tick = 0U; tick < measured_ticks; ++tick) {
        CHECK(cudaEventCreate(&starts[tick]) == cudaSuccess);
        CHECK(cudaEventCreate(&ends[tick]) == cudaSuccess);
        CHECK(cudaEventRecord(starts[tick]) == cudaSuccess);
        cloth.run_tick();
        CHECK(cudaEventRecord(ends[tick]) == cudaSuccess);
    }
    CHECK(cudaEventSynchronize(ends.back()) == cudaSuccess);
    std::vector<float> milliseconds(measured_ticks);
    for (unsigned tick = 0U; tick < measured_ticks; ++tick) {
        CHECK(cudaEventElapsedTime(
                  &milliseconds[tick], starts[tick], ends[tick]) == cudaSuccess);
        cudaEventDestroy(starts[tick]);
        cudaEventDestroy(ends[tick]);
    }
    for (unsigned tick = 720U; tick < 3'600U; ++tick) cloth.run_tick();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaSuccess);

    std::sort(milliseconds.begin(), milliseconds.end());
    const auto percentile = [&](float quantile) {
        const std::size_t index = static_cast<std::size_t>(
            quantile * float(milliseconds.size() - 1U));
        return milliseconds[index];
    };
    const GridMetrics metrics = grid_metrics(
        cloth, edges, rest_positions, side);
    float maximum_correction{};
    for (unsigned vertex = 0U; vertex < cloth.count; ++vertex)
        maximum_correction = std::max(
            maximum_correction, cloth.correction_used[vertex]);
    CHECK(*cloth.status == 0U);
    CHECK(*cloth.time_advances == 3'600U);
    CHECK(metrics.stretch_rms <= 0.03F);
    CHECK(metrics.stretch_max <= 0.10F);
    CHECK(metrics.bend_rms_degrees <= 10.0F);
    CHECK(metrics.pinned_drift <= 0.002F);
    CHECK(metrics.tunneling == 0U);
    CHECK(std::isfinite(metrics.final_energy));
    CHECK(metrics.final_energy < 1.0F);
    CHECK(maximum_correction <= 0.25F * spacing + 1.0e-6F);
    std::printf(
        "F8_CLOTH_60HZ_QUALITY ticks=%u stretch_rms=%.8f "
        "stretch_max=%.8f bend_rms_deg=%.8f pinned_drift_m=%.8f "
        "tunneling=%u final_energy_j=%.8f max_correction_m=%.8f\n",
        *cloth.time_advances, metrics.stretch_rms, metrics.stretch_max,
        metrics.bend_rms_degrees, metrics.pinned_drift,
        metrics.tunneling, metrics.final_energy, maximum_correction);
    std::printf(
        "F8_CLOTH_60HZ_PERF fixture_vertices=%u samples=%u p50_ms=%.6f "
        "p95_ms=%.6f p99_ms=%.6f max_ms=%.6f\n",
        cloth.count, measured_ticks, percentile(0.50F), percentile(0.95F),
        percentile(0.99F), milliseconds.back());
}

} // namespace

int main() {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    std::printf("F8_CLOTH_60HZ_DEVICE name=%s compute=%d.%d\n",
        properties.name, properties.major, properties.minor);
    test_profile_stiff_topology_and_impact();
    test_closed_volume();
    test_hanging_resting_long_run();
    if (failures == 0)
        std::puts("F8-13 cloth realtime physical CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
