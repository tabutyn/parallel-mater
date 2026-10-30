// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "simulation/rope_realtime.hpp"
#include "simulation/types.hpp"
#include "parallel_mater_rope_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(Vec3) == sizeof(Vec3_0));

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

struct Fixture {
    unsigned capacity{};
    unsigned count{};
    float node_spacing{0.1F};
    float radius{0.05F};
    Vec3 *positions{};
    Vec3 *velocities{};
    Vec3 *previous{};
    Vec3 *position_deltas{};
    Vec3 *constraint_forces{};
    Vec3 *anchor_targets{};
    Vec3 *contact_normals{};
    float *correction_used{};
    float *inverse_masses{};
    float *rest_lengths{};
    float *rest_bend_lengths{};
    float *contact_offsets{};
    std::uint8_t *anchor_active{};
    std::uint8_t *contact_active{};
    unsigned *time_advances{};
    unsigned *status{};

    explicit Fixture(unsigned node_capacity)
        : capacity(node_capacity), count(node_capacity),
          positions(managed<Vec3>(capacity)),
          velocities(managed<Vec3>(capacity)),
          previous(managed<Vec3>(capacity)),
          position_deltas(managed<Vec3>(capacity)),
          constraint_forces(managed<Vec3>(capacity)),
          anchor_targets(managed<Vec3>(2U)),
          contact_normals(managed<Vec3>(capacity)),
          correction_used(managed<float>(capacity)),
          inverse_masses(managed<float>(capacity)),
          rest_lengths(managed<float>(capacity > 0U ? capacity - 1U : 0U)),
          rest_bend_lengths(managed<float>(capacity > 1U ? capacity - 2U : 0U)),
          contact_offsets(managed<float>(capacity)),
          anchor_active(managed<std::uint8_t>(2U)),
          contact_active(managed<std::uint8_t>(capacity)),
          time_advances(managed<unsigned>(1U)),
          status(managed<unsigned>(1U)) {
        for (unsigned node = 0U; node < capacity; ++node) {
            inverse_masses[node] = 10.0F;
            contact_normals[node] = {0.0F, 1.0F, 0.0F};
        }
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 16>{positions, velocities,
                 previous, position_deltas, constraint_forces,
                 anchor_targets, contact_normals,
                 correction_used, inverse_masses, rest_lengths,
                 rest_bend_lengths, contact_offsets, anchor_active,
                 contact_active, time_advances, status})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return positions && velocities && previous && position_deltas &&
            constraint_forces &&
            anchor_targets && contact_normals && correction_used &&
            inverse_masses && rest_lengths && rest_bend_lengths &&
            contact_offsets && anchor_active && contact_active &&
            time_advances && status;
    }

    void capture_rest() {
        for (unsigned edge = 0U; edge + 1U < count; ++edge)
            rest_lengths[edge] = length(subtract_host(
                positions[edge + 1U], positions[edge]));
        for (unsigned bend = 0U; bend + 2U < count; ++bend)
            rest_bend_lengths[bend] = length(subtract_host(
                positions[bend + 2U], positions[bend]));
    }

    void run_tick(Vec3 gravity = {0.0F, -9.81F, 0.0F},
                  float velocity_damping = 5.0F,
                  float friction = 0.4F, float maximum_speed = 2.3F,
                  float structural_compliance = 0.0F,
                  unsigned profile = rope_realtime_profile_version,
                  float dt = rope_realtime_timestep) {
        const unsigned blocks = std::max(
            1U, (capacity + threads - 1U) / threads);
        rope_realtime_predict<<<blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(velocities),
            reinterpret_cast<Vec3_0 *>(previous), correction_used,
            inverse_masses, reinterpret_cast<Vec3_0 *>(anchor_targets),
            anchor_active, count, capacity, node_spacing, radius,
            *reinterpret_cast<Vec3_0 *>(&gravity), velocity_damping,
            friction, maximum_speed, dt, profile, time_advances, status);
        for (unsigned sweep = 0U;
             sweep < rope_realtime_constraint_sweeps; ++sweep) {
            rope_realtime_project_constraints<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(positions),
                reinterpret_cast<Vec3_0 *>(position_deltas),
                reinterpret_cast<Vec3_0 *>(constraint_forces), correction_used,
                inverse_masses, rest_lengths, rest_bend_lengths,
                anchor_active, count, capacity, node_spacing,
                structural_compliance, dt, profile, status);
            rope_realtime_apply_position_delta<<<blocks, threads>>>(
                reinterpret_cast<Vec3_0 *>(positions),
                reinterpret_cast<Vec3_0 *>(position_deltas), correction_used,
                inverse_masses, anchor_active, count, capacity,
                node_spacing, dt, profile, status);
            if ((sweep & 1U) != 0U)
                rope_realtime_project_contacts<<<blocks, threads>>>(
                    reinterpret_cast<Vec3_0 *>(positions),
                    reinterpret_cast<Vec3_0 *>(previous), inverse_masses,
                    correction_used, anchor_active,
                    reinterpret_cast<Vec3_0 *>(contact_normals),
                    contact_offsets, contact_active, count, capacity,
                    node_spacing, radius, 0U, friction, dt, profile, status);
        }
        rope_realtime_commit<<<blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(previous),
            reinterpret_cast<Vec3_0 *>(velocities), inverse_masses,
            anchor_active, reinterpret_cast<Vec3_0 *>(contact_normals),
            contact_offsets, contact_active, count, capacity,
            node_spacing, radius, friction, maximum_speed, dt, profile,
            status);
    }
};

void test_profile_stiff_attachment_impact_and_self_contact() {
    Fixture stiff(8U);
    CHECK(stiff.allocated());
    for (unsigned node = 0U; node < stiff.count; ++node)
        stiff.positions[node] = {0.1F * float(node), 0.0F, 0.0F};
    stiff.capture_rest();
    stiff.positions[1].x = 0.12F;
    stiff.anchor_active[0] = 1U;
    stiff.anchor_targets[0] = stiff.positions[0];
    stiff.run_tick(Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const float restored = length(subtract_host(
        stiff.positions[0], stiff.positions[1]));
    CHECK(*stiff.status == 0U);
    CHECK(*stiff.time_advances == 1U);
    CHECK(std::abs(restored - 0.1F) / 0.1F <= 0.08F);
    CHECK(length(subtract_host(stiff.positions[0],
        stiff.anchor_targets[0])) <= 1.0e-7F);
    for (unsigned node = 0U; node < stiff.count; ++node)
        CHECK(stiff.correction_used[node] <= 0.025001F);

    stiff.anchor_targets[0].y = 0.01F;
    stiff.run_tick(Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(length(subtract_host(stiff.positions[0],
        stiff.anchor_targets[0])) <= 1.0e-7F);
    CHECK(std::abs(stiff.velocities[0].y - 0.6F) <= 1.0e-5F);

    // A topology/reset boundary supplies new rest buffers and a new anchor.
    for (unsigned node = 0U; node < stiff.count; ++node) {
        stiff.positions[node] = {0.1F * float(node), 0.0F, 0.0F};
        stiff.velocities[node] = {};
    }
    stiff.anchor_targets[0] = stiff.positions[0];
    stiff.capture_rest();
    stiff.run_tick(Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*stiff.status == 0U);
    CHECK(length(subtract_host(stiff.positions[0],
        stiff.anchor_targets[0])) <= 1.0e-7F);

    Fixture impact(4U);
    CHECK(impact.allocated());
    for (unsigned node = 0U; node < impact.count; ++node) {
        impact.positions[node] = {0.1F * float(node), 0.06F, 0.0F};
        impact.velocities[node] = {0.0F, -1.0F, 0.0F};
        impact.contact_active[node] = 1U;
    }
    impact.capture_rest();
    impact.run_tick(Vec3{}, 0.0F, 0.4F, 2.3F);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*impact.status == 0U);
    CHECK(*impact.time_advances == 1U);
    for (unsigned node = 0U; node < impact.count; ++node) {
        CHECK(impact.positions[node].y >= 0.049999F);
        CHECK(impact.velocities[node].y >= -1.0e-5F);
    }

    Fixture self_contact(5U);
    CHECK(self_contact.allocated());
    for (unsigned node = 0U; node < self_contact.count; ++node)
        self_contact.positions[node] =
            {0.1F * float(node), node == 3U ? 0.03F : 0.05F, 0.0F};
    self_contact.capture_rest();
    // F8-10 supplies this canonical same-owner, non-adjacent constraint after
    // filtering neighboring segments. The domain solve consumes it four times.
    self_contact.contact_active[3] = 1U;
    self_contact.run_tick(Vec3{});
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*self_contact.status == 0U);
    CHECK(self_contact.positions[3].y >= 0.049999F);

    const Vec3 rejected_position = impact.positions[0];
    impact.run_tick(Vec3{}, 0.0F, 0.4F, 2.3F, 0.0F, 99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*impact.status & rope_realtime_status_invalid_profile) != 0U);
    CHECK(*impact.time_advances == 1U);
    CHECK(length(subtract_host(
        impact.positions[0], rejected_position)) == 0.0F);
    std::printf(
        "F8_ROPE_60HZ_STIFF stretch_error=%.8f attachment_drift=%.8f "
        "moving_anchor_speed=%.8f impact_height=%.8f "
        "impact_normal_speed=%.8f self_contact_height=%.8f reset=pass\n",
        std::abs(restored - 0.1F) / 0.1F,
        length(subtract_host(stiff.positions[0], stiff.anchor_targets[0])),
        0.6F, impact.positions[0].y, impact.velocities[0].y,
        self_contact.positions[3].y);
}

struct ChainMetrics {
    float stretch_rms{};
    float stretch_max{};
    float bend_rms_degrees{};
    float attachment_drift{};
    float final_energy{};
    unsigned tunneling{};
};

ChainMetrics chain_metrics(const Fixture &fixture,
                           const std::vector<Vec3> &rest_positions) {
    ChainMetrics result{};
    double stretch_squared{};
    for (unsigned edge = 0U; edge + 1U < fixture.count; ++edge) {
        const float rest = length(subtract_host(
            rest_positions[edge + 1U], rest_positions[edge]));
        const float current = length(subtract_host(
            fixture.positions[edge + 1U], fixture.positions[edge]));
        const float error = std::abs(current - rest) / rest;
        stretch_squared += static_cast<double>(error) * error;
        result.stretch_max = std::max(result.stretch_max, error);
    }
    result.stretch_rms = static_cast<float>(std::sqrt(stretch_squared /
        std::max(1U, fixture.count - 1U)));
    double bend_squared{};
    for (unsigned node = 1U; node + 1U < fixture.count; ++node) {
        const Vec3 before = subtract_host(
            fixture.positions[node], fixture.positions[node - 1U]);
        const Vec3 after = subtract_host(
            fixture.positions[node + 1U], fixture.positions[node]);
        const Vec3 rest_before = subtract_host(
            rest_positions[node], rest_positions[node - 1U]);
        const Vec3 rest_after = subtract_host(
            rest_positions[node + 1U], rest_positions[node]);
        const float current_cosine = std::clamp(dot_host(before, after) /
            std::max(length(before) * length(after), 1.0e-12F), -1.0F, 1.0F);
        const float rest_cosine = std::clamp(dot_host(rest_before, rest_after) /
            std::max(length(rest_before) * length(rest_after), 1.0e-12F),
            -1.0F, 1.0F);
        const float error = std::acos(current_cosine) - std::acos(rest_cosine);
        bend_squared += static_cast<double>(error) * error;
    }
    result.bend_rms_degrees = static_cast<float>(std::sqrt(bend_squared /
        std::max(1U, fixture.count - 2U)) *
        180.0 / 3.14159265358979323846);
    result.attachment_drift = length(subtract_host(
        fixture.positions[0], fixture.anchor_targets[0]));
    for (unsigned node = 0U; node < fixture.count; ++node) {
        if (fixture.positions[node].y < fixture.radius - 1.0e-4F)
            ++result.tunneling;
        if (node != 0U) {
            const float mass = 1.0F / fixture.inverse_masses[node];
            const float speed = length(fixture.velocities[node]);
            result.final_energy += 0.5F * mass * speed * speed;
        }
    }
    return result;
}

void test_long_chain_quality_and_timing() {
    constexpr unsigned count = 72U;
    constexpr float spacing = 0.10F;
    Fixture rope(count);
    CHECK(rope.allocated());
    rope.node_spacing = spacing;
    rope.radius = 0.5F * spacing;
    std::vector<Vec3> rest_positions(count);
    const float top = 0.05F + spacing * float(count - 1U);
    for (unsigned node = 0U; node < count; ++node) {
        const Vec3 position{0.0F, top - spacing * float(node), 0.0F};
        rope.positions[node] = position;
        rest_positions[node] = position;
        rope.inverse_masses[node] = 10.0F;
        rope.contact_active[node] = 1U;
    }
    rope.anchor_active[0] = 1U;
    rope.anchor_targets[0] = rope.positions[0];
    rope.anchor_active[1] = 1U;
    rope.anchor_targets[1] = rope.positions[count - 1U];
    rope.capture_rest();

    // Keep a representable margin below the exact 0.5 h / dt profile edge.
    constexpr float maximum_speed = 2.9F;
    for (unsigned tick = 0U; tick < 120U; ++tick)
        rope.run_tick({0.0F, -9.81F, 0.0F}, 5.0F, 0.4F,
            maximum_speed);
    constexpr unsigned measured_ticks = 600U;
    std::vector<cudaEvent_t> starts(measured_ticks), ends(measured_ticks);
    for (unsigned tick = 0U; tick < measured_ticks; ++tick) {
        CHECK(cudaEventCreate(&starts[tick]) == cudaSuccess);
        CHECK(cudaEventCreate(&ends[tick]) == cudaSuccess);
        CHECK(cudaEventRecord(starts[tick]) == cudaSuccess);
        rope.run_tick({0.0F, -9.81F, 0.0F}, 5.0F, 0.4F,
            maximum_speed);
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
    for (unsigned tick = 720U; tick < 3'600U; ++tick)
        rope.run_tick({0.0F, -9.81F, 0.0F}, 5.0F, 0.4F,
            maximum_speed);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaSuccess);

    std::sort(milliseconds.begin(), milliseconds.end());
    const auto percentile = [&](float quantile) {
        const std::size_t index = static_cast<std::size_t>(
            quantile * float(milliseconds.size() - 1U));
        return milliseconds[index];
    };
    const ChainMetrics metrics = chain_metrics(rope, rest_positions);
    float maximum_correction{};
    for (unsigned node = 0U; node < rope.count; ++node)
        maximum_correction = std::max(
            maximum_correction, rope.correction_used[node]);
    CHECK(*rope.status == 0U);
    CHECK(*rope.time_advances == 3'600U);
    CHECK(metrics.stretch_rms <= 0.03F);
    CHECK(metrics.stretch_max <= 0.08F);
    CHECK(metrics.bend_rms_degrees <= 15.0F);
    CHECK(metrics.attachment_drift <= 0.01F);
    CHECK(metrics.tunneling == 0U);
    CHECK(std::isfinite(metrics.final_energy));
    CHECK(metrics.final_energy < 1.0F);
    CHECK(maximum_correction <= 0.25F * spacing + 1.0e-6F);
    std::printf(
        "F8_ROPE_60HZ_QUALITY ticks=%u stretch_rms=%.8f "
        "stretch_max=%.8f bend_rms_deg=%.8f attachment_drift_m=%.8f "
        "tunneling=%u final_energy_j=%.8f max_correction_m=%.8f\n",
        *rope.time_advances, metrics.stretch_rms, metrics.stretch_max,
        metrics.bend_rms_degrees, metrics.attachment_drift,
        metrics.tunneling, metrics.final_energy, maximum_correction);
    std::printf(
        "F8_ROPE_60HZ_PERF fixture_nodes=%u samples=%u p50_ms=%.6f "
        "p95_ms=%.6f p99_ms=%.6f max_ms=%.6f\n",
        rope.count, measured_ticks, percentile(0.50F), percentile(0.95F),
        percentile(0.99F), milliseconds.back());
}

} // namespace

int main() {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    std::printf("F8_ROPE_60HZ_DEVICE name=%s compute=%d.%d\n",
        properties.name, properties.major, properties.minor);
    test_profile_stiff_attachment_impact_and_self_contact();
    test_long_chain_quality_and_timing();
    if (failures == 0)
        std::puts("F8-15 rope realtime physical CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
