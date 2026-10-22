// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>

#include "simulation/geometry_storage.hpp"
#include "simulation/rigid_realtime.hpp"
#include "parallel_mater_geometry_constraints_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(BodyParameters) == sizeof(BodyParameters_0));
static_assert(sizeof(RigidBodyState) == sizeof(RigidBodyState_0));
static_assert(sizeof(RigidBodyId) == sizeof(RigidBodyId_0));
static_assert(sizeof(ContactManifold) == sizeof(ContactManifold_0));
static_assert(sizeof(RigidConstraintResource) ==
              sizeof(RigidConstraintResource_0));
static_assert(sizeof(GeometryConstraintBody) == sizeof(AvbdBody_0));
static_assert(sizeof(ContactSchedule) == sizeof(ContactSchedule_0));

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    if (cudaMallocManaged(&result,
            std::max<std::size_t>(count, 1U) * sizeof(T)) != cudaSuccess)
        return nullptr;
    std::fill(result, result + count, T{});
    return result;
}

struct Fixture {
    static constexpr std::uint32_t body_count = 8U;
    static constexpr std::uint32_t contact_count = 3U;
    static constexpr std::uint32_t joint_count = 1U;

    BodyParameters *parameters{managed<BodyParameters>(body_count)};
    RigidBodyState *previous{managed<RigidBodyState>(body_count)};
    RigidBodyState *states{managed<RigidBodyState>(body_count)};
    RigidBodyId *ids{managed<RigidBodyId>(body_count)};
    ContactManifold *manifolds{managed<ContactManifold>(contact_count)};
    std::uint32_t *pairs{managed<std::uint32_t>(contact_count)};
    std::uint32_t *pair_count{managed<std::uint32_t>(1U)};
    RigidConstraintResource *joints{
        managed<RigidConstraintResource>(joint_count)};
    GeometryConstraintBody *bodies{
        managed<GeometryConstraintBody>(body_count)};
    ContactSchedule *schedule{managed<ContactSchedule>(1U)};

    Fixture() {
        for (std::uint32_t index = 0U; index < body_count; ++index) {
            parameters[index].motion = index == 0U
                ? MotionType::static_body : MotionType::dynamic;
            parameters[index].inverse_mass = index == 0U ? 0.0F : 1.0F;
            parameters[index].inverse_inertia_local = index == 0U
                ? Vec3{} : Vec3{1.0F, 1.0F, 1.0F};
            parameters[index].friction = 0.5F;
            parameters[index].restitution = 0.5F;
            parameters[index].collision_margin = 0.001F;
            previous[index].orientation.w = 1.0F;
            states[index] = previous[index];
            ids[index] = {index, 1U};
        }

        // Resting penetration against a static floor.
        previous[1].position.y = 0.04F;
        states[1] = previous[1];
        states[1].position.y = 0.0395F;
        states[1].linear_velocity.y = -0.03F;
        pairs[0] = 1U * body_count;
        manifolds[0].count = 1U;
        manifolds[0].contacts[0].normal = {0.0F, 1.0F, 0.0F};
        manifolds[0].contacts[0].point = {0.0F, 0.0F, 0.0F};
        manifolds[0].contacts[0].penetration = 0.01F;

        // Cold impact. Realtime v1 folds restitution into the pose solve.
        previous[2].position.y = 0.1F;
        previous[2].linear_velocity.y = -3.0F;
        states[2] = previous[2];
        states[2].position.y = 0.05F;
        pairs[1] = 2U * body_count;
        manifolds[1].count = 1U;
        manifolds[1].contacts[0].normal = {0.0F, 1.0F, 0.0F};
        manifolds[1].contacts[0].point = {0.0F, 0.0F, 0.0F};
        manifolds[1].contacts[0].penetration = 0.002F;

        // Exact 100:1 dynamic mass-ratio boundary.
        parameters[4].inverse_mass = 0.01F;
        parameters[4].inverse_inertia_local = {0.01F, 0.01F, 0.01F};
        previous[3].position = {-0.002F, 1.0F, 0.0F};
        previous[4].position = {0.002F, 1.0F, 0.0F};
        states[3] = previous[3];
        states[4] = previous[4];
        pairs[2] = 3U * body_count + 4U;
        manifolds[2].count = 1U;
        manifolds[2].contacts[0].normal = {-1.0F, 0.0F, 0.0F};
        manifolds[2].contacts[0].point = {0.0F, 1.0F, 0.0F};
        manifolds[2].contacts[0].penetration = 0.01F;

        // Fixed joint error begins outside the 1 cm quality gate.
        states[5] = previous[5];
        states[5].position.x = 0.02F;
        constexpr float joint_half_angle = 0.05F;
        states[5].orientation = {0.0F, 0.0F, std::sin(joint_half_angle),
                                 std::cos(joint_half_angle)};
        joints[0].alive = 1U;
        joints[0].options.enabled = 1U;
        joints[0].options.type = RigidConstraintType::fixed;
        joints[0].options.body_a = ids[5];
        joints[0].options.body_b = ids[0];
        joints[0].options.local_orientation_a.w = 1.0F;
        joints[0].options.local_orientation_b.w = 1.0F;
        joints[0].options.solver_iterations = 64U; // profile must ignore it

        // Fast rotation at the profile boundary remains finite and bounded.
        constexpr float half_angle = 0.25F;
        previous[6].angular_velocity.z = 30.0F;
        states[6] = previous[6];
        states[6].orientation = {0.0F, 0.0F, std::sin(half_angle),
                                 std::cos(half_angle)};

        // A staged teleport/reset target is the tick start, not swept motion.
        previous[7].position.x = -5.0F;
        previous[7].linear_velocity = {3.0F, 2.0F, 1.0F};
        previous[7].angular_velocity = {1.0F, 2.0F, 3.0F};
        states[7] = previous[7];
        states[7].position.x = 5.0F;
        states[7].linear_velocity = {};
        states[7].angular_velocity = {};
        *pair_count = contact_count;
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 10>{parameters, previous,
                 states, ids, manifolds, pairs, pair_count, joints, bodies,
                 schedule})
            cudaFree(pointer);
    }

    bool allocated() const {
        return parameters && previous && states && ids && manifolds && pairs &&
            pair_count && joints && bodies && schedule;
    }
};

struct StackFixture {
    static constexpr std::uint32_t body_count = 3U;
    static constexpr std::uint32_t contact_count = 2U;

    BodyParameters *parameters{managed<BodyParameters>(body_count)};
    RigidBodyState *previous{managed<RigidBodyState>(body_count)};
    RigidBodyState *states{managed<RigidBodyState>(body_count)};
    RigidBodyId *ids{managed<RigidBodyId>(body_count)};
    ContactManifold *manifolds{managed<ContactManifold>(contact_count)};
    std::uint32_t *pairs{managed<std::uint32_t>(contact_count)};
    std::uint32_t *pair_count{managed<std::uint32_t>(1U)};
    RigidConstraintResource *joints{managed<RigidConstraintResource>(1U)};
    GeometryConstraintBody *bodies{
        managed<GeometryConstraintBody>(body_count)};
    ContactSchedule *schedule{managed<ContactSchedule>(1U)};

    StackFixture() {
        for (std::uint32_t index = 0U; index < body_count; ++index) {
            parameters[index].motion = index == 0U
                ? MotionType::static_body : MotionType::dynamic;
            parameters[index].inverse_mass = index == 0U ? 0.0F : 1.0F;
            parameters[index].inverse_inertia_local = index == 0U
                ? Vec3{} : Vec3{1.0F, 1.0F, 1.0F};
            parameters[index].collision_margin = 0.001F;
            previous[index].orientation.w = 1.0F;
            states[index] = previous[index];
            ids[index] = {index, 1U};
        }
        states[1].position.y = 0.05F;
        states[2].position.y = 0.149F;
        previous[1] = states[1];
        previous[2] = states[2];
        pairs[0] = body_count;
        pairs[1] = 2U * body_count + 1U;
        *pair_count = contact_count;
    }

    ~StackFixture() {
        for (void *pointer : std::array<void *, 10>{parameters, previous,
                 states, ids, manifolds, pairs, pair_count, joints, bodies,
                 schedule})
            cudaFree(pointer);
    }

    bool allocated() const {
        return parameters && previous && states && ids && manifolds && pairs &&
            pair_count && joints && bodies && schedule;
    }
};

void set_stack_contact(ContactManifold &manifold, float penetration,
                       Vec3 point) {
    manifold.count = 1U;
    auto &contact = manifold.contacts[0];
    contact.normal = {0.0F, 1.0F, 0.0F};
    contact.point = point;
    contact.penetration = std::max(0.0F, penetration);
}

void run_stack_window() {
    StackFixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;

    constexpr std::uint32_t cold_ticks = 120U;
    constexpr std::uint32_t settled_ticks = 600U;
    constexpr float half_extent = 0.05F;
    constexpr float gravity = -9.81F;
    float minimum_y1{};
    float maximum_y1{};
    float minimum_y2{};
    float maximum_y2{};
    float settled_energy{};
    float maximum_energy{};
    float maximum_penetration{};
    float maximum_rotation_drift{};
    std::uint32_t tunneling_events{};
    std::array<float, cold_ticks + settled_ticks> gpu_milliseconds{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK(cudaEventCreate(&start) == cudaSuccess);
    CHECK(cudaEventCreate(&stop) == cudaSuccess);
    if (failures != 0) return;

    for (std::uint32_t tick = 0U; tick < cold_ticks + settled_ticks; ++tick) {
        for (std::uint32_t body = 0U; body < StackFixture::body_count; ++body)
            fixture.previous[body] = fixture.states[body];
        for (std::uint32_t body = 1U; body < StackFixture::body_count; ++body) {
            fixture.states[body].linear_velocity.y +=
                gravity * rigid_realtime_timestep;
            fixture.states[body].position.y +=
                fixture.states[body].linear_velocity.y *
                rigid_realtime_timestep;
        }

        set_stack_contact(fixture.manifolds[0],
            half_extent - fixture.states[1].position.y,
            {0.0F, 0.0F, 0.0F});
        set_stack_contact(fixture.manifolds[1],
            fixture.states[1].position.y + 2.0F * half_extent -
                fixture.states[2].position.y,
            {0.0F, 0.5F * (fixture.states[1].position.y + half_extent +
                fixture.states[2].position.y - half_extent), 0.0F});

        CHECK(cudaEventRecord(start) == cudaSuccess);
        prepare_avbd_kernel<<<1U, 128U>>>(
            reinterpret_cast<BodyParameters_0 *>(fixture.parameters),
            reinterpret_cast<RigidBodyState_0 *>(fixture.previous),
            reinterpret_cast<RigidBodyState_0 *>(fixture.states),
            StackFixture::body_count,
            reinterpret_cast<ContactManifold_0 *>(fixture.manifolds),
            fixture.pairs, fixture.pair_count,
            reinterpret_cast<RigidConstraintResource_0 *>(fixture.joints), 0U,
            reinterpret_cast<RigidBodyId_0 *>(fixture.ids),
            reinterpret_cast<AvbdBody_0 *>(fixture.bodies),
            reinterpret_cast<ContactSchedule_0 *>(fixture.schedule),
            rigid_realtime_timestep, Vec3_0{0.0F, gravity, 0.0F}, 64U, 64U,
            rigid_realtime_profile_version, 1U);
        solve_avbd_kernel<<<1U, 32U>>>(
            reinterpret_cast<BodyParameters_0 *>(fixture.parameters),
            reinterpret_cast<RigidBodyState_0 *>(fixture.previous),
            reinterpret_cast<RigidBodyState_0 *>(fixture.states),
            StackFixture::body_count,
            reinterpret_cast<ContactManifold_0 *>(fixture.manifolds),
            fixture.pairs, fixture.pair_count,
            reinterpret_cast<RigidConstraintResource_0 *>(fixture.joints), 0U,
            reinterpret_cast<AvbdBody_0 *>(fixture.bodies),
            reinterpret_cast<ContactSchedule_0 *>(fixture.schedule),
            rigid_realtime_timestep);
        CHECK(cudaGetLastError() == cudaSuccess);
        CHECK(cudaEventRecord(stop) == cudaSuccess);
        CHECK(cudaEventSynchronize(stop) == cudaSuccess);
        CHECK(cudaEventElapsedTime(&gpu_milliseconds[tick], start, stop) ==
              cudaSuccess);

        for (auto &manifold : std::span<ContactManifold>(
                 fixture.manifolds, StackFixture::contact_count))
            manifold.contacts[0].avbd_cached = 1U;
        const float floor_penetration = std::max(0.0F,
            half_extent - fixture.states[1].position.y);
        const float stack_penetration = std::max(0.0F,
            fixture.states[1].position.y + 2.0F * half_extent -
                fixture.states[2].position.y);
        maximum_penetration = std::max(maximum_penetration,
            std::max(floor_penetration, stack_penetration));
        for (std::uint32_t body = 1U; body < StackFixture::body_count; ++body) {
            const auto &orientation = fixture.states[body].orientation;
            const float angle = 2.0F * std::atan2(std::sqrt(
                orientation.x * orientation.x +
                orientation.y * orientation.y +
                orientation.z * orientation.z), std::abs(orientation.w));
            maximum_rotation_drift = std::max(maximum_rotation_drift, angle);
        }
        if (fixture.states[1].position.y < -half_extent ||
            fixture.states[2].position.y <= fixture.states[1].position.y)
            ++tunneling_events;

        if (tick >= cold_ticks) {
            const float energy = 0.5F * (
                fixture.states[1].linear_velocity.y *
                    fixture.states[1].linear_velocity.y +
                fixture.states[2].linear_velocity.y *
                    fixture.states[2].linear_velocity.y) -
                gravity * (fixture.states[1].position.y +
                    fixture.states[2].position.y);
            if (tick == cold_ticks) {
                minimum_y1 = maximum_y1 = fixture.states[1].position.y;
                minimum_y2 = maximum_y2 = fixture.states[2].position.y;
                settled_energy = energy;
                maximum_energy = energy;
            } else {
                minimum_y1 = std::min(minimum_y1, fixture.states[1].position.y);
                maximum_y1 = std::max(maximum_y1, fixture.states[1].position.y);
                minimum_y2 = std::min(minimum_y2, fixture.states[2].position.y);
                maximum_y2 = std::max(maximum_y2, fixture.states[2].position.y);
                maximum_energy = std::max(maximum_energy, energy);
            }
        }
        CHECK(fixture.schedule->budget == rigid_realtime_sweeps);
        CHECK(fixture.schedule->counts[6] == rigid_realtime_sweeps);
        CHECK(fixture.schedule->counts[7] == 0U);
    }

    const float drift = std::max(maximum_y1 - minimum_y1,
                                 maximum_y2 - minimum_y2);
    const float energy_growth = settled_energy > 0.0F
        ? maximum_energy / settled_energy - 1.0F : 0.0F;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    auto sorted_milliseconds = gpu_milliseconds;
    std::sort(sorted_milliseconds.begin(), sorted_milliseconds.end());
    const float p50_ms = sorted_milliseconds[
        sorted_milliseconds.size() * 50U / 100U];
    const float p95_ms = sorted_milliseconds[
        sorted_milliseconds.size() * 95U / 100U];
    const float p99_ms = sorted_milliseconds[
        sorted_milliseconds.size() * 99U / 100U];
    const float maximum_ms = sorted_milliseconds.back();
    std::printf("F8_RIGID_60HZ_STACK ticks=%u maximum_penetration=%.9g "
                "settled_drift=%.9g rotation_drift=%.9g "
                "energy_growth=%.9g tunneling_events=%u\n",
                cold_ticks + settled_ticks, maximum_penetration, drift,
                maximum_rotation_drift, energy_growth, tunneling_events);
    std::printf("F8_RIGID_60HZ_TIMING workload=two-body-stack samples=%zu "
                "p50_ms=%.9g p95_ms=%.9g p99_ms=%.9g maximum_ms=%.9g\n",
                sorted_milliseconds.size(), p50_ms, p95_ms, p99_ms,
                maximum_ms);
    CHECK(maximum_penetration <= 0.005F);
    CHECK(drift <= 0.01F);
    CHECK(maximum_rotation_drift <= 0.035F);
    CHECK(energy_growth <= 0.05F);
    CHECK(tunneling_events == 0U);
    CHECK(p95_ms <= 10.0F);
}

void run_fixture() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;

    const float resting_start = fixture.states[1].position.y;
    const float ratio_start_a = fixture.states[3].position.x;
    const float ratio_start_b = fixture.states[4].position.x;
    const float impact_energy = 0.5F *
        fixture.states[2].linear_velocity.y *
        fixture.states[2].linear_velocity.y;
    prepare_avbd_kernel<<<1U, 128U>>>(
        reinterpret_cast<BodyParameters_0 *>(fixture.parameters),
        reinterpret_cast<RigidBodyState_0 *>(fixture.previous),
        reinterpret_cast<RigidBodyState_0 *>(fixture.states),
        Fixture::body_count,
        reinterpret_cast<ContactManifold_0 *>(fixture.manifolds),
        fixture.pairs, fixture.pair_count,
        reinterpret_cast<RigidConstraintResource_0 *>(fixture.joints),
        Fixture::joint_count,
        reinterpret_cast<RigidBodyId_0 *>(fixture.ids),
        reinterpret_cast<AvbdBody_0 *>(fixture.bodies),
        reinterpret_cast<ContactSchedule_0 *>(fixture.schedule),
        rigid_realtime_timestep, Vec3_0{}, 64U, 64U,
        rigid_realtime_profile_version, 1U);
    solve_avbd_kernel<<<1U, 32U>>>(
        reinterpret_cast<BodyParameters_0 *>(fixture.parameters),
        reinterpret_cast<RigidBodyState_0 *>(fixture.previous),
        reinterpret_cast<RigidBodyState_0 *>(fixture.states),
        Fixture::body_count,
        reinterpret_cast<ContactManifold_0 *>(fixture.manifolds),
        fixture.pairs, fixture.pair_count,
        reinterpret_cast<RigidConstraintResource_0 *>(fixture.joints),
        Fixture::joint_count,
        reinterpret_cast<AvbdBody_0 *>(fixture.bodies),
        reinterpret_cast<ContactSchedule_0 *>(fixture.schedule),
        rigid_realtime_timestep);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    CHECK(fixture.schedule->offsets[0] == rigid_realtime_profile_version);
    CHECK(fixture.schedule->budget == rigid_realtime_sweeps);
    CHECK(fixture.schedule->counts[6] == rigid_realtime_sweeps);
    CHECK(fixture.schedule->counts[7] == 0U);
    CHECK(fixture.schedule->maximum_passes == rigid_realtime_sweeps);
    CHECK(fixture.joints[0].avbd_count == 6U);

    for (std::uint32_t body = 0U; body < Fixture::body_count; ++body) {
        const auto &state = fixture.states[body];
        const float values[]{state.position.x, state.position.y,
            state.position.z, state.orientation.x, state.orientation.y,
            state.orientation.z, state.orientation.w,
            state.linear_velocity.x, state.linear_velocity.y,
            state.linear_velocity.z, state.angular_velocity.x,
            state.angular_velocity.y, state.angular_velocity.z};
        for (const float value : values) CHECK(std::isfinite(value));
    }

    const float resting_correction =
        fixture.states[1].position.y - resting_start;
    const float resting_penetration = std::max(0.0F,
        0.01F - resting_correction);
    CHECK(resting_penetration <= 0.005F);
    CHECK(std::abs(resting_correction) <= 0.01F);
    CHECK(fixture.states[2].linear_velocity.y >= -0.05F);
    const float final_impact_energy = 0.5F *
        fixture.states[2].linear_velocity.y *
        fixture.states[2].linear_velocity.y;
    const float ratio_relative_correction =
        (fixture.states[3].position.x - ratio_start_a) -
        (fixture.states[4].position.x - ratio_start_b);
    const float ratio_penetration = std::max(0.0F,
        0.01F + ratio_relative_correction);
    const auto &joint_orientation = fixture.states[5].orientation;
    const float joint_angle = 2.0F * std::atan2(std::sqrt(
        joint_orientation.x * joint_orientation.x +
        joint_orientation.y * joint_orientation.y +
        joint_orientation.z * joint_orientation.z),
        std::abs(joint_orientation.w));
    std::printf("F8_RIGID_60HZ_RESULT resting_penetration=%.9g "
                "impact_velocity=%.9g impact_energy_ratio=%.9g "
                "mass_ratio_penetration=%.9g joint_error=%.9g joint_angle=%.9g "
                "angular_velocity=%.9g\n",
                resting_penetration, fixture.states[2].linear_velocity.y,
                final_impact_energy / impact_energy,
                ratio_penetration,
                std::abs(fixture.states[5].position.x),
                joint_angle,
                std::abs(fixture.states[6].angular_velocity.z));
    CHECK(final_impact_energy <= 1.05F * impact_energy);
    CHECK(ratio_penetration <= 0.005F);
    CHECK(std::abs(fixture.states[5].position.x) <= 0.01F);
    CHECK(joint_angle <= 0.0523598776F);
    CHECK(std::abs(fixture.states[6].angular_velocity.z) <= 30.0F);
    CHECK(std::abs(fixture.states[7].position.x - 5.0F) < 1.0e-6F);
    CHECK(fixture.states[7].linear_velocity.x == 0.0F &&
          fixture.states[7].linear_velocity.y == 0.0F &&
          fixture.states[7].linear_velocity.z == 0.0F);
    CHECK(fixture.states[7].angular_velocity.x == 0.0F &&
          fixture.states[7].angular_velocity.y == 0.0F &&
          fixture.states[7].angular_velocity.z == 0.0F);
}

} // namespace

int main() {
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    run_fixture();
    run_stack_window();
    if (failures == 0)
        std::printf("F8-11 physical CUDA rigid 60 Hz solve passed on %s\n",
                    properties.name);
    return failures == 0 ? 0 : 1;
}
