// SPDX-License-Identifier: MIT
#include "rope_dispatch.hpp"

#include "parallel_mater_rope_runtime.cu"

#include <cstring>
#include <type_traits>

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

RopeData_0 shader_rope(const RopeDispatchData &value) noexcept {
    static_assert(std::is_trivially_copyable_v<RopeDispatchData>);
    static_assert(sizeof(RopeDispatchData) == sizeof(RopeData_0));
    static_assert(alignof(RopeDispatchData) == alignof(RopeData_0));
    RopeData_0 result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

RopeSoftTarget_0 shader_target(
    const RopeSoftTargetDispatch &value) noexcept {
    static_assert(std::is_trivially_copyable_v<RopeSoftTargetDispatch>);
    static_assert(sizeof(RopeSoftTargetDispatch) == sizeof(RopeSoftTarget_0));
    static_assert(alignof(RopeSoftTargetDispatch) == alignof(RopeSoftTarget_0));
    RopeSoftTarget_0 result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

} // namespace

void advance_rope(device::ExecutionQueue stream, const RopeDispatchData &rope,
    float timestep, Vec3 gravity, int first, int last,
    const void *body_parameters, void *body_states,
    const void *previous_body_states, const void *meshes,
    std::uint32_t body_count, bool first_substep,
    const RopeSoftTargetDispatch &soft_a,
    const RopeSoftTargetDispatch &soft_b) noexcept {
    auto data = shader_rope(rope);
    auto first_target = shader_target(soft_a);
    auto second_target = shader_target(soft_b);
    const Vec3_0 shader_gravity{gravity.x, gravity.y, gravity.z};
    rope_advance<<<1U, 128U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(data, timestep, shader_gravity,
        first, last,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(previous_body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, first_substep, first_target, second_target);
}

} // namespace parallel_mater::simulation
