// SPDX-License-Identifier: MIT
#include <parallel_mater/d3d12.hpp>

#include <cstddef>
#include <type_traits>

using namespace parallel_mater::d3d12;

static_assert(sizeof(Vec2) == 8U);
static_assert(sizeof(Vec3) == 12U);
static_assert(sizeof(Quaternion) == 16U);
static_assert(offsetof(Vec3, z) == 8U);
static_assert(std::is_trivially_copyable_v<NativeContext>);
static_assert(std::is_trivially_copyable_v<BufferSpan<const Vec3>>);
static_assert(std::is_trivially_copyable_v<HostSpan<const Vec3>>);
static_assert(std::is_trivially_copyable_v<RigidBodyId>);
static_assert(std::is_trivially_copyable_v<RigidConstraintOptions>);
static_assert(std::is_trivially_copyable_v<RigidContactEvent>);
static_assert(std::is_trivially_copyable_v<WorldStepTimings>);
static_assert(std::is_same_v<parallel_mater::d3d12::RigidBodyId,
                             parallel_mater::RigidBodyId>);
static_assert(std::is_same_v<parallel_mater::d3d12::WorldOptions,
                             parallel_mater::WorldOptions>);
static_assert(std::is_same_v<parallel_mater::d3d12::RigidBodyOptions,
                             parallel_mater::RigidBodyOptions>);
static_assert(std::is_same_v<parallel_mater::d3d12::RigidConstraintOptions,
                             parallel_mater::RigidConstraintOptions>);
static_assert(std::is_same_v<parallel_mater::d3d12::WorldStatistics,
                             parallel_mater::WorldStatistics>);
static_assert(!std::is_copy_constructible_v<World>);
static_assert(std::is_move_constructible_v<World>);
static_assert(!std::is_copy_constructible_v<FrameToken>);
static_assert(std::is_move_constructible_v<FrameToken>);

constexpr BufferSpan<const Vec3> span{nullptr, 48U, 12U};
static_assert(!span.empty());
static_assert(span.byte_offset == 48U);
static_assert(span.size == 12U);

int main() { return 0; }
