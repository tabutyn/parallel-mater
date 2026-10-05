// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#include <cstddef>
#include <type_traits>

using namespace parallel_mater::metal;

static_assert(sizeof(Vec2) == 8U);
static_assert(sizeof(Vec3) == 12U);
static_assert(sizeof(Quaternion) == 16U);
static_assert(offsetof(Vec3, z) == 8U);
static_assert(std::is_trivially_copyable_v<NativeContext>);
static_assert(std::is_trivially_copyable_v<BufferSpan<const Vec3>>);
static_assert(std::is_trivially_copyable_v<HostSpan<const Vec3>>);
static_assert(std::is_trivially_copyable_v<FluidId>);
static_assert(std::is_trivially_copyable_v<RigidBodyId>);
static_assert(std::is_trivially_copyable_v<RigidConstraintOptions>);
static_assert(std::is_trivially_copyable_v<ContactEvent>);
static_assert(std::is_trivially_copyable_v<WorldStepTimings>);
static_assert(std::is_same_v<parallel_mater::metal::FluidId,
                             parallel_mater::FluidId>);
static_assert(std::is_same_v<parallel_mater::metal::RigidBodyId,
                             parallel_mater::RigidBodyId>);
static_assert(std::is_same_v<parallel_mater::metal::RopeClothCouplingId,
                             parallel_mater::RopeClothCouplingId>);
static_assert(std::is_same_v<parallel_mater::metal::WorldOptions,
                             parallel_mater::WorldOptions>);
static_assert(std::is_same_v<parallel_mater::metal::RigidBodyOptions,
                             parallel_mater::RigidBodyOptions>);
static_assert(std::is_same_v<parallel_mater::metal::HitBox,
                             parallel_mater::HitBox>);
static_assert(std::is_same_v<parallel_mater::metal::HitBoxResult,
                             parallel_mater::HitBoxResult>);
static_assert(std::is_same_v<parallel_mater::metal::RigidConstraintOptions,
                             parallel_mater::RigidConstraintOptions>);
static_assert(std::is_same_v<parallel_mater::metal::FluidOptions,
                             parallel_mater::FluidOptions>);
static_assert(std::is_same_v<parallel_mater::metal::ClothOptions,
                             parallel_mater::ClothOptions>);
static_assert(std::is_same_v<parallel_mater::metal::SoftBodyOptions,
                             parallel_mater::SoftBodyOptions>);
static_assert(std::is_same_v<parallel_mater::metal::RopeOptions,
                             parallel_mater::RopeOptions>);
static_assert(std::is_same_v<parallel_mater::metal::SmokeOptions,
                             parallel_mater::SmokeOptions>);
static_assert(std::is_same_v<parallel_mater::metal::FluidClothCouplingOptions,
                             parallel_mater::FluidClothCouplingOptions>);
static_assert(std::is_same_v<parallel_mater::metal::RopeSoftBodyCouplingOptions,
                             parallel_mater::RopeSoftBodyCouplingOptions>);
static_assert(std::is_same_v<parallel_mater::metal::SmokeRigidCouplingOptions,
                             parallel_mater::SmokeRigidCouplingOptions>);
static_assert(std::is_same_v<parallel_mater::metal::ContactEvent,
                             parallel_mater::ContactEvent>);
static_assert(std::is_same_v<parallel_mater::metal::WorldStatistics,
                             parallel_mater::WorldStatistics>);
static_assert(!std::is_copy_constructible_v<World>);
static_assert(std::is_move_constructible_v<World>);
static_assert(!std::is_copy_constructible_v<FrameToken>);
static_assert(std::is_move_constructible_v<FrameToken>);

constexpr BufferSpan<const Vec3> buffer_span{
    nullptr, 48U, 12U};
static_assert(!buffer_span.empty());
static_assert(buffer_span.byte_offset == 48U);
static_assert(buffer_span.size == 12U);

int main() { return 0; }
