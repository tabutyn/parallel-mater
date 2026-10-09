// SPDX-License-Identifier: MIT
#include <parallel_mater/types.hpp>

#include <cstddef>
#include <type_traits>

using namespace parallel_mater;

static_assert(sizeof(Vec2) == 8U && alignof(Vec2) == 4U);
static_assert(sizeof(Vec3) == 12U && alignof(Vec3) == 4U);
static_assert(sizeof(Quaternion) == 16U && alignof(Quaternion) == 4U);
static_assert(offsetof(Vec3, x) == 0U);
static_assert(offsetof(Vec3, y) == 4U);
static_assert(offsetof(Vec3, z) == 8U);
static_assert(std::is_trivially_copyable_v<Vec2>);
static_assert(std::is_trivially_copyable_v<Vec3>);
static_assert(std::is_trivially_copyable_v<Quaternion>);
static_assert(std::is_trivially_copyable_v<HostSpan<const Vec3>>);
static_assert(std::is_trivially_copyable_v<FluidId>);
static_assert(std::is_trivially_copyable_v<SmokeId>);
static_assert(std::is_trivially_copyable_v<RigidBodyId>);
static_assert(std::is_trivially_copyable_v<RigidConstraintId>);
static_assert(std::is_trivially_copyable_v<TriangleMeshId>);
static_assert(std::is_trivially_copyable_v<ClothId>);
static_assert(std::is_trivially_copyable_v<SoftBodyId>);
static_assert(std::is_trivially_copyable_v<RopeId>);
static_assert(sizeof(FluidId) == 8U);
static_assert(sizeof(ParticleDestroyPlaneId) == 8U);
static_assert(std::is_trivially_copyable_v<WorldOptions>);
static_assert(std::is_trivially_copyable_v<FluidOptions>);
static_assert(std::is_trivially_copyable_v<ClothOptions>);
static_assert(std::is_trivially_copyable_v<SoftBodyOptions>);
static_assert(std::is_trivially_copyable_v<RopeOptions>);
static_assert(std::is_trivially_copyable_v<SmokeOptions>);
static_assert(std::is_trivially_copyable_v<RigidBodyOptions>);
static_assert(std::is_trivially_copyable_v<RigidConstraintOptions>);
static_assert(std::is_trivially_copyable_v<FluidClothCouplingOptions>);
static_assert(std::is_trivially_copyable_v<SoftBodyClothCouplingOptions>);
static_assert(std::is_trivially_copyable_v<FluidSoftBodyCouplingOptions>);
static_assert(std::is_trivially_copyable_v<FluidRopeCouplingOptions>);
static_assert(std::is_trivially_copyable_v<RopeSoftBodyCouplingOptions>);
static_assert(std::is_trivially_copyable_v<RopeClothCouplingOptions>);
static_assert(std::is_trivially_copyable_v<SmokeRigidCouplingOptions>);
static_assert(std::is_trivially_copyable_v<SmokeClothCouplingOptions>);
static_assert(std::is_trivially_copyable_v<SmokeSoftBodyCouplingOptions>);
static_assert(std::is_trivially_copyable_v<SmokeRopeCouplingOptions>);
static_assert(std::is_trivially_copyable_v<ContactEvent>);
static_assert(std::is_trivially_copyable_v<RigidContactEvent>);
static_assert(std::is_trivially_copyable_v<WorldStatistics>);
static_assert(std::is_trivially_copyable_v<WorldStepTimings>);
static_assert(std::is_trivially_copyable_v<StepOptions>);
static_assert(StepOptions{}.rigid_contact_pass_limit == 0U);

int main() { return 0; }
