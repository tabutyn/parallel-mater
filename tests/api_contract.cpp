// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <type_traits>

using namespace parallel_mater;

static_assert(std::is_trivially_copyable_v<Vec2>);
static_assert(std::is_trivially_copyable_v<Vec3>);
static_assert(std::is_trivially_copyable_v<Quaternion>);
static_assert(std::is_trivially_copyable_v<FluidId>);
static_assert(std::is_trivially_copyable_v<RigidBodyId>);
static_assert(std::is_trivially_copyable_v<TriangleMeshId>);
static_assert(std::is_trivially_copyable_v<ParticleSpawnPlaneId>);
static_assert(std::is_trivially_copyable_v<ParticleDestroyPlaneId>);
static_assert(std::is_trivially_copyable_v<FluidParticle>);
static_assert(std::is_trivially_copyable_v<ParticlePlane>);
static_assert(std::is_trivially_copyable_v<RigidBodyState>);
static_assert(std::is_trivially_copyable_v<ContactEvent>);
static_assert(!std::is_copy_constructible_v<World>);
static_assert(std::is_move_constructible_v<World>);
static_assert(!std::is_copy_constructible_v<FrameToken>);
static_assert(std::is_move_constructible_v<FrameToken>);

constexpr RigidBodyOptions triangle_body{.mesh = {2U, 3U}};
static_assert(triangle_body.mesh == TriangleMeshId{2U, 3U});
static_assert(triangle_body.collision_margin == 0.005F);

int main() { return 0; }
