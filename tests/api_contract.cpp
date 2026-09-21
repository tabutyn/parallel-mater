// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <type_traits>

using namespace parallel_mater;

static_assert(std::is_trivially_copyable_v<Vec2>);
static_assert(std::is_trivially_copyable_v<Vec3>);
static_assert(std::is_trivially_copyable_v<Quaternion>);
static_assert(std::is_trivially_copyable_v<FluidId>);
static_assert(std::is_trivially_copyable_v<RigidBodyId>);
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

constexpr auto sphere = CollisionShape::sphere(0.5F);
constexpr auto box = CollisionShape::box({1.0F, 2.0F, 3.0F});
static_assert(sphere.type == ShapeType::sphere && sphere.dimensions.x == 0.5F);
static_assert(box.type == ShapeType::box && box.dimensions.y == 2.0F);

int main() { return 0; }
