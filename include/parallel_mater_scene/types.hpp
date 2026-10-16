// SPDX-License-Identifier: MIT
#pragma once
// Host-side geometry/preparation values shared by tools and reference drivers.
// No solver handles, resource ownership, native API types or device execution.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace parallel_mater {

struct Vec2 {
    float x{};
    float y{};
};

// Three scalar floats are used instead of a SIMD type so CPU, CUDA, and MSL
// agree on the 12-byte storage contract.
struct Vec3 {
    float x{};
    float y{};
    float z{};
};

struct Quaternion {
    float x{};
    float y{};
    float z{};
    float w{1.0F};
};

template <typename T> struct HostSpan {
    const T *data{};
    std::uint64_t size{};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0U; }
};

struct FluidParticle {
    Vec3 position{};
    Vec3 velocity{};
    float temperature{20.0F}; // degrees Celsius
};

struct SoftBodyBond {
    std::uint32_t first{};
    std::uint32_t second{};
    float rest_length{};
};

// Delta-skinning binding from one authored surface vertex to up to four
// physical lattice nodes. Weights must be finite, nonnegative, and sum to one.
struct SoftBodySurfaceBinding {
    std::uint32_t nodes[4]{};
    float weights[4]{};
};

// Mapping back to the input surface, for interpolating renderer-owned UVs,
// normals, or other vertex attributes after conforming triangle refinement.
struct SoftBodySurfaceSource {
    std::uint32_t vertices[3]{};
    float weights[3]{};
};

struct SoftBodyGeometrySource {
    HostSpan<Vec3> vertices{};
    HostSpan<std::uint32_t> triangle_indices{};
    // Optional full-weight Goal pins. Refined edges/faces interpolate weights;
    // only weight one is fixed. Rendering seams share one physical node.
    HostSpan<float> pin_weights{};
    float spacing{0.2F};
    float total_mass{1.0F};
    std::uint32_t maximum_nodes{100'000U};
    std::uint32_t maximum_surface_triangles{200'000U};
};

struct SoftBodyGeometry {
    std::vector<Vec3> nodes{};
    std::vector<SoftBodyBond> bonds{};
    std::vector<float> inverse_masses{};
    std::vector<Vec3> surface_vertices{};
    std::vector<std::uint32_t> surface_triangle_indices{};
    std::vector<SoftBodySurfaceBinding> surface_bindings{};
    std::vector<SoftBodySurfaceSource> surface_sources{};
    float node_mass{};
};

// Arbitrary open or closed triangle surface, in world coordinates. Copied and
// sampled at registration; the caller may release these host buffers afterward.
struct ParticleSourceMesh {
    HostSpan<const Vec3> vertices{};
    HostSpan<const std::uint32_t> triangle_indices{};
    // Minimum distance between emission sites AND clearance from existing water.
    // Zero at registration selects the fluid support radius. Must be at least
    // the particle diameter. A site emits only when its clearance is empty.
    float spacing{};
};

struct RigidBodyState {
    Vec3 position{};
    Quaternion orientation{};
    Vec3 linear_velocity{};
    Vec3 angular_velocity{};
};

// A closed, indexed, body-local triangle mesh sampled once into an HCP
// particle lattice. Host buffers are borrowed only for the duration of the
// call. A volume may have multiple disconnected closed components.
struct FluidGeometrySource {
    HostSpan<Vec3> vertices{};
    HostSpan<std::uint32_t> triangle_indices{};
    RigidBodyState transform{};
    Vec3 initial_velocity{};
    float spacing{0.06F};
};

} // namespace parallel_mater
