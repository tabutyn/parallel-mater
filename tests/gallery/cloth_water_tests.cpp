// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using parallel_mater::ClothDeviceView;
using parallel_mater::FluidDeviceView;
using parallel_mater::Vec3;

Vec3 subtract(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float signed_volume(const std::vector<Vec3> &positions,
                    const std::vector<std::uint32_t> &indices) {
    float result = 0.0F;
    for (std::size_t triangle = 0U; triangle < indices.size(); triangle += 3U)
        result += dot(positions[indices[triangle]], cross(
            positions[indices[triangle + 1U]],
            positions[indices[triangle + 2U]])) / 6.0F;
    return result;
}

Vec3 closest_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    const Vec3 ab = subtract(b, a), ac = subtract(c, a), ap = subtract(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) return a;
    const Vec3 bp = subtract(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        const float v = d1 / (d1 - d3);
        return {a.x + ab.x * v, a.y + ab.y * v, a.z + ab.z * v};
    }
    const Vec3 cp = subtract(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        const float w = d2 / (d2 - d6);
        return {a.x + ac.x * w, a.y + ac.y * w, a.z + ac.z * w};
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vec3 bc = subtract(c, b);
        return {b.x + bc.x * w, b.y + bc.y * w, b.z + bc.z * w};
    }
    const float inverse = 1.0F / (va + vb + vc);
    const float v = vb * inverse, w = vc * inverse;
    return {a.x + ab.x * v + ac.x * w,
            a.y + ab.y * v + ac.y * w,
            a.z + ab.z * v + ac.z * w};
}

std::uint32_t outside_particles(
    const std::vector<Vec3> &particles, const std::vector<Vec3> &cloth,
    const std::vector<std::uint32_t> &indices, float orientation) {
    std::uint32_t outside = 0U;
    for (Vec3 point : particles) {
        float best = std::numeric_limits<float>::max();
        float signed_distance = 0.0F;
        for (std::size_t triangle = 0U; triangle < indices.size(); triangle += 3U) {
            const Vec3 a = cloth[indices[triangle]];
            const Vec3 b = cloth[indices[triangle + 1U]];
            const Vec3 c = cloth[indices[triangle + 2U]];
            const Vec3 nearest = closest_triangle(point, a, b, c);
            const Vec3 offset = subtract(point, nearest);
            const float squared = dot(offset, offset);
            if (squared >= best) continue;
            const Vec3 raw_normal = cross(subtract(b, a), subtract(c, a));
            const float length = std::sqrt(dot(raw_normal, raw_normal));
            if (length <= 1.0e-8F) continue;
            best = squared;
            signed_distance = orientation * dot(offset, raw_normal) / length;
        }
        if (signed_distance > 1.0e-3F) ++outside;
    }
    return outside;
}

bool require(parallel_mater::Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": "
              << (status.message != nullptr ? status.message : "failed") << '\n';
    return false;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;

    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_CLOTH_WATER_SCENE_PATH, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (scene.cloths.size() != 1U || !scene.cloths[0].preserve_volume ||
        !scene.cloths[0].contains_fluid || scene.initial_particles.empty()) {
        std::cerr << "ClothWater metadata did not survive export\n";
        return 1;
    }
    if (std::any_of(scene.cloths[0].inverse_masses.begin(),
                    scene.cloths[0].inverse_masses.end(),
                    [](float mass) { return mass <= 0.0F; })) {
        std::cerr << "ClothWater should be an unpinned pressure skin\n";
        return 1;
    }

    World world;
    SceneInstance instance;
    if (!require(create_scene_world(scene, world, instance,
                                    {.frame_capacity = 1U}),
                 "create scene world"))
        return 1;
    if (!instance.has_fluid || instance.cloths.size() != 1U ||
        instance.fluid_cloth_couplings.size() != 1U) {
        std::cerr << "scene did not instantiate its fluid-cloth coupling\n";
        return 1;
    }

    ClothDeviceView cloth_view;
    if (!require(world.cloth_view(instance.cloths[0], cloth_view), "cloth view"))
        return 1;
    if (cloth_view.rigid_contact_forces.size != cloth_view.vertex_count ||
        cloth_view.fluid_contact_forces.size != cloth_view.vertex_count) {
        std::cerr << "cloth coupling diagnostics are not exposed per vertex\n";
        return 1;
    }
    std::vector<std::uint32_t> indices(cloth_view.triangle_indices.size);
    std::vector<Vec3> cloth(cloth_view.positions.size);
    cudaMemcpy(indices.data(), cloth_view.triangle_indices.data,
               indices.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cloth.data(), cloth_view.positions.data,
               cloth.size() * sizeof(Vec3), cudaMemcpyDeviceToHost);
    std::vector<Vec3> rigid_forces(cloth_view.rigid_contact_forces.size);
    std::vector<Vec3> fluid_forces(cloth_view.fluid_contact_forces.size);
    cudaMemcpy(rigid_forces.data(), cloth_view.rigid_contact_forces.data,
               rigid_forces.size() * sizeof(Vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(fluid_forces.data(), cloth_view.fluid_contact_forces.data,
               fluid_forces.size() * sizeof(Vec3), cudaMemcpyDeviceToHost);
    const auto finite = [](Vec3 value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
               std::isfinite(value.z);
    };
    if (!std::all_of(rigid_forces.begin(), rigid_forces.end(), finite) ||
        !std::all_of(fluid_forces.begin(), fluid_forces.end(), finite)) {
        std::cerr << "cloth coupling diagnostics contain nonfinite forces\n";
        return 1;
    }
    const float initial_signed_volume = signed_volume(cloth, indices);
    const float initial_volume = std::fabs(initial_signed_volume);
    const float orientation = initial_signed_volume < 0.0F ? -1.0F : 1.0F;

    const auto started = std::chrono::steady_clock::now();
    float minimum_ratio = 1.0F, maximum_ratio = 1.0F;
    for (std::uint32_t frame = 0U; frame < 90U; ++frame) {
        if (!require(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 4U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "step water cloth")) return 1;
        if (frame % 10U == 9U) {
            if (!require(world.cloth_view(instance.cloths[0], cloth_view),
                         "updated cloth view")) return 1;
            cudaMemcpy(cloth.data(), cloth_view.positions.data,
                       cloth.size() * sizeof(Vec3), cudaMemcpyDeviceToHost);
            const float ratio = std::fabs(signed_volume(cloth, indices)) /
                                initial_volume;
            minimum_ratio = std::min(minimum_ratio, ratio);
            maximum_ratio = std::max(maximum_ratio, ratio);
        }
    }
    const float milliseconds = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    PhysicsDebugFrameView debug_frame{};
    if (!require(world.physics_debug_frame(debug_frame),
                 "water-cloth physics debug frame")) return 1;
    if (debug_frame.fluid_particles.size != scene.initial_particles.size() ||
        debug_frame.cloth_vertices.size != cloth_view.vertex_count) {
        std::cerr << "physics capture omitted fluid or cloth state\n";
        return 1;
    }

    FluidDeviceView fluid_view;
    if (!require(world.fluid_view(instance.fluid, fluid_view), "fluid view"))
        return 1;
    std::vector<Vec3> particles(fluid_view.particle_count);
    cudaMemcpy(particles.data(), fluid_view.positions.data,
               particles.size() * sizeof(Vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(cloth.data(), cloth_view.positions.data,
               cloth.size() * sizeof(Vec3), cudaMemcpyDeviceToHost);
    const std::uint32_t escaped = outside_particles(
        particles, cloth, indices, orientation);
    std::cout << "water-cloth particles=" << particles.size()
              << " escaped=" << escaped
              << " volume_ratio=[" << minimum_ratio << ',' << maximum_ratio << ']'
              << " step_ms=" << milliseconds / 90.0F << '\n';
    if (particles.empty() || escaped != 0U || minimum_ratio < 0.92F ||
        maximum_ratio > 1.08F) {
        std::cerr << "water-cloth containment or volume stability regressed\n";
        return 1;
    }
    return 0;
}
