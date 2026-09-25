// SPDX-License-Identifier: MIT
#include "fluid_surface.hpp"
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using parallel_mater::Vec3;
using parallel_mater::gallery::SceneDefinition;

float floor_height(const SceneDefinition &scene, float x, float z) {
    float highest = -std::numeric_limits<float>::infinity();
    for (const auto &body : scene.rigid_bodies) {
        if (body.options.motion != parallel_mater::MotionType::static_body)
            continue;
        const auto &meshes = body.collision_mesh_indices.empty()
            ? scene.meshes : scene.collision_meshes;
        const auto &mesh_indices = body.collision_mesh_indices.empty()
            ? body.mesh_indices : body.collision_mesh_indices;
        const auto state = body.options.initial_state;
        const auto transform = [&](Vec3 p) {
            const auto q = state.orientation;
            const Vec3 t{2.0F * (q.y * p.z - q.z * p.y),
                         2.0F * (q.z * p.x - q.x * p.z),
                         2.0F * (q.x * p.y - q.y * p.x)};
            return Vec3{p.x + q.w * t.x + q.y * t.z - q.z * t.y +
                            state.position.x,
                        p.y + q.w * t.y + q.z * t.x - q.x * t.z +
                            state.position.y,
                        p.z + q.w * t.z + q.x * t.y - q.y * t.x +
                            state.position.z};
        };
        for (std::uint32_t mesh_index : mesh_indices) {
            const auto &mesh = meshes[mesh_index];
            for (std::size_t i = 0; i < mesh.indices.size(); i += 3U) {
                const Vec3 a = transform(mesh.vertices[mesh.indices[i]].position);
                const Vec3 b = transform(mesh.vertices[mesh.indices[i + 1U]].position);
                const Vec3 c = transform(mesh.vertices[mesh.indices[i + 2U]].position);
                const float abx = b.x - a.x, abz = b.z - a.z;
                const float acx = c.x - a.x, acz = c.z - a.z;
                const float determinant = abx * acz - abz * acx;
                if (std::fabs(determinant) < 1.0e-8F) continue;
                const float px = x - a.x, pz = z - a.z;
                const float u = (px * acz - pz * acx) / determinant;
                const float v = (abx * pz - abz * px) / determinant;
                if (u < -1.0e-4F || v < -1.0e-4F ||
                    u + v > 1.0001F) continue;
                highest = std::max(highest,
                    a.y + u * (b.y - a.y) + v * (c.y - a.y));
            }
        }
    }
    return highest;
}

} // namespace

int main() {
    using parallel_mater::FluidDeviceView;
    using parallel_mater::Vec3;
    using parallel_mater::gallery::FluidSurface;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    std::vector<Vec3> positions(1'000U);
    for (std::uint32_t i = 0U; i < positions.size(); ++i)
        positions[i] = {0.15F * static_cast<float>(i % 25U), 0.0F,
                        0.15F * static_cast<float>(i / 25U)};
    Vec3 *device = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device),
                   positions.size() * sizeof(Vec3)) != cudaSuccess)
        return 1;

    int result = 0;
    try {
        FluidSurface surface(static_cast<std::uint32_t>(positions.size()));
        FluidDeviceView view{};
        view.positions = {device, positions.size()};
        view.particle_count = static_cast<std::uint32_t>(positions.size());
        view.particle_radius = 0.045F;
        view.support_radius = 0.18F;
        for (Vec3 &position : positions) position.y = view.particle_radius;
        if (cudaMemcpy(device, positions.data(), positions.size() * sizeof(Vec3),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            throw 1;
        (void)surface.update(view, positions);
        if (surface.excluded_particle_count() != 0U)
            throw 2;
        const auto baseline_grid = surface.view().grid;
        const std::size_t baseline_samples =
            static_cast<std::size_t>(baseline_grid.dimensions.x) *
            baseline_grid.dimensions.y * baseline_grid.dimensions.z;
        std::vector<float> baseline_values(baseline_samples);
        if (cudaMemcpy(baseline_values.data(), surface.view().values,
                       baseline_samples * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) throw 9;
        std::size_t below_floor = 0U;
        std::size_t wet_flat_cells = 0U;
        for (std::uint32_t z = 0; z < baseline_grid.dimensions.z; ++z)
            for (std::uint32_t y = 0; y < baseline_grid.dimensions.y; ++y)
                for (std::uint32_t x = 0; x < baseline_grid.dimensions.x; ++x) {
                    const float sample_y = baseline_grid.minimum.y +
                        y * baseline_grid.cell_size.y;
                    const std::size_t index =
                        (static_cast<std::size_t>(z) *
                             baseline_grid.dimensions.y + y) *
                            baseline_grid.dimensions.x + x;
                    below_floor += sample_y < -0.001F &&
                                   baseline_values[index] < 0.0F;
                    wet_flat_cells += baseline_values[index] < 0.0F;
                }
        std::cout << "Flat-floor water field below floor=" << below_floor
                  << " wet_cells=" << wet_flat_cells << '\n';
        if (below_floor != 0U || wet_flat_cells < 100U) throw 10;

        const Vec3 ordinary = positions.back();
        for (int axis = 0; axis < 3; ++axis) {
            positions.back() = ordinary;
            if (axis == 0) positions.back().x = 96.0F;
            if (axis == 1) positions.back().y = -96.0F;
            if (axis == 2) positions.back().z = -96.0F;
            if (cudaMemcpy(device, positions.data(),
                           positions.size() * sizeof(Vec3),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                throw 3;
            (void)surface.update(view, positions);
            const auto grid = surface.view().grid;
            const float maximum_x = grid.minimum.x + grid.cell_size.x *
                static_cast<float>(grid.dimensions.x - 1U);
            if (surface.excluded_particle_count() != 1U ||
                (axis == 0 && maximum_x > 10.0F) ||
                (axis == 1 && grid.minimum.y < -2.0F) ||
                (axis == 2 && grid.minimum.z < -2.0F) ||
                grid.dimensions.x < 40U || grid.dimensions.z < 40U)
                throw 4 + axis;
        }

        // A single-particle-thick sheet can fall between grid columns. It
        // should remain a continuous surface rather than separate bands.
        for (std::uint32_t i = 0U; i < 920U; ++i)
            positions[i] = {0.11F,
                            -1.8F + (i / 46U) * (1.6F / 19.0F),
                            -2.0F + (i % 46U) * (4.0F / 45.0F)};
        for (std::uint32_t i = 920U; i < 940U; ++i)
            positions[i] = {-7.0F, -1.0F, 0.01F * (i - 920U)};
        for (std::uint32_t i = 940U; i < 960U; ++i)
            positions[i] = {7.0F, -1.0F, 0.01F * (i - 940U)};
        for (std::uint32_t i = 960U; i < 980U; ++i)
            positions[i] = {0.0F, -1.0F, -7.0F};
        for (std::uint32_t i = 980U; i < 1'000U; ++i)
            positions[i] = {0.0F, -1.0F, 7.0F};
        if (cudaMemcpy(device, positions.data(), positions.size() * sizeof(Vec3),
                       cudaMemcpyHostToDevice) != cudaSuccess) throw 18;
        (void)surface.update(view, positions);
        const auto sheet_grid = surface.view().grid;
        const std::size_t sheet_samples =
            static_cast<std::size_t>(sheet_grid.dimensions.x) *
            sheet_grid.dimensions.y * sheet_grid.dimensions.z;
        std::vector<float> sheet_values(sheet_samples);
        if (cudaMemcpy(sheet_values.data(), surface.view().values,
                       sheet_samples * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) throw 19;
        const float sheet_x = (0.11F - sheet_grid.minimum.x) /
            sheet_grid.cell_size.x;
        const std::uint32_t sheet_column =
            static_cast<std::uint32_t>(std::floor(sheet_x));
        if (sheet_column + 1U >= sheet_grid.dimensions.x) throw 20;
        const float sheet_fraction = sheet_x - sheet_column;
        std::size_t sheet_wet = 0U, sheet_total = 0U;
        for (std::uint32_t z = 0U; z < sheet_grid.dimensions.z; ++z) {
            const float world_z = sheet_grid.minimum.z +
                z * sheet_grid.cell_size.z;
            if (world_z < -1.7F || world_z > 1.7F) continue;
            for (std::uint32_t y = 0U; y < sheet_grid.dimensions.y; ++y) {
                const float world_y = sheet_grid.minimum.y +
                    y * sheet_grid.cell_size.y;
                if (world_y < -1.6F || world_y > -0.4F) continue;
                ++sheet_total;
                const std::size_t index =
                    (static_cast<std::size_t>(z) *
                         sheet_grid.dimensions.y + y) *
                        sheet_grid.dimensions.x + sheet_column;
                const float interpolated = sheet_values[index] +
                    sheet_fraction * (sheet_values[index + 1U] -
                                      sheet_values[index]);
                sheet_wet += interpolated < 0.0F;
            }
        }
        std::cout << "Thin water sheet covered=" << sheet_wet << '/'
                  << sheet_total << " grid_pitch=" << sheet_grid.cell_size.x
                  << '\n';
        if (sheet_total == 0U || sheet_wet * 100U < sheet_total * 95U)
            throw 20;

        SceneDefinition scene;
        std::string error;
        if (!parallel_mater::gallery::load_glb_scene(
                PARALLEL_MATER_FLUID_SCENE_PATH, scene, error))
            throw std::runtime_error(error);
        parallel_mater::World world;
        if (!parallel_mater::World::create(
                {.rigid_body_capacity =
                     static_cast<std::uint32_t>(scene.rigid_bodies.size()),
                 .triangle_mesh_capacity = static_cast<std::uint32_t>(
                     scene.meshes.size() + scene.collision_meshes.size())},
                world)) throw 11;
        parallel_mater::gallery::SceneInstance instance;
        if (!parallel_mater::gallery::instantiate_scene(scene, world, instance))
            throw 12;
        for (int frame = 0; frame < 2'000; ++frame)
            if (!world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                             .gravity = {0.0F, -9.81F, 0.0F}})) throw 13;
        FluidDeviceView scene_view{};
        if (!world.fluid_view(instance.fluid, scene_view)) throw 14;
        std::vector<Vec3> scene_positions(scene_view.particle_count);
        if (cudaMemcpy(scene_positions.data(), scene_view.positions.data,
                       scene_positions.size() * sizeof(Vec3),
                       cudaMemcpyDeviceToHost) != cudaSuccess) throw 15;
        FluidSurface scene_surface(scene.fluid_options.capacity);
        (void)scene_surface.update(scene_view, scene_positions);
        const auto grid = scene_surface.view().grid;
        const std::size_t samples = static_cast<std::size_t>(grid.dimensions.x) *
            grid.dimensions.y * grid.dimensions.z;
        std::vector<float> values(samples);
        if (cudaMemcpy(values.data(), scene_surface.view().values,
                       samples * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) throw 16;
        std::vector<float> terrain(static_cast<std::size_t>(grid.dimensions.x) *
                                   grid.dimensions.z);
        for (std::uint32_t z = 0; z < grid.dimensions.z; ++z)
            for (std::uint32_t x = 0; x < grid.dimensions.x; ++x)
                terrain[static_cast<std::size_t>(z) * grid.dimensions.x + x] =
                    floor_height(scene, grid.minimum.x + x * grid.cell_size.x,
                                 grid.minimum.z + z * grid.cell_size.z);
        std::size_t below_scene_floor = 0U;
        std::size_t wet_scene_cells = 0U;
        for (std::uint32_t z = 0; z < grid.dimensions.z; ++z)
            for (std::uint32_t y = 0; y < grid.dimensions.y; ++y)
                for (std::uint32_t x = 0; x < grid.dimensions.x; ++x) {
                    const std::size_t index =
                        (static_cast<std::size_t>(z) * grid.dimensions.y + y) *
                            grid.dimensions.x + x;
                    const float floor = terrain[static_cast<std::size_t>(z) *
                                                grid.dimensions.x + x];
                    const float sample_y = grid.minimum.y + y * grid.cell_size.y;
                    below_scene_floor += values[index] < 0.0F &&
                                         sample_y < floor - 0.001F;
                    wet_scene_cells += values[index] < 0.0F;
                }
        std::cout << "Fluid scene water field below terrain="
                  << below_scene_floor << " wet_cells=" << wet_scene_cells
                  << '\n';
        std::size_t below_interpolated_floor = 0U;
        for (std::uint32_t z = 0; z < grid.dimensions.z; ++z)
            for (std::uint32_t x = 0; x < grid.dimensions.x; ++x) {
                const float floor = terrain[static_cast<std::size_t>(z) *
                                            grid.dimensions.x + x];
                const float sample_y = floor - 0.001F;
                const float coordinate =
                    (sample_y - grid.minimum.y) / grid.cell_size.y;
                if (coordinate < 0.0F ||
                    coordinate >= grid.dimensions.y - 1U) continue;
                const std::uint32_t y = static_cast<std::uint32_t>(coordinate);
                const float fraction = coordinate - y;
                const std::size_t first =
                    (static_cast<std::size_t>(z) * grid.dimensions.y + y) *
                        grid.dimensions.x + x;
                const float interpolated = values[first] + fraction *
                    (values[first + grid.dimensions.x] - values[first]);
                below_interpolated_floor += interpolated < 0.0F;
                if (interpolated < 0.0F && below_interpolated_floor <= 3U)
                    std::cout << "Below interpolated floor x=" <<
                        grid.minimum.x + x * grid.cell_size.x
                        << " z=" << grid.minimum.z + z * grid.cell_size.z
                        << " floor=" << floor
                        << " field=" << interpolated << '\n';
            }
        std::cout << "Fluid scene interpolated water below terrain="
                  << below_interpolated_floor << '\n';
        std::size_t below_cell_midpoints = 0U;
        for (std::uint32_t z = 0; z + 1U < grid.dimensions.z; ++z)
            for (std::uint32_t x = 0; x + 1U < grid.dimensions.x; ++x) {
                const float world_x = grid.minimum.x +
                    (x + 0.5F) * grid.cell_size.x;
                const float world_z = grid.minimum.z +
                    (z + 0.5F) * grid.cell_size.z;
                const float sample_y = floor_height(scene, world_x, world_z) -
                    0.001F;
                const float coordinate =
                    (sample_y - grid.minimum.y) / grid.cell_size.y;
                if (coordinate < 0.0F ||
                    coordinate >= grid.dimensions.y - 1U) continue;
                const std::uint32_t y = static_cast<std::uint32_t>(coordinate);
                const float fraction = coordinate - y;
                const std::size_t layer = static_cast<std::size_t>(
                    grid.dimensions.x) * grid.dimensions.y;
                const std::size_t first =
                    (static_cast<std::size_t>(z) * grid.dimensions.y + y) *
                        grid.dimensions.x + x;
                const float lower = 0.25F * (values[first] + values[first + 1U] +
                    values[first + layer] + values[first + layer + 1U]);
                const std::size_t upper_index = first + grid.dimensions.x;
                const float upper = 0.25F * (values[upper_index] +
                    values[upper_index + 1U] + values[upper_index + layer] +
                    values[upper_index + layer + 1U]);
                const float interpolated = lower + fraction * (upper - lower);
                below_cell_midpoints += interpolated < 0.0F;
            }
        std::cout << "Fluid scene midpoint water below terrain="
                  << below_cell_midpoints << '\n';
        if (wet_scene_cells < 100U || below_scene_floor != 0U ||
            below_interpolated_floor != 0U || below_cell_midpoints != 0U)
            throw 17;
    } catch (int failure) {
        std::cerr << "FAIL: fluid surface outlier regression " << failure << '\n';
        result = 1;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: fluid surface exception " << error.what() << '\n';
        result = 1;
    }
    cudaFree(device);
    return result;
}
