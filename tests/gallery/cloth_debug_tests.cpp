// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/overlay.hpp>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::gallery;

namespace {
void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

template<class T> struct Buffer {
    T *data{};
    std::size_t size{};
    Buffer(std::initializer_list<T> values) : size(values.size()) {
        check(cudaMallocManaged(reinterpret_cast<void **>(&data), size * sizeof(T)) ==
              cudaSuccess, "allocate fixture");
        std::copy(values.begin(), values.end(), data);
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    DeviceSpan<const T> span() const { return {data, size}; }
};

constexpr std::uint32_t width = 640U, height = 480U;
const Camera camera{.eye = {0, 2, 6}, .target = {}};

std::vector<std::uint32_t> draw(ClothDeviceView cloth, ClothDebugOptions options) {
    std::vector<std::uint32_t> pixels(width * height, 0xff101010U);
    std::string error;
    const bool okay = draw_cloth_debug_overlay(pixels, width, height, cloth,
                                               camera, options, error);
    if (!okay) throw std::runtime_error(error);
    return pixels;
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        // Original shared connectivity stretches across the gap. The API's
        // triangle-local fracture surface keeps two separate, compact faces.
        Buffer<Vec3> nodes{{-1.5F, -0.4F, 0}, {-1.5F, 0.4F, 0},
                           {1.5F, -0.4F, 0}, {1.5F, 0.4F, 0}};
        Buffer<std::uint32_t> indices{0, 1, 2, 1, 3, 2};
        Buffer<Vec3> surface{{-1.5F, -0.4F, 0}, {-1.5F, 0.4F, 0}, {-1, -0.4F, 0},
                             {1, 0.4F, 0}, {1.5F, 0.4F, 0}, {1.5F, -0.4F, 0}};
        Buffer<std::uint32_t> surface_indices{0, 1, 2, 3, 4, 5};
        ClothDeviceView original{.positions = nodes.span(),
            .triangle_indices = indices.span(), .vertex_count = 4U};
        ClothDeviceView torn = original;
        torn.surface_positions = surface.span();
        torn.surface_triangle_indices = surface_indices.span();
        ClothDeviceView reference{.positions = surface.span(),
            .triangle_indices = surface_indices.span(), .vertex_count = 6U};

        check(draw(original, {.wireframe = true}) != draw(reference, {.wireframe = true}),
              "wireframe fixture must expose stale connections");
        check(draw(torn, {.wireframe = true}) == draw(reference, {.wireframe = true}),
              "torn wireframe still connects detached triangles");
        check(draw(original, {.normals = true}) != draw(reference, {.normals = true}),
              "normal fixture must distinguish nodes from surface corners");
        check(draw(torn, {.normals = true}) == draw(reference, {.normals = true}),
              "torn normals must use triangle-local surface positions");

        Buffer<ClothBond> bonds{{0, 1, 0.8F}, {1, 2, 3.1F}};
        Buffer<std::uint8_t> active{1U, 0U};
        Buffer<ClothBond> surviving{{0, 1, 0.8F}};
        Buffer<std::uint8_t> surviving_active{1U};
        torn.bonds = bonds.span();
        torn.active_bonds = active.span();
        ClothDeviceView live_only = original;
        live_only.bonds = surviving.span();
        live_only.active_bonds = surviving_active.span();
        check(draw(torn, {.bonds = true}) == draw(live_only, {.bonds = true}),
              "broken bond is still drawn between detached vertices");

        // Reactions belong to physical nodes, not the duplicated render
        // corners. Switching wireframe topology must not move force arrows.
        Buffer<Vec3> forces{{5, 5, 0}, {5, 5, 0}, {5, 5, 0}, {5, 5, 0}};
        torn.rigid_contact_forces = forces.span();
        original.rigid_contact_forces = forces.span();
        check(draw(torn, {.rigid_contact_forces = true}) ==
              draw(original, {.rigid_contact_forces = true}),
              "force arrows moved off physical nodes");

        std::vector<std::uint32_t> pixels(width * height);
        std::string error;
        torn.surface_triangle_indices.size = 1U;
        check(!draw_cloth_debug_overlay(pixels, width, height, torn, camera,
                                       {.wireframe = true}, error),
              "invalid surface index count was accepted");
        torn.surface_triangle_indices = surface_indices.span();
        surface_indices.data[5] = 6U;
        check(!draw_cloth_debug_overlay(pixels, width, height, torn, camera,
                                       {.normals = true}, error),
              "out-of-range surface index was accepted");
        torn.surface_positions = {};
        check(!draw_cloth_debug_overlay(pixels, width, height, torn, camera,
                                       {.wireframe = true}, error),
              "surface indices without positions were accepted");
        std::cout << "Cloth fracture wireframe, normals, bonds and force origins passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
