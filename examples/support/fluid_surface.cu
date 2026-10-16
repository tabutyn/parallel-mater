// SPDX-License-Identifier: MIT
#include "fluid_surface.hpp"
#include "fluid_surface_generated.cu"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace parallel_mater::gallery {
namespace {

// Budget for the late scene, where the stream spans almost the whole level.
constexpr std::uint32_t k_samples = 128U * 128U * 128U;

void check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

template <class T> T *allocate(std::size_t count) {
    T *result = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&result), count * sizeof(T)),
          "allocate fluid surface workspace");
    return result;
}


} // namespace

struct FluidSurface::Impl {
    std::uint64_t *keys[2]{};
    std::uint32_t *indices[2]{};
    void *sort_workspace{};
    std::size_t sort_workspace_size{};
    float *values{};
    cudaEvent_t begin{};
    cudaEvent_t end{};
    optix_shared::FluidSurfaceGrid grid{};
    std::uint32_t excluded_particle_count{};

    explicit Impl(std::uint32_t capacity) {
        try {
            for (int i = 0; i < 2; ++i) {
                keys[i] = allocate<std::uint64_t>(capacity);
                indices[i] = allocate<std::uint32_t>(capacity);
            }
            values = allocate<float>(k_samples);
            check(cub::DeviceRadixSort::SortPairs(
                      nullptr, sort_workspace_size, keys[0], keys[1],
                      indices[0], indices[1], capacity, 0, 64),
                  "size fluid surface sort");
            sort_workspace = allocate<std::uint8_t>(sort_workspace_size);
            check(cudaEventCreate(&begin), "create surface timing begin");
            check(cudaEventCreate(&end), "create surface timing end");
        } catch (...) {
            release();
            throw;
        }
    }
    ~Impl() { release(); }

    void release() noexcept {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        cudaFree(sort_workspace);
        cudaFree(values);
        for (int i = 0; i < 2; ++i) {
            cudaFree(keys[i]);
            cudaFree(indices[i]);
        }
    }
};

FluidSurface::FluidSurface(std::uint32_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}
FluidSurface::~FluidSurface() = default;
FluidSurface::FluidSurface(FluidSurface &&) noexcept = default;
FluidSurface &FluidSurface::operator=(FluidSurface &&) noexcept = default;

float FluidSurface::update(const Vec3 *device_positions,
                           std::uint32_t particle_count,
                           float particle_radius, float support_radius,
                           const std::vector<Vec3> &host_positions) {
    impl_->grid = {};
    impl_->excluded_particle_count = 0U;
    if (particle_count == 0U) return 0.0F;
    if (device_positions == nullptr || host_positions.size() != particle_count ||
        !(support_radius > 0.0F) || !(particle_radius > 0.0F))
        throw std::invalid_argument("invalid fluid surface input");
    Vec3 low = host_positions.front();
    Vec3 high = low;
    for (Vec3 p : host_positions) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::runtime_error("fluid surface has nonfinite particle");
        low.x = std::min(low.x, p.x); low.y = std::min(low.y, p.y);
        low.z = std::min(low.z, p.z);
        high.x = std::max(high.x, p.x); high.y = std::max(high.y, p.y);
        high.z = std::max(high.z, p.z);
    }
    // Reconstruct with the solver's neighbor reach; widening it bridges
    // separate streams and extrudes the surface beneath the passive mesh.
    const float radius = support_radius;
    // A few escaped particles can travel dozens of meters below the stream.
    // Using their absolute extrema spends almost the entire fixed voxel budget
    // on empty space, erasing the water where the bulk of particles lives.
    // Trim an axis only when its extreme is well separated from the 0.5% tail.
    if (host_positions.size() >= 200U) {
        std::vector<float> coordinates(host_positions.size());
        const std::size_t trim = host_positions.size() / 200U;
        for (int axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < host_positions.size(); ++i) {
                const Vec3 p = host_positions[i];
                coordinates[i] = axis == 0 ? p.x : axis == 1 ? p.y : p.z;
            }
            auto lower = coordinates.begin() + trim;
            auto upper = coordinates.end() - trim - 1;
            std::nth_element(coordinates.begin(), lower, coordinates.end());
            const float lower_quantile = *lower;
            std::nth_element(lower + 1, upper, coordinates.end());
            const float upper_quantile = *upper;
            const float core_extent = upper_quantile - lower_quantile;
            const float outlier_gap = std::max(4.0F * radius,
                                               0.25F * core_extent);
            float *minimum = axis == 0 ? &low.x : axis == 1 ? &low.y : &low.z;
            float *maximum = axis == 0 ? &high.x : axis == 1 ? &high.y : &high.z;
            if (*minimum < lower_quantile - outlier_gap)
                *minimum = lower_quantile - radius;
            if (*maximum > upper_quantile + outlier_gap)
                *maximum = upper_quantile + radius;
        }
    }
    const float3 extent = make_float3(high.x - low.x + 2.0F * radius,
                                      high.y - low.y + 2.0F * radius,
                                      high.z - low.z + 2.0F * radius);
    auto &grid = impl_->grid;
    grid.minimum = make_float3(low.x - radius, low.y - radius, low.z - radius);
    grid.support_radius = radius;
    grid.particle_radius = particle_radius;
    for (Vec3 p : host_positions) {
        impl_->excluded_particle_count +=
            p.x < grid.minimum.x || p.x > high.x + radius ||
            p.y < grid.minimum.y || p.y > high.y + radius ||
            p.z < grid.minimum.z || p.z > high.z + radius;
    }
    // A thin sheet vanishes when its field is narrower than half a grid cell.
    // Use only the samples needed to resolve a physical particle, up to the
    // fixed budget; empty space should not make early frames expensive.
    float pitch = std::max(particle_radius * 1.5F,
        std::cbrt(extent.x * extent.y * extent.z / static_cast<float>(k_samples)));
    for (;;) {
        const auto dimension = [pitch](float length) {
            return std::min(256U, std::max(2U,
                static_cast<std::uint32_t>(std::ceil(length / pitch)) + 1U));
        };
        grid.dimensions = make_uint3(dimension(extent.x), dimension(extent.y),
                                     dimension(extent.z));
        if (static_cast<std::uint64_t>(grid.dimensions.x) * grid.dimensions.y *
                grid.dimensions.z <= k_samples) break;
        pitch *= 1.025F;
    }
    grid.cell_size = make_float3(
        extent.x / static_cast<float>(grid.dimensions.x - 1U),
        extent.y / static_cast<float>(grid.dimensions.y - 1U),
        extent.z / static_cast<float>(grid.dimensions.z - 1U));
    const std::uint32_t samples =
        grid.dimensions.x * grid.dimensions.y * grid.dimensions.z;
    check(cudaEventRecord(impl_->begin), "begin fluid surface timing");
    constexpr std::uint32_t block = 128U;
    surface_index_particles<<<(particle_count + block - 1U) / block, block>>>(
        reinterpret_cast<float3 *>(const_cast<Vec3 *>(device_positions)), particle_count, 1.0F / radius,
        reinterpret_cast<ulonglong *>(impl_->keys[0]), impl_->indices[0]);
    check(cudaGetLastError(), "index fluid surface particles");
    check(cub::DeviceRadixSort::SortPairs(
              impl_->sort_workspace, impl_->sort_workspace_size,
              impl_->keys[0], impl_->keys[1], impl_->indices[0],
              impl_->indices[1], particle_count, 0, 64),
          "sort fluid surface particles");
    const SurfaceGrid_0 shared_grid{grid.minimum, grid.cell_size, grid.dimensions,
        grid.support_radius, grid.particle_radius};
    surface_build_field<<<(samples + block - 1U) / block, block>>>(
        reinterpret_cast<float3 *>(const_cast<Vec3 *>(device_positions)), particle_count,
        reinterpret_cast<ulonglong *>(impl_->keys[1]), impl_->indices[1],
        shared_grid, samples, impl_->values);
    check(cudaGetLastError(), "build fluid surface field");
    check(cudaEventRecord(impl_->end), "end fluid surface timing");
    check(cudaEventSynchronize(impl_->end), "complete fluid surface field");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, impl_->begin, impl_->end),
          "measure fluid surface field");
    return milliseconds;
}

optix_shared::FluidSurfaceView FluidSurface::view() const noexcept {
    return impl_ ? optix_shared::FluidSurfaceView{impl_->values, impl_->grid}
                 : optix_shared::FluidSurfaceView{};
}

std::uint32_t FluidSurface::excluded_particle_count() const noexcept {
    return impl_ ? impl_->excluded_particle_count : 0U;
}

} // namespace parallel_mater::gallery
