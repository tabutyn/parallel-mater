// SPDX-License-Identifier: MIT
#include "fluid_surface.hpp"

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
constexpr int k_cell_bias = 1 << 20;

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

__device__ std::uint64_t cell_key(int x, int y, int z) {
    x = max(-k_cell_bias, min(k_cell_bias - 1, x));
    y = max(-k_cell_bias, min(k_cell_bias - 1, y));
    z = max(-k_cell_bias, min(k_cell_bias - 1, z));
    return (static_cast<std::uint64_t>(x + k_cell_bias) << 42U) |
           (static_cast<std::uint64_t>(y + k_cell_bias) << 21U) |
           static_cast<std::uint64_t>(z + k_cell_bias);
}

__global__ void index_particles(const Vec3 *positions, std::uint32_t count,
                                float inverse_support, std::uint64_t *keys,
                                std::uint32_t *indices) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Vec3 p = positions[i];
    keys[i] = cell_key(__float2int_rd(p.x * inverse_support),
                       __float2int_rd(p.y * inverse_support),
                       __float2int_rd(p.z * inverse_support));
    indices[i] = i;
}

__device__ std::uint32_t lower_bound(const std::uint64_t *keys,
                                     std::uint32_t count, std::uint64_t key) {
    std::uint32_t first = 0U;
    while (first < count) {
        const std::uint32_t middle = first + (count - first) / 2U;
        if (keys[middle] < key) first = middle + 1U;
        else count = middle;
    }
    return first;
}

__global__ void build_field(const Vec3 *positions, std::uint32_t count,
                            const std::uint64_t *keys,
                            const std::uint32_t *indices,
                            optix_shared::FluidSurfaceGrid grid,
                            std::uint32_t samples, float *values) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= samples) return;
    const std::uint32_t nx = grid.dimensions.x;
    const std::uint32_t ny = grid.dimensions.y;
    const Vec3 p{grid.minimum.x + (i % nx) * grid.cell_size.x,
                 grid.minimum.y + ((i / nx) % ny) * grid.cell_size.y,
                 grid.minimum.z + (i / (nx * ny)) * grid.cell_size.z};
    const float inverse = 1.0F / grid.support_radius;
    const int cx = __float2int_rd(p.x * inverse);
    const int cy = __float2int_rd(p.y * inverse);
    const int cz = __float2int_rd(p.z * inverse);
    const float radius_squared = grid.support_radius * grid.support_radius;
    float weight = 0.0F;
    float3 offset = make_float3(0.0F, 0.0F, 0.0F);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const std::uint64_t key = cell_key(cx + dx, cy + dy, cz + dz);
                for (std::uint32_t item = lower_bound(keys, count, key);
                     item < count && keys[item] == key; ++item) {
                    const Vec3 q = positions[indices[item]];
                    const float rx = p.x - q.x;
                    const float ry = p.y - q.y;
                    const float rz = p.z - q.z;
                    const float distance_squared = rx * rx + ry * ry + rz * rz;
                    if (distance_squared >= radius_squared) continue;
                    const float a = 1.0F - distance_squared / radius_squared;
                    const float w = a * a * a;
                    weight += w;
                    offset.x += w * rx;
                    offset.y += w * ry;
                    offset.z += w * rz;
                }
            }
        }
    }
    float value = grid.support_radius;
    if (weight > 1.0e-10F) {
        const float x = offset.x / weight;
        const float y = offset.y / weight;
        const float z = offset.z / weight;
        const float smooth_surface = sqrtf(x * x + y * y + z * z) -
            fminf(grid.support_radius / 3.0F,
                  grid.particle_radius * 0.8F);
        // The weighted-center field can bulge far below a particle layer.
        // Keep its lower boundary within one physical particle radius of
        // the local weighted particle center.
        value = fmaxf(smooth_surface, -y - grid.particle_radius * 0.9F);
    }
    values[i] = value;
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

float FluidSurface::update(FluidDeviceView particles,
                           const std::vector<Vec3> &host_positions) {
    impl_->grid = {};
    impl_->excluded_particle_count = 0U;
    if (particles.particle_count == 0U) return 0.0F;
    if (host_positions.size() != particles.particle_count ||
        !(particles.support_radius > 0.0F) ||
        !(particles.particle_radius > 0.0F))
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
    const float radius = particles.support_radius;
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
    grid.particle_radius = particles.particle_radius;
    for (Vec3 p : host_positions) {
        impl_->excluded_particle_count +=
            p.x < grid.minimum.x || p.x > high.x + radius ||
            p.y < grid.minimum.y || p.y > high.y + radius ||
            p.z < grid.minimum.z || p.z > high.z + radius;
    }
    // A thin sheet vanishes when its field is narrower than half a grid cell.
    // Use only the samples needed to resolve a physical particle, up to the
    // fixed budget; empty space should not make early frames expensive.
    float pitch = std::max(particles.particle_radius * 1.5F,
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
    index_particles<<<(particles.particle_count + block - 1U) / block, block>>>(
        particles.positions.data, particles.particle_count, 1.0F / radius,
        impl_->keys[0], impl_->indices[0]);
    check(cudaGetLastError(), "index fluid surface particles");
    check(cub::DeviceRadixSort::SortPairs(
              impl_->sort_workspace, impl_->sort_workspace_size,
              impl_->keys[0], impl_->keys[1], impl_->indices[0],
              impl_->indices[1], particles.particle_count, 0, 64),
          "sort fluid surface particles");
    build_field<<<(samples + block - 1U) / block, block>>>(
        particles.positions.data, particles.particle_count, impl_->keys[1],
        impl_->indices[1], grid, samples, impl_->values);
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
