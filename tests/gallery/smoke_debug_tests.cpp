// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_debug.hpp>
#include <parallel_mater_gallery/overlay.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::gallery;

namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

template<class T> struct Buffer {
    T *data{};
    std::size_t size{};
    explicit Buffer(std::size_t count) : size(count) {
        require(cudaMallocManaged(reinterpret_cast<void **>(&data),
                                  count * sizeof(T)) == cudaSuccess,
                "allocate smoke debug fixture");
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    DeviceSpan<const T> span() const { return {data, size}; }
};
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        constexpr std::uint32_t n = 4U, h = 2U;
        constexpr std::size_t count = n * h * n;
        Buffer<Vec3> velocity(count), vorticity(count);
        Buffer<float> pressure(count), density(count), temperature(count),
            divergence(count);
        Buffer<std::uint32_t> solid(count);
        for (std::size_t cell = 0U; cell < count; ++cell) {
            const float x = float(cell % n);
            const float y = float(cell / n % h);
            const float z = float(cell / (n * h));
            velocity.data[cell] = {0.4F + x, y - 0.5F, z - 1.5F};
            vorticity.data[cell] = {z - 1.5F, x - 1.5F, 0.5F - y};
            pressure.data[cell] = x - 1.5F;
            density.data[cell] = 0.1F + 0.2F * x;
            temperature.data[cell] = density.data[cell] * (0.2F + y);
            divergence.data[cell] = y == 0.0F ? -0.25F : 0.25F;
            solid.data[cell] = cell == 13U ? 1U : 0U;
        }
        SmokeDeviceView view{
            .grid_velocity = velocity.span(),
            .grid_pressure = pressure.span(),
            .grid_density = density.span(),
            .grid_temperature = temperature.span(),
            .grid_solid = solid.span(),
            .grid_vorticity = vorticity.span(),
            .grid_divergence = divergence.span(),
            .grid_resolution = n,
            .grid_vertical_resolution = h,
            .grid_minimum = {},
            .grid_spacing = 1.0F,
        };
        constexpr std::uint32_t width = 640U, height = 480U;
        const Camera camera{.eye = {8, 6, 8}, .target = {2, 1, 2}};
        const std::array modes{SmokeDebugMode::grid, SmokeDebugMode::velocity,
            SmokeDebugMode::pressure, SmokeDebugMode::density_temperature,
            SmokeDebugMode::vorticity, SmokeDebugMode::divergence};
        std::vector<std::vector<std::uint32_t>> images;
        std::string error;
        for (const SmokeDebugMode mode : modes) {
            images.emplace_back(width * height, 0xff101010U);
            require(draw_smoke_grid_debug_overlay(
                        images.back(), width, height, view, camera, mode, error),
                    error.c_str());
            require(std::any_of(images.back().begin(), images.back().end(),
                                [](std::uint32_t pixel) {
                                    return pixel != 0xff101010U;
                                }),
                    "smoke debug mode drew no pixels");
        }
        for (std::size_t left = 0U; left < images.size(); ++left)
            for (std::size_t right = left + 1U; right < images.size(); ++right)
                require(images[left] != images[right],
                        "two smoke debug modes produced the same map");

        GalleryDebugState state{};
        state.toggle_smoke(SmokeDebugMode::pressure);
        require(state.smoke_mode == SmokeDebugMode::pressure,
                "smoke mode did not activate");
        state.toggle_smoke(SmokeDebugMode::pressure);
        require(state.smoke_mode == SmokeDebugMode::none,
                "pressing an active smoke mode did not hide it");

        SmokeDeviceView invalid = view;
        invalid.grid_temperature = {};
        std::vector<std::uint32_t> pixels(width * height, 0xff101010U);
        require(!draw_smoke_grid_debug_overlay(
                    pixels, width, height, invalid, camera,
                    SmokeDebugMode::density_temperature, error),
                "incomplete smoke grid debug view was accepted");
        std::cout << "Smoke Z/X/C/V/B/N grid maps passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
