// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gpu_renderer.hpp>
#include <parallel_mater_scene/command_scene.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gallery = parallel_mater::gallery;

namespace {

static_assert(static_cast<std::uint8_t>(gallery::GpuDrawKind::rigid_mesh) == 0U);
static_assert(static_cast<std::uint8_t>(
                  gallery::GpuDrawKind::deformable_surface) == 1U);
static_assert(static_cast<std::uint8_t>(gallery::GpuDrawKind::particles) == 2U);
static_assert(static_cast<std::uint8_t>(
                  gallery::GpuDrawKind::fluid_surface) == 3U);

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Handle>
Handle handle(std::uintptr_t value) {
    return reinterpret_cast<Handle>(value);
}

class MockGpuRenderer final : public gallery::GpuGalleryRenderer {
  public:
    explicit MockGpuRenderer(gallery::GpuRendererDevice device)
        : contract_(device) {}

    const gallery::GpuRendererDevice &renderer_device() const noexcept override {
        return contract_.device();
    }

    bool render(const gallery::GpuRenderRequest &request,
                std::string &error) override {
        if (!contract_.begin(request, error)) return false;
        if (hold_) return true;
        return contract_.finish(request.synchronization.frame_generation, error);
    }

    void hold(bool value) noexcept { hold_ = value; }
    bool complete(std::uint64_t frame, std::string &error) {
        return contract_.finish(frame, error);
    }

  private:
    gallery::GpuRendererContractState contract_;
    bool hold_{};
};

gallery::GpuRenderRequest request_for(
    const gallery::GpuSceneBuffers &views,
    const gallery::GpuRendererDevice &device, std::uint64_t frame) {
    gallery::GpuRenderRequest request;
    request.views = &views;
    request.output = {
        device.api_device,
        views.output,
        device.native,
        {31U, 0U, 4096U, 0U, 0U, 0U},
        views.resource_generation,
    };
    request.synchronization = {
        .physics_queue = device.physics_queue,
        .resource_generation = views.resource_generation,
        .frame_generation = frame,
    };
    request.target = {device.native.backend, device.native.device, 0U, 8U, 8U};
    return request;
}

void validate_interop_and_ownership() {
    gallery::GpuSceneBuffers views;
    views.device = handle<PmDevice>(1U);
    views.queue = handle<PmQueue>(2U);
    views.output = handle<PmBuffer>(3U);
    views.resource_generation = 7U;
    views.rigid = {0U, sizeof(PmRigidState)};
    views.previous_rigid = {sizeof(PmRigidState), sizeof(PmRigidState)};
    gallery::GpuRendererDevice device{
        views.device,
        views.queue,
        {PM_NATIVE_BACKEND_VULKAN, 11U, 12U, 13U},
        {14U, 2U, 0U},
        views.resource_generation,
    };
    MockGpuRenderer renderer(device);
    std::string error;

    auto request = request_for(views, device, 2U);
    require(renderer.render(request, error),
            "mock renderer rejected a valid public interop frame");
    require(renderer.render(request, error),
            "mock renderer rejected a safe paused-frame redraw");

    auto stale_frame = request_for(views, device, 1U);
    require(!renderer.render(stale_frame, error) &&
                error.find("stale frame") != std::string::npos,
            "mock renderer accepted stale frame synchronization");

    auto foreign = request_for(views, device, 3U);
    foreign.output.native_device.device = 99U;
    require(!renderer.render(foreign, error) &&
                error.find("foreign") != std::string::npos,
            "mock renderer accepted a foreign native buffer");

    auto foreign_target = request_for(views, device, 3U);
    foreign_target.target.device = 98U;
    require(!renderer.render(foreign_target, error) &&
                error.find("target") != std::string::npos,
            "mock renderer accepted a foreign render-target device");

    gallery::GpuSceneBuffers stale_views = views;
    stale_views.resource_generation = 6U;
    auto stale_buffer = request_for(stale_views, device, 3U);
    require(!renderer.render(stale_buffer, error) &&
                error.find("stale scene buffers") != std::string::npos,
            "mock renderer accepted a stale scene buffer generation");

    auto unsafe = request_for(views, device, 3U);
    unsafe.synchronization.acquire_from = gallery::GpuBufferOwner::graphics;
    require(!renderer.render(unsafe, error) &&
                error.find("unsafe ownership") != std::string::npos,
            "mock renderer accepted an unsafe ownership transition");

    renderer.hold(true);
    auto held = request_for(views, device, 3U);
    require(renderer.render(held, error),
            "mock renderer could not acquire a frame for overlap test");
    auto overlap = request_for(views, device, 4U);
    require(!renderer.render(overlap, error) &&
                error.find("already owned") != std::string::npos,
            "mock renderer allowed simultaneous graphics ownership");
    require(renderer.complete(3U, error),
            "mock renderer could not return graphics ownership");
}

void validate_draw_semantics() {
    gallery::SceneDefinition scene;
    scene.meshes.resize(3U);
    scene.rigid_bodies.push_back({.mesh_indices = {0U},
                                  .paint_resolution = 16U});
    scene.cloths.push_back({.mesh_index = 1U, .break_strain = 0.1F,
                            .paint_resolution = 32U});
    scene.soft_bodies.push_back({.mesh_index = 2U});
    scene.ropes.push_back({.options = {.radius = 0.04F}});
    scene.fluid_options.capacity = 16U;
    scene.fluid_options.particle_radius = 0.03F;
    scene.has_smoke = true;
    scene.smoke_options.particle_radius = 0.05F;

    gallery::CommandSceneInstance instance;
    instance.rigid_bodies = {0U};
    auto &views = instance.render_buffers;
    views.rigid = {0U, sizeof(PmRigidState)};
    views.previous_rigid = {64U, sizeof(PmRigidState)};
    views.cloth_surfaces = {{128U, 64U}};
    views.soft_surfaces = {{192U, 64U}};
    views.fluid = {256U, 128U};
    views.smoke = {384U, 128U};
    views.ropes = {{512U, 128U}};
    views.paint = {{640U, 128U}, {768U, 128U}};
    instance.paint_bindings = {{0U, 0U, 0U}, {PM_INVALID_INDEX, 1U, 1U}};

    gallery::GpuSceneDrawList list;
    std::string error;
    require(gallery::make_gpu_scene_draw_list(scene, instance, list, error),
            "common GPU draw-list construction failed");
    require(list.commands.size() == 6U,
            "common GPU draw list did not cover every current domain");
    const auto count_domain = [&](gallery::GpuOutputDomain domain) {
        return std::count_if(list.commands.begin(), list.commands.end(),
            [domain](const auto &command) { return command.domain == domain; });
    };
    for (const auto domain : {
             gallery::GpuOutputDomain::rigid,
             gallery::GpuOutputDomain::cloth,
             gallery::GpuOutputDomain::soft_body,
             gallery::GpuOutputDomain::fluid,
             gallery::GpuOutputDomain::smoke,
             gallery::GpuOutputDomain::rope}) {
        require(count_domain(domain) == 1,
                "common GPU draw list omitted or duplicated a domain");
    }
    const auto &rigid = list.commands.front();
    require(rigid.kind == gallery::GpuDrawKind::rigid_mesh &&
                rigid.previous_output.size == sizeof(PmRigidState) &&
                rigid.paint_output.size != 0U,
            "rigid draw semantics omitted interpolation or paint inputs");
    const auto &cloth = list.commands[1];
    require(cloth.kind == gallery::GpuDrawKind::deformable_surface &&
                cloth.output_has_header && cloth.use_triangle_corners &&
                cloth.paint_output.size != 0U,
            "cloth draw semantics omitted fracture or paint inputs");
}

} // namespace

int main() {
    validate_interop_and_ownership();
    validate_draw_semantics();
    return 0;
}
