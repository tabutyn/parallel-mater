// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gpu_renderer.hpp>
#include <parallel_mater_scene/command_scene.hpp>

#include <algorithm>
#include <limits>
#include <new>

namespace parallel_mater::gallery {
namespace {

[[nodiscard]] bool same_native_device(const PmNativeDeviceInfo &left,
                                      const PmNativeDeviceInfo &right) noexcept {
    return left.backend == right.backend && left.instance == right.instance &&
        left.physicalDevice == right.physicalDevice && left.device == right.device;
}

[[nodiscard]] bool valid_region(GpuSceneBuffers::Region region,
                                PmDeviceSize size) noexcept {
    return region.offset <= size && region.size <= size - region.offset;
}

[[nodiscard]] Vec3 color(float red, float green, float blue) noexcept {
    return {red, green, blue};
}

} // namespace

bool resolve_gpu_renderer_device(const GpuSceneBuffers &buffers,
                                 GpuRendererDevice &output,
                                 std::string &error) {
    error.clear();
    output = {};
    if (buffers.device == nullptr || buffers.queue == nullptr ||
        buffers.output == nullptr || buffers.resource_generation == 0U) {
        error = "GPU renderer requires live public scene resources";
        return false;
    }
    PmNativeDeviceInfo native{};
    if (pmGetNativeDevice(buffers.device, &native) != PM_SUCCESS ||
        native.device == 0U) {
        error = "GPU renderer could not export the public native device";
        return false;
    }
    PmNativeQueueInfo native_queue{};
    if (pmGetNativeQueue(buffers.queue, &native_queue) != PM_SUCCESS ||
        native_queue.queue == 0U) {
        error = "GPU renderer could not export the public native queue";
        return false;
    }
    output = {buffers.device, buffers.queue, native, native_queue,
              buffers.resource_generation};
    return true;
}

bool make_gpu_render_request(
    const GpuSceneBuffers &buffers, std::uint64_t frame_generation,
    Camera camera, GpuRenderTarget target, GpuDiagnosticReadback readback,
    FrameTimingCollector *timing, std::span<const std::uint32_t> overlay,
    GpuRenderRequest &output, std::string &error) {
    error.clear();
    output = {};
    GpuRendererDevice device;
    if (!resolve_gpu_renderer_device(buffers, device, error)) return false;
    PmNativeBufferInfo native_buffer{};
    if (pmGetNativeBuffer(buffers.device, buffers.output, &native_buffer) !=
            PM_SUCCESS || native_buffer.buffer == 0U) {
        error = "GPU renderer could not export the public output buffer";
        return false;
    }
    if (frame_generation == 0U) {
        error = "GPU renderer requires a completed physics generation";
        return false;
    }
    if (target.backend == static_cast<PmNativeBackend>(0))
        target.backend = device.native.backend;
    if (target.device == 0U) target.device = device.native.device;
    output.views = &buffers;
    output.output = {buffers.device, buffers.output, device.native,
                     native_buffer, buffers.resource_generation};
    output.synchronization = {
        .mode = GpuSynchronizationMode::queue_idle,
        .physics_queue = buffers.queue,
        .resource_generation = buffers.resource_generation,
        .frame_generation = frame_generation,
    };
    output.target = target;
    output.readback = readback;
    output.camera = camera;
    output.timing = timing;
    output.overlay = overlay;
    return true;
}

bool make_gpu_scene_draw_list(const SceneDefinition &scene,
                              const CommandSceneInstance &instance,
                              GpuSceneDrawList &output,
                              std::string &error) {
    error.clear();
    output.commands.clear();
    const auto &buffers = instance.render_buffers;
    if (scene.rigid_bodies.size() != instance.rigid_bodies.size() ||
        scene.cloths.size() != buffers.cloth_surfaces.size() ||
        scene.soft_bodies.size() != buffers.soft_surfaces.size() ||
        scene.ropes.size() != buffers.ropes.size()) {
        error = "GPU draw list does not match public scene output domains";
        return false;
    }
    try {
        const auto paint = [&](std::uint32_t mesh, std::uint32_t body,
                               GpuDrawCommand &command) {
            const auto binding = std::find_if(
                instance.paint_bindings.begin(), instance.paint_bindings.end(),
                [&](const CommandSceneInstance::PaintBinding &candidate) {
                    return candidate.mesh_index == mesh &&
                        (command.domain != GpuOutputDomain::rigid ||
                         candidate.body_index == body);
                });
            if (binding == instance.paint_bindings.end()) return;
            if (binding->field_index >= buffers.paint.size()) {
                error = "GPU draw paint binding is outside public outputs";
                return;
            }
            command.paint_output = buffers.paint[binding->field_index];
        };
        for (std::uint32_t body = 0U; body < scene.rigid_bodies.size(); ++body) {
            const auto offset = static_cast<PmDeviceSize>(body) * sizeof(PmRigidState);
            if (offset > buffers.rigid.size ||
                sizeof(PmRigidState) > buffers.rigid.size - offset ||
                offset > buffers.previous_rigid.size ||
                sizeof(PmRigidState) > buffers.previous_rigid.size - offset) {
                error = "GPU draw list is missing rigid state output";
                return false;
            }
            for (const std::uint32_t mesh : scene.rigid_bodies[body].mesh_indices) {
                if (mesh >= scene.meshes.size()) {
                    error = "GPU rigid draw references an unknown mesh";
                    return false;
                }
                GpuDrawCommand command{
                    .kind = GpuDrawKind::rigid_mesh,
                    .domain = GpuOutputDomain::rigid,
                    .output = {buffers.rigid.offset + offset, sizeof(PmRigidState)},
                    .previous_output = {buffers.previous_rigid.offset + offset,
                                        sizeof(PmRigidState)},
                    .output_stride = sizeof(PmRigidState),
                    .domain_index = body,
                    .mesh_index = mesh,
                    .paint_resolution = scene.rigid_bodies[body].paint_resolution,
                    .color = scene.meshes[mesh].base_color,
                };
                paint(mesh, body, command);
                if (!error.empty()) return false;
                output.commands.push_back(command);
            }
        }
        for (std::uint32_t index = 0U; index < scene.cloths.size(); ++index) {
            const auto &cloth = scene.cloths[index];
            if (cloth.mesh_index >= scene.meshes.size()) {
                error = "GPU cloth draw references an unknown mesh";
                return false;
            }
            GpuDrawCommand command{
                .kind = GpuDrawKind::deformable_surface,
                .domain = GpuOutputDomain::cloth,
                .output = buffers.cloth_surfaces[index],
                .output_stride = sizeof(PmFloat3),
                .domain_index = index,
                .mesh_index = cloth.mesh_index,
                .paint_resolution = cloth.paint_resolution,
                .color = scene.meshes[cloth.mesh_index].base_color,
                .output_has_header = true,
                .use_triangle_corners = cloth.break_strain > 0.0F ||
                    cloth.impact_break_impulse > 0.0F,
            };
            paint(cloth.mesh_index, 0U, command);
            if (!error.empty()) return false;
            output.commands.push_back(command);
        }
        for (std::uint32_t index = 0U; index < scene.soft_bodies.size(); ++index) {
            const auto mesh = scene.soft_bodies[index].mesh_index;
            if (mesh >= scene.meshes.size()) {
                error = "GPU soft-body draw references an unknown mesh";
                return false;
            }
            output.commands.push_back({
                .kind = GpuDrawKind::deformable_surface,
                .domain = GpuOutputDomain::soft_body,
                .output = buffers.soft_surfaces[index],
                .output_stride = sizeof(PmFloat3),
                .domain_index = index,
                .mesh_index = mesh,
                .color = scene.meshes[mesh].base_color,
                .output_has_header = true,
            });
        }
        if (buffers.fluid.size != 0U) {
            output.commands.push_back({
                .kind = GpuDrawKind::particles,
                .domain = GpuOutputDomain::fluid,
                .output = buffers.fluid,
                .output_stride = sizeof(PmFluidParticleOutput),
                .color = color(0.05F, 0.42F, 0.92F),
                .radius = std::max(scene.fluid_options.particle_radius, 0.02F),
                .output_has_header = true,
            });
        }
        if (buffers.smoke.size != 0U) {
            output.commands.push_back({
                .kind = GpuDrawKind::particles,
                .domain = GpuOutputDomain::smoke,
                .output = buffers.smoke,
                .output_stride = sizeof(PmSmokeParticleState),
                .color = color(0.58F, 0.61F, 0.65F),
                .radius = std::max(scene.smoke_options.particle_radius, 0.03F),
                .output_has_header = true,
            });
        }
        for (std::uint32_t index = 0U; index < scene.ropes.size(); ++index) {
            output.commands.push_back({
                .kind = GpuDrawKind::particles,
                .domain = GpuOutputDomain::rope,
                .output = buffers.ropes[index],
                .output_stride = sizeof(PmRopeNodeState),
                .domain_index = index,
                .color = color(0.85F, 0.62F, 0.18F),
                .radius = std::max(scene.ropes[index].options.radius, 0.025F),
                .output_has_header = true,
            });
        }
    } catch (const std::bad_alloc &) {
        output.commands.clear();
        error = "GPU draw list allocation failed";
        return false;
    }
    return true;
}

bool GpuRendererContractState::begin(const GpuRenderRequest &request,
                                     std::string &error) {
    error.clear();
    if (graphics_owns_buffer_) {
        error = "GPU output is already owned by graphics";
        return false;
    }
    if (!request.views || request.views->device != device_.api_device ||
        request.views->queue != device_.physics_queue ||
        request.output.api_device != device_.api_device ||
        request.output.api_buffer != request.views->output ||
        !same_native_device(request.output.native_device, device_.native)) {
        error = "GPU render request contains a foreign device or buffer";
        return false;
    }
    if (request.output.resource_generation != device_.resource_generation ||
        request.views->resource_generation != device_.resource_generation ||
        request.synchronization.resource_generation != device_.resource_generation) {
        error = "GPU render request contains stale scene buffers";
        return false;
    }
    if (request.output.native.buffer == 0U ||
        request.output.native.offset >
            std::numeric_limits<PmDeviceSize>::max() - request.output.native.size) {
        error = "GPU render request contains an invalid native buffer";
        return false;
    }
    const auto size = request.output.native.size;
    const auto valid = [&](GpuSceneBuffers::Region region) {
        return valid_region(region, size);
    };
    if (!valid(request.views->rigid) ||
        !valid(request.views->previous_rigid) ||
        !valid(request.views->fluid) || !valid(request.views->smoke) ||
        !std::all_of(request.views->cloth_surfaces.begin(),
                     request.views->cloth_surfaces.end(), valid) ||
        !std::all_of(request.views->soft_surfaces.begin(),
                     request.views->soft_surfaces.end(), valid) ||
        !std::all_of(request.views->ropes.begin(), request.views->ropes.end(),
                     valid) ||
        !std::all_of(request.views->paint.begin(), request.views->paint.end(),
                     valid)) {
        error = "GPU render request contains an out-of-bounds output region";
        return false;
    }
    const auto &sync = request.synchronization;
    if (sync.physics_queue != device_.physics_queue ||
        sync.frame_generation == 0U || sync.frame_generation < newest_frame_) {
        error = "GPU render request contains stale frame synchronization";
        return false;
    }
    if (sync.acquire_from != GpuBufferOwner::physics ||
        sync.release_to != GpuBufferOwner::physics) {
        error = "GPU render request has an unsafe ownership transition";
        return false;
    }
    if (sync.mode == GpuSynchronizationMode::timeline_semaphore &&
        (sync.wait_semaphore == 0U || sync.signal_semaphore == 0U ||
         sync.signal_value <= sync.wait_value)) {
        error = "GPU render request has an invalid timeline handoff";
        return false;
    }
    if (request.target.backend != device_.native.backend ||
        request.target.device != device_.native.device ||
        request.target.width == 0U || request.target.height == 0U) {
        error = "GPU render target does not match the renderer device";
        return false;
    }
    const auto pixel_count = static_cast<std::uint64_t>(request.target.width) *
        request.target.height;
    if (!request.overlay.empty() && request.overlay.size() != pixel_count) {
        error = "GPU overlay dimensions do not match the render target";
        return false;
    }
    graphics_owns_buffer_ = true;
    active_frame_ = sync.frame_generation;
    return true;
}

bool GpuRendererContractState::finish(std::uint64_t frame_generation,
                                      std::string &error) {
    error.clear();
    if (!graphics_owns_buffer_ || active_frame_ != frame_generation) {
        error = "GPU renderer released output it does not own";
        return false;
    }
    newest_frame_ = std::max(newest_frame_, frame_generation);
    active_frame_ = 0U;
    graphics_owns_buffer_ = false;
    return true;
}

void GpuRendererContractState::abandon(
    std::uint64_t frame_generation) noexcept {
    if (graphics_owns_buffer_ && active_frame_ == frame_generation) {
        active_frame_ = 0U;
        graphics_owns_buffer_ = false;
    }
}

} // namespace parallel_mater::gallery
