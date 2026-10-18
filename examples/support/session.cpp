// SPDX-License-Identifier: MIT
// Mobile behavior follows Android: bounded work, shared budgets, touch camera.
#include <parallel_mater_gallery/session.hpp>
#include "scene_builder.hpp"
#include "render/render_scene.hpp"
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/runtime_platform.hpp>
#ifdef PM_GALLERY_NATIVE_GPU_RENDERER
#include <parallel_mater_gallery/gpu_renderer.hpp>
#endif
#include <parallel_mater_gallery/gallery_context.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace parallel_mater::gallery {
namespace {
void require(bool ok, const std::string &error) {
    if (!ok) throw std::runtime_error(error.empty() ? "Gallery operation failed" : error);
}

std::optional<PmNativeBackend> required_backend(std::string_view name) {
    const auto backend = gallery_backend_from_name(name);
    if (!backend && name != "slang-cpu")
        throw std::runtime_error("Unknown Gallery backend: " + std::string(name));
    return backend;
}
}
const GalleryCatalogItem &GallerySession::catalog_item(unsigned index) const {
    if (index >= catalog_.size()) throw std::runtime_error("Unknown scene");
    return catalog_[index];
}
GallerySession::GallerySession(GallerySessionConfig config) : config_(std::move(config)) {
#ifndef PM_GALLERY_NATIVE_GPU_RENDERER
    if (config_.gpu_rendering) throw std::runtime_error("Native GPU Gallery renderer is not built");
#endif
    if (config_.workload_limits) {
        limits = *config_.workload_limits;
    } else {
        std::string error;
        DeviceProfileCatalog catalog;
        require(load_device_profiles(config_.profiles, catalog, error), error);
        const auto *profile = select_workload_limits(
            catalog, config_.platform, config_.model, config_.backend);
        if (!profile) throw std::runtime_error("Gallery workload profile missing");
        limits = *profile;
    }
    frame_policy_.apply_workload(&limits);
    parameters_.bricks = config_.bricks.value_or(
        BrickSceneConfig{limits.brick_count, 1.0F, 1U});
    parameters_.dump_spheres = config_.dump_spheres != 0U
        ? config_.dump_spheres : limits.dump_spheres;
    parameters_.fluid_particles = config_.fluid_particles != 0U
        ? config_.fluid_particles : limits.fluid_particles;
    width = limits.render_width;
    height = limits.render_height;
    select(config_.initial_scene);
}
GallerySession::~GallerySession() = default;
void GallerySession::require_scene() const {
    if (!runtime_.ready())
        throw std::runtime_error("Choose a scene or Reset to continue");
}
void GallerySession::select(unsigned index) {
    const auto &item = catalog_item(index);
    if (!item.available)
        throw std::runtime_error(std::string(item.scene.name) + ": " +
                                 item.unavailable_reason);
    const auto &selected = item.scene;
    std::string error;
    GalleryRuntimeCreateInfo request;
    request.context = selected.context;
    request.asset = selected.context == GalleryContext::rigid_body &&
            !config_.rigid_body_asset.empty()
        ? config_.rigid_body_asset : config_.assets / item.asset_file;
    request.backend = config_.native_backend
        ? config_.native_backend : required_backend(config_.backend);
    request.debug_frame_capacity = selected.context ==
            GalleryContext::constraint_fixed
        ? 300U : 30U;
    request.width = width;
    request.height = height;
    request.create_host_renderer = config_.create_host_renderer.value_or(
        !config_.gpu_rendering);
    request.replacement = config_.replacement;
    const GallerySceneParameters parameters = parameters_;
    const GalleryWorkloadLimits selected_limits = limits;
    struct PendingDevice {
        std::optional<GalleryCatalog> catalog;
        std::string name;
    };
    const auto pending_device = std::make_shared<PendingDevice>();
    request.configure_scene =
        [context = selected.context, has_fluid = selected.has_fluid,
         parameters, selected_limits,
         validate = config_.validate_scene](SceneDefinition &scene,
                                            std::string &failure) {
        if (context == GalleryContext::rigid_body) {
            SceneDefinition generated;
            if (!make_brick_scene(scene, parameters.bricks,
                                  generated, failure)) return false;
            scene = std::move(generated);
        }
        if (has_fluid)
            scene.fluid_options.capacity = parameters.fluid_particles;
        if ((context == GalleryContext::fluid ||
             context == GalleryContext::fluid_rigid)) {
            auto applied_limits = selected_limits;
            applied_limits.fluid_particles = parameters.fluid_particles;
            if (!apply_liquid_workload_limits(
                    scene, applied_limits, failure)) return false;
        }
        if (context == GalleryContext::dump &&
            !configure_dump_payload(
                scene, parameters.dump_spheres, failure)) return false;
        scene.smoke_options.capacity = std::min(
            scene.smoke_options.capacity, selected_limits.smoke_particles);
        return !validate || validate(context, scene, failure);
    };
    const bool gpu_rendering = config_.gpu_rendering;
    const auto native_backend = request.backend;
    const bool prime_scene = config_.prime_scene;
    request.prepare_runtime =
        [pending_device, gpu_rendering, native_backend, prime_scene]
        (GalleryRuntime &next, std::string &failure) {
        PmPhysicalDeviceProperties device{};
        pmGetPhysicalDeviceProperties(next.simulation.physical_device(), &device);
        PmPhysicalDeviceFeatures features{};
        pmGetPhysicalDeviceFeatures(next.simulation.physical_device(), &features);
        pending_device->catalog = make_gallery_catalog(&features);
        if (native_backend) {
            PmNativeDeviceInfo native{};
            if (pmGetNativeDevice(next.simulation.device(), &native) != PM_SUCCESS ||
                native.backend != *native_backend) {
                failure =
                    "Requested GPU backend required; CPU fallback is not permitted";
                return false;
            }
        }
#ifdef PM_GALLERY_NATIVE_GPU_RENDERER
        if (gpu_rendering) {
            GpuRendererDevice device;
            if (!resolve_gpu_renderer_device(next.instance.render_buffers,
                                             device, failure)) return false;
            next.native_renderer = create_gpu_gallery_renderer(
                device, next.scene, next.instance, failure);
            if (!next.native_renderer) return false;
        }
#else
        (void)gpu_rendering;
#endif
        if (prime_scene && !next.step({.gravity = {0.0F,
                -9.81F * next.scene.gravity_scale, 0.0F}}, failure)) {
            return false;
        }
        pending_device->name = device.deviceName;
        return true;
    };
    const unsigned previous_selection = selected_;
    if (!runtime_.replace(std::move(request), error)) {
        // Preflight failure preserves the old scene; a later failure under the
        // mobile release-first policy leaves the requested scene retryable.
        selected_ = runtime_.ready() ? previous_selection : index;
        if (!runtime_.ready()) {
            frames = 0;
            pixels.clear();
            timing.clear_history();
            frame_policy_.reset();
            right_ = up_ = 0;
            released_ = false;
        }
        require(false, error);
    }
    if (!pending_device->catalog)
        throw std::runtime_error("Gallery backend did not report scene capabilities");
    catalog_ = *pending_device->catalog;
    device_ = pending_device->name;
    selected_ = index;
    frames = 0;
    pixels.clear();
    timing.clear_history();
    frame_policy_.reset();
    right_ = up_ = 0;
    released_ = false;
    ++revision_;
    focus();
}
void GallerySession::reset() {
    std::string error;
    if (!runtime_.reset(error)) {
        if (!runtime_.ready()) {
            frames = 0;
            pixels.clear();
            timing.clear_history();
            frame_policy_.reset();
            right_ = up_ = 0;
            released_ = false;
        }
        require(false, error);
    }
    frames = 0;
    pixels.clear();
    timing.clear_history();
    frame_policy_.reset();
    right_ = up_ = 0;
    released_ = false;
    ++revision_;
    focus();
}
void GallerySession::resize(unsigned w, unsigned h) {
    const float scale = std::min({1.0F, float(limits.render_width) / std::max(w, 1U),
        float(limits.render_height) / std::max(h, 1U)});
    w = std::max(1U, unsigned(w * scale)); h = std::max(1U, unsigned(h * scale));
    if (w == width && h == height) return;
    std::string error;
    RendererHandle renderer;
    if (runtime_.ready() && !config_.gpu_rendering) {
        require(create_renderer(runtime_.scene, runtime_.instance, w, h,
                                renderer, error), error);
        runtime_.renderer = std::move(renderer);
    }
    width = w; height = h;
    timing.clear_history();
}
void GallerySession::set_render_target(GpuRenderTarget target) noexcept {
    const bool replacing = render_target_.backend != target.backend ||
        render_target_.device != target.device ||
        render_target_.handle != target.handle;
    if (replacing && runtime_.native_renderer)
        runtime_.native_renderer->detach_target();
    render_target_ = target;
}
void GallerySession::detach_render_target() noexcept {
    set_render_target({});
}
void GallerySession::orbit(float x, float y) {
    require_scene();
    runtime_.camera.begin_drag(CameraDragMode::orbit, 0, 0);
    runtime_.camera.move_cursor(x, y, height); runtime_.camera.end_drag();
}
void GallerySession::zoom(float amount) {
    require_scene();
    runtime_.camera.zoom(amount);
}
void GallerySession::focus() {
    require_scene();
    runtime_.camera = CameraController{};
    auto preset = entry(selected_).context == GalleryContext::rigid_body
        ? brick_camera_preset(parameters_.bricks) : entry(selected_).camera;
    if (entry(selected_).context == GalleryContext::constraint_hinge) preset.distance_scale *= 1.6F;
    if (entry(selected_).context == GalleryContext::fluid || entry(selected_).context == GalleryContext::fluid_rigid) {
        Vec3 low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
        const auto include = [&](float x, float y, float z) {
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return;
            low = {std::min(low.x, x), std::min(low.y, y), std::min(low.z, z)};
            high = {std::max(high.x, x), std::max(high.y, y), std::max(high.z, z)};
        };
        for (const auto &p : runtime_.frame.fluid_particles)
            include(p.position[0], p.position[1], p.position[2]);
        for (std::size_t i = 0; i < runtime_.frame.rigid_states.size(); ++i) {
            if (runtime_.scene.rigid_bodies[i].options.motion != MotionType::dynamic)
                continue;
            const auto &p = runtime_.frame.rigid_states[i].position;
            include(p[0] - .35F, p[1] - .35F, p[2] - .35F);
            include(p[0] + .35F, p[1] + .35F, p[2] + .35F);
        }
        if (std::isfinite(low.x)) {
            const Vec3 extent{high.x-low.x, high.y-low.y, high.z-low.z};
            const float radius = .5F * std::sqrt(extent.x*extent.x + extent.y*extent.y + extent.z*extent.z);
            preset = {.target = {(low.x+high.x)*.5F, (low.y+high.y)*.5F, (low.z+high.z)*.5F},
                .distance_scale = std::max(.28F, (radius+.25F)*2.8F/11), .pitch = .8F};
        }
    }
    runtime_.camera.set_preset(preset);
}
GalleryActionOutcome GallerySession::primary_action() {
    require_scene();
    std::string error;
    const GalleryActionOutcome outcome = runtime_.primary_action(error);
    require(error.empty(), error);
    if (outcome == GalleryActionOutcome::reset_requested) reset();
    released_ = runtime_.controller_state().constraints_released;
    return outcome;
}
void GallerySession::release() { (void)primary_action(); }
void GallerySession::set_paused(bool value) {
    if (paused_ == value) return;
    paused_ = value;
    frame_policy_.set_paused(value);
    timing.clear_history();
}
void GallerySession::set_active(bool value) {
    active_ = value;
    frame_policy_.set_active(value);
    right_ = up_ = 0;
    timing.clear_history();
}
void GallerySession::steer(float right, float up) {
    right_ = std::clamp(right, -1.0F, 1.0F); up_ = std::clamp(up, -1.0F, 1.0F);
}
void GallerySession::set_bricks(unsigned count) {
    parameters_.bricks.brick_count = std::clamp(count, 1U, limits.brick_count);
    if (entry(selected_).context == GalleryContext::rigid_body) select(selected_);
}
void GallerySession::set_scene_parameters(
    GallerySceneParameters parameters) noexcept {
    parameters_ = parameters;
}
GallerySceneParameters GallerySession::scene_parameters() const noexcept {
    return parameters_;
}
bool GallerySession::controlled_step(const GalleryControlInput &input,
                                     GalleryStepOptions options,
                                     std::string &error) {
    if (!runtime_.ready()) {
        error = "Choose a scene or Reset to continue";
        return false;
    }
    if (!runtime_.controlled_step(input, options, error)) return false;
    ++frames;
    return true;
}
bool GallerySession::diagnostic_step(const GalleryControlInput &input,
                                     GalleryStepOptions options,
                                     std::string &error) {
    return controlled_step(input, options, error);
}
void GallerySession::frame(const Presenter &present) {
    frame_at(frame_policy_.next_fixed_frame_time(), present);
}
void GallerySession::frame_at(
    FrameClock::time_point timestamp, const Presenter &present) {
    GalleryFramePlan plan;
    plan.controls = {
        .right = right_,
        .forward = up_,
        .camera = runtime_.camera.camera(),
    };
    plan.host_diagnostics = diagnostics_;
    frame_at(timestamp, plan, {}, present);
}
void GallerySession::frame_at(
    FrameClock::time_point timestamp, GalleryFramePlan plan,
    const FrameRenderer &render, const Presenter &present) {
    if (!active_) return;
    require_scene();
    frame_policy_.begin_frame(timestamp, plan.simulation_enabled);
    timing.begin_frame();
    try {
        std::string error;
        while (frame_policy_.can_step()) {
            timing.begin_phase(FramePhase::physics);
            const auto controls = runtime_.controller_state();
            const bool native_host_renderer =
                runtime_.renderer.native_image().handle != 0U;
            const bool native_gpu_renderer = bool(runtime_.native_renderer);
            const bool read_host = !native_gpu_renderer ||
                plan.host_diagnostics ||
                controls.directional == GalleryControlPolicy::collector_gravity;
            require(controlled_step(plan.controls, {
                .timestep = frame_policy_.timestep(),
                .substeps = frame_policy_.substeps(),
                .collect_rigid_contacts = plan.collect_rigid_contacts,
                .collect_fluid_contacts = plan.collect_fluid_contacts,
                .rigid_contact_pass_limit = plan.rigid_contact_pass_limit,
                .read_deformables =
                    (!native_host_renderer && !native_gpu_renderer) ||
                    plan.host_diagnostics,
                .read_host = read_host,
            }, error), error);
            frame_policy_.record_step(timing.end_phase() / 1000.0);
        }
        frame_policy_.finish_frame();
        timing.begin_phase(FramePhase::render);
        if (render) {
            render(*this, frame_policy_.interpolation_alpha());
        } else {
#ifdef PM_GALLERY_NATIVE_GPU_RENDERER
            if (runtime_.native_renderer) {
                pixels.clear();
                GpuRenderTarget target = render_target_;
                if (diagnostics_) target.handle = 0U;
                require(render_native(target, {},
                    diagnostics_ ? &pixels : nullptr, error), error);
            } else
#endif
                require(render_host(pixels, error, nullptr,
                                    FluidRenderMode::particles), error);
        }
        timing.end_phase();
        if (present) {
            timing.begin_phase(FramePhase::present);
            present(*this);
            timing.end_phase();
        }
        timing.finish_frame(); // No scan-out claim. UI handoff excluded unless supplied above.
    } catch (...) {
        frame_policy_.reset();
        timing.cancel_frame();
        throw;
    }
}
std::string GallerySession::scene_file() const {
    return std::string(catalog_item(selected_).asset_file);
}
FrameTimingStatistics GallerySession::qualify(unsigned frame_count) {
    if (frame_count == 0 || frame_count > 100000)
        throw std::runtime_error("Qualification frame count must be 1..100000");
    struct DiagnosticScope { bool &flag; ~DiagnosticScope() { flag = false; } } scope{diagnostics_};
    diagnostics_ = true;
    set_active(true); set_paused(false); reset();
    for (unsigned i = 0; i < frame_count; ++i) { frame(); validate_snapshot(); }
    const auto statistics = timing.statistics();
    require(frames == frame_count, "Qualification simulation count mismatch");
    set_paused(true); frame();
    require(frames == frame_count && timing.latest() && timing.latest()->physics_steps == 0,
        "Paused frame advanced physics");
    set_active(false); frame();
    require(frames == frame_count && !timing.latest(), "Inactive session advanced");
    set_active(true); set_paused(false); frame(); validate_snapshot();
    require(frames == frame_count + 1, "Foreground session did not resume");
    reset();
    require(frames == 0 && !timing.latest(), "Reset retained frames or timing history");
    frame(); validate_snapshot();
    return statistics;
}
std::string GallerySession::build_revision() { return PM_GALLERY_BUILD_REVISION; }
PmNativeDeviceInfo GallerySession::native_render_device() const {
    require_scene();
    if (runtime_.native_renderer)
        return runtime_.native_renderer->renderer_device().native;
    return native_physics_device();
}
PmNativeDeviceInfo GallerySession::native_physics_device() const {
    require_scene();
    PmNativeDeviceInfo native{};
    if (pmGetNativeDevice(runtime_.simulation.device(), &native) != PM_SUCCESS)
        throw std::runtime_error("No native physics device");
    return native;
}
PmNativeQueueInfo GallerySession::native_render_queue() const {
    require_scene();
    PmNativeQueueInfo native{};
    if (pmGetNativeQueue(runtime_.simulation.queue(), &native) != PM_SUCCESS)
        throw std::runtime_error("No native rendering queue");
    return native;
}
bool GallerySession::has_native_renderer() const noexcept {
    return bool(runtime_.native_renderer);
}
NativeRenderImage GallerySession::native_image() const noexcept {
    return runtime_.renderer.native_image();
}
bool GallerySession::render_host(
    std::vector<std::uint32_t> &rgba, std::string &error,
    RendererTimings *renderer_timings, FluidRenderMode fluid_mode,
    bool show_smoke_particles, float rigid_interpolation_alpha) {
    return runtime_.renderer.render(
        runtime_.render_frame, runtime_.camera.camera(), rgba, error,
        renderer_timings, fluid_mode, show_smoke_particles,
        rigid_interpolation_alpha);
}
bool GallerySession::render_native(
    GpuRenderTarget target, std::span<const std::uint32_t> overlay,
    std::vector<std::uint32_t> *readback, std::string &error) {
#ifdef PM_GALLERY_NATIVE_GPU_RENDERER
    if (!runtime_.native_renderer) {
        error = "native Gallery renderer is not initialized";
        return false;
    }
    target.width = width;
    target.height = height;
    GpuRenderRequest request;
    if (!make_gpu_render_request(
            runtime_.instance.render_buffers, runtime_.gpu_frame_generation,
            runtime_.camera.camera(), target, {readback},
            timing.measuring(FramePhase::render) ? &timing : nullptr, overlay,
            request, error))
        return false;
    return runtime_.native_renderer->render(request, error);
#else
    (void)target;
    (void)overlay;
    (void)readback;
    error = "native Gallery renderer is not built";
    return false;
#endif
}
bool GallerySession::read_host_image(
    std::vector<std::uint32_t> &rgba, std::string &error) {
    return runtime_.renderer.read_image(rgba, error);
}
bool GallerySession::read_host_output(std::string &error) {
    return runtime_.simulation.read_output(runtime_.frame, error);
}
bool GallerySession::advance_host_visuals(std::string &error) {
    return runtime_.renderer.advance_visuals(runtime_.render_frame, error);
}
bool GallerySession::paint_coverage(
    std::uint64_t &painted_texels, std::string &error) const {
    return runtime_.renderer.paint_coverage(painted_texels, error);
}
bool GallerySession::set_rigid_state(
    std::size_t scene_body_index, const RigidBodyState &state,
    std::string &error) {
    if (scene_body_index >= runtime_.instance.rigid_bodies.size()) {
        error = "Gallery rigid body index is out of range";
        return false;
    }
    return runtime_.simulation.set_rigid_state(
        runtime_.instance.rigid_bodies[scene_body_index], state, error);
}
std::size_t GallerySession::fixed_attached_count() const noexcept {
    return runtime_.fixed_collector.attached_count();
}
std::size_t GallerySession::fixed_constraint_count() const noexcept {
    return runtime_.fixed_collector.generated_constraint_count();
}
std::size_t GallerySession::particle_count() const {
    return runtime_.ready() ? runtime_.frame.fluid_particles.size() : 0;
}
std::size_t GallerySession::moving_bodies() const {
    if (!runtime_.ready()) return 0;
    return std::count_if(runtime_.scene.rigid_bodies.begin(),
        runtime_.scene.rigid_bodies.end(),
        [](const auto &body) { return body.options.motion == MotionType::dynamic; });
}
std::uint64_t GallerySession::allocated_bytes() const {
    PmMemoryBudget budget{};
    return runtime_.ready() &&
        pmGetDeviceMemoryBudget(runtime_.simulation.device(), &budget) == PM_SUCCESS
        ? budget.applicationAllocationBytes : 0;
}
std::string GallerySession::status() const {
    require_scene();
    std::ostringstream text;
    text << (paused_ ? "Paused · " : "") << config_.backend << " · " << frames << " steps";
    if (config_.gpu_rendering) text << " · GPU graphics";
    else if (!runtime_.frame.fluid_particles.empty())
        text << " · " << runtime_.frame.fluid_particles.size() << " particles";
    return text.str();
}
void GallerySession::validate_snapshot() const {
    require_scene();
    const auto finite = [](float value) {
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite physics state");
    };
    const auto vector = [&](PmFloat3 v) { finite(v.x); finite(v.y); finite(v.z); };
    const auto nodes = [&](const auto &domains) {
        for (const auto &domain : domains) {
            if (domain.empty()) throw std::runtime_error("Empty domain state");
            for (const auto &node : domain) { vector(node.position); vector(node.velocity); }
        }
    };
    const auto &output = runtime_.frame;
    for (const auto &body : output.rigid_states) {
        for (float x : body.position) finite(x);
        for (float x : body.orientation) finite(x);
        for (float x : body.linearVelocity) finite(x);
        for (float x : body.angularVelocity) finite(x);
    }
    for (const auto &particle : output.fluid_particles) {
        for (float x : particle.position) finite(x);
        for (float x : particle.velocity) finite(x);
        for (float x : particle.acceleration) finite(x);
        finite(particle.temperatureCelsius); finite(particle.foam);
    }
    nodes(output.cloth_vertices); nodes(output.soft_body_nodes); nodes(output.rope_nodes);
    for (const auto &particle : output.smoke_particles) {
        vector(particle.position); vector(particle.velocity); vector(particle.vorticity);
        finite(particle.age); finite(particle.numberDensity); finite(particle.pressure);
    }
    if ((entry(selected_).has_fluid && output.fluid_particles.empty()) ||
        (runtime_.scene.has_smoke && output.smoke_particles.empty()) ||
        output.rigid_states.size() != runtime_.scene.rigid_bodies.size() ||
        output.cloth_vertices.size() != runtime_.scene.cloths.size() ||
        output.soft_body_nodes.size() != runtime_.scene.soft_bodies.size() ||
        output.rope_nodes.size() != runtime_.scene.ropes.size())
        throw std::runtime_error("Missing domain output");
    if (pixels.size() != std::size_t(width) * height || pixels.empty())
        throw std::runtime_error("Invalid rendered image size");
    if (std::all_of(pixels.begin(), pixels.end(), [&](auto pixel) { return pixel == pixels.front(); }))
        throw std::runtime_error("Rendered image has no visible scene");
}
std::string GallerySession::details() const {
    require_scene();
    std::ostringstream text;
    const auto &selected = catalog_item(selected_);
    text << device_ << "\nScene: " << selected.id << " · available"
         << "\nRequires: " << selected.required_feature_names
         << (config_.gpu_rendering
        ? "\nShared Slang physics; direct Metal GPU graphics (fenced preview).\n\nProfile: "
        : "\nShared Slang physics; software image renderer.\n\nProfile: ") << limits.name
         << "\n" << limits.fluid_particles << " liquid particles · " << limits.coupled_bodies
         << " coupled bodies\n" << width << " × " << height << " · target " << limits.target_fps
         << " updates/s\nHost work throughput: " << timing.fps() << " frames/s";
    if (timing.latest()) {
        const auto &sample = *timing.latest();
        text << "\nFrame work: " << sample.total_ms << " ms\nPhysics: " << sample.physics_ms
             << " ms · render: " << sample.render_ms << " ms";
    }
    text << (timing.latest() && timing.latest()->present_ms
        ? "\nPresentation callback included; display scan-out excluded."
        : "\nUI handoff/display time excluded.")
         << "\n\nDrag to orbit. Pinch to zoom. Hold arrows for the scene's directional action."
         << "\n\nPreview only: workloads are provisional. Slow updates slow simulation. Missing roadmap interactions and full qualification remain pending."
         << "\n\nBuild " PM_GALLERY_BUILD_REVISION;
    return text.str();
}
}
