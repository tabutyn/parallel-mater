// SPDX-License-Identifier: MIT
// Mobile behavior follows Android: bounded work, shared budgets, touch camera.
#include <parallel_mater_gallery/session.hpp>
#include "scene_builder.hpp"
#include "render/render_scene.hpp"
#include <parallel_mater_gallery/renderer.hpp>
#ifdef PM_GALLERY_METAL_RENDERER
#include <parallel_mater_gallery/gpu_renderer.hpp>
#endif
#include <parallel_mater_gallery/gallery_context.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace parallel_mater::gallery {
namespace {
constexpr std::array files{"RigidBody.glb", "ConstraintFixed.glb", "Celestial.glb",
    "ConstraintHinge.glb", "ConstraintPiston.glb", "ConstraintGeneric.glb",
    "ConstraintMotorSpring.glb", "DumpTruck.glb", "Fluid.glb", "FluidRigid.glb",
    "Pegs.glb", "Cloth.glb", "ClothTear.glb", "ClothPaint.glb", "ClothWater.glb",
    "Softbody.glb", "SoftbodyRigidBody.glb", "SoftbodyCloth.glb", "SoftbodyFluid.glb",
    "Rope.glb", "RopeFluid.glb", "RopeSoftbody.glb", "RopeCloth.glb", "Smoke.glb",
    "SmokeWater.glb", "SmokeSoftbody.glb", "SmokeCloth.glb", "SmokeRope.glb"};
static_assert(files.size() == gallery_entries.size());
const GalleryEntry &scene_entry(GallerySceneSet scenes, unsigned index) {
    constexpr std::array<unsigned, 8> rigid_liquid{0, 1, 2, 3, 4, 5, 8, 9};
    if (scenes == GallerySceneSet::rigid_liquid) {
        if (index >= rigid_liquid.size()) throw std::runtime_error("Unknown scene");
        return gallery_entries[rigid_liquid[index]];
    }
    // Dump requires vehicle controls, which this touch preview does not expose.
    if (index >= gallery_entries.size() - 1) throw std::runtime_error("Unknown scene");
    return gallery_entries[index >= 7 ? index + 1 : index];
}
void require(bool ok, const std::string &error) {
    if (!ok) throw std::runtime_error(error.empty() ? "Gallery operation failed" : error);
}
}
std::vector<std::string> GallerySession::scene_names(GallerySceneSet scenes) {
    std::vector<std::string> names;
    const unsigned count = scenes == GallerySceneSet::rigid_liquid ? 8 : gallery_entries.size() - 1;
    for (unsigned i = 0; i < count; ++i)
        names.emplace_back(scene_entry(scenes, i).name);
    return names;
}
const GalleryEntry &GallerySession::entry(unsigned index) const { return scene_entry(config_.scenes, index); }
GallerySession::GallerySession(GallerySessionConfig config) : config_(std::move(config)) {
#ifndef PM_GALLERY_METAL_RENDERER
    if (config_.gpu_rendering) throw std::runtime_error("Native GPU Gallery renderer is not built");
#endif
    std::string error;
    DeviceProfileCatalog catalog;
    require(load_device_profiles(config_.profiles, catalog, error), error);
    const auto *profile = select_workload_limits(catalog, config_.platform, config_.model, config_.backend);
    if (!profile) throw std::runtime_error("Gallery workload profile missing");
    limits = *profile;
    bricks_ = limits.brick_count;
    width = limits.render_width;
    height = limits.render_height;
    select(0);
}
GallerySession::~GallerySession() = default;
void GallerySession::require_scene() const {
    if (!runtime_.ready())
        throw std::runtime_error("Choose a scene or Reset to continue");
}
void GallerySession::select(unsigned index) {
    const auto &selected = entry(index);
    std::string error;
    GalleryRuntimeCreateInfo request;
    request.context = selected.context;
    request.asset =
        config_.assets / files[static_cast<unsigned>(selected.source)];
    request.backend = config_.native_backend;
    request.width = width;
    request.height = height;
    request.create_host_renderer = !config_.gpu_rendering;
    request.replacement =
        GalleryReplacementPolicy::release_before_device_create;
    const unsigned brick_count = bricks_;
    const GalleryWorkloadLimits selected_limits = limits;
    request.configure_scene =
        [context = selected.context, has_fluid = selected.has_fluid,
         brick_count, selected_limits](SceneDefinition &scene,
                                       std::string &failure) {
        if (context == GalleryContext::rigid_body) {
            SceneDefinition generated;
            if (!make_brick_scene(scene, {brick_count, 1.0F, 1U},
                                  generated, failure)) return false;
            scene = std::move(generated);
        }
        if (has_fluid)
            scene.fluid_options.capacity = selected_limits.fluid_particles;
        if ((context == GalleryContext::fluid ||
             context == GalleryContext::fluid_rigid) &&
            !apply_liquid_workload_limits(scene, selected_limits, failure))
            return false;
        scene.smoke_options.capacity = std::min(
            scene.smoke_options.capacity, selected_limits.smoke_particles);
        return true;
    };
    const bool gpu_rendering = config_.gpu_rendering;
    const auto native_backend = config_.native_backend;
    request.prepare_runtime =
        [this, gpu_rendering, native_backend]
        (GalleryRuntime &next, std::string &failure) {
        PmPhysicalDeviceProperties device{};
        pmGetPhysicalDeviceProperties(next.simulation.physical_device(), &device);
        if (native_backend) {
            PmNativeDeviceInfo native{};
            if (pmGetNativeDevice(next.simulation.device(), &native) != PM_SUCCESS ||
                native.backend != *native_backend) {
                failure =
                    "Requested GPU backend required; CPU fallback is not permitted";
                return false;
            }
        }
#ifdef PM_GALLERY_METAL_RENDERER
        if (gpu_rendering)
            next.native_renderer = std::make_shared<GpuGalleryRenderer>(
                next.simulation.device(), next.scene, next.instance);
#else
        (void)gpu_rendering;
#endif
        if (!next.step({.gravity = {0.0F,
                -9.81F * next.scene.gravity_scale, 0.0F}}, failure)) {
            return false;
        }
        device_ = device.deviceName;
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
            right_ = up_ = 0;
            released_ = false;
        }
        require(false, error);
    }
    selected_ = index;
    frames = 0;
    pixels.clear();
    timing.clear_history();
    right_ = up_ = 0;
    released_ = false;
    focus();
}
void GallerySession::reset() {
    std::string error;
    if (!runtime_.reset(error)) {
        if (!runtime_.ready()) {
            frames = 0;
            pixels.clear();
            timing.clear_history();
            right_ = up_ = 0;
            released_ = false;
        }
        require(false, error);
    }
    frames = 0;
    pixels.clear();
    timing.clear_history();
    right_ = up_ = 0;
    released_ = false;
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
        ? brick_camera_preset({bricks_, 1.0F, 1U}) : entry(selected_).camera;
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
    paused_ = value; timing.clear_history();
}
void GallerySession::set_active(bool value) {
    active_ = value; right_ = up_ = 0; timing.clear_history();
}
void GallerySession::steer(float right, float up) {
    right_ = std::clamp(right, -1.0F, 1.0F); up_ = std::clamp(up, -1.0F, 1.0F);
}
void GallerySession::set_bricks(unsigned count) {
    bricks_ = std::clamp(count, 1U, limits.brick_count);
    if (entry(selected_).context == GalleryContext::rigid_body) select(selected_);
}
void GallerySession::frame(const std::function<void(const GallerySession &)> &present) {
    if (!active_) return;
    require_scene();
    timing.begin_frame();
    try {
        std::string error;
        if (!paused_) {
            timing.begin_phase(FramePhase::physics);
            const auto controls = runtime_.controller_state();
            const bool read_host = !config_.gpu_rendering || diagnostics_ ||
                controls.directional == GalleryControlPolicy::collector_gravity;
            require(runtime_.controlled_step({
                .right = right_,
                .forward = up_,
                .camera = runtime_.camera.camera(),
            }, {
                .timestep = 1.0F / limits.target_fps,
                .substeps = (240U + limits.target_fps - 1) / limits.target_fps,
                .read_deformables = read_host,
                .read_host = read_host,
            }, error), error);
            ++frames;
            timing.end_phase();
        }
        timing.begin_phase(FramePhase::render);
#ifdef PM_GALLERY_METAL_RENDERER
        if (runtime_.native_renderer) {
            pixels.clear();
            static_cast<GpuGalleryRenderer *>(runtime_.native_renderer.get())->render(
                runtime_.instance.render_buffers, runtime_.camera.camera(), width, height,
                diagnostics_ ? nullptr : presentation_layer_, diagnostics_ ? &pixels : nullptr, &timing);
        } else
#endif
            require(runtime_.renderer.render(runtime_.render_frame,
                runtime_.camera.camera(), pixels,
                error, nullptr, FluidRenderMode::particles), error);
        timing.end_phase();
        if (present) {
            timing.begin_phase(FramePhase::present);
            present(*this);
            timing.end_phase();
        }
        timing.finish_frame(); // No scan-out claim. UI handoff excluded unless supplied above.
    } catch (...) { timing.cancel_frame(); throw; }
}
std::string GallerySession::scene_file() const { return files[static_cast<unsigned>(entry(selected_).source)]; }
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
void *GallerySession::native_render_device() const {
    require_scene();
    PmNativeDeviceInfo native{};
    if (pmGetNativeDevice(runtime_.simulation.device(), &native) != PM_SUCCESS)
        throw std::runtime_error("No native rendering device");
    return reinterpret_cast<void *>(native.device);
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
    text << "\n" << (runtime_.controller_state().authored_force
        ? "Push objects" : "Tilt gravity");
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
    text << device_ << (config_.gpu_rendering
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
         << "\n\nDrag to orbit. Pinch to zoom. Hold arrows to push objects or tilt gravity."
         << "\n\nPreview only: workloads are provisional. Slow updates slow simulation. Missing roadmap interactions and full qualification remain pending."
         << "\n\nBuild " PM_GALLERY_BUILD_REVISION;
    return text.str();
}
}
