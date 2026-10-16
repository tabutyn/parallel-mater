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
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>
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
struct GallerySession::Scene {
    SceneDefinition definition;
    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame output;
    RenderSceneFrame render_frame;
    RendererHandle renderer;
#ifdef PM_GALLERY_METAL_RENDERER
    std::unique_ptr<GpuGalleryRenderer> gpu_renderer;
#endif
    CameraController camera;
    ArrowForces arrows;
    FixedContactCollector collector;
};
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
    if (!scene_) throw std::runtime_error("Choose a scene or Reset to continue");
}
void GallerySession::select(unsigned index) {
    const auto &selected = entry(index);
    auto next = std::make_unique<Scene>();
    std::string error;
    require(load_glb_scene(config_.assets / files[static_cast<unsigned>(selected.source)],
                           next->definition, error), error);
    if (selected.context == GalleryContext::rigid_body) {
        SceneDefinition generated;
        require(make_brick_scene(next->definition, {bricks_, 1.0F, 1U}, generated, error), error);
        next->definition = std::move(generated);
    }
    if (selected.has_fluid) next->definition.fluid_options.capacity = limits.fluid_particles;
    if (selected.context == GalleryContext::fluid || selected.context == GalleryContext::fluid_rigid)
        require(apply_liquid_workload_limits(next->definition, limits, error), error);
    next->definition.smoke_options.capacity = std::min(
        next->definition.smoke_options.capacity, limits.smoke_particles);
    // Release the old device before allocating the new one. Mobile drivers can
    // retain substantial private shader scratch; two live scenes exceed budgets.
    // Invalid selections/assets above leave the old scene intact. GPU creation
    // failure below leaves a recoverable empty session, never stale output.
    scene_.reset();
    selected_ = index;
    frames = 0; pixels.clear(); timing.clear_history();
    require(CommandScene::create(next->definition, 0U, next->simulation, next->instance,
                                 error, config_.native_backend), error);
    PmPhysicalDeviceProperties device{};
    pmGetPhysicalDeviceProperties(next->simulation.physical_device(), &device);
    device_ = device.deviceName;
    if (config_.native_backend) {
        PmNativeDeviceInfo native{};
        if (pmGetNativeDevice(next->simulation.device(), &native) != PM_SUCCESS ||
            native.backend != *config_.native_backend)
            throw std::runtime_error("Requested GPU backend required; CPU fallback is not permitted");
    }
    require(next->arrows.initialize(next->definition, next->instance, error), error);
    if (selected.context == GalleryContext::constraint_fixed)
        require(next->collector.initialize(next->definition, next->instance, error), error);
    if (!config_.gpu_rendering)
        require(create_renderer(next->definition, next->instance, width, height, next->renderer, error), error);
#ifdef PM_GALLERY_METAL_RENDERER
    else next->gpu_renderer = std::make_unique<GpuGalleryRenderer>(next->simulation.device(), next->definition, next->instance);
#endif
    require(next->simulation.step(1.0F / 60.0F, 4U,
        {0, -9.81F * next->definition.gravity_scale, 0}, next->output, error), error);
    require(make_render_scene_frame(next->definition, next->output, next->render_frame, error), error);
    scene_ = std::move(next);
    selected_ = index;
    frames = 0;
    right_ = up_ = 0;
    released_ = false;
    timing.clear_history();
    focus();
}
void GallerySession::reset() { select(selected_); }
void GallerySession::resize(unsigned w, unsigned h) {
    const float scale = std::min({1.0F, float(limits.render_width) / std::max(w, 1U),
        float(limits.render_height) / std::max(h, 1U)});
    w = std::max(1U, unsigned(w * scale)); h = std::max(1U, unsigned(h * scale));
    if (w == width && h == height) return;
    std::string error;
    RendererHandle renderer;
    if (scene_ && !config_.gpu_rendering) {
        require(create_renderer(scene_->definition, scene_->instance, w, h, renderer, error), error);
        scene_->renderer = std::move(renderer);
    }
    width = w; height = h;
    timing.clear_history();
}
void GallerySession::orbit(float x, float y) {
    require_scene();
    scene_->camera.begin_drag(CameraDragMode::orbit, 0, 0);
    scene_->camera.move_cursor(x, y, height); scene_->camera.end_drag();
}
void GallerySession::zoom(float amount) { require_scene(); scene_->camera.zoom(amount); }
void GallerySession::focus() {
    require_scene();
    scene_->camera = CameraController{};
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
        for (const auto &p : scene_->output.fluid_particles) include(p.position[0], p.position[1], p.position[2]);
        for (std::size_t i = 0; i < scene_->output.rigid_states.size(); ++i) {
            if (scene_->definition.rigid_bodies[i].options.motion != MotionType::dynamic) continue;
            const auto &p = scene_->output.rigid_states[i].position;
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
    scene_->camera.set_preset(preset);
}
void GallerySession::release() {
    require_scene();
    if (entry(selected_).context != GalleryContext::constraint_point) return;
    if (released_) { reset(); return; }
    std::string error;
    for (auto slot : scene_->instance.rigid_constraints)
        require(scene_->simulation.set_constraint_active(slot, false, error), error);
    released_ = true;
}
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
    if (entry(selected_).context == GalleryContext::rigid_body) reset();
}
void GallerySession::frame(const std::function<void(const GallerySession &)> &present) {
    if (!active_) return;
    require_scene();
    timing.begin_frame();
    try {
        std::string error;
        if (!paused_) {
            timing.begin_phase(FramePhase::physics);
            const float magnitude = 9.81F * scene_->definition.gravity_scale;
            const Vec3 gravity = screen_space_gravity(scene_->camera.camera(),
                scene_->arrows.active() ? 0 : right_, scene_->arrows.active() ? 0 : up_,
                magnitude, 30.0F);
            if (scene_->collector.active())
                require(scene_->collector.apply_loose_gravity(scene_->simulation,
                    scene_->definition, scene_->instance, {0, -magnitude, 0}, gravity, error), error);
            require(scene_->arrows.apply(scene_->simulation, scene_->camera.camera(), right_, up_, error), error);
            require(scene_->simulation.step(1.0F / limits.target_fps,
                (240U + limits.target_fps - 1) / limits.target_fps, gravity,
                scene_->output, error, scene_->collector.active(),
                !config_.gpu_rendering || diagnostics_ || scene_->collector.active(),
                !config_.gpu_rendering || diagnostics_ || scene_->collector.active()), error);
            if (scene_->collector.active())
                require(scene_->collector.collect(scene_->simulation, scene_->definition,
                    scene_->instance, scene_->output, error), error);
            ++frames;
            if (!config_.gpu_rendering)
                require(make_render_scene_frame(scene_->definition, scene_->output, scene_->render_frame, error), error);
            timing.end_phase();
        }
        timing.begin_phase(FramePhase::render);
#ifdef PM_GALLERY_METAL_RENDERER
        if (scene_->gpu_renderer) {
            pixels.clear();
            scene_->gpu_renderer->render(scene_->instance.render_buffers, scene_->camera.camera(), width, height,
                diagnostics_ ? nullptr : presentation_layer_, diagnostics_ ? &pixels : nullptr, &timing);
        } else
#endif
            require(scene_->renderer.render(scene_->render_frame, scene_->camera.camera(), pixels,
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
    if (pmGetNativeDevice(scene_->simulation.device(), &native) != PM_SUCCESS)
        throw std::runtime_error("No native rendering device");
    return reinterpret_cast<void *>(native.device);
}
std::size_t GallerySession::particle_count() const { return scene_ ? scene_->output.fluid_particles.size() : 0; }
std::size_t GallerySession::moving_bodies() const {
    if (!scene_) return 0;
    return std::count_if(scene_->definition.rigid_bodies.begin(), scene_->definition.rigid_bodies.end(),
        [](const auto &body) { return body.options.motion == MotionType::dynamic; });
}
std::uint64_t GallerySession::allocated_bytes() const {
    PmMemoryBudget budget{};
    return scene_ && pmGetDeviceMemoryBudget(scene_->simulation.device(), &budget) == PM_SUCCESS
        ? budget.applicationAllocationBytes : 0;
}
std::string GallerySession::status() const {
    require_scene();
    std::ostringstream text;
    text << (paused_ ? "Paused · " : "") << config_.backend << " · " << frames << " steps";
    if (config_.gpu_rendering) text << " · GPU graphics";
    else if (!scene_->output.fluid_particles.empty()) text << " · " << scene_->output.fluid_particles.size() << " particles";
    text << "\n" << (scene_->arrows.active() ? "Push objects" : "Tilt gravity");
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
    const auto &output = scene_->output;
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
        (scene_->definition.has_smoke && output.smoke_particles.empty()) ||
        output.rigid_states.size() != scene_->definition.rigid_bodies.size() ||
        output.cloth_vertices.size() != scene_->definition.cloths.size() ||
        output.soft_body_nodes.size() != scene_->definition.soft_bodies.size() ||
        output.rope_nodes.size() != scene_->definition.ropes.size())
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
