// SPDX-License-Identifier: MIT
// Mobile behavior follows Android: bounded work, shared budgets, touch camera.
#include "gallery_session.hpp"
#include "scene_builder.hpp"
#include "render/render_scene.hpp"
#include <parallel_mater_gallery/renderer.hpp>
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
const GalleryEntry &entry(unsigned index) {
    // Dump requires vehicle controls, which this touch preview does not expose.
    if (index >= gallery_entries.size() - 1) throw std::runtime_error("Unknown scene");
    return gallery_entries[index >= 7 ? index + 1 : index];
}
void require(bool ok, const std::string &error) {
    if (!ok) throw std::runtime_error(error.empty() ? "Gallery operation failed" : error);
}
}
struct IPadGallerySession::Scene {
    SceneDefinition definition;
    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame output;
    RenderSceneFrame render_frame;
    RendererHandle renderer;
    CameraController camera;
    ArrowForces arrows;
    FixedContactCollector collector;
};
std::vector<std::string> IPadGallerySession::scene_names() {
    std::vector<std::string> names;
    for (unsigned i = 0; i < gallery_entries.size() - 1; ++i)
        names.emplace_back(entry(i).name);
    return names;
}
IPadGallerySession::IPadGallerySession(const std::filesystem::path &resources,
                                      const std::string &model) : resources_(resources) {
    std::string error;
    DeviceProfileCatalog catalog;
    require(load_device_profiles(resources / "config/device-profiles.json", catalog, error), error);
    const auto *profile = select_workload_limits(catalog, "iPadOS", model, "metal");
    if (!profile) throw std::runtime_error("iPad workload profile missing");
    limits = *profile;
    width = limits.render_width;
    height = limits.render_height;
    select(0);
}
IPadGallerySession::~IPadGallerySession() = default;
void IPadGallerySession::select(unsigned index) {
    const auto &selected = entry(index);
    auto next = std::make_unique<Scene>();
    std::string error;
    require(load_glb_scene(resources_ / "assets" / files[static_cast<unsigned>(selected.source)],
                           next->definition, error), error);
    if (selected.context == GalleryContext::rigid_body) {
        SceneDefinition generated;
        require(make_brick_scene(next->definition, {limits.brick_count, 1.0F, 1U}, generated, error), error);
        next->definition = std::move(generated);
    }
    if (selected.has_fluid) next->definition.fluid_options.capacity = limits.fluid_particles;
    if (selected.context == GalleryContext::fluid || selected.context == GalleryContext::fluid_rigid)
        require(apply_liquid_workload_limits(next->definition, limits, error), error);
    next->definition.smoke_options.capacity = std::min(
        next->definition.smoke_options.capacity, limits.smoke_particles);
    require(CommandScene::create(next->definition, 0U, next->simulation, next->instance, error), error);
    PmPhysicalDeviceProperties device{};
    pmGetPhysicalDeviceProperties(next->simulation.physical_device(), &device);
    device_ = device.deviceName;
    if (device_.find("Metal GPU preview") == std::string::npos)
        throw std::runtime_error("Apple GPU required; CPU fallback is not permitted");
    require(next->arrows.initialize(next->definition, next->instance, error), error);
    if (selected.context == GalleryContext::constraint_fixed)
        require(next->collector.initialize(next->definition, next->instance, error), error);
    require(create_renderer(next->definition, next->instance, width, height, next->renderer, error), error);
    require(next->simulation.step(1.0F / 60.0F, 4U,
        {0, -9.81F * next->definition.gravity_scale, 0}, next->output, error), error);
    require(make_render_scene_frame(next->definition, next->output, next->render_frame, error), error);
    scene_ = std::move(next);
    selected_ = index;
    frames = 0;
    right = up = 0;
    released_ = false;
    timing.clear_history();
    focus();
}
void IPadGallerySession::reset() { select(selected_); }
void IPadGallerySession::resize(unsigned w, unsigned h) {
    const float scale = std::min({1.0F, float(limits.render_width) / std::max(w, 1U),
        float(limits.render_height) / std::max(h, 1U)});
    w = std::max(1U, unsigned(w * scale)); h = std::max(1U, unsigned(h * scale));
    if (w == width && h == height) return;
    std::string error;
    RendererHandle renderer;
    require(create_renderer(scene_->definition, scene_->instance, w, h, renderer, error), error);
    scene_->renderer = std::move(renderer); width = w; height = h;
    timing.clear_history();
}
void IPadGallerySession::orbit(float x, float y) {
    scene_->camera.begin_drag(CameraDragMode::orbit, 0, 0);
    scene_->camera.move_cursor(x, y, height); scene_->camera.end_drag();
}
void IPadGallerySession::zoom(float amount) { scene_->camera.zoom(amount); }
void IPadGallerySession::focus() {
    scene_->camera = CameraController{};
    scene_->camera.set_preset(entry(selected_).context == GalleryContext::rigid_body
        ? brick_camera_preset({limits.brick_count, 1.0F, 1U}) : entry(selected_).camera);
}
void IPadGallerySession::release() {
    if (entry(selected_).context != GalleryContext::constraint_point) return;
    if (released_) { reset(); return; }
    std::string error;
    for (auto slot : scene_->instance.rigid_constraints)
        require(scene_->simulation.set_constraint_active(slot, false, error), error);
    released_ = true;
}
void IPadGallerySession::frame() {
    timing.begin_frame();
    try {
        std::string error;
        if (!paused) {
            timing.begin_phase(FramePhase::physics);
            const float magnitude = 9.81F * scene_->definition.gravity_scale;
            const Vec3 gravity = screen_space_gravity(scene_->camera.camera(),
                scene_->arrows.active() ? 0 : right, scene_->arrows.active() ? 0 : up,
                magnitude, 30.0F);
            if (scene_->collector.active())
                require(scene_->collector.apply_loose_gravity(scene_->simulation,
                    scene_->definition, scene_->instance, {0, -magnitude, 0}, gravity, error), error);
            require(scene_->arrows.apply(scene_->simulation, scene_->camera.camera(), right, up, error), error);
            require(scene_->simulation.step(1.0F / limits.target_fps,
                (240U + limits.target_fps - 1) / limits.target_fps, gravity,
                scene_->output, error, scene_->collector.active()), error);
            if (scene_->collector.active())
                require(scene_->collector.collect(scene_->simulation, scene_->definition,
                    scene_->instance, scene_->output, error), error);
            ++frames;
            require(make_render_scene_frame(scene_->definition, scene_->output, scene_->render_frame, error), error);
            timing.end_phase();
        }
        timing.begin_phase(FramePhase::render);
        require(scene_->renderer.render(scene_->render_frame, scene_->camera.camera(), pixels,
            error, nullptr, FluidRenderMode::particles), error);
        timing.end_phase();
        timing.finish_frame(); // Host frame work, before UIKit image handoff; no scan-out claim.
    } catch (...) { timing.cancel_frame(); throw; }
}
std::string IPadGallerySession::status() const {
    std::ostringstream text;
    text << (paused ? "Paused · " : "") << "Apple GPU · " << frames << " steps";
    if (!scene_->output.fluid_particles.empty()) text << " · " << scene_->output.fluid_particles.size() << " particles";
    return text.str();
}
void IPadGallerySession::validate_snapshot() const {
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
std::string IPadGallerySession::details() const {
    std::ostringstream text;
    text << device_ << "\nShared Slang physics; software image renderer.\n\nProfile: " << limits.name
         << "\n" << limits.fluid_particles << " liquid particles · " << limits.coupled_bodies
         << " coupled bodies\n" << width << " × " << height << " · target " << limits.target_fps
         << " updates/s\nHost work throughput: " << timing.fps() << " frames/s";
    if (timing.latest()) {
        const auto &sample = *timing.latest();
        text << "\nFrame work: " << sample.total_ms << " ms\nPhysics: " << sample.physics_ms
             << " ms · render: " << sample.render_ms << " ms";
    }
    text << "\nUI handoff/display time excluded.\n\nDrag to orbit. Pinch to zoom. Hold arrows to push objects or tilt gravity."
         << "\n\nPreview only: workloads are provisional. Slow updates slow simulation. Missing roadmap interactions and full qualification remain pending."
         << "\n\nBuild " PM_IPAD_REVISION;
    return text.str();
}
}
