// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_runtime.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <utility>
#include <vector>

namespace parallel_mater::gallery {
namespace {
std::atomic_size_t live_runtimes{};

constexpr float gravity_magnitude = 9.81F;
constexpr float rigid_gravity_tilt_degrees = 30.0F;
constexpr float deformable_gravity_tilt_degrees = 45.0F;
constexpr float kinematic_speed = 2.0F;
constexpr float motor_speed = 8.0F;
constexpr float pi = 3.14159265358979323846F;
constexpr float motor_steering_angle = 25.0F * pi / 180.0F;
constexpr float motor_steering_speed = 90.0F * pi / 180.0F;

[[nodiscard]] Quaternion multiply(Quaternion left,
                                  Quaternion right) noexcept {
    return {
        left.w * right.x + left.x * right.w + left.y * right.z -
            left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w +
            left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x +
            left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y -
            left.z * right.z};
}

[[nodiscard]] Quaternion rotation_z(float radians) noexcept {
    return {0.0F, 0.0F, std::sin(radians * 0.5F),
            std::cos(radians * 0.5F)};
}

[[nodiscard]] bool apply_acceleration(
    GalleryRuntime &runtime, const std::vector<GravityTiltBody> &bodies,
    Vec3 acceleration, std::string &error) {
    if (bodies.empty()) return true;
    std::vector<PmRigidInput> inputs;
    try {
        inputs.reserve(bodies.size());
        for (const GravityTiltBody &body : bodies) {
            PmRigidInput input{};
            input.bodyIndex = body.body;
            input.force[0] = acceleration.x * body.mass;
            input.force[1] = acceleration.y * body.mass;
            input.force[2] = acceleration.z * body.mass;
            inputs.push_back(input);
        }
    } catch (...) {
        error = "rigid acceleration allocation failed";
        return false;
    }
    return runtime.simulation.apply_rigid_inputs(inputs, error);
}
}

struct GalleryRuntime::LifetimeToken {
    LifetimeToken() noexcept { live_runtimes.fetch_add(1U); }
    ~LifetimeToken() { live_runtimes.fetch_sub(1U); }
};

GalleryRuntime::GalleryRuntime() noexcept = default;
GalleryRuntime::~GalleryRuntime() = default;
GalleryRuntime::GalleryRuntime(GalleryRuntime &&) noexcept = default;
GalleryRuntime &GalleryRuntime::operator=(GalleryRuntime &&other) noexcept {
    if (this == &other) return *this;

    // Default member-wise assignment would release simulation before the
    // renderer and scene instance declared after it. Native renderers borrow
    // the simulation device, queue, and output buffers, so preserve normal
    // reverse-declaration destruction order before reconstructing the runtime.
    std::destroy_at(this);
    std::construct_at(this, std::move(other));
    return *this;
}

std::size_t GalleryRuntime::live_runtime_count() noexcept {
    return live_runtimes.load();
}

void GalleryRuntime::clear() noexcept { *this = GalleryRuntime{}; }

bool GalleryRuntime::replace(GalleryRuntimeCreateInfo info,
                             std::string &error) {
    error.clear();
    GalleryRuntime next;
    next.context = info.context;
    next.capture_history = PhysicsDebugHistory(info.capture_history_capacity);
    if (!load_glb_scene(info.asset, next.scene, error)) {
        error = "scene load failed: " + error;
        return false;
    }
    if (info.configure_scene && !info.configure_scene(next.scene, error)) {
        if (error.empty()) error = "scene configuration failed";
        return false;
    }

    const bool release_first = info.replacement ==
        GalleryReplacementPolicy::release_before_device_create;
    if (release_first) clear();
    const auto retain_retry = [&] {
        if (release_first) recipe_ = info;
    };

    if (!CommandScene::create(next.scene, info.debug_frame_capacity,
                              next.simulation, next.instance, error,
                              info.backend)) {
        if (error.empty()) error = "scene creation failed";
        retain_retry();
        return false;
    }
    next.lifetime_ = std::make_unique<LifetimeToken>();
    if (info.create_host_renderer &&
        !RendererAdapterFactory::create_host(
            next.scene, next.instance, info.width, info.height,
            next.renderer, error)) {
        error = "renderer creation failed: " + error;
        retain_retry();
        return false;
    }
    if (!next.initialize_controllers(error)) {
        if (error.empty()) error = "controller initialization failed";
        retain_retry();
        return false;
    }
    if (info.prepare_runtime && !info.prepare_runtime(next, error)) {
        if (error.empty()) error = "runtime preparation failed";
        retain_retry();
        return false;
    }
    next.recipe_ = std::move(info);
    *this = std::move(next);
    error.clear();
    return true;
}

bool GalleryRuntime::reset(std::string &error) {
    if (!recipe_) {
        error = "Gallery scene has no reset recipe";
        return false;
    }
    GalleryRuntimeCreateInfo recipe = *recipe_;
    return replace(std::move(recipe), error);
}

bool GalleryRuntime::step(const GalleryStepOptions &options,
                          std::string &error) {
    if (!ready()) {
        error = "Choose a scene or Reset to continue";
        return false;
    }
    const bool capture_host_step = capture_history.enabled() && options.read_host;
    if (!simulation.step(options.timestep, options.substeps, options.gravity,
                         frame, error,
                         options.collect_rigid_contacts ||
                             options.collect_fluid_contacts || capture_host_step,
                         options.read_deformables, options.read_host))
        return false;
    gpu_ready = true;
    ++gpu_frame_generation;
    if (!options.read_host) {
        if (capture_history.enabled()) capture_history.mark_gap();
        return true;
    }
    if (capture_history.enabled()) {
        PmStepInfo step{options.timestep, options.substeps,
                        {options.gravity.x, options.gravity.y,
                         options.gravity.z},
                        options.collect_rigid_contacts ||
                                options.collect_fluid_contacts || capture_host_step
                            ? static_cast<PmFlags>(
                                  PM_STEP_COLLECT_CONTACT_EVENTS_BIT)
                            : PmFlags{}};
        if (!capture_history.record(frame, step, error)) return false;
    }
    RenderSceneFrame next_render_frame;
    if (!make_render_scene_frame(scene, frame, next_render_frame, error))
        return false;
    advance_render_scene_snapshots(render_frame, next_render_frame);
    render_frame = std::move(next_render_frame);
    return true;
}

bool GalleryRuntime::initialize_controllers(std::string &error) {
    error.clear();
    const GalleryEntry &entry = gallery_entry(context);
    control_gravity_ = {0.0F, -gravity_magnitude * scene.gravity_scale, 0.0F};
    constraints_released_ = false;
    steering_angle = 0.0F;

    if (entry.primary_action == GalleryPrimaryAction::toggle_dump_bed &&
        !dump_bed.initialize(scene)) {
        error = "dump truck needs its authored DumpLift joint";
        return false;
    }

    if (entry.controls == GalleryControlPolicy::motor_drive) {
        for (std::size_t spring_index = 0U;
             spring_index < scene.rigid_constraints.size(); ++spring_index) {
            const auto &spring = scene.rigid_constraints[spring_index];
            if (spring.options.type != RigidConstraintType::generic_spring)
                continue;
            const auto motor = std::find_if(
                scene.rigid_constraints.begin(), scene.rigid_constraints.end(),
                [&](const auto &candidate) {
                    return candidate.options.type == RigidConstraintType::motor &&
                           candidate.body_a == spring.body_b;
                });
            if (motor == scene.rigid_constraints.end() ||
                (motor->name.find("Front") == std::string::npos &&
                 motor->name.find("Rear") == std::string::npos)) {
                error = spring.name +
                    ": suspension hub needs a named Front or Rear motor";
                return false;
            }
            suspension_steering_joints.push_back({
                spring_index, spring.options.local_orientation_a,
                motor->name.find("Front") != std::string::npos});
        }
        if (suspension_steering_joints.size() != 4U) {
            error = "vehicle scene needs four steerable suspension hubs";
            return false;
        }
    }

    if (!arrow_forces.initialize(scene, instance, error)) return false;
    const bool expects_force =
        entry.controls == GalleryControlPolicy::authored_force;
    if (arrow_forces.active() != expects_force) {
        error = expects_force
            ? "catalog force action has no authored force target"
            : "scene has an authored force target but its catalog action differs";
        return false;
    }

    gravity_tilt_bodies.reserve(scene.rigid_bodies.size());
    for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
        if (index >= instance.rigid_bodies.size()) {
            error = "scene and runtime rigid body counts differ";
            return false;
        }
        const auto &body = scene.rigid_bodies[index];
        if (body.options.motion == MotionType::dynamic &&
            !body.follows_gravity_tilt)
            gravity_tilt_bodies.push_back(
                {instance.rigid_bodies[index], body.options.mass});
        if (entry.moves_kinematic &&
            kinematic_index == static_cast<std::size_t>(-1) &&
            body.options.motion == MotionType::kinematic) {
            kinematic_index = index;
            kinematic_target = body.options.initial_state;
        }
    }

    if (entry.controls == GalleryControlPolicy::collector_gravity &&
        !fixed_collector.initialize(scene, instance, error)) return false;
    return true;
}

GalleryActionOutcome GalleryRuntime::primary_action(std::string &error) {
    error.clear();
    if (!ready()) {
        error = "Choose a scene or Reset to continue";
        return GalleryActionOutcome::ignored;
    }
    switch (gallery_entry(context).primary_action) {
    case GalleryPrimaryAction::none:
        return GalleryActionOutcome::ignored;
    case GalleryPrimaryAction::toggle_dump_bed:
        dump_bed.toggle();
        return GalleryActionOutcome::applied;
    case GalleryPrimaryAction::release_restore_constraints:
        if (constraints_released_)
            return GalleryActionOutcome::reset_requested;
        if (instance.rigid_constraints.empty() ||
            instance.rigid_constraints.size() !=
                scene.rigid_constraints.size()) {
            error = "constraint action needs matching constraints";
            return GalleryActionOutcome::ignored;
        }
        for (std::uint32_t slot : instance.rigid_constraints) {
            if (!simulation.set_constraint_active(slot, false, error))
                return GalleryActionOutcome::ignored;
        }
        constraints_released_ = true;
        return GalleryActionOutcome::applied;
    }
    return GalleryActionOutcome::ignored;
}

bool GalleryRuntime::controlled_step(const GalleryControlInput &source,
                                     GalleryStepOptions options,
                                     std::string &error) {
    error.clear();
    if (!ready()) {
        error = "Choose a scene or Reset to continue";
        return false;
    }
    const GalleryEntry &entry = gallery_entry(context);
    const float right = std::clamp(source.right, -1.0F, 1.0F);
    const float forward = std::clamp(source.forward, -1.0F, 1.0F);
    const Vec3 vertical{
        0.0F, -gravity_magnitude * scene.gravity_scale, 0.0F};

    if (entry.controls == GalleryControlPolicy::motor_drive) {
        const float target_velocity = -forward * motor_speed;
        for (std::size_t index = 0U;
             index < scene.rigid_constraints.size(); ++index) {
            auto &definition = scene.rigid_constraints[index];
            if (definition.options.type != RigidConstraintType::motor ||
                definition.options.motor.angular_target_velocity ==
                    target_velocity)
                continue;
            auto updated = definition;
            updated.options.motor.angular_target_velocity = target_velocity;
            if (!simulation.write_constraint(
                    instance.rigid_constraints[index], updated, error))
                return false;
            definition.options = updated.options;
        }

        const float target = right * motor_steering_angle;
        const float maximum_step = motor_steering_speed * options.timestep;
        const float previous = steering_angle;
        steering_angle += std::clamp(
            target - steering_angle, -maximum_step, maximum_step);
        if (steering_angle != previous) {
            const auto angles = axle_steering_angles(steering_angle, 1.0F);
            for (const auto &steering : suspension_steering_joints) {
                auto &definition =
                    scene.rigid_constraints[steering.constraint_index];
                auto updated = definition;
                const float angle = steering.front
                    ? angles.front : angles.rear;
                updated.options.local_orientation_a = multiply(
                    steering.authored_local_orientation_a,
                    rotation_z(angle));
                if (!simulation.write_constraint(
                        instance.rigid_constraints[
                            steering.constraint_index],
                        updated, error)) return false;
                definition.options = updated.options;
            }
        }
    }

    if (entry.primary_action == GalleryPrimaryAction::toggle_dump_bed &&
        !dump_bed.advance(simulation, scene, instance,
                          options.timestep, error)) return false;

    if (entry.moves_kinematic &&
        kinematic_index < instance.rigid_bodies.size()) {
        kinematic_target.position.x += right * kinematic_speed * options.timestep;
        kinematic_target.position.z -= forward * kinematic_speed * options.timestep;
        if (!simulation.set_kinematic_target(
                instance.rigid_bodies[kinematic_index],
                kinematic_target, error)) return false;
    }

    Vec3 gravity = vertical;
    if (source.gravity_override) {
        gravity = *source.gravity_override;
        control_gravity_ = gravity;
    } else {
        switch (entry.controls) {
        case GalleryControlPolicy::rigid_gravity:
            gravity = screen_space_gravity(source.camera, right, forward,
                gravity_magnitude * scene.gravity_scale,
                rigid_gravity_tilt_degrees);
            break;
        case GalleryControlPolicy::collector_gravity:
            gravity = screen_space_gravity(source.camera, right, forward,
                gravity_magnitude * scene.gravity_scale,
                collector_gravity_tilt_degrees);
            break;
        case GalleryControlPolicy::peg_gravity:
            control_gravity_ = steer_gravity(
                control_gravity_, source.camera, right, forward,
                gravity_magnitude * scene.gravity_scale,
                peg_paint_gravity_tilt_degrees, options.timestep);
            gravity = control_gravity_;
            break;
        case GalleryControlPolicy::cloth_gravity:
            control_gravity_ = steer_gravity(
                control_gravity_, source.camera, right, forward,
                gravity_magnitude * scene.gravity_scale,
                deformable_gravity_tilt_degrees, options.timestep);
            gravity = control_gravity_;
            break;
        case GalleryControlPolicy::motor_drive:
        case GalleryControlPolicy::authored_force:
        case GalleryControlPolicy::none:
            break;
        }
    }
    options.gravity = gravity;

    if (entry.controls == GalleryControlPolicy::collector_gravity) {
        options.collect_rigid_contacts = true;
        options.read_host = true;
        if (!fixed_collector.apply_loose_gravity(
                simulation, scene, instance, vertical, gravity, error))
            return false;
    }
    if (entry.controls == GalleryControlPolicy::authored_force &&
        !arrow_forces.apply(
            simulation, source.camera, right, forward, error)) return false;

    if (!gravity_tilt_bodies.empty()) {
        const Vec3 compensation{vertical.x - gravity.x,
                                vertical.y - gravity.y,
                                vertical.z - gravity.z};
        if ((compensation.x != 0.0F || compensation.y != 0.0F ||
             compensation.z != 0.0F) &&
            !apply_acceleration(
                *this, gravity_tilt_bodies, compensation, error)) return false;
    }

    if (!step(options, error)) return false;
    if (entry.controls == GalleryControlPolicy::collector_gravity &&
        !fixed_collector.collect(
            simulation, scene, instance, frame, error)) return false;
    return true;
}

GalleryControllerState GalleryRuntime::controller_state() const noexcept {
    const GalleryEntry &entry = gallery_entry(context);
    return {
        .directional = entry.controls,
        .primary = entry.primary_action,
        .constraints_released = constraints_released_,
        .authored_force = arrow_forces.active(),
        .moves_kinematic = entry.moves_kinematic &&
            kinematic_index < instance.rigid_bodies.size(),
        .steering_angle = steering_angle,
        .dump_bed_angle = dump_bed.angle(),
        .dump_bed_raised = dump_bed.raised(),
        .gravity = control_gravity_,
    };
}

} // namespace parallel_mater::gallery
