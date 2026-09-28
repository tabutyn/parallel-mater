// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/physics_debug.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <system_error>

namespace parallel_mater::gallery {
namespace {

void vector(std::ostream &stream, Vec3 value) {
    stream << value.x << ' ' << value.y << ' ' << value.z;
}

void quaternion(std::ostream &stream, Quaternion value) {
    stream << value.x << ' ' << value.y << ' ' << value.z << ' ' << value.w;
}

[[nodiscard]] bool default_capture_path(std::uint64_t frame,
                                        std::filesystem::path &output,
                                        std::string &error) {
    std::error_code filesystem_error;
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path(filesystem_error);
    if (filesystem_error) {
        error = "could not find the temporary directory: " +
                filesystem_error.message();
        return false;
    }
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    output = temporary /
        ("parallel-mater-physics-" + std::to_string(stamp) + "-frame-" +
         std::to_string(frame) + ".log");
    return true;
}

} // namespace

bool write_physics_debug_capture(World &world,
                                 const std::filesystem::path &output,
                                 std::string &error) {
    error.clear();
    PhysicsDebugCapture capture{};
    const Status status = world.copy_physics_debug_capture(capture);
    if (!status) {
        error = status.message != nullptr ? status.message
                                          : "physics capture copy failed";
        return false;
    }
    std::ofstream stream(output, std::ios::out | std::ios::trunc);
    if (!stream) {
        error = "could not open physics capture log";
        return false;
    }
    stream << std::setprecision(9);
    stream << "parallel_mater_physics_capture 1\n";
    stream << "frames " << capture.frames.size() << '\n';
    for (const PhysicsDebugFrame &frame : capture.frames) {
        stream << "frame " << frame.frame_index << " timestep "
               << frame.timestep << " gravity ";
        vector(stream, frame.gravity);
        stream << " max_fluid_neighbors "
               << frame.maximum_fluid_neighbor_count << " rigid "
               << frame.rigid_bodies.size() << " fluid "
               << frame.fluid_particles.size() << " cloth "
               << frame.cloth_vertices.size() << " soft_body "
               << frame.soft_body_nodes.size() << " rigid_contacts "
               << frame.rigid_contacts.size() << " fluid_contacts "
               << frame.fluid_contacts.size() << '\n';
        for (const PhysicsDebugRigidSample &sample : frame.rigid_bodies) {
            stream << "rigid " << sample.id.index << ' '
                   << sample.id.generation << " position ";
            vector(stream, sample.state.position);
            stream << " orientation ";
            quaternion(stream, sample.state.orientation);
            stream << " linear_velocity ";
            vector(stream, sample.state.linear_velocity);
            stream << " angular_velocity ";
            vector(stream, sample.state.angular_velocity);
            stream << " applied_force ";
            vector(stream, sample.applied_force);
            stream << " applied_torque ";
            vector(stream, sample.applied_torque);
            stream << '\n';
        }
        for (const PhysicsDebugFluidSample &sample : frame.fluid_particles) {
            stream << "fluid " << sample.fluid.index << ' '
                   << sample.fluid.generation << ' '
                   << sample.stable_particle_id << " position ";
            vector(stream, sample.position);
            stream << " velocity ";
            vector(stream, sample.velocity);
            stream << " acceleration ";
            vector(stream, sample.acceleration);
            stream << " foam " << sample.foam << '\n';
        }
        for (const PhysicsDebugClothSample &sample : frame.cloth_vertices) {
            stream << "cloth " << sample.cloth.index << ' '
                   << sample.cloth.generation << ' ' << sample.vertex
                   << " position ";
            vector(stream, sample.position);
            stream << " velocity ";
            vector(stream, sample.velocity);
            stream << " rigid_contact_force ";
            vector(stream, sample.rigid_contact_force);
            stream << " fluid_contact_force ";
            vector(stream, sample.fluid_contact_force);
            stream << " soft_body_contact_force ";
            vector(stream, sample.soft_body_contact_force);
            stream << '\n';
        }
        for (const PhysicsDebugSoftBodySample &sample : frame.soft_body_nodes) {
            stream << "soft_body " << sample.soft_body.index << ' '
                   << sample.soft_body.generation << ' ' << sample.node
                   << " position ";
            vector(stream, sample.position);
            stream << " velocity ";
            vector(stream, sample.velocity);
            stream << " rigid_contact_force ";
            vector(stream, sample.rigid_contact_force);
            stream << " cloth_contact_force ";
            vector(stream, sample.cloth_contact_force);
            stream << '\n';
        }
        for (const RigidContactEvent &contact : frame.rigid_contacts) {
            stream << "rigid_contact " << contact.body.index << ' '
                   << contact.body.generation << ' ' << contact.collider.index
                   << ' ' << contact.collider.generation << " position ";
            vector(stream, contact.position);
            stream << " normal ";
            vector(stream, contact.normal);
            stream << " penetration " << contact.penetration
                   << " normal_impulse " << contact.normal_impulse
                   << " friction_impulse ";
            vector(stream, contact.friction_impulse);
            stream << '\n';
        }
        for (const ContactEvent &contact : frame.fluid_contacts) {
            stream << "fluid_contact " << contact.fluid.index << ' '
                   << contact.fluid.generation << ' '
                   << contact.stable_particle_id << ' '
                   << contact.rigid_body.index << ' '
                   << contact.rigid_body.generation << " position ";
            vector(stream, contact.position);
            stream << " normal ";
            vector(stream, contact.normal);
            stream << " normal_impulse " << contact.normal_impulse << '\n';
        }
        stream << "end_frame\n";
    }
    if (!stream) {
        error = "could not finish physics capture log";
        return false;
    }
    return true;
}

bool save_physics_debug_capture(World &world, std::filesystem::path &output,
                                std::string &error) {
    output.clear();
    PhysicsDebugFrameView frame{};
    const Status status = world.physics_debug_frame(frame);
    if (!status) {
        error = status.message != nullptr ? status.message
                                          : "physics capture is unavailable";
        return false;
    }
    if (!default_capture_path(frame.frame_index, output, error)) return false;
    return write_physics_debug_capture(world, output, error);
}

} // namespace parallel_mater::gallery
