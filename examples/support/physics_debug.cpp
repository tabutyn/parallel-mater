// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/physics_debug.hpp>
#include <scene_builder.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <system_error>

namespace parallel_mater::gallery {
namespace {

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

bool write_physics_debug_capture(const CommandSceneFrame &frame,
                                 const std::filesystem::path &output,
                                 std::string &error) {
    error.clear();
    std::ofstream stream(output, std::ios::out | std::ios::trunc);
    if (!stream) {
        error = "could not open physics capture log";
        return false;
    }
    stream << std::setprecision(9)
           << "parallel_mater_physics_capture 3\n"
           << "frame " << frame.revision
           << " rigid " << frame.rigid_states.size()
           << " constraints " << frame.constraint_states.size()
           << " fluid " << frame.fluid_particles.size()
           << " cloth " << frame.cloth_vertices.size()
           << " soft_body " << frame.soft_body_nodes.size()
           << " rope " << frame.rope_nodes.size()
           << " smoke " << frame.smoke_particles.size()
           << " contacts " << frame.contacts.size() << '\n';
    for (std::size_t index = 0; index < frame.rigid_states.size(); ++index) {
        const PmRigidState &state = frame.rigid_states[index];
        stream << "rigid " << index << " position "
               << state.position[0] << ' ' << state.position[1] << ' '
               << state.position[2] << " orientation "
               << state.orientation[0] << ' ' << state.orientation[1] << ' '
               << state.orientation[2] << ' ' << state.orientation[3]
               << " linear_velocity " << state.linearVelocity[0] << ' '
               << state.linearVelocity[1] << ' ' << state.linearVelocity[2]
               << " angular_velocity " << state.angularVelocity[0] << ' '
               << state.angularVelocity[1] << ' ' << state.angularVelocity[2]
               << '\n';
    }
    for (std::size_t index = 0; index < frame.constraint_states.size();
         ++index) {
        const PmConstraintState &state = frame.constraint_states[index];
        stream << "constraint " << index
               << " enabled " << state.enabled
               << " broken " << state.broken
               << " applied_impulse " << state.appliedImpulse << '\n';
    }
    for (const PmFluidParticleOutput &particle : frame.fluid_particles) {
        stream << "fluid " << particle.stableId << " position "
               << particle.position[0] << ' ' << particle.position[1] << ' '
               << particle.position[2] << " velocity "
               << particle.velocity[0] << ' ' << particle.velocity[1] << ' '
               << particle.velocity[2] << " acceleration "
               << particle.acceleration[0] << ' ' << particle.acceleration[1]
               << ' ' << particle.acceleration[2] << " foam "
               << particle.foam << '\n';
    }
    for (const PmContactEvent &contact : frame.contacts) {
        stream << "contact " << static_cast<unsigned>(contact.type) << ' '
               << contact.first << ' ' << contact.second << " position "
               << contact.position.x << ' ' << contact.position.y << ' '
               << contact.position.z << " normal " << contact.normal.x << ' '
               << contact.normal.y << ' ' << contact.normal.z
               << " penetration " << contact.penetration
               << " normal_impulse " << contact.normalImpulse << '\n';
    }
    for (const PmDebugRecord &record : frame.debug_records) {
        stream << "debug " << static_cast<unsigned>(record.type) << ' '
               << record.objectIndex << ' ' << record.elementIndex
               << " position " << record.position.x << ' '
               << record.position.y << ' ' << record.position.z
               << " velocity " << record.velocity.x << ' '
               << record.velocity.y << ' ' << record.velocity.z
               << " force " << record.force.x << ' ' << record.force.y << ' '
               << record.force.z << '\n';
    }
    stream << "end_frame\n";
    if (!stream) {
        error = "could not finish physics capture log";
        return false;
    }
    return true;
}

bool save_physics_debug_capture(const CommandSceneFrame &frame,
                                std::filesystem::path &output,
                                std::string &error) {
    output.clear();
    if (!default_capture_path(frame.revision, output, error)) return false;
    return write_physics_debug_capture(frame, output, error);
}

} // namespace parallel_mater::gallery
