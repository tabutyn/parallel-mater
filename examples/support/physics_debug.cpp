// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/physics_debug.hpp>
#include <scene_builder.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <new>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace parallel_mater::gallery {
namespace {

constexpr std::uint32_t capture_format_version = 5U;
constexpr std::size_t capture_history_budget = 32U * 1024U * 1024U;

std::size_t sample_bytes(const RigidPhysicsSample &sample) {
    return sizeof(sample) + sample.bodies.size() * sizeof(PmRigidState) +
        sample.constraints.size() * sizeof(PmConstraintState) +
        sample.contacts.size() * sizeof(PmContactEvent);
}

void write_rigid_states(std::ostream &stream,
                        const std::vector<PmRigidState> &states) {
    for (std::size_t index = 0; index < states.size(); ++index) {
        const auto &state = states[index];
        stream << "rigid " << index << " position";
        for (float value : state.position) stream << ' ' << value;
        stream << " orientation";
        for (float value : state.orientation) stream << ' ' << value;
        stream << " linear_velocity";
        for (float value : state.linearVelocity) stream << ' ' << value;
        stream << " angular_velocity";
        for (float value : state.angularVelocity) stream << ' ' << value;
        stream << '\n';
    }
}

void write_constraint_states(std::ostream &stream,
                             const std::vector<PmConstraintState> &states) {
    for (std::size_t index = 0; index < states.size(); ++index) {
        const auto &state = states[index];
        stream << "constraint " << index << " enabled " << state.enabled
               << " broken " << state.broken << " applied_impulse "
               << state.appliedImpulse << '\n';
    }
}

void write_contacts(std::ostream &stream,
                    const std::vector<PmContactEvent> &contacts) {
    for (const auto &contact : contacts)
        stream << "contact " << static_cast<unsigned>(contact.type)
               << ' ' << contact.first << ' ' << contact.second
               << " position " << contact.position.x << ' '
               << contact.position.y << ' ' << contact.position.z
               << " normal " << contact.normal.x << ' '
               << contact.normal.y << ' ' << contact.normal.z
               << " penetration " << contact.penetration
               << " normal_impulse " << contact.normalImpulse << '\n';
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

[[nodiscard]] bool parse_rigid(std::istringstream &line,
                               RigidPhysicsSample &sample) {
    std::size_t index{};
    std::string position, orientation, linear_velocity, angular_velocity;
    PmRigidState state{};
    if (!(line >> index >> position) || position != "position") return false;
    for (float &value : state.position) if (!(line >> value)) return false;
    if (!(line >> orientation) || orientation != "orientation") return false;
    for (float &value : state.orientation) if (!(line >> value)) return false;
    if (!(line >> linear_velocity) || linear_velocity != "linear_velocity")
        return false;
    for (float &value : state.linearVelocity) if (!(line >> value)) return false;
    if (!(line >> angular_velocity) || angular_velocity != "angular_velocity")
        return false;
    for (float &value : state.angularVelocity) if (!(line >> value)) return false;
    if (index != sample.bodies.size()) return false;
    sample.bodies.push_back(state);
    return true;
}

[[nodiscard]] bool parse_constraint(std::istringstream &line,
                                    RigidPhysicsSample &sample) {
    std::size_t index{};
    std::string enabled, broken, impulse;
    PmConstraintState state{};
    if (!(line >> index >> enabled >> state.enabled >> broken >> state.broken >>
          impulse >> state.appliedImpulse) ||
        enabled != "enabled" || broken != "broken" ||
        impulse != "applied_impulse" || index != sample.constraints.size())
        return false;
    sample.constraints.push_back(state);
    return true;
}

[[nodiscard]] bool parse_contact(std::istringstream &line,
                                 RigidPhysicsSample &sample) {
    unsigned type{};
    std::string position, normal, penetration, impulse;
    PmContactEvent contact{};
    if (!(line >> type >> contact.first >> contact.second >> position >>
          contact.position.x >> contact.position.y >> contact.position.z >>
          normal >> contact.normal.x >> contact.normal.y >> contact.normal.z >>
          penetration >> contact.penetration >> impulse >>
          contact.normalImpulse) ||
        position != "position" || normal != "normal" ||
        penetration != "penetration" || impulse != "normal_impulse")
        return false;
    contact.type = static_cast<PmContactEventType>(type);
    sample.contacts.push_back(contact);
    return true;
}

[[nodiscard]] bool parse_frame_header(std::istringstream &line,
                                      RigidPhysicsSample &sample,
                                      std::size_t &rigid_count,
                                      std::size_t &constraint_count,
                                      std::size_t &contact_count) {
    std::string rigid, constraints, fluid, cloth, soft_body, rope, smoke,
        contacts;
    std::size_t fluid_count{}, cloth_count{}, soft_body_count{}, rope_count{},
        smoke_count{};
    return bool(line >> sample.revision >> rigid >> rigid_count >> constraints >>
                constraint_count >> fluid >> fluid_count >> cloth >> cloth_count >>
                soft_body >> soft_body_count >> rope >> rope_count >> smoke >>
                smoke_count >> contacts >> contact_count) &&
        rigid == "rigid" && constraints == "constraints" &&
        fluid == "fluid" && cloth == "cloth" && soft_body == "soft_body" &&
        rope == "rope" && smoke == "smoke" && contacts == "contacts";
}

[[nodiscard]] RigidBodyState convert(const PmRigidState &state) noexcept {
    return {{state.position[0], state.position[1], state.position[2]},
            {state.orientation[0], state.orientation[1], state.orientation[2],
             state.orientation[3]},
            {state.linearVelocity[0], state.linearVelocity[1],
             state.linearVelocity[2]},
            {state.angularVelocity[0], state.angularVelocity[1],
             state.angularVelocity[2]}};
}

} // namespace

std::vector<std::string> physics_rigid_body_ids(const SceneDefinition &scene) {
    std::unordered_map<std::string, std::size_t> source_counts;
    for (const auto &body : scene.rigid_bodies)
        if (!body.source_name.empty()) ++source_counts[body.source_name];
    std::vector<std::string> ids;
    ids.reserve(scene.rigid_bodies.size());
    for (const auto &body : scene.rigid_bodies) {
        const bool unique_source = !body.source_name.empty() &&
            source_counts[body.source_name] == 1U;
        ids.push_back(unique_source ? body.source_name : body.name);
    }
    return ids;
}

bool PhysicsDebugHistory::record(const CommandSceneFrame &frame,
                                 const PmStepInfo &step, std::string &error) {
    if (!enabled()) return true;
    try {
        RigidPhysicsSample sample{frame.revision, step, frame.rigid_states,
                                  frame.constraint_states, frame.contacts};
        if (!frames_.empty() && frame.revision <= frames_.back().revision) {
            frames_.clear();
            bytes_ = 0U;
            complete_ = true;
        }
        bytes_ += sample_bytes(sample);
        frames_.push_back(std::move(sample));
        while (frames_.size() > capacity_ || bytes_ > capture_history_budget) {
            bytes_ -= sample_bytes(frames_.front());
            frames_.pop_front();
            complete_ = false;
        }
    } catch (const std::bad_alloc &) {
        error = "could not retain physics capture history";
        return false;
    }
    return true;
}

bool write_physics_debug_capture(const CommandSceneFrame &frame,
                                 const std::filesystem::path &output,
                                 std::string &error,
                                 const PhysicsDebugHistory *history,
                                 const PhysicsCaptureProvenance *provenance) {
    error.clear();
    if (provenance != nullptr &&
        provenance->rigid_body_ids.size() != frame.rigid_states.size()) {
        error = "physics capture rigid body identities do not match the frame";
        return false;
    }
    if (provenance != nullptr) {
        std::unordered_set<std::string> unique;
        for (const auto &id : provenance->rigid_body_ids) {
            if (id.empty() || !unique.insert(id).second) {
                error = "physics capture rigid body identities must be nonempty and unique";
                return false;
            }
        }
    }
    std::ofstream stream(output, std::ios::out | std::ios::trunc);
    if (!stream) {
        error = "could not open physics capture log";
        return false;
    }
    const PhysicsCaptureProvenance empty;
    const auto &source = provenance == nullptr ? empty : *provenance;
    stream << std::setprecision(9)
           << "parallel_mater_physics_capture " << capture_format_version << '\n'
           << "provenance scene " << std::quoted(source.scene)
           << " backend " << std::quoted(source.backend)
           << " device " << std::quoted(source.device)
           << " build " << std::quoted(source.build_revision) << '\n'
           << "rigid_ids " << source.rigid_body_ids.size();
    for (const auto &id : source.rigid_body_ids) stream << ' ' << std::quoted(id);
    stream << '\n' << "history_complete "
           << (history != nullptr && history->complete() ? 1 : 0) << '\n'
           << "rigid_history " << (history == nullptr ? 0U : history->frames().size())
           << '\n';
    if (history != nullptr) {
        for (const auto &sample : history->frames()) {
            stream << "history_frame " << sample.revision
                   << " timestep " << sample.step.timestep
                   << " substeps " << sample.step.substeps << " gravity "
                   << sample.step.gravity[0] << ' ' << sample.step.gravity[1]
                   << ' ' << sample.step.gravity[2] << " flags "
                   << sample.step.flags << '\n';
            write_rigid_states(stream, sample.bodies);
            write_constraint_states(stream, sample.constraints);
            write_contacts(stream, sample.contacts);
            stream << "end_history_frame\n";
        }
    }
    stream << "frame " << frame.revision
           << " rigid " << frame.rigid_states.size()
           << " constraints " << frame.constraint_states.size()
           << " fluid " << frame.fluid_particles.size()
           << " cloth " << frame.cloth_vertices.size()
           << " soft_body " << frame.soft_body_nodes.size()
           << " rope " << frame.rope_nodes.size()
           << " smoke " << frame.smoke_particles.size()
           << " contacts " << frame.contacts.size() << '\n';
    write_rigid_states(stream, frame.rigid_states);
    write_constraint_states(stream, frame.constraint_states);
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
    write_contacts(stream, frame.contacts);
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

bool read_physics_debug_capture(const std::filesystem::path &input,
                                PhysicsDebugCapture &capture,
                                std::string &error) {
    error.clear();
    capture = {};
    std::ifstream stream(input);
    std::string line_text;
    if (!stream || !std::getline(stream, line_text)) {
        error = "could not open physics capture log";
        return false;
    }
    {
        std::istringstream line(line_text);
        std::string magic;
        if (!(line >> magic >> capture.format_version) ||
            magic != "parallel_mater_physics_capture" ||
            capture.format_version < 3U || capture.format_version > 5U) {
            error = "unsupported physics capture format";
            return false;
        }
    }
    RigidPhysicsSample *current = nullptr;
    std::size_t history_count_expected{}, rigid_count_expected{},
        constraint_count_expected{}, contact_count_expected{};
    bool saw_provenance = false, saw_rigid_ids = false,
        saw_history_complete = false, saw_history_count = false,
        saw_frame_end = false;
    while (std::getline(stream, line_text)) {
        if (line_text.empty()) continue;
        std::istringstream line(line_text);
        std::string kind;
        line >> kind;
        bool valid = true;
        if (kind == "provenance" && capture.format_version >= 5U) {
            std::string scene, backend, device, build;
            valid = bool(line >> scene >> std::quoted(capture.provenance.scene) >>
                         backend >> std::quoted(capture.provenance.backend) >>
                         device >> std::quoted(capture.provenance.device) >>
                         build >> std::quoted(capture.provenance.build_revision)) &&
                scene == "scene" && backend == "backend" &&
                device == "device" && build == "build";
            valid = valid && !saw_provenance;
            saw_provenance = valid;
        } else if (kind == "rigid_ids" && capture.format_version >= 5U) {
            std::size_t count{};
            valid = !saw_rigid_ids && bool(line >> count);
            for (std::size_t i = 0; valid && i < count; ++i) {
                std::string id;
                valid = bool(line >> std::quoted(id));
                if (valid) capture.provenance.rigid_body_ids.push_back(std::move(id));
            }
            saw_rigid_ids = valid;
        } else if (kind == "history_complete" && capture.format_version >= 5U) {
            unsigned complete{};
            valid = !saw_history_complete && bool(line >> complete) &&
                complete <= 1U;
            capture.history_complete = complete != 0U;
            saw_history_complete = valid;
        } else if (kind == "rigid_history" && capture.format_version >= 4U) {
            valid = !saw_history_count && bool(line >> history_count_expected);
            saw_history_count = valid;
        } else if (kind == "history_frame" && capture.format_version >= 4U) {
            valid = current == nullptr;
            capture.history.emplace_back();
            current = &capture.history.back();
            std::string timestep, substeps, gravity, flags;
            valid = valid && bool(line >> current->revision >> timestep >>
                         current->step.timestep >> substeps >>
                         current->step.substeps >> gravity >>
                         current->step.gravity[0] >> current->step.gravity[1] >>
                         current->step.gravity[2]) &&
                timestep == "timestep" && substeps == "substeps" &&
                gravity == "gravity";
            if (valid && line >> flags)
                valid = flags == "flags" && bool(line >> current->step.flags);
        } else if (kind == "end_history_frame") {
            valid = current != nullptr && current != &capture.frame;
            current = nullptr;
        } else if (kind == "frame") {
            valid = current == nullptr && !capture.has_frame;
            capture.frame = {};
            current = &capture.frame;
            valid = valid && parse_frame_header(
                line, capture.frame, rigid_count_expected,
                constraint_count_expected, contact_count_expected);
            capture.has_frame = valid;
        } else if (kind == "rigid" && current != nullptr) {
            valid = parse_rigid(line, *current);
        } else if (kind == "constraint" && current != nullptr) {
            valid = parse_constraint(line, *current);
        } else if (kind == "contact" && current != nullptr) {
            valid = parse_contact(line, *current);
        } else if (kind == "end_frame") {
            valid = current == &capture.frame &&
                capture.frame.bodies.size() == rigid_count_expected &&
                capture.frame.constraints.size() == constraint_count_expected &&
                capture.frame.contacts.size() == contact_count_expected;
            current = nullptr;
            saw_frame_end = valid;
        } else if (current == &capture.frame &&
                   (kind == "fluid" || kind == "debug")) {
            // Non-rigid domains remain diagnostic text until versioned replay
            // schemas are specified for each domain.
        } else {
            valid = false;
        }
        if (!valid) {
            error = "invalid physics capture line: " + line_text;
            capture = {};
            return false;
        }
    }
    if (current != nullptr || !capture.has_frame || !saw_frame_end ||
        (capture.format_version >= 4U &&
         (!saw_history_count || capture.history.size() != history_count_expected)) ||
        (capture.format_version >= 5U &&
         (!saw_provenance || !saw_rigid_ids || !saw_history_complete))) {
        error = "truncated physics capture log";
        capture = {};
        return false;
    }
    return true;
}

bool replay_rigid_physics_sample(
    CommandScene &simulation, const CommandSceneInstance &instance,
    const SceneDefinition &scene, const RigidPhysicsSample &sample,
    std::span<const std::string> rigid_body_ids, PhysicsReplayPolicy policy,
    std::string &error) {
    error.clear();
    if (rigid_body_ids.empty() || rigid_body_ids.size() != sample.bodies.size()) {
        error = "physics replay requires one stable identity per rigid state";
        return false;
    }
    if (instance.rigid_bodies.size() != scene.rigid_bodies.size()) {
        error = "physics replay scene bindings are incomplete";
        return false;
    }
    if (policy == PhysicsReplayPolicy::require_complete &&
        rigid_body_ids.size() != scene.rigid_bodies.size()) {
        error = "physics replay capture is partial";
        return false;
    }
    std::unordered_map<std::string, std::size_t> scene_indices;
    const auto scene_ids = physics_rigid_body_ids(scene);
    for (std::size_t index = 0; index < scene_ids.size(); ++index) {
        const std::string &id = scene_ids[index];
        if (id.empty() || !scene_indices.emplace(id, index).second) {
            error = "physics replay scene identities must be nonempty and unique";
            return false;
        }
    }
    std::unordered_set<std::string> seen;
    std::vector<std::size_t> destinations;
    destinations.reserve(rigid_body_ids.size());
    for (const auto &id : rigid_body_ids) {
        const auto found = scene_indices.find(id);
        if (id.empty() || !seen.insert(id).second || found == scene_indices.end()) {
            error = "physics replay capture has a duplicate or unknown rigid identity";
            return false;
        }
        destinations.push_back(found->second);
    }
    for (std::size_t index = 0; index < sample.bodies.size(); ++index) {
        if (!simulation.set_rigid_state(
                instance.rigid_bodies[destinations[index]],
                convert(sample.bodies[index]), error)) {
            if (error.empty()) error = "could not apply physics replay state";
            return false;
        }
    }
    return true;
}

bool save_physics_debug_capture(const CommandSceneFrame &frame,
                                std::filesystem::path &output,
                                std::string &error,
                                const PhysicsDebugHistory *history,
                                const PhysicsCaptureProvenance *provenance) {
    output.clear();
    if (!default_capture_path(frame.revision, output, error)) return false;
    return write_physics_debug_capture(frame, output, error, history, provenance);
}

} // namespace parallel_mater::gallery
