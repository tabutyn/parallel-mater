// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

class CommandScene;
struct CommandSceneInstance;
struct CommandSceneFrame;
struct SceneDefinition;

struct PhysicsCaptureProvenance {
    std::string scene;
    std::string backend;
    std::string device;
    std::string build_revision;
    std::vector<std::string> rigid_body_ids;
};

struct RigidPhysicsSample {
    std::uint64_t revision{};
    PmStepInfo step{};
    std::vector<PmRigidState> bodies;
    std::vector<PmConstraintState> constraints;
    std::vector<PmContactEvent> contacts;
};

struct PhysicsDebugCapture {
    std::uint32_t format_version{};
    PhysicsCaptureProvenance provenance;
    std::vector<RigidPhysicsSample> history;
    RigidPhysicsSample frame;
    bool has_frame{};
    bool history_complete{};
};

[[nodiscard]] std::vector<std::string> physics_rigid_body_ids(
    const SceneDefinition &scene);

enum class PhysicsReplayPolicy : std::uint8_t {
    require_complete,
    allow_partial,
};

// Bounded history contains only host-visible public API output. Eviction or a
// GPU-only step marks the sequence partial instead of silently claiming a
// continuous replay.
class PhysicsDebugHistory {
  public:
    explicit PhysicsDebugHistory(std::size_t capacity = 0U)
        : capacity_(capacity) {}
    [[nodiscard]] bool record(const CommandSceneFrame &frame,
                              const PmStepInfo &step, std::string &error);
    void mark_gap() noexcept { complete_ = false; }
    [[nodiscard]] bool enabled() const noexcept { return capacity_ != 0U; }
    [[nodiscard]] bool complete() const noexcept { return complete_; }
    [[nodiscard]] const std::deque<RigidPhysicsSample> &frames() const {
        return frames_;
    }

  private:
    std::size_t capacity_{};
    std::size_t bytes_{};
    bool complete_{true};
    std::deque<RigidPhysicsSample> frames_;
};

// Version 5 adds quoted provenance, stable rigid-body identities, and explicit
// history completeness. The reader accepts legacy v3 snapshots and archived v4
// rolling histories, but identity-free captures cannot be replayed safely.
[[nodiscard]] bool read_physics_debug_capture(
    const std::filesystem::path &input, PhysicsDebugCapture &capture,
    std::string &error);

[[nodiscard]] bool replay_rigid_physics_sample(
    CommandScene &simulation, const CommandSceneInstance &instance,
    const SceneDefinition &scene,
    const RigidPhysicsSample &sample,
    std::span<const std::string> rigid_body_ids,
    PhysicsReplayPolicy policy, std::string &error);

[[nodiscard]] bool save_physics_debug_capture(
    const CommandSceneFrame &frame, std::filesystem::path &output,
    std::string &error, const PhysicsDebugHistory *history = nullptr,
    const PhysicsCaptureProvenance *provenance = nullptr);

[[nodiscard]] bool write_physics_debug_capture(
    const CommandSceneFrame &frame, const std::filesystem::path &output,
    std::string &error, const PhysicsDebugHistory *history = nullptr,
    const PhysicsCaptureProvenance *provenance = nullptr);

} // namespace parallel_mater::gallery
