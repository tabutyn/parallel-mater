// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_scene/scene.hpp>
#include <parallel_mater_scene/gpu_frame.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

// Stable gallery bindings are public-API slot indices, never backend handles.
struct CommandSceneInstance {
    struct PaintBinding {
        std::uint32_t body_index{PM_INVALID_INDEX};
        std::uint32_t mesh_index{};
        std::uint32_t field_index{};
    };

    std::vector<std::uint32_t> rigid_bodies;
    std::vector<std::uint32_t> rigid_constraints;
    std::vector<std::uint32_t> cloths;
    std::vector<std::uint32_t> ropes;
    std::vector<std::uint32_t> soft_bodies;
    std::vector<PaintBinding> paint_bindings;
    std::uint32_t fluid{PM_INVALID_INDEX};
    std::uint32_t smoke{PM_INVALID_INDEX};
    GpuSceneBuffers render_buffers;
};

struct CommandSceneFrame {
    std::vector<PmRigidState> rigid_states;
    std::vector<PmRigidState> previous_rigid_states;
    std::vector<PmFluidParticleOutput> fluid_particles;
    std::vector<std::vector<PmClothVertexState>> cloth_vertices;
    std::vector<std::vector<PmFloat3>> cloth_surface_vertices;
    std::vector<std::vector<PmRopeNodeState>> rope_nodes;
    std::vector<std::vector<PmSoftBodyNodeState>> soft_body_nodes;
    std::vector<std::vector<PmFloat3>> soft_body_surface_vertices;
    std::vector<PmSmokeParticleState> smoke_particles;
    std::vector<std::vector<std::uint32_t>> paint_fields;
    std::vector<PmConstraintState> constraint_states;
    std::vector<PmContactEvent> contacts;
    std::vector<PmDebugRecord> debug_records;
    PmStepStatistics statistics{};
    std::uint64_t revision{};
    bool full_host_output{};
};

// Gallery client for the public command API. It owns only pm* resources and
// records one command stream for scene creation, stepping, updates and output.
class CommandScene {
  public:
    CommandScene() noexcept;
    ~CommandScene();
    CommandScene(CommandScene &&) noexcept;
    CommandScene &operator=(CommandScene &&) noexcept;
    CommandScene(const CommandScene &) = delete;
    CommandScene &operator=(const CommandScene &) = delete;

    // creation_category distinguishes unsupported advertised scene capabilities
    // from general creation failure. Detailed diagnostics remain in error.
    [[nodiscard]] static bool create(const SceneDefinition &scene,
                                     std::uint32_t debug_frame_capacity,
                                     CommandScene &output,
                                     CommandSceneInstance &instance,
                                     std::string &error,
                                     std::optional<PmNativeBackend> backend =
                                         std::nullopt,
                                     PmResult *creation_category = nullptr);

    [[nodiscard]] bool step(float timestep, std::uint32_t substeps,
                            Vec3 gravity, CommandSceneFrame &frame,
                            std::string &error,
                            bool collect_contacts = false,
                            bool read_deformables = true, bool read_host = true);
    // Explicit diagnostics/capture readback, including while the viewer is paused.
    [[nodiscard]] bool read_output(CommandSceneFrame &frame, std::string &error,
                                   bool read_deformables = true);
    [[nodiscard]] bool set_rigid_state(std::uint32_t body,
                                       const RigidBodyState &state,
                                       std::string &error);
    [[nodiscard]] bool set_kinematic_target(std::uint32_t body,
                                            const RigidBodyState &state,
                                            std::string &error);
    [[nodiscard]] bool apply_rigid_inputs(const std::vector<PmRigidInput> &inputs,
                                          std::string &error);
    [[nodiscard]] bool write_constraint(
        std::uint32_t constraint, const RigidConstraintDefinition &definition,
        std::string &error);
    [[nodiscard]] bool set_constraint_active(std::uint32_t constraint,
                                             bool active,
                                             std::string &error);
    [[nodiscard]] bool reset(std::string &error);

    [[nodiscard]] PmDevice device() const noexcept;
    [[nodiscard]] PmPhysicalDevice physical_device() const noexcept;
    [[nodiscard]] PmSimulation simulation() const noexcept;
    [[nodiscard]] PmQueue queue() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
