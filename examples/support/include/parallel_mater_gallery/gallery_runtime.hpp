// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/physics_debug.hpp>
#include <parallel_mater_gallery/renderer_adapter.hpp>
#include <parallel_mater_gallery/render_scene.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

struct GalleryStepOptions {
    float timestep{1.0F / 60.0F};
    std::uint32_t substeps{4U};
    Vec3 gravity{0.0F, -9.81F, 0.0F};
    bool collect_rigid_contacts{};
    bool collect_fluid_contacts{};
    std::uint32_t rigid_contact_pass_limit{};
    bool read_deformables{true};
    bool read_host{true};
};

// Platform-neutral input for one physics step. Positive forward means screen
// up. Scripted/headless clients may supply an exact gravity while retaining the
// same motor, force, dump-bed, collector, and kinematic controller path.
struct GalleryControlInput {
    float right{};
    float forward{};
    Camera camera{};
    std::optional<Vec3> gravity_override{};
};

enum class GalleryActionOutcome : std::uint8_t {
    ignored,
    applied,
    reset_requested,
};

struct GalleryControllerState {
    GalleryControlPolicy directional{GalleryControlPolicy::none};
    GalleryPrimaryAction primary{GalleryPrimaryAction::none};
    bool constraints_released{};
    bool authored_force{};
    bool moves_kinematic{};
    float steering_angle{};
    float dump_bed_angle{};
    bool dump_bed_raised{};
    Vec3 gravity{};
};

struct SuspensionSteeringJoint {
    std::size_t constraint_index{};
    Quaternion authored_local_orientation_a{};
    bool front{};
};

struct GravityTiltBody {
    std::uint32_t body{};
    float mass{};
};

enum class GalleryReplacementPolicy : std::uint8_t {
    preserve_until_ready,
    release_before_device_create,
};

class GalleryRuntime;

struct GalleryRuntimeCreateInfo {
    GalleryContext context{GalleryContext::rigid_body};
    std::filesystem::path asset;
    std::optional<PmNativeBackend> backend;
    std::uint32_t debug_frame_capacity{};
    std::size_t capture_history_capacity{};
    std::uint32_t width{1U};
    std::uint32_t height{1U};
    bool create_host_renderer{true};
    GalleryReplacementPolicy replacement{
        GalleryReplacementPolicy::preserve_until_ready};
    std::function<bool(SceneDefinition &, std::string &)> configure_scene;
    std::function<bool(GalleryRuntime &, std::string &)> prepare_runtime;
};

// Stable, reusable scene owner for every Gallery shell. Configuration and GLB
// loading finish before a memory-constrained replacement releases its old
// device. Once device creation begins, failure leaves either the preserved old
// scene or an explicit empty/retryable state according to replacement policy.
class GalleryRuntime {
  private:
    struct LifetimeToken;
    std::unique_ptr<LifetimeToken> lifetime_;
    std::optional<GalleryRuntimeCreateInfo> recipe_;
    bool constraints_released_{};
    Vec3 control_gravity_{};

    [[nodiscard]] bool initialize_controllers(std::string &error);

  public:
    GalleryRuntime() noexcept;
    ~GalleryRuntime();
    GalleryRuntime(GalleryRuntime &&) noexcept;
    GalleryRuntime &operator=(GalleryRuntime &&) noexcept;
    GalleryRuntime(const GalleryRuntime &) = delete;
    GalleryRuntime &operator=(const GalleryRuntime &) = delete;

    [[nodiscard]] bool replace(GalleryRuntimeCreateInfo info,
                               std::string &error);
    [[nodiscard]] bool reset(std::string &error);
    void clear() noexcept;
    [[nodiscard]] bool step(const GalleryStepOptions &options,
                            std::string &error);
    [[nodiscard]] bool controlled_step(const GalleryControlInput &input,
                                       GalleryStepOptions options,
                                       std::string &error);
    [[nodiscard]] GalleryActionOutcome primary_action(std::string &error);
    [[nodiscard]] GalleryControllerState controller_state() const noexcept;
    [[nodiscard]] bool ready() const noexcept { return lifetime_ != nullptr; }
    [[nodiscard]] static std::size_t live_runtime_count() noexcept;

    GalleryContext context{GalleryContext::rigid_body};
    SceneDefinition scene{};
    CommandScene simulation{};
    CommandSceneInstance instance{};
    CommandSceneFrame frame{};
    PhysicsDebugHistory capture_history{};
    RenderSceneFrame render_frame{};
    RendererHandle renderer{};
    std::unique_ptr<GpuGalleryRenderer> native_renderer{};
    bool gpu_ready{};
    std::uint64_t gpu_frame_generation{};
    CameraController camera{};
    FixedContactCollector fixed_collector{};
    ArrowForces arrow_forces{};
    DumpTruckBed dump_bed{};
    std::vector<SuspensionSteeringJoint> suspension_steering_joints{};
    float steering_angle{};
    std::vector<GravityTiltBody> gravity_tilt_bodies{};
    std::size_t kinematic_index{static_cast<std::size_t>(-1)};
    RigidBodyState kinematic_target{};
};

} // namespace parallel_mater::gallery
