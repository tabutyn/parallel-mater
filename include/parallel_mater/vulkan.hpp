// SPDX-License-Identifier: MIT
#pragma once

// Public Vulkan 1.2 rigid-body API. This first backend slice intentionally
// exposes only operations implemented by its compute pipeline.

#include <parallel_mater/types.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace parallel_mater::vulkan {

using ::parallel_mater::HostSpan;
using ::parallel_mater::MotionType;
using ::parallel_mater::Quaternion;
using ::parallel_mater::RigidBodyId;
using ::parallel_mater::RigidBodyOptions;
using ::parallel_mater::RigidBodyState;
using ::parallel_mater::StepOptions;
using ::parallel_mater::TriangleMeshId;
using ::parallel_mater::Vec3;
using ::parallel_mater::WorldOptions;
using ::parallel_mater::WorldStatistics;
using ::parallel_mater::WorldStepTimings;

// When supplied to borrowed World creation, every handle remains caller-owned.
// The caller keeps them alive until World and all submitted FrameTokens are
// destroyed and externally synchronizes queue submission. native_context()
// returns the same kind of non-owning view for an owned World.
struct NativeContext {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    std::uint32_t queue_family_index{};
    // Vulkan cannot query which optional features were enabled at device
    // creation. The caller explicitly promises timelineSemaphore was enabled.
    bool timeline_semaphore_enabled{};
};

template <typename T> struct BufferSpan {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceSize byte_offset{};
    std::uint64_t size{}; // Element count, not byte count.

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0U; }
};

enum class StatusCode : std::uint8_t {
    success,
    invalid_argument,
    not_supported,
    invalid_handle,
    capacity_exceeded,
    busy,
    out_of_memory,
    vulkan_failure,
    internal_error,
};

struct Status {
    StatusCode code{StatusCode::success};
    VkResult vulkan_result{VK_SUCCESS};
    const char *message{};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == StatusCode::success;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return ok();
    }
};

struct NativeCompletion {
    // Borrowed from FrameToken. Do not destroy the semaphore; keep that token
    // alive until every external GPU wait using this value has completed.
    VkSemaphore semaphore{VK_NULL_HANDLE};
    std::uint64_t value{};
};

struct RigidBodyDeviceView {
    // Buffers remain stable for World lifetime. Counts/revision describe the
    // last completed frame and must be reacquired after later completions.
    BufferSpan<const RigidBodyId> ids{};
    BufferSpan<const RigidBodyState> states{};
    BufferSpan<const RigidBodyState> previous_states{};
    std::uint64_t revision{};
};

class FrameToken {
  public:
    FrameToken() noexcept;
    ~FrameToken();
    FrameToken(FrameToken &&) noexcept;
    FrameToken &operator=(FrameToken &&) noexcept;
    FrameToken(const FrameToken &) = delete;
    FrameToken &operator=(const FrameToken &) = delete;

    [[nodiscard]] bool pending() const noexcept;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] Status wait() noexcept;
    [[nodiscard]] NativeCompletion native_completion() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class World;
};

class World {
  public:
    World() noexcept;
    ~World();
    World(World &&) noexcept;
    World &operator=(World &&) noexcept;
    World(const World &) = delete;
    World &operator=(const World &) = delete;

    // Creates a headless Vulkan 1.2 instance/device/compute queue owned by the
    // returned World. No surface or presentation extension is requested.
    [[nodiscard]] static Status create(WorldOptions options,
                                       World &output) noexcept;
    // Uses a caller-owned context. All handles must belong to one device and
    // queue family. The queue must support VK_QUEUE_COMPUTE_BIT.
    [[nodiscard]] static Status create(WorldOptions options,
                                       NativeContext context,
                                       World &output) noexcept;

    [[nodiscard]] Status add_triangle_mesh(
        HostSpan<const Vec3> vertices,
        HostSpan<const std::uint32_t> triangle_indices,
        TriangleMeshId &output) noexcept;
    [[nodiscard]] Status remove_triangle_mesh(TriangleMeshId mesh) noexcept;

    [[nodiscard]] Status add_rigid_body(RigidBodyOptions options,
                                        RigidBodyId &output) noexcept;
    [[nodiscard]] Status remove_rigid_body(RigidBodyId body) noexcept;
    [[nodiscard]] Status set_rigid_body_state(RigidBodyId body,
                                              RigidBodyState state) noexcept;
    [[nodiscard]] Status set_kinematic_target(RigidBodyId body,
                                              RigidBodyState target) noexcept;
    [[nodiscard]] Status apply_force(RigidBodyId body, Vec3 force,
                                     Vec3 world_point) noexcept;
    [[nodiscard]] Status apply_central_acceleration(
        HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept;
    [[nodiscard]] Status apply_impulse(RigidBodyId body, Vec3 impulse,
                                       Vec3 world_point) noexcept;

    [[nodiscard]] Status rigid_body_view(
        RigidBodyDeviceView &output) const noexcept;
    [[nodiscard]] Status read_rigid_body_state(
        RigidBodyId body, RigidBodyState &output) const noexcept;

    // One frame per World may be in flight. A borrowed queue must not be
    // submitted concurrently by another thread while this call executes.
    [[nodiscard]] Status step_async(StepOptions options,
                                    FrameToken &completion) noexcept;
    [[nodiscard]] Status step(StepOptions options) noexcept;

    [[nodiscard]] Status collect_step_timings(
        WorldStepTimings &output) const noexcept;
    [[nodiscard]] Status collect_statistics(
        WorldStatistics &output) const noexcept;
    [[nodiscard]] NativeContext native_context() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::vulkan
