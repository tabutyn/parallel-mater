// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_scene/gpu_frame.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

class FrameTimingCollector;
struct CommandSceneInstance;

// Complete backend-neutral boundary between the Gallery runtime and a native
// graphics adapter. Native handles come only from the public interop API.
struct GpuRendererDevice {
    PmDevice api_device{};
    PmQueue physics_queue{};
    PmNativeDeviceInfo native{};
    PmNativeQueueInfo native_queue{};
    std::uint64_t resource_generation{};
};

struct GpuRendererBuffer {
    PmDevice api_device{};
    PmBuffer api_buffer{};
    PmNativeDeviceInfo native_device{};
    PmNativeBufferInfo native{};
    std::uint64_t resource_generation{};
};

enum class GpuSynchronizationMode : std::uint8_t {
    queue_idle,
    timeline_semaphore,
};

enum class GpuBufferOwner : std::uint8_t {
    physics,
    graphics,
};

struct GpuFrameSynchronization {
    GpuSynchronizationMode mode{GpuSynchronizationMode::queue_idle};
    PmQueue physics_queue{};
    PmNativeHandle wait_semaphore{};
    std::uint64_t wait_value{};
    PmNativeHandle signal_semaphore{};
    std::uint64_t signal_value{};
    std::uint64_t resource_generation{};
    std::uint64_t frame_generation{};
    GpuBufferOwner acquire_from{GpuBufferOwner::physics};
    GpuBufferOwner release_to{GpuBufferOwner::physics};
};

struct GpuRenderTarget {
    PmNativeBackend backend{};
    PmNativeHandle device{};
    PmNativeHandle handle{}; // zero requests an adapter-owned offscreen target
    std::uint32_t width{};
    std::uint32_t height{};
};

struct GpuDiagnosticReadback {
    // Null keeps the rendered image entirely on the GPU.
    std::vector<std::uint32_t> *rgba{};
};

enum class GpuDrawKind : std::uint8_t {
    rigid_mesh = 0,
    deformable_surface = 1,
    particles = 2,
    fluid_surface = 3,
};

enum class GpuOutputDomain : std::uint8_t {
    rigid,
    cloth,
    soft_body,
    fluid,
    smoke,
    rope,
};

// One graphics operation over a public output region. Backends choose their
// raster API and shader representation; these records contain draw meaning,
// layout and presentation material only, never solver equations.
struct GpuDrawCommand {
    GpuDrawKind kind{};
    GpuOutputDomain domain{};
    GpuSceneBuffers::Region output{};
    GpuSceneBuffers::Region previous_output{};
    GpuSceneBuffers::Region paint_output{};
    std::uint32_t output_stride{};
    std::uint32_t domain_index{};
    std::uint32_t mesh_index{PM_INVALID_INDEX};
    std::uint32_t paint_resolution{};
    Vec3 color{};
    float radius{};
    bool output_has_header{};
    bool use_triangle_corners{};
};

struct GpuSceneDrawList {
    std::vector<GpuDrawCommand> commands;
};

struct GpuRenderRequest {
    const GpuSceneBuffers *views{};
    GpuRendererBuffer output{};
    GpuFrameSynchronization synchronization{};
    GpuRenderTarget target{};
    GpuDiagnosticReadback readback{};
    Camera camera{};
    FrameTimingCollector *timing{};
    std::span<const std::uint32_t> overlay{};
};

[[nodiscard]] bool resolve_gpu_renderer_device(
    const GpuSceneBuffers &buffers, GpuRendererDevice &output,
    std::string &error);
[[nodiscard]] bool make_gpu_render_request(
    const GpuSceneBuffers &buffers, std::uint64_t frame_generation,
    Camera camera, GpuRenderTarget target, GpuDiagnosticReadback readback,
    FrameTimingCollector *timing, std::span<const std::uint32_t> overlay,
    GpuRenderRequest &output, std::string &error);
[[nodiscard]] bool make_gpu_scene_draw_list(
    const SceneDefinition &scene, const CommandSceneInstance &instance,
    GpuSceneDrawList &output, std::string &error);

// Shared state machine used by every native adapter. begin() borrows the
// physics output for graphics; finish() returns it before another physics step.
class GpuRendererContractState {
  public:
    explicit GpuRendererContractState(GpuRendererDevice device) noexcept
        : device_(device) {}

    [[nodiscard]] bool begin(const GpuRenderRequest &request,
                             std::string &error);
    [[nodiscard]] bool finish(std::uint64_t frame_generation,
                              std::string &error);
    void abandon(std::uint64_t frame_generation) noexcept;
    [[nodiscard]] const GpuRendererDevice &device() const noexcept {
        return device_;
    }

  private:
    GpuRendererDevice device_{};
    std::uint64_t newest_frame_{};
    std::uint64_t active_frame_{};
    bool graphics_owns_buffer_{};
};

class GpuGalleryRenderer {
  public:
    virtual ~GpuGalleryRenderer() = default;
    [[nodiscard]] virtual const GpuRendererDevice &renderer_device()
        const noexcept = 0;
    [[nodiscard]] virtual bool render(const GpuRenderRequest &request,
                                      std::string &error) = 0;
};

// Implemented by the native graphics adapter selected by the Gallery target.
[[nodiscard]] std::unique_ptr<GpuGalleryRenderer> create_gpu_gallery_renderer(
    GpuRendererDevice device, const SceneDefinition &scene,
    const CommandSceneInstance &instance, std::string &error);

} // namespace parallel_mater::gallery
