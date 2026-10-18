// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer_adapter.hpp>
#include <parallel_mater_scene/command_scene.hpp>

#include "renderer_backends.hpp"

#include <utility>

namespace parallel_mater::gallery {

bool RendererAdapterFactory::native_gpu_available() noexcept {
#if defined(PM_GALLERY_NATIVE_RENDERER_VULKAN) || \
    defined(PM_GALLERY_NATIVE_RENDERER_METAL)
    return true;
#else
    return false;
#endif
}

bool RendererAdapterFactory::create_host(
    const SceneDefinition &scene, const CommandSceneInstance &instance,
    std::uint32_t width, std::uint32_t height, RendererHandle &output,
    std::string &error) {
    error.clear();
    output = {};
#ifdef PM_GALLERY_HOST_RENDERER_D3D12
    PmNativeDeviceInfo native{};
    if (instance.render_buffers.device != nullptr &&
        pmGetNativeDevice(instance.render_buffers.device, &native) == PM_SUCCESS &&
        native.backend == PM_NATIVE_BACKEND_D3D12) {
        auto renderer = create_d3d12_renderer(
            scene, instance.render_buffers, width, height, error);
        if (!renderer) return false;
        output = RendererHandle(std::move(renderer));
        return true;
    }
#endif
#ifdef PM_GALLERY_HOST_RENDERER_OPTIX
    auto renderer = create_optix_renderer(
        scene, instance, width, height, error);
#else
    (void)instance;
    auto renderer = create_software_renderer(scene, width, height, error);
#endif
    if (!renderer) return false;
    output = RendererHandle(std::move(renderer));
    return true;
}

std::unique_ptr<GpuGalleryRenderer> RendererAdapterFactory::create_native(
    GpuRendererDevice device, const SceneDefinition &scene,
    const CommandSceneInstance &instance, std::string &error) {
    error.clear();
#if !defined(PM_GALLERY_NATIVE_RENDERER_VULKAN) && \
    !defined(PM_GALLERY_NATIVE_RENDERER_METAL)
    (void)device;
    (void)scene;
    (void)instance;
#endif
#ifdef PM_GALLERY_NATIVE_RENDERER_VULKAN
    if (device.native.backend == PM_NATIVE_BACKEND_VULKAN)
        return create_vulkan_gallery_renderer(device, scene, instance, error);
#endif
#ifdef PM_GALLERY_NATIVE_RENDERER_METAL
    if (device.native.backend == PM_NATIVE_BACKEND_METAL)
        return create_metal_gallery_renderer(device, scene, instance, error);
#endif
    error = "No native Gallery renderer adapter supports the selected backend";
    return {};
}

} // namespace parallel_mater::gallery
