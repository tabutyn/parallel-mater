// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/window.hpp>
#include <parallel_mater_gallery/gpu_renderer.hpp>

#ifdef PM_GALLERY_VULKAN_WINDOW
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>
#ifdef PM_GALLERY_VULKAN_WINDOW
#include <GL/gl.h>
#include <type_traits>
#endif

#include <utility>

namespace parallel_mater::gallery {

struct GalleryWindow::Impl {
    GLFWwindow *window{};
    std::uint32_t image_width{};
    std::uint32_t image_height{};
#ifdef PM_GALLERY_VULKAN_WINDOW
    bool vulkan{};
    VkInstance instance{};
    VkSurfaceKHR surface{};
#endif

    ~Impl() {
#ifdef PM_GALLERY_VULKAN_WINDOW
        if (surface && instance)
            vkDestroySurfaceKHR(instance, surface, nullptr);
#endif
        if (window != nullptr) glfwDestroyWindow(window);
        glfwTerminate();
    }
};

GalleryWindow::GalleryWindow() noexcept = default;
GalleryWindow::~GalleryWindow() = default;
GalleryWindow::GalleryWindow(GalleryWindow &&) noexcept = default;
GalleryWindow &GalleryWindow::operator=(GalleryWindow &&) noexcept = default;

bool GalleryWindow::create(std::uint32_t window_width,
                           std::uint32_t window_height,
                           std::uint32_t image_width,
                           std::uint32_t image_height,
                           std::string_view title, GalleryWindow &output,
                           std::string &error,
                           std::optional<PmNativeBackend> backend) {
    error.clear();
    if (image_width == 0U || image_height == 0U) {
        error = "presentation image dimensions must be nonzero";
        return false;
    }
    if (glfwInit() == GLFW_FALSE) {
        error = "GLFW initialization failed";
        return false;
    }
    auto implementation = std::make_unique<Impl>();
#ifdef PM_GALLERY_VULKAN_WINDOW
    implementation->vulkan = backend == PM_NATIVE_BACKEND_VULKAN;
    if (implementation->vulkan)
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    else {
#endif
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#ifdef PM_GALLERY_VULKAN_WINDOW
    }
#endif
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#if defined(__linux__) && defined(GLFW_X11_CLASS_NAME)
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "parallel-mater");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "parallel-mater");
#endif
    const std::string window_title(title);
    implementation->window = glfwCreateWindow(
        static_cast<int>(window_width), static_cast<int>(window_height),
        window_title.c_str(), nullptr, nullptr);
    if (implementation->window == nullptr) {
        error = "GLFW window creation failed";
        return false;
    }
    implementation->image_width = image_width;
    implementation->image_height = image_height;
#ifdef PM_GALLERY_VULKAN_WINDOW
    if (implementation->vulkan) {
        output.impl_ = std::move(implementation);
        return true;
    }
#endif
    glfwMakeContextCurrent(implementation->window);
    glfwSwapInterval(1);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    output.impl_ = std::move(implementation);
    return true;
}

bool GalleryWindow::present(const std::vector<std::uint32_t> &rgba,
                            std::string &error, NativeRenderImage image) {
    if (image.handle != 0U) { error = "native image is not supported by OpenGL presentation"; return false; }
    error.clear();
#ifdef PM_GALLERY_VULKAN_WINDOW
    if (impl_ && impl_->vulkan) {
        error = "Vulkan window requires direct native rendering";
        return false;
    }
#endif
    if (!impl_ || rgba.size() !=
            static_cast<std::size_t>(impl_->image_width) *
                impl_->image_height) {
        error = "presentation image size does not match the window adapter";
        return false;
    }
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(
        impl_->window, &framebuffer_width, &framebuffer_height);
    glViewport(0, 0, framebuffer_width, framebuffer_height);
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(-1.0F, -1.0F);
    glPixelZoom(static_cast<float>(framebuffer_width) / impl_->image_width,
                static_cast<float>(framebuffer_height) /
                    impl_->image_height);
    glDrawPixels(static_cast<int>(impl_->image_width),
                 static_cast<int>(impl_->image_height), GL_RGBA,
                 GL_UNSIGNED_BYTE, rgba.data());
    glfwSwapBuffers(impl_->window);
    return true;
}

void GalleryWindow::set_vsync(bool enabled) noexcept {
    if (!impl_) return;
#ifdef PM_GALLERY_VULKAN_WINDOW
    if (impl_->vulkan) return; // Vulkan presentation currently uses FIFO.
#endif
    glfwMakeContextCurrent(impl_->window);
    glfwSwapInterval(enabled ? 1 : 0);
}

GLFWwindow *GalleryWindow::native_handle() const noexcept {
    return impl_ ? impl_->window : nullptr;
}

GpuRenderTarget GalleryWindow::native_render_target(
    const PmNativeDeviceInfo &native, std::uint32_t width,
    std::uint32_t height, std::string &error) {
#ifdef PM_GALLERY_VULKAN_WINDOW
    if (!impl_ || !impl_->vulkan || native.backend != PM_NATIVE_BACKEND_VULKAN ||
        native.instance == 0U || native.device == 0U || width == 0U ||
        height == 0U) {
        error = "Vulkan presentation target is invalid";
        return {};
    }
    const auto instance = reinterpret_cast<VkInstance>(
        static_cast<std::uintptr_t>(native.instance));
    if (impl_->surface && impl_->instance != instance) {
        error = "Vulkan presentation target changed instances";
        return {};
    }
    if (!impl_->surface) {
        const VkResult result = glfwCreateWindowSurface(
            instance, impl_->window, nullptr, &impl_->surface);
        if (result != VK_SUCCESS) {
            error = "GLFW Vulkan surface creation failed (VkResult " +
                std::to_string(static_cast<int>(result)) + ")";
            return {};
        }
        impl_->instance = instance;
    }
    error.clear();
    return {native.backend, native.device,
        static_cast<PmNativeHandle>(reinterpret_cast<std::uintptr_t>(
            impl_->surface)), width, height};
#else
    (void)native;
    (void)width;
    (void)height;
    error = "Native GPU rendering is unavailable in the OpenGL window";
    return {};
#endif
}

} // namespace parallel_mater::gallery
