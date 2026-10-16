// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/window.hpp>

#include <GLFW/glfw3.h>

#include <utility>

namespace parallel_mater::gallery {

struct GalleryWindow::Impl {
    GLFWwindow *window{};
    std::uint32_t image_width{};
    std::uint32_t image_height{};

    ~Impl() {
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
                           std::string &error) {
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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
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
    glfwMakeContextCurrent(impl_->window);
    glfwSwapInterval(enabled ? 1 : 0);
}

GLFWwindow *GalleryWindow::native_handle() const noexcept {
    return impl_ ? impl_->window : nullptr;
}

} // namespace parallel_mater::gallery
