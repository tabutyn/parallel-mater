// SPDX-License-Identifier: MIT
// Native surface/input adapter. No physics, session logic or timing collector.
#include <parallel_mater_gallery/window.hpp>
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "metal_gallery_shaders.hpp"
#include <utility>

namespace parallel_mater::gallery {
struct GalleryWindow::Impl {
    GLFWwindow *window{};
    unsigned width{}, height{};
    CAMetalLayer *layer;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLRenderPipelineState> pipeline;
    ~Impl() { if (window) glfwDestroyWindow(window); glfwTerminate(); }
};
GalleryWindow::GalleryWindow() noexcept = default;
GalleryWindow::~GalleryWindow() = default;
GalleryWindow::GalleryWindow(GalleryWindow &&) noexcept = default;
GalleryWindow &GalleryWindow::operator=(GalleryWindow &&) noexcept = default;
bool GalleryWindow::create(unsigned window_width, unsigned window_height, unsigned width, unsigned height,
                          std::string_view title, GalleryWindow &output, std::string &error) {
    if (!width || !height || !glfwInit()) { error = "Invalid dimensions or GLFW unavailable"; return false; }
    auto next = std::make_unique<Impl>(); next->width = width; next->height = height;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    next->window = glfwCreateWindow(window_width, window_height, std::string(title).c_str(), nullptr, nullptr);
    if (!next->window) { error = "Native Mac window creation failed"; return false; }
    next->device = MTLCreateSystemDefaultDevice(); next->queue = [next->device newCommandQueue];
    if (!next->queue) { error = "Metal presentation unavailable"; return false; }
    next->layer = [CAMetalLayer layer]; next->layer.device = next->device;
    next->layer.pixelFormat = MTLPixelFormatBGRA8Unorm; next->layer.framebufferOnly = YES;
    next->layer.drawableSize = CGSizeMake(width,height); next->layer.displaySyncEnabled = YES;
    NSWindow *native_window = glfwGetCocoaWindow(next->window);
    NSView *view = native_window.contentView;
    view.wantsLayer = YES; view.layer = next->layer;
    NSError *native_error = nil;
    id<MTLLibrary> library = [next->device newLibraryWithSource:[NSString stringWithUTF8String:gallery_metal_source]
        options:nil error:&native_error];
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"overlay_vertex"];
    desc.fragmentFunction = [library newFunctionWithName:@"overlay_fragment"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    next->pipeline = [next->device newRenderPipelineStateWithDescriptor:desc error:&native_error];
    if (!next->pipeline) { error = native_error.localizedDescription.UTF8String; return false; }
    output.impl_ = std::move(next); error.clear(); return true;
}
bool GalleryWindow::present(const std::vector<std::uint32_t> &rgba, std::string &error, NativeRenderImage native) {
    if (native.handle) { error = "Unexpected native image for Metal CPU presenter"; return false; }
    @autoreleasepool {
        if (!impl_ || rgba.size() != std::size_t(impl_->width)*impl_->height) {
            error = "Presentation image size mismatch"; return false;
        }
        id<CAMetalDrawable> drawable = [impl_->layer nextDrawable];
        if (!drawable) return true; // Minimized/occluded window; do not fail the session.
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:impl_->width height:impl_->height mipmapped:NO];
        desc.storageMode = MTLStorageModeShared; desc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [impl_->device newTextureWithDescriptor:desc];
        if (!texture) { error = "Metal presentation texture allocation failed"; return false; }
        [texture replaceRegion:MTLRegionMake2D(0,0,impl_->width,impl_->height) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:impl_->width*4];
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionDontCare; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
        if (!command || !encoder) { error = "Metal presentation command allocation failed"; return false; }
        [encoder setRenderPipelineState:impl_->pipeline]; [encoder setFragmentTexture:texture atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [encoder endEncoding];
        [command presentDrawable:drawable]; [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) { error = "Metal presentation failed"; return false; }
        return true;
    }
}
void GalleryWindow::set_vsync(bool enabled) noexcept { if (impl_) impl_->layer.displaySyncEnabled = enabled; }
GLFWwindow *GalleryWindow::native_handle() const noexcept { return impl_ ? impl_->window : nullptr; }
void *GalleryWindow::metal_layer(void *device) {
    if (!impl_) return nullptr;
    impl_->layer.device = (__bridge id<MTLDevice>)device;
    impl_->layer.drawableSize = CGSizeMake(impl_->width,impl_->height);
    return (__bridge void *)impl_->layer;
}
}
