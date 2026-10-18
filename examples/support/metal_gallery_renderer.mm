// SPDX-License-Identifier: MIT
#if !__has_feature(objc_arc)
#error "Metal Gallery rendering requires ARC (including mixed-language Xcode targets)"
#endif
#include <parallel_mater_gallery/gpu_renderer.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include "scene_builder.hpp"
#include "vector_math.hpp"
#include "metal_gallery_shaders.hpp"
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace parallel_mater::gallery {
namespace {
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
struct VertexData { Vec3 position; Vec2 uv; std::uint32_t output_index; };
struct Uniforms {
    std::array<float,4> eye{}, right{}, up{}, forward{}, color{};
    std::uint32_t offset{}, stride{}, kind{}, count{};
    std::uint32_t paint_offset{}, paint_resolution{}, paint_count{}, padding{};
    float radius{}, height{}, padding2{}, padding3{};
};
static_assert(sizeof(VertexData) == 24 && sizeof(Uniforms) == 128);
static_assert(sizeof(PmRigidState) == 52 && offsetof(PmRigidState, orientation) == 12);
static_assert(sizeof(PmDomainOutputHeader) == 16 && sizeof(PmFloat3) == 12);
std::array<float,4> vector(Vec3 v, float w = 0) { return {v.x,v.y,v.z,w}; }
}
struct MetalGalleryRenderer final : GpuGalleryRenderer {
    GpuRendererContractState contract;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLRenderPipelineState> mesh_pipeline, particle_pipeline, overlay_pipeline;
    id<MTLDepthStencilState> depth_state;
    struct Draw {
        id<MTLBuffer> vertices;
        unsigned count;
        GpuDrawCommand semantics;
    };
    std::vector<Draw> draws;
    std::vector<GpuDrawCommand> particle_draws;
    MetalGalleryRenderer(GpuRendererDevice renderer_device,
                         const SceneDefinition &scene,
                         const CommandSceneInstance &instance)
        : contract(renderer_device) {
        const auto native = renderer_device.native;
        require(native.backend == PM_NATIVE_BACKEND_METAL && native.device != 0U,
            "GPU Gallery rendering requires the native Metal physics device");
        device = (__bridge id<MTLDevice>)(reinterpret_cast<void *>(native.device));
        queue = [device newCommandQueue];
        require(queue != nil, "Metal graphics queue allocation failed");
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:gallery_metal_source]
            options:nil error:&error];
        if (!library) throw std::runtime_error(std::string("Gallery graphics shaders: ") + error.localizedDescription.UTF8String);
        const auto pipeline = [&](NSString *vertex, NSString *fragment, bool overlay = false) {
            MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
            desc.vertexFunction = [library newFunctionWithName:vertex];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
            desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            if (overlay) {
                auto attachment = desc.colorAttachments[0];
                attachment.blendingEnabled = YES;
                attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
                attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
                attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            }
            id<MTLRenderPipelineState> result = [device newRenderPipelineStateWithDescriptor:desc error:&error];
            if (!result) throw std::runtime_error(std::string("Gallery graphics pipeline: ") + error.localizedDescription.UTF8String);
            return result;
        };
        mesh_pipeline = pipeline(@"mesh_vertex", @"mesh_fragment");
        particle_pipeline = pipeline(@"particle_vertex", @"particle_fragment");
        overlay_pipeline = pipeline(@"overlay_vertex", @"overlay_fragment", true);
        MTLDepthStencilDescriptor *depth = [MTLDepthStencilDescriptor new];
        depth.depthCompareFunction = MTLCompareFunctionLess; depth.depthWriteEnabled = YES;
        depth_state = [device newDepthStencilStateWithDescriptor:depth];
        require(depth_state != nil, "Metal depth state allocation failed");
        GpuSceneDrawList draw_list;
        std::string draw_error;
        require(make_gpu_scene_draw_list(scene, instance, draw_list, draw_error),
                draw_error.c_str());
        for (const auto &semantics : draw_list.commands) {
            if (semantics.kind == GpuDrawKind::particles) {
                particle_draws.push_back(semantics);
                continue;
            }
            const auto &source = scene.meshes.at(semantics.mesh_index);
            if (!source.visible || source.indices.empty()) continue;
            std::vector<VertexData> vertices;
            vertices.reserve(source.indices.size());
            for (unsigned i = 0; i < source.indices.size(); ++i) {
                const auto &v = source.vertices.at(source.indices[i]);
                vertices.push_back({v.position, v.uv,
                    semantics.use_triangle_corners ? i : source.indices[i]});
            }
            Draw draw{[device newBufferWithBytes:vertices.data() length:vertices.size()*sizeof(VertexData)
                options:MTLResourceStorageModeShared],
                static_cast<unsigned>(vertices.size()), semantics};
            require(draw.vertices != nil, "Metal scene vertices allocation failed");
            draws.push_back(std::move(draw));
        }
    }
    void render_frame(const GpuRenderRequest &request) {
        const auto &native = request.output.native;
        const auto camera = request.camera;
        unsigned width = request.target.width;
        unsigned height = request.target.height;
        auto *capture = request.readback.rgba;
        auto *timing = request.timing;
        const auto overlay = request.overlay;
        require(request.synchronization.mode == GpuSynchronizationMode::queue_idle &&
            pmQueueWaitIdle(request.synchronization.physics_queue) == PM_SUCCESS,
            "Metal graphics could not acquire completed physics output");
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(native.buffer));
        require(buffer.device == device && native.offset <= buffer.length &&
            native.size <= buffer.length - native.offset, "Foreign or out-of-bounds graphics buffer");
        CAMetalLayer *layer = (__bridge CAMetalLayer *)(reinterpret_cast<void *>(
            request.target.handle));
        id<CAMetalDrawable> drawable = nil;
        if (layer) {
            require(layer.device == device, "Presentation layer must use the physics Metal device");
            if (timing) { timing->end_phase(); timing->begin_phase(FramePhase::present); }
            drawable = [layer nextDrawable];
            if (timing) { timing->end_phase(); timing->begin_phase(FramePhase::render); }
        }
        if (drawable) { width = static_cast<unsigned>(drawable.texture.width); height = static_cast<unsigned>(drawable.texture.height); }
        MTLTextureDescriptor *color_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
            width:width height:height mipmapped:NO];
        color_desc.storageMode = MTLStorageModePrivate; color_desc.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> color = drawable ? drawable.texture : [device newTextureWithDescriptor:color_desc];
        MTLTextureDescriptor *depth_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
            width:width height:height mipmapped:NO];
        depth_desc.storageMode = MTLStorageModePrivate; depth_desc.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> depth = [device newTextureWithDescriptor:depth_desc];
        require(color && depth, "Metal render targets allocation failed");
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = color;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(.055,.08,.11,1);
        pass.depthAttachment.texture = depth; pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = 1;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
        require(command && encoder, "Metal graphics command allocation failed");
        [encoder setDepthStencilState:depth_state]; [encoder setCullMode:MTLCullModeNone];
        [encoder setVertexBuffer:buffer offset:native.offset atIndex:1];
        [encoder setFragmentBuffer:buffer offset:native.offset atIndex:1];
        using namespace math;
        const auto forward = normalize_or(subtract(camera.target, camera.eye), {0,0,-1});
        const auto right = normalize_or(cross(forward, camera.up), {1,0,0});
        const auto up = normalize_or(cross(right, forward), {0,1,0});
        const float vertical = std::tan(camera.vertical_field_of_view_degrees * .00872664626F);
        Uniforms u;
        u.eye = vector(camera.eye); u.forward = vector(forward);
        u.right = vector(right, vertical * float(width)/height); u.up = vector(up, vertical); u.height = height;
        const auto region = [&](GpuSceneBuffers::Region r, unsigned stride, bool header) {
            require(r.offset <= native.size && r.size <= native.size - r.offset &&
                r.offset <= UINT32_MAX && r.size <= UINT32_MAX && (!header || r.size >= sizeof(PmDomainOutputHeader)),
                "Invalid public graphics output region");
            u.offset = static_cast<unsigned>(r.offset); u.stride = stride;
            u.count = static_cast<unsigned>((r.size - (header ? sizeof(PmDomainOutputHeader) : 0)) / stride);
        };
        [encoder setRenderPipelineState:mesh_pipeline];
        for (const auto &draw : draws) {
            const auto &semantics = draw.semantics;
            region(semantics.output, semantics.output_stride,
                   semantics.output_has_header);
            u.kind = semantics.kind == GpuDrawKind::rigid_mesh ? 0 : 1;
            u.color = vector(semantics.color, 1); u.paint_resolution = 0;
            if (semantics.paint_output.size != 0U) {
                const auto paint = semantics.paint_output;
                require(paint.offset <= UINT32_MAX && paint.size >= 16 && paint.offset <= native.size &&
                    paint.size <= native.size - paint.offset, "Invalid GPU paint region");
                u.paint_offset = static_cast<unsigned>(paint.offset);
                u.paint_resolution = semantics.paint_resolution;
                u.paint_count = static_cast<unsigned>((paint.size-16)/4);
            }
            [encoder setVertexBuffer:draw.vertices offset:0 atIndex:0];
            [encoder setVertexBytes:&u length:sizeof(u) atIndex:2];
            [encoder setFragmentBytes:&u length:sizeof(u) atIndex:2];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:draw.count];
        }
        [encoder setRenderPipelineState:particle_pipeline];
        const auto particles = [&](const GpuDrawCommand &semantics) {
            region(semantics.output, semantics.output_stride, true);
            u.radius = semantics.radius; u.color = vector(semantics.color, 1);
            [encoder setVertexBytes:&u length:sizeof(u) atIndex:2];
            [encoder setFragmentBytes:&u length:sizeof(u) atIndex:2];
            [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:u.count];
        };
        for (const auto &draw : particle_draws) particles(draw);
        if (!overlay.empty()) {
            require(overlay.size() == std::size_t(width)*height, "Overlay dimensions do not match Metal target");
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                width:width height:height mipmapped:NO];
            desc.storageMode = MTLStorageModeShared; desc.usage = MTLTextureUsageShaderRead;
            id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
            require(texture != nil, "Metal overlay allocation failed");
            [texture replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 withBytes:overlay.data() bytesPerRow:width*4];
            [encoder setDepthStencilState:nil]; [encoder setRenderPipelineState:overlay_pipeline];
            [encoder setFragmentTexture:texture atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [encoder endEncoding];
        id<MTLBuffer> readback = nil;
        const NSUInteger row_bytes = (width * 4ULL + 255) & ~255ULL;
        if (capture) {
            readback = [device newBufferWithLength:row_bytes * height options:MTLResourceStorageModeShared];
            require(readback != nil, "Metal image capture allocation failed");
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                sourceSize:MTLSizeMake(width,height,1) toBuffer:readback destinationOffset:0
                destinationBytesPerRow:row_bytes destinationBytesPerImage:row_bytes*height];
            [blit endEncoding];
        }
        if (drawable) {
            if (timing) { timing->end_phase(); timing->begin_phase(FramePhase::present); }
            [command presentDrawable:drawable];
            if (timing) { timing->end_phase(); timing->begin_phase(FramePhase::render); }
        }
        [command commit]; [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Metal graphics execution failed");
        // Rendering has finished reading the borrowed buffer before the next
        // physics step can overwrite it. This preview is fenced, not pipelined.
        if (capture) {
            capture->resize(std::size_t(width)*height);
            for (unsigned y = 0; y < height; ++y) {
                const auto *row = reinterpret_cast<const std::uint32_t *>(static_cast<const char *>(readback.contents) + y * row_bytes);
                for (unsigned x = 0; x < width; ++x) {
                    const auto p = row[x];
                    (*capture)[std::size_t(height-1-y)*width+x] = (p & 0xff00ff00U) | ((p & 255U)<<16) | ((p>>16)&255U);
                }
            }
        }
    }
    const GpuRendererDevice &renderer_device() const noexcept override {
        return contract.device();
    }
    bool render(const GpuRenderRequest &request,
                std::string &error) override {
        if (!contract.begin(request, error)) return false;
        try {
            @autoreleasepool { render_frame(request); }
        } catch (const std::exception &exception) {
            contract.abandon(request.synchronization.frame_generation);
            error = exception.what();
            return false;
        }
        return contract.finish(request.synchronization.frame_generation,
                               error);
    }
};

std::unique_ptr<GpuGalleryRenderer> create_gpu_gallery_renderer(
    GpuRendererDevice device, const SceneDefinition &scene,
    const CommandSceneInstance &instance, std::string &error) {
    error.clear();
    try {
        @autoreleasepool {
            return std::make_unique<MetalGalleryRenderer>(
                device, scene, instance);
        }
    } catch (const std::exception &exception) {
        error = exception.what();
        return {};
    }
}
}
