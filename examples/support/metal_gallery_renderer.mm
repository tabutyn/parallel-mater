// SPDX-License-Identifier: MIT
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
struct GpuGalleryRenderer::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLRenderPipelineState> mesh_pipeline, particle_pipeline, overlay_pipeline;
    id<MTLDepthStencilState> depth_state;
    struct Draw {
        id<MTLBuffer> vertices;
        unsigned count, domain, index, paint{PM_INVALID_INDEX}, paint_resolution{};
        Vec3 color;
    };
    std::vector<Draw> draws;
    float fluid_radius{}, smoke_radius{};
    std::vector<float> rope_radii;
    Impl(PmDevice physics, const SceneDefinition &scene, const CommandSceneInstance &instance) {
        PmNativeDeviceInfo native{};
        require(pmGetNativeDevice(physics, &native) == PM_SUCCESS && native.backend == PM_NATIVE_BACKEND_METAL,
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
        const auto mesh = [&](unsigned mesh_index, unsigned domain, unsigned index, bool corners) {
            const auto &source = scene.meshes.at(mesh_index);
            if (!source.visible || source.indices.empty()) return;
            std::vector<VertexData> vertices;
            vertices.reserve(source.indices.size());
            for (unsigned i = 0; i < source.indices.size(); ++i) {
                const auto &v = source.vertices.at(source.indices[i]);
                vertices.push_back({v.position, v.uv, corners ? i : source.indices[i]});
            }
            Draw draw{[device newBufferWithBytes:vertices.data() length:vertices.size()*sizeof(VertexData)
                options:MTLResourceStorageModeShared], static_cast<unsigned>(vertices.size()), domain, index,
                PM_INVALID_INDEX, 0, source.base_color};
            require(draw.vertices != nil, "Metal scene vertices allocation failed");
            for (const auto &binding : instance.paint_bindings) {
                if (binding.mesh_index != mesh_index || (domain == 0 && binding.body_index != index)) continue;
                draw.paint = binding.field_index;
                draw.paint_resolution = domain == 0 ? scene.rigid_bodies.at(index).paint_resolution
                    : scene.cloths.at(index).paint_resolution;
            }
            draws.push_back(std::move(draw));
        };
        for (unsigned i = 0; i < scene.rigid_bodies.size(); ++i)
            for (auto index : scene.rigid_bodies[i].mesh_indices) mesh(index, 0, i, false);
        for (unsigned i = 0; i < scene.cloths.size(); ++i) {
            const auto &cloth = scene.cloths[i];
            mesh(cloth.mesh_index, 1, i, cloth.break_strain > 0 || cloth.impact_break_impulse > 0);
        }
        for (unsigned i = 0; i < scene.soft_bodies.size(); ++i) mesh(scene.soft_bodies[i].mesh_index, 2, i, false);
        fluid_radius = std::max(scene.fluid_options.particle_radius, .02F);
        smoke_radius = std::max(scene.smoke_options.particle_radius, .03F);
        for (const auto &rope : scene.ropes) rope_radii.push_back(std::max(rope.options.radius, .025F));
    }
    void render(const CommandSceneGpuFrame &frame, Camera camera, unsigned width, unsigned height,
                void *target, std::vector<std::uint32_t> *capture, FrameTimingCollector *timing,
                std::span<const std::uint32_t> overlay) {
        require(width && height && frame.buffer.buffer, "Invalid Metal render frame");
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(frame.buffer.buffer));
        require(buffer.device == device && frame.buffer.offset <= buffer.length &&
            frame.buffer.size <= buffer.length - frame.buffer.offset, "Foreign or out-of-bounds graphics buffer");
        CAMetalLayer *layer = (__bridge CAMetalLayer *)target;
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
        [encoder setVertexBuffer:buffer offset:frame.buffer.offset atIndex:1];
        [encoder setFragmentBuffer:buffer offset:frame.buffer.offset atIndex:1];
        using namespace math;
        const auto forward = normalize_or(subtract(camera.target, camera.eye), {0,0,-1});
        const auto right = normalize_or(cross(forward, camera.up), {1,0,0});
        const auto up = normalize_or(cross(right, forward), {0,1,0});
        const float vertical = std::tan(camera.vertical_field_of_view_degrees * .00872664626F);
        Uniforms u;
        u.eye = vector(camera.eye); u.forward = vector(forward);
        u.right = vector(right, vertical * float(width)/height); u.up = vector(up, vertical); u.height = height;
        const auto region = [&](CommandSceneGpuFrame::Region r, unsigned stride, bool header) {
            require(r.offset <= frame.buffer.size && r.size <= frame.buffer.size - r.offset &&
                r.offset <= UINT32_MAX && r.size <= UINT32_MAX && (!header || r.size >= sizeof(PmDomainOutputHeader)),
                "Invalid public graphics output region");
            u.offset = static_cast<unsigned>(r.offset); u.stride = stride;
            u.count = static_cast<unsigned>((r.size - (header ? sizeof(PmDomainOutputHeader) : 0)) / stride);
        };
        [encoder setRenderPipelineState:mesh_pipeline];
        for (const auto &draw : draws) {
            if (draw.domain == 0) {
                const auto offset = frame.rigid.offset + draw.index * sizeof(PmRigidState);
                require((draw.index + 1ULL) * sizeof(PmRigidState) <= frame.rigid.size, "Missing GPU rigid state");
                region({offset, sizeof(PmRigidState)}, sizeof(PmRigidState), false);
            } else region(draw.domain == 1 ? frame.cloth_surfaces.at(draw.index) : frame.soft_surfaces.at(draw.index), sizeof(PmFloat3), true);
            u.kind = draw.domain == 0 ? 0 : 1; u.color = vector(draw.color, 1); u.paint_resolution = 0;
            if (draw.paint != PM_INVALID_INDEX) {
                const auto paint = frame.paint.at(draw.paint);
                require(paint.offset <= UINT32_MAX && paint.size >= 16 && paint.offset <= frame.buffer.size &&
                    paint.size <= frame.buffer.size - paint.offset, "Invalid GPU paint region");
                u.paint_offset = static_cast<unsigned>(paint.offset); u.paint_resolution = draw.paint_resolution;
                u.paint_count = static_cast<unsigned>((paint.size-16)/4);
            }
            [encoder setVertexBuffer:draw.vertices offset:0 atIndex:0];
            [encoder setVertexBytes:&u length:sizeof(u) atIndex:2];
            [encoder setFragmentBytes:&u length:sizeof(u) atIndex:2];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:draw.count];
        }
        [encoder setRenderPipelineState:particle_pipeline];
        const auto particles = [&](CommandSceneGpuFrame::Region r, unsigned stride, Vec3 tint, float radius) {
            if (!r.size) return;
            region(r, stride, true); u.radius = radius; u.color = vector(tint, 1);
            [encoder setVertexBytes:&u length:sizeof(u) atIndex:2];
            [encoder setFragmentBytes:&u length:sizeof(u) atIndex:2];
            [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:u.count];
        };
        particles(frame.fluid, sizeof(PmFluidParticleOutput), {.05F,.42F,.92F}, fluid_radius);
        particles(frame.smoke, sizeof(PmSmokeParticleState), {.58F,.61F,.65F}, smoke_radius);
        for (unsigned i = 0; i < frame.ropes.size(); ++i)
            particles(frame.ropes[i], sizeof(PmRopeNodeState), {.85F,.62F,.18F}, rope_radii.at(i));
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
};
GpuGalleryRenderer::GpuGalleryRenderer(PmDevice device, const SceneDefinition &scene, const CommandSceneInstance &instance)
    : impl_(std::make_unique<Impl>(device, scene, instance)) {}
GpuGalleryRenderer::~GpuGalleryRenderer() = default;
void GpuGalleryRenderer::render(const CommandSceneGpuFrame &frame, Camera camera, unsigned width, unsigned height,
                                void *layer, std::vector<std::uint32_t> *capture, FrameTimingCollector *timing,
                                std::span<const std::uint32_t> overlay) {
    @autoreleasepool { impl_->render(frame, camera, width, height, layer, capture, timing, overlay); }
}
}
