// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <iostream>

using namespace parallel_mater::metal;

namespace {
id<MTLBuffer> private_buffer(id<MTLDevice> device,
                             id<MTLCommandQueue> queue,
                             const void *bytes, NSUInteger length) {
    id<MTLBuffer> staging =
        [device newBufferWithBytes:bytes
                           length:length
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> output =
        [device newBufferWithLength:length
                            options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (staging == nil || output == nil || command == nil || blit == nil)
        return nil;
    [blit copyFromBuffer:staging
            sourceOffset:0U
                toBuffer:output
       destinationOffset:0U
                    size:length];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted ? output : nil;
}
} // namespace

int main() {
    World world;
    Status status = World::create({}, world);
    if (!status) {
        std::cerr << status.message << '\n';
        return 1;
    }

    id<MTLDevice> device =
        (__bridge id<MTLDevice>)world.native_context().device;
    id<MTLCommandQueue> upload_queue = [device newCommandQueue];
    if (upload_queue == nil) {
        std::cerr << "Private-buffer upload queue creation failed\n";
        return 1;
    }
    const std::array<Vec3, 4> vertices{{
        {-0.5F, 0.0F, -0.5F},
        {0.5F, 0.0F, -0.5F},
        {0.0F, 0.0F, 0.5F},
        {0.0F, 1.0F, 0.0F},
    }};
    const std::array<std::uint32_t, 12> indices{
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};

    id<MTLBuffer> vertex_buffer =
        [device newBufferWithLength:sizeof(vertices)
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer =
        [device newBufferWithLength:sizeof(indices)
                            options:MTLResourceStorageModeShared];
    std::memcpy(vertex_buffer.contents, vertices.data(), sizeof(vertices));
    std::memcpy(index_buffer.contents, indices.data(), sizeof(indices));

    TriangleMeshId mesh{};
    status = world.add_triangle_mesh(
        {(__bridge void *)vertex_buffer, 0, vertices.size()},
        {(__bridge void *)index_buffer, 0, indices.size()}, mesh);
    if (!status) {
        std::cerr << (status.message == nullptr ? "Buffer upload failed"
                                               : status.message)
                  << '\n';
        return 1;
    }
    status = world.remove_triangle_mesh(mesh);
    if (!status) {
        std::cerr << "Removing buffer-uploaded mesh failed\n";
        return 1;
    }

    id<MTLBuffer> private_vertex_buffer =
        private_buffer(device, upload_queue, vertices.data(), sizeof(vertices));
    id<MTLBuffer> private_index_buffer =
        private_buffer(device, upload_queue, indices.data(), sizeof(indices));
    if (private_vertex_buffer == nil || private_index_buffer == nil) {
        std::cerr << "Private mesh-buffer staging failed\n";
        return 1;
    }
    status = world.add_triangle_mesh(
        {(__bridge void *)private_vertex_buffer, 0, vertices.size()},
        {(__bridge void *)private_index_buffer, 0, indices.size()}, mesh);
    if (!status) {
        std::cerr << (status.message == nullptr ? "Private buffer upload failed"
                                               : status.message)
                  << '\n';
        return 1;
    }
    status = world.remove_triangle_mesh(mesh);
    if (!status) {
        std::cerr << "Removing private-buffer mesh failed\n";
        return 1;
    }

    const std::array<FluidParticle, 2> particles{{
        {{0.0F, 1.0F, 0.0F}, {}, 20.0F},
        {{0.1F, 1.0F, 0.0F}, {}, 25.0F},
    }};
    id<MTLBuffer> private_particles =
        private_buffer(device, upload_queue, particles.data(), sizeof(particles));
    if (private_particles == nil) {
        std::cerr << "Private particle-buffer staging failed\n";
        return 1;
    }
    FluidId fluid{};
    status = world.add_fluid(
        {.capacity = 4U},
        {(__bridge void *)private_particles, 0U, particles.size()}, fluid);
    if (!status) {
        std::cerr << (status.message == nullptr
                          ? "Private-buffer fluid upload failed"
                          : status.message)
                  << '\n';
        return 1;
    }
    FluidDeviceView fluid_view{};
    status = world.fluid_view(fluid, fluid_view);
    if (!status || fluid_view.particle_count != particles.size()) {
        std::cerr << "Private-buffer fluid view is invalid\n";
        return 1;
    }
    status = world.remove_fluid(fluid);
    if (!status) {
        std::cerr << "Removing private-buffer fluid failed\n";
        return 1;
    }

    status = world.add_triangle_mesh(
        {(__bridge void *)vertex_buffer, 0, vertices.size()},
        {(__bridge void *)index_buffer, 0, indices.size()}, mesh);
    RigidBodyId body{};
    status = status ? world.add_rigid_body(
                          {.motion = MotionType::static_body, .mesh = mesh},
                          body)
                    : status;
    const std::array<Vec2, 4> uvs{{
        {0.0F, 0.0F}, {1.0F, 0.0F}, {0.5F, 1.0F}, {0.5F, 0.5F},
    }};
    id<MTLBuffer> private_uvs =
        private_buffer(device, upload_queue, uvs.data(), sizeof(uvs));
    if (private_uvs == nil) {
        std::cerr << "Private UV-buffer staging failed\n";
        return 1;
    }
    PaintFieldId field{};
    status = status ? world.add_paint_field(
                          {.body = body,
                           .mesh = mesh,
                           .vertex_uvs = {(__bridge void *)private_uvs, 0U,
                                          uvs.size()},
                           .width = 8U,
                           .height = 8U},
                          field)
                    : status;
    PaintFieldDeviceView paint_view{};
    status = status ? world.paint_field_view(field, paint_view) : status;
    if (!status || paint_view.pixels.size != 64U) {
        std::cerr << (status.message == nullptr
                          ? "Private-buffer paint UV upload failed"
                          : status.message)
                  << '\n';
        return 1;
    }

    RigidBodyDeviceView rigid_view{};
    status = world.rigid_body_view(rigid_view);
    if (!status || rigid_view.revision != paint_view.revision) {
        std::cerr << "Paint and rigid views do not share a world revision\n";
        return 1;
    }
    std::uint64_t revision = rigid_view.revision;

    status = world.add_fluid(
        {.capacity = 4U},
        {(__bridge void *)private_particles, 0U, particles.size()}, fluid);
    status = status ? world.rigid_body_view(rigid_view) : status;
    status = status ? world.fluid_view(fluid, fluid_view) : status;
    if (!status || rigid_view.revision != revision + 1U ||
        fluid_view.revision != rigid_view.revision) {
        std::cerr << "Fluid registration did not advance the world revision\n";
        return 1;
    }
    revision = rigid_view.revision;

    SmokeId smoke{};
    status = world.add_smoke({.capacity = 4U, .particles_per_second = 1.0F},
                             smoke);
    SmokeDeviceView smoke_view{};
    status = status ? world.rigid_body_view(rigid_view) : status;
    status = status ? world.smoke_view(smoke, smoke_view) : status;
    if (!status || rigid_view.revision != revision + 1U ||
        smoke_view.revision != rigid_view.revision) {
        std::cerr << "Smoke registration did not advance the world revision\n";
        return 1;
    }
    revision = rigid_view.revision;

    FluidSmokeCouplingId coupling{};
    status = world.add_fluid_smoke_coupling(
        {.fluid = fluid, .smoke = smoke}, coupling);
    status = status ? world.rigid_body_view(rigid_view) : status;
    if (!status || rigid_view.revision != revision + 1U) {
        std::cerr << "Coupling registration did not advance the world revision\n";
        return 1;
    }
    revision = rigid_view.revision;

    status = world.step({.timestep = 1.0F / 120.0F, .substeps = 2U,
                         .gravity = {}});
    status = status ? world.rigid_body_view(rigid_view) : status;
    status = status ? world.fluid_view(fluid, fluid_view) : status;
    status = status ? world.smoke_view(smoke, smoke_view) : status;
    status = status ? world.paint_field_view(field, paint_view) : status;
    if (!status || rigid_view.revision != revision + 1U ||
        fluid_view.revision != rigid_view.revision ||
        smoke_view.revision != rigid_view.revision ||
        paint_view.revision != rigid_view.revision) {
        std::cerr << "Completed frame views do not share one revision\n";
        return 1;
    }

    revision = rigid_view.revision;
    status = world.clear_paint_field(field);
    status = status ? world.rigid_body_view(rigid_view) : status;
    if (!status || rigid_view.revision != revision) {
        std::cerr << "Paint clear unexpectedly invalidated CUDA-style views\n";
        return 1;
    }

    status = world.remove_fluid_smoke_coupling(coupling);
    status = status ? world.rigid_body_view(rigid_view) : status;
    if (!status || rigid_view.revision != revision + 1U) {
        std::cerr << "Coupling removal did not advance the world revision\n";
        return 1;
    }
    revision = rigid_view.revision;
    const Status stale_remove = world.remove_fluid_smoke_coupling(coupling);
    status = world.rigid_body_view(rigid_view);
    if (stale_remove.code != StatusCode::invalid_handle || !status ||
        rigid_view.revision != revision) {
        std::cerr << "Failed mutation unexpectedly advanced the revision\n";
        return 1;
    }
    status = status ? world.remove_smoke(smoke) : status;
    status = status ? world.remove_fluid(fluid) : status;
    status = status ? world.remove_paint_field(field) : status;
    status = status ? world.remove_rigid_body(body) : status;
    status = status ? world.remove_triangle_mesh(mesh) : status;
    if (!status) {
        std::cerr << "Removing private-buffer paint resources failed\n";
        return 1;
    }
    return 0;
}
