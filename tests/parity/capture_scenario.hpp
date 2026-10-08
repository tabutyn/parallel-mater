// SPDX-License-Identifier: MIT
#if defined(PARALLEL_MATER_PARITY_CUDA)
#include <parallel_mater/parallel_mater.hpp>
#else
#include <parallel_mater/metal.hpp>
#endif

#include "capture_format.hpp"

#if !defined(PARALLEL_MATER_PARITY_CUDA)
#import <Metal/Metal.h>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

#if defined(PARALLEL_MATER_PARITY_CUDA)
using namespace parallel_mater;
template <typename T> using BufferSpan = DeviceSpan<T>;
constexpr const char *capture_backend = "cuda";
#else
using namespace parallel_mater::metal;
constexpr const char *capture_backend = "metal";
#endif
namespace parity = parallel_mater::test::parity;

namespace {

#if defined(PARALLEL_MATER_PARITY_CUDA)
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <typename T> struct Upload {
    T *data = nullptr;
    Upload(const T *source, std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
        const auto error = cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) { cudaFree(data); data = nullptr; cuda_check(error); }
    }
    ~Upload() { if (data) cudaFree(data); }
    Upload(const Upload &) = delete;
    Upload &operator=(const Upload &) = delete;
};
#else
template <typename T> const T *contents(BufferSpan<const T> span) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)span.buffer;
    return reinterpret_cast<const T *>(
        static_cast<const std::byte *>(buffer.contents) + span.byte_offset);
}

#endif

bool checked(Status status, std::string_view operation) {
    if (status.ok()) return true;
    std::cerr << operation;
    if (status.message != nullptr) std::cerr << ": " << status.message;
    std::cerr << '\n';
    return false;
}

std::string member(std::string_view collection, std::size_t index,
                   std::string_view field) {
    return std::string(collection) + '[' + std::to_string(index) + "]." +
           std::string(field);
}

void write_vec3(parity::Writer &writer, std::string_view path, Vec3 value) {
    writer.floating_value(std::string(path) + ".x", value.x);
    writer.floating_value(std::string(path) + ".y", value.y);
    writer.floating_value(std::string(path) + ".z", value.z);
}

void write_quaternion(parity::Writer &writer, std::string_view path,
                      Quaternion value) {
    writer.floating_value(std::string(path) + ".x", value.x);
    writer.floating_value(std::string(path) + ".y", value.y);
    writer.floating_value(std::string(path) + ".z", value.z);
    writer.floating_value(std::string(path) + ".w", value.w);
}

template <typename Id>
void write_id(parity::Writer &writer, std::string_view path, Id value) {
    writer.unsigned_value(std::string(path) + ".index", value.index);
    writer.unsigned_value(std::string(path) + ".generation", value.generation);
}

void write_debug_frame(parity::Writer &writer,
                       const PhysicsDebugFrame &frame) {
    writer.unsigned_value("frame.index", frame.frame_index);
    writer.floating_value("frame.timestep", frame.timestep);
    write_vec3(writer, "frame.gravity", frame.gravity);
    writer.unsigned_value("frame.maximum_fluid_neighbor_count",
                          frame.maximum_fluid_neighbor_count);

    writer.unsigned_value("rigid.count", frame.rigid_bodies.size());
    for (std::size_t i = 0; i < frame.rigid_bodies.size(); ++i) {
        const PhysicsDebugRigidSample &sample = frame.rigid_bodies[i];
        write_id(writer, member("rigid", i, "id"), sample.id);
        write_vec3(writer, member("rigid", i, "position"),
                   sample.state.position);
        write_quaternion(writer, member("rigid", i, "orientation"),
                         sample.state.orientation);
        write_vec3(writer, member("rigid", i, "linear_velocity"),
                   sample.state.linear_velocity);
        write_vec3(writer, member("rigid", i, "angular_velocity"),
                   sample.state.angular_velocity);
        write_vec3(writer, member("rigid", i, "applied_force"),
                   sample.applied_force);
        write_vec3(writer, member("rigid", i, "applied_torque"),
                   sample.applied_torque);
    }

    writer.unsigned_value("fluid.count", frame.fluid_particles.size());
    for (std::size_t i = 0; i < frame.fluid_particles.size(); ++i) {
        const PhysicsDebugFluidSample &sample = frame.fluid_particles[i];
        write_id(writer, member("fluid", i, "system"), sample.fluid);
        writer.unsigned_value(member("fluid", i, "stable_id"),
                              sample.stable_particle_id);
        write_vec3(writer, member("fluid", i, "position"), sample.position);
        write_vec3(writer, member("fluid", i, "velocity"), sample.velocity);
        write_vec3(writer, member("fluid", i, "acceleration"),
                   sample.acceleration);
        writer.floating_value(member("fluid", i, "foam"), sample.foam);
    }

    writer.unsigned_value("cloth.count", frame.cloth_vertices.size());
    for (std::size_t i = 0; i < frame.cloth_vertices.size(); ++i) {
        const PhysicsDebugClothSample &sample = frame.cloth_vertices[i];
        write_id(writer, member("cloth", i, "system"), sample.cloth);
        writer.unsigned_value(member("cloth", i, "vertex"), sample.vertex);
        write_vec3(writer, member("cloth", i, "position"), sample.position);
        write_vec3(writer, member("cloth", i, "velocity"), sample.velocity);
        write_vec3(writer, member("cloth", i, "rigid_contact_force"),
                   sample.rigid_contact_force);
        write_vec3(writer, member("cloth", i, "fluid_contact_force"),
                   sample.fluid_contact_force);
        write_vec3(writer, member("cloth", i, "soft_contact_force"),
                   sample.soft_body_contact_force);
    }

    writer.unsigned_value("soft.count", frame.soft_body_nodes.size());
    for (std::size_t i = 0; i < frame.soft_body_nodes.size(); ++i) {
        const PhysicsDebugSoftBodySample &sample = frame.soft_body_nodes[i];
        write_id(writer, member("soft", i, "system"), sample.soft_body);
        writer.unsigned_value(member("soft", i, "node"), sample.node);
        write_vec3(writer, member("soft", i, "position"), sample.position);
        write_vec3(writer, member("soft", i, "velocity"), sample.velocity);
        write_vec3(writer, member("soft", i, "rigid_contact_force"),
                   sample.rigid_contact_force);
        write_vec3(writer, member("soft", i, "cloth_contact_force"),
                   sample.cloth_contact_force);
        write_vec3(writer, member("soft", i, "fluid_contact_force"),
                   sample.fluid_contact_force);
    }

    writer.unsigned_value("rope.count", frame.rope_nodes.size());
    for (std::size_t i = 0; i < frame.rope_nodes.size(); ++i) {
        const PhysicsDebugRopeSample &sample = frame.rope_nodes[i];
        write_id(writer, member("rope", i, "system"), sample.rope);
        writer.unsigned_value(member("rope", i, "node"), sample.node);
        write_vec3(writer, member("rope", i, "position"), sample.position);
        write_vec3(writer, member("rope", i, "velocity"), sample.velocity);
        write_vec3(writer, member("rope", i, "constraint_force"),
                   sample.constraint_force);
        write_vec3(writer, member("rope", i, "contact_force"),
                   sample.contact_force);
        write_vec3(writer, member("rope", i, "fluid_contact_force"),
                   sample.fluid_contact_force);
    }

    writer.unsigned_value("rigid_contact.count", frame.rigid_contacts.size());
    for (std::size_t i = 0; i < frame.rigid_contacts.size(); ++i) {
        const RigidContactEvent &event = frame.rigid_contacts[i];
        write_id(writer, member("rigid_contact", i, "body"), event.body);
        write_id(writer, member("rigid_contact", i, "collider"),
                 event.collider);
        write_vec3(writer, member("rigid_contact", i, "position"),
                   event.position);
        write_vec3(writer, member("rigid_contact", i, "normal"), event.normal);
        writer.floating_value(member("rigid_contact", i, "penetration"),
                              event.penetration);
        writer.floating_value(member("rigid_contact", i, "normal_impulse"),
                              event.normal_impulse);
        write_vec3(writer, member("rigid_contact", i, "friction_impulse"),
                   event.friction_impulse);
    }

    writer.unsigned_value("fluid_contact.count", frame.fluid_contacts.size());
    for (std::size_t i = 0; i < frame.fluid_contacts.size(); ++i) {
        const ContactEvent &event = frame.fluid_contacts[i];
        write_id(writer, member("fluid_contact", i, "fluid"), event.fluid);
        writer.unsigned_value(member("fluid_contact", i, "stable_id"),
                              event.stable_particle_id);
        write_id(writer, member("fluid_contact", i, "rigid"),
                 event.rigid_body);
        write_vec3(writer, member("fluid_contact", i, "position"),
                   event.position);
        write_vec3(writer, member("fluid_contact", i, "normal"), event.normal);
        writer.floating_value(member("fluid_contact", i, "normal_impulse"),
                              event.normal_impulse);
    }
}

template <typename T, typename Write>
void write_buffer(parity::Writer &writer, std::string_view name,
                  BufferSpan<const T> span, Write write) {
    writer.unsigned_value(std::string(name) + ".count", span.size);
#if defined(PARALLEL_MATER_PARITY_CUDA)
    std::vector<T> downloaded(span.size);
    if (span.size) cuda_check(cudaMemcpy(downloaded.data(), span.data,
                                        span.size * sizeof(T), cudaMemcpyDeviceToHost));
    const T *values = downloaded.data();
#else
    const T *values = contents(span);
#endif
    for (std::uint64_t i = 0; i < span.size; ++i)
        write(writer, member(name, i, "value"), values[i]);
}

void write_statistics(parity::Writer &writer, const WorldStatistics &stats) {
    writer.unsigned_value("stats.frame_index", stats.frame_index);
    writer.unsigned_value("stats.fluid_count", stats.fluid_count);
    writer.unsigned_value("stats.particle_count", stats.particle_count);
    writer.unsigned_value("stats.smoke_system_count", stats.smoke_system_count);
    writer.unsigned_value("stats.smoke_particle_count",
                          stats.smoke_particle_count);
    writer.unsigned_value("stats.emitted_smoke_particle_count",
                          stats.emitted_smoke_particle_count);
    writer.unsigned_value("stats.boiled_particle_count",
                          stats.boiled_particle_count);
    writer.unsigned_value("stats.rigid_body_count", stats.rigid_body_count);
    writer.unsigned_value("stats.rigid_constraint_count",
                          stats.rigid_constraint_count);
    writer.unsigned_value("stats.triangle_mesh_count",
                          stats.triangle_mesh_count);
    writer.unsigned_value("stats.contact_count", stats.contact_count);
    writer.unsigned_value("stats.contact_overflow_count",
                          stats.contact_overflow_count);
    writer.unsigned_value("stats.maximum_fluid_neighbor_count",
                          stats.maximum_fluid_neighbor_count);
    writer.unsigned_value("stats.cloth_count", stats.cloth_count);
    writer.unsigned_value("stats.cloth_vertex_count", stats.cloth_vertex_count);
    writer.unsigned_value("stats.soft_body_count", stats.soft_body_count);
    writer.unsigned_value("stats.soft_body_node_count",
                          stats.soft_body_node_count);
    writer.unsigned_value("stats.rope_count", stats.rope_count);
    writer.unsigned_value("stats.rope_node_count", stats.rope_node_count);
    writer.unsigned_value("stats.fluid_soft_contact_count",
                          stats.fluid_soft_body_contact_count);
    writer.floating_value("stats.maximum_fluid_soft_penetration",
                          stats.maximum_fluid_soft_body_penetration);
    writer.unsigned_value("stats.fluid_rope_contact_count",
                          stats.fluid_rope_contact_count);
    writer.floating_value("stats.maximum_fluid_rope_penetration",
                          stats.maximum_fluid_rope_penetration);
    writer.unsigned_value("stats.rope_soft_contact_count",
                          stats.rope_soft_body_contact_count);
    writer.floating_value("stats.maximum_rope_soft_penetration",
                          stats.maximum_rope_soft_body_penetration);
}

bool build_and_capture(std::ostream &output) {
    World world;
    Status status = World::create(
        {.fluid_capacity = 1U,
         .smoke_capacity = 1U,
         .rigid_body_capacity = 3U,
         .rigid_constraint_capacity = 1U,
         .triangle_mesh_capacity = 1U,
         .contact_capacity = 256U,
         .cloth_capacity = 1U,
         .soft_body_capacity = 1U,
         .rope_capacity = 1U,
         .physics_debug = {.frame_capacity = 1U, .frame_stride = 1U}},
        world);
    if (!checked(status, "create parity world")) return false;

    const std::array<Vec3, 4> mesh_vertices{{
        {-0.2F, -0.2F, -0.2F}, {0.2F, -0.2F, -0.2F},
        {0.0F, -0.2F, 0.2F},   {0.0F, 0.2F, 0.0F},
    }};
    const std::array<std::uint32_t, 12> mesh_indices{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U};
    TriangleMeshId mesh{};
#if defined(PARALLEL_MATER_PARITY_CUDA)
    const Upload<Vec3> uploaded_vertices(mesh_vertices.data(), mesh_vertices.size());
    const Upload<std::uint32_t> uploaded_indices(mesh_indices.data(), mesh_indices.size());
    status = world.add_triangle_mesh(
        {uploaded_vertices.data, mesh_vertices.size()},
        {uploaded_indices.data, mesh_indices.size()}, mesh);
#else
    status = world.add_triangle_mesh(
        {mesh_vertices.data(), mesh_vertices.size()},
        {mesh_indices.data(), mesh_indices.size()}, mesh);
#endif
    if (!checked(status, "add parity mesh")) return false;

    RigidBodyId obstacle{}, anchor{}, constrained{};
    status = world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = mesh}, obstacle);
    status = status ? world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = mesh,
                           .initial_state = {.position = {10.0F, 0.0F, 0.0F}}},
                          anchor)
                    : status;
    status = status ? world.add_rigid_body(
                          {.mesh = mesh,
                           .initial_state = {
                               .position = {10.0F, 0.0F, 0.0F},
                               .linear_velocity = {0.5F, 0.0F, 0.0F}},
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F},
                          constrained)
                    : status;
    if (!checked(status, "add parity rigid bodies")) return false;
    RigidConstraintId constraint{};
    status = world.add_rigid_constraint(
        {.type = RigidConstraintType::fixed,
         .body_a = anchor,
         .body_b = constrained},
        constraint);
    if (!checked(status, "add parity rigid constraint")) return false;

    const std::array<FluidParticle, 2> water{{
        {{-0.03F, 1.0F, 0.0F}, {}, 20.0F},
        {{0.03F, 1.0F, 0.0F}, {}, 20.0F},
    }};
    FluidId fluid{};
#if defined(PARALLEL_MATER_PARITY_CUDA)
    const Upload<FluidParticle> uploaded_water(water.data(), water.size());
    const auto *water_data = uploaded_water.data;
#else
    const auto *water_data = water.data();
#endif
    status = world.add_fluid(
        {.capacity = 8U,
         .particle_radius = 0.03F,
         .support_radius = 0.12F,
         .solver_iterations = 4U,
         .repulsion = 20.0F,
         .maximum_speed = 10.0F},
        {water_data, water.size()}, fluid);
    if (!checked(status, "add parity fluid")) return false;

    SmokeId smoke{};
    status = world.add_smoke(
        {.capacity = 32U,
         .initial_velocity = {0.5F, 0.0F, 0.0F},
         .wind = {0.5F, 0.0F, 0.0F},
         .particles_per_second = 60.0F,
         .lifetime = 2.0F,
         .grid_resolution = 16U,
         .grid_vertical_resolution = 8U,
         .grid_pressure_iterations = 4U,
         .grid_minimum = {-0.25F, -1.0F, -1.0F},
         .grid_edge_length = 2.0F},
        smoke);
    if (!checked(status, "add parity smoke")) return false;

    std::array<Vec3, 4> cloth_vertices{{
        {-1.0F, 0.0F, -1.0F}, {3.0F, 0.0F, -1.0F},
        {-1.0F, 4.0F, -1.0F}, {-1.0F, 0.0F, 3.0F},
    }};
    std::array<std::uint32_t, 12> cloth_indices{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U};
    std::array<float, 4> cloth_inverse{};
    ClothId cloth{};
    status = world.add_cloth(
        {.vertices = {cloth_vertices.data(), cloth_vertices.size()},
         .triangle_indices = {cloth_indices.data(), cloth_indices.size()},
         .inverse_masses = {cloth_inverse.data(), cloth_inverse.size()},
         .solver_iterations = 8U,
         .preserve_volume = true},
        cloth);
    if (!checked(status, "add parity cloth")) return false;

    std::array<Vec3, 4> soft_nodes{{
        {2.0F, 1.0F, 0.0F}, {2.3F, 1.0F, 0.0F},
        {2.0F, 1.3F, 0.0F}, {2.0F, 1.0F, 0.3F},
    }};
    const auto distance = [&](std::uint32_t a, std::uint32_t b) {
        const Vec3 d{soft_nodes[b].x - soft_nodes[a].x,
                     soft_nodes[b].y - soft_nodes[a].y,
                     soft_nodes[b].z - soft_nodes[a].z};
        return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    };
    std::array<SoftBodyBond, 6> soft_bonds{{
        {0U, 1U, distance(0U, 1U)}, {0U, 2U, distance(0U, 2U)},
        {0U, 3U, distance(0U, 3U)}, {1U, 2U, distance(1U, 2U)},
        {1U, 3U, distance(1U, 3U)}, {2U, 3U, distance(2U, 3U)},
    }};
    std::array<SoftBodySurfaceBinding, 4> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].nodes[0] = i;
        bindings[i].weights[0] = 1.0F;
    }
    SoftBodyId soft{};
    status = world.add_soft_body(
        {.nodes = {soft_nodes.data(), soft_nodes.size()},
         .bonds = {soft_bonds.data(), soft_bonds.size()},
         .surface_vertices = {soft_nodes.data(), soft_nodes.size()},
         .surface_triangle_indices = {cloth_indices.data(), cloth_indices.size()},
         .surface_bindings = {bindings.data(), bindings.size()},
         .solver_iterations = 8U},
        soft);
    if (!checked(status, "add parity soft body")) return false;

    std::array<Vec3, 2> rope_line{{cloth_vertices[0],
                                   {-2.0F, 0.0F, -1.0F}}};
    RopeId rope{};
    status = world.add_rope(
        {.centerline = {rope_line.data(), rope_line.size()},
         .node_spacing = 0.04F,
         .radius = 0.02F,
         .solver_iterations = 16U},
        rope);
    if (!checked(status, "add parity rope")) return false;

    FluidSmokeCouplingId fluid_smoke{};
    SmokeSoftBodyCouplingId smoke_soft{};
    SmokeClothCouplingId smoke_cloth{};
    SmokeRopeCouplingId smoke_rope{};
    SmokeRigidCouplingId smoke_rigid{};
    FluidClothCouplingId fluid_cloth{};
    FluidSoftBodyCouplingId fluid_soft{};
    FluidRopeCouplingId fluid_rope{};
    SoftBodyClothCouplingId soft_cloth{};
    RopeSoftBodyCouplingId rope_soft{};
    RopeClothCouplingId rope_cloth{};
    status = world.add_fluid_smoke_coupling(
        {.fluid = fluid, .smoke = smoke}, fluid_smoke);
    status = status ? world.add_smoke_soft_body_coupling(
                          {.smoke = smoke, .soft_body = soft}, smoke_soft)
                    : status;
    status = status ? world.add_smoke_cloth_coupling(
                          {.smoke = smoke, .cloth = cloth}, smoke_cloth)
                    : status;
    status = status ? world.add_smoke_rope_coupling(
                          {.smoke = smoke, .rope = rope}, smoke_rope)
                    : status;
    status = status ? world.add_smoke_rigid_coupling(
                          {.smoke = smoke, .body = obstacle}, smoke_rigid)
                    : status;
    status = status ? world.add_fluid_cloth_coupling(
                          {.fluid = fluid, .cloth = cloth}, fluid_cloth)
                    : status;
    status = status ? world.add_fluid_soft_body_coupling(
                          {.fluid = fluid, .soft_body = soft}, fluid_soft)
                    : status;
    status = status ? world.add_fluid_rope_coupling(
                          {.fluid = fluid, .rope = rope}, fluid_rope)
                    : status;
    status = status ? world.add_soft_body_cloth_coupling(
                          {.soft_body = soft, .cloth = cloth}, soft_cloth)
                    : status;
    status = status ? world.add_rope_soft_body_coupling(
                          {.rope = rope, .soft_body = soft}, rope_soft)
                    : status;
    status = status ? world.add_rope_cloth_coupling(
                          {.rope = rope, .cloth = cloth, .first_vertex = 0U},
                          rope_cloth)
                    : status;
    if (!checked(status, "add parity couplings")) return false;

    for (std::uint32_t frame = 0; frame < 3U; ++frame) {
        status = world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 4U,
                             .gravity = {0.0F, -9.81F, 0.0F},
                             .collect_rigid_contacts = true,
                             .collect_fluid_contacts = true});
        if (!checked(status, "step parity world")) return false;
    }

    PhysicsDebugCapture capture{};
    WorldStatistics stats{};
    SmokeDeviceView smoke_view{};
    RigidConstraintState constraint_state{};
    status = world.copy_physics_debug_capture(capture);
    status = status ? world.collect_statistics(stats) : status;
    status = status ? world.smoke_view(smoke, smoke_view) : status;
    status = status ? world.read_rigid_constraint_state(constraint,
                                                        constraint_state)
                    : status;
    if (!checked(status, "read parity checkpoint") ||
        capture.frames.size() != 1U) {
        if (capture.frames.size() != 1U)
            std::cerr << "parity debug capture did not contain one frame\n";
        return false;
    }

    parity::Writer writer(output, "all-systems-coupled-v1", capture_backend);
    write_debug_frame(writer, capture.frames.front());
    writer.unsigned_value("constraint.enabled", constraint_state.enabled);
    writer.unsigned_value("constraint.broken", constraint_state.broken);
    writer.floating_value("constraint.applied_impulse",
                          constraint_state.applied_impulse);
    write_statistics(writer, stats);

    writer.unsigned_value("smoke.particle_count", smoke_view.particle_count);
    writer.floating_value("smoke.lifetime", smoke_view.lifetime);
    writer.floating_value("smoke.particle_radius", smoke_view.particle_radius);
    writer.unsigned_value("smoke.grid_resolution", smoke_view.grid_resolution);
    writer.unsigned_value("smoke.grid_vertical_resolution",
                          smoke_view.grid_vertical_resolution);
    write_vec3(writer, "smoke.grid_minimum", smoke_view.grid_minimum);
    writer.floating_value("smoke.grid_spacing", smoke_view.grid_spacing);
    writer.floating_value("smoke.grid_pressure_relative_residual",
                          smoke_view.grid_pressure_relative_residual);
    const auto vector_writer = [](parity::Writer &w, std::string_view path,
                                  Vec3 value) { write_vec3(w, path, value); };
    const auto float_writer = [](parity::Writer &w, std::string_view path,
                                 float value) { w.floating_value(path, value); };
    const auto uint_writer = [](parity::Writer &w, std::string_view path,
                                std::uint32_t value) {
        w.unsigned_value(path, value);
    };
    write_buffer(writer, "smoke.position", smoke_view.positions, vector_writer);
    write_buffer(writer, "smoke.velocity", smoke_view.velocities, vector_writer);
    write_buffer(writer, "smoke.age", smoke_view.ages, float_writer);
    write_buffer(writer, "smoke.number_density", smoke_view.number_densities,
                 float_writer);
    write_buffer(writer, "smoke.pressure", smoke_view.pressures, float_writer);
    write_buffer(writer, "smoke.vorticity", smoke_view.vorticities,
                 vector_writer);
    write_buffer(writer, "smoke.grid_velocity", smoke_view.grid_velocity,
                 vector_writer);
    write_buffer(writer, "smoke.grid_pressure", smoke_view.grid_pressure,
                 float_writer);
    write_buffer(writer, "smoke.grid_density", smoke_view.grid_density,
                 float_writer);
    write_buffer(writer, "smoke.grid_temperature", smoke_view.grid_temperature,
                 float_writer);
    write_buffer(writer, "smoke.grid_solid", smoke_view.grid_solid, uint_writer);
    write_buffer(writer, "smoke.grid_vorticity", smoke_view.grid_vorticity,
                 vector_writer);
    write_buffer(writer, "smoke.grid_divergence", smoke_view.grid_divergence,
                 float_writer);
    return output.good();
}

int self_test() {
    std::ostringstream first;
    if (!build_and_capture(first)) return EXIT_FAILURE;
    const std::string reference_text = first.str();
    for (std::uint32_t run = 1U; run < 10U; ++run) {
        std::ostringstream replay;
        if (!build_and_capture(replay)) return EXIT_FAILURE;
        if (reference_text != replay.str()) {
            std::cerr << capture_backend << " parity capture run " << run
                      << " is not byte-identical\n";
            return EXIT_FAILURE;
        }
    }
    parity::Capture first_capture;
    parity::Capture replay_capture;
    std::string error;
    std::istringstream first_input(reference_text);
    std::istringstream replay_input(reference_text);
    if (!parity::parse(first_input, first_capture, error) ||
        !parity::parse(replay_input, replay_capture, error)) {
        std::cerr << capture_backend << " parity capture parse failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    const parity::ComparisonResult comparison =
        parity::compare(first_capture, replay_capture,
                        {.absolute_tolerance = 0.0,
                         .relative_tolerance = 0.0});
    if (!comparison.matches) {
        std::cerr << comparison.message << '\n';
        return EXIT_FAILURE;
    }
    std::cout << capture_backend << " all-system parity capture is byte-identical across ten "
                 "runs ("
              << first_capture.records.size() << " records)\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) try {
#if defined(PARALLEL_MATER_PARITY_CUDA)
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
#endif
    if (argc == 2 && std::string_view(argv[1]) == "--self-test")
        return self_test();
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " OUTPUT | --self-test\n";
        return 2;
    }
    if (std::string_view(argv[1]) == "-")
        return build_and_capture(std::cout) ? EXIT_SUCCESS : EXIT_FAILURE;
    std::ofstream output(argv[1]);
    if (!output) {
        std::cerr << "cannot open parity capture: " << argv[1] << '\n';
        return 2;
    }
    return build_and_capture(output) ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch (const std::exception &error) {
    std::cerr << capture_backend << " parity capture: " << error.what() << '\n';
    return EXIT_FAILURE;
}
