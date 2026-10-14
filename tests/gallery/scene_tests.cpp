// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <scene_builder.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

parallel_mater::Quaternion rotation_z(float angle) {
    return {0.0F, 0.0F, std::sin(angle * 0.5F), std::cos(angle * 0.5F)};
}

parallel_mater::RigidBodyState rigid_state(const PmRigidState &source) {
    return {{source.position[0], source.position[1], source.position[2]},
            {source.orientation[0], source.orientation[1],
             source.orientation[2], source.orientation[3]},
            {source.linearVelocity[0], source.linearVelocity[1],
             source.linearVelocity[2]},
            {source.angularVelocity[0], source.angularVelocity[1],
             source.angularVelocity[2]}};
}

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    const SceneDefinition minimum_dump = make_dump_scene(10U);
    check(minimum_dump.meshes.size() == 3U &&
              minimum_dump.rigid_bodies.size() == 12U,
          "DUMP must contain three shared meshes, two containers, and N spheres");
    if (minimum_dump.meshes.size() == 3U) {
        check(minimum_dump.meshes[0].indices.size() == 4U * 2U * 3U,
              "DUMP hopper must omit top and right cube faces");
        check(minimum_dump.meshes[1].indices.size() == 5U * 2U * 3U,
              "DUMP receiver must omit only the top cube face");
        const TriangleMesh &sphere = minimum_dump.meshes[2];
        check(sphere.vertices.size() == 14U && sphere.indices.size() == 72U,
              "DUMP sphere must use eight corners, six face centers, and 24 triangles");
        for (const Vertex &vertex : sphere.vertices) {
            const float radius = std::sqrt(
                vertex.position.x * vertex.position.x +
                vertex.position.y * vertex.position.y +
                vertex.position.z * vertex.position.z);
            check(std::fabs(radius - 0.1F) < 1.0e-6F,
                  "every DUMP sphere vertex must lie on the requested radius");
        }
    }
    if (minimum_dump.rigid_bodies.size() == 12U) {
        check(minimum_dump.rigid_bodies[0].options.motion ==
                  MotionType::kinematic &&
                  minimum_dump.rigid_bodies[1].options.motion ==
                      MotionType::static_body,
              "DUMP hopper must be kinematic and receiver static");
        for (std::size_t index = 2U; index < minimum_dump.rigid_bodies.size();
             ++index) {
            check(minimum_dump.rigid_bodies[index].options.motion ==
                          MotionType::dynamic &&
                      minimum_dump.rigid_bodies[index].mesh_indices ==
                          std::vector<std::uint32_t>{2U},
                  "every DUMP sphere must be dynamic and share one mesh");
        }
    }
    check(make_dump_scene(1'000U).rigid_bodies.size() == 1'002U,
          "DUMP must support the 1,000-sphere maximum");

    SceneDefinition fluid_scene;
    std::string fluid_error;
    check(load_glb_scene(PARALLEL_MATER_FLUID_SCENE_PATH,
                         fluid_scene, fluid_error),
          fluid_error.empty() ? "load Fluid GLB scene" : fluid_error.c_str());
    check(fluid_scene.rigid_bodies.size() == 1U &&
          fluid_scene.rigid_bodies[0].options.motion == MotionType::static_body,
          "Fluid scene must retain its passive triangle collider");
    check(fluid_scene.particle_sources.size() == 1U &&
          fluid_scene.destroy_planes.size() == 1U,
          "Blender Liquid Inflow and Outflow must export as lifecycle planes");
    if (fluid_scene.particle_sources.size() == 1U) {
        check(fluid_scene.particle_sources[0].options.initial_velocity.y < -0.9F,
              "Blender downward flow velocity must become gallery -Y");
        check(fluid_scene.particle_sources[0].vertices.size() >= 4 &&
              fluid_scene.particle_sources[0].indices.size() == 6,
              "inflow triangles must survive Blender export");
    }
    SceneDefinition fluid_rigid_scene;
    std::string fluid_rigid_error;
    check(load_glb_scene(PARALLEL_MATER_FLUID_RIGID_SCENE_PATH,
                         fluid_rigid_scene, fluid_rigid_error),
          fluid_rigid_error.empty() ? "load FluidRigid GLB scene"
                                     : fluid_rigid_error.c_str());
    std::set<std::array<int, 3>> sphere_centers;
    std::size_t dynamic_spheres = 0U;
    for (const auto &body : fluid_rigid_scene.rigid_bodies) {
        if (body.options.motion != MotionType::dynamic) continue;
        ++dynamic_spheres;
        const auto p = body.options.initial_state.position;
        sphere_centers.insert({static_cast<int>(std::lround(p.x * 1'000.0F)),
                               static_cast<int>(std::lround(p.y * 1'000.0F)),
                               static_cast<int>(std::lround(p.z * 1'000.0F))});
        check(body.mesh_indices.size() == 1U &&
              fluid_rigid_scene.meshes[body.mesh_indices[0]].indices.size() == 60U,
              "each Array sphere is one 20-triangle rigid mesh");
    }
    check(fluid_rigid_scene.rigid_bodies.size() == 65U &&
          dynamic_spheres == 64U && sphere_centers.size() == 64U &&
          fluid_rigid_scene.meshes.size() == 2U &&
          fluid_rigid_scene.particle_sources.size() == 1U &&
          fluid_rigid_scene.destroy_planes.size() == 1U,
          "FluidRigid exports 64 detached shared-mesh spheres and flow planes");

    SceneDefinition pegs_scene;
    std::string pegs_error;
    check(load_glb_scene(PARALLEL_MATER_PEGS_SCENE_PATH,
                         pegs_scene, pegs_error),
          pegs_error.empty() ? "load Pegs GLB scene" : pegs_error.c_str());
    std::size_t passive_peg_bodies = 0U, active_peg_bodies = 0U;
    std::size_t paintable_peg_bodies = 0U;
    for (const auto &peg_body : pegs_scene.rigid_bodies) {
        passive_peg_bodies +=
            peg_body.options.motion == MotionType::static_body;
        active_peg_bodies +=
            peg_body.options.motion == MotionType::dynamic;
        paintable_peg_bodies += peg_body.paintable;
        if (peg_body.paintable)
            check(peg_body.name.starts_with("Sphere") &&
                      peg_body.options.motion == MotionType::static_body,
                  "only the passive Peg bowl receives paint");
        if (peg_body.name.starts_with("Sphere") &&
            peg_body.options.motion == MotionType::static_body) {
            check(std::fabs(peg_body.options.collision_margin - 0.02F) < 0.001F,
                  "authored bowl skin keeps resting sphere contacts continuous");
            check(std::fabs(peg_body.options.friction - 0.05F) < 0.001F,
                  "authored bowl friction lets water drain down the wall");
            check(peg_body.paint_resolution == 64U,
                  "authored Peg paint grid matches the smaller bowl");
        }
        if (peg_body.options.motion == MotionType::dynamic) {
            check(std::fabs(peg_body.options.mass - 125.0F) < 0.01F,
                  "authored Peg sphere mass is heavy enough to sink");
            check(peg_body.options.initial_state.position.y > 0.3F,
                  "authored Peg sphere starts above the bowl");
        }
    }
    check(passive_peg_bodies == 6U && active_peg_bodies == 1U &&
          paintable_peg_bodies == 1U &&
          pegs_scene.particle_sources.empty() &&
          pegs_scene.destroy_planes.empty() &&
          pegs_scene.initial_particles.size() > 500U &&
          pegs_scene.initial_particles.size() < 10'000U,
          "Pegs Geometry flow creates one finite initial fill, not an emitter");
    check(std::fabs(pegs_scene.gravity_scale - 2.0F) < 0.01F &&
          std::fabs(pegs_scene.fluid_options.particle_radius - 0.026F) < 0.001F &&
          pegs_scene.fluid_options.rest_particle_volume > 0.0F &&
          pegs_scene.fluid_options.normal_damping > 0.0F,
          "authored Geometry flow carries scale-aware generic fluid settings");

    SceneDefinition rigid_body_scene;
    std::string rigid_body_error;
    check(load_glb_scene(PARALLEL_MATER_RIGID_BODY_SCENE_PATH,
                         rigid_body_scene, rigid_body_error),
          rigid_body_error.empty() ? "load RigidBody GLB scene"
                                   : rigid_body_error.c_str());
    std::size_t layer_one = 0U, layer_two = 0U;
    std::set<std::array<int, 3>> brick_centers;
    for (const RigidBodyDefinition &body : rigid_body_scene.rigid_bodies) {
        if (body.source_name != "Layer1" && body.source_name != "Layer2")
            continue;
        layer_one += body.source_name == "Layer1";
        layer_two += body.source_name == "Layer2";
        const Vec3 position = body.options.initial_state.position;
        brick_centers.insert({
            static_cast<int>(std::lround(position.x * 1'000.0F)),
            static_cast<int>(std::lround(position.y * 1'000.0F)),
            static_cast<int>(std::lround(position.z * 1'000.0F))});
    }
    check(rigid_body_scene.rigid_bodies.size() == 386U &&
              layer_one == 192U && layer_two == 192U &&
              brick_centers.size() == 384U &&
              rigid_body_scene.meshes.size() == 4U,
          "RigidBody arrays export 384 independent shared-mesh bricks");
    check(rigid_body_scene.hit_boxes.size() == 1U &&
              rigid_body_scene.hit_boxes[0].name == "LoadBox" &&
              std::fabs(rigid_body_scene.hit_boxes[0].box.center.z + 16.0F) <
                  1.0e-4F &&
              std::fabs(
                  rigid_body_scene.hit_boxes[0].box.half_extents.x -
                  4.3983493F) < 1.0e-4F,
          "RigidBody loads its authored LoadBox trigger volume");

    SceneDefinition scene;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_PASSIVE_ACTIVE_SCENE_PATH, scene, error),
          error.empty() ? "load PassiveActive GLB scene" : error.c_str());
    check(scene.rigid_bodies.size() == 5U,
          "PassiveActive must contain the bowl and four test bodies");
    check(scene.meshes.size() == 5U,
          "PassiveActive must contain five render meshes");
    check(scene.collision_meshes.size() == 3U,
          "three dynamic bodies must carry authored collision proxies");
    if (scene.rigid_bodies.size() == 5U) {
        check(scene.rigid_bodies[0].name == "Plane" &&
                  scene.rigid_bodies[0].options.motion ==
                      MotionType::static_body,
              "Blender PASSIVE plane must load as static");
        check(scene.rigid_bodies[1].name == "Cube" &&
                  scene.rigid_bodies[1].options.motion == MotionType::kinematic,
              "Blender ACTIVE Animated cube must load as kinematic");
        check(scene.rigid_bodies[2].name == "Icosphere" &&
                  scene.rigid_bodies[2].options.motion == MotionType::dynamic,
              "Blender ACTIVE sphere mesh must load as dynamic");
        check(scene.rigid_bodies[3].name == "Suzanne" &&
                  scene.rigid_bodies[3].options.motion == MotionType::dynamic,
              "Blender ACTIVE Suzanne must load as dynamic");
        check(scene.rigid_bodies[4].name == "Suzanne.001" &&
                  scene.rigid_bodies[4].options.motion == MotionType::dynamic,
              "the second Blender ACTIVE Suzanne must load as dynamic");
        for (const TriangleMesh &mesh : scene.meshes) {
            check(!mesh.vertices.empty() && !mesh.indices.empty() &&
                      mesh.indices.size() % 3U == 0U,
                  "every Blender object must be triangulated");
        }
        check(scene.meshes[0].indices.size() == 2'048U * 3U,
              "the Blender bowl quads must export as 2,048 triangles");
        check(scene.rigid_bodies[0].collision_mesh_indices.empty() &&
                  scene.rigid_bodies[1].collision_mesh_indices.empty(),
              "the detailed bowl and cube must use their render triangles");
        for (std::size_t index = 2U; index < 5U; ++index) {
            check(!scene.rigid_bodies[index].collision_mesh_indices.empty(),
                  "each detailed dynamic body must select a collision proxy");
            if (!scene.rigid_bodies[index].collision_mesh_indices.empty()) {
                const TriangleMesh &proxy = scene.collision_meshes[
                    scene.rigid_bodies[index].collision_mesh_indices.front()];
                check(proxy.indices.size() <
                          scene.meshes[index].indices.size(),
                      "a collision proxy must contain fewer triangles than its render mesh");
            }
        }
        float cube_extent = 0.0F;
        for (const Vertex &vertex : scene.meshes[1].vertices) {
            cube_extent = std::fmax(cube_extent, std::fabs(vertex.position.x));
            cube_extent = std::fmax(cube_extent, std::fabs(vertex.position.y));
            cube_extent = std::fmax(cube_extent, std::fabs(vertex.position.z));
        }
        check(std::fabs(cube_extent - 0.6837F) < 1.0e-3F,
              "the Blender cube scale must be baked into its vertices");
    }

    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    if (!CommandScene::create(scene, 2U, simulation, instance, error)) {
        if (error.find("physical device") != std::string::npos) {
            std::cout << "SKIP: GLB structure passed; no Parallel Mater device\n";
            return failures == 0 ? 77 : 1;
        }
        std::cerr << "FAIL: create PassiveActive command scene: "
                  << error << '\n';
        return 1;
    }
    check(instance.rigid_bodies.size() == 5U,
          "each rigid body must have a public command resource index");

    RigidBodyState cube_target =
        scene.rigid_bodies[1].options.initial_state;
    cube_target.position.x += 0.1F;
    check(simulation.set_kinematic_target(
              instance.rigid_bodies[1], cube_target, error),
          error.empty() ? "set authored cube target" : error.c_str());
    check(simulation.step(1.0F / 60.0F, 4U,
                          {4.905F, -8.495709F, 0.0F}, frame, error, true),
          error.empty() ? "step authored scene" : error.c_str());
    if (frame.rigid_states.size() == 5U) {
        const RigidBodyState after_cube = rigid_state(frame.rigid_states[1]);
        const RigidBodyState after_icosphere =
            rigid_state(frame.rigid_states[2]);
        const auto initial_icosphere =
            scene.rigid_bodies[2].options.initial_state;
        check(std::fabs(after_cube.position.x - cube_target.position.x) <
                  1.0e-5F,
              "kinematic cube must follow its public-API target");
        check(after_icosphere.position.x > initial_icosphere.position.x &&
                  after_icosphere.position.y < initial_icosphere.position.y,
              "tilted gravity must affect the dynamic icosphere");
    } else {
        check(false, "public rigid output must contain every scene body");
    }

    CommandScene dump_simulation;
    CommandSceneInstance dump_instance;
    CommandSceneFrame dump_frame;
    if (!CommandScene::create(minimum_dump, 0U, dump_simulation,
                              dump_instance, error)) {
        std::cerr << "FAIL: create DUMP command scene: " << error << '\n';
        return 1;
    }
    check(dump_instance.rigid_bodies.size() == 12U,
          "DUMP resources must reuse the authored public geometry");
    RigidBodyState hopper_target =
        minimum_dump.rigid_bodies[0].options.initial_state;
    hopper_target.orientation = rotation_z(0.0F);
    check(dump_simulation.set_kinematic_target(
              dump_instance.rigid_bodies[0], hopper_target, error) &&
              dump_simulation.step(1.0F / 60.0F, 4U,
                                   {0.0F, -9.81F, 0.0F}, dump_frame,
                                   error, true),
          error.empty() ? "step DUMP command scene" : error.c_str());
    check(dump_frame.rigid_states.size() == 12U,
          "DUMP public output must contain every body");

    fluid_scene.fluid_options.capacity =
        std::max<std::uint32_t>(fluid_scene.fluid_options.capacity, 100U);
    CommandScene fluid_simulation;
    CommandSceneInstance fluid_instance;
    CommandSceneFrame fluid_frame;
    if (!CommandScene::create(fluid_scene, 0U, fluid_simulation,
                              fluid_instance, error)) {
        std::cerr << "FAIL: create Fluid command scene: " << error << '\n';
        return 1;
    }
    bool fluid_stepped = true;
    for (int index = 0; index < 30 && fluid_stepped; ++index)
        fluid_stepped = fluid_simulation.step(
            1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F},
            fluid_frame, error);
    check(fluid_stepped,
          error.empty() ? "step authored Fluid command scene" : error.c_str());
    check(fluid_instance.fluid != PM_INVALID_INDEX &&
              !fluid_frame.fluid_particles.empty() &&
              fluid_frame.statistics.peakParticleCount >=
                  fluid_frame.fluid_particles.size(),
          "inflow must emit live particles through public output queries");

    if (failures != 0) {
        std::cerr << failures << " gallery scene test(s) failed\n";
        return 1;
    }
    std::cout << "PassiveActive Blender scene tests passed\n";
    return 0;
}
