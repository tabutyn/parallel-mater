// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/dump_truck.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace {
void check(bool okay, const char *message) { if (!okay) throw std::runtime_error(message); }
Vec3 subtract(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
float length(Vec3 v) { return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); }
Quaternion conjugate(Quaternion q) { return {-q.x,-q.y,-q.z,q.w}; }
Quaternion multiply(Quaternion a, Quaternion b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y, a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w, a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}
Vec3 rotate(Quaternion q, Vec3 v) {
    const auto r = multiply(multiply(q, {v.x,v.y,v.z,0}), conjugate(q));
    return {r.x,r.y,r.z};
}
RigidBodyState rigid_state(const PmRigidState &source) {
    return {{source.position[0], source.position[1], source.position[2]},
            {source.orientation[0], source.orientation[1],
             source.orientation[2], source.orientation[3]},
            {source.linearVelocity[0], source.linearVelocity[1],
             source.linearVelocity[2]},
            {source.angularVelocity[0], source.angularVelocity[1],
             source.angularVelocity[2]}};
}

void check_cluster_payload(SceneDefinition &scene, unsigned requested) {
    const auto mesh_count = scene.meshes.size();
    const auto source_joints = scene.rigid_constraints;
    const auto anchor = scene.sphere_clusters.front().options.initial_state;
    std::string error;
    check(configure_dump_payload(scene, requested, error), error.c_str());
    check(scene.rigid_bodies.size() == requested+11 && dump_payload_count(scene) == requested,
          "P must replace the exact payload count without deleting the truck");
    check(scene.meshes.size() == mesh_count, "count edits must share the exported template mesh");
    check(std::count_if(scene.meshes.begin(), scene.meshes.end(),
              [](const auto &mesh) {
                  return mesh.name.ends_with("-sloped-liner");
              }) == 1,
          "count edits must retain one idempotent sloped bucket liner");
    Vec3 minimum{1e9F,1e9F,1e9F}, maximum{-1e9F,-1e9F,-1e9F};
    std::vector<Vec3> positions;
    for (const auto &body : scene.rigid_bodies) {
        if (body.source_name != "DumpPayload") continue;
        const auto &pose = body.options.initial_state;
        const auto p = rotate(conjugate(anchor.orientation), subtract(pose.position, anchor.position));
        minimum = {std::min(minimum.x,p.x),std::min(minimum.y,p.y),std::min(minimum.z,p.z)};
        maximum = {std::max(maximum.x,p.x),std::max(maximum.y,p.y),std::max(maximum.z,p.z)};
        check(body.mesh_indices == scene.sphere_clusters.front().mesh_indices,
              "every sphere must instance the same exported mesh");
        check(pose.orientation.x == anchor.orientation.x && pose.orientation.y == anchor.orientation.y &&
              pose.orientation.z == anchor.orientation.z && pose.orientation.w == anchor.orientation.w,
              "payload must follow the Empty rotation");
        for (const auto other : positions)
            check(length(subtract(p,other)) > 0.229F, "cluster packing must not overlap spheres");
        positions.push_back(p);
    }
    check(length({minimum.x+maximum.x,minimum.y+maximum.y,minimum.z+maximum.z}) < 1e-5F,
          "every payload count must be centered on SphereCluster, not the old load volume");
    for (std::size_t i = 0; i < source_joints.size(); ++i) {
        check(scene.rigid_constraints[i].body_a == source_joints[i].body_a &&
              scene.rigid_constraints[i].body_b == source_joints[i].body_b,
              "count edits must preserve truck joints");
    }
}
}

int main(int argc, char **argv) try {
    bool metadata_only = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view option(argv[argument]);
        if (option == "--metadata-only") metadata_only = true;
        else { std::cerr << "Invalid dump test option: " << option << '\n'; return 2; }
    }
    SceneDefinition source;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH, source, error), error.c_str());
    check(source.rigid_constraints.size() == 9, "truck needs four motors, four springs, one bucket actuator");
    check(source.sphere_clusters.size() == 1 && source.sphere_clusters[0].name == "SphereCluster",
          "loader must retain the Blender Empty as one non-simulated template");
    check(dump_payload_count(source) == 0, "source no longer needs an authored sphere array");
    // Simulate P edits, including changing away from and back to the default.
    auto edited = source;
    for (unsigned count : {100U,10U,24U,101U,1000U,100U}) check_cluster_payload(edited,count);
    auto &pose = edited.sphere_clusters[0].options.initial_state;
    pose.position = {10,5,-3};
    pose.orientation = {0,0.70710678F,0,0.70710678F};
    edited.hit_boxes.clear();
    for (unsigned count : {10U,100U,1000U}) check_cluster_payload(edited,count);
    for (unsigned invalid : {0U,1U,1001U}) {
        auto rejected = source;
        check(!configure_dump_payload(rejected, invalid, error), "reject out-of-range count overrides");
    }
    auto missing = source;
    missing.sphere_clusters.clear();
    check(!configure_dump_payload(missing,100,error), "missing Empty must report an actionable error");
    std::cout << "Dump truck: SphereCluster anchor, rotation and repeated P count edits passed\n";
    if (metadata_only) return 0;
    auto scene = source;
    check(configure_dump_payload(scene, default_dump_payload_count, error),
          error.c_str());
    const auto index_for = [&](const char *name) {
        const auto found = std::find_if(
            scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
            [&](const auto &body) { return body.source_name == name; });
        check(found != scene.rigid_bodies.end(), "truck body missing");
        return std::size_t(found - scene.rigid_bodies.begin());
    };
    const auto chassis = index_for("MotorChassis");
    const auto bucket = index_for("DumpBucket");

    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    if (!CommandScene::create(scene, 0U, simulation, instance, error)) {
        if (error.find("physical device") != std::string::npos) {
            std::cout << "SKIP: dump truck metadata passed; no Parallel Mater device\n";
            return 77;
        }
        throw std::runtime_error(error);
    }
    DumpTruckBed bed;
    check(bed.initialize(scene), "initialize dump lift");

    unsigned total_frames{};
    const auto step = [&](unsigned frames) {
        for (unsigned index = 0; index < frames; ++index) {
            check(bed.advance(simulation, scene, instance, frame,
                              1.0F / 60.0F,
                              error),
                  error.c_str());
            check(simulation.step(1.0F / 60.0F, 1U,
                                  {0.0F, -9.81F, 0.0F}, frame, error),
                  error.c_str());
            const auto truck = rigid_state(frame.rigid_states[chassis]);
            const auto load = rigid_state(frame.rigid_states[bucket]);
            ++total_frames;
            const bool bounded = std::isfinite(length(truck.position)) &&
                      std::isfinite(length(load.position)) &&
                      length(truck.linear_velocity) < 50.0F &&
                      length(load.linear_velocity) < 50.0F;
            if (!bounded)
                std::cerr << "dump-truck unbounded frame=" << total_frames
                          << " target-degrees="
                          << bed.angle() * 180.0F / 3.14159265358979323846F
                          << " truck-position=(" << truck.position.x << ','
                          << truck.position.y << ',' << truck.position.z
                          << ") truck-speed=" << length(truck.linear_velocity)
                          << " bucket-position=(" << load.position.x << ','
                          << load.position.y << ',' << load.position.z
                          << ") bucket-speed=" << length(load.linear_velocity)
                          << '\n';
            check(bounded, "truck and bucket must stay finite and bounded");
            const auto truck_up = rotate(
                truck.orientation, {0.0F, 1.0F, 0.0F});
            if (truck_up.y <= 0.8F)
                std::cerr << "dump-truck tipped frame=" << total_frames
                          << " target-degrees="
                          << bed.angle() * 180.0F / 3.14159265358979323846F
                          << " up=(" << truck_up.x << ',' << truck_up.y
                          << ',' << truck_up.z << ")\n";
            check(truck_up.y > 0.8F, "truck must remain on its wheels");
        }
    };
    step(1U);
    const auto state = [&](std::size_t index) {
        return rigid_state(frame.rigid_states[index]);
    };
    const auto authored_relative = multiply(
        conjugate(state(chassis).orientation), state(bucket).orientation);
    const auto bucket_angle = [&]() {
        const auto relative = multiply(
            conjugate(state(chassis).orientation), state(bucket).orientation);
        const auto delta = multiply(conjugate(authored_relative), relative);
        return 2.0F * std::atan2(
            length({delta.x, delta.y, delta.z}), std::fabs(delta.w));
    };
    const auto inside_count = [&]() {
        const auto box = state(bucket);
        unsigned inside{};
        for (std::size_t index = 0; index < scene.rigid_bodies.size();
             ++index) {
            if (scene.rigid_bodies[index].source_name != "DumpPayload")
                continue;
            const auto position = rotate(
                conjugate(box.orientation),
                subtract(state(index).position, box.position));
            inside += std::fabs(position.x) < 1.42F &&
                      position.y > -1.17F && position.y < 1.30F &&
                      std::fabs(position.z) < 1.17F;
        }
        return inside;
    };
    const auto drive = [&](float velocity) {
        for (std::size_t index = 0;
             index < scene.rigid_constraints.size(); ++index) {
            auto &definition = scene.rigid_constraints[index];
            if (definition.options.type != RigidConstraintType::motor)
                continue;
            auto updated = definition;
            updated.options.motor.angular_target_velocity = velocity;
            check(simulation.write_constraint(
                      instance.rigid_constraints[index], updated, error),
                  error.c_str());
            definition.options = updated.options;
        }
    };

    step(119U);
    check(inside_count() == 100U,
          "upright bucket must retain every payload sphere");
    const auto start = state(chassis);
    drive(-4.0F);
    step(120U);
    drive(0.0F);
    step(90U);
    check(length(subtract(state(chassis).position, start.position)) > 1.0F,
          "wheel motors must propel the loaded truck");
    check(inside_count() == 100U,
          "driving upright must retain the payload");
    check(bed.angle() == 0.0F, "driving must not tip the bucket");

    bed.toggle();
    step(480U);
    const float degrees =
        bucket_angle() * 180.0F / 3.14159265358979323846F;
    if (std::fabs(degrees - 100.0F) >= 1.0F)
        std::cerr << "dump-truck physical bucket angle=" << degrees
                  << " commanded="
                  << bed.angle() * 180.0F / 3.14159265358979323846F
                  << '\n';
    check(std::fabs(degrees - 100.0F) < 1.0F,
          "bucket must physically reach 100 degrees relative to chassis");
    const unsigned remaining = inside_count();
    if (remaining != 0U) {
        Vec3 minimum{1e9F,1e9F,1e9F}, maximum{-1e9F,-1e9F,-1e9F};
        float maximum_speed{};
        const auto box = state(bucket);
        for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index) {
            if (scene.rigid_bodies[index].source_name != "DumpPayload")
                continue;
            const auto payload = state(index);
            const auto local = rotate(conjugate(box.orientation),
                                      subtract(payload.position, box.position));
            minimum = {std::min(minimum.x, local.x),
                       std::min(minimum.y, local.y),
                       std::min(minimum.z, local.z)};
            maximum = {std::max(maximum.x, local.x),
                       std::max(maximum.y, local.y),
                       std::max(maximum.z, local.z)};
            maximum_speed = std::max(maximum_speed,
                                     length(payload.linear_velocity));
        }
        const auto local_gravity = rotate(conjugate(box.orientation),
                                          {0.0F, -9.81F, 0.0F});
        std::cerr << "dump-truck payload remaining=" << remaining
                  << " physical bucket angle=" << degrees
                  << " local-bounds=(" << minimum.x << ',' << minimum.y
                  << ',' << minimum.z << ")..(" << maximum.x << ','
                  << maximum.y << ',' << maximum.z << ") local-gravity=("
                  << local_gravity.x << ',' << local_gravity.y << ','
                  << local_gravity.z << ") max-speed=" << maximum_speed
                  << '\n';
    }
    check(remaining == 0U,
          "100-degree tip must empty the complete payload through the open top");
    bed.toggle();
    step(240U);
    if (!(bucket_angle() < 0.02F && bed.angle() == 0.0F))
        std::cerr << "dump-truck return angle="
                  << bucket_angle() * 180.0F / 3.14159265358979323846F
                  << " target="
                  << bed.angle() * 180.0F / 3.14159265358979323846F
                  << '\n';
    check(bucket_angle() < 0.02F && bed.angle() == 0.0F,
          "second Space must return bucket upright without resetting the truck");
    std::cout << "Dump truck command-API drive and unload passed\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
}
