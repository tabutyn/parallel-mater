// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/dump_truck.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace {
void check(bool okay, const char *message) { if (!okay) throw std::runtime_error(message); }
void require(Status status) { check(bool(status), status.message ? status.message : "physics operation failed"); }
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

void check_cluster_payload(SceneDefinition &scene, unsigned requested) {
    const auto mesh_count = scene.meshes.size();
    const auto source_joints = scene.rigid_constraints;
    const auto anchor = scene.sphere_clusters.front().options.initial_state;
    std::string error;
    check(configure_dump_payload(scene, requested, error), error.c_str());
    check(scene.rigid_bodies.size() == requested+11 && dump_payload_count(scene) == requested,
          "P must replace the exact payload count without deleting the truck");
    check(scene.meshes.size() == mesh_count, "count edits must share the exported template mesh");
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
    unsigned pass_limit = 0;
    bool metadata_only = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view option(argv[argument]);
        if (option == "--metadata-only") metadata_only = true;
        else if (option == "--passes" && argument+1 < argc) {
            const std::string_view value(argv[++argument]);
            const auto parsed = std::from_chars(value.data(),value.data()+value.size(),pass_limit);
            if (parsed.ec == std::errc{} && parsed.ptr == value.data()+value.size() && pass_limit <= 64) continue;
            std::cerr << "Invalid dump test pass limit: " << value << '\n'; return 2;
        } else { std::cerr << "Invalid dump test option: " << option << '\n'; return 2; }
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
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::cout << "SKIP: dump truck metadata passed; CUDA unavailable\n";
        return 77;
    }
    auto scene = source;
    check(configure_dump_payload(scene, default_dump_payload_count, error), error.c_str());
    const auto index_for = [&](const char *name) {
        const auto found = std::find_if(scene.rigid_bodies.begin(),scene.rigid_bodies.end(),
            [&](const auto &body) { return body.source_name == name; });
        check(found != scene.rigid_bodies.end(), "truck body missing");
        return std::size_t(found-scene.rigid_bodies.begin());
    };
    const auto chassis = index_for("MotorChassis"), bucket = index_for("DumpBucket");
    World world;
    SceneInstance instance;
    require(create_scene_world(scene, world, instance));
    DumpTruckBed bed;
    check(bed.initialize(scene), "initialize dump lift");
    std::cout << "Dump truck AVBD pass_limit=" << pass_limit << '\n';
    const auto state = [&](std::size_t i) {
        RigidBodyState result;
        require(world.read_rigid_body_state(instance.rigid_bodies[i],result));
        return result;
    };
    const auto authored_relative = multiply(conjugate(state(chassis).orientation),
                                            state(bucket).orientation);
    const auto bucket_angle = [&]() {
        const auto relative = multiply(conjugate(state(chassis).orientation),
                                        state(bucket).orientation);
        const auto delta = multiply(conjugate(authored_relative), relative);
        // Quaternion sign and world heading are arbitrary after Blender edits.
        return 2*std::atan2(length({delta.x,delta.y,delta.z}),std::fabs(delta.w));
    };
    const auto inside_count = [&]() {
        const auto box = state(bucket);
        unsigned inside{};
        for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
            if (scene.rigid_bodies[i].source_name != "DumpPayload") continue;
            const auto p = rotate(conjugate(box.orientation), subtract(state(i).position, box.position));
            inside += std::fabs(p.x) < 1.42F && p.y > -1.17F && p.y < 1.30F && std::fabs(p.z) < 1.17F;
        }
        return inside;
    };
    float maximum_pivot_error{};
    const auto step = [&](unsigned frames) {
        for (unsigned frame = 0; frame < frames; ++frame) {
            require(bed.advance(world, scene, instance, 1.0F/60));
            require(world.step({.timestep = 1.0F/60, .substeps = 8U,
                                .rigid_contact_pass_limit = pass_limit}));
            const auto truck = state(chassis), load = state(bucket);
            if (!(std::isfinite(length(truck.position)) && std::isfinite(length(load.position)) &&
                  length(truck.linear_velocity) < 50 && length(load.linear_velocity) < 50))
                std::cerr << "dump stability: frame=" << frame << " truck_speed="
                          << length(truck.linear_velocity) << " bucket_speed="
                          << length(load.linear_velocity) << " truck_y=" << truck.position.y
                          << " bucket_y=" << load.position.y << '\n';
            check(std::isfinite(length(truck.position)) && std::isfinite(length(load.position)) &&
                  length(truck.linear_velocity) < 50 && length(load.linear_velocity) < 50,
                  "truck and bucket must stay finite and bounded");
            const float upright = rotate(truck.orientation,{0,1,0}).y;
            if (!(upright > 0.8F))
                std::cerr << "dump attitude: frame=" << frame << " up_y=" << upright
                          << " speed=" << length(truck.linear_velocity) << " spin="
                          << length(truck.angular_velocity) << " bucket_angle="
                          << bucket_angle()*180/3.14159265358979323846F << '\n';
            check(upright > 0.8F, "truck must remain on its wheels");
            const auto lift = std::find_if(scene.rigid_constraints.begin(),scene.rigid_constraints.end(),
                [](const auto &joint) { return joint.name == "DumpLift"; });
            const auto a = rotate(truck.orientation, lift->options.local_anchor_a);
            const auto b = rotate(load.orientation, lift->options.local_anchor_b);
            const float drift = length({truck.position.x+a.x-load.position.x-b.x,
                truck.position.y+a.y-load.position.y-b.y, truck.position.z+a.z-load.position.z-b.z});
            maximum_pivot_error = std::max(maximum_pivot_error,drift);
            check(drift < 0.08F, "bucket rear pivot must remain attached while driving and tipping");
        }
    };
    const auto drive = [&](float velocity) {
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            if (joint.options.type != RigidConstraintType::motor) continue;
            auto options = joint.options;
            options.body_a = instance.rigid_bodies[joint.body_a];
            options.body_b = instance.rigid_bodies[joint.body_b];
            options.motor.angular_target_velocity = velocity;
            require(world.update_rigid_constraint(instance.rigid_constraints[i],options));
        }
    };
    step(120);
    std::cout << "upright payload=" << inside_count() << "/100\n";
    check(inside_count() == 100, "upright bucket must retain every payload sphere");
    const auto start = state(chassis);
    drive(-4);
    step(120);
    drive(0);
    step(90);
    const float distance = length(subtract(state(chassis).position,start.position));
    std::cout << "drive distance=" << distance << "m payload=" << inside_count() << "/100\n";
    check(distance > 1, "wheel motors must propel the loaded truck");
    check(inside_count() == 100, "driving upright must retain the payload");
    check(bed.angle() == 0, "driving must not tip the bucket");
    bed.toggle();
    step(480);
    const float degrees = bucket_angle()*180/3.14159265358979323846F;
    std::cout << "tip=" << degrees << "deg remaining=" << inside_count() << "/100 max pivot error=" << maximum_pivot_error << "m\n";
    check(std::fabs(degrees-100) < 1, "bucket must physically reach 100 degrees relative to chassis");
    check(inside_count() == 0, "100-degree tip must empty the complete payload through the open top");
    bed.toggle();
    step(240);
    check(bucket_angle() < 0.02F && bed.angle() == 0,
          "second Space must return bucket upright without resetting the truck");
    std::cout << "Dump truck: payload counts, loaded drive, 100-degree unload and lowering passed\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
}
