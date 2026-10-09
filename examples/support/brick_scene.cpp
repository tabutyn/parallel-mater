// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

#if defined(PARALLEL_MATER_GALLERY_METAL)
namespace parallel_mater::metal::gallery {
#else
namespace parallel_mater::gallery {
#endif
namespace {

bool is_brick(const RigidBodyDefinition &body) {
    return body.source_name == "Layer1" || body.source_name == "Layer2";
}

struct Bounds {
    Vec3 minimum{std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()};
};

Bounds mesh_bounds(const TriangleMesh &mesh) {
    Bounds result;
    for (const Vertex &vertex : mesh.vertices) {
        result.minimum.x = std::min(result.minimum.x, vertex.position.x);
        result.minimum.y = std::min(result.minimum.y, vertex.position.y);
        result.minimum.z = std::min(result.minimum.z, vertex.position.z);
        result.maximum.x = std::max(result.maximum.x, vertex.position.x);
        result.maximum.y = std::max(result.maximum.y, vertex.position.y);
        result.maximum.z = std::max(result.maximum.z, vertex.position.z);
    }
    return result;
}

float extent(float minimum, float maximum) {
    return std::max(maximum - minimum, 1.0e-4F);
}

TriangleMesh scaled_mesh(const TriangleMesh &source, float scale,
                         std::string name) {
    TriangleMesh result = source;
    result.name = std::move(name);
    for (Vertex &vertex : result.vertices) {
        vertex.position.x *= scale;
        vertex.position.y *= scale;
        vertex.position.z *= scale;
    }
    return result;
}

std::uint32_t columns_for(std::uint32_t count) {
    // Brick aspect is 2:1. This produces a physical wall near 4:3.
    return std::max(1U, static_cast<std::uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(count) * 2.0 / 3.0))));
}

} // namespace

bool make_brick_scene(const SceneDefinition &authored,
                      const ::parallel_mater::gallery::BrickSceneConfig &config,
                      SceneDefinition &output,
                      std::string &error) {
    if (!parallel_mater::gallery::validate_brick_config(config, error))
        return false;

    const RigidBodyDefinition *first = nullptr;
    const RigidBodyDefinition *second = nullptr;
    for (const auto &body : authored.rigid_bodies) {
        if (body.source_name == "Layer1" && first == nullptr) first = &body;
        if (body.source_name == "Layer2" && second == nullptr) second = &body;
    }
    if (first == nullptr || second == nullptr || first->mesh_indices.empty() ||
        second->mesh_indices.empty() ||
        first->mesh_indices.front() >= authored.meshes.size() ||
        second->mesh_indices.front() >= authored.meshes.size()) {
        error = "authored RigidBody scene has no Layer1/Layer2 brick templates";
        return false;
    }

    output = authored;
    output.rigid_bodies.erase(
        std::remove_if(output.rigid_bodies.begin(), output.rigid_bodies.end(),
                       is_brick),
        output.rigid_bodies.end());

    const std::uint32_t first_mesh = static_cast<std::uint32_t>(output.meshes.size());
    output.meshes.push_back(scaled_mesh(
        authored.meshes[first->mesh_indices.front()], config.brick_scale,
        "ProceduralBrickA"));
    const std::uint32_t second_mesh = static_cast<std::uint32_t>(output.meshes.size());
    output.meshes.push_back(scaled_mesh(
        authored.meshes[second->mesh_indices.front()], config.brick_scale,
        "ProceduralBrickB"));

    const Bounds brick_bounds = mesh_bounds(output.meshes[first_mesh]);
    const float width = extent(brick_bounds.minimum.x, brick_bounds.maximum.x);
    const float height = extent(brick_bounds.minimum.y, brick_bounds.maximum.y);
    const float depth = extent(brick_bounds.minimum.z, brick_bounds.maximum.z);

    float ball_diameter = 2.0F;
    for (const auto &body : authored.rigid_bodies) {
        if (body.source_name != "Icosphere" || body.mesh_indices.empty()) continue;
        const Bounds ball = mesh_bounds(authored.meshes[body.mesh_indices.front()]);
        ball_diameter = std::max({extent(ball.minimum.x, ball.maximum.x),
                                  extent(ball.minimum.y, ball.maximum.y),
                                  extent(ball.minimum.z, ball.maximum.z)});
        break;
    }

    const std::uint32_t base_per_plane = config.brick_count / config.wall_planes;
    const std::uint32_t extra_planes = config.brick_count % config.wall_planes;
    output.rigid_bodies.reserve(output.rigid_bodies.size() + config.brick_count);
    std::uint32_t serial = 0U;
    float wall_minimum_x = std::numeric_limits<float>::max();
    float wall_maximum_x = std::numeric_limits<float>::lowest();
    float wall_minimum_z = std::numeric_limits<float>::max();
    for (std::uint32_t plane = 0U; plane < config.wall_planes; ++plane) {
        const std::uint32_t plane_count = base_per_plane + (plane < extra_planes);
        const std::uint32_t columns = columns_for(plane_count);
        const float wall_z = -static_cast<float>(plane) *
            (depth + ball_diameter);
        std::uint32_t placed = 0U;
        for (std::uint32_t row = 0U; placed < plane_count; ++row) {
            // A full-width row alternates with a centered row one brick
            // shorter. This supplies the half-brick bond without balancing an
            // outer brick on exactly half of the brick below it.
            const std::uint32_t capacity = row % 2U == 0U || columns == 1U
                ? columns : columns - 1U;
            const std::uint32_t row_count =
                std::min(capacity, plane_count - placed);
            const float row_width = static_cast<float>(row_count) * width;
            for (std::uint32_t column = 0U; column < row_count; ++column) {
                RigidBodyDefinition brick = row % 2U == 0U ? *first : *second;
                brick.name = "Brick_" + std::to_string(serial);
                brick.source_name = row % 2U == 0U ? "Layer1" : "Layer2";
                brick.mesh_indices = {row % 2U == 0U ? first_mesh : second_mesh};
                brick.collision_mesh_indices.clear();
                brick.options.mass *= config.brick_scale * config.brick_scale *
                                      config.brick_scale;
                brick.options.collision_margin *= config.brick_scale;
                brick.options.inertia_diagonal = {};
                brick.options.initial_state.position = {
                    -row_width * 0.5F + width * 0.5F +
                        static_cast<float>(column) * width,
                    height * 0.5F + static_cast<float>(row) * height,
                    wall_z};
                wall_minimum_x = std::min(
                    wall_minimum_x, brick.options.initial_state.position.x - width * 0.5F);
                wall_maximum_x = std::max(
                    wall_maximum_x, brick.options.initial_state.position.x + width * 0.5F);
                wall_minimum_z = std::min(wall_minimum_z, wall_z - depth * 0.5F);
                output.rigid_bodies.push_back(std::move(brick));
                ++placed;
                ++serial;
            }
        }
    }

    if (serial != config.brick_count) {
        error = "procedural brick generator did not place the exact count";
        return false;
    }

    // Extend a private copy of the authored floor so every wall and the ball's
    // approach remain supported. No physics or renderer scaling is required.
    for (auto &body : output.rigid_bodies) {
        if ((body.source_name != "Ground" && body.name != "Ground.001") ||
            body.mesh_indices.empty() ||
            body.mesh_indices.front() >= output.meshes.size()) continue;
        TriangleMesh floor = output.meshes[body.mesh_indices.front()];
        floor.name = "ProceduralBrickFloor";
        const Bounds floor_bounds = mesh_bounds(floor);
        const float source_width = extent(floor_bounds.minimum.x, floor_bounds.maximum.x);
        const float source_depth = extent(floor_bounds.minimum.z, floor_bounds.maximum.z);
        float approach_z = 12.0F;
        for (const auto &candidate : output.rigid_bodies)
            if (candidate.source_name == "Icosphere")
                approach_z = candidate.options.initial_state.position.z;
        const float requested_minimum_x = wall_minimum_x - 2.0F;
        const float requested_maximum_x = wall_maximum_x + 2.0F;
        const float requested_minimum_z = wall_minimum_z - 3.0F;
        const float requested_maximum_z = std::max(approach_z + 3.0F, 3.0F);
        const float target_minimum_x =
            std::min(floor_bounds.minimum.x, requested_minimum_x);
        const float target_maximum_x =
            std::max(floor_bounds.maximum.x, requested_maximum_x);
        const float target_minimum_z =
            std::min(floor_bounds.minimum.z, requested_minimum_z);
        const float target_maximum_z =
            std::max(floor_bounds.maximum.z, requested_maximum_z);
        if (target_minimum_x == floor_bounds.minimum.x &&
            target_maximum_x == floor_bounds.maximum.x &&
            target_minimum_z == floor_bounds.minimum.z &&
            target_maximum_z == floor_bounds.maximum.z) {
            break;
        }
        for (Vertex &vertex : floor.vertices) {
            const float x = (vertex.position.x - floor_bounds.minimum.x) / source_width;
            const float z = (vertex.position.z - floor_bounds.minimum.z) / source_depth;
            vertex.position.x = target_minimum_x +
                x * (target_maximum_x - target_minimum_x);
            vertex.position.z = target_minimum_z +
                z * (target_maximum_z - target_minimum_z);
        }
        body.mesh_indices = {
            static_cast<std::uint32_t>(output.meshes.size())};
        // The enlarged render triangles are also the collision surface. The
        // authored floor may carry a private collision mesh with the original
        // bounds, which would leave later walls unsupported.
        body.collision_mesh_indices.clear();
        output.meshes.push_back(std::move(floor));
        break;
    }
    error.clear();
    return true;
}

#if defined(PARALLEL_MATER_GALLERY_METAL)
} // namespace parallel_mater::metal::gallery
#else
} // namespace parallel_mater::gallery
#endif
