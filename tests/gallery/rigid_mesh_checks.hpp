// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>
#include "../../examples/support/vector_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace rigid_mesh_checks {
using namespace parallel_mater;
using namespace parallel_mater::gallery;
using namespace parallel_mater::gallery::math;

inline Vec3 rotate(Quaternion q, Vec3 value) {
    const Vec3 axis{q.x, q.y, q.z};
    const Vec3 twice = multiply(cross(axis, value), 2.0F);
    return add(value, add(multiply(twice, q.w), cross(axis, twice)));
}

inline float triangle_distance_squared(Vec3 point, const std::array<Vec3, 3> &triangle) {
    const Vec3 ab = subtract(triangle[1], triangle[0]);
    const Vec3 ac = subtract(triangle[2], triangle[0]);
    const Vec3 normal = cross(ab, ac);
    const Vec3 relative = subtract(point, triangle[0]);
    const float normal_squared = dot(normal, normal);
    float distance = 1.0e30F;
    if (normal_squared > 1.0e-20F) {
        const Vec3 projection = subtract(relative,
            multiply(normal, dot(relative, normal) / normal_squared));
        const float v = dot(cross(projection, ac), normal) / normal_squared;
        const float w = dot(cross(ab, projection), normal) / normal_squared;
        if (v >= 0.0F && w >= 0.0F && v + w <= 1.0F)
            distance = dot(relative, normal) * dot(relative, normal) / normal_squared;
    }
    for (std::size_t edge = 0; edge < 3; ++edge) {
        const Vec3 origin = triangle[edge];
        const Vec3 direction = subtract(triangle[(edge + 1) % 3], origin);
        const float t = std::clamp(dot(subtract(point, origin), direction) /
            std::max(dot(direction, direction), 1.0e-20F), 0.0F, 1.0F);
        const Vec3 delta = subtract(point, add(origin, multiply(direction, t)));
        distance = std::min(distance, dot(delta, delta));
    }
    return distance;
}

// Independent CPU check of the actual collision geometry, not the solver's
// reported contact depths. Sample vertices, edge midpoints and face interiors
// in both directions: a contact event alone does not prove non-penetration.
// The authored gear/rack are closed meshes. Parity ignores winding, and equal
// ray hits on shared triangle edges are counted only once.
class MeshSamples {
  public:
    MeshSamples(const SceneDefinition &scene, std::size_t body_index) {
        const auto &body = scene.rigid_bodies[body_index];
        const auto &meshes = body.collision_mesh_indices.empty()
            ? scene.meshes : scene.collision_meshes;
        const auto &indices = body.collision_mesh_indices.empty()
            ? body.mesh_indices : body.collision_mesh_indices;
        for (const auto mesh_index : indices) {
            const auto &mesh = meshes[mesh_index];
            for (std::size_t index = 0; index < mesh.indices.size(); index += 3) {
                const std::array<Vec3, 3> triangle{
                    mesh.vertices[mesh.indices[index]].position,
                    mesh.vertices[mesh.indices[index + 1]].position,
                    mesh.vertices[mesh.indices[index + 2]].position};
                triangles_.push_back(triangle);
                for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                    samples_.push_back(triangle[vertex]);
                    samples_.push_back(multiply(add(triangle[vertex],
                        triangle[(vertex + 1) % 3]), 0.5F));
                }
                samples_.push_back(multiply(add(add(triangle[0], triangle[1]),
                                               triangle[2]), 1.0F / 3.0F));
            }
        }
        for (const auto point : samples_) {
            minimum_ = {std::min(minimum_.x, point.x), std::min(minimum_.y, point.y),
                        std::min(minimum_.z, point.z)};
            maximum_ = {std::max(maximum_.x, point.x), std::max(maximum_.y, point.y),
                        std::max(maximum_.z, point.z)};
        }
    }

    float penetration(const RigidBodyState &state, const MeshSamples &other,
                      const RigidBodyState &other_state) const {
        const auto q = other_state.orientation;
        const Quaternion inverse{-q.x, -q.y, -q.z, q.w};
        float maximum = 0.0F;
        for (const auto sample : samples_) {
            const Vec3 world = add(state.position, rotate(state.orientation, sample));
            maximum = std::max(maximum, other.depth(
                rotate(inverse, subtract(world, other_state.position))));
        }
        return maximum;
    }

  private:
    float depth(Vec3 point) const {
        if (point.x <= minimum_.x || point.x >= maximum_.x ||
            point.y <= minimum_.y || point.y >= maximum_.y ||
            point.z <= minimum_.z || point.z >= maximum_.z) return 0.0F;
        // Use double precision for parity. A grazing ray can enter and leave
        // a tooth only micrometers apart; merging those distinct crossings
        // turns an exterior point into a false, deeply penetrating sample.
        using PreciseVector = std::array<double, 3>;
        const auto difference = [](Vec3 a, Vec3 b) -> PreciseVector {
            return {double(a.x) - b.x, double(a.y) - b.y, double(a.z) - b.z};
        };
        const auto cross_product = [](PreciseVector a, PreciseVector b) -> PreciseVector {
            return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0]};
        };
        const auto inner_product = [](PreciseVector a, PreciseVector b) {
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        };
        constexpr PreciseVector ray{0.927, 0.357, 0.111};
        std::vector<double> hits;
        for (const auto &triangle : triangles_) {
            const auto ab = difference(triangle[1], triangle[0]);
            const auto ac = difference(triangle[2], triangle[0]);
            const auto h = cross_product(ray, ac);
            const double determinant = inner_product(ab, h);
            if (std::fabs(determinant) < 1.0e-14) continue;
            const auto relative = difference(point, triangle[0]);
            const double u = inner_product(relative, h) / determinant;
            if (u < 0.0 || u > 1.0) continue;
            const auto q = cross_product(relative, ab);
            const double v = inner_product(ray, q) / determinant;
            if (v < 0.0 || u + v > 1.0) continue;
            const double time = inner_product(ac, q) / determinant;
            if (time > 1.0e-9) hits.push_back(time);
        }
        std::sort(hits.begin(), hits.end());
        hits.erase(std::unique(hits.begin(), hits.end(), [](double a, double b) {
            return std::fabs(a - b) < 1.0e-10;
        }), hits.end());
        if (hits.size() % 2 == 0) return 0.0F;
        float distance_squared = 1.0e30F;
        for (const auto &triangle : triangles_)
            distance_squared = std::min(distance_squared,
                triangle_distance_squared(point, triangle));
        return std::sqrt(distance_squared);
    }

    std::vector<std::array<Vec3, 3>> triangles_;
    std::vector<Vec3> samples_;
    Vec3 minimum_{1.0e30F, 1.0e30F, 1.0e30F};
    Vec3 maximum_{-1.0e30F, -1.0e30F, -1.0e30F};
};
} // namespace rigid_mesh_checks
