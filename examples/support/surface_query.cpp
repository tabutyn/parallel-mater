// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/surface_query.hpp>

#include "vector_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace parallel_mater::gallery {
namespace {

[[nodiscard]] Vec3 rotate(Quaternion q, Vec3 p) noexcept {
    const Vec3 t{2.0F * (q.y * p.z - q.z * p.y),
                 2.0F * (q.z * p.x - q.x * p.z),
                 2.0F * (q.x * p.y - q.y * p.x)};
    return {p.x + q.w * t.x + q.y * t.z - q.z * t.y,
            p.y + q.w * t.y + q.z * t.x - q.x * t.z,
            p.z + q.w * t.z + q.x * t.y - q.y * t.x};
}

} // namespace

bool StaticTriangleSurface::create(const SceneDefinition &scene,
                                   StaticTriangleSurface &output,
                                   std::string &error) {
    StaticTriangleSurface next;
    next.minimum_ = {std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max()};
    next.maximum_ = {-std::numeric_limits<float>::max(),
                     -std::numeric_limits<float>::max(),
                     -std::numeric_limits<float>::max()};
    std::vector<Triangle> pending;
    try {
        for (const RigidBodyDefinition &body : scene.rigid_bodies) {
            if (body.options.motion != MotionType::static_body) continue;
            const bool collision = !body.collision_mesh_indices.empty();
            const auto &meshes = collision ? scene.collision_meshes : scene.meshes;
            const auto &mesh_indices = collision ? body.collision_mesh_indices
                                                 : body.mesh_indices;
            const auto world_vertex = [&](Vec3 local) {
                const Vec3 p = rotate(body.options.initial_state.orientation,
                                      local);
                return math::add(p, body.options.initial_state.position);
            };
            for (std::uint32_t mesh_index : mesh_indices) {
                if (mesh_index >= meshes.size()) continue;
                const TriangleMesh &mesh = meshes[mesh_index];
                for (const Vertex &vertex : mesh.vertices) {
                    const Vec3 point = world_vertex(vertex.position);
                    next.minimum_.x = std::min(next.minimum_.x, point.x);
                    next.minimum_.y = std::min(next.minimum_.y, point.y);
                    next.minimum_.z = std::min(next.minimum_.z, point.z);
                    next.maximum_.x = std::max(next.maximum_.x, point.x);
                    next.maximum_.y = std::max(next.maximum_.y, point.y);
                    next.maximum_.z = std::max(next.maximum_.z, point.z);
                }
                for (std::size_t offset = 0U; offset + 2U < mesh.indices.size();
                     offset += 3U) {
                    const Vec3 a = world_vertex(
                        mesh.vertices[mesh.indices[offset]].position);
                    const Vec3 b = world_vertex(
                        mesh.vertices[mesh.indices[offset + 1U]].position);
                    const Vec3 c = world_vertex(
                        mesh.vertices[mesh.indices[offset + 2U]].position);
                    const Vec3 normal = math::cross(math::subtract(b, a),
                                                    math::subtract(c, a));
                    const float normal_length = math::length(normal);
                    if (normal_length < 1.0e-8F ||
                        std::fabs(normal.y) < normal_length * 0.2F) continue;
                    pending.push_back({a, b, c, normal.y / normal_length});
                }
            }
        }
        if (pending.empty()) {
            error = "scene has no queryable static collider surface";
            return false;
        }
        next.minimum_x_ = next.minimum_.x;
        next.minimum_z_ = next.minimum_.z;
        next.columns_ = static_cast<std::size_t>(std::ceil(
            (next.maximum_.x - next.minimum_.x) / next.cell_size_)) + 1U;
        next.rows_ = static_cast<std::size_t>(std::ceil(
            (next.maximum_.z - next.minimum_.z) / next.cell_size_)) + 1U;
        next.cells_.resize(next.columns_ * next.rows_);
        next.triangles_.reserve(pending.size());
        for (const Triangle &source : pending) {
            const std::uint32_t triangle = static_cast<std::uint32_t>(
                next.triangles_.size());
            next.triangles_.push_back(source);
            const float left = std::min({source.a.x, source.b.x, source.c.x});
            const float right = std::max({source.a.x, source.b.x, source.c.x});
            const float front = std::min({source.a.z, source.b.z, source.c.z});
            const float back = std::max({source.a.z, source.b.z, source.c.z});
            const auto cell = [&](float value, float minimum,
                                  std::size_t limit) {
                return std::min(limit - 1U, static_cast<std::size_t>(
                    std::max(0.0F, std::floor((value - minimum) /
                                              next.cell_size_))));
            };
            const std::size_t first_column = cell(
                left, next.minimum_.x, next.columns_);
            const std::size_t last_column = cell(
                right, next.minimum_.x, next.columns_);
            const std::size_t first_row = cell(
                front, next.minimum_.z, next.rows_);
            const std::size_t last_row = cell(
                back, next.minimum_.z, next.rows_);
            for (std::size_t row = first_row; row <= last_row; ++row)
                for (std::size_t column = first_column;
                     column <= last_column; ++column)
                    next.cells_[row * next.columns_ + column].push_back(triangle);
        }
    } catch (...) {
        error = "could not build static collider surface query";
        return false;
    }
    output = std::move(next);
    error.clear();
    return true;
}

std::optional<float> StaticTriangleSurface::height(
    float x, float z, SurfaceSelection selection,
    float *normal_y) const noexcept {
    const int column = static_cast<int>(std::floor((x - minimum_x_) / cell_size_));
    const int row = static_cast<int>(std::floor((z - minimum_z_) / cell_size_));
    if (column < 0 || row < 0 || static_cast<std::size_t>(column) >= columns_ ||
        static_cast<std::size_t>(row) >= rows_) return std::nullopt;
    std::optional<float> result;
    for (std::uint32_t index :
         cells_[static_cast<std::size_t>(row) * columns_ +
                static_cast<std::size_t>(column)]) {
        const Triangle &triangle = triangles_[index];
        const float abx = triangle.b.x - triangle.a.x;
        const float abz = triangle.b.z - triangle.a.z;
        const float acx = triangle.c.x - triangle.a.x;
        const float acz = triangle.c.z - triangle.a.z;
        const float determinant = abx * acz - abz * acx;
        if (std::fabs(determinant) < 1.0e-8F) continue;
        const float px = x - triangle.a.x, pz = z - triangle.a.z;
        const float u = (px * acz - pz * acx) / determinant;
        const float v = (abx * pz - abz * px) / determinant;
        if (u < -1.0e-4F || v < -1.0e-4F || u + v > 1.0001F) continue;
        const float value = triangle.a.y + u * (triangle.b.y - triangle.a.y) +
                            v * (triangle.c.y - triangle.a.y);
        const bool preferred = !result ||
            (selection == SurfaceSelection::lowest ? value < *result
                                                   : value > *result);
        if (preferred) {
            result = value;
            if (normal_y != nullptr) *normal_y = triangle.normal_y;
        }
    }
    return result;
}

} // namespace parallel_mater::gallery
