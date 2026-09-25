// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace parallel_mater::gallery {
namespace {

constexpr float k_bounds_epsilon = 1.0e-4F;

class FlatJson {
  public:
    explicit FlatJson(const char *json) : json_(json != nullptr ? json : "") {}

    [[nodiscard]] std::optional<std::string> string(std::string_view key) const {
        const std::optional<std::string_view> value = find(key);
        if (!value || value->empty() || value->front() != '"') {
            return std::nullopt;
        }
        std::string result;
        bool escaped = false;
        for (std::size_t index = 1; index < value->size(); ++index) {
            const char character = (*value)[index];
            if (escaped) {
                result.push_back(character);
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                return result;
            } else {
                result.push_back(character);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<double> number(std::string_view key) const {
        const std::optional<std::string_view> value = find(key);
        if (!value) {
            return std::nullopt;
        }
        double result = 0.0;
        const char *begin = value->data();
        const char *end = begin + value->size();
        const auto parsed = std::from_chars(begin, end, result);
        if (parsed.ec != std::errc{}) {
            return std::nullopt;
        }
        return result;
    }

    [[nodiscard]] std::optional<bool> boolean(std::string_view key) const {
        const std::optional<std::string_view> value = find(key);
        if (!value) {
            return std::nullopt;
        }
        if (value->starts_with("true")) {
            return true;
        }
        if (value->starts_with("false")) {
            return false;
        }
        return std::nullopt;
    }

  private:
    [[nodiscard]] std::optional<std::string_view>
    find(std::string_view key) const {
        std::string quoted;
        quoted.reserve(key.size() + 2U);
        quoted.push_back('"');
        quoted.append(key);
        quoted.push_back('"');
        const std::size_t key_position = json_.find(quoted);
        if (key_position == std::string_view::npos) {
            return std::nullopt;
        }
        const std::size_t colon = json_.find(':', key_position + quoted.size());
        if (colon == std::string_view::npos) {
            return std::nullopt;
        }
        const std::size_t value = json_.find_first_not_of(" \t\r\n", colon + 1U);
        if (value == std::string_view::npos) {
            return std::nullopt;
        }
        return json_.substr(value);
    }

    std::string_view json_;
};

[[nodiscard]] bool finite(float value) { return std::isfinite(value); }

[[nodiscard]] bool finite(Vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

[[nodiscard]] Vec3 add(Vec3 first, Vec3 second) {
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

[[nodiscard]] Vec3 subtract(Vec3 first, Vec3 second) {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

[[nodiscard]] Vec3 multiply(Vec3 value, Vec3 scale) {
    return {value.x * scale.x, value.y * scale.y, value.z * scale.z};
}

[[nodiscard]] Vec3 multiply(Vec3 value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] float dot(Vec3 first, Vec3 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

[[nodiscard]] Vec3 cross(Vec3 first, Vec3 second) {
    return {first.y * second.z - first.z * second.y,
            first.z * second.x - first.x * second.z,
            first.x * second.y - first.y * second.x};
}

[[nodiscard]] Vec3 normalize(Vec3 value) {
    const float length = std::sqrt(dot(value, value));
    return length > 1.0e-8F ? multiply(value, 1.0F / length)
                            : Vec3{0.0F, 1.0F, 0.0F};
}

[[nodiscard]] Vec3 rotate(Quaternion orientation, Vec3 value) {
    const Vec3 q{orientation.x, orientation.y, orientation.z};
    const Vec3 twice_cross = multiply(cross(q, value), 2.0F);
    return add(value,
               add(multiply(twice_cross, orientation.w), cross(q, twice_cross)));
}

[[nodiscard]] bool read_metadata(const cgltf_node &node,
                                 RigidBodyOptions &options,
                                 bool &checkerboard,
                                 std::string &error) {
    const FlatJson extras(node.extras.data);
    const std::optional<double> schema = extras.number("pm_schema");
    const std::optional<std::string> system = extras.string("pm_system");
    if (!schema && !system) {
        return false;
    }
    const std::string node_name = node.name != nullptr ? node.name : "unnamed node";
    if (!schema || *schema != 2.0) {
        error = node_name + ": unsupported or missing pm_schema";
        return false;
    }
    if (!system || *system != "rigid_body") {
        error = node_name + ": unsupported pm_system";
        return false;
    }
    const std::optional<std::string> motion = extras.string("pm_motion");
    if (!motion) {
        error = node_name + ": pm_motion is required";
        return false;
    }
    if (*motion == "static") {
        options.motion = MotionType::static_body;
    } else if (*motion == "kinematic") {
        options.motion = MotionType::kinematic;
    } else if (*motion == "dynamic") {
        options.motion = MotionType::dynamic;
    } else {
        error = node_name + ": invalid pm_motion";
        return false;
    }

    const std::optional<double> mass = extras.number("pm_mass");
    if (mass) {
        options.mass = static_cast<float>(*mass);
    } else if (options.motion == MotionType::dynamic) {
        error = node_name + ": dynamic bodies require pm_mass";
        return false;
    }
    if (const std::optional<double> friction = extras.number("pm_friction")) {
        options.friction = static_cast<float>(*friction);
    }
    if (const std::optional<double> restitution =
            extras.number("pm_restitution")) {
        options.restitution = static_cast<float>(*restitution);
    }
    if (const std::optional<double> damping =
            extras.number("pm_linear_damping")) {
        options.linear_damping = static_cast<float>(*damping);
    }
    if (const std::optional<double> damping =
            extras.number("pm_angular_damping")) {
        options.angular_damping = static_cast<float>(*damping);
    }
    if (const std::optional<double> margin =
            extras.number("pm_collision_margin")) {
        options.collision_margin = static_cast<float>(*margin);
    }
    checkerboard = extras.boolean("pm_checkerboard").value_or(false);
    return true;
}

[[nodiscard]] Vec3 node_scale(const cgltf_node &node) {
    return node.has_scale
               ? Vec3{node.scale[0], node.scale[1], node.scale[2]}
               : Vec3{1.0F, 1.0F, 1.0F};
}

[[nodiscard]] RigidBodyState node_state(const cgltf_node &node) {
    RigidBodyState state{};
    if (node.has_translation) {
        state.position = {node.translation[0], node.translation[1],
                          node.translation[2]};
    }
    if (node.has_rotation) {
        state.orientation = {node.rotation[0], node.rotation[1], node.rotation[2],
                             node.rotation[3]};
    }
    return state;
}

[[nodiscard]] Vec3 material_color(const cgltf_material *material) {
    if (material == nullptr || !material->has_pbr_metallic_roughness) {
        return {0.7F, 0.7F, 0.7F};
    }
    const cgltf_float *color =
        material->pbr_metallic_roughness.base_color_factor;
    return {color[0], color[1], color[2]};
}

[[nodiscard]] bool material_visible(const cgltf_material *material) {
    if (material == nullptr || !material->has_pbr_metallic_roughness ||
        material->alpha_mode == cgltf_alpha_mode_opaque) {
        return true;
    }
    const float alpha = material->pbr_metallic_roughness.base_color_factor[3];
    if (material->alpha_mode == cgltf_alpha_mode_mask)
        return alpha >= material->alpha_cutoff;
    return alpha > 0.0F;
}

[[nodiscard]] bool append_primitive(const cgltf_primitive &primitive,
                                    Vec3 scale, bool checkerboard,
                                    std::string_view name,
                                    TriangleMesh &mesh,
                                    std::string &error) {
    if (primitive.type != cgltf_primitive_type_triangles) {
        error = std::string(name) + ": only triangle primitives are supported";
        return false;
    }
    const cgltf_accessor *positions =
        cgltf_find_accessor(&primitive, cgltf_attribute_type_position, 0);
    const cgltf_accessor *normals =
        cgltf_find_accessor(&primitive, cgltf_attribute_type_normal, 0);
    const cgltf_accessor *uvs =
        cgltf_find_accessor(&primitive, cgltf_attribute_type_texcoord, 0);
    if (positions == nullptr || positions->type != cgltf_type_vec3) {
        error = std::string(name) + ": POSITION vec3 data is required";
        return false;
    }
    if (positions->count > std::numeric_limits<std::uint32_t>::max()) {
        error = std::string(name) + ": vertex count exceeds uint32 range";
        return false;
    }

    mesh.name = std::string(name);
    mesh.base_color = material_color(primitive.material);
    mesh.visible = material_visible(primitive.material);
    mesh.checkerboard = checkerboard;
    mesh.vertices.resize(positions->count);
    const Vec3 inverse_scale{1.0F / scale.x, 1.0F / scale.y, 1.0F / scale.z};
    for (cgltf_size index = 0; index < positions->count; ++index) {
        std::array<cgltf_float, 3> position{};
        if (!cgltf_accessor_read_float(positions, index, position.data(),
                                       position.size())) {
            error = std::string(name) + ": failed to read a position";
            return false;
        }
        mesh.vertices[index].position =
            multiply({position[0], position[1], position[2]}, scale);
        if (!finite(mesh.vertices[index].position)) {
            error = std::string(name) + ": non-finite position";
            return false;
        }
        if (normals != nullptr) {
            std::array<cgltf_float, 3> normal{};
            if (!cgltf_accessor_read_float(normals, index, normal.data(),
                                           normal.size())) {
                error = std::string(name) + ": failed to read a normal";
                return false;
            }
            mesh.vertices[index].normal = normalize(
                multiply({normal[0], normal[1], normal[2]}, inverse_scale));
        }
        if (uvs != nullptr) {
            std::array<cgltf_float, 2> uv{};
            if (uvs->type != cgltf_type_vec2 ||
                !cgltf_accessor_read_float(uvs, index, uv.data(), uv.size()) ||
                !std::isfinite(uv[0]) || !std::isfinite(uv[1])) {
                error = std::string(name) + ": invalid TEXCOORD_0";
                return false;
            }
            mesh.vertices[index].uv = {uv[0], uv[1]};
        }
    }

    const cgltf_size index_count = primitive.indices != nullptr
                                       ? primitive.indices->count
                                       : positions->count;
    if (index_count % 3U != 0U ||
        index_count > std::numeric_limits<std::uint32_t>::max()) {
        error = std::string(name) + ": triangle index count is invalid";
        return false;
    }
    mesh.indices.resize(index_count);
    for (cgltf_size index = 0; index < index_count; ++index) {
        const cgltf_size source = primitive.indices != nullptr
                                      ? cgltf_accessor_read_index(primitive.indices,
                                                                  index)
                                      : index;
        if (source >= mesh.vertices.size()) {
            error = std::string(name) + ": index is outside the vertex buffer";
            return false;
        }
        mesh.indices[index] = static_cast<std::uint32_t>(source);
    }

    if (normals == nullptr) {
        for (std::size_t index = 0; index < mesh.indices.size(); index += 3U) {
            Vertex &first = mesh.vertices[mesh.indices[index]];
            Vertex &second = mesh.vertices[mesh.indices[index + 1U]];
            Vertex &third = mesh.vertices[mesh.indices[index + 2U]];
            const Vec3 normal = cross(subtract(second.position, first.position),
                                      subtract(third.position, first.position));
            first.normal = add(first.normal, normal);
            second.normal = add(second.normal, normal);
            third.normal = add(third.normal, normal);
        }
        for (Vertex &vertex : mesh.vertices) {
            vertex.normal = normalize(vertex.normal);
        }
    }
    return true;
}

[[nodiscard]] bool sample_initial_volume(
    const cgltf_node &node, Vec3 scale, Vec3 velocity, float spacing,
    std::vector<FluidParticle> &particles, std::string &error) {
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    for (cgltf_size primitive_index = 0U;
         primitive_index < node.mesh->primitives_count; ++primitive_index) {
        TriangleMesh mesh{};
        if (!append_primitive(node.mesh->primitives[primitive_index], scale,
                              false, "fluid initial volume", mesh, error))
            return false;
        const auto base = static_cast<std::uint32_t>(vertices.size());
        for (const Vertex &vertex : mesh.vertices)
            vertices.push_back(vertex.position);
        for (const std::uint32_t index : mesh.indices)
            indices.push_back(base + index);
    }
    const FluidGeometrySource source{{vertices.data(), vertices.size()},
        {indices.data(), indices.size()}, node_state(node), velocity, spacing};
    const Status status = sample_fluid_geometry(source, particles);
    if (!status)
        error = status.message != nullptr ? status.message :
            "fluid geometry sampling failed";
    return status.ok();
}

[[nodiscard]] bool finalize_body_geometry(SceneDefinition &scene,
                                          RigidBodyDefinition &body,
                                          std::string &error,
                                          Vec3 *local_center = nullptr) {
    Vec3 minimum{std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{-std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max()};
    for (const std::uint32_t mesh_index : body.mesh_indices) {
        for (const Vertex &vertex : scene.meshes[mesh_index].vertices) {
            minimum.x = std::min(minimum.x, vertex.position.x);
            minimum.y = std::min(minimum.y, vertex.position.y);
            minimum.z = std::min(minimum.z, vertex.position.z);
            maximum.x = std::max(maximum.x, vertex.position.x);
            maximum.y = std::max(maximum.y, vertex.position.y);
            maximum.z = std::max(maximum.z, vertex.position.z);
        }
    }
    const Vec3 center = multiply(add(minimum, maximum), 0.5F);
    if (local_center != nullptr) *local_center = center;
    const Vec3 half = multiply(subtract(maximum, minimum), 0.5F);
    if (!finite(center) || !finite(half) || half.x < k_bounds_epsilon ||
        half.z < k_bounds_epsilon) {
        error = body.name + ": render bounds are invalid";
        return false;
    }
    for (const std::uint32_t mesh_index : body.mesh_indices) {
        for (Vertex &vertex : scene.meshes[mesh_index].vertices) {
            vertex.position = subtract(vertex.position, center);
        }
    }
    for (const std::uint32_t mesh_index : body.collision_mesh_indices) {
        for (Vertex &vertex : scene.collision_meshes[mesh_index].vertices) {
            vertex.position = subtract(vertex.position, center);
        }
    }
    body.options.initial_state.position = add(
        body.options.initial_state.position,
        rotate(body.options.initial_state.orientation, center));

    return true;
}

struct CollisionProxy {
    RigidBodyState state{};
    std::vector<std::uint32_t> mesh_indices{};
};

[[nodiscard]] bool same_transform(const RigidBodyState &first,
                                  const RigidBodyState &second) {
    constexpr float tolerance = 1.0e-5F;
    const auto near = [](float left, float right) {
        return std::fabs(left - right) <= tolerance;
    };
    const bool same_position = near(first.position.x, second.position.x) &&
                               near(first.position.y, second.position.y) &&
                               near(first.position.z, second.position.z);
    const float orientation_dot =
        first.orientation.x * second.orientation.x +
        first.orientation.y * second.orientation.y +
        first.orientation.z * second.orientation.z +
        first.orientation.w * second.orientation.w;
    return same_position && std::fabs(std::fabs(orientation_dot) - 1.0F) <=
                                tolerance;
}

[[nodiscard]] Quaternion rotation_z(float radians) {
    return {0.0F, 0.0F, std::sin(radians * 0.5F),
            std::cos(radians * 0.5F)};
}

void append_quad(std::vector<std::uint32_t> &indices, std::uint32_t first,
                 std::uint32_t second, std::uint32_t third,
                 std::uint32_t fourth) {
    indices.insert(indices.end(), {first, second, third, first, third, fourth});
}

[[nodiscard]] TriangleMesh make_open_cube(std::string name, float half_extent,
                                          bool remove_right, Vec3 color,
                                          bool checkerboard) {
    const std::array<Vec3, 8> corners{{
        {-half_extent, -half_extent, -half_extent},
        {half_extent, -half_extent, -half_extent},
        {half_extent, half_extent, -half_extent},
        {-half_extent, half_extent, -half_extent},
        {-half_extent, -half_extent, half_extent},
        {half_extent, -half_extent, half_extent},
        {half_extent, half_extent, half_extent},
        {-half_extent, half_extent, half_extent},
    }};
    TriangleMesh result{};
    result.name = std::move(name);
    result.base_color = color;
    result.checkerboard = checkerboard;
    result.vertices.reserve(corners.size());
    for (const Vec3 corner : corners) {
        result.vertices.push_back({corner, normalize(corner)});
    }

    // Bottom, front, back, and left are always present. Top is open.
    append_quad(result.indices, 0U, 1U, 5U, 4U);
    append_quad(result.indices, 0U, 3U, 2U, 1U);
    append_quad(result.indices, 4U, 5U, 6U, 7U);
    append_quad(result.indices, 0U, 4U, 7U, 3U);
    if (!remove_right) {
        append_quad(result.indices, 1U, 2U, 6U, 5U);
    }
    return result;
}

[[nodiscard]] TriangleMesh make_cube_projected_sphere(float radius) {
    const std::array<Vec3, 14> cube_vertices{{
        {-1.0F, -1.0F, -1.0F}, {1.0F, -1.0F, -1.0F},
        {1.0F, 1.0F, -1.0F},   {-1.0F, 1.0F, -1.0F},
        {-1.0F, -1.0F, 1.0F},  {1.0F, -1.0F, 1.0F},
        {1.0F, 1.0F, 1.0F},    {-1.0F, 1.0F, 1.0F},
        {0.0F, -1.0F, 0.0F},   {0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, -1.0F},   {0.0F, 0.0F, 1.0F},
        {-1.0F, 0.0F, 0.0F},   {1.0F, 0.0F, 0.0F},
    }};
    TriangleMesh result{};
    result.name = "dump_sphere";
    result.base_color = {0.98F, 0.48F, 0.12F};
    result.vertices.reserve(cube_vertices.size());
    for (const Vec3 cube_vertex : cube_vertices) {
        const Vec3 position = multiply(normalize(cube_vertex), radius);
        result.vertices.push_back({position, normalize(position)});
    }
    const auto append_face = [&](std::uint32_t center,
                                 std::array<std::uint32_t, 4> corners) {
        for (std::size_t index = 0U; index < corners.size(); ++index) {
            result.indices.insert(result.indices.end(),
                                  {center, corners[index],
                                   corners[(index + 1U) % corners.size()]});
        }
    };
    append_face(8U, {0U, 1U, 5U, 4U});
    append_face(9U, {3U, 7U, 6U, 2U});
    append_face(10U, {0U, 3U, 2U, 1U});
    append_face(11U, {4U, 5U, 6U, 7U});
    append_face(12U, {0U, 4U, 7U, 3U});
    append_face(13U, {1U, 2U, 6U, 5U});
    return result;
}

} // namespace

bool load_glb_scene(const std::filesystem::path &path, SceneDefinition &output,
                    std::string &error) {
    output = {};
    error.clear();
    cgltf_options options{};
    cgltf_data *data = nullptr;
    const std::string path_string = path.string();
    cgltf_result result = cgltf_parse_file(&options, path_string.c_str(), &data);
    if (result != cgltf_result_success) {
        error = "failed to parse GLB: cgltf result " +
                std::to_string(static_cast<int>(result));
        return false;
    }
    const std::unique_ptr<cgltf_data, decltype(&cgltf_free)> owner(data,
                                                                  &cgltf_free);
    result = cgltf_load_buffers(&options, data, path_string.c_str());
    if (result != cgltf_result_success) {
        error = "failed to load GLB buffers: cgltf result " +
                std::to_string(static_cast<int>(result));
        return false;
    }
    result = cgltf_validate(data);
    if (result != cgltf_result_success) {
        error = "invalid GLB: cgltf result " +
                std::to_string(static_cast<int>(result));
        return false;
    }

    std::unordered_map<std::string, CollisionProxy> collision_proxies;
    for (cgltf_size node_index = 0; node_index < data->nodes_count; ++node_index) {
        const cgltf_node &node = data->nodes[node_index];
        if (node.extras.data == nullptr) {
            continue;
        }
        const FlatJson extras(node.extras.data);
        if (extras.string("pm_system").value_or("") != "collision_mesh") {
            continue;
        }
        const std::string node_name =
            extras.string("pm_name")
                .value_or(node.name != nullptr
                              ? node.name
                              : "collision_node_" + std::to_string(node_index));
        if (extras.number("pm_schema").value_or(0.0) != 2.0) {
            error = node_name + ": unsupported or missing pm_schema";
            return false;
        }
        if (node.parent != nullptr || node.has_matrix) {
            error = node_name +
                    ": collision proxies must be scene-root TRS nodes";
            return false;
        }
        if (node.mesh == nullptr || node.mesh->primitives_count == 0U) {
            error = node_name + ": collision proxy requires geometry";
            return false;
        }
        const Vec3 scale = node_scale(node);
        if (!finite(scale) || scale.x < k_bounds_epsilon ||
            scale.y < k_bounds_epsilon || scale.z < k_bounds_epsilon) {
            error = node_name + ": collision proxy scale is invalid";
            return false;
        }
        CollisionProxy proxy{.state = node_state(node)};
        for (cgltf_size primitive_index = 0;
             primitive_index < node.mesh->primitives_count; ++primitive_index) {
            TriangleMesh mesh{};
            const std::string mesh_name =
                node_name + "/primitive_" + std::to_string(primitive_index);
            if (!append_primitive(node.mesh->primitives[primitive_index], scale,
                                  false, mesh_name, mesh, error)) {
                return false;
            }
            proxy.mesh_indices.push_back(
                static_cast<std::uint32_t>(output.collision_meshes.size()));
            output.collision_meshes.push_back(std::move(mesh));
        }
        if (!collision_proxies.emplace(node_name, std::move(proxy)).second) {
            error = node_name + ": duplicate collision proxy name";
            return false;
        }
    }

    std::unordered_set<std::string> used_collision_proxies;
    struct SharedRenderMesh {
        Vec3 scale{};
        bool checkerboard{};
        std::vector<std::uint32_t> indices{};
        Vec3 center{};
    };
    std::unordered_map<const cgltf_mesh *, SharedRenderMesh> shared_render_meshes;
    for (cgltf_size node_index = 0; node_index < data->nodes_count; ++node_index) {
        const cgltf_node &node = data->nodes[node_index];
        if (node.extras.data == nullptr) {
            continue;
        }
        const FlatJson extras(node.extras.data);
        if (extras.string("pm_system").value_or("") != "rigid_body") {
            continue;
        }
        RigidBodyDefinition body{};
        body.name = node.name != nullptr ? node.name
                                         : "node_" + std::to_string(node_index);
        bool checkerboard = false;
        if (!read_metadata(node, body.options, checkerboard, error)) {
            if (error.empty()) {
                continue;
            }
            return false;
        }
        body.name = extras.string("pm_name").value_or(body.name);
        body.paintable = extras.boolean("pm_paintable").value_or(false);
        if (const auto resolution = extras.number("pm_paint_resolution")) {
            if (!std::isfinite(*resolution) || *resolution < 32.0 ||
                *resolution > 2048.0 || std::floor(*resolution) != *resolution) {
                error = body.name + ": pm_paint_resolution must be an integer from 32 to 2048";
                return false;
            }
            body.paint_resolution = static_cast<std::uint32_t>(*resolution);
        }
        if (node.parent != nullptr) {
            error = body.name + ": physics objects must be scene-root nodes";
            return false;
        }
        if (node.has_matrix) {
            error = body.name + ": matrix transforms are unsupported; export TRS";
            return false;
        }
        if (node.mesh == nullptr || node.mesh->primitives_count == 0U) {
            error = body.name + ": physics objects require render geometry";
            return false;
        }
        const Vec3 scale = node_scale(node);
        if (!finite(scale) || scale.x < k_bounds_epsilon ||
            scale.y < k_bounds_epsilon || scale.z < k_bounds_epsilon) {
            error = body.name + ": scale must be finite and positive";
            return false;
        }
        body.options.initial_state = node_state(node);
        const bool has_proxy = extras.string("pm_collision_proxy").has_value();
        const auto cached = shared_render_meshes.find(node.mesh);
        const bool reuse = !has_proxy && cached != shared_render_meshes.end() &&
            cached->second.scale.x == scale.x &&
            cached->second.scale.y == scale.y &&
            cached->second.scale.z == scale.z &&
            cached->second.checkerboard == checkerboard;
        if (reuse) {
            body.mesh_indices = cached->second.indices;
        } else {
            for (cgltf_size primitive_index = 0;
                 primitive_index < node.mesh->primitives_count; ++primitive_index) {
                TriangleMesh mesh{};
                const std::string mesh_name =
                    body.name + "/primitive_" + std::to_string(primitive_index);
                if (!append_primitive(node.mesh->primitives[primitive_index], scale,
                                      checkerboard, mesh_name, mesh, error)) {
                    return false;
                }
                body.mesh_indices.push_back(
                    static_cast<std::uint32_t>(output.meshes.size()));
                output.meshes.push_back(std::move(mesh));
            }
        }
        if (const std::optional<std::string> collision_name =
                extras.string("pm_collision_proxy")) {
            const auto proxy = collision_proxies.find(*collision_name);
            if (proxy == collision_proxies.end()) {
                error = body.name + ": collision proxy '" + *collision_name +
                        "' was not exported";
                return false;
            }
            if (!used_collision_proxies.insert(*collision_name).second) {
                error = body.name + ": collision proxy '" + *collision_name +
                        "' is already assigned to another body";
                return false;
            }
            if (!same_transform(body.options.initial_state,
                                proxy->second.state)) {
                error = body.name +
                        ": collision proxy must share the rigid-body transform";
                return false;
            }
            body.collision_mesh_indices = proxy->second.mesh_indices;
        }
        if (reuse) {
            body.options.initial_state.position = add(
                body.options.initial_state.position,
                rotate(body.options.initial_state.orientation,
                       cached->second.center));
        } else {
            Vec3 center{};
            if (!finalize_body_geometry(output, body, error, &center))
                return false;
            if (!has_proxy)
                shared_render_meshes.emplace(node.mesh,
                    SharedRenderMesh{scale, checkerboard, body.mesh_indices,
                                     center});
        }
        output.rigid_bodies.push_back(std::move(body));
    }
    for (cgltf_size node_index = 0; node_index < data->nodes_count; ++node_index) {
        const cgltf_node &node = data->nodes[node_index];
        if (node.extras.data == nullptr) continue;
        const FlatJson extras(node.extras.data);
        if (extras.string("pm_system").value_or("") != "cloth") continue;
        const std::string name = extras.string("pm_name").value_or(
            node.name != nullptr ? node.name : "cloth");
        if (extras.number("pm_schema").value_or(0.0) != 2.0 ||
            node.parent != nullptr || node.has_matrix || node.mesh == nullptr ||
            node.mesh->primitives_count != 1U ||
            extras.number("pm_pin_stiffness").value_or(0.0) != 1.0) {
            error = name + ": cloth needs schema 2, one root TRS mesh, and full pin stiffness";
            return false;
        }
        const float mass = static_cast<float>(
            extras.number("pm_vertex_mass").value_or(0.001));
        const float thickness = static_cast<float>(
            extras.number("pm_thickness").value_or(0.025));
        const float tear_ratio = static_cast<float>(
            extras.number("pm_tear_ratio").value_or(0.0));
        const float stretch_compliance = static_cast<float>(
            extras.number("pm_stretch_compliance").value_or(1.0e-6));
        const double solver_iterations =
            extras.number("pm_solver_iterations").value_or(8.0);
        const double paint_resolution =
            extras.number("pm_paint_resolution").value_or(512.0);
        const Vec3 scale = node_scale(node);
        if (!finite(scale) || scale.x <= 0.0F || scale.y <= 0.0F ||
            scale.z <= 0.0F || !finite(mass) || mass <= 0.0F ||
            !finite(thickness) || thickness <= 0.0F) {
            error = name + ": invalid cloth mass, thickness, or transform";
            return false;
        }
        if (!finite(tear_ratio) ||
            (tear_ratio != 0.0F && (tear_ratio <= 1.0F || tear_ratio > 10.0F)) ||
            !finite(stretch_compliance) || stretch_compliance < 0.0F ||
            solver_iterations < 1.0 || solver_iterations > 64.0 ||
            std::floor(solver_iterations) != solver_iterations ||
            paint_resolution < 32.0 || paint_resolution > 2048.0 ||
            std::floor(paint_resolution) != paint_resolution) {
            error = name + ": invalid cloth tear, solver, or paint settings";
            return false;
        }
        const auto encoded_pins = extras.string("pm_pin_vertices");
        if (!encoded_pins || encoded_pins->empty()) {
            error = name + ": cloth has no exported pin vertices";
            return false;
        }
        struct Pin { Vec3 position; float weight; bool matched{}; };
        std::vector<Pin> pins;
        const char *cursor = encoded_pins->c_str();
        const char *end = cursor + encoded_pins->size();
        while (cursor < end) {
            float fields[4]{};
            for (int component = 0; component < 4; ++component) {
                char *next = nullptr;
                fields[component] = std::strtof(cursor, &next);
                if (next == cursor || !std::isfinite(fields[component]) ||
                    (component < 3 && *next != ',') ||
                    (component == 3 && *next != ';' && *next != '\0')) {
                    error = name + ": invalid exported cloth pin coordinate";
                    return false;
                }
                cursor = next + (component < 3 || *next == ';' ? 1 : 0);
            }
            if (fields[3] <= 0.0F || fields[3] > 1.0F) {
                error = name + ": invalid cloth pin weight";
                return false;
            }
            pins.push_back({{fields[0], fields[1], fields[2]}, fields[3]});
        }
        TriangleMesh mesh{};
        if (!append_primitive(node.mesh->primitives[0], scale, false,
                              name, mesh, error)) return false;
        ClothDefinition cloth{};
        cloth.name = name;
        cloth.vertex_mass = mass;
        cloth.thickness = thickness;
        cloth.tear_ratio = tear_ratio;
        cloth.tear_requires_contact = extras.boolean(
            "pm_tear_requires_contact").value_or(false);
        cloth.stretch_compliance = stretch_compliance;
        cloth.solver_iterations = static_cast<std::uint32_t>(solver_iterations);
        cloth.paintable = extras.boolean("pm_paintable").value_or(false);
        cloth.paint_resolution = static_cast<std::uint32_t>(paint_resolution);
        cloth.mesh_index = static_cast<std::uint32_t>(output.meshes.size());
        cloth.inverse_masses.assign(mesh.vertices.size(), 1.0F / mass);
        const RigidBodyState state = node_state(node);
        for (std::size_t vertex = 0U; vertex < mesh.vertices.size(); ++vertex) {
            Vertex &point = mesh.vertices[vertex];
            for (Pin &pin : pins) {
                const Vec3 delta = subtract(point.position, pin.position);
                if (dot(delta, delta) > 1.0e-8F) continue;
                cloth.inverse_masses[vertex] =
                    (1.0F - pin.weight) / mass;
                pin.matched = true;
            }
            point.position = add(state.position,
                rotate(state.orientation, point.position));
            point.normal = rotate(state.orientation, point.normal);
        }
        if (std::any_of(pins.begin(), pins.end(),
                        [](const Pin &pin) { return !pin.matched; })) {
            error = name + ": cloth pin positions did not match exported mesh";
            return false;
        }
        output.meshes.push_back(std::move(mesh));
        output.cloths.push_back(std::move(cloth));
    }
    std::optional<float> initial_spacing;
    std::optional<float> initial_gravity_scale;
    for (cgltf_size node_index = 0; node_index < data->nodes_count;
         ++node_index) {
        const cgltf_node &node = data->nodes[node_index];
        if (node.extras.data == nullptr) continue;
        const FlatJson extras(node.extras.data);
        const std::string system = extras.string("pm_system").value_or("");
        if (system != "fluid_inflow" && system != "fluid_outflow" &&
            system != "fluid_initial_volume") continue;
        const std::string name = node.name != nullptr ? node.name : "fluid plane";
        if (extras.number("pm_schema").value_or(0.0) != 2.0 ||
            node.parent != nullptr || node.has_matrix || node.mesh == nullptr ||
            node.mesh->primitives_count == 0U) {
            error = name + ": fluid plane needs schema 2 and root TRS mesh";
            return false;
        }
        const Vec3 scale = node_scale(node);
        if (!finite(scale) || scale.x <= 0.0F || scale.y <= 0.0F ||
            scale.z <= 0.0F) {
            error = name + ": fluid plane scale is invalid";
            return false;
        }
        if (system == "fluid_initial_volume") {
            const float spacing = static_cast<float>(
                extras.number("pm_particle_spacing").value_or(0.06));
            const float gravity_scale = static_cast<float>(
                extras.number("pm_gravity_scale").value_or(1.0));
            const Vec3 velocity{
                static_cast<float>(extras.number("pm_velocity_x").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_y").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_z").value_or(0.0))};
            if (!finite(velocity) || !std::isfinite(spacing) ||
                spacing < 0.01F || spacing > 0.2F ||
                !std::isfinite(gravity_scale) ||
                gravity_scale <= 0.0F || gravity_scale > 10.0F ||
                (initial_spacing && std::fabs(*initial_spacing - spacing) > 1.0e-5F) ||
                (initial_gravity_scale &&
                 std::fabs(*initial_gravity_scale - gravity_scale) > 1.0e-5F)) {
                error = name + ": invalid or inconsistent initial fluid settings";
                return false;
            }
            initial_spacing = spacing;
            initial_gravity_scale = gravity_scale;
            if (!sample_initial_volume(node, scale, velocity, spacing,
                                       output.initial_particles, error)) {
                return false;
            }
            continue;
        }
        Vec3 minimum{FLT_MAX, FLT_MAX, FLT_MAX};
        Vec3 maximum{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (cgltf_size primitive_index = 0U;
             primitive_index < node.mesh->primitives_count; ++primitive_index) {
            const cgltf_accessor *positions = cgltf_find_accessor(
                &node.mesh->primitives[primitive_index],
                cgltf_attribute_type_position, 0);
            if (positions == nullptr || positions->type != cgltf_type_vec3) {
                error = name + ": fluid plane needs POSITION vec3 data";
                return false;
            }
            for (cgltf_size vertex = 0U; vertex < positions->count; ++vertex) {
                std::array<cgltf_float, 3> value{};
                if (!cgltf_accessor_read_float(positions, vertex,
                                               value.data(), value.size())) {
                    error = name + ": cannot read fluid plane vertex";
                    return false;
                }
                const Vec3 p{value[0] * scale.x, value[1] * scale.y,
                             value[2] * scale.z};
                minimum = {std::min(minimum.x, p.x),
                           std::min(minimum.y, p.y),
                           std::min(minimum.z, p.z)};
                maximum = {std::max(maximum.x, p.x),
                           std::max(maximum.y, p.y),
                           std::max(maximum.z, p.z)};
            }
        }
        const Vec3 extent = subtract(maximum, minimum);
        if (!finite(minimum) || !finite(maximum) ||
            extent.x <= 0.0F || extent.z <= 0.0F || extent.y > 1.0e-3F) {
            error = name + ": fluid mesh must be a local XZ rectangle";
            return false;
        }
        const RigidBodyState transform = node_state(node);
        const Vec3 local_center = multiply(add(minimum, maximum), 0.5F);
        ParticlePlane plane{
            .center = add(transform.position,
                          rotate(transform.orientation, local_center)),
            .orientation = transform.orientation,
            .half_extents = {extent.x * 0.5F, extent.z * 0.5F}};
        if (system == "fluid_inflow") {
            const float rate = static_cast<float>(
                extras.number("pm_particles_per_second").value_or(2400.0));
            const Vec3 velocity{
                static_cast<float>(extras.number("pm_velocity_x").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_y").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_z").value_or(0.0))};
            if (!finite(rate) || rate < 0.0F || !finite(velocity)) {
                error = name + ": invalid fluid inflow rate or velocity";
                return false;
            }
            output.spawn_planes.push_back({.plane = plane,
                                           .particles_per_second = rate,
                                           .initial_velocity = velocity});
        } else {
            output.destroy_planes.push_back({.plane = plane});
        }
    }
    if (!output.spawn_planes.empty()) {
        output.fluid_options = {.capacity = 30'000U,
                                .particle_radius = 0.045F,
                                .support_radius = 0.18F,
                                .solver_iterations = 2U,
                                .maximum_neighbors = 128U,
                                .repulsion = 50.0F};
    } else if (!output.initial_particles.empty()) {
        const float spacing = *initial_spacing;
        const float radius = 0.5F * spacing;
        output.gravity_scale = *initial_gravity_scale;
        output.fluid_options = {.capacity = 30'000U,
                                .particle_radius = radius,
                                .support_radius = std::max(0.12F, spacing),
                                .solver_iterations = 2U,
                                .maximum_neighbors = 256U,
                                .repulsion = 30.0F,
                                .viscosity = 0.0F,
                                .velocity_damping = 0.4F,
                                .maximum_speed = 3.0F,
                                .normal_damping = 2.0F,
                                .rest_particle_volume =
                                    std::sqrt(0.5F) * spacing * spacing * spacing,
                                .maximum_pair_acceleration = 55.0F};
    }
    if (output.rigid_bodies.empty()) {
        error = "GLB contains no ParallelMater rigid bodies";
        return false;
    }
    return true;
}

SceneDefinition make_dump_scene(std::uint32_t sphere_count) {
    constexpr std::uint32_t minimum_spheres = 10U;
    constexpr std::uint32_t maximum_spheres = 1'000U;
    constexpr float sphere_radius = 0.1F;
    constexpr float sphere_spacing = sphere_radius * 2.15F;
    constexpr float source_half_extent = 1.3F;
    constexpr float pi = 3.14159265358979323846F;
    sphere_count = std::clamp(sphere_count, minimum_spheres, maximum_spheres);

    SceneDefinition result{};
    result.meshes.reserve(3U);
    result.rigid_bodies.reserve(static_cast<std::size_t>(sphere_count) + 2U);
    result.meshes.push_back(make_open_cube("dump_hopper", source_half_extent,
                                           true, {0.42F, 0.48F, 0.56F}, false));
    result.meshes.push_back(make_open_cube("dump_receiver", 2.5F, false,
                                           {0.22F, 0.31F, 0.42F}, true));
    result.meshes.push_back(make_cube_projected_sphere(sphere_radius));

    const RigidBodyState hopper_state{
        .position = {0.0F, 4.5F, 0.0F},
        .orientation = rotation_z(pi * 0.25F),
    };
    result.rigid_bodies.push_back(
        {"Dump hopper", {.motion = MotionType::kinematic,
                          .initial_state = hopper_state,
                          .friction = 0.65F,
                          .restitution = 0.02F,
                          .collision_margin = 0.01F},
         {0U}});
    result.rigid_bodies.push_back(
        {"Dump receiver", {.motion = MotionType::static_body,
                            .initial_state = {.position = {1.5F, 0.0F, 0.0F}},
                            .friction = 0.7F,
                            .restitution = 0.02F,
                            .collision_margin = 0.01F},
         {1U}});

    std::uint32_t width = 1U;
    while (width * width * width < sphere_count) {
        ++width;
    }
    for (std::uint32_t index = 0U; index < sphere_count; ++index) {
        const std::uint32_t x = index % width;
        const std::uint32_t y = (index / width) % width;
        const std::uint32_t z = index / (width * width);
        const Vec3 local{
            (static_cast<float>(x) - static_cast<float>(width - 1U) * 0.5F) *
                sphere_spacing,
            (static_cast<float>(y) - static_cast<float>(width - 1U) * 0.5F) *
                sphere_spacing,
            (static_cast<float>(z) - static_cast<float>(width - 1U) * 0.5F) *
                sphere_spacing,
        };
        const RigidBodyState initial{
            .position = add(hopper_state.position,
                            rotate(hopper_state.orientation, local)),
        };
        result.rigid_bodies.push_back(
            {"Dump sphere " + std::to_string(index + 1U),
             {.motion = MotionType::dynamic,
              .initial_state = initial,
              .mass = 0.06F,
              .friction = 0.5F,
              .restitution = 0.04F,
              .linear_damping = 0.01F,
              .angular_damping = 0.01F,
              .maximum_linear_speed = 40.0F,
              .maximum_angular_speed = 80.0F,
              .collision_margin = 0.004F},
             {2U}});
    }
    return result;
}

Status instantiate_scene(const SceneDefinition &scene, World &world,
                         SceneInstance &output) noexcept {
    output = {};
    std::unordered_map<std::string, TriangleMeshId> mesh_cache;
    try {
        output.rigid_bodies.reserve(scene.rigid_bodies.size());
        mesh_cache.reserve(scene.meshes.size() + scene.collision_meshes.size());
    } catch (...) {
        return {StatusCode::out_of_memory, cudaSuccess,
                "failed to allocate gallery body bindings"};
    }

    const auto upload_mesh = [&](const std::vector<TriangleMesh> &source_meshes,
                                 const std::vector<std::uint32_t> &source_indices,
                                 TriangleMeshId &mesh_id) -> Status {
        std::vector<Vec3> vertices;
        std::vector<std::uint32_t> indices;
        try {
            std::size_t vertex_count = 0U;
            std::size_t index_count = 0U;
            for (const std::uint32_t mesh_index : source_indices) {
                vertex_count += source_meshes[mesh_index].vertices.size();
                index_count += source_meshes[mesh_index].indices.size();
            }
            if (vertex_count > std::numeric_limits<std::uint32_t>::max() ||
                index_count > std::numeric_limits<std::uint32_t>::max()) {
                return {StatusCode::capacity_exceeded, cudaSuccess,
                        "gallery triangle mesh exceeds uint32 range"};
            }
            vertices.reserve(vertex_count);
            indices.reserve(index_count);
            for (const std::uint32_t mesh_index : source_indices) {
                const TriangleMesh &mesh = source_meshes[mesh_index];
                const std::uint32_t base =
                    static_cast<std::uint32_t>(vertices.size());
                for (const Vertex &vertex : mesh.vertices) {
                    vertices.push_back(vertex.position);
                }
                for (const std::uint32_t index : mesh.indices) {
                    indices.push_back(base + index);
                }
            }
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "failed to assemble gallery triangle mesh"};
        }

        Vec3 *device_vertices = nullptr;
        std::uint32_t *device_indices = nullptr;
        cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&device_vertices),
                                       vertices.size() * sizeof(Vec3));
        if (error == cudaSuccess) {
            error = cudaMalloc(reinterpret_cast<void **>(&device_indices),
                               indices.size() * sizeof(std::uint32_t));
        }
        if (error == cudaSuccess) {
            error = cudaMemcpy(device_vertices, vertices.data(),
                               vertices.size() * sizeof(Vec3),
                               cudaMemcpyHostToDevice);
        }
        if (error == cudaSuccess) {
            error = cudaMemcpy(device_indices, indices.data(),
                               indices.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice);
        }
        if (error != cudaSuccess) {
            cudaFree(device_indices);
            cudaFree(device_vertices);
            return {StatusCode::cuda_failure, error,
                    "failed to upload gallery triangle mesh"};
        }
        const Status mesh_status = world.add_triangle_mesh(
            {device_vertices, vertices.size()}, {device_indices, indices.size()},
            mesh_id);
        cudaFree(device_indices);
        cudaFree(device_vertices);
        if (!mesh_status) {
            return mesh_status;
        }
        return {};
    };

    for (const RigidBodyDefinition &definition : scene.rigid_bodies) {
        RigidBodyOptions options = definition.options;
        const bool uses_collision_proxy =
            !definition.collision_mesh_indices.empty();
        const std::vector<TriangleMesh> &source_meshes =
            uses_collision_proxy ? scene.collision_meshes : scene.meshes;
        const std::vector<std::uint32_t> &source_indices =
            uses_collision_proxy ? definition.collision_mesh_indices
                                 : definition.mesh_indices;
        std::string cache_key = uses_collision_proxy ? "collision" : "render";
        try {
            for (const std::uint32_t mesh_index : source_indices) {
                cache_key.push_back(':');
                cache_key.append(std::to_string(mesh_index));
            }
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "failed to identify shared gallery mesh"};
        }

        TriangleMeshId mesh_id{};
        const auto cached = mesh_cache.find(cache_key);
        if (cached != mesh_cache.end()) {
            mesh_id = cached->second;
        } else {
            const Status mesh_status =
                upload_mesh(source_meshes, source_indices, mesh_id);
            if (!mesh_status) {
                return mesh_status;
            }
            try {
                mesh_cache.emplace(std::move(cache_key), mesh_id);
            } catch (...) {
                return {StatusCode::out_of_memory, cudaSuccess,
                        "failed to cache shared gallery mesh"};
            }
        }
        options.mesh = mesh_id;
        RigidBodyId body{};
        const Status status = world.add_rigid_body(options, body);
        if (!status) {
            return status;
        }
        output.rigid_bodies.push_back(body);
    }
    for (const ClothDefinition &definition : scene.cloths) {
        if (definition.mesh_index >= scene.meshes.size())
            return {StatusCode::invalid_argument, cudaSuccess,
                    "gallery cloth mesh index is invalid"};
        const TriangleMesh &mesh = scene.meshes[definition.mesh_index];
        std::vector<Vec3> positions;
        try {
            positions.reserve(mesh.vertices.size());
            for (const Vertex &vertex : mesh.vertices)
                positions.push_back(vertex.position);
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "failed to assemble gallery cloth vertices"};
        }
        ClothId cloth{};
        const Status status = world.add_cloth({
            .vertices = {positions.data(), positions.size()},
            .triangle_indices = {mesh.indices.data(), mesh.indices.size()},
            .inverse_masses = {definition.inverse_masses.data(),
                               definition.inverse_masses.size()},
            .vertex_mass = definition.vertex_mass,
            .thickness = definition.thickness,
            .stretch_compliance = definition.stretch_compliance,
            .solver_iterations = definition.solver_iterations,
            .tear_ratio = definition.tear_ratio,
            .tear_requires_contact = definition.tear_requires_contact}, cloth);
        if (!status) return status;
        output.cloths.push_back(cloth);
    }
    if (scene.fluid_options.capacity != 0U) {
        const std::size_t requested = std::min<std::size_t>(
            scene.fluid_options.capacity, scene.initial_particles.size());
        std::vector<FluidParticle> initial;
        try {
            initial.reserve(requested);
            for (std::size_t index = 0U; index < requested; ++index) {
                const std::size_t source = index * scene.initial_particles.size() /
                                           requested;
                initial.push_back(scene.initial_particles[source]);
            }
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "failed to select initial gallery particles"};
        }
        FluidParticle *device_initial = nullptr;
        if (!initial.empty()) {
            cudaError_t error = cudaMalloc(
                reinterpret_cast<void **>(&device_initial),
                initial.size() * sizeof(FluidParticle));
            if (error == cudaSuccess)
                error = cudaMemcpy(device_initial, initial.data(),
                    initial.size() * sizeof(FluidParticle), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                cudaFree(device_initial);
                return {StatusCode::cuda_failure, error,
                        "failed to upload initial gallery particles"};
            }
        }
        Status status = world.add_fluid(scene.fluid_options,
            {device_initial, initial.size()}, output.fluid);
        cudaFree(device_initial);
        if (!status) return status;
        output.has_fluid = true;
        for (ParticleSpawnPlaneOptions options : scene.spawn_planes) {
            options.fluid = output.fluid;
            ParticleSpawnPlaneId id{};
            status = world.add_particle_spawn_plane(options, id);
            if (!status) return status;
        }
        for (ParticleDestroyPlaneOptions options : scene.destroy_planes) {
            options.fluid = output.fluid;
            ParticleDestroyPlaneId id{};
            status = world.add_particle_destroy_plane(options, id);
            if (!status) return status;
        }
    }
    try {
    const auto bind_paint = [&](std::uint32_t owner,
                                std::uint32_t mesh_index,
                                PaintFieldOptions options) -> Status {
        const TriangleMesh &mesh = scene.meshes[mesh_index];
        std::vector<Vec2> uvs;
        uvs.reserve(mesh.vertices.size());
        for (const Vertex &vertex : mesh.vertices) uvs.push_back(vertex.uv);
        Vec2 *device_uvs = nullptr;
        cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&device_uvs),
                                      uvs.size() * sizeof(Vec2));
        if (error == cudaSuccess)
            error = cudaMemcpy(device_uvs, uvs.data(),
                uvs.size() * sizeof(Vec2), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            cudaFree(device_uvs);
            return {StatusCode::cuda_failure, error, "failed to upload paint UVs"};
        }
        options.vertex_uvs = {device_uvs, uvs.size()};
        PaintFieldId field{};
        Status status = world.add_paint_field(options, field);
        cudaFree(device_uvs);
        if (!status) return status;
        PaintRuleId rule{};
        status = world.add_paint_rule({.source = output.fluid,
                                      .target = field}, rule);
        if (!status) return status;
        output.paint_bindings.push_back({owner, mesh_index, field});
        return {};
    };
    for (std::uint32_t body_index = 0;
         body_index < scene.rigid_bodies.size(); ++body_index) {
        const RigidBodyDefinition &body = scene.rigid_bodies[body_index];
        if (!body.paintable) continue;
        if (!output.has_fluid)
            return {StatusCode::invalid_argument, cudaSuccess,
                    "paintable gallery body needs a fluid source"};
        for (const std::uint32_t mesh_index : body.mesh_indices) {
            const std::string key = "paint:" + std::to_string(mesh_index);
            TriangleMeshId mesh_id{};
            const auto cached = mesh_cache.find(key);
            if (cached != mesh_cache.end()) {
                mesh_id = cached->second;
            } else {
                Status status = upload_mesh(scene.meshes, {mesh_index}, mesh_id);
                if (!status) return status;
                mesh_cache.emplace(key, mesh_id);
            }
            const Status status = bind_paint(body_index, mesh_index,
                {.body = output.rigid_bodies[body_index],
                 .mesh = mesh_id,
                 .width = body.paint_resolution,
                 .height = body.paint_resolution});
            if (!status) return status;
        }
    }
    for (std::uint32_t cloth_index = 0U;
         cloth_index < scene.cloths.size(); ++cloth_index) {
        const ClothDefinition &cloth = scene.cloths[cloth_index];
        if (!cloth.paintable) continue;
        if (!output.has_fluid)
            return {StatusCode::invalid_argument, cudaSuccess,
                    "paintable cloth needs a fluid source"};
        const Status status = bind_paint(UINT32_MAX, cloth.mesh_index,
            {.cloth = output.cloths[cloth_index],
             .width = cloth.paint_resolution,
             .height = cloth.paint_resolution});
        if (!status) return status;
    }
    } catch (...) {
        return {StatusCode::out_of_memory, cudaSuccess,
                "failed to assemble gallery paint bindings"};
    }
    return {};
}

} // namespace parallel_mater::gallery
