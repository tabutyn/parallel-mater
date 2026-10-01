// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include "vector_math.hpp"

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

using math::add;
using math::cross;
using math::dot;
using math::multiply;
using math::subtract;

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

[[nodiscard]] Vec3 normalize(Vec3 value) {
    return math::normalize_or(value, {0.0F, 1.0F, 0.0F});
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

// glTF may split one Blender vertex at normal or UV seams. Pressure physics
// needs the authored topological vertex, so merge coincident render vertices
// before constructing a closed cloth graph.
void weld_pressure_cloth(TriangleMesh &mesh) {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> remap(mesh.vertices.size());
    vertices.reserve(mesh.vertices.size());
    for (std::size_t source = 0U; source < mesh.vertices.size(); ++source) {
        const Vec3 position = mesh.vertices[source].position;
        std::size_t target = 0U;
        for (; target < vertices.size(); ++target) {
            const Vec3 difference = subtract(position, vertices[target].position);
            if (dot(difference, difference) <= 1.0e-12F) break;
        }
        if (target == vertices.size()) vertices.push_back(mesh.vertices[source]);
        remap[source] = static_cast<std::uint32_t>(target);
    }
    for (std::uint32_t &index : mesh.indices) index = remap[index];
    mesh.vertices = std::move(vertices);
}

struct Pin { Vec3 position; float weight; bool matched{}; };

[[nodiscard]] bool read_pins(const FlatJson &extras, const std::string &name,
                            std::vector<Pin> &pins, std::string &error) {
    const std::string encoded = extras.string("pm_pin_vertices").value_or("");
    const char *cursor = encoded.c_str();
    while (*cursor != '\0') {
        float fields[4]{};
        for (int component = 0; component < 4; ++component) {
            char *next = nullptr;
            fields[component] = std::strtof(cursor, &next);
            if (next == cursor || !std::isfinite(fields[component]) ||
                (component < 3 && *next != ',') ||
                (component == 3 && *next != ';' && *next != '\0')) {
                error = name + ": invalid exported pin coordinate";
                return false;
            }
            cursor = next + (component < 3 || *next == ';' ? 1 : 0);
        }
        if (fields[3] <= 0.0F || fields[3] > 1.0F) {
            error = name + ": invalid pin weight";
            return false;
        }
        pins.push_back({{fields[0], fields[1], fields[2]}, fields[3]});
    }
    return true;
}

[[nodiscard]] bool build_soft_body_lattice(
    TriangleMesh &mesh, float spacing, float total_mass,
    const std::vector<float> &pin_weights,
    SoftBodyDefinition &body, std::string &error) {
    std::vector<Vec3> surface;
    surface.reserve(mesh.vertices.size());
    for (const Vertex &vertex : mesh.vertices)
        surface.push_back(vertex.position);
    SoftBodyGeometry geometry;
    const Status status = build_soft_body_geometry(
        {{surface.data(), surface.size()},
         {mesh.indices.data(), mesh.indices.size()},
         {pin_weights.data(), pin_weights.size()}, spacing, total_mass}, geometry);
    if (!status) {
        error = body.name + ": " +
            (status.message != nullptr ? status.message : "soft-body meshing failed");
        return false;
    }

    // Rendering attributes stay outside physics. The API supplies the source
    // interpolation for each newly simulated surface vertex, including seams.
    std::vector<Vertex> refined;
    refined.reserve(geometry.surface_vertices.size());
    for (std::size_t i = 0; i < geometry.surface_vertices.size(); ++i) {
        Vertex vertex{};
        vertex.position = geometry.surface_vertices[i];
        const auto &source = geometry.surface_sources[i];
        for (unsigned j = 0; j < 3; ++j) {
            const Vertex &original = mesh.vertices[source.vertices[j]];
            vertex.normal = math::add(vertex.normal,
                math::multiply(original.normal, source.weights[j]));
            vertex.uv.x += original.uv.x * source.weights[j];
            vertex.uv.y += original.uv.y * source.weights[j];
        }
        vertex.normal = math::normalize_or(vertex.normal, {0, 1, 0});
        refined.push_back(vertex);
    }
    mesh.vertices = std::move(refined);
    mesh.indices = std::move(geometry.surface_triangle_indices);
    body.nodes = std::move(geometry.nodes);
    body.bonds = std::move(geometry.bonds);
    body.inverse_masses = std::move(geometry.inverse_masses);
    body.surface_bindings = std::move(geometry.surface_bindings);
    body.node_mass = geometry.node_mass;
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
        body.source_name = extras.string("pm_source_name").value_or(body.name);
        body.paintable = extras.boolean("pm_paintable").value_or(false);
        body.smoke_collider = extras.boolean("pm_smoke_collider").value_or(false);
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
        body.options.initial_state.linear_velocity = {
            static_cast<float>(extras.number("pm_initial_velocity_x").value_or(0.0)),
            static_cast<float>(extras.number("pm_initial_velocity_y").value_or(0.0)),
            static_cast<float>(extras.number("pm_initial_velocity_z").value_or(0.0))};
        if (!finite(body.options.initial_state.linear_velocity)) {
            error = body.name + ": initial rigid velocity must be finite";
            return false;
        }
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
        const float break_strain = static_cast<float>(
            extras.number("pm_break_strain").value_or(0.0));
        const float impact_break_impulse = static_cast<float>(
            extras.number("pm_impact_break_impulse").value_or(0.0));
        const double fracture_persistence = extras.number(
            "pm_fracture_persistence_substeps").value_or(4.0);
        const float stretch_compliance = static_cast<float>(
            extras.number("pm_stretch_compliance").value_or(1.0e-6));
        const double solver_iterations =
            extras.number("pm_solver_iterations").value_or(8.0);
        const float velocity_damping = static_cast<float>(
            extras.number("pm_velocity_damping").value_or(5.0));
        const float contact_friction = static_cast<float>(
            extras.number("pm_contact_friction").value_or(0.4));
        const bool pressure_enabled =
            extras.boolean("pm_pressure_enabled").value_or(false);
        const float pressure_scale = static_cast<float>(
            extras.number("pm_pressure_scale").value_or(1.0));
        const float uniform_pressure = static_cast<float>(
            extras.number("pm_uniform_pressure").value_or(0.0));
        const bool pressure_custom_volume =
            extras.boolean("pm_pressure_custom_volume").value_or(false);
        const float pressure_target_volume = static_cast<float>(
            extras.number("pm_pressure_target_volume").value_or(0.0));
        const float pressure_fluid_density = static_cast<float>(
            extras.number("pm_pressure_fluid_density").value_or(0.0));
        const double paint_resolution =
            extras.number("pm_paint_resolution").value_or(512.0);
        const Vec3 scale = node_scale(node);
        if (!finite(scale) || scale.x <= 0.0F || scale.y <= 0.0F ||
            scale.z <= 0.0F || !finite(mass) || mass <= 0.0F ||
            !finite(thickness) || thickness <= 0.0F) {
            error = name + ": invalid cloth mass, thickness, or transform";
            return false;
        }
        if (!finite(break_strain) || break_strain < 0.0F ||
            break_strain > 9.0F || !finite(impact_break_impulse) ||
            impact_break_impulse < 0.0F ||
            fracture_persistence < 1.0 || fracture_persistence > 64.0 ||
            std::floor(fracture_persistence) != fracture_persistence ||
            !finite(stretch_compliance) || stretch_compliance < 0.0F ||
            !finite(velocity_damping) || velocity_damping < 0.0F ||
            !finite(contact_friction) || contact_friction < 0.0F ||
            !finite(pressure_scale) || pressure_scale <= 0.0F ||
            !finite(uniform_pressure) || uniform_pressure != 0.0F ||
            !finite(pressure_target_volume) || pressure_target_volume < 0.0F ||
            !finite(pressure_fluid_density) || pressure_fluid_density < 0.0F ||
            solver_iterations < 1.0 || solver_iterations > 64.0 ||
            std::floor(solver_iterations) != solver_iterations ||
            paint_resolution < 32.0 || paint_resolution > 2048.0 ||
            std::floor(paint_resolution) != paint_resolution) {
            error = name + ": invalid cloth tear, solver, or paint settings";
            return false;
        }
        std::vector<Pin> pins;
        if (!read_pins(extras, name, pins, error)) return false;
        TriangleMesh mesh{};
        if (!append_primitive(node.mesh->primitives[0], scale, false,
                              name, mesh, error)) return false;
        if (pressure_enabled || extras.boolean("pm_weld_vertices").value_or(false))
            weld_pressure_cloth(mesh);
        ClothDefinition cloth{};
        cloth.name = name;
        cloth.vertex_mass = mass;
        cloth.thickness = thickness;
        cloth.break_strain = break_strain;
        cloth.fracture_persistence_substeps =
            static_cast<std::uint32_t>(fracture_persistence);
        cloth.impact_break_impulse = impact_break_impulse;
        cloth.stretch_compliance = stretch_compliance;
        cloth.velocity_damping = velocity_damping;
        cloth.contact_friction = contact_friction;
        cloth.solver_iterations = static_cast<std::uint32_t>(solver_iterations);
        cloth.preserve_volume = pressure_enabled;
        cloth.target_volume = pressure_custom_volume
            ? pressure_target_volume : 0.0F;
        cloth.volume_compliance = 1.0e-7F / pressure_scale;
        cloth.contains_fluid =
            extras.boolean("pm_contains_fluid").value_or(false);
        if (cloth.contains_fluid && !cloth.preserve_volume) {
            error = name + ": contained fluid requires Cloth Pressure";
            return false;
        }
        cloth.paintable = extras.boolean("pm_paintable").value_or(false);
        cloth.paint_resolution = static_cast<std::uint32_t>(paint_resolution);
        cloth.paint_source = extras.string("pm_paint_source").value_or("");
        cloth.paint_brush_radius = static_cast<float>(
            extras.number("pm_paint_brush_radius").value_or(0.15));
        if (!std::isfinite(cloth.paint_brush_radius) ||
            cloth.paint_brush_radius <= 0.0F ||
            cloth.paint_brush_radius > 10.0F) {
            error = name + ": invalid cloth paint brush radius";
            return false;
        }
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
    for (cgltf_size node_index = 0; node_index < data->nodes_count; ++node_index) {
        const cgltf_node &node = data->nodes[node_index];
        if (node.extras.data == nullptr) continue;
        const FlatJson extras(node.extras.data);
        if (extras.string("pm_system").value_or("") != "soft_body") continue;
        SoftBodyDefinition body{};
        body.name = extras.string("pm_name").value_or(
            node.name != nullptr ? node.name : "soft_body");
        if (extras.number("pm_schema").value_or(0.0) != 2.0 ||
            node.parent != nullptr || node.has_matrix || node.mesh == nullptr ||
            node.mesh->primitives_count != 1U) {
            error = body.name +
                ": soft body needs schema 2 and one scene-root TRS mesh";
            return false;
        }
        const Vec3 scale = node_scale(node);
        const float total_mass = static_cast<float>(
            extras.number("pm_total_mass").value_or(1.0));
        const float spacing = static_cast<float>(
            extras.number("pm_node_spacing").value_or(0.22));
        body.node_radius = static_cast<float>(
            extras.number("pm_node_radius").value_or(spacing * 0.35));
        body.stretch_compliance = static_cast<float>(
            extras.number("pm_stretch_compliance").value_or(1.0e-7));
        body.velocity_damping = static_cast<float>(
            extras.number("pm_velocity_damping").value_or(0.8));
        body.spring_damping = static_cast<float>(
            extras.number("pm_spring_damping").value_or(0.85));
        body.contact_friction = static_cast<float>(
            extras.number("pm_contact_friction").value_or(0.5));
        body.shape_matching_stiffness = static_cast<float>(extras.number(
            "pm_shape_matching_stiffness").value_or(0.0));
        body.maximum_projection_fraction = static_cast<float>(extras.number(
            "pm_maximum_projection_fraction").value_or(0.20));
        body.constraint_velocity_response = static_cast<float>(extras.number(
            "pm_constraint_velocity_response").value_or(0.70));
        body.maximum_speed = static_cast<float>(
            extras.number("pm_maximum_speed").value_or(2.0));
        const double solver_iterations =
            extras.number("pm_solver_iterations").value_or(16.0);
        if (!finite(scale) || scale.x <= 0.0F || scale.y <= 0.0F ||
            scale.z <= 0.0F || !finite(total_mass) || total_mass <= 0.0F ||
            !finite(spacing) || spacing <= 0.0F || spacing > 10.0F ||
            !finite(body.node_radius) || body.node_radius <= 0.0F ||
            body.node_radius > spacing || !finite(body.stretch_compliance) ||
            body.stretch_compliance < 0.0F ||
            !finite(body.velocity_damping) || body.velocity_damping < 0.0F ||
            !finite(body.spring_damping) || body.spring_damping < 0.0F ||
            body.spring_damping > 1.0F || !finite(body.contact_friction) ||
            body.contact_friction < 0.0F ||
            !finite(body.shape_matching_stiffness) ||
            body.shape_matching_stiffness < 0.0F ||
            body.shape_matching_stiffness > 1.0F ||
            !finite(body.maximum_projection_fraction) ||
            body.maximum_projection_fraction <= 0.0F ||
            body.maximum_projection_fraction > 1.0F ||
            !finite(body.constraint_velocity_response) ||
            body.constraint_velocity_response < 0.0F ||
            body.constraint_velocity_response > 1.0F ||
            !finite(body.maximum_speed) ||
            body.maximum_speed <= 0.0F || solver_iterations < 1.0 ||
            solver_iterations > 64.0 ||
            std::floor(solver_iterations) != solver_iterations) {
            error = body.name + ": invalid soft-body solver settings";
            return false;
        }
        body.solver_iterations =
            static_cast<std::uint32_t>(solver_iterations);
        TriangleMesh mesh{};
        if (!append_primitive(node.mesh->primitives[0], scale, false,
                              body.name, mesh, error)) return false;
        const RigidBodyState state = node_state(node);
        for (Vertex &vertex : mesh.vertices) {
            vertex.position = add(state.position,
                rotate(state.orientation, vertex.position));
            vertex.normal = rotate(state.orientation, vertex.normal);
        }
        std::vector<Pin> pins;
        if (!read_pins(extras, body.name, pins, error)) return false;
        std::vector<float> pin_weights(mesh.vertices.size(), 0.0F);
        for (Pin &pin : pins) {
            if (pin.weight != 1.0F) {
                error = body.name + ": soft-body pins require full Goal weight";
                return false;
            }
            const Vec3 target = add(state.position, rotate(state.orientation, pin.position));
            for (std::size_t vertex = 0; vertex < mesh.vertices.size(); ++vertex) {
                if (math::length_squared(subtract(mesh.vertices[vertex].position, target)) > 1.0e-8F)
                    continue;
                pin_weights[vertex] = 1.0F;
                pin.matched = true;
            }
            if (!pin.matched) {
                error = body.name + ": Goal pin did not match exported mesh";
                return false;
            }
        }
        if (!build_soft_body_lattice(mesh, spacing, total_mass, pin_weights, body, error))
            return false;
        body.mesh_index = static_cast<std::uint32_t>(output.meshes.size());
        output.meshes.push_back(std::move(mesh));
        output.soft_bodies.push_back(std::move(body));
    }
    for(cgltf_size n=0;n<data->nodes_count;++n) {
        const auto &node=data->nodes[n];
        if(!node.extras.data)continue;
        const FlatJson extras(node.extras.data);
        if(extras.string("pm_system").value_or("")!="rope")continue;
        RopeDefinition rope;rope.name=node.name?node.name:"rope";
        if(extras.number("pm_schema").value_or(0)!=2 || node.parent){error=rope.name+": invalid rope root/schema";return false;}
        const std::string encoded=extras.string("pm_rope_points").value_or("");
        const char *cursor=encoded.c_str();
        while(*cursor) {
            float values[3];
            for(int c=0;c<3;++c) {
                char *next=nullptr;values[c]=std::strtof(cursor,&next);
                if(next==cursor || !std::isfinite(values[c]) || (c<2 && *next!=',') || (c==2 && *next!=';' && *next!='\0')) {
                    error=rope.name+": invalid rope centerline";return false;
                }
                cursor=next+((c<2 || *next==';')?1:0);
            }
            rope.centerline.push_back({values[0],values[1],values[2]});
        }
        auto &options=rope.options;
        options.radius=static_cast<float>(extras.number("pm_rope_radius").value_or(0.01));
        options.node_spacing=static_cast<float>(extras.number("pm_rope_spacing").value_or(2*options.radius));
        options.mass=static_cast<float>(extras.number("pm_rope_mass").value_or(0.1));
        options.stretch_compliance=static_cast<float>(extras.number("pm_rope_compliance").value_or(0));
        options.friction=static_cast<float>(extras.number("pm_rope_friction").value_or(0.4));
        options.velocity_damping=static_cast<float>(extras.number("pm_rope_damping").value_or(0.1));
        options.maximum_substep_timestep=static_cast<float>(extras.number("pm_rope_maximum_substep_timestep").value_or(1.0/480));
        const double iterations=extras.number("pm_rope_iterations").value_or(24);
        if(!finite(options.radius)||options.radius<=0 || !finite(options.node_spacing)||options.node_spacing<=0 || options.node_spacing>2*options.radius ||
           !finite(options.mass)||options.mass<=0 || !finite(options.stretch_compliance)||options.stretch_compliance<0 ||
           !finite(options.friction)||options.friction<0 || !finite(options.velocity_damping)||options.velocity_damping<0 ||
           !finite(options.maximum_substep_timestep)||options.maximum_substep_timestep<=0 ||
           !std::isfinite(iterations)||iterations<1||iterations>128||std::floor(iterations)!=iterations) {
            error=rope.name+": invalid rope material";return false;
        }
        options.solver_iterations=static_cast<unsigned>(iterations);
        std::vector<Vec3> nodes;
        const auto sampled=sample_rope_centerline({rope.centerline.data(),rope.centerline.size()},options.node_spacing,nodes);
        if(!sampled){error=sampled.message;return false;}
        for(unsigned end=0;end<2;++end) {
            const auto target=extras.string(end?"pm_rope_last_body":"pm_rope_first_body").value_or("");
            const auto soft_target=extras.string(end?"pm_rope_last_soft_body":"pm_rope_first_soft_body").value_or("");
            const auto cloth_target=extras.string(end?"pm_rope_last_cloth":"pm_rope_first_cloth").value_or("");
            if(unsigned(!target.empty())+unsigned(!soft_target.empty())+
               unsigned(!cloth_target.empty())>1U) {
                error=rope.name+": endpoint has multiple attachment targets";return false;
            }
            if(!cloth_target.empty()) {
                int match=-1;
                for(unsigned sheet=0;sheet<output.cloths.size();++sheet)
                    if(output.cloths[sheet].name==cloth_target) {
                        if(match>=0){error=rope.name+": cloth target is ambiguous";return false;}
                        match=int(sheet);
                    }
                if(match<0){error=rope.name+": missing cloth target "+cloth_target;return false;}
                const auto &mesh=output.meshes[output.cloths[match].mesh_index];
                const Vec3 point=end?nodes.back():nodes.front();
                int vertex=-1;
                for(unsigned index=0;index<mesh.vertices.size();++index)
                    if(math::length_squared(subtract(mesh.vertices[index].position,point))<=1.0e-8F) {
                        if(vertex>=0){error=rope.name+": cloth vertex is ambiguous";return false;}
                        vertex=int(index);
                    }
                if(vertex<0){error=rope.name+": endpoint misses cloth vertex";return false;}
                (end?rope.last_cloth:rope.first_cloth)=match;
                (end?rope.last_cloth_vertex:rope.first_cloth_vertex)=unsigned(vertex);
                continue;
            }
            if(!soft_target.empty()) {
                int match=-1;
                for(unsigned body=0;body<output.soft_bodies.size();++body)
                    if(output.soft_bodies[body].name==soft_target) {
                        if(match>=0){error=rope.name+": soft Hook target is ambiguous";return false;}
                        match=int(body);
                    }
                if(match<0){error=rope.name+": missing soft Hook target "+soft_target;return false;}
                (end?rope.last_soft_body:rope.first_soft_body)=match;
                continue;
            }
            if(target.empty())continue;
            int match=-1;
            for(unsigned body=0;body<output.rigid_bodies.size();++body)
                if(output.rigid_bodies[body].source_name==target || output.rigid_bodies[body].name==target) {
                    if(match>=0){error=rope.name+": Hook target is ambiguous (instanced body)";return false;}
                    match=int(body);
                }
            if(match<0){error=rope.name+": missing rigid Hook target "+target;return false;}
            const auto state=output.rigid_bodies[match].options.initial_state;
            const auto q=state.orientation;
            auto &anchor=end?options.last:options.first;
            anchor.enabled=true;
            anchor.local_anchor=rotate({-q.x,-q.y,-q.z,q.w},subtract(end?nodes.back():nodes.front(),state.position));
            (end?rope.last_body:rope.first_body)=match;
        }
        TriangleMesh mesh;mesh.name=rope.name;mesh.base_color={0.85F,0.28F,0.06F};
        update_rope_render_mesh(nodes,options.radius,mesh);
        rope.mesh_index=static_cast<unsigned>(output.meshes.size());
        output.meshes.push_back(std::move(mesh));output.ropes.push_back(std::move(rope));
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
            system != "fluid_initial_volume" &&
            system != "smoke_emitter" &&
            system != "thermal_surface") continue;
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
        if (system == "thermal_surface") {
            Vec3 low{FLT_MAX, FLT_MAX, FLT_MAX};
            Vec3 high{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            RigidBodyDefinition plate{};
            plate.name = name;
            plate.source_name = name;
            plate.options.motion = MotionType::static_body;
            plate.options.initial_state = node_state(node);
            for (cgltf_size primitive = 0; primitive < node.mesh->primitives_count;
                 ++primitive) {
                TriangleMesh mesh;
                if (!append_primitive(node.mesh->primitives[primitive], scale,
                                      false, name, mesh, error)) return false;
                for (const Vertex &vertex : mesh.vertices) {
                    low = {std::min(low.x, vertex.position.x),
                           std::min(low.y, vertex.position.y),
                           std::min(low.z, vertex.position.z)};
                    high = {std::max(high.x, vertex.position.x),
                            std::max(high.y, vertex.position.y),
                            std::max(high.z, vertex.position.z)};
                }
                plate.mesh_indices.push_back(
                    static_cast<std::uint32_t>(output.meshes.size()));
                output.meshes.push_back(std::move(mesh));
            }
            const Vec3 extent = subtract(high, low);
            if (!finite(low) || !finite(high) || extent.x <= 0.0F ||
                extent.z <= 0.0F || extent.y > 0.001F) {
                error = name + ": thermal surface must be a local XZ plane";
                return false;
            }
            const RigidBodyState state = node_state(node);
            SceneDefinition::ThermalSurfaceDefinition heater{};
            heater.plane = {
                .center = add(state.position,
                              rotate(state.orientation, multiply(add(low, high), 0.5F))),
                .orientation = state.orientation,
                .half_extents = {extent.x * 0.5F, extent.z * 0.5F}};
            heater.temperature = static_cast<float>(
                extras.number("pm_temperature").value_or(500.0));
            heater.heat_transfer_rate = static_cast<float>(
                extras.number("pm_heat_transfer_rate").value_or(0.2));
            heater.smoke_drag = static_cast<float>(
                extras.number("pm_smoke_drag").value_or(2.0));
            heater.steam_rise_speed = static_cast<float>(
                extras.number("pm_steam_rise_speed").value_or(2.0));
            if (!std::isfinite(heater.temperature) ||
                !std::isfinite(heater.heat_transfer_rate) ||
                heater.heat_transfer_rate < 0.0F ||
                !std::isfinite(heater.smoke_drag) || heater.smoke_drag < 0.0F ||
                !std::isfinite(heater.steam_rise_speed) ||
                heater.steam_rise_speed < 0.0F) {
                error = name + ": invalid thermal surface settings";
                return false;
            }
            output.thermal_surfaces.push_back(heater);
            output.rigid_bodies.push_back(std::move(plate));
            continue;
        }
        if (system == "smoke_emitter") {
            if (output.has_smoke) {
                error = name + ": only one smoke emitter is supported";
                return false;
            }
            Vec3 low{FLT_MAX, FLT_MAX, FLT_MAX};
            Vec3 high{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            const RigidBodyState state = node_state(node);
            for (cgltf_size primitive = 0; primitive < node.mesh->primitives_count;
                 ++primitive) {
                TriangleMesh mesh;
                if (!append_primitive(node.mesh->primitives[primitive], scale,
                                      false, name, mesh, error)) return false;
                for (const Vertex &vertex : mesh.vertices) {
                    const Vec3 point = add(state.position,
                                           rotate(state.orientation, vertex.position));
                    low = {std::min(low.x, point.x), std::min(low.y, point.y),
                           std::min(low.z, point.z)};
                    high = {std::max(high.x, point.x), std::max(high.y, point.y),
                            std::max(high.z, point.z)};
                }
            }
            const Vec3 extent = subtract(high, low);
            if (!finite(low) || !finite(high) || extent.x > 0.01F ||
                extent.y < 0.01F || extent.z < 0.01F) {
                error = name + ": smoke emitter must be a world-YZ plane";
                return false;
            }
            const std::string obstacle =
                extras.string("pm_smoke_obstacle").value_or("");
            const double capacity =
                extras.number("pm_smoke_capacity").value_or(4500.0);
            if (!std::isfinite(capacity) || capacity < 1.0 ||
                capacity > 1'000'000.0 || std::floor(capacity) != capacity) {
                error = name + ": smoke capacity must be an integer from 1 to 1000000";
                return false;
            }
            const double grid_resolution =
                extras.number("pm_smoke_grid_resolution").value_or(128.0);
            const double vertical_resolution =
                extras.number("pm_smoke_grid_vertical_resolution").value_or(32.0);
            const double pressure_iterations =
                extras.number("pm_smoke_grid_pressure_iterations").value_or(24.0);
            const double kinematic_viscosity = extras.number(
                "pm_smoke_grid_kinematic_viscosity").value_or(1.5e-5);
            const double les_coefficient = extras.number(
                "pm_smoke_grid_les_coefficient").value_or(0.12);
            const double pressure_tolerance = extras.number(
                "pm_smoke_grid_pressure_tolerance").value_or(1.0e-3);
            if (!std::isfinite(grid_resolution) ||
                std::floor(grid_resolution) != grid_resolution ||
                (grid_resolution != 0.0 &&
                 (grid_resolution < 16.0 || grid_resolution > 256.0)) ||
                !std::isfinite(vertical_resolution) ||
                std::floor(vertical_resolution) != vertical_resolution ||
                vertical_resolution < 8.0 || vertical_resolution > 256.0 ||
                !std::isfinite(pressure_iterations) ||
                std::floor(pressure_iterations) != pressure_iterations ||
                pressure_iterations < 4.0 || pressure_iterations > 128.0 ||
                !std::isfinite(kinematic_viscosity) ||
                kinematic_viscosity < 0.0 ||
                !std::isfinite(les_coefficient) || les_coefficient < 0.0 ||
                !std::isfinite(pressure_tolerance) ||
                pressure_tolerance <= 0.0 || pressure_tolerance > 1.0) {
                error = name + ": invalid smoke grid settings";
                return false;
            }
            const Vec3 velocity{
                static_cast<float>(extras.number("pm_velocity_x").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_y").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_z").value_or(0.0))};
            output.smoke_options = {
                .capacity = static_cast<std::uint32_t>(capacity),
                .emitter_center = multiply(add(low, high), 0.5F),
                .emitter_half_extents = {extent.y * 0.5F, extent.z * 0.5F},
                .initial_velocity = velocity,
                .wind = velocity,
                .particles_per_second = static_cast<float>(
                    extras.number("pm_smoke_rate").value_or(900.0)),
                .lifetime = static_cast<float>(
                    extras.number("pm_smoke_lifetime").value_or(5.0)),
                .particle_radius = static_cast<float>(
                    extras.number("pm_smoke_radius").value_or(0.085)),
                .buoyancy = static_cast<float>(
                    extras.number("pm_smoke_buoyancy").value_or(0.12)),
                .response = static_cast<float>(
                    extras.number("pm_smoke_wind_response").value_or(0.5)),
                .rest_number_density = static_cast<float>(
                    extras.number("pm_smoke_rest_number_density").value_or(12.0)),
                .pressure_stiffness = static_cast<float>(
                    extras.number("pm_smoke_pressure_stiffness").value_or(2.0)),
                .viscosity = static_cast<float>(
                    extras.number("pm_smoke_viscosity").value_or(0.02)),
                .vorticity_confinement = static_cast<float>(
                    extras.number("pm_smoke_vorticity_confinement").value_or(0.1)),
                .grid_resolution = static_cast<std::uint32_t>(grid_resolution),
                .grid_vertical_resolution = static_cast<std::uint32_t>(
                    vertical_resolution),
                .grid_pressure_iterations = static_cast<std::uint32_t>(
                    pressure_iterations),
                .grid_kinematic_viscosity = static_cast<float>(
                    kinematic_viscosity),
                .grid_les_coefficient = static_cast<float>(les_coefficient),
                .grid_pressure_tolerance = static_cast<float>(
                    pressure_tolerance),
            };
            output.smoke_obstacle_name = obstacle;
            output.has_smoke = true;
            continue;
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
            const auto first = output.initial_particles.size();
            if (!sample_initial_volume(node, scale, velocity, spacing,
                                       output.initial_particles, error)) {
                return false;
            }
            const float temperature = static_cast<float>(
                extras.number("pm_temperature").value_or(20.0));
            if (!std::isfinite(temperature)) {
                error = name + ": invalid fluid temperature";
                return false;
            }
            for (auto index = first; index < output.initial_particles.size(); ++index)
                output.initial_particles[index].temperature = temperature;
            continue;
        }
        if (system == "fluid_inflow") {
            ParticleSourceDefinition source;
            source.spacing = static_cast<float>(extras.number("pm_source_spacing").value_or(0.0));
            source.options.initial_velocity = {
                static_cast<float>(extras.number("pm_velocity_x").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_y").value_or(0.0)),
                static_cast<float>(extras.number("pm_velocity_z").value_or(0.0))};
            source.options.initial_temperature = static_cast<float>(
                extras.number("pm_temperature").value_or(20.0));
            if (!finite(source.spacing) || source.spacing < 0 || !finite(source.options.initial_velocity)) {
                error = name + ": invalid fluid source spacing or velocity";
                return false;
            }
            const auto state = node_state(node);
            for (cgltf_size primitive = 0; primitive < node.mesh->primitives_count; ++primitive) {
                TriangleMesh mesh;
                if (!append_primitive(node.mesh->primitives[primitive], scale, false, name, mesh, error)) return false;
                const auto offset = static_cast<std::uint32_t>(source.vertices.size());
                for (const auto &vertex : mesh.vertices)
                    source.vertices.push_back(add(state.position, rotate(state.orientation, vertex.position)));
                for (const auto index : mesh.indices) source.indices.push_back(offset + index);
            }
            output.particle_sources.push_back(std::move(source));
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
        output.destroy_planes.push_back({.plane = plane});
    }
    if (!output.particle_sources.empty()) {
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
                                // Pressure cloth can transiently compress a
                                // valid geometry volume above the ordinary
                                // free-surface density. The solver visits all
                                // neighbors; this is a safety threshold, not
                                // a storage allocation.
                                .maximum_neighbors = 512U,
                                .repulsion = 30.0F,
                                .viscosity = 0.0F,
                                .velocity_damping = 0.4F,
                                .maximum_speed = 3.0F,
                                .normal_damping = 2.0F,
                                .rest_particle_volume =
                                    std::sqrt(0.5F) * spacing * spacing * spacing,
                                .maximum_pair_acceleration = 55.0F};
    }
    if (output.rigid_bodies.empty() && output.cloths.empty() &&
        output.soft_bodies.empty() && output.ropes.empty() &&
        output.particle_sources.empty() && output.destroy_planes.empty() &&
        output.initial_particles.empty() && !output.has_smoke) {
        error = "GLB contains no ParallelMater physics objects";
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

Status scene_world_options(const SceneDefinition &scene, WorldOptions &output,
                           PhysicsDebugOptions physics_debug) noexcept {
    std::size_t triangle_meshes = 0U;
    for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
        const RigidBodyDefinition &body = scene.rigid_bodies[index];
        const bool collision = !body.collision_mesh_indices.empty();
        const auto &indices = collision ? body.collision_mesh_indices
                                        : body.mesh_indices;
        bool already_uploaded = false;
        for (std::size_t previous = 0U; previous < index; ++previous) {
            const RigidBodyDefinition &candidate = scene.rigid_bodies[previous];
            const bool candidate_collision =
                !candidate.collision_mesh_indices.empty();
            const auto &candidate_indices = candidate_collision
                ? candidate.collision_mesh_indices : candidate.mesh_indices;
            if (collision == candidate_collision && indices == candidate_indices) {
                already_uploaded = true;
                break;
            }
        }
        triangle_meshes += already_uploaded ? 0U : 1U;
    }
    std::size_t paint_fields = 0U;
    for (std::size_t body_index = 0U; body_index < scene.rigid_bodies.size();
         ++body_index) {
        const RigidBodyDefinition &body = scene.rigid_bodies[body_index];
        if (!body.paintable) continue;
        paint_fields += body.mesh_indices.size();
        for (std::uint32_t mesh : body.mesh_indices) {
            bool already_uploaded = false;
            for (std::size_t previous = 0U; previous < body_index; ++previous) {
                const RigidBodyDefinition &candidate =
                    scene.rigid_bodies[previous];
                already_uploaded = candidate.paintable &&
                    std::find(candidate.mesh_indices.begin(),
                              candidate.mesh_indices.end(), mesh) !=
                        candidate.mesh_indices.end();
                if (already_uploaded) break;
            }
            triangle_meshes += already_uploaded ? 0U : 1U;
        }
    }
    for (const ClothDefinition &cloth : scene.cloths)
        paint_fields += cloth.paintable ? 1U : 0U;
    const std::size_t maximum = std::numeric_limits<std::uint32_t>::max();
    if (scene.rigid_bodies.size() > maximum || triangle_meshes > maximum ||
        scene.particle_sources.size() > maximum ||
        scene.destroy_planes.size() > maximum || paint_fields > maximum ||
        scene.cloths.size() > maximum || scene.soft_bodies.size() > maximum ||
        scene.ropes.size() > maximum ||
        (!scene.cloths.empty() && scene.soft_bodies.size() > maximum / scene.cloths.size())) {
        return {StatusCode::capacity_exceeded, cudaSuccess,
                "gallery scene exceeds world capacity range"};
    }
    output = {
        .fluid_capacity = 1U,
        .smoke_capacity = scene.has_smoke ? 1U : 0U,
        .fluid_smoke_coupling_capacity = static_cast<std::uint32_t>(
            scene.thermal_surfaces.size()),
        .smoke_soft_body_coupling_capacity = scene.has_smoke
            ? static_cast<std::uint32_t>(scene.soft_bodies.size()) : 0U,
        .smoke_cloth_coupling_capacity = scene.has_smoke
            ? static_cast<std::uint32_t>(scene.cloths.size()) : 0U,
        .smoke_rope_coupling_capacity = scene.has_smoke
            ? static_cast<std::uint32_t>(scene.ropes.size()) : 0U,
        .smoke_rigid_coupling_capacity = scene.has_smoke
            ? static_cast<std::uint32_t>(std::count_if(
                  scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
                  [&](const RigidBodyDefinition &body) {
                      return body.smoke_collider ||
                          body.options.motion == MotionType::dynamic ||
                          (!scene.smoke_obstacle_name.empty() &&
                           (body.name == scene.smoke_obstacle_name ||
                            body.source_name == scene.smoke_obstacle_name));
                  })) : 0U,
        .rigid_body_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, scene.rigid_bodies.size())),
        .triangle_mesh_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, triangle_meshes)),
        .particle_source_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, scene.particle_sources.size())),
        .particle_destroy_plane_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, scene.destroy_planes.size())),
        .paint_field_capacity = static_cast<std::uint32_t>(paint_fields),
        .paint_rule_capacity = static_cast<std::uint32_t>(paint_fields),
        .cloth_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, scene.cloths.size())),
        .soft_body_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, scene.soft_bodies.size())),
        .fluid_cloth_coupling_capacity = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, scene.cloths.size())),
        .soft_body_cloth_coupling_capacity = static_cast<std::uint32_t>(
            scene.cloths.size() * scene.soft_bodies.size()),
        .fluid_soft_body_coupling_capacity = static_cast<std::uint32_t>(
            scene.soft_bodies.size()),
        .rope_capacity = static_cast<std::uint32_t>(scene.ropes.size()),
        .fluid_rope_coupling_capacity = static_cast<std::uint32_t>(scene.ropes.size()),
        .rope_soft_body_coupling_capacity = static_cast<std::uint32_t>(
            scene.ropes.size()*scene.soft_bodies.size()),
        .rope_cloth_coupling_capacity = static_cast<std::uint32_t>(
            scene.ropes.size()*scene.cloths.size()),
        .physics_debug = physics_debug};
    return {};
}

Status instantiate_scene(const SceneDefinition &scene, World &world,
                         SceneInstance &output) noexcept {
    output = {};
    std::unordered_map<std::string, TriangleMeshId> mesh_cache;
    try {
        output.rigid_bodies.reserve(scene.rigid_bodies.size());
        output.ropes.reserve(scene.ropes.size());
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
    if (scene.has_smoke) {
        int obstacle = -1;
        for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index) {
            if (!scene.smoke_obstacle_name.empty() &&
                (scene.rigid_bodies[index].name == scene.smoke_obstacle_name ||
                 scene.rigid_bodies[index].source_name == scene.smoke_obstacle_name)) {
                if (obstacle >= 0)
                    return {StatusCode::invalid_argument, cudaSuccess,
                            "smoke obstacle name is ambiguous"};
                obstacle = static_cast<int>(index);
            }
        }
        if (!scene.smoke_obstacle_name.empty() && obstacle < 0)
            return {StatusCode::invalid_argument, cudaSuccess,
                    "smoke obstacle rigid mesh was not found"};
        SmokeOptions options = scene.smoke_options;
        const Status status = world.add_smoke(options, output.smoke);
        if (!status) return status;
        output.has_smoke = true;
        for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index) {
            if (static_cast<int>(index) != obstacle &&
                !scene.rigid_bodies[index].smoke_collider &&
                scene.rigid_bodies[index].options.motion != MotionType::dynamic)
                continue;
            SmokeRigidCouplingId coupling{};
            const Status coupled = world.add_smoke_rigid_coupling(
                {.smoke = output.smoke, .body = output.rigid_bodies[index],
                 .tracer_contact = true},
                coupling);
            if (!coupled) return coupled;
            output.smoke_rigid_couplings.push_back(coupling);
        }
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
            .velocity_damping = definition.velocity_damping,
            .contact_friction = definition.contact_friction,
            .solver_iterations = definition.solver_iterations,
            .break_strain = definition.break_strain,
            .fracture_persistence_substeps =
                definition.fracture_persistence_substeps,
            .impact_break_impulse = definition.impact_break_impulse,
            .preserve_volume = definition.preserve_volume,
            .target_volume = definition.target_volume,
            .volume_compliance = definition.volume_compliance}, cloth);
        if (!status) return status;
        output.cloths.push_back(cloth);
    }
    if (output.has_smoke) {
        for (const ClothId cloth : output.cloths) {
            SmokeClothCouplingId coupling{};
            const Status status = world.add_smoke_cloth_coupling(
                {.smoke = output.smoke, .cloth = cloth}, coupling);
            if (!status) return status;
            output.smoke_cloth_couplings.push_back(coupling);
        }
    }
    for (const SoftBodyDefinition &definition : scene.soft_bodies) {
        if (definition.mesh_index >= scene.meshes.size())
            return {StatusCode::invalid_argument, cudaSuccess,
                    "gallery soft-body mesh index is invalid"};
        const TriangleMesh &mesh = scene.meshes[definition.mesh_index];
        std::vector<Vec3> surface_vertices;
        try {
            surface_vertices.reserve(mesh.vertices.size());
            for (const Vertex &vertex : mesh.vertices)
                surface_vertices.push_back(vertex.position);
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "failed to assemble gallery soft-body surface"};
        }
        SoftBodyId body{};
        const Status status = world.add_soft_body({
            .nodes = {definition.nodes.data(), definition.nodes.size()},
            .bonds = {definition.bonds.data(), definition.bonds.size()},
            .inverse_masses = {definition.inverse_masses.data(),
                               definition.inverse_masses.size()},
            .surface_vertices = {surface_vertices.data(),
                                 surface_vertices.size()},
            .surface_triangle_indices = {mesh.indices.data(),
                                         mesh.indices.size()},
            .surface_bindings = {definition.surface_bindings.data(),
                                 definition.surface_bindings.size()},
            .node_mass = definition.node_mass,
            .node_radius = definition.node_radius,
            .stretch_compliance = definition.stretch_compliance,
            .velocity_damping = definition.velocity_damping,
            .spring_damping = definition.spring_damping,
            .contact_friction = definition.contact_friction,
            .shape_matching_stiffness =
                definition.shape_matching_stiffness,
            .maximum_projection_fraction =
                definition.maximum_projection_fraction,
            .constraint_velocity_response =
                definition.constraint_velocity_response,
            .maximum_speed = definition.maximum_speed,
            .solver_iterations = definition.solver_iterations}, body);
        if (!status) return status;
        output.soft_bodies.push_back(body);
    }
    if (output.has_smoke) {
        for (const SoftBodyId body : output.soft_bodies) {
            SmokeSoftBodyCouplingId coupling{};
            const Status status = world.add_smoke_soft_body_coupling(
                {.smoke = output.smoke, .soft_body = body}, coupling);
            if (!status) return status;
            output.smoke_soft_body_couplings.push_back(coupling);
        }
    }
    for (const auto &rope : scene.ropes) {
        auto options=rope.options;
        options.centerline={rope.centerline.data(),rope.centerline.size()};
        for (int body : {rope.first_body, rope.last_body})
            if (body >= 0 && static_cast<std::size_t>(body) >= output.rigid_bodies.size())
                return {StatusCode::invalid_argument, cudaSuccess,
                        "gallery rope attachment index is invalid"};
        if(rope.first_body>=0)options.first.body=output.rigid_bodies[rope.first_body];
        if(rope.last_body>=0)options.last.body=output.rigid_bodies[rope.last_body];
        RopeId id;const auto status=world.add_rope(options,id);if(!status)return status;
        output.ropes.push_back(id);
        for(std::size_t soft=0;soft<output.soft_bodies.size();++soft) {
            RopeSoftBodyCouplingId coupling{};
            const auto coupled=world.add_rope_soft_body_coupling({
                .rope=id,.soft_body=output.soft_bodies[soft],
                .attach_first=rope.first_soft_body==static_cast<int>(soft),
                .attach_last=rope.last_soft_body==static_cast<int>(soft)},coupling);
            if(!coupled)return coupled;
            output.rope_soft_body_couplings.push_back(coupling);
        }
        for(std::size_t sheet=0;sheet<output.cloths.size();++sheet) {
            const bool first=rope.first_cloth==static_cast<int>(sheet);
            const bool last=rope.last_cloth==static_cast<int>(sheet);
            if(!first && !last)continue;
            RopeClothCouplingId coupling{};
            const auto coupled=world.add_rope_cloth_coupling({
                .rope=id,.cloth=output.cloths[sheet],
                .first_vertex=first?rope.first_cloth_vertex:UINT32_MAX,
                .last_vertex=last?rope.last_cloth_vertex:UINT32_MAX},coupling);
            if(!coupled)return coupled;
            output.rope_cloth_couplings.push_back(coupling);
        }
    }
    if (output.has_smoke) {
        for (const RopeId rope : output.ropes) {
            SmokeRopeCouplingId coupling{};
            const Status status = world.add_smoke_rope_coupling(
                {.smoke = output.smoke, .rope = rope}, coupling);
            if (!status) return status;
            output.smoke_rope_couplings.push_back(coupling);
        }
    }
    for (SoftBodyId body : output.soft_bodies) {
        for (std::size_t sheet = 0U; sheet < output.cloths.size(); ++sheet) {
            SoftBodyClothCouplingId coupling{};
            const Status status = world.add_soft_body_cloth_coupling(
                {.soft_body = body, .cloth = output.cloths[sheet],
                 .friction = scene.cloths[sheet].contact_friction}, coupling);
            if (!status) return status;
            output.soft_body_cloth_couplings.push_back(coupling);
        }
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
        for (SoftBodyId body : output.soft_bodies) {
            FluidSoftBodyCouplingId coupling{};
            status = world.add_fluid_soft_body_coupling(
                {.fluid = output.fluid, .soft_body = body}, coupling);
            if (!status) return status;
            output.fluid_soft_body_couplings.push_back(coupling);
        }
        for (RopeId rope : output.ropes) {
            FluidRopeCouplingId coupling{};
            status = world.add_fluid_rope_coupling(
                {.fluid = output.fluid, .rope = rope}, coupling);
            if (!status) return status;
            output.fluid_rope_couplings.push_back(coupling);
        }
        for (const auto &source : scene.particle_sources) {
            auto options = source.options;
            options.fluid = output.fluid;
            ParticleSourceId id{};
            status = world.add_particle_source(
                {{source.vertices.data(), source.vertices.size()},
                 {source.indices.data(), source.indices.size()}, source.spacing}, options, id);
            if (!status) return status;
        }
        for (ParticleDestroyPlaneOptions options : scene.destroy_planes) {
            options.fluid = output.fluid;
            ParticleDestroyPlaneId id{};
            status = world.add_particle_destroy_plane(options, id);
            if (!status) return status;
        }
    }
    if (!scene.thermal_surfaces.empty() &&
        (!output.has_fluid || !output.has_smoke))
        return {StatusCode::invalid_argument, cudaSuccess,
                "thermal surfaces require both water and smoke"};
    for (const auto &surface : scene.thermal_surfaces) {
        FluidSmokeCouplingId coupling{};
        const Status status = world.add_fluid_smoke_coupling({
            .fluid = output.fluid, .smoke = output.smoke,
            .heater = surface.plane,
            .heater_temperature = surface.temperature,
            .heat_transfer_rate = surface.heat_transfer_rate,
            .wind_drag = surface.smoke_drag,
            .steam_rise_speed = surface.steam_rise_speed}, coupling);
        if (!status) return status;
        output.fluid_smoke_couplings.push_back(coupling);
    }
    for (std::size_t index = 0U; index < scene.cloths.size(); ++index) {
        if (!scene.cloths[index].contains_fluid) continue;
        if (!output.has_fluid || index >= output.cloths.size())
            return {StatusCode::invalid_argument, cudaSuccess,
                    "contained cloth needs a scene fluid"};
        FluidClothCouplingId coupling{};
        const Status status = world.add_fluid_cloth_coupling(
            {.fluid = output.fluid, .cloth = output.cloths[index]}, coupling);
        if (!status) return status;
        output.fluid_cloth_couplings.push_back(coupling);
    }
    try {
    const auto bind_paint = [&](std::uint32_t owner,
                                std::uint32_t mesh_index,
                                PaintFieldOptions options,
                                PaintRuleOptions rule_options) -> Status {
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
        rule_options.target = field;
        status = world.add_paint_rule(rule_options, rule);
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
                 .height = body.paint_resolution},
                {.source = output.fluid});
            if (!status) return status;
        }
    }
    for (std::uint32_t cloth_index = 0U;
         cloth_index < scene.cloths.size(); ++cloth_index) {
        const ClothDefinition &cloth = scene.cloths[cloth_index];
        if (!cloth.paintable) continue;
        if (cloth.paint_source.empty())
            return {StatusCode::invalid_argument, cudaSuccess,
                    "paintable cloth needs an authored rigid paint source"};
        const auto source = std::find_if(scene.rigid_bodies.begin(),
            scene.rigid_bodies.end(), [&](const RigidBodyDefinition &body) {
                return body.source_name == cloth.paint_source;
            });
        if (source == scene.rigid_bodies.end() ||
            source->options.motion != MotionType::dynamic)
            return {StatusCode::invalid_argument, cudaSuccess,
                    "cloth paint source must name a dynamic rigid body"};
        if (std::any_of(source + 1, scene.rigid_bodies.end(),
                [&](const RigidBodyDefinition &body) {
                    return body.source_name == cloth.paint_source;
                }))
            return {StatusCode::invalid_argument, cudaSuccess,
                    "cloth paint source names more than one rigid body"};
        const auto source_index = static_cast<std::size_t>(
            source - scene.rigid_bodies.begin());
        const Status status = bind_paint(UINT32_MAX, cloth.mesh_index,
            {.cloth = output.cloths[cloth_index],
             .width = cloth.paint_resolution,
             .height = cloth.paint_resolution},
            {.rigid_source = output.rigid_bodies[source_index],
             .brush_radius = cloth.paint_brush_radius});
        if (!status) return status;
    }
    } catch (...) {
        return {StatusCode::out_of_memory, cudaSuccess,
                "failed to assemble gallery paint bindings"};
    }
    return {};
}

Status create_scene_world(const SceneDefinition &scene, World &world,
                          SceneInstance &output,
                          PhysicsDebugOptions physics_debug) noexcept {
    WorldOptions options{};
    Status status = scene_world_options(scene, options, physics_debug);
    if (!status) return status;
    status = World::create(options, world);
    return status ? instantiate_scene(scene, world, output) : status;
}

} // namespace parallel_mater::gallery
