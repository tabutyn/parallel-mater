// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
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
    if (!schema || *schema != 1.0) {
        error = node_name + ": unsupported or missing pm_schema";
        return false;
    }
    if (!system || *system != "rigid_body") {
        error = node_name + ": unsupported pm_system";
        return false;
    }
    const std::optional<std::string> motion = extras.string("pm_motion");
    const std::optional<std::string> collider = extras.string("pm_collider");
    if (!motion || !collider) {
        error = node_name + ": pm_motion and pm_collider are required";
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

    if (*collider == "sphere") {
        options.shape.type = ShapeType::sphere;
    } else if (*collider == "box") {
        options.shape.type = ShapeType::box;
    } else if (*collider == "capsule") {
        options.shape.type = ShapeType::capsule;
    } else if (*collider == "plane") {
        options.shape.type = ShapeType::plane;
    } else if (*collider == "triangles") {
        options.shape.type = ShapeType::triangle_mesh;
    } else {
        error = node_name + ": invalid pm_collider";
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

[[nodiscard]] bool finalize_body_geometry(SceneDefinition &scene,
                                          RigidBodyDefinition &body,
                                          std::string &error) {
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
    body.options.initial_state.position = add(
        body.options.initial_state.position,
        rotate(body.options.initial_state.orientation, center));

    switch (body.options.shape.type) {
    case ShapeType::sphere: {
        const float largest = std::max({half.x, half.y, half.z});
        const float smallest = std::min({half.x, half.y, half.z});
        if (largest - smallest > largest * 0.02F) {
            error = body.name + ": sphere scale must be uniform";
            return false;
        }
        body.options.shape = CollisionShape::sphere(largest);
        break;
    }
    case ShapeType::box:
        body.options.shape = CollisionShape::box(half);
        break;
    case ShapeType::capsule: {
        const float radius = std::max(half.x, half.z);
        if (std::fabs(half.x - half.z) > radius * 0.02F ||
            half.y + k_bounds_epsilon < radius) {
            error = body.name + ": capsule must be local-Y with circular XZ radius";
            return false;
        }
        body.options.shape = CollisionShape::capsule(radius, half.y - radius);
        break;
    }
    case ShapeType::plane:
        body.options.shape = CollisionShape::plane();
        break;
    case ShapeType::triangle_mesh:
        // The World-owned mesh handle is created during scene instantiation.
        break;
    }
    return true;
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

    for (cgltf_size node_index = 0; node_index < data->nodes_count; ++node_index) {
        const cgltf_node &node = data->nodes[node_index];
        if (node.extras.data == nullptr) {
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
        if (!finalize_body_geometry(output, body, error)) {
            return false;
        }
        output.rigid_bodies.push_back(std::move(body));
    }
    if (output.rigid_bodies.empty()) {
        error = "GLB contains no ParallelMater rigid bodies";
        return false;
    }
    return true;
}

Status instantiate_scene(const SceneDefinition &scene, World &world,
                         SceneInstance &output) noexcept {
    output = {};
    try {
        output.rigid_bodies.reserve(scene.rigid_bodies.size());
        output.collision_meshes.reserve(scene.rigid_bodies.size());
    } catch (...) {
        return {StatusCode::out_of_memory, cudaSuccess,
                "failed to allocate gallery body bindings"};
    }
    for (const RigidBodyDefinition &definition : scene.rigid_bodies) {
        RigidBodyOptions options = definition.options;
        if (options.shape.type == ShapeType::triangle_mesh) {
            std::vector<Vec3> vertices;
            std::vector<std::uint32_t> indices;
            try {
                std::size_t vertex_count = 0U;
                std::size_t index_count = 0U;
                for (const std::uint32_t mesh_index : definition.mesh_indices) {
                    vertex_count += scene.meshes[mesh_index].vertices.size();
                    index_count += scene.meshes[mesh_index].indices.size();
                }
                if (vertex_count > std::numeric_limits<std::uint32_t>::max() ||
                    index_count > std::numeric_limits<std::uint32_t>::max()) {
                    return {StatusCode::capacity_exceeded, cudaSuccess,
                            "gallery triangle collider exceeds uint32 range"};
                }
                vertices.reserve(vertex_count);
                indices.reserve(index_count);
                for (const std::uint32_t mesh_index : definition.mesh_indices) {
                    const TriangleMesh &mesh = scene.meshes[mesh_index];
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
                        "failed to assemble gallery triangle collider"};
            }

            Vec3 *device_vertices = nullptr;
            std::uint32_t *device_indices = nullptr;
            cudaError_t error = cudaMalloc(
                reinterpret_cast<void **>(&device_vertices),
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
                        "failed to upload gallery triangle collider"};
            }
            TriangleMeshId mesh_id{};
            const Status mesh_status = world.add_triangle_mesh(
                {device_vertices, vertices.size()},
                {device_indices, indices.size()}, mesh_id);
            cudaFree(device_indices);
            cudaFree(device_vertices);
            if (!mesh_status) {
                return mesh_status;
            }
            output.collision_meshes.push_back(mesh_id);
            options.shape = CollisionShape::triangle_mesh(mesh_id);
        }
        RigidBodyId body{};
        const Status status = world.add_rigid_body(options, body);
        if (!status) {
            return status;
        }
        output.rigid_bodies.push_back(body);
    }
    return {};
}

} // namespace parallel_mater::gallery
