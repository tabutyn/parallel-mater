// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>

#include "renderer_shared.hpp"
#include "fluid_surface.hpp"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::gallery {
namespace {

using optix_shared::HitData;
using optix_shared::LaunchParameters;

static_assert(sizeof(Vertex) == sizeof(optix_shared::Vertex));
static_assert(offsetof(Vertex, position) == offsetof(optix_shared::Vertex, position));
static_assert(offsetof(Vertex, normal) == offsetof(optix_shared::Vertex, normal));

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void check_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void check_driver(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *description = nullptr;
    cuGetErrorString(result, &description);
    fail(std::string(operation) + ": " +
         (description != nullptr ? description : "unknown CUDA driver error"));
}

void check_optix(OptixResult result, const char *operation) {
    if (result != OPTIX_SUCCESS) {
        fail(std::string(operation) + ": " + optixGetErrorName(result));
    }
}

[[nodiscard]] std::string read_text(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("cannot open OptiX PTX: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    input.seekg(0, std::ios::beg);
    if (length <= 0) {
        fail("OptiX PTX is empty: " + path.string());
    }
    std::string result(static_cast<std::size_t>(length), '\0');
    input.read(result.data(), length);
    if (!input) {
        fail("cannot read OptiX PTX: " + path.string());
    }
    return result;
}

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          size_(std::exchange(other.size_, 0U)) {}
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            pointer_ = std::exchange(other.pointer_, nullptr);
            size_ = std::exchange(other.size_, 0U);
        }
        return *this;
    }

    void resize(std::size_t size) {
        if (size == size_) {
            return;
        }
        reset();
        if (size != 0U) {
            check_cuda(cudaMalloc(&pointer_, size), "cudaMalloc");
            size_ = size;
        }
    }

    void upload(const void *source, std::size_t size) {
        resize(size);
        if (size != 0U) {
            check_cuda(cudaMemcpy(pointer_, source, size, cudaMemcpyHostToDevice),
                       "cudaMemcpy host to device");
        }
    }

    template <typename T> void upload(const std::vector<T> &source) {
        upload(source.data(), source.size() * sizeof(T));
    }

    [[nodiscard]] CUdeviceptr device_pointer() const noexcept {
        return reinterpret_cast<CUdeviceptr>(pointer_);
    }

    [[nodiscard]] void *pointer() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

  private:
    void reset() noexcept {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
            pointer_ = nullptr;
            size_ = 0U;
        }
    }

    void *pointer_{};
    std::size_t size_{};
};

struct EmptyRecordData {};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) ShaderRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    T data{};
};

using RaygenRecord = ShaderRecord<EmptyRecordData>;
using MissRecord = ShaderRecord<EmptyRecordData>;
using HitRecord = ShaderRecord<HitData>;

struct Geometry {
    DeviceBuffer vertices{};
    DeviceBuffer triangles{};
    DeviceBuffer acceleration{};
    OptixTraversableHandle handle{};
};

struct RenderBinding {
    std::uint32_t body_index{};
    std::uint32_t mesh_index{};
};

[[nodiscard]] float3 make_float(Vec3 value) {
    return make_float3(value.x, value.y, value.z);
}

[[nodiscard]] Vec3 subtract(Vec3 first, Vec3 second) {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

[[nodiscard]] Vec3 multiply(Vec3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
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
    const float length = std::sqrt(std::max(dot(value, value), 1.0e-20F));
    return multiply(value, 1.0F / length);
}

void write_transform(const RigidBodyState &state, float output[12]) {
    const Quaternion q = state.orientation;
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;
    output[0] = 1.0F - 2.0F * (yy + zz);
    output[1] = 2.0F * (xy - wz);
    output[2] = 2.0F * (xz + wy);
    output[3] = state.position.x;
    output[4] = 2.0F * (xy + wz);
    output[5] = 1.0F - 2.0F * (xx + zz);
    output[6] = 2.0F * (yz - wx);
    output[7] = state.position.y;
    output[8] = 2.0F * (xz - wy);
    output[9] = 2.0F * (yz + wx);
    output[10] = 1.0F - 2.0F * (xx + yy);
    output[11] = state.position.z;
}

void optix_log(unsigned int level, const char *tag, const char *message,
               void *) {
    if (level <= 1U) {
        (void)tag;
        (void)message;
    }
}

} // namespace

struct OptixRenderer::Impl {
    std::uint32_t width{};
    std::uint32_t height{};
    OptixDeviceContext context{};
    OptixModule module{};
    OptixProgramGroup raygen_group{};
    OptixProgramGroup miss_group{};
    OptixProgramGroup hit_group{};
    OptixPipeline pipeline{};
    OptixShaderBindingTable shader_binding_table{};
    std::vector<Geometry> geometry{};
    std::vector<RenderBinding> bindings{};
    DeviceBuffer raygen_record{};
    DeviceBuffer miss_record{};
    DeviceBuffer hit_records{};
    DeviceBuffer instances{};
    DeviceBuffer instance_acceleration{};
    DeviceBuffer instance_scratch{};
    DeviceBuffer launch_parameters{};
    DeviceBuffer image{};
    DeviceBuffer depth{};
    std::vector<float> host_depth{};
    std::unique_ptr<FluidSurface> fluid_surface{};
    OptixTraversableHandle scene_handle{};
    std::size_t instance_scratch_build_size{};
    std::size_t instance_scratch_update_size{};

    ~Impl() {
        cudaDeviceSynchronize();
        if (pipeline != nullptr) {
            optixPipelineDestroy(pipeline);
        }
        if (hit_group != nullptr) {
            optixProgramGroupDestroy(hit_group);
        }
        if (miss_group != nullptr) {
            optixProgramGroupDestroy(miss_group);
        }
        if (raygen_group != nullptr) {
            optixProgramGroupDestroy(raygen_group);
        }
        if (module != nullptr) {
            optixModuleDestroy(module);
        }
        if (context != nullptr) {
            optixDeviceContextDestroy(context);
        }
    }

    void create_context() {
        check_cuda(cudaFree(nullptr), "initialize CUDA runtime");
        check_driver(cuInit(0), "cuInit");
        CUcontext cuda_context = nullptr;
        check_driver(cuCtxGetCurrent(&cuda_context), "cuCtxGetCurrent");
        if (cuda_context == nullptr) {
            fail("CUDA did not create a current context");
        }
        check_optix(optixInit(), "optixInit");
        OptixDeviceContextOptions options{};
        options.logCallbackFunction = &optix_log;
        options.logCallbackLevel = 2U;
        check_optix(optixDeviceContextCreate(cuda_context, &options, &context),
                    "optixDeviceContextCreate");
    }

    void create_pipeline(const std::filesystem::path &ptx_path) {
        const std::string ptx = read_text(ptx_path);
        OptixModuleCompileOptions module_options{};
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        OptixPipelineCompileOptions pipeline_options{};
        pipeline_options.usesMotionBlur = false;
        pipeline_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_options.numPayloadValues = 4;
        pipeline_options.numAttributeValues = 2;
        pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_options.pipelineLaunchParamsVariableName = "params";
        pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        std::array<char, 4096> log{};
        std::size_t log_size = log.size();
        const OptixResult module_result = optixModuleCreate(
            context, &module_options, &pipeline_options, ptx.data(), ptx.size(),
            log.data(), &log_size, &module);
        if (module_result != OPTIX_SUCCESS) {
            fail(std::string("optixModuleCreate: ") + log.data());
        }

        OptixProgramGroupOptions group_options{};
        OptixProgramGroupDesc raygen_description{};
        raygen_description.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_description.raygen.module = module;
        raygen_description.raygen.entryFunctionName = "__raygen__primary";
        log_size = log.size();
        check_optix(optixProgramGroupCreate(
                        context, &raygen_description, 1U, &group_options,
                        log.data(), &log_size, &raygen_group),
                    "create OptiX raygen program");

        OptixProgramGroupDesc miss_description{};
        miss_description.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_description.miss.module = module;
        miss_description.miss.entryFunctionName = "__miss__sky";
        log_size = log.size();
        check_optix(optixProgramGroupCreate(
                        context, &miss_description, 1U, &group_options, log.data(),
                        &log_size, &miss_group),
                    "create OptiX miss program");

        OptixProgramGroupDesc hit_description{};
        hit_description.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_description.hitgroup.moduleCH = module;
        hit_description.hitgroup.entryFunctionNameCH = "__closesthit__surface";
        log_size = log.size();
        check_optix(optixProgramGroupCreate(
                        context, &hit_description, 1U, &group_options, log.data(),
                        &log_size, &hit_group),
                    "create OptiX hit program");

        const std::array<OptixProgramGroup, 3> groups{raygen_group, miss_group,
                                                      hit_group};
        OptixPipelineLinkOptions link_options{};
        link_options.maxTraceDepth = 1U;
        log_size = log.size();
        check_optix(optixPipelineCreate(
                        context, &pipeline_options, &link_options, groups.data(),
                        groups.size(), log.data(), &log_size, &pipeline),
                    "optixPipelineCreate");
        check_optix(optixPipelineSetStackSizeFromCallDepths(pipeline, 1U, 0U, 0U,
                                                            0U, 2U),
                    "optixPipelineSetStackSizeFromCallDepths");
    }

    void create_geometry(const SceneDefinition &scene) {
        geometry.resize(scene.meshes.size());
        for (std::size_t index = 0; index < scene.meshes.size(); ++index) {
            const TriangleMesh &mesh = scene.meshes[index];
            Geometry &gpu = geometry[index];
            gpu.vertices.upload(mesh.vertices);
            gpu.triangles.upload(mesh.indices);
            CUdeviceptr vertex_buffer = gpu.vertices.device_pointer();
            std::uint32_t geometry_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            OptixBuildInput input{};
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
            input.triangleArray.numVertices =
                static_cast<unsigned int>(mesh.vertices.size());
            input.triangleArray.vertexBuffers = &vertex_buffer;
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3U * sizeof(std::uint32_t);
            input.triangleArray.numIndexTriplets =
                static_cast<unsigned int>(mesh.indices.size() / 3U);
            input.triangleArray.indexBuffer = gpu.triangles.device_pointer();
            input.triangleArray.flags = &geometry_flags;
            input.triangleArray.numSbtRecords = 1U;

            OptixAccelBuildOptions build_options{};
            build_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            build_options.operation = OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes sizes{};
            check_optix(optixAccelComputeMemoryUsage(
                            context, &build_options, &input, 1U, &sizes),
                        "compute OptiX geometry memory");
            DeviceBuffer scratch;
            scratch.resize(sizes.tempSizeInBytes);
            gpu.acceleration.resize(sizes.outputSizeInBytes);
            check_optix(optixAccelBuild(
                            context, nullptr, &build_options, &input, 1U,
                            scratch.device_pointer(), scratch.size(),
                            gpu.acceleration.device_pointer(),
                            gpu.acceleration.size(), &gpu.handle, nullptr, 0U),
                        "build OptiX geometry acceleration");
        }
        check_cuda(cudaDeviceSynchronize(), "synchronize geometry build");
    }

    [[nodiscard]] std::vector<OptixInstance>
    make_instances(const std::vector<RigidBodyState> &states) const {
        std::vector<OptixInstance> result(bindings.size());
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            const RenderBinding binding = bindings[index];
            OptixInstance &instance = result[index];
            write_transform(states[binding.body_index], instance.transform);
            instance.instanceId = static_cast<unsigned int>(index);
            instance.sbtOffset = static_cast<unsigned int>(index);
            instance.visibilityMask = 255U;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = geometry[binding.mesh_index].handle;
        }
        return result;
    }

    void create_instances(const SceneDefinition &scene) {
        std::vector<RigidBodyState> states;
        states.reserve(scene.rigid_bodies.size());
        for (std::uint32_t body_index = 0;
             body_index < scene.rigid_bodies.size(); ++body_index) {
            const RigidBodyDefinition &body = scene.rigid_bodies[body_index];
            states.push_back(body.options.initial_state);
            for (const std::uint32_t mesh_index : body.mesh_indices) {
                bindings.push_back({body_index, mesh_index});
            }
        }
        std::vector<OptixInstance> authored_instances = make_instances(states);
        instances.upload(authored_instances);

        OptixBuildInput input{};
        input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        input.instanceArray.instances = instances.device_pointer();
        input.instanceArray.numInstances =
            static_cast<unsigned int>(authored_instances.size());
        OptixAccelBuildOptions options{};
        options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                             OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        check_optix(optixAccelComputeMemoryUsage(context, &options, &input, 1U,
                                                 &sizes),
                    "compute OptiX instance memory");
        instance_scratch_build_size = sizes.tempSizeInBytes;
        instance_scratch_update_size = sizes.tempUpdateSizeInBytes;
        instance_scratch.resize(
            std::max(instance_scratch_build_size, instance_scratch_update_size));
        instance_acceleration.resize(sizes.outputSizeInBytes);
        check_optix(optixAccelBuild(
                        context, nullptr, &options, &input, 1U,
                        instance_scratch.device_pointer(),
                        instance_scratch_build_size,
                        instance_acceleration.device_pointer(),
                        instance_acceleration.size(), &scene_handle, nullptr, 0U),
                    "build OptiX instance acceleration");
        check_cuda(cudaDeviceSynchronize(), "synchronize instance build");
    }

    void create_shader_binding_table(const SceneDefinition &scene) {
        RaygenRecord raygen{};
        check_optix(optixSbtRecordPackHeader(raygen_group, &raygen),
                    "pack raygen record");
        raygen_record.upload(&raygen, sizeof(raygen));
        MissRecord miss{};
        check_optix(optixSbtRecordPackHeader(miss_group, &miss),
                    "pack miss record");
        miss_record.upload(&miss, sizeof(miss));
        std::vector<HitRecord> hits(bindings.size());
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            const std::uint32_t mesh_index = bindings[index].mesh_index;
            const TriangleMesh &mesh = scene.meshes[mesh_index];
            check_optix(optixSbtRecordPackHeader(hit_group, &hits[index]),
                        "pack hit record");
            hits[index].data.vertices =
                reinterpret_cast<const optix_shared::Vertex *>(
                    geometry[mesh_index].vertices.device_pointer());
            hits[index].data.triangles = reinterpret_cast<const uint3 *>(
                geometry[mesh_index].triangles.device_pointer());
            hits[index].data.base_color = make_float(mesh.base_color);
            hits[index].data.checkerboard = mesh.checkerboard ? 1U : 0U;
        }
        hit_records.upload(hits);
        shader_binding_table = {};
        shader_binding_table.raygenRecord = raygen_record.device_pointer();
        shader_binding_table.missRecordBase = miss_record.device_pointer();
        shader_binding_table.missRecordStrideInBytes = sizeof(MissRecord);
        shader_binding_table.missRecordCount = 1U;
        shader_binding_table.hitgroupRecordBase = hit_records.device_pointer();
        shader_binding_table.hitgroupRecordStrideInBytes = sizeof(HitRecord);
        shader_binding_table.hitgroupRecordCount =
            static_cast<unsigned int>(hits.size());
    }

    [[nodiscard]] std::vector<RigidBodyState>
    read_states(const World &world, const SceneInstance &scene_instance) const {
        RigidBodyDeviceView view{};
        const Status status = world.rigid_body_view(view);
        if (!status) {
            fail(status.message != nullptr ? status.message
                                           : "cannot borrow rigid-body view");
        }
        std::vector<RigidBodyId> ids(view.ids.size);
        std::vector<RigidBodyState> dense_states(view.states.size);
        check_cuda(cudaMemcpy(ids.data(), view.ids.data,
                              ids.size() * sizeof(RigidBodyId),
                              cudaMemcpyDeviceToHost),
                   "copy rigid IDs for rendering");
        check_cuda(cudaMemcpy(dense_states.data(), view.states.data,
                              dense_states.size() * sizeof(RigidBodyState),
                              cudaMemcpyDeviceToHost),
                   "copy rigid states for rendering");
        std::vector<RigidBodyState> result(scene_instance.rigid_bodies.size());
        for (std::size_t body_index = 0;
             body_index < scene_instance.rigid_bodies.size(); ++body_index) {
            const RigidBodyId wanted = scene_instance.rigid_bodies[body_index];
            const auto found = std::find(ids.begin(), ids.end(), wanted);
            if (found == ids.end()) {
                fail("rendered rigid-body handle is no longer alive");
            }
            result[body_index] = dense_states[static_cast<std::size_t>(
                std::distance(ids.begin(), found))];
        }
        return result;
    }

    void update_instances(const std::vector<RigidBodyState> &states) {
        const std::vector<OptixInstance> updated = make_instances(states);
        instances.upload(updated);
        OptixBuildInput input{};
        input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        input.instanceArray.instances = instances.device_pointer();
        input.instanceArray.numInstances = static_cast<unsigned int>(updated.size());
        OptixAccelBuildOptions options{};
        options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                             OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        options.operation = OPTIX_BUILD_OPERATION_UPDATE;
        check_optix(optixAccelBuild(
                        context, nullptr, &options, &input, 1U,
                        instance_scratch.device_pointer(),
                        instance_scratch_update_size,
                        instance_acceleration.device_pointer(),
                        instance_acceleration.size(), &scene_handle, nullptr, 0U),
                    "update OptiX instance acceleration");
    }

    void render(Camera camera, optix_shared::FluidSurfaceView fluid,
                std::vector<std::uint32_t> &rgba) {
        const Vec3 forward = normalize(subtract(camera.target, camera.eye));
        const Vec3 right = normalize(cross(forward, camera.up));
        const Vec3 corrected_up = normalize(cross(right, forward));
        constexpr float radians = 3.14159265358979323846F / 180.0F;
        const float vertical_scale =
            std::tan(camera.vertical_field_of_view_degrees * radians * 0.5F);
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        LaunchParameters parameters{};
        parameters.image = reinterpret_cast<uchar4 *>(image.pointer());
        parameters.depth = reinterpret_cast<float *>(depth.pointer());
        parameters.width = width;
        parameters.height = height;
        parameters.scene = scene_handle;
        parameters.eye = make_float(camera.eye);
        parameters.camera_w = make_float(forward);
        parameters.camera_u = make_float(multiply(right, vertical_scale * aspect));
        parameters.camera_v = make_float(multiply(corrected_up, vertical_scale));
        parameters.fluid = fluid;
        launch_parameters.upload(&parameters, sizeof(parameters));
        check_optix(optixLaunch(pipeline, nullptr,
                                launch_parameters.device_pointer(),
                                launch_parameters.size(), &shader_binding_table,
                                width, height, 1U),
                    "optixLaunch");
        check_cuda(cudaDeviceSynchronize(), "synchronize OptiX render");
        rgba.resize(static_cast<std::size_t>(width) * height);
        check_cuda(cudaMemcpy(rgba.data(), image.pointer(), image.size(),
                              cudaMemcpyDeviceToHost),
                   "copy OptiX image");
        host_depth.resize(static_cast<std::size_t>(width) * height);
        check_cuda(cudaMemcpy(host_depth.data(), depth.pointer(), depth.size(),
                              cudaMemcpyDeviceToHost),
                   "copy OptiX depth");
    }
};

void paint_fluid_particles(const std::vector<Vec3> &positions,
                           const std::vector<float> &foam,
                           const std::vector<std::uint32_t> &ids,
                           float radius, Camera camera,
                           std::uint32_t width, std::uint32_t height,
                           std::vector<float> &depth,
                           FluidRenderMode mode,
                           std::vector<std::uint32_t> &rgba) {
    const bool particle_view = mode == FluidRenderMode::particles;
    const Vec3 forward = normalize(subtract(camera.target, camera.eye));
    const Vec3 right = normalize(cross(forward, camera.up));
    const Vec3 up = normalize(cross(right, forward));
    constexpr float radians = 3.14159265358979323846F / 180.0F;
    const float tangent = std::tan(
        camera.vertical_field_of_view_degrees * radians * 0.5F);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    for (std::size_t particle = 0U; particle < positions.size(); ++particle) {
        if (!particle_view && foam[particle] < 0.25F) continue;
        const std::uint32_t foam_hash =
            (ids[particle] ^ (ids[particle] >> 16U)) * 0x7feb352dU;
        if (!particle_view && foam_hash % 9U != 0U) continue;
        const Vec3 offset = subtract(positions[particle], camera.eye);
        const float forward_distance = dot(offset, forward);
        if (forward_distance <= radius) continue;
        const float normalized_x = dot(offset, right) /
            (forward_distance * tangent * aspect);
        const float normalized_y = dot(offset, up) /
            (forward_distance * tangent);
        const float center_x = (normalized_x + 1.0F) * 0.5F * width;
        const float center_y = (normalized_y + 1.0F) * 0.5F * height;
        const float pixel_radius = std::max(
            particle_view ? 1.2F : 1.0F,
            radius * (particle_view ? 1.0F : 0.65F) * height /
                (2.0F * forward_distance * tangent));
        if (center_x + pixel_radius < 0.0F || center_x - pixel_radius >= width ||
            center_y + pixel_radius < 0.0F || center_y - pixel_radius >= height)
            continue;
        const int x0 = std::max(0, static_cast<int>(center_x - pixel_radius));
        const int y0 = std::max(0, static_cast<int>(center_y - pixel_radius));
        const int x1 = std::min(static_cast<int>(width) - 1,
                                static_cast<int>(center_x + pixel_radius));
        const int y1 = std::min(static_cast<int>(height) - 1,
                                static_cast<int>(center_y + pixel_radius));
        const float source = std::clamp(foam[particle], 0.0F, 1.0F);
        const float center_depth = std::sqrt(dot(offset, offset));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float dx = (x + 0.5F - center_x) / pixel_radius;
                const float dy = (y + 0.5F - center_y) / pixel_radius;
                const float squared = dx * dx + dy * dy;
                if (squared > 1.0F) continue;
                const std::size_t index = static_cast<std::size_t>(y) * width + x;
                const float surface_depth = center_depth -
                    radius * std::sqrt(1.0F - squared);
                if (surface_depth > depth[index] +
                    (particle_view ? 0.01F : 0.12F)) continue;
                if (particle_view) depth[index] = surface_depth;
                const float highlight = particle_view && foam_hash % 9U != 0U
                    ? std::clamp(source * 0.25F, 0.0F, 0.2F)
                    : std::clamp(source * 4.0F, 0.0F, 1.0F);
                const float shading = 0.65F + 0.35F *
                    std::sqrt(1.0F - squared);
                const std::uint32_t old = rgba[index];
                const float alpha = particle_view ? 0.78F + 0.18F * highlight
                                                  : 0.25F + 0.65F * source;
                const auto blend = [&](std::uint32_t original,
                                       float water, float white) {
                    const float target = (water * (1.0F - highlight) +
                                          white * highlight) * shading;
                    return static_cast<std::uint32_t>(std::clamp(
                        original * (1.0F - alpha) + target * alpha,
                        0.0F, 255.0F));
                };
                const std::uint32_t red = blend(old & 255U,
                                                particle_view ? 25.0F : 190.0F,
                                                245.0F);
                const std::uint32_t green = blend((old >> 8U) & 255U,
                                                  particle_view ? 125.0F : 215.0F,
                                                  250.0F);
                const std::uint32_t blue = blend((old >> 16U) & 255U,
                                                 245.0F, 255.0F);
                rgba[index] = 0xff000000U | (blue << 16U) |
                              (green << 8U) | red;
            }
        }
    }
}

OptixRenderer::OptixRenderer() noexcept = default;
OptixRenderer::~OptixRenderer() = default;
OptixRenderer::OptixRenderer(OptixRenderer &&) noexcept = default;
OptixRenderer &OptixRenderer::operator=(OptixRenderer &&) noexcept = default;

bool OptixRenderer::create(const SceneDefinition &scene,
                           const std::filesystem::path &ptx_path,
                           std::uint32_t width, std::uint32_t height,
                           OptixRenderer &output, std::string &error) {
    error.clear();
    if (scene.meshes.empty() || scene.rigid_bodies.empty() || width == 0U ||
        height == 0U) {
        error = "renderer requires a nonempty scene and image dimensions";
        return false;
    }
    try {
        auto implementation = std::make_unique<Impl>();
        implementation->width = width;
        implementation->height = height;
        implementation->create_context();
        implementation->create_pipeline(ptx_path);
        implementation->create_geometry(scene);
        implementation->create_instances(scene);
        implementation->create_shader_binding_table(scene);
        implementation->image.resize(static_cast<std::size_t>(width) * height *
                                     sizeof(std::uint32_t));
        implementation->depth.resize(static_cast<std::size_t>(width) * height *
                                     sizeof(float));
        implementation->launch_parameters.resize(sizeof(LaunchParameters));
        if (!scene.spawn_planes.empty())
            implementation->fluid_surface =
                std::make_unique<FluidSurface>(scene.fluid_options.capacity);
        output.impl_ = std::move(implementation);
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

bool OptixRenderer::render(const World &world, const SceneInstance &instance,
                           Camera camera, std::vector<std::uint32_t> &rgba,
                           std::string &error, RendererTimings *timings,
                           FluidRenderMode fluid_mode) {
    error.clear();
    if (!impl_) {
        error = "renderer is not initialized";
        return false;
    }
    try {
        using clock = std::chrono::steady_clock;
        const auto total_begin = clock::now();
        RendererTimings sample{};
        sample.particle_view = fluid_mode == FluidRenderMode::particles;
        const std::vector<RigidBodyState> states =
            impl_->read_states(world, instance);
        impl_->update_instances(states);
        std::vector<Vec3> positions;
        std::vector<float> foam;
        std::vector<std::uint32_t> ids;
        float particle_radius = 0.0F;
        optix_shared::FluidSurfaceView surface{};
        if (instance.has_fluid) {
            FluidDeviceView view{};
            const Status status = world.fluid_view(instance.fluid, view);
            if (!status) fail(status.message != nullptr ? status.message
                                                       : "cannot borrow fluid view");
            sample.particle_count = view.particle_count;
            particle_radius = view.particle_radius;
            positions.resize(view.particle_count);
            foam.resize(view.particle_count);
            ids.resize(view.particle_count);
            if (view.particle_count != 0U) {
                check_cuda(cudaMemcpy(positions.data(), view.positions.data,
                                      positions.size() * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost),
                           "copy fluid positions for rendering");
                check_cuda(cudaMemcpy(foam.data(), view.foam.data,
                                      foam.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost),
                           "copy foam signal for rendering");
                check_cuda(cudaMemcpy(ids.data(), view.stable_particle_ids.data,
                                      ids.size() * sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToHost),
                           "copy fluid IDs for rendering");
                if (impl_->fluid_surface &&
                    fluid_mode == FluidRenderMode::surface) {
                    sample.surface_gpu_milliseconds =
                        impl_->fluid_surface->update(view, positions);
                    sample.surface_excluded_particle_count =
                        impl_->fluid_surface->excluded_particle_count();
                    surface = impl_->fluid_surface->view();
                }
            }
        }
        const auto raytrace_begin = clock::now();
        impl_->render(camera, surface, rgba);
        const auto foam_begin = clock::now();
        if (instance.has_fluid && !positions.empty()) {
            paint_fluid_particles(positions, foam, ids,
                                  particle_radius,
                                  camera, impl_->width, impl_->height,
                                  impl_->host_depth, fluid_mode, rgba);
        }
        const auto finish = clock::now();
        const auto milliseconds = [](auto start, auto end) {
            return std::chrono::duration<float, std::milli>(end - start).count();
        };
        sample.raytrace_wall_milliseconds = milliseconds(raytrace_begin, foam_begin);
        sample.foam_wall_milliseconds = milliseconds(foam_begin, finish);
        sample.total_wall_milliseconds = milliseconds(total_begin, finish);
        if (timings) *timings = sample;
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

std::uint32_t OptixRenderer::width() const noexcept {
    return impl_ ? impl_->width : 0U;
}

std::uint32_t OptixRenderer::height() const noexcept {
    return impl_ ? impl_->height : 0U;
}

} // namespace parallel_mater::gallery
