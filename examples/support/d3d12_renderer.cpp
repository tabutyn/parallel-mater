// SPDX-License-Identifier: MIT
#include "renderer_backends.hpp"
#include "gallery_shaders.hpp"
#include "vector_math.hpp"
#include "render/render_scene.hpp"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace parallel_mater::gallery {
namespace {
using Microsoft::WRL::ComPtr;

void check(HRESULT result, const char *operation) {
    if (SUCCEEDED(result)) return;
    std::ostringstream message;
    message << operation << " failed: 0x" << std::hex << static_cast<unsigned long>(result);
    throw std::runtime_error(message.str());
}

void barrier(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource,
             D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER value{};
    value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    value.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    commands->ResourceBarrier(1U, &value);
}

struct DrawConstants {
    float eye[4]{}, right[4]{}, up[4]{}, forward[4]{}, color[4]{};
    std::uint32_t draw[4]{};
    std::uint32_t paint[4]{};
    float settings[4]{};
};
static_assert(sizeof(DrawConstants) == 128U);
static_assert(sizeof(PmRigidState) == 52U && sizeof(PmFluidParticleOutput) == 48U);
static_assert(sizeof(PmDomainOutputHeader) == 16U);

class D3D12Renderer final : public Renderer {
    struct Mesh {
        ComPtr<ID3D12Resource> vertices, indices;
        D3D12_VERTEX_BUFFER_VIEW vertex_view{};
        D3D12_INDEX_BUFFER_VIEW index_view{};
        std::uint32_t count{};
        Vec3 color{};
        bool visible{};
    };
    struct Draw { std::uint32_t mesh{}, mode{}, body{}, offset{}; };
    GpuSceneBuffers buffers_;
    std::uint32_t width_{}, height_{};
    float particle_radius_{};
    float support_radius_{};
    std::uint32_t surface_hash_size_{256U};
    static constexpr std::uint32_t surface_samples = 128U * 128U * 128U;
    static constexpr std::uint32_t surface_vertex_capacity = 3U * 524288U;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12Resource> physics_, target_, depth_, constants_, particle_vertices_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> commands_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    ComPtr<ID3D12PipelineState> wireframe_;
    std::array<ComPtr<ID3D12PipelineState>, 6U> surface_pipelines_;
    ComPtr<ID3D12Resource> surface_metadata_, surface_heads_, surface_next_, surface_field_, surface_vertices_, surface_status_;
    ComPtr<ID3D12CommandSignature> surface_draw_;
    ComPtr<ID3D12DescriptorHeap> rtv_, dsv_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_{}, image_handle_{};
    std::uint64_t fence_value_{}, physics_offset_{};
    std::byte *constant_data_{};
    std::vector<Mesh> meshes_;
    std::vector<Draw> draws_;

    ComPtr<ID3D12Resource> storage(std::size_t size, bool readback = false) {
        ComPtr<ID3D12Resource> buffer;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = readback ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = size; description.Height = 1U;
        description.DepthOrArraySize = description.MipLevels = 1U;
        description.SampleDesc.Count = 1U;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        description.Flags = readback ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
            readback ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, IID_PPV_ARGS(&buffer)), "create GPU surface buffer");
        return buffer;
    }

    ComPtr<ID3D12Resource> upload(const void *data, std::size_t size) {
        ComPtr<ID3D12Resource> buffer;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = std::max<std::size_t>(size, 256U);
        description.Height = 1U;
        description.DepthOrArraySize = description.MipLevels = 1U;
        description.SampleDesc.Count = 1U;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buffer)), "create drawing upload");
        if (data && size) {
            void *mapped{};
            D3D12_RANGE no_read{};
            check(buffer->Map(0U, &no_read, &mapped), "map drawing upload");
            std::memcpy(mapped, data, size);
            buffer->Unmap(0U, nullptr);
        }
        return buffer;
    }

    void wait() {
        if (!queue_ || !fence_) return;
        check(queue_->Signal(fence_.Get(), ++fence_value_), "signal drawing fence");
        if (fence_->GetCompletedValue() >= fence_value_) return;
        check(fence_->SetEventOnCompletion(fence_value_, fence_event_), "arm drawing fence");
        if (WaitForSingleObject(fence_event_, INFINITE) != WAIT_OBJECT_0)
            throw std::runtime_error("drawing fence wait failed");
    }

  public:
    D3D12Renderer(const SceneDefinition &scene, const GpuSceneBuffers &buffers,
                  std::uint32_t width, std::uint32_t height)
        : buffers_(buffers), width_(width), height_(height),
          particle_radius_(scene.fluid_options.particle_radius),
          support_radius_(scene.fluid_options.support_radius) {}

    ~D3D12Renderer() override {
        try { wait(); } catch (...) {}
        if (constant_data_) constants_->Unmap(0U, nullptr);
        if (fence_event_) CloseHandle(fence_event_);
        if (image_handle_) CloseHandle(image_handle_);
    }

    void initialize(const SceneDefinition &scene) {
        PmNativeDeviceInfo native{};
        PmNativeBufferInfo output{};
        if (!width_ || !height_ || pmGetNativeDevice(buffers_.device, &native) != PM_SUCCESS ||
            native.backend != PM_NATIVE_BACKEND_D3D12 ||
            pmGetNativeBuffer(buffers_.device, buffers_.output, &output) != PM_SUCCESS)
            throw std::runtime_error("D3D12 drawing requires native public output buffers");
        device_ = reinterpret_cast<ID3D12Device *>(static_cast<std::uintptr_t>(native.device));
        physics_ = reinterpret_cast<ID3D12Resource *>(static_cast<std::uintptr_t>(output.buffer));
        physics_offset_ = output.offset;
        D3D12_COMMAND_QUEUE_DESC queue_description{};
        queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device_->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue_)), "create drawing queue");
        check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator_)), "create drawing allocator");
        check(device_->CreateCommandList(0U, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(),
            nullptr, IID_PPV_ARGS(&commands_)), "create drawing commands");
        check(commands_->Close(), "close drawing commands");
        check(device_->CreateFence(0U, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "create drawing fence");
        fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event_) throw std::runtime_error("create drawing fence event failed");

        D3D12_ROOT_PARAMETER parameters[8]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[2].Descriptor.ShaderRegister = 1U;
        parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        for (UINT index = 3U; index < 8U; ++index) {
            parameters[index].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            parameters[index].Descriptor.ShaderRegister = index - 3U;
            parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC signature{};
        signature.NumParameters = 8U;
        signature.pParameters = parameters;
        signature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> serialized, errors;
        check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1,
            &serialized, &errors), "serialize drawing root signature");
        check(device_->CreateRootSignature(0U, serialized->GetBufferPointer(), serialized->GetBufferSize(),
            IID_PPV_ARGS(&root_)), "create drawing root signature");
        D3D12_INPUT_ELEMENT_DESC elements[]{
            {"POSITION",0U,DXGI_FORMAT_R32G32B32_FLOAT,0U,0U,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0U},
            {"NORMAL",0U,DXGI_FORMAT_R32G32B32_FLOAT,0U,12U,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0U},
            {"TEXCOORD",0U,DXGI_FORMAT_R32G32_FLOAT,0U,24U,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0U}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = root_.Get();
        pipeline.VS = {shaders::vertex, sizeof(shaders::vertex)};
        pipeline.PS = {shaders::fragment, sizeof(shaders::fragment)};
        pipeline.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pipeline.SampleMask = UINT_MAX;
        pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pipeline.RasterizerState.DepthClipEnable = TRUE;
        pipeline.DepthStencilState.DepthEnable = TRUE;
        pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        pipeline.InputLayout = {elements, 3U};
        pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pipeline.NumRenderTargets = 1U;
        pipeline.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pipeline.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        pipeline.SampleDesc.Count = 1U;
        check(device_->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(&pipeline_)), "create Slang drawing pipeline");
        pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
        check(device_->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(&wireframe_)), "create Slang wireframe pipeline");
        if (buffers_.fluid_capacity) {
            const D3D12_SHADER_BYTECODE bytecodes[]{
                {shaders::surfaceClear, sizeof(shaders::surfaceClear)},
                {shaders::surfaceHash, sizeof(shaders::surfaceHash)},
                {shaders::surfaceBounds, sizeof(shaders::surfaceBounds)},
                {shaders::surfaceField, sizeof(shaders::surfaceField)},
                {shaders::surfaceMesh, sizeof(shaders::surfaceMesh)},
                {shaders::surfaceFinalize, sizeof(shaders::surfaceFinalize)}};
            for (std::size_t index = 0U; index < surface_pipelines_.size(); ++index) {
                D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
                compute.pRootSignature = root_.Get(); compute.CS = bytecodes[index];
                check(device_->CreateComputePipelineState(&compute, IID_PPV_ARGS(&surface_pipelines_[index])), "create shared fluid surface pipeline");
            }
            while (surface_hash_size_ < buffers_.fluid_capacity * 2ULL) surface_hash_size_ *= 2U;
            surface_metadata_ = storage(64U);
            surface_heads_ = storage(surface_hash_size_ * sizeof(std::uint32_t));
            surface_next_ = storage(buffers_.fluid_capacity * sizeof(std::uint32_t));
            surface_field_ = storage(surface_samples * sizeof(float));
            surface_vertices_ = storage(surface_vertex_capacity * sizeof(Vec3) * 2U);
            surface_status_ = storage(sizeof(std::uint32_t), true);
            D3D12_INDIRECT_ARGUMENT_DESC argument{};
            argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
            D3D12_COMMAND_SIGNATURE_DESC indirect{};
            indirect.ByteStride = sizeof(D3D12_DRAW_ARGUMENTS);
            indirect.NumArgumentDescs = 1U; indirect.pArgumentDescs = &argument;
            check(device_->CreateCommandSignature(&indirect, nullptr, IID_PPV_ARGS(&surface_draw_)), "create GPU surface indirect draw");
        }

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = width_; texture.Height = height_;
        texture.DepthOrArraySize = texture.MipLevels = 1U;
        texture.SampleDesc.Count = 1U;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = texture.Format;
        check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &texture,
            D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&target_)), "create shared drawing target");
        check(device_->CreateSharedHandle(target_.Get(), nullptr, GENERIC_ALL, nullptr, &image_handle_),
            "share drawing target");
        texture.Format = DXGI_FORMAT_D32_FLOAT;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        clear.Format = texture.Format;
        clear.DepthStencil.Depth = 1.0F;
        check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(&depth_)), "create drawing depth");
        D3D12_DESCRIPTOR_HEAP_DESC descriptor{};
        descriptor.NumDescriptors = 1U;
        descriptor.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        check(device_->CreateDescriptorHeap(&descriptor, IID_PPV_ARGS(&rtv_)), "create drawing RTV");
        descriptor.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        check(device_->CreateDescriptorHeap(&descriptor, IID_PPV_ARGS(&dsv_)), "create drawing DSV");
        device_->CreateRenderTargetView(target_.Get(), nullptr, rtv_->GetCPUDescriptorHandleForHeapStart());
        device_->CreateDepthStencilView(depth_.Get(), nullptr, dsv_->GetCPUDescriptorHandleForHeapStart());

        for (const auto &source : scene.meshes) {
            Mesh mesh{};
            mesh.color = source.base_color;
            mesh.visible = source.visible;
            mesh.count = static_cast<std::uint32_t>(source.indices.size());
            if (mesh.visible && mesh.count) {
                auto vertices = source.vertices;
                auto indices = source.indices;
                const bool fracture_surface = std::any_of(scene.cloths.begin(), scene.cloths.end(),
                    [&](const auto &cloth) { return cloth.mesh_index == meshes_.size() &&
                        (cloth.break_strain > 0.0F || cloth.impact_break_impulse > 0.0F); });
                if (fracture_surface) {
                    vertices.clear();
                    vertices.reserve(indices.size());
                    for (std::uint32_t corner = 0U; corner < indices.size(); ++corner) {
                        vertices.push_back(source.vertices.at(indices[corner]));
                        indices[corner] = corner;
                    }
                }
                mesh.vertices = upload(vertices.data(), vertices.size() * sizeof(Vertex));
                mesh.indices = upload(indices.data(), indices.size() * sizeof(std::uint32_t));
                mesh.vertex_view = {mesh.vertices->GetGPUVirtualAddress(),
                    static_cast<UINT>(vertices.size() * sizeof(Vertex)), sizeof(Vertex)};
                mesh.index_view = {mesh.indices->GetGPUVirtualAddress(),
                    static_cast<UINT>(indices.size() * sizeof(std::uint32_t)), DXGI_FORMAT_R32_UINT};
            }
            meshes_.push_back(std::move(mesh));
        }
        for (std::uint32_t body = 0U; body < scene.rigid_bodies.size(); ++body)
            for (const auto mesh : scene.rigid_bodies[body].mesh_indices)
                draws_.push_back({mesh, 0U, body, static_cast<std::uint32_t>(buffers_.rigid.offset)});
        for (std::uint32_t cloth = 0U; cloth < scene.cloths.size(); ++cloth)
            draws_.push_back({scene.cloths[cloth].mesh_index, 1U, 0U,
                static_cast<std::uint32_t>(buffers_.cloth_surfaces.at(cloth).offset)});
        constants_ = upload(nullptr, (draws_.size() + 2U) * 256U);
        D3D12_RANGE no_read{};
        check(constants_->Map(0U, &no_read, reinterpret_cast<void **>(&constant_data_)), "map drawing constants");
        const std::array<Vertex, 6U> vertices{};
        particle_vertices_ = upload(vertices.data(), sizeof(vertices));
    }

    bool render(const RenderSceneFrame &, Camera camera, std::vector<std::uint32_t> &rgba,
                std::string &error, RendererTimings *timings, FluidRenderMode fluid_mode,
                bool, float rigid_interpolation_alpha) override {
        try {
            const auto start = std::chrono::steady_clock::now();
            if (pmQueueWaitIdle(buffers_.queue) != PM_SUCCESS)
                throw std::runtime_error("physics queue failed before GPU drawing");
            wait();
            check(allocator_->Reset(), "reset drawing allocator");
            check(commands_->Reset(allocator_.Get(), pipeline_.Get()), "reset drawing commands");
            barrier(commands_.Get(), physics_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            barrier(commands_.Get(), target_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
            const auto rtv = rtv_->GetCPUDescriptorHandleForHeapStart(), dsv = dsv_->GetCPUDescriptorHandleForHeapStart();
            const float background[]{0.045F, 0.07F, 0.10F, 1.0F};
            commands_->ClearRenderTargetView(rtv, background, 0U, nullptr);
            commands_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0F, 0U, 0U, nullptr);
            commands_->OMSetRenderTargets(1U, &rtv, FALSE, &dsv);
            const D3D12_VIEWPORT viewport{0.0F, 0.0F, static_cast<float>(width_), static_cast<float>(height_), 0.0F, 1.0F};
            const D3D12_RECT scissor{0, 0, static_cast<LONG>(width_), static_cast<LONG>(height_)};
            commands_->RSSetViewports(1U, &viewport);
            commands_->RSSetScissorRects(1U, &scissor);
            commands_->SetGraphicsRootSignature(root_.Get());
            commands_->SetGraphicsRootShaderResourceView(1U, physics_->GetGPUVirtualAddress() + physics_offset_);
            commands_->SetGraphicsRootShaderResourceView(2U, physics_->GetGPUVirtualAddress() + physics_offset_);
            commands_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            DrawConstants values{};
            const auto forward = math::normalize_or(math::subtract(camera.target, camera.eye), {0,0,-1});
            const auto right = math::normalize_or(math::cross(forward, camera.up), {1,0,0});
            const auto up = math::cross(right, forward);
            std::memcpy(values.eye, &camera.eye, sizeof(Vec3));
            std::memcpy(values.right, &right, sizeof(Vec3));
            std::memcpy(values.up, &up, sizeof(Vec3));
            std::memcpy(values.forward, &forward, sizeof(Vec3));
            values.up[3] = std::tan(camera.vertical_field_of_view_degrees * 0.00872664626F);
            values.right[3] = values.up[3] * static_cast<float>(width_) / static_cast<float>(height_);
            values.settings[1] = rigid_interpolation_alpha;
            std::size_t constant_index{};
            const auto bind = [&] {
                std::memcpy(constant_data_ + constant_index * 256U, &values, sizeof(values));
                const auto address = constants_->GetGPUVirtualAddress() + constant_index++ * 256U;
                commands_->SetGraphicsRootConstantBufferView(0U, address);
                return address;
            };
            for (const auto &draw : draws_) {
                const auto &mesh = meshes_.at(draw.mesh);
                if (!mesh.visible || !mesh.count) continue;
                std::memcpy(values.color, &mesh.color, sizeof(Vec3));
                values.draw[0] = draw.mode;
                values.draw[1] = draw.mode == 0U
                    ? static_cast<std::uint32_t>(buffers_.rigid.offset) +
                        draw.body * sizeof(PmRigidState)
                    : draw.offset;
                values.draw[2] = 0U;
                values.draw[3] = draw.mode == 0U
                    ? static_cast<std::uint32_t>(buffers_.previous_rigid.offset) +
                        draw.body * sizeof(PmRigidState)
                    : 0U;
                bind();
                commands_->IASetVertexBuffers(0U, 1U, &mesh.vertex_view);
                commands_->IASetIndexBuffer(&mesh.index_view);
                commands_->DrawIndexedInstanced(mesh.count, 1U, 0U, 0, 0U);
            }
            if (buffers_.fluid_capacity) {
                values.color[0] = 0.05F; values.color[1] = 0.42F; values.color[2] = 0.92F;
                values.draw[0] = 2U;
                values.draw[1] = static_cast<std::uint32_t>(buffers_.fluid.offset);
                values.draw[2] = sizeof(PmFluidParticleOutput);
                values.settings[0] = particle_radius_;
                values.settings[3] = 1.0F;
                values.settings[2] = support_radius_;
                if (fluid_mode == FluidRenderMode::particles) {
                    bind();
                    const D3D12_VERTEX_BUFFER_VIEW vertices{particle_vertices_->GetGPUVirtualAddress(), 6U * sizeof(Vertex), sizeof(Vertex)};
                    commands_->IASetVertexBuffers(0U, 1U, &vertices);
                    commands_->DrawInstanced(6U, buffers_.fluid_capacity, 0U, 0U);
                } else {
                    values.draw[2] = surface_hash_size_;
                    values.draw[3] = surface_vertex_capacity;
                    const auto constants = bind();
                    commands_->SetComputeRootSignature(root_.Get());
                    commands_->SetComputeRootConstantBufferView(0U, constants);
                    commands_->SetComputeRootShaderResourceView(1U, physics_->GetGPUVirtualAddress() + physics_offset_);
                    const std::array buffers{surface_metadata_.Get(), surface_heads_.Get(), surface_next_.Get(), surface_field_.Get(), surface_vertices_.Get()};
                    for (UINT index = 0U; index < buffers.size(); ++index)
                        commands_->SetComputeRootUnorderedAccessView(index + 3U, buffers[index]->GetGPUVirtualAddress());
                    const std::array<UINT, 6U> groups{(surface_hash_size_+127U)/128U,
                        (buffers_.fluid_capacity+127U)/128U, 1U, surface_samples/128U, surface_samples/128U, 1U};
                    D3D12_RESOURCE_BARRIER uav{};
                    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    for (std::size_t stage = 0U; stage < groups.size(); ++stage) {
                        commands_->SetPipelineState(surface_pipelines_[stage].Get());
                        commands_->Dispatch(groups[stage], 1U, 1U);
                        commands_->ResourceBarrier(1U, &uav);
                    }
                    barrier(commands_.Get(), surface_metadata_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    commands_->CopyBufferRegion(surface_status_.Get(), 0U, surface_metadata_.Get(), 60U, sizeof(std::uint32_t));
                    barrier(commands_.Get(), surface_metadata_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                    barrier(commands_.Get(), surface_vertices_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    commands_->SetPipelineState(fluid_mode == FluidRenderMode::wireframe ? wireframe_.Get() : pipeline_.Get());
                    values.draw[0] = 3U;
                    bind();
                    commands_->SetGraphicsRootShaderResourceView(2U, surface_vertices_->GetGPUVirtualAddress());
                    // Constant zero-stride input; all positions come from GPU vertices.
                    const D3D12_VERTEX_BUFFER_VIEW vertices{particle_vertices_->GetGPUVirtualAddress(), sizeof(Vertex), 0U};
                    commands_->IASetVertexBuffers(0U, 1U, &vertices);
                    commands_->ExecuteIndirect(surface_draw_.Get(), 1U, surface_metadata_.Get(), 0U, nullptr, 0U);
                    barrier(commands_.Get(), surface_metadata_.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    barrier(commands_.Get(), surface_vertices_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
            }
            barrier(commands_.Get(), physics_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrier(commands_.Get(), target_.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
            check(commands_->Close(), "close drawing commands");
            ID3D12CommandList *lists[]{commands_.Get()};
            queue_->ExecuteCommandLists(1U, lists);
            wait();
            if (buffers_.fluid_capacity && fluid_mode != FluidRenderMode::particles) {
                void *status{};
                const D3D12_RANGE range{0U, sizeof(std::uint32_t)};
                check(surface_status_->Map(0U, &range, &status), "read GPU surface capacity status");
                const auto overflow = *static_cast<const std::uint32_t *>(status);
                surface_status_->Unmap(0U, nullptr);
                if (overflow) throw std::runtime_error("GPU fluid surface vertex capacity exceeded");
            }
            rgba.assign(static_cast<std::size_t>(width_) * height_, 0U); // common UI overlay only
            if (timings) {
                *timings = {};
                timings->particle_view = fluid_mode == FluidRenderMode::particles;
                timings->total_wall_milliseconds = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
            }
            return true;
        } catch (const std::exception &exception) { error = exception.what(); return false; }
    }

    NativeRenderImage native_image() const noexcept override {
        return {PM_NATIVE_BACKEND_D3D12, static_cast<PmNativeHandle>(reinterpret_cast<std::uintptr_t>(image_handle_))};
    }
    bool read_image(std::vector<std::uint32_t> &rgba, std::string &error) override {
        try {
            wait();
            const auto description = target_->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            UINT64 size{};
            device_->GetCopyableFootprints(&description, 0U, 1U, 0U, &footprint, nullptr, nullptr, &size);
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = size; buffer.Height = 1U;
            buffer.DepthOrArraySize = buffer.MipLevels = 1U;
            buffer.SampleDesc.Count = 1U;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            ComPtr<ID3D12Resource> readback;
            check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)), "create capture readback");
            check(allocator_->Reset(), "reset capture allocator");
            check(commands_->Reset(allocator_.Get(), nullptr), "reset capture commands");
            barrier(commands_.Get(), target_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
            source.pResource = target_.Get();
            source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            destination.pResource = readback.Get();
            destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            destination.PlacedFootprint = footprint;
            commands_->CopyTextureRegion(&destination, 0U, 0U, 0U, &source, nullptr);
            barrier(commands_.Get(), target_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            check(commands_->Close(), "close capture commands");
            ID3D12CommandList *lists[]{commands_.Get()};
            queue_->ExecuteCommandLists(1U, lists);
            wait();
            void *mapped{};
            const D3D12_RANGE range{0U, static_cast<SIZE_T>(size)};
            check(readback->Map(0U, &range, &mapped), "map captured image");
            rgba.resize(static_cast<std::size_t>(width_) * height_);
            for (std::uint32_t row = 0U; row < height_; ++row)
                std::memcpy(rgba.data() + static_cast<std::size_t>(height_ - 1U - row) * width_,
                    static_cast<const std::byte *>(mapped) + footprint.Offset +
                        static_cast<std::size_t>(row) * footprint.Footprint.RowPitch,
                    width_ * sizeof(std::uint32_t));
            readback->Unmap(0U, nullptr);
            return true;
        } catch (const std::exception &exception) { error = exception.what(); return false; }
    }
    bool advance_visuals(const RenderSceneFrame &, std::string &) override { return true; }
    bool paint_coverage(std::uint64_t &texels, std::string &) const override { texels = 0U; return true; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
};
} // namespace

std::unique_ptr<Renderer> create_d3d12_renderer(const SceneDefinition &scene,
    const GpuSceneBuffers &buffers, std::uint32_t width, std::uint32_t height, std::string &error) {
    try {
        auto renderer = std::make_unique<D3D12Renderer>(scene, buffers, width, height);
        renderer->initialize(scene);
        return renderer;
    } catch (const std::exception &exception) { error = exception.what(); return {}; }
}
} // namespace parallel_mater::gallery
