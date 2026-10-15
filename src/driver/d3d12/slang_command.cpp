// SPDX-License-Identifier: MIT
#include "slang_command.hpp"

#include "parallel_mater_slang_dxil.hpp"

#include <algorithm>
#include <new>

namespace parallel_mater::d3d12 {
namespace {

constexpr std::size_t no_kernel = static_cast<std::size_t>(-1);

} // namespace

HRESULT SlangCommandPath::initialize(
    ID3D12Device *device, std::uint32_t descriptor_capacity) noexcept {
    if (device == nullptr || descriptor_capacity == 0U) return E_INVALIDARG;

    D3D12_FEATURE_DATA_ROOT_SIGNATURE root_support{
        D3D_ROOT_SIGNATURE_VERSION_1_1};
    if (FAILED(device->CheckFeatureSupport(
            D3D12_FEATURE_ROOT_SIGNATURE, &root_support,
            sizeof(root_support))) ||
        root_support.HighestVersion < D3D_ROOT_SIGNATURE_VERSION_1_1)
        return E_NOINTERFACE;

    D3D12_DESCRIPTOR_HEAP_DESC heap_description{};
    heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_description.NumDescriptors = descriptor_capacity;
    heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    HRESULT result = device->CreateDescriptorHeap(
        &heap_description, IID_PPV_ARGS(&descriptors_));
    if (FAILED(result)) return result;

    D3D12_ROOT_PARAMETER1 parameters[2]{};
    for (UINT index = 0U; index < 2U; ++index) {
        parameters[index].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[index].Descriptor.ShaderRegister = index;
        parameters[index].Descriptor.RegisterSpace = 0U;
        parameters[index].Descriptor.Flags =
            D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
        parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_description{};
    root_description.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_description.Desc_1_1.NumParameters = 2U;
    root_description.Desc_1_1.pParameters = parameters;
    root_description.Desc_1_1.Flags =
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    result = D3D12SerializeVersionedRootSignature(
        &root_description, &signature, &error);
    if (FAILED(result)) return result;
    result = device->CreateRootSignature(
        0U, signature->GetBufferPointer(), signature->GetBufferSize(),
        IID_PPV_ARGS(&root_));
    if (FAILED(result)) return result;

    try {
        pipelines_.resize(generated::slang_kernel_count);
        free_descriptors_.reserve(descriptor_capacity);
    } catch (const std::bad_alloc &) {
        return E_OUTOFMEMORY;
    }
    device_ = device;
    descriptor_capacity_ = descriptor_capacity;
    return S_OK;
}

HRESULT SlangCommandPath::create_raw_buffer_view(
    ID3D12Resource *resource, std::uint64_t byte_offset,
    std::uint64_t byte_size, std::uint32_t *descriptor) noexcept {
    if (resource == nullptr || descriptor == nullptr || byte_size < 4U ||
        byte_offset % 4U != 0U || byte_size % 4U != 0U ||
        byte_offset / 4U > std::numeric_limits<UINT>::max() ||
        byte_size / 4U > std::numeric_limits<UINT>::max())
        return E_INVALIDARG;
    *descriptor = invalid_descriptor;
    const D3D12_RESOURCE_DESC resource_description = resource->GetDesc();
    if (resource_description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        byte_offset > resource_description.Width ||
        byte_size > resource_description.Width - byte_offset)
        return E_INVALIDARG;

    std::lock_guard lock(descriptor_mutex_);
    std::uint32_t index = invalid_descriptor;
    if (!free_descriptors_.empty()) {
        index = free_descriptors_.back();
        free_descriptors_.pop_back();
    } else if (next_descriptor_ < descriptor_capacity_) {
        index = next_descriptor_++;
    } else {
        return E_OUTOFMEMORY;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    view.Buffer.FirstElement = byte_offset / 4U;
    view.Buffer.NumElements = static_cast<UINT>(byte_size / 4U);
    view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        descriptors_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) *
        device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device_->CreateUnorderedAccessView(resource, nullptr, &view, handle);
    *descriptor = index;
    return S_OK;
}

void SlangCommandPath::release_descriptor(std::uint32_t descriptor) noexcept {
    if (descriptor == invalid_descriptor ||
        descriptor >= descriptor_capacity_)
        return;
    std::lock_guard lock(descriptor_mutex_);
    free_descriptors_.push_back(descriptor);
}

std::size_t SlangCommandPath::find(
    std::string_view module, std::string_view entry) const noexcept {
    for (std::size_t index = 0U; index < generated::slang_kernel_count;
         ++index) {
        const auto &candidate = generated::slang_kernels[index];
        if (module == candidate.module && entry == candidate.entry)
            return index;
    }
    return no_kernel;
}

bool SlangCommandPath::contains(
    std::string_view module, std::string_view entry) const noexcept {
    return find(module, entry) != no_kernel;
}

std::size_t SlangCommandPath::kernel_count() const noexcept {
    return generated::slang_kernel_count;
}

HRESULT SlangCommandPath::pipeline(
    std::size_t index, ID3D12PipelineState **output) noexcept {
    if (output == nullptr || index >= pipelines_.size()) return E_INVALIDARG;
    std::lock_guard lock(pipeline_mutex_);
    if (!pipelines_[index]) {
        const auto &kernel = generated::slang_kernels[index];
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
        description.pRootSignature = root_.Get();
        description.CS = {kernel.code, kernel.size};
        const HRESULT result = device_->CreateComputePipelineState(
            &description, IID_PPV_ARGS(&pipelines_[index]));
        if (FAILED(result)) return result;
    }
    *output = pipelines_[index].Get();
    return S_OK;
}

HRESULT SlangCommandPath::record_dispatch(
    ID3D12GraphicsCommandList *commands, std::string_view module,
    std::string_view entry, D3D12_GPU_VIRTUAL_ADDRESS global_constants,
    D3D12_GPU_VIRTUAL_ADDRESS entry_constants,
    std::uint32_t group_count_x, std::uint32_t group_count_y,
    std::uint32_t group_count_z) noexcept {
    if (commands == nullptr || global_constants == 0U ||
        entry_constants == 0U || group_count_x == 0U ||
        group_count_y == 0U || group_count_z == 0U ||
        global_constants % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT != 0U ||
        entry_constants % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT != 0U)
        return E_INVALIDARG;
    const std::size_t index = find(module, entry);
    if (index == no_kernel) return E_INVALIDARG;
    ID3D12PipelineState *state = nullptr;
    const HRESULT result = pipeline(index, &state);
    if (FAILED(result)) return result;

    ID3D12DescriptorHeap *heaps[]{descriptors_.Get()};
    commands->SetDescriptorHeaps(1U, heaps);
    commands->SetComputeRootSignature(root_.Get());
    commands->SetComputeRootConstantBufferView(0U, global_constants);
    commands->SetComputeRootConstantBufferView(1U, entry_constants);
    commands->SetPipelineState(state);
    commands->Dispatch(group_count_x, group_count_y, group_count_z);
    return S_OK;
}

} // namespace parallel_mater::d3d12
