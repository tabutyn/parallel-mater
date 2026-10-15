// SPDX-License-Identifier: MIT
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string_view>
#include <vector>

namespace parallel_mater::d3d12 {

inline constexpr std::uint32_t invalid_descriptor =
    std::numeric_limits<std::uint32_t>::max();

// This is the host half of PMGraphicsBuffer<T>. Element offsets remain in the
// schedule constants; the native adapter supplies only descriptor ownership.
struct GraphicsBufferAddress {
    std::uint32_t descriptor_index{invalid_descriptor};
    std::uint32_t byte_offset{};
    std::uint32_t available{};
};
static_assert(sizeof(GraphicsBufferAddress) == 12U);

struct DispatchConstants {
    std::uint32_t group_count_x{};
    std::uint32_t group_count_y{};
    std::uint32_t group_count_z{};
    std::uint32_t group_size_x{};
};
static_assert(sizeof(DispatchConstants) == 16U);

class SlangCommandPath {
public:
    HRESULT initialize(ID3D12Device *device,
                       std::uint32_t descriptor_capacity = 65536U) noexcept;

    HRESULT create_raw_buffer_view(ID3D12Resource *resource,
                                   std::uint64_t byte_offset,
                                   std::uint64_t byte_size,
                                   std::uint32_t *descriptor) noexcept;
    void release_descriptor(std::uint32_t descriptor) noexcept;

    HRESULT record_dispatch(ID3D12GraphicsCommandList *commands,
                            std::string_view module,
                            std::string_view entry,
                            D3D12_GPU_VIRTUAL_ADDRESS global_constants,
                            D3D12_GPU_VIRTUAL_ADDRESS entry_constants,
                            std::uint32_t group_count_x,
                            std::uint32_t group_count_y = 1U,
                            std::uint32_t group_count_z = 1U) noexcept;

    [[nodiscard]] bool contains(std::string_view module,
                                std::string_view entry) const noexcept;
    [[nodiscard]] std::size_t kernel_count() const noexcept;

private:
    HRESULT pipeline(std::size_t index,
                     ID3D12PipelineState **output) noexcept;
    [[nodiscard]] std::size_t find(std::string_view module,
                                   std::string_view entry) const noexcept;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptors_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12PipelineState>> pipelines_;
    std::vector<std::uint32_t> free_descriptors_;
    std::mutex pipeline_mutex_;
    std::mutex descriptor_mutex_;
    std::uint32_t descriptor_capacity_{};
    std::uint32_t next_descriptor_{};
    std::uint32_t descriptor_stride_{};
};

} // namespace parallel_mater::d3d12
