// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/window.hpp>
#include <parallel_mater_gallery/gpu_renderer.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

namespace parallel_mater::gallery {
namespace {

using Microsoft::WRL::ComPtr;

std::string hresult_error(const char *operation, HRESULT result) {
    std::ostringstream stream;
    stream << operation << " failed (HRESULT 0x" << std::hex
           << static_cast<unsigned long>(result) << ')';
    return stream.str();
}

} // namespace

struct GalleryWindow::Impl {
    GLFWwindow *window{};
    ComPtr<IDXGIFactory6> factory{};
    ComPtr<ID3D12Device> device{};
    ComPtr<ID3D12CommandQueue> queue{};
    ComPtr<IDXGISwapChain3> swap_chain{};
    ComPtr<ID3D12CommandAllocator> allocator{};
    ComPtr<ID3D12GraphicsCommandList> command_list{};
    ComPtr<ID3D12Resource> upload{};
    ComPtr<ID3D12Fence> fence{};
    HANDLE fence_event{};
    std::uint64_t fence_value{};
    std::uint32_t image_width{};
    std::uint32_t image_height{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT upload_footprint{};
    bool vsync{true};
    bool allow_tearing{};

    ~Impl() {
        wait();
        if (fence_event != nullptr) CloseHandle(fence_event);
        if (window != nullptr) glfwDestroyWindow(window);
        glfwTerminate();
    }

    bool wait() noexcept {
        if (!queue || !fence) return true;
        const std::uint64_t next = ++fence_value;
        if (FAILED(queue->Signal(fence.Get(), next))) return false;
        if (fence->GetCompletedValue() >= next) return true;
        if (FAILED(fence->SetEventOnCompletion(next, fence_event)))
            return false;
        return WaitForSingleObject(fence_event, INFINITE) == WAIT_OBJECT_0;
    }
};

GalleryWindow::GalleryWindow() noexcept = default;
GalleryWindow::~GalleryWindow() = default;
GalleryWindow::GalleryWindow(GalleryWindow &&) noexcept = default;
GalleryWindow &GalleryWindow::operator=(GalleryWindow &&) noexcept = default;

bool GalleryWindow::create(std::uint32_t window_width,
                           std::uint32_t window_height,
                           std::uint32_t image_width,
                           std::uint32_t image_height,
                           std::string_view title, GalleryWindow &output,
                           std::string &error,
                           std::optional<PmNativeBackend>) {
    error.clear();
    if (image_width == 0U || image_height == 0U) {
        error = "presentation image dimensions must be nonzero";
        return false;
    }
    if (glfwInit() == GLFW_FALSE) {
        error = "GLFW initialization failed";
        return false;
    }
    auto implementation = std::make_unique<Impl>();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    const std::string window_title(title);
    implementation->window = glfwCreateWindow(
        static_cast<int>(window_width), static_cast<int>(window_height),
        window_title.c_str(), nullptr, nullptr);
    if (implementation->window == nullptr) {
        error = "GLFW window creation failed";
        return false;
    }
    implementation->image_width = image_width;
    implementation->image_height = image_height;

    HRESULT result = CreateDXGIFactory2(
        0U, IID_PPV_ARGS(&implementation->factory));
    if (FAILED(result)) {
        error = hresult_error("CreateDXGIFactory2", result);
        return false;
    }

    for (UINT index = 0U;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        result = implementation->factory->EnumAdapterByGpuPreference(
            index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter));
        if (result == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(result)) continue;
        DXGI_ADAPTER_DESC1 description{};
        adapter->GetDesc1(&description);
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0U) continue;
        result = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&implementation->device));
        if (SUCCEEDED(result)) break;
    }
    if (!implementation->device) {
        ComPtr<IDXGIAdapter> warp;
        result = implementation->factory->EnumWarpAdapter(
            IID_PPV_ARGS(&warp));
        if (SUCCEEDED(result))
            result = D3D12CreateDevice(
                warp.Get(), D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&implementation->device));
    }
    if (!implementation->device) {
        error = hresult_error("D3D12CreateDevice", result);
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    result = implementation->device->CreateCommandQueue(
        &queue_description, IID_PPV_ARGS(&implementation->queue));
    if (FAILED(result)) {
        error = hresult_error("CreateCommandQueue", result);
        return false;
    }

    DXGI_SWAP_CHAIN_DESC1 swap_description{};
    swap_description.Width = image_width;
    swap_description.Height = image_height;
    swap_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swap_description.SampleDesc.Count = 1U;
    swap_description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap_description.BufferCount = 2U;
    swap_description.Scaling = DXGI_SCALING_STRETCH;
    swap_description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swap_description.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    BOOL allow_tearing = FALSE;
    if (SUCCEEDED(implementation->factory->CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing,
            sizeof(allow_tearing))) && allow_tearing == TRUE) {
        implementation->allow_tearing = true;
        swap_description.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    }
    ComPtr<IDXGISwapChain1> swap_chain;
    const HWND window_handle = glfwGetWin32Window(implementation->window);
    result = implementation->factory->CreateSwapChainForHwnd(
        implementation->queue.Get(), window_handle, &swap_description,
        nullptr, nullptr, &swap_chain);
    if (FAILED(result)) {
        error = hresult_error("CreateSwapChainForHwnd", result);
        return false;
    }
    implementation->factory->MakeWindowAssociation(
        window_handle, DXGI_MWA_NO_ALT_ENTER);
    result = swap_chain.As(&implementation->swap_chain);
    if (FAILED(result)) {
        error = hresult_error("IDXGISwapChain3", result);
        return false;
    }

    result = implementation->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&implementation->allocator));
    if (FAILED(result)) {
        error = hresult_error("CreateCommandAllocator", result);
        return false;
    }
    result = implementation->device->CreateCommandList(
        0U, D3D12_COMMAND_LIST_TYPE_DIRECT, implementation->allocator.Get(),
        nullptr, IID_PPV_ARGS(&implementation->command_list));
    if (FAILED(result)) {
        error = hresult_error("CreateCommandList", result);
        return false;
    }
    implementation->command_list->Close();

    D3D12_RESOURCE_DESC image_description{};
    image_description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image_description.Width = image_width;
    image_description.Height = image_height;
    image_description.DepthOrArraySize = 1U;
    image_description.MipLevels = 1U;
    image_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    image_description.SampleDesc.Count = 1U;
    image_description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    UINT64 upload_size = 0U;
    implementation->device->GetCopyableFootprints(
        &image_description, 0U, 1U, 0U,
        &implementation->upload_footprint, nullptr, nullptr, &upload_size);
    D3D12_HEAP_PROPERTIES upload_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC upload_description{};
    upload_description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    upload_description.Width = upload_size;
    upload_description.Height = 1U;
    upload_description.DepthOrArraySize = 1U;
    upload_description.MipLevels = 1U;
    upload_description.SampleDesc.Count = 1U;
    upload_description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result = implementation->device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &upload_description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&implementation->upload));
    if (FAILED(result)) {
        error = hresult_error("CreateCommittedResource(upload)", result);
        return false;
    }

    result = implementation->device->CreateFence(
        0U, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&implementation->fence));
    if (FAILED(result)) {
        error = hresult_error("CreateFence", result);
        return false;
    }
    implementation->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (implementation->fence_event == nullptr) {
        error = "CreateEventW failed for the D3D12 presentation fence";
        return false;
    }

    output.impl_ = std::move(implementation);
    return true;
}

bool GalleryWindow::present(const std::vector<std::uint32_t> &rgba,
                            std::string &error, NativeRenderImage image) {
    error.clear();
    if (!impl_ || rgba.size() !=
            static_cast<std::size_t>(impl_->image_width) *
                impl_->image_height) {
        error = "presentation image size does not match the window adapter";
        return false;
    }
    ComPtr<ID3D12Resource> gpu_image;
    if (image.handle != 0U) {
        if (image.backend != PM_NATIVE_BACKEND_D3D12) {
            error = "D3D12 presentation received an incompatible native image";
            return false;
        }
        const auto opened = impl_->device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(image.handle)),
            IID_PPV_ARGS(&gpu_image));
        if (FAILED(opened)) { error = hresult_error("OpenSharedHandle(render image)", opened); return false; }
        const auto description = gpu_image->GetDesc();
        if (description.Width != impl_->image_width || description.Height != impl_->image_height ||
            description.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
            error = "native render image does not match the presentation target";
            return false;
        }
    }

    std::byte *mapped = nullptr;
    D3D12_RANGE no_read{0U, 0U};
    HRESULT result = impl_->upload->Map(
        0U, &no_read, reinterpret_cast<void **>(&mapped));
    if (FAILED(result)) {
        error = hresult_error("Map(upload)", result);
        return false;
    }
    mapped += impl_->upload_footprint.Offset;
    const std::size_t source_pitch =
        static_cast<std::size_t>(impl_->image_width) * sizeof(std::uint32_t);
    for (std::uint32_t row = 0U; row < impl_->image_height; ++row) {
        auto *target = mapped + static_cast<std::size_t>(row) *
            impl_->upload_footprint.Footprint.RowPitch;
        const auto *pixels = rgba.data() + static_cast<std::size_t>(
            impl_->image_height - 1U - row) * impl_->image_width;
        if (!gpu_image) {
            std::memcpy(target, pixels, source_pitch);
            continue;
        }
        // Upload only UI spans. The transparent canvas is not image data and
        // should consume neither upload bandwidth nor a full-image GPU copy.
        for (UINT column = 0U; column < impl_->image_width;) {
            if (pixels[column] == 0U) { ++column; continue; }
            const UINT first = column++;
            while (column < impl_->image_width && pixels[column] != 0U) ++column;
            std::memcpy(target + first * sizeof(std::uint32_t), pixels + first,
                (column - first) * sizeof(std::uint32_t));
        }
    }
    impl_->upload->Unmap(0U, nullptr);

    result = impl_->allocator->Reset();
    if (SUCCEEDED(result))
        result = impl_->command_list->Reset(impl_->allocator.Get(), nullptr);
    if (FAILED(result)) {
        error = hresult_error("Reset(command list)", result);
        return false;
    }
    ComPtr<ID3D12Resource> back_buffer;
    result = impl_->swap_chain->GetBuffer(
        impl_->swap_chain->GetCurrentBackBufferIndex(),
        IID_PPV_ARGS(&back_buffer));
    if (FAILED(result)) {
        error = hresult_error("GetBuffer", result);
        return false;
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffer.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    impl_->command_list->ResourceBarrier(1U, &barrier);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = back_buffer.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.SubresourceIndex = 0U;
    D3D12_TEXTURE_COPY_LOCATION source_location{};
    source_location.pResource = impl_->upload.Get();
    source_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source_location.PlacedFootprint = impl_->upload_footprint;
    if (!gpu_image) {
        impl_->command_list->CopyTextureRegion(
            &destination, 0U, 0U, 0U, &source_location, nullptr);
    } else {
        D3D12_RESOURCE_BARRIER source_barrier{};
        source_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        source_barrier.Transition = {gpu_image.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE};
        impl_->command_list->ResourceBarrier(1U, &source_barrier);
        impl_->command_list->CopyResource(back_buffer.Get(), gpu_image.Get());
        std::swap(source_barrier.Transition.StateBefore, source_barrier.Transition.StateAfter);
        impl_->command_list->ResourceBarrier(1U, &source_barrier);
        // Shared UI code emits opaque pixels over a transparent canvas. Copy
        // only its occupied row spans; the scene image never visits the CPU.
        for (UINT row = 0U; row < impl_->image_height; ++row) {
            const auto *pixels = rgba.data() +
                static_cast<std::size_t>(impl_->image_height - 1U - row) * impl_->image_width;
            for (UINT column = 0U; column < impl_->image_width;) {
                if (pixels[column] == 0U) { ++column; continue; }
                const UINT first = column++;
                while (column < impl_->image_width && pixels[column] != 0U) ++column;
                const D3D12_BOX box{first, row, 0U, column, row + 1U, 1U};
                impl_->command_list->CopyTextureRegion(&destination, first, row, 0U, &source_location, &box);
            }
        }
    }
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    impl_->command_list->ResourceBarrier(1U, &barrier);
    result = impl_->command_list->Close();
    if (FAILED(result)) {
        error = hresult_error("Close(command list)", result);
        return false;
    }
    ID3D12CommandList *lists[]{impl_->command_list.Get()};
    impl_->queue->ExecuteCommandLists(1U, lists);
    const UINT present_flags = !impl_->vsync && impl_->allow_tearing
        ? DXGI_PRESENT_ALLOW_TEARING : 0U;
    result = impl_->swap_chain->Present(
        impl_->vsync ? 1U : 0U, present_flags);
    if (FAILED(result)) {
        error = hresult_error("Present", result);
        return false;
    }
    if (!impl_->wait()) {
        error = "D3D12 presentation fence wait failed";
        return false;
    }
    return true;
}

void GalleryWindow::set_vsync(bool enabled) noexcept {
    if (impl_) impl_->vsync = enabled;
}

GLFWwindow *GalleryWindow::native_handle() const noexcept {
    return impl_ ? impl_->window : nullptr;
}

GpuRenderTarget GalleryWindow::native_render_target(
    const PmNativeDeviceInfo &, std::uint32_t, std::uint32_t,
    std::string &error) {
    error = "D3D12 Gallery does not use the native GPU renderer contract";
    return {};
}

} // namespace parallel_mater::gallery
