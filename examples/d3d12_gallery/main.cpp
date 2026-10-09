// SPDX-License-Identifier: MIT
#include <parallel_mater/d3d12.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/bitmap_font.hpp>
#include "ui.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "pm_d3d12_gallery_ps.h"
#include "pm_d3d12_gallery_vs.h"
#include "pm_d3d12_gallery_ui_vs.h"
#include "pm_d3d12_gallery_ui_ps.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace pm = parallel_mater::d3d12;
namespace scene = parallel_mater::d3d12::gallery;
namespace catalog = parallel_mater::gallery;
using Microsoft::WRL::ComPtr;

namespace {

constexpr auto rigid_scene_count = pm::viewer::rigid_scene_count;
using pm::viewer::InputState;
using pm::viewer::Key;
using pm::viewer::ArrowKey;

void install_input(GLFWwindow *window,InputState &input) {
    glfwSetWindowUserPointer(window,&input);
    glfwSetKeyCallback(window,[](GLFWwindow *w,int key,int,int action,int) {
        auto &state=*static_cast<InputState *>(glfwGetWindowUserPointer(w));
        const bool pressed=action!=GLFW_RELEASE;
        switch(key) {
        case GLFW_KEY_LEFT: state.arrow(ArrowKey::left,pressed); break;
        case GLFW_KEY_RIGHT: state.arrow(ArrowKey::right,pressed); break;
        case GLFW_KEY_UP: state.arrow(ArrowKey::up,pressed); break;
        case GLFW_KEY_DOWN: state.arrow(ArrowKey::down,pressed); break;
        default: break;
        }
        if(action!=GLFW_PRESS && action!=GLFW_REPEAT) return;
        if(action==GLFW_REPEAT && key!=GLFW_KEY_DOWN && key!=GLFW_KEY_UP) return;
        switch(key) {
        case GLFW_KEY_F: state.key(Key::fps); break;
        case GLFW_KEY_TAB: state.key(Key::catalog); break;
        case GLFW_KEY_ESCAPE: state.key(Key::escape); break;
        case GLFW_KEY_DOWN: state.key(Key::next); break;
        case GLFW_KEY_UP: state.key(Key::previous); break;
        case GLFW_KEY_ENTER: case GLFW_KEY_KP_ENTER: state.key(Key::enter); break;
        case GLFW_KEY_R: state.key(Key::reset); break;
        case GLFW_KEY_P: state.key(Key::pause); break;
        case GLFW_KEY_SPACE: state.key(Key::action); break;
        }
    });
    glfwSetMouseButtonCallback(window,[](GLFWwindow *w,int button,int action,int mods) {
        auto &state=*static_cast<InputState *>(glfwGetWindowUserPointer(w));
        if(action==GLFW_RELEASE) { state.camera.end_drag(); return; }
        if(button!=GLFW_MOUSE_BUTTON_LEFT && button!=GLFW_MOUSE_BUTTON_RIGHT) return;
        double x{},y{}; glfwGetCursorPos(w,&x,&y);
        if(state.catalog_visible) {
            if(button!=GLFW_MOUSE_BUTTON_LEFT) return;
            int width{},height{}; glfwGetWindowSize(w,&width,&height);
            const int scale=width>=1000 && height>=600 ? 2 : 1;
            const int first_y=scale*32+62,spacing=scale*10+6;
            const int column_width=(width-32)/2;
            const int column=static_cast<int>((x-16)/column_width);
            const int row=static_cast<int>((y-first_y+4)/spacing);
            if(x<16 || y<first_y-4 || column<0 || column>1 || row<0 || row>=14) return;
            const auto selected=static_cast<std::size_t>(column*14+row);
            if(selected>=catalog::gallery_entries.size()) return;
            state.selected=selected;
            state.key(Key::enter);
            return;
        }
        state.camera.begin_drag(button==GLFW_MOUSE_BUTTON_RIGHT || (mods&GLFW_MOD_SHIFT)
            ?catalog::CameraDragMode::pan:catalog::CameraDragMode::orbit,x,y);
    });
    glfwSetCursorPosCallback(window,[](GLFWwindow *w,double x,double y) {
        auto &state=*static_cast<InputState *>(glfwGetWindowUserPointer(w));
        int width{},height{}; glfwGetWindowSize(w,&width,&height);
        state.camera.move_cursor(x,y,height,!state.catalog_visible);
    });
    glfwSetScrollCallback(window,[](GLFWwindow *w,double,double y) {
        auto &state=*static_cast<InputState *>(glfwGetWindowUserPointer(w));
        if(!state.catalog_visible) state.camera.zoom(y);
    });
    glfwSetWindowFocusCallback(window,[](GLFWwindow *w,int focused) {
        if(!focused) {
            auto &state=*static_cast<InputState *>(glfwGetWindowUserPointer(w));
            state.camera.end_drag();
            state.clear_arrows();
        }
    });
}

struct Options {
    bool list_adapters{};
    bool list_scenes{};
    bool warp{};
    bool headless{};
    bool validate{};
    bool gpu_validation{};
    bool all_rigid_scenes{};
    bool all_scenes{};
    bool show_fps{};
    bool show_catalog{};
    bool frames_set{};
    bool benchmark{};
    std::uint32_t adapter{};
    bool adapter_set{};
    std::uint32_t frames{240U};
    std::uint32_t substeps{4U};
    std::uint32_t width{1280U};
    std::uint32_t height{720U};
    std::uint32_t dump_spheres{scene::default_dump_payload_count};
    std::uint32_t repeat{};
    std::size_t scene_index{};
    std::filesystem::path output{};
};

[[nodiscard]] bool parse_u32(const char *text, std::uint32_t &output) {
    if (text == nullptr || *text == '\0') return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (*end != '\0' || value > std::numeric_limits<std::uint32_t>::max())
        return false;
    output = static_cast<std::uint32_t>(value);
    return true;
}

void usage() {
    std::cout <<
        "parallel-mater-d3d12-gallery [options]\n"
        "  --list-adapters          print D3D12 adapters and capability\n"
        "  --adapter N              select hardware adapter index\n"
        "  --warp                   use Microsoft WARP\n"
        "  --list-scenes            print the shared gallery catalog\n"
        "  --all-rigid-scenes       run all supported rigid scenes\n"
        "  --all-scenes             reserved until all subsystem gates land\n"
        "  --headless               render without a window\n"
        "  --output PATH            PPM file or all-scenes directory\n"
        "  --frames N               frame limit (headless default 240)\n"
        "  --show-fps --show-catalog show overlays at startup\n"
        "  --benchmark              report average frame and physics times\n"
        "  --substeps N             physics substeps per frame (default 4)\n"
        "  --repeat N               deterministic repetitions\n"
        "  --width N --height N     render dimensions\n"
        "  --validate               enable D3D12 validation and checks\n"
        "  --gpu-validation         add slow GPU-based validation\n"
        "  --dump-spheres N         DUMP payload count (10..1000)\n"
        "Keys: F FPS, Tab catalog, arrows/Enter select, R reset, P pause,\n"
        "      Space runs scene action, Esc closes catalog/quit. Drag orbit, Shift-drag pan, wheel zoom.\n";
}

[[nodiscard]] bool parse_options(int argc, char **argv, Options &result) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        auto number = [&](std::uint32_t &value) {
            return i + 1 < argc && parse_u32(argv[++i], value);
        };
        if (argument == "--help" || argument == "-h") { usage(); return false; }
        if (argument == "--list-adapters") result.list_adapters = true;
        else if (argument == "--list-scenes") result.list_scenes = true;
        else if (argument == "--warp") result.warp = true;
        else if (argument == "--headless") result.headless = true;
        else if (argument == "--show-fps") result.show_fps = true;
        else if (argument == "--show-catalog") result.show_catalog = true;
        else if (argument == "--benchmark") result.benchmark = true;
        else if (argument == "--validate") result.validate = true;
        else if (argument == "--gpu-validation") {
            result.validate = true;
            result.gpu_validation = true;
        }
        else if (argument == "--all-rigid-scenes") result.all_rigid_scenes = true;
        else if (argument == "--all-scenes") result.all_scenes = true;
        else if (argument == "--adapter") {
            if (!number(result.adapter)) return false;
            result.adapter_set = true;
        } else if (argument == "--frames") {
            if (!number(result.frames) || result.frames == 0U) return false;
            result.frames_set = true;
        } else if (argument == "--substeps") {
            if (!number(result.substeps) || result.substeps == 0U) return false;
        } else if (argument == "--width") {
            if (!number(result.width) || result.width == 0U) return false;
        } else if (argument == "--height") {
            if (!number(result.height) || result.height == 0U) return false;
        } else if (argument == "--dump-spheres") {
            if (!number(result.dump_spheres) || result.dump_spheres < 10U ||
                result.dump_spheres > 1000U) return false;
        } else if (argument == "--repeat") {
            if (!number(result.repeat) || result.repeat == 0U) return false;
        } else if (argument == "--output") {
            if (i + 1 >= argc) return false;
            result.output = argv[++i];
        } else {
            bool found = false;
            for (std::size_t entry = 0; entry < rigid_scene_count; ++entry) {
                const auto &candidate = catalog::gallery_entries[entry];
                if (!candidate.command_line_option.empty() &&
                    argument == candidate.command_line_option) {
                    result.scene_index = entry;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Unknown option: " << argument << '\n';
                return false;
            }
        }
    }
    if (result.warp && result.adapter_set) {
        std::cerr << "--warp and --adapter are mutually exclusive\n";
        return false;
    }
    return true;
}

[[nodiscard]] std::string status_error(pm::Status status,
                                       const char *fallback) {
    std::ostringstream stream;
    stream << (status.message != nullptr ? status.message : fallback);
    if (status.hresult != 0)
        stream << " (HRESULT 0x" << std::hex << std::uppercase
               << static_cast<std::uint32_t>(status.hresult) << ')';
    return stream.str();
}

struct AdapterInfo {
    ComPtr<IDXGIAdapter1> adapter{};
    DXGI_ADAPTER_DESC1 description{};
    D3D_FEATURE_LEVEL feature_level{D3D_FEATURE_LEVEL_11_0};
    D3D12_RESOURCE_BINDING_TIER binding_tier{D3D12_RESOURCE_BINDING_TIER_1};
    D3D_SHADER_MODEL shader_model{D3D_SHADER_MODEL_5_1};
    bool device_created{};
    bool qualifies{};
};

[[nodiscard]] const char *feature_level_name(D3D_FEATURE_LEVEL level) {
    switch (level) {
    case D3D_FEATURE_LEVEL_12_1: return "12_1";
    case D3D_FEATURE_LEVEL_12_0: return "12_0";
    case D3D_FEATURE_LEVEL_11_1: return "11_1";
    default: return "11_0";
    }
}

[[nodiscard]] AdapterInfo inspect_adapter(ComPtr<IDXGIAdapter1> adapter) {
    AdapterInfo result{};
    result.adapter = std::move(adapter);
    result.adapter->GetDesc1(&result.description);
    constexpr D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0};
    ComPtr<ID3D12Device> device;
    for (D3D_FEATURE_LEVEL level : levels) {
        if (SUCCEEDED(D3D12CreateDevice(result.adapter.Get(), level,
                IID_PPV_ARGS(device.ReleaseAndGetAddressOf())))) {
            result.feature_level = level;
            result.device_created = true;
            break;
        }
    }
    if (!device) return result;
    D3D12_FEATURE_DATA_D3D12_OPTIONS binding{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
            &binding, sizeof(binding))))
        result.binding_tier = binding.ResourceBindingTier;
    D3D12_FEATURE_DATA_SHADER_MODEL shader{D3D_SHADER_MODEL_5_1};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,
            &shader, sizeof(shader))))
        result.shader_model = shader.HighestShaderModel;
    result.qualifies = result.shader_model >= D3D_SHADER_MODEL_5_1 &&
        (result.binding_tier >= D3D12_RESOURCE_BINDING_TIER_2 ||
         result.feature_level >= D3D_FEATURE_LEVEL_11_1);
    return result;
}

[[nodiscard]] std::vector<AdapterInfo> enumerate_adapters(IDXGIFactory6 *factory) {
    std::vector<AdapterInfo> result;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT hr = factory->EnumAdapters1(index,
            adapter.ReleaseAndGetAddressOf());
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (SUCCEEDED(hr)) result.push_back(inspect_adapter(std::move(adapter)));
    }
    return result;
}

[[nodiscard]] std::string narrow(const wchar_t *text) {
    if (text == nullptr) return {};
    const int count = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0,
                                          nullptr, nullptr);
    std::string result(count > 0 ? static_cast<std::size_t>(count) : 0U, '\0');
    if (count > 1) {
        WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), count,
                            nullptr, nullptr);
        result.pop_back();
    }
    return result;
}

struct GraphicsContext {
    ComPtr<IDXGIFactory6> factory{};
    AdapterInfo adapter{};
    ComPtr<ID3D12Device> device{};
    ComPtr<ID3D12CommandQueue> queue{};
    ComPtr<ID3D12InfoQueue> info_queue{};
};

[[nodiscard]] bool debug_queue_clean(GraphicsContext &graphics,
                                     std::string &error) {
    if (!graphics.info_queue) return true;
    const UINT64 count =
        graphics.info_queue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 index = 0; index < count; ++index) {
        SIZE_T bytes = 0U;
        if (FAILED(graphics.info_queue->GetMessage(index, nullptr, &bytes)) ||
            bytes < sizeof(D3D12_MESSAGE))
            continue;
        std::vector<std::byte> storage(bytes);
        auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        if (FAILED(graphics.info_queue->GetMessage(index, message, &bytes)))
            continue;
        if (message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION &&
            message->Severity != D3D12_MESSAGE_SEVERITY_ERROR)
            continue;
        error = "D3D12 validation error";
        if (message->pDescription != nullptr) {
            error += ": ";
            error += message->pDescription;
        }
        return false;
    }
    graphics.info_queue->ClearStoredMessages();
    return true;
}

[[nodiscard]] bool create_graphics_context(const Options &options,
                                           GraphicsContext &output,
                                           std::string &error) {
    UINT flags = 0U;
    if (options.validate) {
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred_settings;
        if (SUCCEEDED(D3D12GetDebugInterface(
                IID_PPV_ARGS(dred_settings.ReleaseAndGetAddressOf())))) {
            dred_settings->SetAutoBreadcrumbsEnablement(
                D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred_settings->SetPageFaultEnablement(
                D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(
                IID_PPV_ARGS(debug.ReleaseAndGetAddressOf())))) {
            error = "D3D12 debug layer is unavailable";
            return false;
        }
        debug->EnableDebugLayer();
        flags |= DXGI_CREATE_FACTORY_DEBUG;
        ComPtr<ID3D12Debug1> debug1;
        if (options.gpu_validation && SUCCEEDED(debug.As(&debug1)))
            debug1->SetEnableGPUBasedValidation(TRUE);
    }
    HRESULT hr = CreateDXGIFactory2(flags,
        IID_PPV_ARGS(output.factory.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) { error = "CreateDXGIFactory2 failed"; return false; }
    if (options.warp) {
        ComPtr<IDXGIAdapter> warp;
        hr = output.factory->EnumWarpAdapter(
            IID_PPV_ARGS(warp.ReleaseAndGetAddressOf()));
        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(hr) || FAILED(warp.As(&adapter))) {
            error = "WARP adapter is unavailable"; return false;
        }
        output.adapter = inspect_adapter(std::move(adapter));
    } else {
        std::vector<AdapterInfo> adapters = enumerate_adapters(output.factory.Get());
        if (options.adapter_set) {
            if (options.adapter >= adapters.size()) {
                error = "Adapter index is out of range"; return false;
            }
            output.adapter = std::move(adapters[options.adapter]);
        } else {
            auto found = std::find_if(adapters.begin(), adapters.end(),
                [](const AdapterInfo &item) {
                    return item.qualifies &&
                        (item.description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0U;
                });
            if (found == adapters.end()) {
                error = "No hardware adapter has SM5.1 and 64 UAV capability";
                return false;
            }
            output.adapter = std::move(*found);
        }
    }
    if (!output.adapter.qualifies) {
        error = "Selected adapter lacks SM5.1 or 64 UAV capability";
        return false;
    }
    if (!options.warp &&
        (output.adapter.description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0U) {
        error = "Software adapters require the explicit --warp option";
        return false;
    }
    hr = D3D12CreateDevice(output.adapter.adapter.Get(),
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(output.device.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) { error = "D3D12CreateDevice failed"; return false; }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = output.device->CreateCommandQueue(&queue_desc,
        IID_PPV_ARGS(output.queue.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) { error = "CreateCommandQueue failed"; return false; }
    if (options.validate) {
        hr = output.device.As(&output.info_queue);
        if (FAILED(hr)) {
            error = "D3D12 info queue is unavailable";
            return false;
        }
        output.info_queue->ClearStoredMessages();
    }
    return true;
}

[[nodiscard]] std::string dred_diagnostics(ID3D12Device *device) {
    ComPtr<ID3D12DeviceRemovedExtendedData> dred;
    if (device == nullptr || FAILED(device->QueryInterface(
            IID_PPV_ARGS(dred.ReleaseAndGetAddressOf())))) return {};
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs{};
    if (FAILED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs))) return {};
    std::ostringstream stream;
    for (const D3D12_AUTO_BREADCRUMB_NODE *node = breadcrumbs.pHeadAutoBreadcrumbNode;
         node != nullptr; node = node->pNext) {
        const UINT completed = node->pLastBreadcrumbValue != nullptr
            ? *node->pLastBreadcrumbValue : 0U;
        std::string name = "unnamed command list";
        if (node->pCommandListDebugNameA != nullptr)
            name = node->pCommandListDebugNameA;
        else if (node->pCommandListDebugNameW != nullptr)
            name = narrow(node->pCommandListDebugNameW);
        stream << "DRED: "
               << name
               << ", completed " << completed << '/' << node->BreadcrumbCount;
        if (completed < node->BreadcrumbCount && node->pCommandHistory != nullptr)
            stream << ", next operation "
                   << static_cast<unsigned>(node->pCommandHistory[completed]);
        stream << '\n';
        if (node->pCommandHistory != nullptr) {
            stream << "DRED history:";
            for (UINT index = 0U; index < node->BreadcrumbCount; ++index)
                stream << ' ' << index << ':'
                       << static_cast<unsigned>(node->pCommandHistory[index]);
            stream << '\n';
        }
    }
    return stream.str();
}

struct Matrix4 { float value[16]{}; };

[[nodiscard]] Matrix4 identity() {
    Matrix4 result{};
    result.value[0]=result.value[5]=result.value[10]=result.value[15]=1.0F;
    return result;
}
[[nodiscard]] Matrix4 multiply(Matrix4 a, Matrix4 b) {
    Matrix4 result{};
    for (int row=0;row<4;++row) for(int column=0;column<4;++column)
        for(int inner=0;inner<4;++inner)
            result.value[row*4+column] +=
                a.value[row*4+inner]*b.value[inner*4+column];
    return result;
}
[[nodiscard]] parallel_mater::Vec3 subtract(parallel_mater::Vec3 a,
                                             parallel_mater::Vec3 b) {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
[[nodiscard]] float dot(parallel_mater::Vec3 a, parallel_mater::Vec3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
[[nodiscard]] parallel_mater::Vec3 cross(parallel_mater::Vec3 a,
                                         parallel_mater::Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
[[nodiscard]] parallel_mater::Vec3 normalize(parallel_mater::Vec3 v) {
    const float length=std::sqrt(std::max(dot(v,v),1.0e-12F));
    return {v.x/length,v.y/length,v.z/length};
}
[[nodiscard]] Matrix4 model_matrix(const parallel_mater::RigidBodyState &state) {
    const auto q=state.orientation;
    Matrix4 result=identity();
    result.value[0]=1-2*(q.y*q.y+q.z*q.z);
    result.value[1]=2*(q.x*q.y+q.z*q.w);
    result.value[2]=2*(q.x*q.z-q.y*q.w);
    result.value[4]=2*(q.x*q.y-q.z*q.w);
    result.value[5]=1-2*(q.x*q.x+q.z*q.z);
    result.value[6]=2*(q.y*q.z+q.x*q.w);
    result.value[8]=2*(q.x*q.z+q.y*q.w);
    result.value[9]=2*(q.y*q.z-q.x*q.w);
    result.value[10]=1-2*(q.x*q.x+q.y*q.y);
    result.value[12]=state.position.x;
    result.value[13]=state.position.y;
    result.value[14]=state.position.z;
    return result;
}
[[nodiscard]] Matrix4 view_matrix(parallel_mater::Vec3 eye,
                                   parallel_mater::Vec3 target) {
    const auto forward=normalize(subtract(target,eye));
    const auto right=normalize(cross({0,1,0},forward));
    const auto up=cross(forward,right);
    Matrix4 result=identity();
    result.value[0]=right.x; result.value[4]=right.y; result.value[8]=right.z;
    result.value[1]=up.x; result.value[5]=up.y; result.value[9]=up.z;
    result.value[2]=forward.x; result.value[6]=forward.y; result.value[10]=forward.z;
    result.value[12]=-dot(right,eye); result.value[13]=-dot(up,eye);
    result.value[14]=-dot(forward,eye);
    return result;
}
[[nodiscard]] Matrix4 projection_matrix(float aspect, float field_of_view) {
    constexpr float near_plane=0.05F, far_plane=1000.0F;
    const float y=1.0F/std::tan(field_of_view*3.14159265359F/360.0F);
    Matrix4 result{};
    result.value[0]=y/aspect; result.value[5]=y;
    result.value[10]=far_plane/(far_plane-near_plane);
    result.value[11]=1.0F;
    result.value[14]=-near_plane*far_plane/(far_plane-near_plane);
    return result;
}

struct DrawConstants {
    Matrix4 model_view_projection{};
    Matrix4 model{};
    float color[4]{};
    std::uint32_t checkerboard{};
    std::uint32_t padding[3]{};
};
static_assert(sizeof(DrawConstants)==40U*sizeof(std::uint32_t));

struct GeometryRange {
    std::uint32_t first_index{};
    std::uint32_t index_count{};
    std::int32_t base_vertex{};
};

class Renderer {
  public:
    ~Renderer() {
        wait();
        if (fence_event_) CloseHandle(fence_event_);
    }

    [[nodiscard]] bool create(GraphicsContext &graphics, GLFWwindow *window,
                              std::uint32_t width, std::uint32_t height,
                              bool headless, std::string &error) {
        device_=graphics.device; queue_=graphics.queue; factory_=graphics.factory;
        width_=width; height_=height; headless_=headless;
        HRESULT hr=device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(allocator_.ReleaseAndGetAddressOf()));
        if(FAILED(hr)) return set_error(error,"render command allocator");
        hr=device_->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator_.Get(),nullptr,IID_PPV_ARGS(commands_.ReleaseAndGetAddressOf()));
        if(FAILED(hr)) return set_error(error,"render command list");
        commands_->SetName(L"Parallel Mater gallery rendering");
        commands_->Close();
        hr=device_->CreateFence(0,D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(fence_.ReleaseAndGetAddressOf()));
        if(FAILED(hr)) return set_error(error,"render fence");
        fence_event_=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!fence_event_) return set_error(error,"render fence event");

        D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{};
        rtv_desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv_desc.NumDescriptors=headless?1U:2U;
        if(FAILED(device_->CreateDescriptorHeap(&rtv_desc,
            IID_PPV_ARGS(rtv_heap_.ReleaseAndGetAddressOf()))))
            return set_error(error,"RTV heap");
        D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{};
        dsv_desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; dsv_desc.NumDescriptors=1;
        if(FAILED(device_->CreateDescriptorHeap(&dsv_desc,
            IID_PPV_ARGS(dsv_heap_.ReleaseAndGetAddressOf()))))
            return set_error(error,"DSV heap");
        rtv_stride_=device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        if(headless) {
            if(!create_headless_target(error)) return false;
        } else {
            DXGI_SWAP_CHAIN_DESC1 desc{};
            desc.Width=width;desc.Height=height;desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
            desc.BufferCount=2;desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
            ComPtr<IDXGISwapChain1> swap;
            hr=factory_->CreateSwapChainForHwnd(queue_.Get(),
                glfwGetWin32Window(window),&desc,nullptr,nullptr,
                swap.ReleaseAndGetAddressOf());
            if(FAILED(hr)||FAILED(swap.As(&swap_chain_)))
                return set_error(error,"DXGI swap chain");
            for(UINT i=0;i<2;++i) {
                swap_chain_->GetBuffer(i,IID_PPV_ARGS(targets_[i].ReleaseAndGetAddressOf()));
                auto handle=rtv_heap_->GetCPUDescriptorHandleForHeapStart();
                handle.ptr+=static_cast<SIZE_T>(i)*rtv_stride_;
                device_->CreateRenderTargetView(targets_[i].Get(),nullptr,handle);
            }
        }
        if(!create_depth(error) || !create_pipeline(error)) return false;
        if(!make_upload_buffer(nullptr,overlay_capacity*sizeof(OverlayVertex),overlay_buffer_,error)) return false;
        overlay_vertices_.reserve(overlay_capacity);
        return true;
    }

    [[nodiscard]] bool upload_scene(const scene::SceneDefinition &definition,
                                    std::string &error) {
        wait();
        ranges_.clear();
        std::vector<scene::Vertex> vertices;
        std::vector<std::uint32_t> indices;
        try {
            for(const scene::TriangleMesh &mesh:definition.meshes) {
                GeometryRange range{};
                range.base_vertex=static_cast<std::int32_t>(vertices.size());
                range.first_index=static_cast<std::uint32_t>(indices.size());
                range.index_count=static_cast<std::uint32_t>(mesh.indices.size());
                vertices.insert(vertices.end(),mesh.vertices.begin(),mesh.vertices.end());
                indices.insert(indices.end(),mesh.indices.begin(),mesh.indices.end());
                ranges_.push_back(range);
            }
        } catch(...) { error="render geometry allocation failed";return false; }
        if(vertices.empty()||indices.empty()) { error="scene has no render geometry";return false; }
        if(!make_upload_buffer(vertices.data(),vertices.size()*sizeof(scene::Vertex),
                               vertex_buffer_,error) ||
           !make_upload_buffer(indices.data(),indices.size()*sizeof(std::uint32_t),
                               index_buffer_,error)) return false;
        vertex_view_={vertex_buffer_->GetGPUVirtualAddress(),
            static_cast<UINT>(vertices.size()*sizeof(scene::Vertex)),sizeof(scene::Vertex)};
        index_view_={index_buffer_->GetGPUVirtualAddress(),
            static_cast<UINT>(indices.size()*sizeof(std::uint32_t)),DXGI_FORMAT_R32_UINT};
        return true;
    }

    [[nodiscard]] bool render(const scene::SceneDefinition &definition,
                              const scene::SceneInstance &instance,
                              pm::World &world,const catalog::GalleryEntry &entry,
                              const InputState &input, std::string &error) {
        wait();
        allocator_->Reset();
        commands_->Reset(allocator_.Get(),pipeline_.Get());
        const UINT frame=headless_?0U:swap_chain_->GetCurrentBackBufferIndex();
        ID3D12Resource *target=targets_[frame].Get();
        if(!headless_) transition(target,D3D12_RESOURCE_STATE_PRESENT,
                                  D3D12_RESOURCE_STATE_RENDER_TARGET);
        auto rtv=rtv_heap_->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr+=static_cast<SIZE_T>(frame)*rtv_stride_;
        const auto dsv=dsv_heap_->GetCPUDescriptorHandleForHeapStart();
        const float clear[]={entry.background.red/255.0F,entry.background.green/255.0F,
                             entry.background.blue/255.0F,1.0F};
        commands_->ClearRenderTargetView(rtv,clear,0,nullptr);
        commands_->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,1.0F,0,0,nullptr);
        D3D12_VIEWPORT viewport{0,0,static_cast<float>(width_),
            static_cast<float>(height_),0,1};
        D3D12_RECT scissor{0,0,static_cast<LONG>(width_),static_cast<LONG>(height_)};
        commands_->RSSetViewports(1,&viewport);commands_->RSSetScissorRects(1,&scissor);
        commands_->OMSetRenderTargets(1,&rtv,FALSE,&dsv);
        commands_->SetGraphicsRootSignature(root_signature_.Get());
        commands_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commands_->IASetVertexBuffers(0,1,&vertex_view_);
        commands_->IASetIndexBuffer(&index_view_);
        const auto camera=input.camera.camera();
        const Matrix4 view_projection=multiply(view_matrix(camera.eye,camera.target),
            projection_matrix(static_cast<float>(width_)/height_,
                              camera.vertical_field_of_view_degrees));
        for(std::size_t body=0;body<instance.rigid_bodies.size();++body) {
            parallel_mater::RigidBodyState state{};
            const pm::Status status=world.read_rigid_body_state(
                instance.rigid_bodies[body],state);
            if(!status) { error=status_error(status,"rigid readback failed");return false; }
            const Matrix4 model=model_matrix(state);
            for(std::uint32_t mesh_index:definition.rigid_bodies[body].mesh_indices) {
                if(mesh_index>=ranges_.size() || !definition.meshes[mesh_index].visible) continue;
                DrawConstants constants{};
                constants.model=model;
                constants.model_view_projection=multiply(model,view_projection);
                const auto color=definition.meshes[mesh_index].base_color;
                constants.color[0]=color.x;constants.color[1]=color.y;
                constants.color[2]=color.z;constants.color[3]=1.0F;
                commands_->SetGraphicsRoot32BitConstants(0,
                    static_cast<UINT>(sizeof(constants)/4),&constants,0);
                const GeometryRange range=ranges_[mesh_index];
                commands_->DrawIndexedInstanced(range.index_count,1,
                    range.first_index,range.base_vertex,0);
            }
        }
        // Native selector ribbon: entry color, captured in headless validation too.
        D3D12_RECT ribbon{0,static_cast<LONG>(height_)-8,
                          static_cast<LONG>(width_),static_cast<LONG>(height_)};
        const float accent[]={entry.icon.red/255.0F,entry.icon.green/255.0F,
                              entry.icon.blue/255.0F,1.0F};
        commands_->ClearRenderTargetView(rtv,accent,1,&ribbon);
        draw_overlay(entry,input);
        if(overlay_vertices_.size()>overlay_capacity) { error="Overlay capacity exceeded";return false; }
        void *overlay_data{}; D3D12_RANGE no_read{0,0};
        if(FAILED(overlay_buffer_->Map(0,&no_read,&overlay_data))) {
            error="Overlay upload map failed";return false;
        }
        const auto overlay_bytes=overlay_vertices_.size()*sizeof(OverlayVertex);
        std::memcpy(overlay_data,overlay_vertices_.data(),overlay_bytes);
        D3D12_RANGE written{0,overlay_bytes}; overlay_buffer_->Unmap(0,&written);
        D3D12_VERTEX_BUFFER_VIEW overlay_view{overlay_buffer_->GetGPUVirtualAddress(),
            static_cast<UINT>(overlay_bytes),sizeof(OverlayVertex)};
        commands_->SetPipelineState(overlay_pipeline_.Get());
        commands_->IASetVertexBuffers(0,1,&overlay_view);
        commands_->DrawInstanced(static_cast<UINT>(overlay_vertices_.size()),1,0,0);
        if(!headless_) transition(target,D3D12_RESOURCE_STATE_RENDER_TARGET,
                                  D3D12_RESOURCE_STATE_PRESENT);
        commands_->Close();
        ID3D12CommandList *lists[]={commands_.Get()};queue_->ExecuteCommandLists(1,lists);
        signal();
        if(!headless_ && FAILED(swap_chain_->Present(1,0))) {
            error="Present failed";return false;
        }
        return true;
    }

    [[nodiscard]] bool capture(const std::filesystem::path &path,
                               std::string &error) {
        if(!headless_) { error="capture requires --headless";return false; }
        wait();allocator_->Reset();commands_->Reset(allocator_.Get(),nullptr);
        transition(targets_[0].Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource=targets_[0].Get();source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource=capture_buffer_.Get();destination.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint=capture_footprint_;
        commands_->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition(targets_[0].Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_RENDER_TARGET);
        commands_->Close();ID3D12CommandList *lists[]={commands_.Get()};
        queue_->ExecuteCommandLists(1,lists);signal();wait();
        std::error_code filesystem_error;
        if(path.has_parent_path())
            std::filesystem::create_directories(path.parent_path(),filesystem_error);
        std::ofstream stream(path,std::ios::binary);
        if(!stream) { error="could not open PPM output";return false; }
        stream << "P6\n" << width_ << ' ' << height_ << "\n255\n";
        void *mapped=nullptr;D3D12_RANGE range{0,static_cast<SIZE_T>(capture_bytes_)};
        if(FAILED(capture_buffer_->Map(0,&range,&mapped))) {
            error="could not map capture buffer";return false;
        }
        const auto *bytes=static_cast<const std::uint8_t *>(mapped);
        for(std::uint32_t y=0;y<height_;++y) {
            const std::uint8_t *row=bytes+y*capture_footprint_.Footprint.RowPitch;
            for(std::uint32_t x=0;x<width_;++x)
                stream.write(reinterpret_cast<const char *>(row+x*4),3);
        }
        D3D12_RANGE written{0,0};capture_buffer_->Unmap(0,&written);
        return static_cast<bool>(stream);
    }

  private:
    void draw_overlay(const catalog::GalleryEntry &entry,const InputState &input) {
        overlay_vertices_.clear();
        const std::array<float,4> background{0.025F,0.04F,0.07F,1};
        const std::array<float,4> white{0.9F,0.94F,1,1};
        const std::array<float,4> muted{0.38F,0.43F,0.5F,1};
        const std::array<float,4> selected{0.1F,0.32F,0.5F,1};
        const int scale=width_>=1000U && height_>=600U ? 2 : 1;
        const auto rectangle=[&](int x,int y,int w,int h,const auto &color) {
            D3D12_RECT rect{std::max(0,x),std::max(0,y),
                std::min(static_cast<int>(width_),x+w),
                std::min(static_cast<int>(height_),y+h)};
            if(rect.right<=rect.left || rect.bottom<=rect.top) return;
            const float left=2.0F*rect.left/width_-1,right=2.0F*rect.right/width_-1;
            const float top=1-2.0F*rect.top/height_,bottom=1-2.0F*rect.bottom/height_;
            for(const auto position:std::array<std::array<float,2>,6>{{
                {left,top},{right,top},{left,bottom},{left,bottom},{right,top},{right,bottom}}})
                overlay_vertices_.push_back({position[0],position[1],color});
        };
        const auto text=[&](int x,int y,std::string_view value,const auto &color) {
            for(char c:value) {
                const auto rows=catalog::glyph(c>='a'&&c<='z'?char(c-'a'+'A'):c);
                for(int row=0;row<7;++row) for(int column=0;column<5;++column) {
                    if((rows[row]&(1U<<(4-column)))==0U) continue;
                    const int left=x+column*scale,top=y+row*scale;
                    if(left>=0 && top>=0 && left+scale<=static_cast<int>(width_) &&
                       top+scale<=static_cast<int>(height_))
                        rectangle(left,top,scale,scale,color);
                }
                x+=6*scale;
            }
        };
        rectangle(8,8,static_cast<int>(width_)-16,scale*10+12,background);
        text(16,14,std::string(entry.name)+"   "+std::string(entry.help)+
             "   F: FPS   TAB: SCENES   R: RESET   P: PAUSE",white);
        if(input.show_fps) {
            std::ostringstream label;
            label<<std::fixed<<std::setprecision(1)<<input.fps<<" FPS  "
                 <<input.frame_ms<<" MS/FRAME  GPU PHYSICS "<<input.gpu_ms<<" MS";
            if(input.paused || input.catalog_visible) label<<"  PAUSED";
            rectangle(8,scale*10+24,static_cast<int>(width_)-16,scale*10+12,background);
            text(16,scale*10+30,label.str(),white);
        }
        if(!input.catalog_visible) return;
        const int top=scale*20+52, rows=14, spacing=scale*10+6;
        rectangle(8,top-8,static_cast<int>(width_)-16,rows*spacing+scale*22+24,background);
        text(20,top,"SCENES - ARROWS / ENTER OR CLICK - TAB TO CLOSE",white);
        const int column_width=(static_cast<int>(width_)-32)/2;
        for(std::size_t i=0;i<catalog::gallery_entries.size();++i) {
            const int x=16+static_cast<int>(i/rows)*column_width;
            const int y=top+scale*12+10+static_cast<int>(i%rows)*spacing;
            if(i==input.selected) rectangle(x,y-4,column_width-8,spacing,selected);
            std::string label(catalog::gallery_entries[i].name);
            if(i>=rigid_scene_count) label+=" - UNAVAILABLE";
            text(x+6,y,label,i<rigid_scene_count?white:muted);
        }
        text(20,top+scale*12+10+rows*spacing,
            "GRAY ENTRIES NEED UNIMPLEMENTED DIRECTCOMPUTE SUBSYSTEMS",muted);
    }
    [[nodiscard]] bool set_error(std::string &error,const char *what) {
        error=std::string("Could not create ")+what;return false;
    }
    void transition(ID3D12Resource *resource,D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};
        commands_->ResourceBarrier(1,&barrier);
    }
    void signal() { queue_->Signal(fence_.Get(),++fence_value_); }
    void wait() {
        if(!fence_ || fence_->GetCompletedValue()>=fence_value_) return;
        fence_->SetEventOnCompletion(fence_value_,fence_event_);
        WaitForSingleObject(fence_event_,INFINITE);
    }
    [[nodiscard]] bool make_upload_buffer(const void *data,std::size_t bytes,
        ComPtr<ID3D12Resource> &output,std::string &error) {
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_UPLOAD;
        heap.CreationNodeMask=heap.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width=bytes;desc.Height=1;desc.DepthOrArraySize=1;desc.MipLevels=1;
        desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if(FAILED(device_->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,
            IID_PPV_ARGS(output.ReleaseAndGetAddressOf()))))
            return set_error(error,"geometry upload buffer");
        void *mapped=nullptr;D3D12_RANGE read{0,0};
        if(FAILED(output->Map(0,&read,&mapped))) return set_error(error,"geometry map");
        if(data) std::memcpy(mapped,data,bytes);
        output->Unmap(0,nullptr);return true;
    }
    [[nodiscard]] bool create_headless_target(std::string &error) {
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask=heap.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=width_;desc.Height=height_;desc.DepthOrArraySize=1;desc.MipLevels=1;
        desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;
        desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};clear.Format=desc.Format;clear.Color[3]=1;
        if(FAILED(device_->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,&clear,
            IID_PPV_ARGS(targets_[0].ReleaseAndGetAddressOf()))))
            return set_error(error,"headless target");
        device_->CreateRenderTargetView(targets_[0].Get(),nullptr,
            rtv_heap_->GetCPUDescriptorHandleForHeapStart());
        UINT rows{};std::uint64_t row_size{};
        device_->GetCopyableFootprints(&desc,0,1,0,&capture_footprint_,&rows,
                                       &row_size,&capture_bytes_);
        heap.Type=D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width=capture_bytes_;buffer.Height=1;buffer.DepthOrArraySize=1;
        buffer.MipLevels=1;buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if(FAILED(device_->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
            IID_PPV_ARGS(capture_buffer_.ReleaseAndGetAddressOf()))))
            return set_error(error,"capture readback");
        return true;
    }
    [[nodiscard]] bool create_depth(std::string &error) {
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask=heap.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=width_;desc.Height=height_;desc.DepthOrArraySize=1;desc.MipLevels=1;
        desc.Format=DXGI_FORMAT_D32_FLOAT;desc.SampleDesc.Count=1;
        desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_CLEAR_VALUE clear{};clear.Format=desc.Format;clear.DepthStencil.Depth=1;
        if(FAILED(device_->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,
            IID_PPV_ARGS(depth_.ReleaseAndGetAddressOf()))))
            return set_error(error,"depth target");
        device_->CreateDepthStencilView(depth_.Get(),nullptr,
            dsv_heap_->GetCPUDescriptorHandleForHeapStart());return true;
    }
    [[nodiscard]] bool create_pipeline(std::string &error) {
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameter.Constants={0,0,static_cast<UINT>(sizeof(DrawConstants)/4)};
        parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC root{};root.NumParameters=1;root.pParameters=&parameter;
        root.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> blob,error_blob;
        if(FAILED(D3D12SerializeRootSignature(&root,D3D_ROOT_SIGNATURE_VERSION_1_0,
            blob.ReleaseAndGetAddressOf(),error_blob.ReleaseAndGetAddressOf())))
            return set_error(error,"render root signature blob");
        if(FAILED(device_->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),
            IID_PPV_ARGS(root_signature_.ReleaseAndGetAddressOf()))))
            return set_error(error,"render root signature");
        constexpr D3D12_INPUT_ELEMENT_DESC input[]={{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,
            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},{"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,
            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},{"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,
            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};desc.pRootSignature=root_signature_.Get();
        desc.VS={pm_d3d12_gallery_vs,sizeof(pm_d3d12_gallery_vs)};
        desc.PS={pm_d3d12_gallery_ps,sizeof(pm_d3d12_gallery_ps)};
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.SampleMask=UINT_MAX;desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable=TRUE;
        desc.DepthStencilState.DepthEnable=TRUE;desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS;
        desc.InputLayout={input,static_cast<UINT>(std::size(input))};
        desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets=1;desc.RTVFormats[0]=DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.DSVFormat=DXGI_FORMAT_D32_FLOAT;desc.SampleDesc.Count=1;
        if(FAILED(device_->CreateGraphicsPipelineState(&desc,
            IID_PPV_ARGS(pipeline_.ReleaseAndGetAddressOf()))))
            return set_error(error,"render pipeline");
        constexpr D3D12_INPUT_ELEMENT_DESC overlay_input[]={
            {"POSITION",0,DXGI_FORMAT_R32G32_FLOAT,0,0,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,8,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
        desc.VS={pm_d3d12_gallery_ui_vs,sizeof(pm_d3d12_gallery_ui_vs)};
        desc.PS={pm_d3d12_gallery_ui_ps,sizeof(pm_d3d12_gallery_ui_ps)};
        desc.InputLayout={overlay_input,2};
        desc.DepthStencilState.DepthEnable=FALSE;
        desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO;
        if(FAILED(device_->CreateGraphicsPipelineState(&desc,
            IID_PPV_ARGS(overlay_pipeline_.ReleaseAndGetAddressOf()))))
            return set_error(error,"overlay pipeline");
        return true;
    }

    ComPtr<IDXGIFactory6> factory_{};ComPtr<ID3D12Device> device_{};
    ComPtr<ID3D12CommandQueue> queue_{};ComPtr<ID3D12CommandAllocator> allocator_{};
    ComPtr<ID3D12GraphicsCommandList> commands_{};ComPtr<ID3D12Fence> fence_{};
    HANDLE fence_event_{};std::uint64_t fence_value_{};
    ComPtr<IDXGISwapChain3> swap_chain_{};ComPtr<ID3D12DescriptorHeap> rtv_heap_,dsv_heap_;
    std::array<ComPtr<ID3D12Resource>,2> targets_{};ComPtr<ID3D12Resource> depth_{};
    ComPtr<ID3D12Resource> capture_buffer_,vertex_buffer_,index_buffer_;
    ComPtr<ID3D12RootSignature> root_signature_{};ComPtr<ID3D12PipelineState> pipeline_{};
    D3D12_VERTEX_BUFFER_VIEW vertex_view_{};D3D12_INDEX_BUFFER_VIEW index_view_{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT capture_footprint_{};std::uint64_t capture_bytes_{};
    std::vector<GeometryRange> ranges_{};std::uint32_t rtv_stride_{},width_{},height_{};
    bool headless_{};
    struct OverlayVertex { float x,y; std::array<float,4> color; };
    static constexpr std::size_t overlay_capacity=262144;
    std::vector<OverlayVertex> overlay_vertices_;
    ComPtr<ID3D12Resource> overlay_buffer_;
    ComPtr<ID3D12PipelineState> overlay_pipeline_;
};

[[nodiscard]] std::filesystem::path scene_asset(const char *source_path) {
    std::array<wchar_t,32768> executable{};
    const DWORD size=GetModuleFileNameW(nullptr,executable.data(),
                                        static_cast<DWORD>(executable.size()));
    if(size!=0U && size<executable.size()) {
        const auto directory=std::filesystem::path(executable.data()).parent_path()/L"assets";
        // An installed package must use its own assets, even when a file is
        // missing. Do not silently fall back to a developer's source checkout.
        if(std::filesystem::is_directory(directory))
            return directory/std::filesystem::path(source_path).filename();
    }
    return source_path;
}

[[nodiscard]] std::filesystem::path scene_path(std::size_t index) {
    using Source=catalog::GallerySceneSource;
    switch(catalog::gallery_entries.at(index).source) {
    case Source::default_scene: return scene_asset(PARALLEL_MATER_DEFAULT_SCENE_PATH);
    case Source::constraint_fixed: return scene_asset(PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH);
    case Source::constraint_point: return scene_asset(PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH);
    case Source::constraint_hinge: return scene_asset(PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH);
    case Source::constraint_piston: return scene_asset(PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH);
    case Source::constraint_generic: return scene_asset(PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH);
    case Source::constraint_motor_spring: return scene_asset(PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH);
    case Source::dump_truck: return scene_asset(PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH);
    default: return {};
    }
}

[[nodiscard]] std::string file_stem(const catalog::GalleryEntry &entry) {
    std::string result;
    for(char character:entry.name) {
        if(character>='A'&&character<='Z') result.push_back(character-'A'+'a');
        else if(character>='a'&&character<='z') result.push_back(character);
        else if(character>='0'&&character<='9') result.push_back(character);
        else if(result.empty()||result.back()!='-') result.push_back('-');
    }
    while(!result.empty()&&result.back()=='-') result.pop_back();
    return result;
}

[[nodiscard]] bool toggle_scene_constraints(
    scene::SceneDefinition &definition,const scene::SceneInstance &instance,
    pm::World &world,std::string &error) {
    if(definition.rigid_constraints.empty() ||
       definition.rigid_constraints.size()!=instance.rigid_constraints.size()) {
        error="constraint toggle scene needs matching constraints";
        return false;
    }
    bool enable=false;
    for(const auto id:instance.rigid_constraints) {
        parallel_mater::RigidConstraintState state{};
        const auto status=world.read_rigid_constraint_state(id,state);
        if(!status) { error=status_error(status,"constraint state read failed");return false; }
        enable=enable || !state.enabled;
    }
    if(enable) {
        // Point-scene arms may be far from their anchors after release. Match
        // CUDA gallery behavior by restoring authored poses before reattaching.
        for(std::size_t index=0;index<definition.rigid_bodies.size();++index) {
            const auto &body=definition.rigid_bodies[index];
            if(body.options.motion!=parallel_mater::MotionType::dynamic) continue;
            const auto status=world.set_rigid_body_state(
                instance.rigid_bodies[index],body.options.initial_state);
            if(!status) { error=status_error(status,"point assembly restore failed");return false; }
        }
    }
    for(std::size_t index=0;index<definition.rigid_constraints.size();++index) {
        auto &joint=definition.rigid_constraints[index];
        auto options=joint.options;
        options.body_a=instance.rigid_bodies[joint.body_a];
        options.body_b=instance.rigid_bodies[joint.body_b];
        options.enabled=enable;
        const auto status=world.update_rigid_constraint(
            instance.rigid_constraints[index],options);
        if(!status) { error=status_error(status,"constraint toggle failed");return false; }
        joint.options=options;
    }
    std::cout<<"Scene action: constraints "
             <<(enable?"restored":"released")<<std::endl;
    return true;
}

[[nodiscard]] bool run_scene(const Options &options,std::size_t scene_index,
    GraphicsContext &graphics,Renderer &renderer,GLFWwindow *window,std::filesystem::path output,
    InputState &input,std::uint64_t &state_hash,std::string &error) {
    scene::SceneDefinition definition;
    const auto asset_path=scene_path(scene_index);
    std::cout<<"Scene asset: "<<asset_path<<std::endl;
    if(!scene::load_glb_scene(asset_path,definition,error)) return false;
    const bool dump=scene_index==catalog::gallery_context_index(catalog::GalleryContext::dump);
    if(dump && !scene::configure_dump_payload(definition,options.dump_spheres,error)) return false;
    parallel_mater::WorldOptions world_options{};
    pm::Status status=scene::scene_world_options(definition,world_options);
    if(!status) { error=status_error(status,"scene options failed");return false; }
    pm::World world;
    status=pm::World::create(world_options,
        {graphics.device.Get(),graphics.queue.Get()},world);
    if(!status) { error=status_error(status,"D3D12 world failed");return false; }
    scene::SceneInstance instance;
    status=scene::instantiate_scene(definition,world,instance);
    if(!status) { error=status_error(status,"scene instantiation failed");return false; }
    scene::ArrowForces forces;
    status=forces.initialize(definition,instance);
    if(!status) { error=status_error(status,"arrow force initialization failed");return false; }
    std::vector<parallel_mater::RigidBodyId> vertical_gravity_bodies;
    for(std::size_t index=0;index<definition.rigid_bodies.size();++index) {
        const auto &body=definition.rigid_bodies[index];
        if(body.options.motion==parallel_mater::MotionType::dynamic &&
           !body.follows_gravity_tilt)
            vertical_gravity_bodies.push_back(instance.rigid_bodies[index]);
    }
    scene::DumpTruckBed bed;
    if(dump && !bed.initialize(definition)) { error="DumpLift constraint missing";return false; }
    struct Steering { std::size_t index; parallel_mater::Quaternion orientation; bool front; };
    std::vector<Steering> steering;
    for(std::size_t i=0;i<definition.rigid_constraints.size();++i) {
        const auto &spring=definition.rigid_constraints[i];
        if(spring.options.type!=parallel_mater::RigidConstraintType::generic_spring) continue;
        const auto motor=std::find_if(definition.rigid_constraints.begin(),
            definition.rigid_constraints.end(),[&](const auto &candidate) {
                return candidate.options.type==parallel_mater::RigidConstraintType::motor &&
                    candidate.body_a==spring.body_b;
            });
        if(motor!=definition.rigid_constraints.end())
            steering.push_back({i,spring.options.local_orientation_a,
                               motor->name.find("Front")!=std::string::npos});
    }
    float steering_angle{};
    if(!renderer.upload_scene(definition,error)) return false;
    const auto &entry=catalog::gallery_entries[scene_index];
    input.current=scene_index; input.selected=scene_index; input.scene_action=false;
    input.clear_arrows();
    input.camera.set_preset(entry.camera);
    input.fps=0; input.frame_ms=0; input.gpu_ms=0;
    if(window) glfwSetWindowTitle(window,(std::string(entry.name)+" - DirectCompute Gallery").c_str());
    std::cout<<"Scene: "<<entry.name<<std::endl;
    using Clock=std::chrono::steady_clock;
    auto last=Clock::now();
    double accumulator=1.0/60.0, sample_seconds{},total_seconds{},total_gpu{};
    float logged_x{},logged_z{};
    std::uint32_t sample_frames{},physics_frames{};
    const auto limit=options.headless || options.frames_set ? options.frames : UINT32_MAX;
    for(std::uint32_t frame=0;frame<limit;++frame) {
        const auto started=Clock::now();
        accumulator=std::min(accumulator+std::chrono::duration<double>(started-last).count(),1.0/30.0);
        last=started;
        if(window) glfwPollEvents();
        if(window && (glfwWindowShouldClose(window) || input.close_requested || input.requested_scene)) break;
        const bool frozen=window && (input.catalog_visible || input.paused);
        if(frozen) accumulator=0;
        if(!frozen && (options.headless || options.frames_set || accumulator>=1.0/60.0)) {
        accumulator=std::max(0.0,accumulator-1.0/60.0);
        parallel_mater::StepOptions step{};
        step.timestep=1.0F/60.0F;step.substeps=options.substeps;
        const float x=window && !input.catalog_visible ? input.right_input() : 0.0F;
        const float z=window && !input.catalog_visible ? input.up_input() : 0.0F;
        if(window && (x!=logged_x || z!=logged_z)) {
            std::cout<<"Arrow input: right="<<x<<" up="<<z<<std::endl;
            logged_x=x;logged_z=z;
        }
        step.gravity=pm::viewer::control_gravity(
            entry.controls,forces.active(),input.camera.camera(),x,z,
            definition.gravity_scale);
        if(entry.controls==catalog::GalleryControlPolicy::motor_drive) {
            for(std::size_t i=0;i<definition.rigid_constraints.size();++i) {
                auto &joint=definition.rigid_constraints[i];
                if(joint.options.type!=parallel_mater::RigidConstraintType::motor ||
                   joint.options.motor.angular_target_velocity==-z*8.0F) continue;
                auto motor_options=joint.options;
                motor_options.body_a=instance.rigid_bodies[joint.body_a];
                motor_options.body_b=instance.rigid_bodies[joint.body_b];
                motor_options.motor.angular_target_velocity=-z*8.0F;
                status=world.update_rigid_constraint(instance.rigid_constraints[i],motor_options);
                if(!status) { error=status_error(status,"motor update failed");return false; }
                joint.options=motor_options;
            }
            constexpr float radians=3.14159265359F/180.0F;
            const float previous=steering_angle;
            steering_angle+=std::clamp(x*25.0F*radians-steering_angle,-90.0F*radians/60,90.0F*radians/60);
            if(previous!=steering_angle) for(const auto &binding:steering) {
                auto &joint=definition.rigid_constraints[binding.index];
                auto steering_options=joint.options;
                steering_options.body_a=instance.rigid_bodies[joint.body_a];
                steering_options.body_b=instance.rigid_bodies[joint.body_b];
                const float angle=(binding.front?steering_angle:-steering_angle)*0.5F;
                const float s=std::sin(angle),c=std::cos(angle);
                const auto q=binding.orientation;
                steering_options.local_orientation_a={q.x*c+q.y*s,q.y*c-q.x*s,q.z*c+q.w*s,q.w*c-q.z*s};
                status=world.update_rigid_constraint(instance.rigid_constraints[binding.index],steering_options);
                if(!status) { error=status_error(status,"steering update failed");return false; }
                joint.options=steering_options;
            }
        }
        const parallel_mater::Vec3 vertical_gravity{
            0.0F,-9.81F*definition.gravity_scale,0.0F};
        const parallel_mater::Vec3 compensation{
            vertical_gravity.x-step.gravity.x,
            vertical_gravity.y-step.gravity.y,
            vertical_gravity.z-step.gravity.z};
        if(!vertical_gravity_bodies.empty() &&
           (compensation.x!=0.0F || compensation.y!=0.0F || compensation.z!=0.0F)) {
            status=world.apply_central_acceleration(
                {vertical_gravity_bodies.data(),vertical_gravity_bodies.size()},
                compensation);
            if(!status) { error=status_error(status,"gravity override failed");return false; }
        }
        status=forces.apply(world,input.camera.camera(),x,z);
        if(!status) { error=status_error(status,"arrow force failed");return false; }
        if(input.scene_action) {
            if(dump) bed.toggle();
            else if(catalog::toggles_constraint(entry.controls) &&
                    !toggle_scene_constraints(definition,instance,world,error)) return false;
        }
        if(dump) {
            status=bed.advance(world,definition,instance,step.timestep);
            if(!status) { error=status_error(status,"dump bed failed");return false; }
        }
        input.scene_action=false;
        step.collect_kernel_timings=options.validate || options.benchmark || input.show_fps;
        step.collect_rigid_contacts=options.validate;
        status=world.step(step);
        if(!status) { error=status_error(status,"physics step failed");return false; }
        ++physics_frames;
        if(step.collect_kernel_timings) {
            parallel_mater::WorldStepTimings timings{};
            status=world.collect_step_timings(timings);
            if(!status) { error=status_error(status,"GPU timing failed");return false; }
            input.gpu_ms=timings.total_gpu_milliseconds;
            total_gpu+=input.gpu_ms;
        }
        }
        if(!renderer.render(definition,instance,world,entry,input,error)) return false;
        const double seconds=std::chrono::duration<double>(Clock::now()-started).count();
        total_seconds+=seconds; sample_seconds+=seconds; ++sample_frames;
        if(sample_seconds>=0.25) {
            input.fps=sample_frames/sample_seconds;
            input.frame_ms=sample_seconds*1000/sample_frames;
            sample_seconds=0; sample_frames=0;
        }
    }
    if(options.benchmark && physics_frames)
        std::cout<<entry.name<<": frames="<<physics_frames<<" wall_ms/frame="
                 <<total_seconds*1000/physics_frames<<" gpu_ms/frame="<<total_gpu/physics_frames<<std::endl;
    if(options.benchmark && physics_frames) {
        parallel_mater::WorldStepTimings timings{};
        if(world.collect_step_timings(timings))
            std::cout<<"Last frame phases (ms): integrate="<<timings.rigid_integration.total_milliseconds
                <<" prepare="<<timings.rigid_world_bounds.total_milliseconds
                <<" contacts="<<timings.rigid_contact_evaluation.total_milliseconds
                <<" compact="<<timings.rigid_pair_compaction.total_milliseconds
                <<" solve="<<timings.rigid_contact_solve.total_milliseconds<<std::endl;
        parallel_mater::WorldStatistics stats{};
        if(world.collect_statistics(stats))
            std::cout<<"Live pairs="<<stats.rigid_contact_live_pairs
                     <<" colors="<<stats.rigid_contact_color_count
                     <<" passes="<<stats.rigid_contact_maximum_passes<<std::endl;
    }
    if(options.headless && !renderer.capture(output,error)) return false;
    state_hash=1469598103934665603ULL;
    const auto hash_bytes=[&](const void *data,std::size_t size) {
        const auto *bytes=static_cast<const std::uint8_t *>(data);
        for(std::size_t i=0;i<size;++i) {
            state_hash^=bytes[i];state_hash*=1099511628211ULL;
        }
    };
    for(const parallel_mater::RigidBodyId id:instance.rigid_bodies) {
        parallel_mater::RigidBodyState state{};
        status=world.read_rigid_body_state(id,state);
        if(!status) { error=status_error(status,"state hash readback failed");return false; }
        hash_bytes(&id,sizeof(id));hash_bytes(&state,sizeof(state));
    }
    if(options.validate) {
        parallel_mater::WorldStatistics statistics{};
        parallel_mater::WorldStepTimings timings{};
        status=world.collect_statistics(statistics);
        if(status) status=world.collect_step_timings(timings);
        if(!status || statistics.rigid_body_count!=instance.rigid_bodies.size()) {
            error="validation statistics mismatch";return false;
        }
        std::cout << entry.name << ": bodies=" << statistics.rigid_body_count
                  << " constraints=" << statistics.rigid_constraint_count
                  << " gpu_ms=" << timings.total_gpu_milliseconds << '\n';
    }
    return true;
}

} // namespace

int main(int argc,char **argv) {
    Options options;
    if(!parse_options(argc,argv,options)) return argc>1?2:0;
    if(options.all_scenes) {
        std::cerr << "--all-scenes is unavailable in the rigid-body gate; "
                     "use --all-rigid-scenes\n";
        return 3;
    }
    if(options.list_scenes) {
        for(std::size_t i=0;i<catalog::gallery_entries.size();++i)
            std::cout << i << (i<rigid_scene_count?" [D3D12 rigid] ":" [later gate] ")
                      << catalog::gallery_entries[i].name << '\n';
        if(!options.list_adapters) return 0;
    }
    Options context_options=options;
    context_options.validate=options.validate;
    GraphicsContext graphics;
    std::string error;
    // Adapter listing must include unsupported devices, so create only factory first.
    if(options.list_adapters) {
        ComPtr<IDXGIFactory6> factory;
        if(FAILED(CreateDXGIFactory2(0,IID_PPV_ARGS(factory.ReleaseAndGetAddressOf())))) {
            std::cerr << "Could not create DXGI factory\n";return 1;
        }
        const auto adapters=enumerate_adapters(factory.Get());
        for(std::size_t i=0;i<adapters.size();++i) {
            const auto &item=adapters[i];
            std::cout << i << ": " << narrow(item.description.Description)
                      << " FL" << feature_level_name(item.feature_level)
                      << " tier " << static_cast<unsigned>(item.binding_tier)
                      << (item.qualifies?" [supported]":" [requires 64 UAVs]") << '\n';
        }
        ComPtr<IDXGIAdapter> warp_base;ComPtr<IDXGIAdapter1> warp;
        if(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(warp_base.ReleaseAndGetAddressOf())))&&
           SUCCEEDED(warp_base.As(&warp))) {
            const auto item=inspect_adapter(std::move(warp));
            std::cout << "warp: " << narrow(item.description.Description)
                      << " FL" << feature_level_name(item.feature_level)
                      << " tier " << static_cast<unsigned>(item.binding_tier)
                      << (item.qualifies?" [supported]":" [unsupported]") << '\n';
        }
        return 0;
    }
    if(!create_graphics_context(context_options,graphics,error)) {
        std::cerr << error << '\n';return 1;
    }
    std::cout << "Adapter: " << narrow(graphics.adapter.description.Description)
              << " FL" << feature_level_name(graphics.adapter.feature_level)
              << " binding tier " << static_cast<unsigned>(graphics.adapter.binding_tier)
              << std::endl;
    GLFWwindow *window=nullptr;
    InputState input;
    input.show_fps=options.show_fps;
    input.catalog_visible=options.show_catalog;
    if(!options.headless) {
        if(!glfwInit()) { std::cerr << "GLFW initialization failed\n";return 1; }
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);
        window=glfwCreateWindow(static_cast<int>(options.width),
            static_cast<int>(options.height),"Parallel Mater D3D12 Gallery",nullptr,nullptr);
        if(!window) { glfwTerminate();std::cerr << "Window creation failed\n";return 1; }
        install_input(window,input);
    }
    bool success=true;
    auto renderer=std::make_unique<Renderer>();
    if(!renderer->create(graphics,window,options.width,options.height,options.headless,error)) {
        std::cerr<<error<<'\n';
        renderer.reset();
        if(window) { glfwDestroyWindow(window);glfwTerminate(); }
        return 1;
    }
    const std::uint32_t repetitions=options.repeat!=0U ? options.repeat : 1U;
    const auto repeat_scene=[&](std::size_t index,
                                const std::filesystem::path &output) {
        std::optional<std::uint64_t> expected;
        for(std::uint32_t repetition=0;repetition<repetitions;++repetition) {
            std::uint64_t hash{};
            if(!run_scene(options,index,graphics,*renderer,window,output,input,hash,error)) return false;
            if(expected && *expected!=hash) {
                error="determinism hash changed across same-adapter repetitions";
                return false;
            }
            expected=hash;
        }
        if(options.validate)
            std::cout << catalog::gallery_entries[index].name
                      << ": deterministic hash 0x" << std::hex << *expected
                      << std::dec << " x" << repetitions << '\n';
        return true;
    };
    if(options.all_rigid_scenes) {
        const std::filesystem::path directory=options.output.empty()
            ? std::filesystem::path("d3d12-gallery-captures") : options.output;
        for(std::size_t index=0;index<rigid_scene_count && success;++index) {
            const auto output=directory/(file_stem(catalog::gallery_entries[index])+".ppm");
            success=repeat_scene(index,output);
        }
    } else {
        const std::filesystem::path output=options.output.empty()
            ? std::filesystem::path("d3d12-gallery.ppm") : options.output;
        std::size_t index=options.scene_index;
        do {
            input.requested_scene.reset();
            success=repeat_scene(index,output);
            if(!success || !window || !input.requested_scene || input.close_requested ||
               glfwWindowShouldClose(window)) break;
            index=*input.requested_scene;
        } while(true);
    }
    renderer.reset();
    if(window) glfwDestroyWindow(window);
    if(!options.headless) glfwTerminate();
    if(success && options.validate)
        success=debug_queue_clean(graphics,error);
    if(!success) {
        error += "\n" + dred_diagnostics(graphics.device.Get());
        if(options.validate) {
            std::string validation_error;
            if(!debug_queue_clean(graphics,validation_error))
                error += "\n" + validation_error;
        }
        std::cerr << error << '\n';return 1;
    }
    return 0;
}
