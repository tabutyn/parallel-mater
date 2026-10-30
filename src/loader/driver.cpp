// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace parallel_mater::loader {

const LoaderDispatch loader_dispatch{};

namespace {

enum class ManifestKind { driver, layer };

using NativeString = std::filesystem::path::string_type;

[[nodiscard]] std::filesystem::path path_from_utf8(const std::string &value) {
    return std::filesystem::path(std::u8string(value.begin(), value.end()));
}

void *open_library(const std::filesystem::path &path) noexcept {
#if defined(_WIN32)
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(path, error);
    const std::filesystem::path &load_path = error ? path : absolute;
    return reinterpret_cast<void *>(LoadLibraryExW(
        load_path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

std::string library_error(const char *fallback) {
#if defined(_WIN32)
    const DWORD code = GetLastError();
    if (code != ERROR_SUCCESS)
        return std::error_code(static_cast<int>(code), std::system_category())
            .message();
    return fallback;
#else
    const char *message = dlerror();
    return message == nullptr ? fallback : message;
#endif
}

void close_library(void *library) noexcept {
#if defined(_WIN32)
    if (library != nullptr) FreeLibrary(reinterpret_cast<HMODULE>(library));
#else
    if (library != nullptr) dlclose(library);
#endif
}

void *symbol(void *library, const char *name) noexcept {
#if defined(_WIN32)
    return reinterpret_cast<void *>(
        GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

std::filesystem::path library_from_manifest(
    const std::filesystem::path &manifest, ManifestKind kind) {
    std::ifstream input(manifest);
    if (!input) return {};
    std::ostringstream contents;
    contents << input.rdbuf();
    std::smatch match;
    const std::string text = contents.str();
    if (!std::regex_search(text, match,
            std::regex(R"re("file_format_version"\s*:\s*"1\.0\.0")re")))
        return {};
    const std::regex object_pattern(kind == ManifestKind::driver
        ? R"re("PMD"\s*:)re"
        : R"re("layer"\s*:)re");
    if (!std::regex_search(text, object_pattern)) return {};
    if (std::regex_search(text, match,
            std::regex(R"re("library_arch"\s*:\s*"([0-9]+)")re")) &&
        match[1].str() != std::to_string(sizeof(void *) * 8U))
        return {};
    if (std::regex_search(text, match,
            std::regex(R"re("api_version"\s*:\s*"([0-9]+)\.([0-9]+)\.[0-9]+")re"))) {
        unsigned major{};
        unsigned minor{};
        const std::string major_text = match[1].str();
        const std::string minor_text = match[2].str();
        const auto major_result = std::from_chars(
            major_text.data(), major_text.data() + major_text.size(), major);
        const auto minor_result = std::from_chars(
            minor_text.data(), minor_text.data() + minor_text.size(), minor);
        if (major_result.ec != std::errc{} ||
            minor_result.ec != std::errc{})
            return {};
        if (major != PM_API_VERSION_MAJOR(PM_API_VERSION_0_3) ||
            minor != PM_API_VERSION_MINOR(PM_API_VERSION_0_3))
            return {};
    }
    if (!std::regex_search(text, match,
            std::regex(R"re("library_path"\s*:\s*"([^"]+)")re")))
        return {};
    std::filesystem::path path = path_from_utf8(match[1].str());
    if (path.is_relative()) path = manifest.parent_path() / path;
    return path.lexically_normal();
}

void append_manifest(std::vector<std::filesystem::path> &paths,
                     const std::filesystem::path &path, ManifestKind kind) {
    const std::filesystem::path library = library_from_manifest(path, kind);
    if (!library.empty()) paths.push_back(library);
}

void append_directory(std::vector<std::filesystem::path> &paths,
                      const std::filesystem::path &directory,
                      ManifestKind kind) {
    std::error_code error;
    std::vector<std::filesystem::path> manifests;
    for (const auto &entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (entry.path().extension() == ".json")
            manifests.push_back(entry.path());
    }
    std::sort(manifests.begin(), manifests.end());
    for (const auto &manifest : manifests)
        append_manifest(paths, manifest, kind);
}

void append_list(std::vector<std::filesystem::path> &paths,
                 const NativeString &value,
                 bool manifests, ManifestKind kind) {
#if defined(_WIN32)
    constexpr wchar_t separator = L';';
    std::wstringstream stream(value);
    std::wstring item;
#else
    constexpr char separator = ':';
    std::stringstream stream(value);
    std::string item;
#endif
    while (std::getline(stream, item, separator)) {
        if (item.empty()) continue;
        if (manifests) append_manifest(paths, item, kind);
        else append_directory(paths, item, kind);
    }
}

std::optional<NativeString> environment_value(
#if defined(_WIN32)
    const wchar_t *name
#else
    const char *name
#endif
) {
#if defined(_WIN32)
    const DWORD required = GetEnvironmentVariableW(name, nullptr, 0U);
    if (required == 0U) return std::nullopt;
    std::wstring value(required, L'\0');
    const DWORD written = GetEnvironmentVariableW(name, value.data(), required);
    if (written == 0U || written >= required) return std::nullopt;
    value.resize(written);
    return value;
#else
    const char *value = std::getenv(name);
    if (value == nullptr) return std::nullopt;
    return std::string(value);
#endif
}

void append_user_directory(std::vector<std::filesystem::path> &paths,
                           const char *leaf, ManifestKind kind) {
#if defined(_WIN32)
    if (const auto root = environment_value(L"LOCALAPPDATA"))
        append_directory(paths, std::filesystem::path(*root) /
             "ParallelMater" / leaf, kind);
#else
    if (const auto root = environment_value("XDG_DATA_HOME"))
        append_directory(paths, std::filesystem::path(*root) /
             "parallel-mater" / leaf, kind);
    else if (const auto home = environment_value("HOME"))
        append_directory(paths, std::filesystem::path(*home) /
             ".local/share/parallel-mater" / leaf, kind);
#endif
}

std::filesystem::path loader_directory() {
#if defined(_WIN32)
    HMODULE module{};
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&loader_dispatch), &module))
        return {};
    std::vector<wchar_t> path(MAX_PATH);
    for (;;) {
        const DWORD written = GetModuleFileNameW(
            module, path.data(), static_cast<DWORD>(path.size()));
        if (written == 0U) return {};
        if (written < path.size())
            return std::filesystem::path(path.data()).parent_path();
        path.resize(path.size() * 2U);
    }
#else
    Dl_info information{};
    if (dladdr(reinterpret_cast<const void *>(&loader_dispatch), &information) ==
            0 ||
        information.dli_fname == nullptr)
        return {};
    return std::filesystem::path(information.dli_fname).parent_path();
#endif
}

void append_loader_relative_directories(
    std::vector<std::filesystem::path> &paths, const char *leaf,
    ManifestKind kind) {
    const std::filesystem::path directory = loader_directory();
    if (directory.empty()) return;
    // Build trees place manifests beside the loader. Installed trees place
    // them under <prefix>/share while the loader resides in <prefix>/lib.
    append_directory(paths, directory / "manifests", kind);
    append_directory(paths, directory.parent_path() / "share" /
                                "parallel-mater" / leaf,
                     kind);
}

} // namespace

Driver::~Driver() { close_library(library); }

Layer::~Layer() { close_library(library); }

std::shared_ptr<Layer> Layer::open(const std::filesystem::path &filename,
                                   std::string &error) {
    auto layer = std::shared_ptr<Layer>(new Layer);
    layer->path = filename;
    layer->library = open_library(filename);
    if (layer->library == nullptr) {
        error = library_error("unable to load layer");
        return {};
    }
    auto negotiate = reinterpret_cast<PFN_pmLayerNegotiateLoaderInterfaceVersion>(
        symbol(layer->library, "pm_layerNegotiateLoaderInterfaceVersion"));
    layer->pre_call = reinterpret_cast<PFN_pmLayerPreCall>(
        symbol(layer->library, "pm_layerPreCall"));
    layer->post_call = reinterpret_cast<PFN_pmLayerPostCall>(
        symbol(layer->library, "pm_layerPostCall"));
    std::uint32_t version = PM_LAYER_INTERFACE_VERSION;
    if (negotiate == nullptr || layer->pre_call == nullptr ||
        layer->post_call == nullptr || negotiate(&version) != PM_SUCCESS ||
        version == 0U || version > PM_LAYER_INTERFACE_VERSION) {
        error = "incompatible Parallel Mater layer interface";
        return {};
    }
    return layer;
}

PmResult Layer::pre(const PmLayerCallInfo &call) const {
    return pre_call(&call);
}

void Layer::post(const PmLayerCallInfo &call, PmResult result) const {
    post_call(&call, result);
}

std::shared_ptr<Driver> Driver::open(const std::filesystem::path &filename,
                                     std::string &error) {
    auto driver = std::shared_ptr<Driver>(new Driver);
    driver->path = filename;
    driver->library = open_library(filename);
    if (driver->library == nullptr) {
        error = library_error("unable to load driver");
        return {};
    }
    auto negotiate = reinterpret_cast<PFN_pmDriverNegotiateLoaderInterfaceVersion>(
        symbol(driver->library, "pm_driverNegotiateLoaderInterfaceVersion"));
    driver->get_instance_proc = reinterpret_cast<PFN_pmDriverGetInstanceProcAddr>(
        symbol(driver->library, "pm_driverGetInstanceProcAddr"));
    driver->get_device_proc = reinterpret_cast<PFN_pmDriverGetDeviceProcAddr>(
        symbol(driver->library, "pm_driverGetDeviceProcAddr"));
    driver->get_physical_proc =
        reinterpret_cast<PFN_pmDriverGetPhysicalDeviceProcAddr>(
            symbol(driver->library, "pm_driverGetPhysicalDeviceProcAddr"));
    if (negotiate == nullptr || driver->get_instance_proc == nullptr ||
        driver->get_device_proc == nullptr || driver->get_physical_proc == nullptr) {
        error = "missing Parallel Mater driver entry points";
        return {};
    }
    driver->interface_version = PM_DRIVER_INTERFACE_VERSION;
    if (negotiate(&driver->interface_version) != PM_SUCCESS ||
        driver->interface_version == 0U ||
        driver->interface_version > PM_DRIVER_INTERFACE_VERSION) {
        error = "incompatible Parallel Mater driver interface";
        return {};
    }
    return driver;
}

PFN_pmVoidFunction Driver::instance_proc(PmInstance instance,
                                         const char *name) const {
    return get_instance_proc(instance, name);
}

PFN_pmVoidFunction Driver::device_proc(PmDevice device,
                                       const char *name) const {
    return get_device_proc(device, name);
}

PFN_pmVoidFunction Driver::resolve(const Route &route, const char *name) const {
    switch (route.scope) {
    case Scope::instance:
        return get_instance_proc(reinterpret_cast<PmInstance>(route.context), name);
    case Scope::physical:
        return get_physical_proc(reinterpret_cast<PmInstance>(route.context), name);
    case Scope::device:
        return get_device_proc(reinterpret_cast<PmDevice>(route.context), name);
    }
    return nullptr;
}

std::vector<std::shared_ptr<Driver>> discover_drivers() {
    std::vector<std::filesystem::path> paths;
#if defined(_WIN32)
    const auto files = environment_value(L"PM_DRIVER_FILES");
#else
    const auto files = environment_value("PM_DRIVER_FILES");
#endif
    if (files)
        append_list(paths, *files, true, ManifestKind::driver);
    else {
#if defined(_WIN32)
        const auto driver_path = environment_value(L"PM_DRIVER_PATH");
#else
        const auto driver_path = environment_value("PM_DRIVER_PATH");
#endif
        if (driver_path)
            append_list(paths, *driver_path, false, ManifestKind::driver);
        append_user_directory(paths, "drivers.d", ManifestKind::driver);
#if defined(PARALLEL_MATER_SYSTEM_DRIVER_MANIFEST_DIR)
        append_directory(paths, PARALLEL_MATER_SYSTEM_DRIVER_MANIFEST_DIR,
                         ManifestKind::driver);
#endif
#if !defined(_WIN32)
        append_directory(paths, "/etc/parallel-mater/drivers.d",
                         ManifestKind::driver);
        append_directory(paths, "/usr/local/share/parallel-mater/drivers.d",
                         ManifestKind::driver);
        append_directory(paths, "/usr/share/parallel-mater/drivers.d",
                         ManifestKind::driver);
#endif
        append_loader_relative_directories(paths, "drivers.d",
                                           ManifestKind::driver);
    }
    std::vector<std::shared_ptr<Driver>> result;
    std::set<std::filesystem::path> unique;
    for (const std::filesystem::path &path : paths) {
        std::error_code error_code;
        const std::filesystem::path canonical =
            std::filesystem::weakly_canonical(path, error_code);
        const std::filesystem::path &key = error_code ? path : canonical;
        if (!unique.insert(key).second) continue;
        std::string error;
        if (auto driver = Driver::open(path, error)) result.push_back(std::move(driver));
    }
    return result;
}

std::shared_ptr<LayerChain> load_validation_layers(
    bool enabled, PmValidationDiagnosticFn diagnostic,
    void *diagnostic_user) {
    auto result = std::make_shared<LayerChain>();
    result->diagnostic = diagnostic;
    result->diagnostic_user = diagnostic_user;
    if (!enabled) return result;
    std::vector<std::filesystem::path> paths;
#if defined(_WIN32)
    const auto files = environment_value(L"PM_LAYER_FILES");
#else
    const auto files = environment_value("PM_LAYER_FILES");
#endif
    if (files)
        append_list(paths, *files, true, ManifestKind::layer);
    else {
#if defined(_WIN32)
        const auto layer_path = environment_value(L"PM_LAYER_PATH");
#else
        const auto layer_path = environment_value("PM_LAYER_PATH");
#endif
        if (layer_path)
            append_list(paths, *layer_path, false, ManifestKind::layer);
        append_user_directory(paths, "layers.d", ManifestKind::layer);
#if defined(PARALLEL_MATER_SYSTEM_LAYER_MANIFEST_DIR)
        append_directory(paths, PARALLEL_MATER_SYSTEM_LAYER_MANIFEST_DIR,
                         ManifestKind::layer);
#endif
#if !defined(_WIN32)
        append_directory(paths, "/etc/parallel-mater/layers.d",
                         ManifestKind::layer);
        append_directory(paths, "/usr/local/share/parallel-mater/layers.d",
                         ManifestKind::layer);
        append_directory(paths, "/usr/share/parallel-mater/layers.d",
                         ManifestKind::layer);
#endif
        append_loader_relative_directories(paths, "layers.d",
                                           ManifestKind::layer);
    }
    std::set<std::filesystem::path> unique;
    for (const std::filesystem::path &path : paths) {
        std::error_code error_code;
        const std::filesystem::path canonical =
            std::filesystem::weakly_canonical(path, error_code);
        const std::filesystem::path &key = error_code ? path : canonical;
        if (!unique.insert(key).second) continue;
        std::string error;
        if (auto layer = Layer::open(path, error))
            result->entries.push_back(std::move(layer));
    }
    return result;
}

PmResult pre_call(const Route &route, const char *command,
                  std::uint32_t size, const void *arguments) noexcept {
    if (route.layers == nullptr) return PM_SUCCESS;
    for (const auto &layer : route.layers->entries) {
        const PmResult result = layer->pre({
            command, route.native, size, arguments, route.object_type,
            route.object, route.layers->diagnostic,
            route.layers->diagnostic_user});
        if (result != PM_SUCCESS) return result;
    }
    return PM_SUCCESS;
}

void post_call(const Route &route, const char *command, PmResult result,
               std::uint32_t size, const void *arguments) noexcept {
    if (route.layers == nullptr) return;
    for (auto layer = route.layers->entries.rbegin();
         layer != route.layers->entries.rend(); ++layer)
        (*layer)->post({command, route.native, size, arguments,
                        route.object_type, route.object,
                        route.layers->diagnostic,
                        route.layers->diagnostic_user}, result);
}

Route route(PmPhysicalDevice value) noexcept {
    if (value == nullptr || value->dispatch != &loader_dispatch) return {};
    return {value->driver, value->native, value->native_instance,
            Scope::physical, PM_VALIDATION_OBJECT_PHYSICAL_DEVICE, value,
            value->layers};
}

Route route(PmInstance value) noexcept {
    if (value == nullptr || value->dispatch != &loader_dispatch ||
        value->children.empty())
        return {};
    const auto &child = value->children.front();
    return {child.driver, child.native, child.native, Scope::instance,
            PM_VALIDATION_OBJECT_INSTANCE, value, value->layers};
}

Route route(PmDevice value) noexcept {
    if (value == nullptr || value->dispatch != &loader_dispatch) return {};
    return {value->driver, value->native, value->native, Scope::device,
            PM_VALIDATION_OBJECT_DEVICE, value, value->layers};
}

Route route(PmQueue value) noexcept {
    if (value == nullptr || value->dispatch != &loader_dispatch) return {};
    return {value->driver, value->native, value->native_device, Scope::device,
            PM_VALIDATION_OBJECT_QUEUE, value, value->layers};
}

Route route(PmCommandBuffer value) noexcept {
    if (value == nullptr || value->dispatch != &loader_dispatch) return {};
    return {value->driver, value->native, value->native_device, Scope::device,
            PM_VALIDATION_OBJECT_COMMAND_BUFFER, value, value->layers};
}

} // namespace parallel_mater::loader
