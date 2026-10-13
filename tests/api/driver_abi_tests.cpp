// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_driver.h>

#include <cstdio>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
using Library = HMODULE;
static Library open_library(const char *path) { return LoadLibraryA(path); }
static void *find_symbol(Library value, const char *name) {
    return reinterpret_cast<void *>(GetProcAddress(value, name));
}
static void close_library(Library value) { if (value) FreeLibrary(value); }
#else
#include <dlfcn.h>
using Library = void *;
static Library open_library(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void *find_symbol(Library value, const char *name) { return dlsym(value, name); }
static void close_library(Library value) { if (value) dlclose(value); }
#endif

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main(int argc, char **argv) {
    CHECK(argc == 3);
    Library driver = open_library(argv[1]);
    CHECK(driver != nullptr);
    auto negotiate = reinterpret_cast<PFN_pmDriverNegotiateLoaderInterfaceVersion>(
        find_symbol(driver, "pm_driverNegotiateLoaderInterfaceVersion"));
    auto get_instance = reinterpret_cast<PFN_pmDriverGetInstanceProcAddr>(
        find_symbol(driver, "pm_driverGetInstanceProcAddr"));
    CHECK(negotiate != nullptr && get_instance != nullptr);
    CHECK(find_symbol(driver, "pmCreateInstance") == nullptr);
    std::uint32_t version = UINT32_MAX;
    CHECK(negotiate(&version) == PM_SUCCESS);
    CHECK(version == PM_DRIVER_INTERFACE_VERSION);
    CHECK(get_instance(nullptr, "pmCreateInstance") != nullptr);
    CHECK(get_instance(nullptr, "pmQueueSubmit") == nullptr);

    Library loader = open_library(argv[2]);
    CHECK(loader != nullptr);
    CHECK(find_symbol(loader, "pmCreateInstance") != nullptr);
    CHECK(find_symbol(loader, "pm_driverGetInstanceProcAddr") == nullptr);
    close_library(loader);
    close_library(driver);
    std::puts("PASS: installable driver ABI and loader symbol ownership");
    return 0;
}
