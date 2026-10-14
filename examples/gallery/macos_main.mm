// SPDX-License-Identifier: MIT
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#include <cstdlib>
#include <filesystem>
#include <string>

int parallel_mater_gallery_application(int argc, char **argv);

namespace {

void configure_manifest_paths() {
    @autoreleasepool {
        NSString *resources = NSBundle.mainBundle.resourcePath;
        if (resources == nil) return;
        const std::filesystem::path root(resources.fileSystemRepresentation);
        if (std::getenv("PM_DRIVER_FILES") == nullptr &&
            std::getenv("PM_DRIVER_PATH") == nullptr) {
            const std::string path = (root / "drivers.d").string();
            setenv("PM_DRIVER_PATH", path.c_str(), 1);
        }
        if (std::getenv("PM_LAYER_FILES") == nullptr &&
            std::getenv("PM_LAYER_PATH") == nullptr) {
            const std::string path = (root / "layers.d").string();
            setenv("PM_LAYER_PATH", path.c_str(), 1);
        }
    }
}

void show_startup_failure() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"ParallelMater Gallery could not start";
        alert.informativeText =
            @"No compatible Parallel Mater Slang driver may be installed. "
             "The Gallery uses the manifest-discovered driver API; it does "
             "not contain the retired Metal physics runtime.";
        alert.alertStyle = NSAlertStyleCritical;
        [alert addButtonWithTitle:@"OK"];
        [alert runModal];
    }
}

} // namespace

int main(int argc, char **argv) {
    configure_manifest_paths();
    const int result = parallel_mater_gallery_application(argc, argv);
    if (result != 0 && argc == 1) show_startup_failure();
    return result;
}
