#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#include <parallel_mater_conformance/runner.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

NSString *argument(NSString *name) {
    NSArray<NSString *> *arguments = NSProcessInfo.processInfo.arguments;
    const NSUInteger index = [arguments indexOfObject:name];
    if (index == NSNotFound || index + 1 >= arguments.count) return nil;
    return arguments[index + 1];
}

void write_status(NSURL *directory, int result) {
    NSDictionary *status = @{
        @"schema": @"parallel-mater-ipados-conformance-job/v1",
        @"completed": @YES,
        @"return_code": @(result),
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:status options:0
                                                      error:nil];
    [data writeToURL:[directory URLByAppendingPathComponent:@"status.json"]
             options:NSDataWritingAtomic error:nil];
}

int run_job(NSURL *directory) {
    NSFileManager *manager = NSFileManager.defaultManager;
    NSURL *documents = [manager URLsForDirectory:NSDocumentDirectory
                                        inDomains:NSUserDomainMask].firstObject;
    NSURL *jobs = [documents URLByAppendingPathComponent:@"conformance-jobs"
                                             isDirectory:YES];
    [manager removeItemAtURL:jobs error:nil];
    [manager createDirectoryAtURL:directory withIntermediateDirectories:YES
                       attributes:nil error:nil];

    NSURL *standardOutput = [directory URLByAppendingPathComponent:@"stdout.txt"];
    std::freopen(standardOutput.fileSystemRepresentation, "w", stdout);
    std::freopen(standardOutput.fileSystemRepresentation, "a", stderr);

    NSString *resources = NSBundle.mainBundle.resourcePath;
    NSString *driver = [resources stringByAppendingPathComponent:
        @"manifests/slang-metal.json"];
    NSString *layer = [resources stringByAppendingPathComponent:
        @"manifests/validation.json"];
    setenv("PM_DRIVER_FILES", driver.fileSystemRepresentation, 1);
    setenv("PM_LAYER_FILES", layer.fileSystemRepresentation, 1);
    if ([NSProcessInfo.processInfo.arguments containsObject:@"--pm-validation"])
        setenv("PM_TEST_VALIDATION", "1", 1);
    else
        unsetenv("PM_TEST_VALIDATION");

    std::vector<std::string> values{"parallel-mater-conformance"};
    NSArray<NSString *> *arguments = NSProcessInfo.processInfo.arguments;
    for (NSUInteger index = 1; index < arguments.count; ++index) {
        NSString *value = arguments[index];
        if ([value isEqualToString:@"--pm-job-token"]) {
            ++index;
            continue;
        }
        if ([value isEqualToString:@"--pm-validation"]) continue;
        if ([value isEqualToString:@"--root"] && index + 1 < arguments.count) {
            values.emplace_back("--root");
            values.emplace_back(resources.fileSystemRepresentation);
            ++index;
            continue;
        }
        if ([value isEqualToString:@"--output"] && index + 1 < arguments.count) {
            values.emplace_back("--output");
            values.emplace_back([[directory URLByAppendingPathComponent:@"output"
                                                   isDirectory:YES]
                fileSystemRepresentation]);
            ++index;
            continue;
        }
        values.emplace_back(value.UTF8String ?: "");
    }

    std::vector<char *> command;
    command.reserve(values.size());
    for (std::string &value : values) command.push_back(value.data());
    const int result = parallel_mater::conformance::conformance_command_main(
        static_cast<int>(command.size()), command.data());
    std::fflush(stdout);
    std::fflush(stderr);
    write_status(directory, result);
    return result;
}

} // namespace

@interface PMConformanceDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow *window;
@end

@implementation PMConformanceDelegate
- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)options {
    (void)options;
    application.idleTimerDisabled = YES;
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController *controller = [UIViewController new];
    controller.view.backgroundColor = UIColor.blackColor;
    self.window.rootViewController = controller;
    [self.window makeKeyAndVisible];

    NSString *token = argument(@"--pm-job-token");
    if (!token.length) return NO;
    NSURL *documents = [NSFileManager.defaultManager
        URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
    NSURL *directory = [[documents URLByAppendingPathComponent:
        @"conformance-jobs" isDirectory:YES]
        URLByAppendingPathComponent:token isDirectory:YES];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        const int result = run_job(directory);
        dispatch_async(dispatch_get_main_queue(), ^{
            UIApplication.sharedApplication.idleTimerDisabled = NO;
            exit(result);
        });
    });
    return YES;
}
@end

int main(int argc, char **argv) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil,
                                 NSStringFromClass(PMConformanceDelegate.class));
    }
}
