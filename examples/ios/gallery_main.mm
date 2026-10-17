// SPDX-License-Identifier: MIT
#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>
#import <Metal/Metal.h>
#include <parallel_mater_gallery/session.hpp>
#if PM_IOS_QUALIFICATION_RUNNER
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#endif
#include <sys/utsname.h>
#include <unistd.h>
#include <mach/mach.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>

using parallel_mater::gallery::GallerySession;
namespace {
NSString *text(const std::string &value) { return [NSString stringWithUTF8String:value.c_str()]; }
std::uint64_t memory_footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS
        ? info.phys_footprint : 0;
}
NSString *argument(NSString *name) {
    NSArray<NSString *> *args = NSProcessInfo.processInfo.arguments;
    NSUInteger index = [args indexOfObject:name];
    return index != NSNotFound && index + 1 < args.count ? args[index + 1] : nil;
}
void write_report(NSString *name, id object) {
    NSError *error = nil;
    NSURL *directory = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
    NSData *data = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted error:&error];
    if (!data || ![data writeToURL:[directory URLByAppendingPathComponent:name] options:NSDataWritingAtomic error:&error])
        throw std::runtime_error(std::string("Writing device report: ") + error.localizedDescription.UTF8String);
}
#if PM_IOS_QUALIFICATION_RUNNER
void write_text_report(NSString *name, const std::string &contents) {
    NSError *error = nil;
    NSURL *directory = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory
                                                             inDomains:NSUserDomainMask].firstObject;
    NSURL *path = [directory URLByAppendingPathComponent:name];
    if (![NSFileManager.defaultManager createDirectoryAtURL:path.URLByDeletingLastPathComponent
                                withIntermediateDirectories:YES attributes:nil error:&error])
        throw std::runtime_error(std::string("Creating report directory: ") +
                                 error.localizedDescription.UTF8String);
    NSData *data = [NSData dataWithBytes:contents.data() length:contents.size()];
    if (![data writeToURL:path options:NSDataWritingAtomic error:&error])
        throw std::runtime_error(std::string("Writing device report: ") +
                                 error.localizedDescription.UTF8String);
}
NSString *file_sha256(NSString *path) {
    return text(parallel_mater::conformance::sha256_file(
        std::filesystem::path(path.fileSystemRepresentation)));
}
NSString *metal_version(id<MTLDevice> device) {
    if (@available(iOS 26.0, *)) {
        if ([device supportsFamily:MTLGPUFamilyMetal4]) return @"Metal 4";
    }
    if (@available(iOS 16.0, *)) {
        if ([device supportsFamily:MTLGPUFamilyMetal3]) return @"Metal 3";
    }
    return @"pre-Metal 3";
}
NSString *metal_gpu_family(id<MTLDevice> device) {
    const struct { MTLGPUFamily family; __unsafe_unretained NSString *name; } families[] = {
        {MTLGPUFamilyApple10, @"Apple10"}, {MTLGPUFamilyApple9, @"Apple9"},
        {MTLGPUFamilyApple8, @"Apple8"}, {MTLGPUFamilyApple7, @"Apple7"},
        {MTLGPUFamilyApple6, @"Apple6"}, {MTLGPUFamilyApple5, @"Apple5"},
        {MTLGPUFamilyApple4, @"Apple4"}, {MTLGPUFamilyApple3, @"Apple3"},
        {MTLGPUFamilyApple2, @"Apple2"}, {MTLGPUFamilyApple1, @"Apple1"}};
    for (const auto &family : families)
        if ([device supportsFamily:family.family]) return family.name;
    return @"unknown";
}
#endif
UIImage *frame_image(const GallerySession &session) {
    // Shared renderer pixels are bottom-up RGBA. UIKit images are top-down.
    NSMutableData *data = [NSMutableData dataWithLength:session.pixels.size() * 4];
    auto *destination = static_cast<std::uint32_t *>(data.mutableBytes);
    for (unsigned y = 0; y < session.height; ++y)
        std::copy_n(session.pixels.data() + (session.height - 1 - y) * session.width,
                    session.width, destination + y * session.width);
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGColorSpaceRef color = CGColorSpaceCreateDeviceRGB();
    CGImageRef image = CGImageCreate(session.width, session.height, 8, 32,
        session.width * 4, color, CGBitmapInfo(kCGBitmapByteOrder32Big) | CGBitmapInfo(kCGImageAlphaNoneSkipLast),
        provider, nullptr, false, kCGRenderingIntentDefault);
    UIImage *result = [UIImage imageWithCGImage:image];
    CGImageRelease(image); CGColorSpaceRelease(color); CGDataProviderRelease(provider);
    return result;
}
}

@interface PMMetalView : UIView
@end
@implementation PMMetalView
+ (Class)layerClass { return CAMetalLayer.class; }
@end

@interface PMGalleryController : UIViewController <UIGestureRecognizerDelegate> {
    std::unique_ptr<GallerySession> _session;
    dispatch_queue_t _worker;
    PMMetalView *_image;
    UILabel *_status;
    UIButton *_sceneButton, *_pauseButton, *_releaseButton;
    CADisplayLink *_display;
    BOOL _ready, _pending, _active, _paused, _failed;
    float _orbitX, _orbitY, _zoom, _right, _up;
    NSInteger _selection;
    BOOL _reset, _focus, _release;
    NSString *_details;
    unsigned _backgrounds, _foregrounds;
    std::uint64_t _backgroundFrames;
    unsigned _renderWidth, _renderHeight;
}
@end

@implementation PMGalleryController
- (UIButton *)button:(NSString *)title action:(SEL)action {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
    UIButtonConfiguration *style = [UIButtonConfiguration filledButtonConfiguration];
    style.title = title;
    style.baseBackgroundColor = [UIColor colorWithWhite:0.12 alpha:0.9];
    style.baseForegroundColor = UIColor.whiteColor;
    style.cornerStyle = UIButtonConfigurationCornerStyleMedium;
    button.configuration = style;
    button.accessibilityLabel = title;
    if (action) [button addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
    return button;
}
- (void)installCatalog:(NSArray<NSDictionary *> *)catalog {
    NSMutableArray<UIMenuElement *> *scenes = [NSMutableArray array];
    __weak PMGalleryController *weakSelf = self;
    for (NSUInteger index = 0; index < catalog.count; ++index) {
        NSDictionary *item = catalog[index];
        const BOOL available = [item[@"available"] boolValue];
        NSString *name = item[@"name"];
        NSString *title = available ? name :
            [NSString stringWithFormat:@"%@ — %@", name, item[@"reason"]];
        UIAction *action = [UIAction actionWithTitle:title image:nil identifier:nil
            handler:^(UIAction *) {
                PMGalleryController *owner = weakSelf;
                if (!owner) return;
                owner->_selection = static_cast<NSInteger>(index);
                owner->_failed = NO;
                owner->_releaseButton.enabled = index == 2U;
                [owner->_sceneButton setTitle:[name stringByAppendingString:@" ▾"]
                                      forState:UIControlStateNormal];
            }];
        action.attributes = available ? 0 : UIMenuElementAttributesDisabled;
        [scenes addObject:action];
    }
    _sceneButton.menu = [UIMenu menuWithTitle:@"Apple GPU Preview" children:scenes];
    _sceneButton.showsMenuAsPrimaryAction = YES;
    if (catalog.count != 0U) {
        NSString *name = catalog[0][@"name"];
        [_sceneButton setTitle:[name stringByAppendingString:@" ▾"]
                      forState:UIControlStateNormal];
    }
}
- (void)viewDidLoad {
    [super viewDidLoad];
    UIApplication.sharedApplication.idleTimerDisabled = YES;
    _active = YES; _selection = -1;
    _worker = dispatch_queue_create("org.parallelmater.gallery.simulation", DISPATCH_QUEUE_SERIAL);
    self.view.backgroundColor = [UIColor colorWithWhite:0.075 alpha:1];
    _image = [PMMetalView new];
    _image.userInteractionEnabled = YES;
    _image.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:_image];
    [NSLayoutConstraint activateConstraints:@[
        [_image.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [_image.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [_image.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [_image.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]]];
    UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(orbit:)];
    pan.maximumNumberOfTouches = 1; pan.delegate = self;
    [_image addGestureRecognizer:pan];
    UIPinchGestureRecognizer *pinch = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(zoom:)];
    pinch.delegate = self; [_image addGestureRecognizer:pinch];

    _sceneButton = [self button:@"Loading scenes…" action:nil];
    _pauseButton = [self button:@"Pause" action:@selector(pause)];
    UIStackView *toolbar = [[UIStackView alloc] initWithArrangedSubviews:@[_sceneButton,
        _pauseButton, [self button:@"Reset" action:@selector(reset)],
        [self button:@"Focus" action:@selector(focus)], [self button:@"Info" action:@selector(info)]]];
    toolbar.spacing = 8; toolbar.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:toolbar];
    _status = [UILabel new]; _status.text = @"Preparing Apple GPU…";
    _status.textColor = UIColor.whiteColor;
    _status.font = [UIFont monospacedSystemFontOfSize:13 weight:UIFontWeightMedium];
    _status.numberOfLines = 2; _status.backgroundColor = [UIColor colorWithWhite:0.08 alpha:0.8];
    _status.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:_status];
    [NSLayoutConstraint activateConstraints:@[
        [toolbar.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:12],
        [toolbar.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:10],
        [toolbar.trailingAnchor constraintLessThanOrEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-12],
        [_status.leadingAnchor constraintEqualToAnchor:toolbar.leadingAnchor],
        [_status.topAnchor constraintEqualToAnchor:toolbar.bottomAnchor constant:8]]];

    NSMutableArray<UIView *> *controls = [NSMutableArray array];
    NSArray *titles = @[@"←", @"↓", @"↑", @"→"];
    for (NSInteger i = 0; i < 4; ++i) {
        UIButton *button = [self button:titles[i] action:nil]; button.tag = i;
        [button.widthAnchor constraintEqualToConstant:60].active = YES;
        [button.heightAnchor constraintEqualToConstant:52].active = YES;
        [button addTarget:self action:@selector(press:) forControlEvents:UIControlEventTouchDown];
        [button addTarget:self action:@selector(releaseDirections) forControlEvents:
            UIControlEventTouchUpInside | UIControlEventTouchUpOutside | UIControlEventTouchCancel | UIControlEventTouchDragExit];
        [controls addObject:button];
    }
    _releaseButton = [self button:@"Release / Restore" action:@selector(releaseConstraint)];
    _releaseButton.enabled = NO;
    [controls addObject:_releaseButton];
    UIStackView *bottom = [[UIStackView alloc] initWithArrangedSubviews:controls];
    bottom.spacing = 8; bottom.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:bottom];
    [NSLayoutConstraint activateConstraints:@[
        [bottom.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:12],
        [bottom.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-12]]];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(background)
        name:UIApplicationWillResignActiveNotification object:nil];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(foreground)
        name:UIApplicationDidBecomeActiveNotification object:nil];
    _display = [CADisplayLink displayLinkWithTarget:self selector:@selector(tick:)];
    _display.preferredFrameRateRange = CAFrameRateRangeMake(15, 30, 30);
    [_display addToRunLoop:NSRunLoop.mainRunLoop forMode:NSRunLoopCommonModes];
    dispatch_async(_worker, ^{
        @autoreleasepool {
            try {
                NSString *root = NSBundle.mainBundle.resourcePath;
                const auto resources = std::filesystem::path(root.fileSystemRepresentation);
                setenv("PM_DRIVER_FILES", (resources / "manifests/slang-metal.json").c_str(), 1);
                setenv("PM_LAYER_FILES", (resources / "manifests/validation.json").c_str(), 1);
                struct utsname machine{}; uname(&machine);
                self->_session = std::make_unique<GallerySession>(parallel_mater::gallery::GallerySessionConfig{
                    .assets = resources / "assets", .profiles = resources / "config/device-profiles.json",
                    .platform = "iPadOS", .model = machine.machine, .backend = "metal",
                    .native_backend = PM_NATIVE_BACKEND_METAL, .gpu_rendering = true});
                NSMutableArray<NSDictionary *> *catalog = [NSMutableArray array];
                for (const auto &item : self->_session->catalog()) {
                    [catalog addObject:@{
                        @"id":text(std::string(item.id)),
                        @"name":text(std::string(item.scene.name)),
                        @"description":text(std::string(item.scene.help)),
                        @"requirements":text(item.required_feature_names),
                        @"available":@(item.available),
                        @"reason":text(item.unavailable_reason)}];
                }
                NSArray<NSDictionary *> *catalogSnapshot = [catalog copy];
                dispatch_sync(dispatch_get_main_queue(), ^{
                    [self installCatalog:catalogSnapshot];
                });
                self->_session->set_presentation_layer((__bridge void *)self->_image.layer);
                NSLog(@"ParallelMater: %@", text(self->_session->details()));
                id<MTLDevice> gpu = (__bridge id<MTLDevice>)self->_session->native_render_device();
                const auto limits = self->_session->limits;
                dispatch_sync(dispatch_get_main_queue(), ^{
                    CAMetalLayer *layer = (CAMetalLayer *)self->_image.layer;
                    layer.device = gpu; layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
                    layer.framebufferOnly = YES;
                    layer.drawableSize = CGSizeMake(limits.render_width, limits.render_height);
                    self->_renderWidth = limits.render_width; self->_renderHeight = limits.render_height;
                });
                // Publish a real drawable before a long offscreen diagnostic.
                // The system must never wait for qualification to see first draw.
                self->_session->frame();
                [self writeRuntimeInfo];
                if ([NSProcessInfo.processInfo.arguments containsObject:@"--smoke-test"] ||
                    [NSProcessInfo.processInfo.arguments containsObject:@"--qualify"])
                    [self runSmokeTests];
#if PM_IOS_QUALIFICATION_RUNNER
                if ([NSProcessInfo.processInfo.arguments containsObject:@"--physics-qualification"])
                    [self runPhysicsQualification];
#endif
                dispatch_async(dispatch_get_main_queue(), ^{ self->_ready = YES; });
            } catch (const std::exception &error) {
                NSString *message = text(error.what());
                dispatch_async(dispatch_get_main_queue(), ^{ [self failure:message]; });
            }
        }
    });
}
- (void)tick:(CADisplayLink *)sender {
    (void)sender;
    if (!_ready || !_active || _pending || _failed) return;
    _pending = YES;
    const float x = _orbitX, y = _orbitY, z = _zoom, right = _right, up = _up;
    const BOOL paused = _paused, reset = _reset, focus = _focus, release = _release;
    const NSInteger selection = _selection;
    const unsigned width = std::max(1.0, self.view.bounds.size.width);
    const unsigned height = std::max(1.0, self.view.bounds.size.height);
    const float scale = std::min({1.0F, float(_renderWidth)/width, float(_renderHeight)/height});
    ((CAMetalLayer *)_image.layer).drawableSize = CGSizeMake(std::max(1U, unsigned(width*scale)), std::max(1U, unsigned(height*scale)));
    _orbitX = _orbitY = _zoom = 0; _reset = _focus = _release = NO; _selection = -1;
    dispatch_async(_worker, ^{
        @autoreleasepool {
            try {
                auto &session = *self->_session;
                if (selection >= 0) session.select(static_cast<unsigned>(selection));
                else if (reset) session.reset();
                if (selection >= 0 || reset) {
                    id<MTLDevice> gpu = (__bridge id<MTLDevice>)session.native_render_device();
                    dispatch_sync(dispatch_get_main_queue(), ^{ ((CAMetalLayer *)self->_image.layer).device = gpu; });
                }
                session.resize(width, height);
                if (focus) session.focus();
                if (release) session.release();
                session.orbit(x * session.height / height, y * session.height / height);
                session.zoom(z); session.steer(right, up); session.set_paused(paused);
                session.frame();
                NSString *status = text(session.status());
                NSString *details = text(session.details());
                dispatch_async(dispatch_get_main_queue(), ^{
                    self->_status.text = status;
                    self->_details = details; self->_pending = NO;
                });
            } catch (const std::exception &error) {
                NSString *message = text(error.what());
                dispatch_async(dispatch_get_main_queue(), ^{ self->_pending = NO; [self failure:message]; });
            }
        }
    });
}
- (void)failure:(NSString *)message {
    NSLog(@"ParallelMater error: %@", message); _failed = YES; _status.text = message;
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Gallery preview error"
        message:message preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}
- (void)info {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"ParallelMater · Apple GPU Preview"
        message:_details ? _details : @"Preparing shared Slang physics…" preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"Done" style:UIAlertActionStyleDefault handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}
- (void)pause { _paused = !_paused; [_pauseButton setTitle:_paused ? @"Resume" : @"Pause" forState:UIControlStateNormal]; }
- (void)reset { _reset = YES; _failed = NO; }
- (void)focus { _focus = YES; }
- (void)releaseConstraint { _release = YES; }
- (void)press:(UIButton *)button { _right = button.tag == 0 ? -1 : button.tag == 3 ? 1 : 0; _up = button.tag == 1 ? -1 : button.tag == 2 ? 1 : 0; }
- (void)releaseDirections { _right = _up = 0; }
- (void)orbit:(UIPanGestureRecognizer *)gesture {
    CGPoint delta = [gesture translationInView:_image]; _orbitX += delta.x; _orbitY += delta.y;
    [gesture setTranslation:CGPointZero inView:_image];
}
- (void)zoom:(UIPinchGestureRecognizer *)gesture { _zoom += std::log(gesture.scale) * 10; gesture.scale = 1; }
- (BOOL)gestureRecognizer:(UIGestureRecognizer *)a shouldRecognizeSimultaneouslyWithGestureRecognizer:(UIGestureRecognizer *)b { (void)a; (void)b; return YES; }
- (void)background {
    UIApplication.sharedApplication.idleTimerDisabled = NO;
    _active = NO; [self releaseDirections]; _orbitX = _orbitY = _zoom = 0;
    dispatch_async(_worker, ^{
        if (!self->_session) return;
        self->_session->set_active(false); ++self->_backgrounds;
        self->_backgroundFrames = self->_session->frames;
        [self writeRuntimeInfo];
    });
}
- (void)foreground {
    UIApplication.sharedApplication.idleTimerDisabled = YES;
    _active = YES;
    dispatch_async(_worker, ^{
        if (!self->_session) return;
        self->_session->set_active(true); ++self->_foregrounds;
        [self writeRuntimeInfo];
    });
}
- (void)writeRuntimeInfo {
    try {
        write_report(@"runtime-info.json", @{
            @"revision":text(GallerySession::build_revision()), @"backend":text(_session->backend_name()),
            @"pid":@(getpid()),
            @"device":text(_session->device_name()), @"nonce":argument(@"--deployment-nonce") ?: @"",
            @"backgrounds":@(_backgrounds), @"foregrounds":@(_foregrounds),
            @"active":@(_session->active()), @"frames":@(_session->frames), @"background_frames":@(_backgroundFrames),
            @"release_channel":@"development-preview", @"renderer":@"metal-direct-fenced"});
    } catch (const std::exception &error) { NSLog(@"Runtime report failed: %s", error.what()); }
}
- (void)shutdown {
    UIApplication.sharedApplication.idleTimerDisabled = NO;
    _active = NO; [_display invalidate];
    [NSNotificationCenter.defaultCenter removeObserver:self];
    dispatch_async(_worker, ^{ self->_session.reset(); });
}
- (void)runSmokeTests {
    NSMutableArray *results = [NSMutableArray array];
    const bool extended = [NSProcessInfo.processInfo.arguments containsObject:@"--qualify"];
    const NSInteger requested = argument(@"--qualification-frames") ? [argument(@"--qualification-frames") integerValue] : 120;
    if (extended && (requested < 1 || requested > 100000))
        throw std::runtime_error("--qualification-frames must be 1..100000");
    const unsigned steps = extended ? static_cast<unsigned>(requested) : 3;
    NSArray<NSURL *> *directories = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask];
    const auto &catalog = _session->catalog();
    for (unsigned i = 0; i < catalog.size(); ++i) {
        @autoreleasepool {
            const auto &item = catalog[i];
            if (!item.available) {
                [results addObject:@{
                    @"id":text(std::string(item.id)),
                    @"scene":text(std::string(item.scene.name)),
                    @"requirements":text(item.required_feature_names),
                    @"available":@NO, @"skipped":@YES,
                    @"reason":text(item.unavailable_reason)}];
                write_report(@"smoke-report.json", @{
                    @"revision":text(GallerySession::build_revision()),
                    @"backend":text(_session->backend_name()),
                    @"device":text(_session->device_name()),
                    @"nonce":argument(@"--deployment-nonce") ?: @"",
                    @"completed":@(results.count == catalog.size()),
                    @"frames_per_scene":@(steps),
                    @"scope":@"scene output and shared controls; not numerical, thermal, or full physics qualification",
                    @"scenes":results});
                continue;
            }
            try {
                NSLog(@"ParallelMater smoke START: %@, footprint=%llu MiB",
                      text(std::string(item.scene.name)), memory_footprint() / (1024ULL*1024));
                _session->select(i);
                const auto statistics = _session->qualify(steps);
                [results addObject:@{@"id":text(std::string(item.id)),
                    @"scene":text(std::string(item.scene.name)),
                    @"requirements":text(item.required_feature_names),
                    @"available":@YES, @"steps":@(steps),
                    @"mean_ms":@(statistics.mean_ms), @"p95_ms":@(statistics.p95_ms),
                    @"timing_window_frames":@(statistics.count), @"session_controls":@YES,
                    @"memory_footprint_bytes":@(memory_footprint()),
                    @"passed":@YES, @"details":text(_session->details())}];
                NSLog(@"ParallelMater smoke PASS: %@", text(std::string(item.scene.name)));
                [UIImagePNGRepresentation(frame_image(*_session)) writeToURL:
                    [directories.firstObject URLByAppendingPathComponent:@"last-smoke-frame.png"] atomically:YES];
            } catch (const std::exception &error) {
                [results addObject:@{@"id":text(std::string(item.id)),
                    @"scene":text(std::string(item.scene.name)), @"available":@YES,
                    @"passed":@NO, @"error":text(error.what())}];
                NSLog(@"ParallelMater smoke FAIL: %@: %@",
                      text(std::string(item.scene.name)), text(error.what()));
            }
            write_report(@"smoke-report.json", @{
                @"revision":text(GallerySession::build_revision()), @"backend":text(_session->backend_name()),
                @"device":text(_session->device_name()), @"nonce":argument(@"--deployment-nonce") ?: @"",
                @"completed":@(results.count == catalog.size()), @"frames_per_scene":@(steps),
                @"scope":@"scene output and shared controls; not numerical, thermal, or full physics qualification",
                @"scenes":results});
        }
    }
    NSLog(@"ParallelMater smoke complete: %lu scenes", (unsigned long)results.count);
    _session->select(0);
    [self writeRuntimeInfo];
}
#if PM_IOS_QUALIFICATION_RUNNER
- (void)runPhysicsQualification {
    NSString *nonce = argument(@"--deployment-nonce");
    if (!nonce.length)
        throw std::runtime_error("--physics-qualification requires --deployment-nonce");
    const auto root = std::filesystem::path(NSBundle.mainBundle.resourcePath.fileSystemRepresentation);
    NSMutableArray<NSDictionary *> *artifacts = [NSMutableArray array];
    NSMutableDictionary *summary = [@{
        @"schema":@"parallel-mater-ipados-qualification-run/v1",
        @"report_kind":@"public-api-conformance",
        @"gallery_scene_sweep":@NO,
        @"nonce":nonce,
        @"revision":text(GallerySession::build_revision()),
        @"backend":@"slang-metal",
        @"completed":@NO,
        @"artifacts":artifacts
    } mutableCopy];
    write_report(@"physics-qualification.json", summary);

    const auto save = [&](NSString *relative, const std::string &contents,
                          NSString *kind, NSString *status) {
        write_text_report(relative, contents);
        [artifacts addObject:@{@"path":relative, @"kind":kind, @"status":status,
            @"sha256":text(parallel_mater::conformance::sha256(contents))}];
        write_report(@"physics-qualification.json", summary);
    };

    save(@"physics-qualification/provenance.json",
         parallel_mater::conformance::conformance_device_provenance(root),
         @"device-provenance", @"completed");
    save(@"physics-qualification/capabilities.json",
         parallel_mater::conformance::conformance_advertised_capabilities(root),
         @"advertised-capabilities", @"completed");

    save(@"physics-qualification/quality/rigid-brick-penetration-v1.json",
         parallel_mater::conformance::conformance_rigid_brick_quality(root),
         @"physical-quality", @"completed");
    const auto fluidTemporary = std::filesystem::path(
        [NSTemporaryDirectory() stringByAppendingPathComponent:
            @"fluid-pair-quality.json"].fileSystemRepresentation);
    save(@"physics-qualification/quality/core-conservation-stability-v1.json",
         parallel_mater::conformance::conformance_fluid_pair_quality(fluidTemporary),
         @"physical-quality", @"unsupported");
    save(@"physics-qualification/lifecycle/adverse-capacity-three-family-v1.json",
         parallel_mater::conformance::conformance_adverse_lifecycle(root),
         @"adverse-lifecycle", @"missing-checkpoints");

    NSUInteger completed = 0, unsupported = 0, failed = 0;
    for (const std::string &caseId :
         parallel_mater::conformance::conformance_case_ids()) {
        @autoreleasepool {
            const auto execution =
                parallel_mater::conformance::execute_conformance_case(caseId, root);
            NSString *status = @"failed";
            if (execution.status == parallel_mater::conformance::ExecutionStatus::completed) {
                status = @"completed"; ++completed;
            } else if (execution.status == parallel_mater::conformance::ExecutionStatus::unsupported) {
                status = @"unsupported"; ++unsupported;
            } else {
                ++failed;
            }
            NSString *path = [NSString stringWithFormat:@"physics-qualification/cases/%@.json",
                              text(caseId)];
            save(path, execution.json, @"trajectory", status);
            NSLog(@"ParallelMater physics qualification %@: %@", status, text(caseId));
        }
    }

    NSString *executable = NSBundle.mainBundle.executablePath;
    NSString *driver = [NSBundle.mainBundle.bundlePath stringByAppendingPathComponent:
        @"Frameworks/ParallelMaterMetal.framework/ParallelMaterMetal"];
    id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
    struct utsname machine{}; uname(&machine);
    NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
    summary[@"binary_identity"] = @{
        @"app_executable_sha256":file_sha256(executable),
        @"driver_sha256":file_sha256(driver)};
    summary[@"device"] = @{
        @"model":text(machine.machine),
        @"gpu":metal.name ?: @"unknown",
        @"operating_system":@"iPadOS",
        @"operating_system_version":[NSString stringWithFormat:@"%ld.%ld.%ld",
            (long)version.majorVersion, (long)version.minorVersion,
            (long)version.patchVersion],
        @"metal_api":@"Metal",
        @"metal_version":metal_version(metal),
        @"metal_gpu_family":metal_gpu_family(metal)};
    summary[@"toolchain"] = @{
        @"compiler":@PM_IPAD_TOOLCHAIN_COMPILER,
        @"compiler_version":@PM_IPAD_TOOLCHAIN_VERSION,
        @"slang_version":@PM_IPAD_SLANG_VERSION};
    summary[@"trajectory"] = @{
        @"case_count":@(completed + unsupported + failed),
        @"completed":@(completed), @"unsupported":@(unsupported),
        @"failed":@(failed)};
    summary[@"physical_quality"] = @{
        @"rigid_brick":@"completed",
        @"equal_fluid":@"unsupported-not-advertised"};
    summary[@"adverse_lifecycle"] = @{
        @"status":@"executed-missing-checkpoints",
        @"reason":@"Public-API adverse fixtures execute, but do not yet emit F5-03 checkpoint fingerprints."};
    summary[@"completed"] = @YES;
    write_report(@"physics-qualification.json", summary);
    NSLog(@"ParallelMater public-API qualification complete: %lu completed, %lu unsupported, %lu failed",
          (unsigned long)completed, (unsigned long)unsupported, (unsigned long)failed);
}
#endif
- (void)dealloc { [_display invalidate]; [NSNotificationCenter.defaultCenter removeObserver:self]; }
@end

@interface PMGallerySceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow *window;
@end
@implementation PMGallerySceneDelegate
- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)options {
    (void)session; (void)options;
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *)scene];
    self.window.rootViewController = [PMGalleryController new];
    [self.window makeKeyAndVisible];
}
- (void)sceneDidDisconnect:(UIScene *)scene {
    (void)scene; [(PMGalleryController *)self.window.rootViewController shutdown];
}
@end
@interface PMGalleryDelegate : UIResponder <UIApplicationDelegate>
@end
@implementation PMGalleryDelegate
- (UISceneConfiguration *)application:(UIApplication *)application configurationForConnectingSceneSession:(UISceneSession *)session options:(UISceneConnectionOptions *)options {
    (void)application; (void)options;
    UISceneConfiguration *configuration = [[UISceneConfiguration alloc] initWithName:@"Gallery" sessionRole:session.role];
    configuration.delegateClass = PMGallerySceneDelegate.class;
    return configuration;
}
@end
int main(int argc, char **argv) {
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(PMGalleryDelegate.class)); }
}
