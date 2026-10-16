// SPDX-License-Identifier: MIT
#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>
#import <Metal/Metal.h>
#include <parallel_mater_gallery/session.hpp>
#include <sys/utsname.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

using parallel_mater::gallery::GallerySession;
namespace {
NSString *text(const std::string &value) { return [NSString stringWithUTF8String:value.c_str()]; }
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

    _sceneButton = [self button:@"RIGID BODY ▾" action:nil];
    NSMutableArray<UIMenuElement *> *scenes = [NSMutableArray array];
    const auto names = GallerySession::scene_names();
    __weak PMGalleryController *weakSelf = self;
    for (unsigned i = 0; i < names.size(); ++i) {
        NSString *name = text(names[i]);
        [scenes addObject:[UIAction actionWithTitle:name image:nil identifier:nil handler:^(UIAction *) {
            PMGalleryController *owner = weakSelf;
            if (!owner) return;
            owner->_selection = i; owner->_failed = NO;
            owner->_releaseButton.enabled = i == 2;
            [owner->_sceneButton setTitle:[name stringByAppendingString:@" ▾"] forState:UIControlStateNormal];
        }]];
    }
    _sceneButton.menu = [UIMenu menuWithTitle:@"Apple GPU Preview" children:scenes];
    _sceneButton.showsMenuAsPrimaryAction = YES;
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
    const auto names = GallerySession::scene_names();
    for (unsigned i = 0; i < names.size(); ++i) {
        @autoreleasepool {
            try {
                _session->select(i);
                const auto statistics = _session->qualify(steps);
                [results addObject:@{@"scene":text(names[i]), @"steps":@(steps),
                    @"mean_ms":@(statistics.mean_ms), @"p95_ms":@(statistics.p95_ms),
                    @"timing_window_frames":@(statistics.count), @"session_controls":@YES,
                    @"passed":@YES, @"details":text(_session->details())}];
                NSLog(@"ParallelMater smoke PASS: %@", text(names[i]));
                [UIImagePNGRepresentation(frame_image(*_session)) writeToURL:
                    [directories.firstObject URLByAppendingPathComponent:@"last-smoke-frame.png"] atomically:YES];
            } catch (const std::exception &error) {
                [results addObject:@{@"scene":text(names[i]), @"passed":@NO, @"error":text(error.what())}];
                NSLog(@"ParallelMater smoke FAIL: %@: %@", text(names[i]), text(error.what()));
            }
            write_report(@"smoke-report.json", @{
                @"revision":text(GallerySession::build_revision()), @"backend":text(_session->backend_name()),
                @"device":text(_session->device_name()), @"nonce":argument(@"--deployment-nonce") ?: @"",
                @"completed":@(results.count == names.size()), @"frames_per_scene":@(steps),
                @"scope":@"scene output and shared controls; not numerical, thermal, or full physics qualification",
                @"scenes":results});
        }
    }
    NSLog(@"ParallelMater smoke complete: %lu scenes", (unsigned long)results.count);
    _session->select(0);
    [self writeRuntimeInfo];
}
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
