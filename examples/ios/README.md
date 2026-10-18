# ParallelMater Gallery for iPad

Native UIKit development preview using the same public loader, validation layer,
shared Gallery scene builder, and **Slang Metal physics driver** as macOS.
Physics kernels execute on the iPad's Apple GPU. There is no CPU physics
fallback. A shared Metal graphics renderer consumes the public GPU output
buffer directly and presents through `CAMetalLayer`. Normal drawing does not
copy physics positions or rendered images to the CPU. Explicit diagnostic
captures and the Fixed scene's CPU controller still read state. No desktop
streaming or runtime network is needed.

The Android app informed the worker-thread lifecycle, touch controls, bounded
frame scheduling, and shared device-profile budgets. iPad does not use Android's
CPU physics adapter or duplicate the physics equations.

## Build and install

Requires macOS with Xcode and its iOS/Metal toolchains, CMake 3.25+, the pinned
host Slang compiler, a development signing team, and a paired iPad with Developer
Mode enabled. This build requires **iPadOS 26+** because shared C++ scene parsing
uses the system floating-point `std::from_chars` implementation. The Metal
driver requires Apple GPU family 7 or newer; older devices fail explicitly.

From the repository root:

```sh
export PM_APPLE_TEAM=YOUR_TEAM_ID
export SLANGC=/path/to/slang/bin/slangc # optional; CMake can fetch the pinned host compiler
bash scripts/Build-IPadGallery.sh
xcrun devicectl list devices
python3 scripts/Deploy-Gallery.py ipad --device DEVICE_ID
```

Unlock the iPad before launching. Xcode handles development signing; do not
store certificates, provisioning profiles, or private keys in the repository.
`PM_IPAD_BUILD_DIR` overrides the build directory; `PM_CGLTF_SOURCE` can point to
an existing cgltf checkout. The generated Xcode project can also be opened to
run/debug on a selected device. Frameworks, GLBs, profiles, manifests, and 136
compiled Metal shader entries are bundled and signed; runtime does not access
the workstation or its removable volumes.

The build uses the `ipad-metal-preview` CMake preset. `ipad-ci` builds the same
app without signing for CI. The approved logo is compiled into the AppIcon
asset catalog. Deployment rejects an old revision or dirty build by default;
`--allow-dirty` explicitly permits a development iteration. It verifies the
running app's compiled revision and actual backend using a fresh launch nonce,
not a stale file from an earlier installation. A locked iPad is an installation
success but a verification failure, never a claimed successful launch.

For Mac: configure/build/test `macos-metal-preview`, then run
`python3 scripts/Deploy-Gallery.py macos`. The wrapper checks the installed
bundle and runs a native-window direct-GPU smoke test. Linux/Vulkan and
Windows/D3D12 presets provide equivalent explicit backend selection; native
runtime qualification still needs suitable hardware. See the mobile build
workflow for Android's pinned Gradle/SDK/NDK configuration.

## Scenes and controls

The picker exposes all 28 catalog scenes: rigid bodies and constraints; fluid, cloth,
soft-body, rope, smoke, paint, and the currently implemented mixed-domain scenes.
Existing scene availability does not establish complete physics qualification
or add missing roadmap interactions.

- Tap the scene name to switch; Pause/Resume controls simulation.
- Drag to orbit; pinch to zoom; Focus restores the initial view.
- Hold an arrow to push authored objects or tilt gravity.
- The shared primary-action button releases/restores Point constraints and
  raises/lowers the Dump bed; Reset reloads the scene.
- Info reports the actual device, profile, revision, and shared frame timings.

Only one worker frame can be pending. Backgrounding stops scheduling and clears
held input. A completed worker frame can still finish during the transition.
Android and iPad use one C++ `MobileGallerySession`, including controls,
scene/reset logic, surface admission, failure recovery, workload selection and
`FrameTimingCollector`. Native adapters own only lifecycle, semantic input
translation, presentation and reporting. Metal drawable acquisition/presentation is
recorded under presentation; GPU command completion is render work. These are
host wall-clock measurements, neither GPU-kernel durations nor scan-out FPS.
Physics and graphics are explicitly fenced before the shared output buffer is
reused: this removes the rendering readback, not all host waits or solver-internal
readbacks. Multiple frames in flight and external engine imports remain separate work.

The provisional `ipad-metal-preview` profile in `config/device-profiles.json`
uses four bricks, 256 liquid particles, four independent dynamic bodies in the
Fluid + Rigid scene, 128 smoke particles, at most 960×720 pixels, and a 30-update/s
target. Indexed mixed-scene attachments are retained. These budgets are not
measured performance guarantees or quality-equivalent desktop benchmarks.
Each update advances 1/30 second with eight substeps, preserving Android's 240
substeps/second. Slow frame work slows simulation rather than building a queue.

## Validation

On a Mac Metal preview build, run:

```sh
ctest --test-dir build-macos-metal-preview --output-on-failure \
  -R 'ipad-session|gallery-workload'
```

The session test loads all 28 shared catalog scenes supported by the queried
Metal device with software and native Metal
rendering, checks finite domain output and image dimensions, and exercises
pause, reset, timing-history reset, invalid selection, and Point release/restore.
It also checks Android's workload budget against the same catalog, plus
image-readback exclusion on ordinary GPU frames. This tests shared code with
the Mac GPU, not Android hardware or UIKit.

For an on-device diagnostic run:

```sh
python3 scripts/Deploy-Gallery.py ipad --device DEVICE_ID --qualify 120
```

The app records all 28 catalog entries, runs 120 frames for each available
scene, records exact capability reasons for unavailable scenes, validates
output/images, and checks shared surface replacement, pause, inactive/resume,
presentation-failure recovery and reset behavior. The host then backgrounds Gallery
using Settings and returns to Gallery, checking actual OS lifecycle counters and
that simulation did not advance while inactive. Reports are nonce-bound and
saved under `out/gallery-deployment/`. `--smoke-test` remains a short three-frame
variant. The app writes `Documents/smoke-report.json`, `runtime-info.json` and
`last-smoke-frame.png`; report schema now includes identity and a `scenes` array.
These bounded sustained/control checks do not qualify touch accuracy, thermal
behavior, numerical agreement or all missing interactions.

The separate public-API physics runner is available only when
`PARALLEL_MATER_ENABLE_IOS_QUALIFICATION=ON`; the development preview preset
enables it, while ordinary iOS builds default it off. Run and collect it with:

```sh
python3 scripts/Deploy-Gallery.py ipad --device DEVICE_ID \
  --physics-qualification --timeout 3600
```

This path does not reuse Gallery scene results. It executes the canonical
30-case registry through the public `pm*` API, the fixed 300-frame/eight-brick
independent penetration oracle, the equal-fluid quality fixture, and both
adverse lifecycle fixtures. Metal currently does not advertise separate-system
equal-fluid coupling, so that fixture records `unsupported`. The adverse
lifecycle operations execute, but remain `missing-checkpoints` until they emit
the exact F5-03 state fingerprints and write ordinals; a console pass is not
promoted to lifecycle qualification.

The app writes 35 raw artifacts under
`Documents/physics-qualification/` and updates a nonce-bound manifest only
after all attempts finish. The host verifies the installed app executable and
embedded Metal-driver SHA-256 against the staged bundle, source revision,
artifact hashes, public iPad model, iPadOS/Metal and compiler/Slang metadata.
It rejects stale, partial, mismatched-revision, binary-mismatched and
Gallery-only reports and rejects private device-identifier fields. Collected
files remain ignored under `out/gallery-deployment/` until a roadmap evidence
task explicitly reviews and registers them.
The device manifest records the runtime Metal generation and highest supported
Apple GPU family separately from the GPU name; host collection requires both.

2026-10-16 evidence: the original `43dcfac` iPad preview launched after unlocking
and passed all 27 scenes × three frames on the connected Apple M1 iPad Pro
12.9-inch (5th generation), `iPad13,8`, iPadOS 26.6.2. The shared-session/icon
development build based on `051b50c` subsequently passed 27 × 120 frames plus
shared control checks using software drawing. This is separate from native Metal
graphics qualification. Mac regression tests pass 36/36, including native Metal
rendering across all 27 mobile scenes. These are explicitly preview results;
no mobile performance profile or full physics qualification was promoted.

The integration build combining `e89cf1a` with upstream `2625740` then passed
**27 × 120 native Metal-rendered frames** on the same iPad, plus shared
pause/inactive/resume/reset checks. Per-scene reported process footprints peaked
at 306.4 MiB. Testing caught a mixed-language Xcode target silently omitting ARC:
the renderer leaked roughly 1 GiB per scene and was terminated. ARC is now an
explicit Xcode setting and a compile-time requirement. Completed command-buffer
autoreleases are also bounded independently of the caller's outer test loop.

The merged Mac build passes **38/38 CTest checks**, the portable no-driver build
passes **3/3**, and all 27 shared scenes pass with Metal API/shader validation.
A six-frame native-window check produces six CSV samples and an upright image.
These are bounded Gallery integration checks, not the outstanding numerical,
thermal/performance, or complete-interaction qualification gates. Local reports
are saved under ignored `out/gallery-deployment/`; final deployment records the
exact installed revision separately from this pre-commit test build.

2026-10-17 F7-07 evidence: clean revision `342997a9` passed all **28 × 120**
direct-Metal Gallery frames on the same physical Apple M1 iPad. Before the scene
sweep, the app verified shared pause/resume, background admission, surface
replacement and deliberate presentation-failure recovery. The deployment tool
then performed a real Settings background/foreground cycle and proved the
simulation frame count stayed unchanged. Binary hashes and the scoped result
are tracked in
[`F7-07-mobile-shells.json`](../../docs/roadmap/evidence/F7-07-mobile-shells.json).
