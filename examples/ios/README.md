# ParallelMater Gallery for iPad

Native UIKit development preview using the same public loader, validation layer,
shared Gallery scene builder, and **Slang Metal physics driver** as macOS.
Physics kernels execute on the iPad's Apple GPU. There is no CPU physics
fallback. The shared software renderer still draws the image on the CPU;
UIKit displays that image. No desktop streaming or runtime network is needed.

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
xcrun devicectl device install app --device DEVICE_ID \
  'build-ipad-gallery/Release-iphoneos/ParallelMater Gallery.app'
xcrun devicectl device process launch --device DEVICE_ID \
  com.ballrollergames.parallelmater.gallery
```

Unlock the iPad before launching. Xcode handles development signing; do not
store certificates, provisioning profiles, or private keys in the repository.
`PM_IPAD_BUILD_DIR` overrides the build directory; `PM_CGLTF_SOURCE` can point to
an existing cgltf checkout. The generated Xcode project can also be opened to
run/debug on a selected device. Frameworks, GLBs, profiles, manifests, and 136
compiled Metal shader entries are bundled and signed; runtime does not access
the workstation or its removable volumes.

## Scenes and controls

The picker exposes 27 scenes: rigid bodies and constraints; fluid, cloth,
soft-body, rope, smoke, paint, and the currently implemented mixed-domain scenes.
Dump Truck is omitted until touch vehicle controls are implemented. Existing
scene availability does not establish complete physics qualification or add
missing roadmap interactions.

- Tap the scene name to switch; Pause/Resume controls simulation.
- Drag to orbit; pinch to zoom; Focus restores the initial view.
- Hold an arrow to push authored objects or tilt gravity.
- Release / Restore acts on the Point scene; Reset reloads the scene.
- Info reports the actual device, profile, revision, and shared frame timings.

Only one worker frame can be pending. Backgrounding stops scheduling and clears
held input. A completed worker frame can still finish during the transition.
UI image handoff and display latency are excluded from the displayed host-work
timings; these are neither GPU-kernel durations nor scan-out FPS.

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

The session test loads all 27 exposed scenes, simulates and renders three
frames each, checks finite domain output and image dimensions, and exercises
pause, reset, timing-history reset, invalid selection, and Point release/restore.
This tests the shared iPad session with the Mac GPU, not UIKit or iPad hardware.

For an on-device diagnostic run:

```sh
xcrun devicectl device process launch --device DEVICE_ID --terminate-existing \
  --console com.ballrollergames.parallelmater.gallery --smoke-test
```

The app checks three rendered frames per exposed scene, logs each result, and
writes `Documents/smoke-report.json` and `Documents/last-smoke-frame.png` inside
its own container. It then returns to Rigid Body for interactive use. Reports
can be copied with `devicectl device copy from` or Finder file sharing. This is
a bounded startup/render smoke test, not sustained stability, touch-control,
thermal, numerical-reference, or full interaction qualification.

2026-10-16 validation: Release iPadOS build and deep code-signature verification
passed; all 136 Metal entries were bundled. Installation succeeded on the
connected iPad Pro 12.9-inch (5th generation), `iPad13,8`, iPadOS 26.6.2. Launch
was blocked by the device lock screen, so on-device physics and UIKit interaction
are **not yet verified**. The Mac regression suite passed all 36 tests, including
the 27-scene session check. No mobile performance profile was promoted.
