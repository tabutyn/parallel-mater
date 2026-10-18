# Parallel Mater Gallery for Android

This native APK uses the shared gallery scene builder, public command API,
loader, validation layer, and software renderer. Android and iPad share one C++
`MobileGallerySession` for controls, scene/reset logic, surface admission,
failure recovery, budgets and frame timing, plus one C++ qualification
coordinator and report schema. Java/JNI retain only Android lifecycle, semantic
input translation, surface presentation and atomic report storage. The default build bundles the
same loadable Vulkan driver and common Slang physics as the desktop preview.
Vulkan physics runs on the phone; rendering currently copies results into the
software renderer. This is not yet native GPU-renderer interoperability.
An explicit `-PphysicsBackend=cpu` build retains the experimental Slang CPU
preview. There is no silent CPU fallback when the Vulkan driver is unavailable.

The picker always shows the common 28-scene catalog and annotates unsupported
records with the exact missing backend feature. The current Vulkan driver
advertises the first 15 records: rigid bodies and constraints, Dump, Fluid,
Fluid + Rigid, Peg Paint, Cloth, Cloth Tear, Cloth Paint, and Water Cloth. Rope,
soft bodies, smoke, and their couplings remain unavailable. The new cloth set
has desktop Vulkan evidence; Pixel qualification remains a separate roadmap
gate.

## Device workload resource

The APK bundles `config/device-profiles.json`. Native startup selects a
`startup_limits` record by platform, Android `Build.MODEL`, and physics backend.
Exact device matches beat the platform fallback; an exact backend breaks ties.

The Pixel 7 preview selects four bricks, 128 fluid particles, four moving bodies
in the liquid scene, at most 640×360 pixels, and a 30 FPS target. Unknown Android
devices use 64 fluid particles and two moving bodies in liquid scenes.
Both are conservative, unmeasured starting budgets, separate from
`measurements` and `verified_profiles`. They are not sustained performance
claims. Smoke and dump limits are reserved for backends/scenes that support
them; they do not enable those scenes here.

Fluid and coupled-body budgets are applied before simulation allocation. Static
and kinematic scenery is retained; a deterministic subset of independent dynamic
bodies is retained. Reduction rejects indexed attachments instead of silently
invalidating their body references.

Fluid capacity is applied before simulation allocation. The shared scene builder
samples the authored initial population down to capacity, and emitters retain
that capacity. Lower counts change the physical amount and sampling of liquid;
this is not a quality-equivalent benchmark against the full desktop workload.
Solver timestep, material parameters and contact iteration quality are retained.
The common Gallery clock advances fixed 1/target_fps steps with approximately
240 solver substeps/second. Slow display frames may catch up by at most four
steps and only within the shared 8 ms physics-work budget; excess whole steps
are discarded so rendering and input cannot accumulate an unbounded queue.
Pause, backgrounding, surface loss, and profile changes reset elapsed time. The
UI schedules one worker frame from display callbacks; it does not implement a
second target-FPS clock.

## Build and install

Vulkan requires Android 13/API 33 or newer and the GPU features listed in
[the Vulkan driver documentation](../../docs/VULKAN.md). The CPU build supports
API 29 or newer. Both install under the same app ID.

Requirements: JDK 17+ (CI uses 21), Android SDK platform 36, NDK 29.0.14206865, SDK CMake
4.1.2, and host Python/CMake build tools. Gradle 9.1.0 and Android Gradle Plugin
9.0.1 are pinned. Slang is the repository's pinned host compiler. The NDK build
does not require CUDA or GLFW. Native libraries use the NDK's 16 KB-compatible
alignment defaults.

From the repository root:

```sh
export ANDROID_HOME="$HOME/Android/Sdk"
export JAVA_HOME=/path/to/jdk-21
export SLANGC=/path/to/slang/bin/slangc # optional; otherwise CMake fetches it
./scripts/Build-AndroidGallery.sh
"$ANDROID_HOME/platform-tools/adb" -d install -r \
  examples/android/app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/platform-tools/adb" -d shell am start \
  -n org.parallelmater.gallery/.GalleryActivity
```

The default build is ARM64 with Vulkan physics. Use `-PphysicsBackend=cpu`
for the CPU build. Set `ANDROID_ABIS=arm64-v8a,x86_64` to include an
emulator binary. For an existing dependency checkout, pass
`-PcgltfSource=/path/to/cgltf` to the script. Debug APKs are locally signed and
include optimized native code. Builds default to one compile job, a 1 GiB Gradle
heap, and ARM64 in the build script to limit workstation memory pressure. No account, Internet permission, or server is
required at runtime.

The simulation fills the screen beneath compact overlay controls. Tap the scene
name to open the scene picker. Drag to orbit, pinch or use +/- to zoom; Focus
reframes the liquid particles and moving bodies, or restores the other scenes'
authored views. Hold arrow buttons for the displayed action: tilt gravity, push
an authored force, or drive and steer a motorized scene. Their accessibility
labels come from the same common semantic control metadata as iPad. Orbiting
bodies expose Release and Restore; Dump exposes Raise bed and Lower bed. Reset
reloads the scene. Info contains backend, profile limits, frame-work averages,
and one copy of the scene requirements and gesture help; the main view shows
only population and paused state. Backgrounding stops frame scheduling and clears held movement. Physics
and rendering run on a worker thread; Android UI controls remain responsive.

For diagnostic logs:

```sh
"$ANDROID_HOME/platform-tools/adb" -d logcat -s ParallelMater AndroidRuntime
```

## Device qualification

The mobile CI workflow builds the Vulkan APK and unsigned iPad app. It also runs
an x86_64 Slang CPU APK on a Pixel 6 phone emulator. The emulator taps every
declared control family, checks accessible names and user-facing language, and
uploads screenshots, UI hierarchies, and a revision-tagged summary. This proves
Android control reachability and presentation independently of Vulkan device
qualification. On a phone
with the new debug APK installed, run shared sustained/control and actual OS
background-recovery checks with:

```sh
python3 scripts/Test-AndroidGallery.py --serial PHONE_SERIAL --frames 120
```

To repeat only the phone UI evidence against an installed debug APK:

```sh
python3 scripts/Test-AndroidGalleryUi.py --serial PHONE_SERIAL
```

Reports use `parallel-mater-mobile-gallery-qualification/v1` and include the
compiled identity, renderer/readback mode, common lifecycle and control checks,
failures, all scenes, and a fresh nonce; old or platform-specific reports cannot
satisfy a new run. The same C++ scenario and fields run on iPad. This does not
replace the native API suites below or qualify thermal/physical accuracy.
Release APKs ignore diagnostic intents and do not write these reports.

Build existing public-API tests as Android executables alongside the APK:

```sh
./scripts/Build-AndroidGallery.sh -PphysicsBackend=vulkan -Pqualification=true
python3 scripts/Test-AndroidVulkan.py --serial PHONE_SERIAL \
  --build-directory examples/android/app/.cxx/Debug/BUILD_HASH/arm64-v8a
```

Use the CMake directory whose cache has `PARALLEL_MATER_ANDROID_QUALIFICATION=ON`.
The runner stages only these tests, their libraries, two scenes and profile data
in `/data/local/tmp/parallel-mater-qualification`. Command/API, asynchronous
submission and analytic/bounded liquid/reset checks execute on the phone.
Logs and results are saved in ignored `out/android/native-vulkan-tests/`.
These tests enable the Parallel Mater command validation layer; they do not
implicitly install or enable the Khronos Vulkan validation layer.

Debug APKs accept `pm_scene` (0–27) and `pm_action` (`reset`, `pause`, `resume`,
`focus`, `info`, `freeze`, `unfreeze`) extras for repeatable UI testing. Freeze
stops presentation callbacks so Android accessibility can observe an idle
window; it does not add or replace a physics control. Release builds ignore them.
They invoke the same worker commands as the visible controls:

```sh
adb -s PHONE_SERIAL shell am start -f 0x20000000 \
  -n org.parallelmater.gallery/.GalleryActivity --ei pm_scene 7
```

Every 60 running frames, `FrameWindow` log records report mean/median/p95 frame
work, mean physics and rendering work, observed frame-start cadence, population,
scene and actual backend. These are measurements at the declared small workload,
not quality-equivalent desktop benchmarks. Vulkan pipeline cache files live in
the app's private cache directory and are checked against the device identity.

## Vulkan status (2026-10-16)

ARM64 packaging builds successfully with the same public loader and shared Slang
Vulkan driver. APK inspection confirms Vulkan is the only bundled physics driver,
with no CUDA runtime dependency. Pixel 7/Mali-G710 reports Vulkan 1.4.343 and the
required features, including 256-byte push constants, `storagePushConstant8`,
`maintenance4`, physical buffer addresses and 64-bit buffer atomics.

The initial command/API test crashed inside Mali's SPIR-V compiler
(`SPIR_MBS2Helper::apply_decorations`). A minimal 1496-byte shader reproduced
the failure. Ordering acyclic pointer types before use resolves it without
changing physics instructions or memory layouts; see the
[Vulkan workaround](../../docs/VULKAN.md#mali-pointer-declaration-workaround).

With normalized shaders, Pixel 7 passes the earlier three native suites:
command/API, asynchronous submission, and analytic liquid free fall plus bounded
Fluid and FluidRigid scenes/reset. That run covered six modules and eight scenes.
The driver now builds nine modules and advertises cloth, paint, and liquid–cloth
coupling after desktop qualification. The expanded set still needs the F7-18
Pixel run. Scene replacement releases the old GPU simulation first to avoid
overlapping memory footprints. Performance remains below target; see the
[qualification record](../../docs/ANDROID-QUALIFICATION.md) for prior scene
timings, memory observations and remaining gates.

## Earlier CPU validation

Host tests cover hardware-profile matching, malformed numeric limits,
preservation of limits when saving a verified profile, analytic liquid free
fall across dispatch boundaries, capped Fluid and FluidRigid gallery scenes,
and reset. These require `BUILD_TESTING`, `PARALLEL_MATER_BUILD_GALLERY`,
`PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER`, and
`PARALLEL_MATER_SLANG_CPU_LIQUIDS` enabled. Run CTest with
`-R 'gallery-workload|mobile-liquid|async-submission'`.

APK compilation and host tests do not establish phone stability or GPU
performance. Record phone runtime evidence separately before promoting budgets
to measured or verified profiles.

Pixel 7 smoke test (Android 17, 2026-10-16): installed and launched the ARM64
APK; startup selected `pixel-7-preview`. Rigid Body, Fluid, and Fluid + Rigid
rendered; both liquid scenes reported 128/128 particles. Pause/resume, orbit,
reset, and background/foreground worked without Gallery errors after correcting
Android's escaped manifest path separators. The other five scenes and sustained
thermal behavior have not been qualified on the phone.

Initial frame-work samples were 16.1 ms for Rigid Body, 19.9 ms for Fluid,
and 156.1–194.5 ms for Fluid + Rigid with 65 rigid bodies. The coupled-body
budget now limits that scene to four moving bodies plus its static geometry
on Pixel 7. A follow-up run averaged 36.9 ms over its first 60 running frames
at 640x288. These observations are not a matched benchmark or a sustained FPS
qualification. Local screenshots and the tested APK are saved under the
ignored `out/android/` directory.
