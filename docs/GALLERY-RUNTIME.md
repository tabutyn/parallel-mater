# Common Gallery runtime contract

The Gallery is one application with desktop, Android, and Apple shells. Its
portable C++ runtime is an example client of the installed Parallel Mater API;
it is not part of the physics ABI. Every shell must observe the same scene and
control semantics when its selected driver advertises the required features.

## Current ownership

Every shell now delegates low-level scene lifecycle to
`parallel_mater::gallery::GalleryRuntime`. It owns GLB loading, scene creation,
physics resources, host-renderer creation, replacement, reset, stepping, frame
assembly, and teardown. Android and iPad reach it through the reusable
`GallerySession`; desktop keeps `DesktopGalleryState` for its current UI and
calibration data. Controllers, timing policy, catalog presentation, and native
renderers remain partly shell-specific and move behind common contracts in the
remaining F7 tasks.

| Responsibility | Current desktop owner | Current mobile owner | Required owner |
|---|---|---|---|
| Ordered scene identity, name, help, source, control policy, and feature requirements | `gallery_context.hpp`, plus desktop selection code | `gallery_context.hpp`, plus `GallerySceneSet` remapping in `session.cpp` | Common runtime catalog |
| Driver discovery, explicit backend choice, and capability query | Desktop startup and backend picker | Platform startup configuration; Android fixes Vulkan | Platform shell requests a backend; common runtime reports devices and scene availability |
| Asset and device-profile paths | Desktop options and compile-time resource paths | Android/iPad shell configuration | Platform shell supplies paths once |
| Workload limits and user-adjustable counts | Desktop options, calibration, and local profile logic | `GallerySession` device-profile selection | Common runtime policy; shell edits declared parameters |
| Asset loading and scene construction | `GalleryRuntime`, configured by desktop | `GalleryRuntime`, configured by `GallerySession` | Common runtime catalog and workload policy |
| Physics device, resources, command submission, scene replacement, reset, and teardown | `GalleryRuntime` through `DesktopGalleryState` | `GalleryRuntime` through `GallerySession` | Common runtime (done) |
| Fixed, point, motor, steering, dump-bed, kinematic, gravity, force, paint, and deformable controllers | Mostly desktop application loop with shared helper types | Gravity/force, fixed collector, and point release in `GallerySession` | Common runtime semantic commands |
| Fixed-step accumulation, catch-up limits, substeps, pause, and foreground recovery | Desktop application loop | One profiled step per `GallerySession::frame` | Common runtime clock and workload policy |
| Camera state and focus policy | `GalleryRuntime::camera`, driven by desktop input | `GalleryRuntime::camera`, driven by `GallerySession` touch methods | Common runtime, driven by semantic pointer/zoom commands |
| Host render-frame assembly | `GalleryRuntime::step` | `GalleryRuntime::step` | Common runtime (done) |
| Software, native GPU, or external-engine drawing | Desktop renderer/window path; optional Metal path | Software renderer or iPad Metal renderer | Renderer adapter |
| Graphics API resources, swapchain/layer binding, upload, and presentation | Desktop window adapter | Android surface/ANativeWindow or Apple presentation layer | Renderer adapter, with the platform shell supplying the native surface |
| Keyboard, touch, gamepad, lifecycle, window, intent, and accessibility events | GLFW/native desktop glue | Java/JNI or UIKit glue | Platform shell translating events into common commands |
| Catalog, parameter, timing, status, and error view models | Desktop overlays | Java/UIKit widgets and session strings | Common runtime data; platform shell chooses widgets and layout |
| Physics capture, validation snapshots, and qualification sequencing | Desktop application and shared debug helpers | `GallerySession` qualification helpers | Common runtime diagnostics; shell chooses destination and presentation |
| Filesystem writes, share sheets, logs, and platform telemetry | Desktop platform helpers | Java/UIKit | Platform shell |

## Runtime boundary

The common runtime owns exactly one selected backend and at most one live
scene. It owns all physics objects, controller state, workload policy, camera,
frame timing, diagnostics, and the render handoff. A renderer adapter may hold
graphics resources, but it must not select scenes, advance physics, or interpret
controls. A platform shell may discover paths and native surfaces and translate
events, but it must not create physics resources or implement a scene rule.

The common command vocabulary must cover:

- select scene by stable identity; reset the current scene;
- pause/resume and deactivate/activate the application;
- orbit, pan, zoom, focus, and directional input;
- primary scene action, including point release/restore and dump-bed toggle;
- editable scene parameters such as brick, fluid, and dump-payload counts;
- debug-view selection, capture, and qualification requests.

Keyboard keys, touch gestures, Android intents, and engine input systems map to
those commands outside the runtime. The runtime interprets a command according
to the selected catalog entry's control policy. It must never let one input
simultaneously drive an authored force, tilt gravity, move a kinematic body, or
steer a vehicle unless the catalog explicitly declares that combination.

## Observable behavior

Scene selection is transactional through validation and asset loading. An
unknown or unavailable identity leaves the live scene unchanged. On
memory-constrained devices the runtime may release the old device before it
creates the replacement; a later creation failure then produces an explicit
empty/error state and never presents stale output under the new selection.
Successful selection and reset clear frame counters, timing history, held
directional input, transient controllers, and renderer history. Reset retains
the selected backend, scene identity, and applied workload parameters.

Pause stops physics and controller advancement while allowing the current
frame and UI to render. Inactive/background state performs no physics,
rendering, or presentation, clears held input, and discards timing history.
Activation resets the wall-clock baseline so elapsed background time cannot
become catch-up work. The next unpaused frame advances according to the same
fixed-step policy used on every shell.

Each successful physics step publishes one coherent output generation.
Renderer adapters receive that generation, camera, dimensions, visual/debug
mode, and timing sink. A host adapter consumes `RenderSceneFrame`; a native GPU
adapter consumes public borrowed device views from the selected backend.
Borrowed views remain valid only for the documented frame/submission lifetime.
Interactive native rendering performs no host readback unless an explicit
diagnostic or capture command requests it. Presentation failure cancels the
in-progress timing sample and cannot publish a completed frame.

Status and qualification output identify source revision, driver/backend,
physical device, selected stable scene identity, availability reason, workload
profile, dimensions, physics generation, and whether host readback was used.
Physics validity, visual execution, and performance are reported separately.

## Characterization gates

These tests protect behavior while later F7 tasks move ownership:

| Behavior | Test |
|---|---|
| Common-session selection, rejected selection, reset, pause, inactive frame, foreground recovery, and point release/restore | `parallel-mater-gallery-session-contract-tests` |
| Catalog selection, modal reset/action gating, and held-input clearing | `parallel-mater-gallery-input-tests` |
| Camera, gravity/force policies, motor/dump classification, and four-wheel steering math | `parallel-mater-camera-controller-tests` |
| Full mobile session scene loop, presentation failure, and qualification lifecycle on Metal | `parallel-mater-ipad-session-tests` |
| Desktop fixed-step timing and headless render behavior | `parallel-mater-gallery-frame-timing-tests` and `parallel-mater-gallery-frame-timing-headless` |

F7-02 through F7-07 may change types and file layout only while these observable
contracts remain covered. F7-08 defines the renderer interface in detail; F7-09
through F7-18 add native Vulkan coverage and complete Android qualification.
