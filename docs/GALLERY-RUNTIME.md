# Common Gallery runtime contract

The Gallery is one application with desktop, Android, and Apple shells. Its
portable C++ runtime is an example client of the installed Parallel Mater API;
it is not part of the physics ABI. Every shell must observe the same scene and
control semantics when its selected driver advertises the required features.

## Current ownership

Every shell delegates scene lifecycle to
`parallel_mater::gallery::GallerySession`. The session resolves workload and
scene parameters, selects scenes, resets them, applies semantic actions,
advances controlled steps, owns camera focus, and reports lifecycle revisions.
Its `GalleryRuntime` owns GLB loading, physics resources, host-renderer
creation, frame assembly, and teardown. Desktop reads the runtime only for
rendering, capture, calibration, and diagnostics; it has no parallel scene
lifecycle wrapper. Scene metadata and backend availability come from one
ordered `GalleryCatalog`; platform shells only choose how to display its
records. `GalleryFramePolicy` supplies the same profiled fixed-step clock and
catch-up behavior to desktop scheduling and the session-managed mobile loop.
Native rendering uses a common device, buffer, synchronization, target,
readback, and draw-semantics contract. Metal implements that contract today;
F7-09 adds the Vulkan adapter used by desktop and Android.

| Responsibility | Current desktop owner | Current mobile owner | Required owner |
|---|---|---|---|
| Ordered scene identity, name, help, source, control policy, feature requirements, availability, and rejection reason | `GalleryCatalog` | `GallerySession::catalog()` | Common runtime catalog (done) |
| Driver discovery, explicit backend choice, and capability query | Desktop startup and backend picker | Platform startup configuration; Android fixes Vulkan | Platform shell requests a backend; common runtime reports devices and scene availability |
| Asset and device-profile paths | Desktop options and compile-time resource paths | Android/iPad shell configuration | Platform shell supplies paths once |
| Workload limits and user-adjustable counts | Desktop supplies resolved limits and edits `GallerySession` parameters | `GallerySession` profile selection and shared frame policy | Common session policy; shell edits declared parameters |
| Asset loading and scene construction | `GallerySession`/`GalleryRuntime` | `GallerySession`/`GalleryRuntime` | Common runtime catalog and workload policy (done) |
| Physics device, resources, command submission, scene replacement, reset, and teardown | `GallerySession`/`GalleryRuntime` | `GallerySession`/`GalleryRuntime` | Common runtime (done) |
| Fixed, point, motor, steering, dump-bed, kinematic, gravity, force, paint, and deformable controllers | Semantic input adapter around `GalleryRuntime` | Semantic input adapter around `GallerySession`/`GalleryRuntime` | Common runtime semantic commands (done) |
| Fixed-step accumulation, catch-up limits, substeps, pause, and foreground recovery | `GalleryFramePolicy` | `GalleryFramePolicy` through `GallerySession` | Common runtime clock and workload policy (done) |
| Camera state and focus policy | `GallerySession::camera()`, driven by desktop input | `GallerySession` touch methods | Common runtime, driven by semantic pointer/zoom commands |
| Host render-frame assembly | `GalleryRuntime::step` | `GalleryRuntime::step` | Common runtime (done) |
| Software, native GPU, or external-engine drawing | Desktop renderer/window path; optional Metal path | Software renderer or iPad Metal renderer | Renderer adapter |
| Graphics API resources, swapchain/layer binding, upload, and presentation | Desktop window adapter | Android surface/ANativeWindow or Apple presentation layer | Renderer adapter, with the platform shell supplying the native surface |
| Keyboard, touch, gamepad, lifecycle, window, intent, and accessibility events | GLFW/native desktop glue | Java/JNI or UIKit glue | Platform shell translating events into common commands |
| Catalog, parameter, timing, status, and error view models | Shared catalog plus desktop overlays | Shared catalog plus Java/UIKit widgets and session strings | Common runtime data; platform shell chooses widgets and layout |
| Physics capture, validation snapshots, and qualification sequencing | Desktop application and shared debug helpers | `GallerySession` qualification helpers | Common runtime diagnostics; shell chooses destination and presentation |
| Filesystem writes, share sheets, logs, and platform telemetry | Desktop platform helpers | Java/UIKit | Platform shell |

## Runtime boundary

The common runtime owns exactly one selected backend and at most one live
scene. It owns all physics objects, controller state, workload policy, camera,
frame timing, diagnostics, and the render handoff. A renderer adapter may hold
graphics resources, but it must not select scenes, advance physics, or interpret
controls. A platform shell may discover paths and native surfaces and translate
events, but it must not create physics resources or implement a scene rule.

## Native GPU renderer boundary

`GpuGalleryRenderer` receives typed `GpuRenderRequest` records. The common
resolver obtains `PmNativeDeviceInfo` and `PmNativeBufferInfo` only through the
public interoperability API. A request names the physics queue, synchronization
mode, ownership direction, resource generation, completed frame generation,
native target backend/device/handle, dimensions, overlay, and optional
diagnostic readback. A null readback keeps ordinary frames on the GPU.

Every scene-buffer allocation receives a new resource generation. Every
successful physics submission advances a frame generation. The shared contract
state permits a paused frame to be drawn again, but rejects an older frame, an
old scene allocation, a foreign device or target, overlapping graphics
ownership, and a handoff that does not return the buffer to physics. The
current Metal adapter uses a queue-idle handoff and completes drawing before it
returns ownership. Timeline semaphore fields are part of the same contract for
the pipelined Vulkan adapter.

`GpuSceneDrawList` describes rigid meshes with current/previous poses, cloth
and soft-body surfaces, liquid/smoke/rope particles, fracture corner indexing,
paint fields, colors, and presentation radii. The records contain public output
layout and drawing meaning only. Named draw kinds in
`gallery_uniforms.slangh` are shared graphics shader inputs; physics behavior
remains in the physics Slang modules.

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

`GalleryControlInput` is the shared directional command. Its `right` and
`forward` values are camera-relative semantics rather than key or widget codes;
an optional gravity override supports deterministic scripted/headless runs.
`GallerySession::primary_action` interprets the catalog's declared primary
action and performs any requested reset. `GallerySession::controlled_step`
applies every controller before and after the shared physics step. Each
successful selection, reset, or restore increments the session revision so a
shell can clear presentation counters without duplicating lifecycle state.

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

Each workload profile's `target_fps` selects both the fixed timestep and enough
solver substeps to retain an approximately 240 Hz solver cadence. Interactive
adapters pass monotonic timestamps into `GalleryFramePolicy`. A display frame
may submit at most four due physics steps, and additional catch-up is admitted
only while the previous step predicts that the 8 ms physics budget remains.
A wall-clock delta is clamped to 250 ms; work still due after either limit is
discarded while its fractional remainder is retained. Modal UI, reset, profile
changes, pause, and background transitions clear accumulated elapsed time.
Scripted qualification uses the same policy with exact one-interval timestamps.

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
Every catalog consumer preserves the shared order and stable identity. A
backend capability query annotates each record with its required feature names,
availability, and an exact missing-feature reason. Unsupported records remain
visible and selection is rejected before the live scene is replaced; a shell
must not remap indexes, hide the record, or retry it on another backend.

## Characterization gates

These tests protect behavior while later F7 tasks move ownership:

| Behavior | Test |
|---|---|
| Common-session selection, rejected selection, reset, pause, inactive frame, foreground recovery, and point release/restore | `parallel-mater-gallery-session-contract-tests` |
| Shared catalog ordering, stable unique identities, required features, and precise backend rejection reasons | `parallel-mater-gallery-registry-tests` |
| Catalog selection, modal reset/action gating, and held-input clearing | `parallel-mater-gallery-input-tests` |
| Camera, gravity/force policies, motor/dump classification, and four-wheel steering math | `parallel-mater-camera-controller-tests` |
| Full mobile session scene loop, presentation failure, and qualification lifecycle on Metal | `parallel-mater-ipad-session-tests` |
| Cross-adapter fixed-step traces, catch-up limits, stalls, pause/background, modal UI, and profile changes | `parallel-mater-gallery-frame-policy-tests` |
| Native interop identity, stale generations, ownership transitions, target matching, and all-domain draw semantics | `parallel-mater-gallery-gpu-renderer-contract-tests` |
| Frame measurement and headless render behavior | `parallel-mater-gallery-frame-timing-tests` and `parallel-mater-gallery-frame-timing-headless` |

F7-02 through F7-07 may change types and file layout only while these observable
contracts remain covered. F7-08 defines the renderer interface in detail; F7-09
through F7-18 add native Vulkan coverage and complete Android qualification.
