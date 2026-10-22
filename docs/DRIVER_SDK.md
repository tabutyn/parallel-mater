# Parallel Mater driver SDK

Parallel Mater separates the application API, loader, layers, and installable
drivers. Applications link only `parallel-mater` and resolve commands through
`pmGetInstanceProcAddr` or `pmGetDeviceProcAddr`. A driver does not export the
ordinary `pm*` application symbols.

## Packages

| Package | Responsibility |
|---|---|
| `parallel_mater.h` | Versioned application ABI generated from `registry/pm.xml` |
| `libparallel-mater` | Driver/layer discovery, physical-device aggregation, dispatch |
| `pm_driver.h` | Versioned private loader-driver ABI |
| `pm_layer.h` | Versioned loader-layer observation and validation ABI |
| `parallel_mater_driver_slang_cuda` | NVIDIA GPU driver for Linux/Windows; shared Slang compiled to CUDA/PTX |
| `parallel_mater_driver_slang_metal` | Apple GPU driver for macOS/iPadOS; shared Slang compiled to Metal libraries |
| `parallel_mater_driver_slang_d3d12` | Windows GPU driver; shared Slang compiled to DXIL |
| `parallel_mater_driver_slang_vulkan` | Linux/Android GPU driver; shared Slang compiled to SPIR-V |
| `parallel_mater_driver_slang_cpu` | Serial C++ development driver for Linux/macOS; shared Slang compiled to host code |
| `PM_LAYER_PARALLEL_MATER_validation` | Optional validation layer |

## Required driver exports

```c
PmResult pm_driverNegotiateLoaderInterfaceVersion(uint32_t* pVersion);
PFN_pmVoidFunction pm_driverGetInstanceProcAddr(
    PmInstance instance, const char* pName);
PFN_pmVoidFunction pm_driverGetDeviceProcAddr(
    PmDevice device, const char* pName);
PFN_pmVoidFunction pm_driverGetPhysicalDeviceProcAddr(
    PmInstance instance, const char* pName);
```

`pm_driverNegotiateLoaderInterfaceVersion` replaces `*pVersion` with the newest
interface version supported by both parties. Version zero and any returned version greater than the loader requested are invalid. The same rule applies to layer negotiation. Driver
libraries use hidden symbol visibility and export only these four functions.

Global lookup with a null instance exposes `pmGetInstanceProcAddr`,
`pmEnumerateInstanceVersion`, and `pmCreateInstance`. Instance, physical-device,
and device lookups return commands implemented for that object. Unsupported
commands return null.

No C++ standard-library type crosses the driver boundary.

## Driver manifest

```json
{
  "file_format_version": "1.0.0",
  "PMD": {
    "library_path": "/usr/lib/libparallel_mater_vendor.so",
    "api_version": "0.2.0",
    "library_arch": "64"
  }
}
```

The loader reads manifests named by `PM_DRIVER_FILES`. The variable is a
colon-separated list on Unix and semicolon-separated list on Windows.
`PM_DRIVER_PATH` names directories containing manifests. Relative library paths
are resolved from the manifest directory. Duplicate canonical library paths are
loaded once.

Installed manifests live under `share/parallel-mater/drivers.d`. Platform
packages should register that directory through the platform loader policy or
set an equivalent compiled default.

## Dispatch and ownership

The loader owns application-visible `PmInstance`, `PmPhysicalDevice`,
`PmDevice`, `PmQueue`, and `PmCommandBuffer` wrappers. Their first word is the
loader dispatch pointer. The wrapper holds the native driver object and routes
commands directly to its owner.

All other handles are opaque driver-owned objects. They may be used only with
their parent device or a command buffer from that device. Cross-driver sharing
requires an explicit external-memory and external-synchronization extension.

An instance may contain physical devices from several drivers. No backend enum
selects a physics implementation; selecting a physical device selects its
driver.

## Driver responsibilities

A conforming driver implements:

- physical-device features, properties, limits, formats, and queue families;
- device and queue creation;
- explicit allocation, mapping, resource binding, and imported memory;
- physics-pipeline creation and simulation resources;
- command recording, submission, barriers, and asynchronous execution;
- fences, timeline semaphores, timestamps, statistics, and memory queries;
- advertised interoperability extensions;
- advertised execution-profile negotiation and immutable per-simulation
  selection;
- the behavior required by the Parallel Mater specification and conformance
  cases.

The loader does not compile Slang, allocate device memory, record native command
buffers, schedule physics, or implement equations.

`PM_EXT_simulation_profile` advertises the negotiation ABI, not a solver.
`pmEnumerateExecutionProfiles` must return only exact
name/version pairs the device can execute. Physical qualification is separate
revisioned evidence and never appears implicitly from this runtime query. A
driver that understands the extension but has not implemented
`realtime-single-step` 1.0 enumerates no such record and returns
`PM_ERROR_FEATURE_NOT_PRESENT` for the exact request.

## Reference implementation

The reference drivers reuse the shared C++ runtime in `src/simulation` and the
Slang physics sources in `src/slang`. A native device adapter inside each
driver supplies allocation, transfer, binding, dispatch, synchronization,
queries, and supported native interoperability. Collision, constraint,
material, and coupling equations stay in Slang. The host schedules the work;
CUDA, Metal, D3D12, and Vulkan execute the generated physics kernels on GPUs.
Only the separately selected Slang CPU driver executes generated host code.

CUDA, Metal, and Vulkan expose rigid bodies/constraints, liquids, cloth, ropes,
soft bodies, smoke, paint, and implemented cross-domain interactions. D3D12
exposes rigid bodies/constraints, liquids, cloth, and their implemented
couplings. The CPU driver exposes rigid bodies/constraints, with explicitly
enabled experimental liquids. Exact interaction and native-import support
must be queried; shared source or a broad domain bit does not establish it.
For build and execution details see [Slang](SLANG.md), [Vulkan](VULKAN.md), and
[Metal](METAL-QUALIFICATION.md). The [qualification index](QUALIFICATION.md)
separately records physical evidence and failures for each driver revision.

These reference implementation choices are not requirements to use Slang in an
independent driver. The loader negotiates the versioned C driver ABI and routes
commands; an independent driver must satisfy the API's observable contracts
and its advertised capability obligations.

A hardware vendor may reuse its compiler, memory manager, queue scheduler, and
kernel runtime between its Vulkan driver and its Parallel Mater driver. Parallel
Mater does not require the vendor driver to execute through Vulkan.

## Qualification

Before advertising a capability, run:

1. loader/driver ABI and symbol-ownership tests;
2. public API lifecycle, memory, command, synchronization, and query tests;
3. Slang shader reflection and architecture checks;
4. the canonical conformance corpus without changing its tolerances;
5. all gallery scenes through the loader and public API;
6. device-specific performance qualification.

Record these as separate evidence scopes and validate every advertised
interaction record with `conformance/v1/qualification.py`. The hashed
`capability-obligations-v1` profile rejects missing contracts, stale case/profile
hashes, unsupported execution and attempts to substitute compilation,
reference agreement or timing for analytic physical quality.

Register the raw result in `conformance/v1/evidence/evidence-sources-v1.json`
and regenerate `evidence/index-v1.json`. Release review runs
`evidence_index.py --check`, so a documentation claim cannot outlive or exceed
the revisioned result that supports it.

The build target `parallel-mater-driver-abi-tests` verifies negotiation,
function lookup, and the absence of ordinary `pm*` exports from a driver. The
architecture test rejects CUDA types or physics details in the loader and native
launch intrinsics outside the shared device-intrinsic module.

## Compatibility and validation

API 0.2 currently accepts `PM_API_VERSION_0_2` or zero (the 0.2 default).
Other API versions return `PM_ERROR_INCOMPATIBLE_VERSION`; pre-stable minor
versions do not imply backward compatibility. Loader/driver and loader/layer
interface versions are separate from the application API version.

The validation layer is advertised as `PM_LAYER_PARALLEL_MATER_validation`.
The initial `PM_LAYER_KHRONOS_validation` spelling remains an unadvertised
compatibility alias for existing 0.2 applications; it implies no Khronos
ownership or endorsement. The command arguments and state validation contract
is tracked in [API contract rules](API-CONTRACT.md).

Layer ABI 2 appended `argumentSize` and `pArguments` to `PmLayerCallInfo`.
Layer ABI 3 appends the application-visible dispatch-object type/handle and the
per-instance structured-diagnostic callback/user data. The installed,
registry-generated `pm_layer_args.h` defines `PmLayerArgs_pmCommand` for each
command. Typed payloads use application-visible handles; `primaryHandle` retains
the native handle for v1 compatibility. Pointers and payloads are valid only
during their callback; post-call sees driver-written output parameters.
Pre-call failure suppresses the driver call. Post-call is observation only and
must not throw or convert a successful resource creation into an error.
Existing ABI 1 and 2 layers remain loadable; this version of the validation
layer requires a loader offering ABI 3. Unsupported returned ABI versions are
rejected. Applications configure diagnostics by chaining
`PmValidationDiagnosticsCreateInfo` through instance creation; this is an
application/loader contract and does not change the loader/driver interface.

The relocated installed-SDK gate builds separate C layer fixtures for frozen
ABI 1, ABI 2 and current ABI 3 prefixes, plus successful zero/999 negotiation
negative controls for both drivers and layers. Separate public-C executables
exercise API 0/default/old/future versions and instance/device extension
discovery. These fixtures compile against installed headers only; private C++
loader, layer and solver types are never present in the external source tree.

For version compatibility, independently built consumers/drivers and install gates,
see [SDK releases](SDK-RELEASES.md). Runtime qualification is scoped separately in
the [qualification index](QUALIFICATION.md).
