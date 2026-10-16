# Parallel Mater driver SDK

Parallel Mater separates the application API, loader, layers, and installable
drivers. Applications link only `parallel-mater` and resolve commands through
`pmGetInstanceProcAddr` or `pmGetDeviceProcAddr`. A driver does not export the
ordinary `pm*` application symbols.

## Packages

| Package | Responsibility |
|---|---|
| `parallel_mater.h` | Stable application ABI generated from `registry/pm.xml` |
| `libparallel-mater` | Driver/layer discovery, physical-device aggregation, dispatch |
| `pm_driver.h` | Versioned private loader-driver ABI |
| `pm_layer.h` | Versioned loader-layer observation and validation ABI |
| `parallel_mater_driver_slang_cuda` | First reference driver and conformance baseline |
| `parallel_mater_driver_slang_d3d12` | Windows transfer driver and shared DXIL rigid-body/constraint adapter |
| `parallel_mater_driver_slang_cpu` | Serial macOS rigid-capability development driver |
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
- the behavior required by the Parallel Mater specification and conformance
  cases.

The loader does not compile Slang, allocate device memory, record native command
buffers, schedule physics, or implement equations.

## Reference implementation

The reference drivers use the shared Slang sources in `src/slang`. CUDA,
macOS-hosted C++, DXIL, and Vulkan SPIR-V are compiler targets of the same
modules; they are not separate physics implementations. The D3D12 and serial
macOS drivers advertise only the rigid capabilities their shared schedules
execute and no unported simulation domain. Platform device code may allocate,
transfer, bind, dispatch, and synchronize but cannot contain collision,
constraint, material, or coupling equations.

The previous standalone Vulkan executor scaffold has been removed. The
[loadable Vulkan preview](VULKAN.md) binds the shared portable schedule to
SPIR-V and advertises rigid bodies, constraints, liquids, and liquid–rigid
coupling. Other domains and native imports remain unavailable. Desktop checks
cover this preview scope; full conformance, engine handoff, sustained performance,
and phone qualification remain separate gates.

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
