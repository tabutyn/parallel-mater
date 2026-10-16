# SDK builds and release contract

Parallel Mater's application boundary is a loadable C library. Physics stays in
shared Slang; native drivers own compilation, allocation and execution services.
Engine integrations consume that boundary. The optional C++ scene/preparation
libraries are host tooling, with no stable C++ binary ABI promise.

## Versions and packages

The current SDK/package and application API line is **0.2**. Package patch
versions identify compatible fixes within that line. Until 1.0, a minor version
may break the contract: CMake accepts the same minor line and rejects 0.1/0.3
against 0.2.0. ELF/macOS loader ABI naming follows 0.2. Driver-interface version 1
and layer-interface version 2 are separate negotiation domains; increasing the
SDK patch number never silently changes either interface. Applications request
their API explicitly; the documented zero alias is not a compatibility promise
for arbitrary future versions. See [API contract](API-CONTRACT.md).

| Build profile or option | Contents and dependencies |
|---|---|
| `sdk-minimal` | Loader, validation, generated C/driver/layer headers; no Slang, GPU SDK, scene parser or renderer |
| `sdk-ci` | Minimal SDK, host tests, optional scene/preparation libraries, headless conformance and pinned corpus; no native driver |
| `sdk-cpu` | `sdk-ci` plus the pinned Slang CPU development driver; only advertised CPU capabilities |
| `PARALLEL_MATER_BUILD_SCENE_SDK` | Host scene libraries without requiring the runner or Gallery |
| `PARALLEL_MATER_BUILD_CONFORMANCE` | Headless public-command runner and corpus; selects installed drivers |
| `PARALLEL_MATER_BUILD_{CUDA,D3D12,VULKAN,SLANG_CPU_DRIVER}` | Independently selected reference driver builds |
| `PARALLEL_MATER_BUILD_GALLERY` | Renderer/window/application client |
| `PARALLEL_MATER_BUILD_BENCHMARKS` | Existing rendered/profiler benchmarks; still require Gallery/CUDA, not a portable SDK performance qualification gate |

Build profiles select components before installation. Partial `cmake --install
--component` packages are not supported by the combined target export. Consumers
can require CMake components `loader`, `scene`, `preparation`, and `validation`.
Missing components fail configuration. Installed imported targets contain no
private source include directories. Use `ParallelMater::api` (alias of
`ParallelMater::parallel_mater`) for the public C interface.

```bash
cmake --preset sdk-ci
cmake --build --preset sdk-ci
ctest --preset sdk-ci
cmake --install build-sdk-ci --prefix /path/to/sdk
cmake -S examples/sdk -B build-example -DCMAKE_PREFIX_PATH=/path/to/sdk
cmake --build build-example --parallel 1
PM_DRIVER_FILES=/path/to/vendor/driver.json build-example/pm-devices
```

These presets use one compiler job, suitable for the development laptop. Slang
is downloaded at a fixed version and archive SHA-256 when a driver requires it;
an explicit compiler must match the pinned version. cgltf's source and MIT
notice are pinned in `third_party/cgltf`. Gallery's optional dependencies are
outside the minimal SDK's dependency closure. A release must record their
resolved revisions if Gallery is included.

The conformance install contains `bin/parallel-mater-conformance` and its
manifest-pinned input data under `share/parallel-mater`. After relocation:

```bash
/moved/sdk/bin/parallel-mater-conformance --check-inputs \
  --root /moved/sdk/share/parallel-mater
```

No driver is needed to validate input hashes. Running physics requires a vendor
manifest. Relative driver/layer library paths resolve from their manifest;
Windows DLLs install in `bin`, Unix shared libraries in the configured `libdir`.

## Required release evidence

`tools/check_sdk_install.py --build-dir build-sdk-ci --output out/sdk-evidence/new`
requires a new output directory. It installs, moves the SDK, then stages only
public consumer and independent mock-driver sources into an external project.
It builds C and C++ consumers, tests driver discovery/ABI negotiation and host
preparation, rejects incompatible package versions, and independently builds
conformance against the installed SDK and installed corpus. It also distinguishes
an unsupported capability from a missing-driver setup failure. No reference
physics implementation is linked into the mock or application.

`report.json` records source revision/dirty state, host, checks, exit codes and
SDK file SHA-256 values. Scope is `packaging_and_public_abi`; physical quality is
explicitly `not_evaluated`. The installed-SDK CI workflow enforces these checks
on Linux. Other platform packaging remains unqualified until its own run is
recorded. Successful local checks do not imply that a newly added remote job has
already run.

A release requires a clean source revision, exact compiler/SDK/dependency
versions, generated contract checks, retained install/consumer/driver evidence,
and the qualification record for each claimed capability. Include source,
licenses, relative manifests, binary hashes and build options. Generate the full
reference PDF/layout artifacts with the documented reference toolchain for a
release; routine text-only generation is not that artifact step. Reproducible
configuration and pinned inputs are required; bit-identical binaries have not
yet been established across hosts/toolchains.

No release may label a backend or the whole library "conformant" from compiler
success, a demo, or agreement with a single historical capture. See
[qualification scope](QUALIFICATION.md) and [contribution rules](../CONTRIBUTING.md).
