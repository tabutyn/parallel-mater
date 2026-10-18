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

Build profiles select what can be installed. The package has independent CMake
install/export components named `loader`, `validation`, `preparation`, `scene`,
`driver`, `conformance`, and `benchmark`. Install `loader` first because it owns
the public headers and package metadata, then install only the selected optional
components into the same prefix. `scene` additionally requires `preparation`.
Missing components and incomplete dependency combinations fail during
`find_package`, before a dangling imported target can be used.

Consumers request the same names with `find_package(ParallelMater 0.2 CONFIG
REQUIRED COMPONENTS ...)`. Use `ParallelMater::api` for the public C interface,
`ParallelMater::validation`, `ParallelMater::preparation`, and
`ParallelMater::scene` for the corresponding libraries. The `driver` component
sets `ParallelMater_DRIVER_TARGETS` to the imported drivers built by that SDK.
The `conformance` and `benchmark` components publish
`ParallelMater_CONFORMANCE_EXECUTABLE`, `ParallelMater_CONFORMANCE_DATA_ROOT`,
and `ParallelMater_BENCHMARK_QUALIFICATION`. Installed imported targets contain
no private source include directories. The optional C++ targets retain no stable
binary ABI promise.

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

On Windows, run the presets and installed-SDK checker from an x64 Visual Studio
Developer Command Prompt with Ninja on `PATH`. The checker applies its
`--config` value to both independently configured consumer builds; a Release
SDK therefore cannot be linked accidentally into a Debug MSVC consumer.

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

Before packaging a release, run two isolated builds from a clean checkout and
retain their manifests and comparison report:

```bash
python3 tools/check_release_reproducibility.py \
  --preset sdk-ci \
  --output out/release-reproducibility/sdk-ci
```

The output directory must not already exist. Each build has an independent
build and install prefix. `manifest-a.json` and `manifest-b.json` record the
source revision and tracked-tree digest, host/toolchain executable identities,
pinned dependencies, normalized CMake options, generated artifacts, installed
artifacts and every artifact SHA-256. The checker rejects installed runtime
metadata that requires a source, build or install path from the workstation.
Compiler paths are represented by executable name and checksum rather than by
their workstation location.

`report.json` compares the complete artifact sets. A difference is retained and
classified as text, binary-container content, known PE/archive metadata, or an
embedded build path. The check succeeds when both builds used equivalent
recorded inputs and neither installed runtime configuration retained a local
path; artifact differences do not become a false failure or disappear. The
report says that output is bit-identical only when every installed and generated
artifact checksum agrees. That observation applies only to the exact recorded
host, toolchain and input tuple and is not a cross-host reproducibility claim.

The repository presets use Ninja. A development host without Ninja can select
its native generator while preserving the same preset options, for example:

```powershell
python tools/check_release_reproducibility.py `
  --preset sdk-ci `
  --generator "Visual Studio 18 2026" --architecture x64 `
  --output out/release-reproducibility/sdk-ci-vs
```

`--allow-dirty` exists only for diagnostics. Evidence submitted for a release
must come from the default clean-source path.

`tools/check_sdk_install.py --build-dir build-sdk-ci --output out/sdk-evidence/new`
requires a new output directory. It installs, moves the SDK, then stages only
public consumer and independent mock-driver sources into an external project.
It builds C and C++ consumers, tests driver discovery/ABI negotiation and host
preparation, rejects incompatible package versions, and independently builds
conformance against the installed SDK and installed corpus. It also distinguishes
an unsupported capability from a missing-driver setup failure. No reference
physics implementation is linked into the mock or application.

`report.json` records source revision/dirty state, host architecture, requested
configuration, the exact CMake C++ compiler identity, checks, exit codes and SDK
file SHA-256 values. Windows reports also record the selected Windows SDK and
resource compiler.

`tools/check_sdk_components.py --build-dir build-sdk-ci --output
out/sdk-components/new` independently installs each supported component
combination into a relocated prefix and builds an external consumer. It also
requires clean configuration failures for absent components and for `scene`
without `preparation`. Backend builds can repeat `--expect-driver-target` to
assert the exact imported CUDA, D3D12, Vulkan, CPU, or Metal targets they ship.
Scope is `packaging_and_public_abi`; physical quality is explicitly
`not_evaluated`. The installed-SDK CI workflow enforces these checks on Linux
Apple Silicon macOS and Windows x64. Successful local checks do not imply that a
newly added remote job has already run.

A release requires a clean source revision, exact compiler/SDK/dependency
versions, generated contract checks, retained install/consumer/driver evidence,
and the qualification record for each claimed capability. Include source,
licenses, relative manifests, binary hashes and build options. Generate the full
reference PDF/layout artifacts with the documented reference toolchain for a
release; routine text-only generation is not that artifact step. Reproducible
configuration and pinned inputs are required; bit-identical binaries have not
yet been established across hosts/toolchains.

Register each raw capability result in
`conformance/v1/evidence/evidence-sources-v1.json`, regenerate the published
index, and require this release check:

```text
python3 conformance/v1/evidence_index.py --check
```

The check re-evaluates raw results against the pinned obligations and verifies
the per-driver block in `docs/QUALIFICATION.md`. Supplemental historical
reports remain visible but contribute no qualified capability records.

Any published performance score must also pass the installed headless
`benchmarks/qualification.py` gate. The resulting throughput, median/p95 timing
and memory metrics retain the exact physical-quality profile hash and
source/binary/device/toolchain tuple; an unqualified diagnostic measurement is
not release performance evidence.

No release may label a backend or the whole library "conformant" from compiler
success, a demo, or agreement with a single historical capture. See
[qualification scope](QUALIFICATION.md) and [contribution rules](../CONTRIBUTING.md).
