# F4-04 reflected binding evidence

Date: 2026-10-18

Machine: `windows-gtx1080`

GPU/driver: NVIDIA GeForce GTX 1080 / 582.66

Toolchain: pinned Slang 2026.18, Visual Studio 2022 generator, MSVC 19.44.35228,
Windows SDK 10.0.26100.0, Python 3.12.9

## Scope

The common `ReflectedABI` model now validates uniform parameter names, types,
offsets, sizes, record fields and padded array strides. D3D12 uses it for both
domain-wrapper generation and DXIL embedding. The embedder emits named symbolic
layouts used by direct rigid, AVBD and primitive dispatches, replacing their
literal offsets. The old CMake JSON embed/layout parser was deleted. No Slang
physics source, equation, schedule or capability was changed.

## Reproduction

```text
cmake -S . -B build-f4-04-d3d12 -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON -DPARALLEL_MATER_BUILD_CUDA=OFF -DPARALLEL_MATER_BUILD_D3D12=ON -DPARALLEL_MATER_BUILD_GALLERY=OFF -DPARALLEL_MATER_BUILD_CONFORMANCE=OFF -DPARALLEL_MATER_BUILD_SCENE_SDK=OFF -DPARALLEL_MATER_SLANGC_EXECUTABLE=<pinned-slang-2026.18>/bin/slangc.exe
cmake --build build-f4-04-d3d12 --config Release --target parallel-mater-driver-slang-d3d12 parallel-mater-d3d12-command-api-tests --parallel 4
ctest --test-dir build-f4-04-d3d12 -C Release --output-on-failure -R "^(parallel-mater-reflected-abi-tests|parallel-mater-dxil-binding-generator-tests|parallel-mater-slang-architecture|parallel-mater-d3d12-command-api-tests)$"
```

Result: PASS, 4/4 tests. The native D3D12 command API case executed in 12.52
seconds on its first focused run and 6.81 seconds on the final rerun.

The build compiled and validated all 141 isolated DXIL entries and generated 52
functions from the eight D3D12 domain adapter sources. A separate generation
check consumed the existing Vulkan artifacts available on this workstation and
regenerated all 14 modules (84 entries) through the same validator.

Negative unit cases reject duplicate manifest identities, mismatched reflected
entry names, changed wrapper argument counts, overlapping parameters and
unsupported uniform shapes. The architecture check rejects reintroduction of
the deleted CMake parser or numeric D3D12 geometry offsets.

## Limits

This evidence covers layout/signature generation, native D3D12 compilation and
command-path execution. It does not qualify physical correctness, lifecycle
coverage, performance, Vulkan execution or any new capability.
