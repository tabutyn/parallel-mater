#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL = re.compile(r"__global__\s+void\s+([A-Za-z0-9_]+)")

failures: list[str] = []

cmake = (ROOT / "CMakeLists.txt").read_text()
loader_cmake = (ROOT / "cmake" / "ParallelMaterLoader.cmake").read_text()
build_contract = cmake + "\n" + loader_cmake
runtime_path = ROOT / "src" / "simulation" / "runtime.cu"
runtime = runtime_path.read_text()
runtime_header = (ROOT / "src" / "simulation" / "runtime.hpp").read_text()
registry = (ROOT / "registry" / "pm.xml")
if not registry.exists():
    failures.append("registry/pm.xml: canonical API registry missing")
if "parallel-mater-driver-slang-cuda SHARED" not in cmake:
    failures.append("CMakeLists.txt: Slang/CUDA runtime is not an installable driver")
cpu_driver_cmake_path = ROOT / "cmake" / "ParallelMaterSlangCpu.cmake"
if not cpu_driver_cmake_path.exists():
    failures.append("cmake/ParallelMaterSlangCpu.cmake: macOS Slang driver missing")
else:
    cpu_driver_cmake = cpu_driver_cmake_path.read_text()
    for required in (
        "parallel-mater-driver-slang-cpu SHARED",
        "-target cpp",
        "parallel_mater_geometry_runtime.cpp",
        "parallel_mater_geometry_constraints_runtime.cpp",
        "parallel_mater_queries_runtime.cpp",
    ):
        if required not in cpu_driver_cmake:
            failures.append(
                "cmake/ParallelMaterSlangCpu.cmake: missing shared-Slang "
                f"driver contract {required}"
            )
    cpu_dispatch_contracts = {
        "src/simulation/cpu_geometry_dispatch.cpp":
            "parallel_mater_geometry_runtime.cpp",
        "src/simulation/cpu_geometry_constraints_dispatch.cpp":
            "parallel_mater_geometry_constraints_runtime.cpp",
        "src/simulation/cpu_exports.cpp":
            "parallel_mater_queries_runtime.cpp",
    }
    for relative, generated_source in cpu_dispatch_contracts.items():
        dispatch_text = (ROOT / relative).read_text()
        if generated_source not in dispatch_text:
            failures.append(
                f"{relative}: not backed by generated Slang C++"
            )
        for native_launch in ("__global__", "<<<"):
            if native_launch in dispatch_text:
                failures.append(
                    f"{relative}: native physics launch {native_launch} remains"
                )
if "include(cmake/ParallelMaterLoader.cmake)" not in cmake or \
        "add_library(parallel-mater SHARED" not in loader_cmake:
    failures.append("CMakeLists.txt: public API is not a standalone loader")
for embedded_path in (
    "PARALLEL_MATER_DEFAULT_DRIVER_PATH",
    "PARALLEL_MATER_DEFAULT_VALIDATION_LAYER_PATH",
):
    if embedded_path in build_contract:
        failures.append(
            f"CMakeLists.txt: loader embeds non-relocatable {embedded_path}"
        )
loader_driver = (ROOT / "src" / "loader" / "driver.cpp").read_text()
if "append_loader_relative_directories" not in loader_driver:
    failures.append("src/loader/driver.cpp: relocatable manifest discovery missing")
if "ManifestKind::driver" not in loader_driver or "ManifestKind::layer" not in loader_driver:
    failures.append("src/loader/driver.cpp: driver/layer manifests are not separated")
for ansi_loader_call in ("LoadLibraryA", "GetModuleHandleExA", "GetModuleFileNameA"):
    if ansi_loader_call in loader_driver:
        failures.append(
            f"src/loader/driver.cpp: Windows loader is not Unicode-safe: {ansi_loader_call}")
if "pm_driverNegotiateLoaderInterfaceVersion" not in (
        ROOT / "include" / "parallel_mater" / "pm_driver.h").read_text():
    failures.append("pm_driver.h: loader/driver ABI negotiation missing")
for loader_source in sorted((ROOT / "src" / "loader").glob("*")):
    if not loader_source.is_file():
        continue
    loader_text = loader_source.read_text()
    for forbidden in ("cudaStream", "cudaEvent", "parallel_mater::World", ".slang"):
        if forbidden in loader_text:
            failures.append(
                f"{loader_source.relative_to(ROOT)}: loader owns device/physics detail {forbidden}")
if "install(DIRECTORY include/" in build_contract:
    failures.append("CMakeLists.txt: legacy backend-specific headers are installed")
if (ROOT / "include" / "parallel_mater" / "parallel_mater.hpp").exists():
    failures.append("include/parallel_mater/parallel_mater.hpp: private runtime is public")
if (ROOT / "include" / "parallel_mater" / "solver" / "contact.hpp").exists():
    failures.append(
        "include/parallel_mater/solver/contact.hpp: alternate contact solver remains"
    )
if re.search(r"\bclass\s+World\b", runtime_header):
    failures.append("src/simulation/runtime.hpp: legacy World class remains")
if re.search(r"\busing\s+World\s*=", runtime_header):
    failures.append("src/simulation/runtime.hpp: legacy World alias remains")
for boundary_header in (
    ROOT / "src" / "api" / "internal.hpp",
    ROOT / "src" / "device" / "device.hpp",
):
    boundary_text = boundary_header.read_text()
    for cuda_type in ("cudaStream_t", "cudaEvent_t", "cuda_runtime_api.h"):
        if cuda_type in boundary_text:
            failures.append(
                f"{boundary_header.relative_to(ROOT)}: device boundary "
                f"exposes {cuda_type}")
for api_source in sorted((ROOT / "src" / "api").glob("*")):
    if not api_source.is_file():
        continue
    api_text = api_source.read_text()
    if re.search(r"parallel_mater::World\b", api_text):
        failures.append(
            f"{api_source.relative_to(ROOT)}: command API uses legacy World type")
    for cuda_service in (
        "cudaMemcpy",
        "cudaMalloc",
        "cudaFree",
        "cudaSetDevice",
        "cudaLaunchHostFunc",
        "cudaStreamSynchronize",
    ):
        if cuda_service in api_text:
            failures.append(
                f"{api_source.relative_to(ROOT)}: API bypasses device services "
                f"with {cuda_service}"
            )
for retired_metal_path in (
    "cmake/BuildMetalPhysics.cmake",
    "include/parallel_mater/metal.hpp",
    "src/metal",
    "tests/metal",
    "examples/metal_gallery",
    "tests/parity/capture_scenario.hpp",
    "conformance/v1/src/metal_runner.mm",
):
    retired_path = ROOT / retired_metal_path
    if retired_path.is_file() or (
        retired_path.is_dir()
        and any(item.is_file() for item in retired_path.rglob("*"))
    ):
        failures.append(f"{retired_metal_path}: retired Metal adapter remains")
for retired_metal_build_token in (
    "option(PARALLEL_MATER_BUILD_METAL",
    "ParallelMater::metal",
    "parallel-mater-metal",
):
    if retired_metal_build_token in build_contract:
        failures.append(
            f"CMakeLists.txt: retired Metal adapter token {retired_metal_build_token}"
        )
for legacy_root in (
    ROOT / "tests",
    ROOT / "benchmarks",
    ROOT / "conformance" / "v1" / "src",
):
    for source in sorted(legacy_root.rglob("*")):
        if not source.is_file() or "metal" in source.parts:
            continue
        if source.suffix not in {".cpp", ".cu", ".hpp", ".h"}:
            continue
        source_text = source.read_text()
        if re.search(
            r"\busing\s+World\s*=|\bWorld::|"
            r"\b(?:create_scene_world|instantiate_scene|scene_world_options)\b",
            source_text,
        ):
            failures.append(
                f"{source.relative_to(ROOT)}: legacy World test path remains")
if "Backend-specific physics implementations are retired" not in cmake:
    failures.append("CMakeLists.txt: retired native physics builds are not rejected")
retired_headers = {
    "cloth.cuh",
    "cloth_rope.cuh",
    "cloth_smoke.cuh",
    "fluid.cuh",
    "fluid_cloth.cuh",
    "fluid_rope.cuh",
    "fluid_smoke.cuh",
    "fluid_soft_body.cuh",
    "geometry_cloth.cuh",
    "geometry_constraints.cuh",
    "geometry_constraints_cuda.cuh",
    "geometry_fluid.cuh",
    "geometry_smoke.cuh",
    "geometry_soft_body.cuh",
    "geometry_rope.cuh",
    "rope_smoke.cuh",
    "smoke_grid.cuh",
    "soft_body.cuh",
    "soft_body_cloth.cuh",
    "soft_body_smoke.cuh",
    "soft_body_rope.cuh",
}
for retired_header in sorted(retired_headers):
    if (ROOT / "src" / retired_header).exists():
        failures.append(f"src/{retired_header}: retired native physics source remains")
    if f'src/{retired_header}' in cmake:
        failures.append(f"CMakeLists.txt: migrated native physics source {retired_header}")
    if f'#include "{retired_header}"' in runtime:
        failures.append(
            f"src/simulation/runtime.cu: migrated native physics include {retired_header}")

for module in (
    "cloth",
    "cloth_rope",
    "cloth_smoke",
    "fluid",
    "fluid_cloth",
    "fluid_rope",
    "fluid_smoke",
    "fluid_soft_body",
    "geometry_cloth",
    "geometry_constraints",
    "geometry_fluid",
    "geometry_smoke",
    "geometry_soft_body",
    "rope",
    "rope_smoke",
    "soft_body",
    "soft_body_cloth",
    "soft_body_smoke",
    "soft_body_rope",
    "smoke_grid",
):
    dispatch = ROOT / "src" / "simulation" / f"{module}_dispatch.cu"
    if not dispatch.exists():
        failures.append(f"missing {dispatch.relative_to(ROOT)}")
    elif f'parallel_mater_{module}_runtime.cu' not in dispatch.read_text():
        failures.append(f"{dispatch.relative_to(ROOT)}: not backed by generated Slang")

for shader in sorted((ROOT / "src" / "slang").rglob("*.slang")):
    if shader.name == "device_intrinsics.slang":
        continue
    text = shader.read_text()
    for retired_metal_token in ("PM_TARGET_METAL", "import metal_common"):
        if retired_metal_token in text:
            failures.append(
                f"{shader.relative_to(ROOT)}: retired Metal adapter token "
                f"{retired_metal_token}"
            )
    for intrinsic in ("cudaBlockIdx()", "cudaThreadIdx()", "cudaBlockDim()"):
        if intrinsic in text:
            failures.append(
                f"{shader.relative_to(ROOT)}: native launch intrinsic {intrinsic}")

migrated_launches = {
    "cloth_project_volume", "cloth_break_bonds", "cloth_limit_strain",
    "cloth_update_surface", "rope_cloth_sample_anchor",
    "rope_cloth_apply_anchor", "smoke_cloth_wind", "smoke_cloth_contact",
    "fluid_reserve_contact_events", "fluid_gather_contact_events",
    "fluid_source_vacancies", "fluid_source_emit", "fluid_destroy_flags",
    "fluid_gather", "fluid_copy_initial",
    "fluid_cloth_containment_forces", "cloth_apply_fluid_forces",
    "fluid_project_inside_cloth", "fluid_rope_contacts", "fluid_rope_apply",
    "fluid_smoke_drag", "fluid_smoke_grid_drag", "fluid_smoke_exchange",
    "fluid_clear_inactive_keep", "fluid_soft_refit", "fluid_soft_detect",
    "fluid_soft_solve", "fluid_soft_apply", "fluid_soft_recover",
    "smoke_rope_wind", "smoke_rope_contact", "deformable_predict",
    "soft_body_measure_momentum", "soft_body_restore_momentum",
    "soft_body_project_rest_shape", "deformable_project_links",
    "deformable_project_links_warp", "soft_body_finalize_velocities",
    "soft_body_apply_contact_friction", "soft_body_damp_springs",
    "soft_body_damp_springs_warp", "soft_body_update_surface",
    "soft_cloth_detect", "soft_cloth_apply_soft", "soft_cloth_apply_cloth",
    "rope_soft_sample_anchor", "rope_soft_scatter_anchor", "rope_soft_apply",
    "smoke_soft_body_wind", "smoke_soft_body_contact",
    "query_rigid_hit_box_kernel", "query_particle_hit_box_kernel",
    "cloth_constrain_bodies", "cloth_apply_body_corrections",
    "fluid_static_contacts", "fluid_body_bounds_kernel",
    "fluid_index_body_cells", "fluid_detect_moving_contacts",
    "fluid_resolve_moving_contacts", "reduce_point_body_impulses",
    "smoke_emit_cells", "smoke_density_pressure", "smoke_pair_forces",
    "smoke_compute_vorticity", "smoke_advect", "smoke_emit",
    "smoke_deformable_bounds", "smoke_rigid_contact",
    "smoke_apply_rigid_impulses",
    "deformable_collide",
    "soft_body_surface_contacts", "soft_body_apply_surface_contacts",
    "smoke_grid_initialize_cells", "smoke_grid_initialize_faces",
    "smoke_grid_clear_obstacles", "smoke_grid_raster_rigid",
    "smoke_grid_raster_deformable", "smoke_grid_raster_soft",
    "smoke_grid_resolve_particle_fields", "smoke_grid_splat_particles",
    "smoke_grid_advect_face", "smoke_grid_correct_face",
    "smoke_grid_cell_diagnostics", "smoke_grid_subgrid_force",
    "smoke_grid_apply_face_forces", "smoke_grid_apply_face_boundaries",
    "smoke_grid_divergence", "smoke_grid_pressure_smooth",
    "smoke_grid_pressure_residual", "smoke_grid_restrict_residual",
    "smoke_grid_clear_scalar", "smoke_grid_restrict_open",
    "smoke_grid_mark_domain_boundaries", "smoke_grid_prolong_add",
    "smoke_grid_compare_residual", "smoke_grid_reset_residual",
    "smoke_grid_project_face", "smoke_grid_trace",
    "smoke_grid_soft_body_force", "smoke_grid_cloth_force",
    "smoke_grid_apply_cloth_force", "smoke_grid_rigid_force",
    "clear_rigid_inputs_kernel", "capture_rigid_inputs_kernel",
    "prepare_parallel_contact_events_kernel",
    "load_rigid_contact_cache_kernel", "save_rigid_contact_cache_kernel",
    "gather_rigid_debug_samples_kernel", "clamp_rigid_speeds_kernel",
    "build_fixed_collision_groups_kernel",
    "prepare_avbd_kernel", "solve_avbd_kernel",
    "rope_advance",
}
for kernel in sorted(migrated_launches):
    if re.search(rf"\b{re.escape(kernel)}\s*<<<", runtime):
        failures.append(
            f"src/simulation/runtime.cu: native launch remains for {kernel}")

consumer_paths = [ROOT / "examples" / "gallery" / "application.cpp"]
consumer_paths.extend(sorted((ROOT / "examples" / "gallery" / "render").glob("*")))
for path in consumer_paths:
    if not path.is_file():
        continue
    if re.search(
        r"\bWorld(?:Options|StepTimings|Statistics)?\b", path.read_text()
    ):
        failures.append(f"{path.relative_to(ROOT)}: legacy World dependency")

backend_neutral_sources = (
    ROOT / "src" / "fluid_geometry.cpp",
    ROOT / "src" / "fluid_source.cpp",
    ROOT / "src" / "soft_body_geometry.cpp",
    ROOT / "src" / "rope_geometry.cpp",
    ROOT / "examples" / "gallery" / "scene_builder.cpp",
    ROOT / "examples" / "support" / "scene.cpp",
    ROOT / "examples" / "support" / "vector_math.hpp",
)
for path in backend_neutral_sources:
    text = path.read_text()
    for backend_detail in (
        "simulation/runtime.hpp", "cuda_runtime", "cudaSuccess", "cudaError_t",
    ):
        if backend_detail in text:
            failures.append(
                f"{path.relative_to(ROOT)}: host preparation owns backend detail "
                f"{backend_detail}"
            )
if "parallel-mater-asset-preparation" in cmake:
    failures.append("CMakeLists.txt: duplicate gallery asset-preparation target remains")

gallery_application = (ROOT / "examples" / "gallery" / "application.cpp").read_text()
for platform_detail in (
    "cuda_runtime_api.h", "cudaMemGetInfo", "cudaGetDevice",
    "cuda_runtime", "OptixRenderer", "PARALLEL_MATER_OPTIX_PTX_PATH",
    "glDrawPixels",
):
    if platform_detail in gallery_application:
        failures.append(
            "examples/gallery/application.cpp: shared application owns "
            f"platform detail {platform_detail}")
macos_gallery_main = ROOT / "examples" / "gallery" / "macos_main.mm"
macos_gallery_cmake = ROOT / "cmake" / "ParallelMaterMacOSGallery.cmake"
for path in (macos_gallery_main, macos_gallery_cmake):
    if path.exists() and "PARALLEL_MATER_DEVELOPMENT_DRIVER_PATH" in path.read_text():
        failures.append(
            f"{path.relative_to(ROOT)}: installed app embeds a development "
            "driver path")
renderer_interface = (
    ROOT / "examples" / "support" / "include" /
    "parallel_mater_gallery" / "renderer.hpp"
).read_text()
if "OptixRenderer" in renderer_interface:
    failures.append(
        "examples/support/include/parallel_mater_gallery/renderer.hpp: "
        "drawing interface exposes the OptiX adapter")
for removed_vulkan_executor in (
    ROOT / "src" / "api" / "interop_vulkan.cpp",
    ROOT / "src" / "device" / "vulkan_executor.cpp",
    ROOT / "src" / "device" / "vulkan_executor.hpp",
):
    if removed_vulkan_executor.exists():
        failures.append(
            f"{removed_vulkan_executor.relative_to(ROOT)}: unused partial backend remains")

d3d12_window_path = ROOT / "examples" / "support" / "d3d12_window.cpp"
if not d3d12_window_path.exists():
    failures.append("examples/support/d3d12_window.cpp: adapter missing")
else:
    d3d12_window = d3d12_window_path.read_text()
    for required in ("D3D12CreateDevice", "CreateSwapChainForHwnd"):
        if required not in d3d12_window:
            failures.append(
                f"examples/support/d3d12_window.cpp: missing {required}"
            )
    for forbidden in ("solve_avbd_kernel", "prepare_avbd_kernel", "runtime.cu"):
        if forbidden in d3d12_window:
            failures.append(
                "examples/support/d3d12_window.cpp: physics leaked into "
                f"adapter: {forbidden}"
            )

slang_constraints_cmake = (
    ROOT / "cmake" / "SlangGeometryConstraints.cmake"
).read_text()
for required in (
    "-target dxil",
    "-DPM_TARGET_D3D12=1",
    "parallel-mater-slang-avbd-dxil",
):
    if required not in slang_constraints_cmake:
        failures.append(
            f"cmake/SlangGeometryConstraints.cmake: missing DXIL contract {required}"
        )

# Native CUDA is limited to device services and generated-Slang dispatch. Host
# scheduling and shared helper headers must never regain a physics kernel.
native_sources = [runtime_path]
native_sources.extend(
    path for path in sorted((ROOT / "src").glob("*.cuh"))
)
for native in native_sources:
    kernels = set(CUDA_KERNEL.findall(native.read_text()))
    if kernels:
        failures.append(
            f"{native.relative_to(ROOT)}: native CUDA physics kernels remain: "
            + ", ".join(sorted(kernels))
        )

for forbidden in ("avbd_cuda.slang", "cuda_common.slang", "fluid_shared.slang"):
    if (ROOT / "src" / "slang" / forbidden).exists():
        failures.append(f"src/slang/{forbidden}: forbidden transitional module")

if failures:
    print("\n".join(failures), file=sys.stderr)
    raise SystemExit(1)

print(
    "Slang architecture valid; runtime contains no native physics kernels "
    "or direct physics launches"
)
