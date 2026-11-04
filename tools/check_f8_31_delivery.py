#!/usr/bin/env python3
"""Validate the F8-31 single-path integration delivery and retained failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_PATH = ROOT / "docs/roadmap/evidence/F8-31-INTEGRATED-DELIVERY.json"
RELEASE_NOTES_PATH = ROOT / "docs/F8-INTEGRATION-RELEASE.md"
MIGRATION_PATH = ROOT / "docs/reference/MIGRATION.md"

CANDIDATE_ID = "realtime-single-step-v1-5bacbe2"
SOURCE_REVISION = "5bacbe2b1e0fbfad565fb9477bb6873396e31d5d"
SOURCE_TREE = "67f5088c2e9f50938c55856e3683326e42675310"
MANIFEST_SHA256 = "29e051d94a89a39d51c2128b212165d60da6da7de4b2c107a72ae7cc5b77f20a"

EXPECTED_DEVICE_VERDICTS = {
    "windows-gtx1080-slang-cuda": ("passed", "passed", "not_qualified"),
    "windows-gtx1080-slang-d3d12": ("passed", "failed", "not_adjudicated"),
    "macos-m4-slang-metal": ("passed", "passed", "not_qualified"),
    "ipados-m1-slang-metal": (
        "passed", "passed_with_scoped_scene_debt", "passed"
    ),
    "android-pixel7-slang-vulkan": ("passed", "failed", "not_qualified"),
    "linux-thomas-laptop-slang-cpu": ("passed", "failed", "passed"),
    "linux-rtx3050ti-slang-cuda": (
        "passed", "failed", "passed_with_stalls_retained"
    ),
    "linux-rtx3050ti-slang-vulkan": (
        "passed", "failed", "passed_with_stalls_retained"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def source_path(root: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe evidence path: {path}")
    return root / path


def validate_delivery(
    delivery: Mapping[str, Any] | None = None,
    *,
    root: Path = ROOT,
    verify_files: bool = True,
) -> list[str]:
    errors: list[str] = []
    if delivery is None:
        delivery = load_json(root / DELIVERY_PATH.relative_to(ROOT))

    expect(errors, delivery.get("schema") == "parallel-mater-f8-integrated-delivery/v1",
           "delivery schema drift")
    expect(errors, delivery.get("task") == "F8-31", "delivery task drift")
    expect(errors, delivery.get("status") == "shipped-development-integration",
           "delivery status drift")

    release = delivery.get("release", {})
    expect(errors, release.get("classification") == "development integration delivery",
           "release classification drift")
    expect(errors, release.get("broad_conformance_qualified") is False,
           "broad conformance failure was promoted")
    expect(errors, release.get("broad_60hz_performance_qualified") is False,
           "broad 60 Hz failure was promoted")
    expect(errors, release.get("scene_quality_qualified") is False,
           "scene-quality debt was promoted")
    expect(errors, release.get("normal_creation_selects_single_path") is True,
           "normal creation no longer selects the sole path")
    expect(errors, release.get("user_facing_execution_toggle") is False,
           "a user-facing execution toggle was introduced")
    expect(errors, release.get("api_version") == "0.3.0", "API version drift")
    expect(errors, release.get("driver_interface_version") == 2,
           "driver interface drift")
    expect(errors, release.get("execution_contract") == "fixed-tick-60hz-v1",
           "execution contract drift")

    candidate = delivery.get("candidate", {})
    expect(errors, candidate.get("id") == CANDIDATE_ID, "candidate ID drift")
    expect(errors, candidate.get("source_revision") == SOURCE_REVISION,
           "candidate source revision drift")
    expect(errors, candidate.get("source_tree") == SOURCE_TREE,
           "candidate source tree drift")
    expect(errors, candidate.get("manifest_sha256") == MANIFEST_SHA256,
           "candidate manifest hash drift")
    if verify_files:
        manifest_path = root / str(candidate.get("manifest", ""))
        expect(errors, manifest_path.is_file(), "candidate manifest is missing")
        if manifest_path.is_file():
            expect(errors, file_sha256(manifest_path) == MANIFEST_SHA256,
                   "candidate manifest content drift")

    compatibility = delivery.get("compatibility", {})
    expect(errors, compatibility.get("api_0_2") ==
           "rejected with PM_ERROR_INCOMPATIBLE_VERSION",
           "API 0.2 rejection drift")
    expect(errors, compatibility.get("simulation_create_pnext") == "must be NULL",
           "simulation creation compatibility drift")
    expect(errors, compatibility.get("step_substeps") == 1,
           "fixed substep contract drift")
    expect(errors, set(compatibility.get("removed_public_selection", [])) == {
        "PM_EXT_simulation_profile",
        "PmSimulationProfileCreateInfo",
        "pmEnumerateExecutionProfiles",
    }, "removed execution-selection API drift")

    rows = {row.get("target"): row for row in delivery.get("device_rows", [])}
    expect(errors, set(rows) == set(EXPECTED_DEVICE_VERDICTS),
           "physical device row set drift")
    for target, expected in EXPECTED_DEVICE_VERDICTS.items():
        row = rows.get(target, {})
        actual = (
            row.get("fixed_tick_sdk_gallery_integration"),
            row.get("frozen_correctness"),
            row.get("performance"),
        )
        expect(errors, actual == expected, f"{target} verdict drift")
        expect(errors, bool(row.get("limitations")),
               f"{target} limitations were omitted")

    dispatch = delivery.get("dispatch_reduction", {})
    rigid = dispatch.get("metal_rigid_public_fixture", {})
    expect(errors, rigid.get("baseline_total_dispatches") == 112 and
           rigid.get("final_total_dispatches") == 35,
           "Metal rigid total dispatch reduction drift")
    expect(errors, rigid.get("baseline_domain_dispatches") == 88 and
           rigid.get("final_domain_dispatches") == 11,
           "Metal rigid domain dispatch reduction drift")
    expect(errors, rigid.get("host_waits") == 0 and
           rigid.get("intra_tick_host_readbacks") == 0,
           "Metal rigid host-wait/readback limit drift")
    coupling = dispatch.get("coupling_ceiling", {})
    expect(errors, coupling.get("total_dispatches_maximum") == 512 and
           coupling.get("spatial_builds") == 1 and
           coupling.get("spatial_publications") == 1,
           "coupling work ceiling drift")
    apple = dispatch.get("apple_smoke_soft_body", {})
    expect(errors, apple.get("baseline_dispatches_per_tick") == 1277 and
           apple.get("final_dispatches_per_tick") == 269 and
           apple.get("frame_throughput_claim") == "not established",
           "Apple dispatch reduction or throughput boundary drift")

    gallery = delivery.get("gallery_contract", {})
    expect(errors, gallery.get("display_rates_tested_hz") == [30, 60, 90, 120, 144],
           "Gallery presentation-rate coverage drift")
    expect(errors, gallery.get("maximum_ticks_per_callback") == 1 and
           gallery.get("catch_up_ticks") == 0,
           "Gallery fixed-tick callback policy drift")
    expect(errors, gallery.get("ipados_all_scenes_frames_each") == 120 and
           gallery.get("ipados_peak_memory_footprint_bytes") == 361923352,
           "iPad scene or memory evidence drift")

    timing = delivery.get("timing_policy", {})
    expect(errors, timing.get("desktop_physics_p95_ms_maximum") == 10.0 and
           timing.get("frame_p95_ms_maximum") == 16.666666666666668 and
           timing.get("ipados_physics_p95_ms_maximum") == 8.5 and
           timing.get("simulated_to_wall_ratio_minimum") == 0.99,
           "timing targets were weakened")
    expect(errors, timing.get("remaining_owners") == ["F9-07", "F9-08"],
           "timing debt routing drift")

    notes = delivery.get("release_notes", {})
    expect(errors, notes.get("historical_evidence_relabelled") is False,
           "historical evidence was relabelled")

    if verify_files:
        records = list(delivery.get("evidence_sources", []))
        records += list(delivery.get("packages", {}).get("archives", []))
        for record in records:
            try:
                path = source_path(root, record)
            except ValueError as error:
                errors.append(str(error))
                continue
            expect(errors, path.is_file(), f"missing retained artifact: {record.get('path')}")
            if path.is_file():
                expect(errors, file_sha256(path) == record.get("sha256"),
                       f"retained artifact hash drift: {record.get('path')}")

        ipad = delivery.get("installed_applications", {}).get("ipados", {})
        try:
            runtime_path = source_path(root, {"path": ipad.get("runtime_report")})
        except ValueError as error:
            errors.append(str(error))
        else:
            expect(errors, runtime_path.is_file(), "missing fresh iPad runtime report")
            if runtime_path.is_file():
                expect(errors, file_sha256(runtime_path) == ipad.get("runtime_report_sha256"),
                       "fresh iPad runtime report hash drift")
                runtime = load_json(runtime_path)
                expect(errors, runtime.get("revision") == SOURCE_REVISION and
                       runtime.get("backend") == "metal" and
                       runtime.get("execution_contract") == "fixed-tick-60hz-v1",
                       "fresh iPad runtime identity drift")

        evidence = {
            item["path"]: load_json(root / item["path"])
            for item in delivery.get("evidence_sources", [])
            if str(item.get("path", "")).endswith(".json") and
            (root / str(item.get("path", ""))).is_file()
        }
        expect(errors, evidence[
            "docs/roadmap/evidence/F8-28-CANDIDATE-V10-MACOS-METAL.json"
        ]["verdicts"]["f8_integration_correctness"]["status"] == "passed",
               "macOS passing correctness evidence drift")
        expect(errors, evidence[
            "docs/roadmap/evidence/F8-29-CANDIDATE-V10-IPADOS-METAL.json"
        ]["verdicts"]["f8_integration_correctness"]["status"] == "passed",
               "iPad passing correctness evidence drift")
        expect(errors, evidence[
            "docs/roadmap/evidence/F8-30-CANDIDATE-V10-ANDROID-VULKAN.json"
        ]["verdicts"]["f8_integration_correctness"]["status"] == "failed",
               "Android correctness failure was lost")
        expect(errors, evidence[
            "docs/roadmap/evidence/F8-27-CANDIDATE-V10-WINDOWS-D3D12.json"
        ]["verdicts"]["f8_windows_d3d12_integration"] == "failed",
               "D3D12 integration failure was lost")
        for path in (
            "docs/roadmap/evidence/F8-85-CANDIDATE-V10-LINUX-CPU.json",
            "docs/roadmap/evidence/F8-86-CANDIDATE-V10-LINUX-CUDA.json",
            "docs/roadmap/evidence/F8-87-CANDIDATE-V10-LINUX-VULKAN.json",
        ):
            expect(errors, evidence[path]["verdicts"]["f8_integration_correctness"]
                   ["status"] == "failed",
                   f"Linux frozen-correctness failure was lost: {path}")

        applications = delivery.get("installed_applications", {})
        mac_evidence = evidence[
            "docs/roadmap/evidence/F8-28-CANDIDATE-V10-MACOS-METAL.json"
        ]
        expect(errors, applications["macos"]["build_info_sha256"] ==
               mac_evidence["artifacts"]["installed_build_info_sha256"],
               "installed Mac build-info identity drift")
        expect(errors, applications["macos"]["executable_sha256"] ==
               mac_evidence["artifacts"]["installed_gallery_executable_sha256"],
               "installed Mac executable identity drift")
        ipad_evidence = evidence[
            "docs/roadmap/evidence/F8-29-CANDIDATE-V10-IPADOS-METAL.json"
        ]
        ipad_build = ipad_evidence["results"]["exact_release_build"]
        expect(errors, applications["ipados"]["gallery_executable_sha256"] ==
               ipad_build["gallery_executable_sha256"],
               "installed iPad executable identity drift")
        expect(errors, applications["ipados"]["driver_sha256"] ==
               ipad_build["metal_driver_sha256"],
               "installed iPad driver identity drift")
        expect(errors, applications["ipados"]["runtime_sdk_sha256"] ==
               ipad_build["runtime_sdk_sha256"],
               "installed iPad runtime SDK identity drift")
        android_evidence = evidence[
            "docs/roadmap/evidence/F8-30-CANDIDATE-V10-ANDROID-VULKAN.json"
        ]
        android_build = android_evidence["results"]["clean_android_build_and_deployment"]
        expect(errors, applications["android"]["apk_sha256"] ==
               android_build["apk_sha256"],
               "installed Android APK identity drift")
        expect(errors, applications["android"]["driver_sha256"] ==
               android_build["native_driver_sha256"],
               "installed Android driver identity drift")

        f854 = evidence[
            "docs/roadmap/evidence/F8-54-metal-rigid-dispatch-batching.json"
        ]
        expect(errors, rigid["baseline_total_dispatches"] ==
               f854["pre_change_physical_baseline"]["total_dispatches"] and
               rigid["final_total_dispatches"] ==
               f854["post_change"]["public_production_path"]["total_dispatches"],
               "Metal rigid evidence no longer matches delivery")
        f855 = evidence[
            "docs/roadmap/evidence/F8-55-shared-bvh-coupling-batching.json"
        ]
        expect(errors, coupling["total_dispatches_maximum"] ==
               f855["independent_ceilings"]["total_dispatches"] and
               coupling["coupling_dispatches_maximum"] ==
               f855["independent_ceilings"]["coupling_dispatches"],
               "coupling ceiling evidence no longer matches delivery")
        f856 = evidence[
            "docs/roadmap/evidence/F8-56-apple-gallery-dispatch/summary.json"
        ]
        expect(errors, apple["baseline_dispatches_per_tick"] ==
               f856["macos"]["smoke_soft_body"]["prefix"]["dispatches_per_tick"] and
               apple["final_dispatches_per_tick"] ==
               f856["macos"]["smoke_soft_body"]["postfix"]["dispatches_per_tick"],
               "Apple dispatch evidence no longer matches delivery")

        packages = delivery.get("packages", {}).get("relocation", {})
        windows_cuda = evidence[
            "docs/roadmap/evidence/F8-26-CANDIDATE-V10-WINDOWS-CUDA.json"
        ]
        windows_d3d12 = evidence[
            "docs/roadmap/evidence/F8-27-CANDIDATE-V10-WINDOWS-D3D12.json"
        ]
        expect(errors, packages.get("windows_cuda_checks_passed") ==
               len(windows_cuda["relocated_sdk"]["checks"]),
               "Windows CUDA relocation count drift")
        expect(errors, packages.get("windows_d3d12_checks_passed") ==
               len(windows_d3d12["relocated_sdk"]["checks"]),
               "Windows D3D12 relocation count drift")
        for key, path in (
            ("linux_cpu_checks_passed", "docs/roadmap/evidence/F8-85-CANDIDATE-V10-LINUX-CPU.json"),
            ("linux_cuda_checks_passed", "docs/roadmap/evidence/F8-86-CANDIDATE-V10-LINUX-CUDA.json"),
            ("linux_vulkan_checks_passed", "docs/roadmap/evidence/F8-87-CANDIDATE-V10-LINUX-VULKAN.json"),
        ):
            expect(errors, packages.get(key) ==
                   evidence[path]["results"]["installed_sdk_relocation"]["checks_passed"],
                   f"{key} relocation count drift")

        release_notes = (root / RELEASE_NOTES_PATH.relative_to(ROOT)).read_text(
            encoding="utf-8")
        migration = (root / MIGRATION_PATH.relative_to(ROOT)).read_text(
            encoding="utf-8")
        for phrase in (
            "not a universal 60-FPS claim",
            "There is no application, Gallery, environment, backend, or build",
            "Windows GTX 1080, D3D12",
            "Pixel 7, Vulkan",
            "112 | 35",
            "F9-07 and F9-08",
        ):
            expect(errors, phrase in release_notes,
                   f"release-note disclosure missing: {phrase}")
        expect(errors, "CUDA is neither the default driver" in migration,
               "driver migration default wording drift")

        for directory in (root / "include", root / "src", root / "examples"):
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for symbol in compatibility.get("removed_public_selection", []):
                    expect(errors, symbol not in text,
                           f"retired public selection symbol remains: {path}: {symbol}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DELIVERY_PATH)
    args = parser.parse_args()
    delivery = load_json(args.manifest)
    errors = validate_delivery(delivery)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: {CANDIDATE_ID} F8-31 delivery and retained failures are exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
