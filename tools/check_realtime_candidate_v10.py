#!/usr/bin/env python3
"""Validate the F8-74 replacement candidate and its exact qualification routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from check_realtime_candidate_v9 import (
    closure_identity,
    file_sha256,
    load_json,
    revision_closure_identities,
    safe_revision_file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "conformance/v1/qualification/realtime-candidate-v10.json"
ACTIVE_POINTER_PATH = (
    ROOT / "conformance/v1/qualification/active-realtime-candidate.json"
)
V9_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v9.json"
)
V9_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v9-rejection.json"
)
INVENTORY_PATH = (
    ROOT / "conformance/v1/qualification/kernel-device-obligations-v1.json"
)
TASK_DIR = ROOT / "docs/roadmap/tasks"

ACTIVE_ID = "realtime-single-step-v1-5bacbe2"
ACTIVE_SOURCE_REVISION = "5bacbe2b1e0fbfad565fb9477bb6873396e31d5d"
ACTIVE_SOURCE_TREE = "67f5088c2e9f50938c55856e3683326e42675310"
ACTIVE_CI_RUN_ID = 41708083151
V9_ID = "realtime-single-step-v1-ba67db3"
V9_SHA256 = "4ed4ed7a097928944ac5667e8120e2447affda962a258e56c01dfd0d90d9dd4b"
F8_78_PATH = ROOT / "docs/roadmap/evidence/F8-78-WINDOWS-INTEGRATION.json"
F8_78_SHA256 = "35d04d6deaa34957d7644b82b4ae18b833ab09f382e990f25a3dbe5fc22aacda"
CI_EVIDENCE_PATH = ROOT / "docs/roadmap/evidence/F8-74-PROTECTED-CI.json"
CI_EVIDENCE_SHA256 = (
    "bc7103710434ab9636e2a4a2f72bfec5820669130185333bd6bbf71506f58cfe"
)
V9_FAILURES = {
    "docs/roadmap/evidence/F8-28-CANDIDATE-V9-MACOS-METAL-FAILURE.json": (
        "e244ba5b67dd7c6584e822b57df26a9b2c870098854c08b7b51efa9400936da7"
    ),
    "docs/roadmap/evidence/F8-26-CANDIDATE-V9-WINDOWS-CUDA-FAILURE.json": (
        "b0858b7ff180c76bd1ee15af09c3fd3cb59f229d2ff655852773404c7e4b35f8"
    ),
}

REQUIRED_CLOSURES = {
    "api_abi",
    "kernel",
    "adapter_binding",
    "runtime",
    "schedule_telemetry",
    "contract",
    "public_oracle",
    "work_budget",
    "gallery",
}
REQUIRED_CI_ARTIFACTS = {
    "sdk-install-contracts",
    "sdk-install-contracts-macos-arm64",
    "sdk-install-contracts-windows-x64",
    "shader-portability-ubuntu-latest",
    "shader-portability-windows-latest",
    "shader-portability-macos-26",
    "android-vulkan-preview",
    "android-gallery-ui-evidence",
    "ipad-unsigned-preview",
    "reference-artifacts",
}
REQUIRED_CI_JOBS = {
    "Roadmap lease contract",
    "sdk / sdk (Windows x64)",
    "windows / loader",
    "sdk / sdk",
    "timing / portable-timing (windows-latest)",
    "mobile / ipad",
    "sdk / sdk (macOS ARM64)",
    "timing / portable-timing (macos-latest)",
    "shaders / Shader compilation (macos-26)",
    "mobile / android",
    "shaders / Shader compilation (ubuntu-latest)",
    "reference / reproduce",
    "shaders / Shader compilation (windows-latest)",
    "timing / portable-timing (ubuntu-latest)",
    "mobile / android-ui",
    "shaders / Shader portability required",
    "CI required",
}
TARGET_OWNERS = {
    "windows-gtx1080-slang-cuda": "F8-26",
    "windows-gtx1080-slang-d3d12": "F8-27",
    "linux-thomas-laptop-slang-cpu": "F8-85",
    "linux-rtx3050ti-slang-cuda": "F8-86",
    "linux-rtx3050ti-slang-vulkan": "F8-87",
    "macos-m4-slang-metal": "F8-28",
    "ipados-m1-slang-metal": "F8-29",
    "android-pixel7-slang-vulkan": "F8-30",
}
QUALIFICATION_TASKS = set(TARGET_OWNERS.values())
HISTORICAL_ROWS = {
    "F8-53": (
        "docs/roadmap/evidence/F8-53-CANDIDATE-V9-LINUX-CPU.json",
        "37ea24ee3db589f94481c6cc6ad11a47a846d52a6ae506017103f0de16bfe131",
    ),
    "F8-68": (
        "docs/roadmap/evidence/F8-68-CANDIDATE-V9-LINUX-CUDA.json",
        "7fd249e6665f9674e053de00c48dbf4a76cb0996f32a4cce7c9e9502d1184d36",
    ),
    "F8-71": (
        "docs/roadmap/evidence/F8-71-CANDIDATE-V9-LINUX-VULKAN.json",
        "a5398863cf27a25d9c6438c6723f0f3e99539563dc1c9f0e91124d063eec2104",
    ),
}
INTEGRATION_EVIDENCE = {
    "F8-73": (
        "docs/roadmap/evidence/F8-73-DEVELOPMENT-ACCEPTANCE.json",
        "4a3a172637a077fdf62eff6b53b1a56dd759a3de8abb5fafb5f75abc8233b3fe",
    ),
    "F8-75": (
        "docs/roadmap/evidence/F8-75-WINDOWS-CUDA-FLUID-FIXTURE.json",
        "93e9d5784a1cda13585c7b75574c5d4a0984face1c73ed7bfcc301ccf04c134c",
    ),
    "F8-76": (
        "docs/roadmap/evidence/F8-76-WINDOWS-CUDA-OWNERSHIP-ACCEPTANCE.json",
        "b06d9df95dc6aec5b16dc82a84a3659fce74e1fd9f85ba05bff01b312c6efd48",
    ),
    "F8-77": (
        "docs/roadmap/evidence/F8-77-WINDOWS-PUBLIC-COMPLETION-WAITS-V2.json",
        "2d0f96f180f87d1f6438e8910a2f35493e91522fe8f89b642a964792a4868717",
    ),
    "F8-78": (
        "docs/roadmap/evidence/F8-78-WINDOWS-INTEGRATION.json",
        F8_78_SHA256,
    ),
}


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _revision_tree(root: Path, revision: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "show", "-s", "--format=%T", revision],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_windows_integration(errors: list[str], root: Path) -> None:
    path = root / F8_78_PATH.relative_to(ROOT)
    _expect(errors, path.is_file(), "F8-78 Windows integration evidence is missing")
    if not path.is_file():
        return
    _expect(errors, file_sha256(path) == F8_78_SHA256,
            "F8-78 Windows integration evidence hash drift")
    evidence = load_json(path)
    source = evidence.get("source", {})
    _expect(errors,
            source.get("revision") == ACTIVE_SOURCE_REVISION and
            source.get("tree") == ACTIVE_SOURCE_TREE,
            "F8-78 did not execute the frozen source identity")
    clean = evidence.get("clean_release", {})
    _expect(errors,
            clean.get("registered") == 146 and clean.get("passed") == 146 and
            clean.get("skipped") == 0 and
            clean.get("all_previous_tests_retained") is True,
            "F8-78 clean Release regression suite is incomplete")
    sdk = evidence.get("relocated_sdk", {})
    checks = sdk.get("checks", [])
    _expect(errors,
            sdk.get("status") == "passed" and len(checks) == 12 and
            all(item.get("status") == "passed" for item in checks) and
            sdk.get("source_revision") == ACTIVE_SOURCE_REVISION and
            sdk.get("source_dirty") is False,
            "F8-78 relocated installed SDK gate is incomplete")
    capture = evidence.get("capture", {})
    _expect(errors,
            capture.get("measured_ticks_total") == 72000 and
            capture.get("raw_timing_values_total") == 288000 and
            capture.get("all_samples_and_stalls_retained") is True,
            "F8-78 did not retain the complete raw timing population")
    for mode_name in ("plain", "validation"):
        mode = capture.get("modes", {}).get(mode_name, {})
        _expect(errors,
                mode.get("public_cases_passed") == 30 and
                mode.get("physical_scenarios_passed") == 12 and
                mode.get("correctness_work_recovery_passed") is True,
                f"F8-78 {mode_name} correctness/work/recovery gate failed")
        _expect(errors,
                mode.get("provenance", {}).get("source_commit") ==
                ACTIVE_SOURCE_REVISION,
                f"F8-78 {mode_name} capture source drift")
        workloads = mode.get("workloads", {})
        _expect(errors,
                set(workloads) == {
                    "rigid-wall", "fluid-rigid", "deformable-mixed",
                    "smoke-coupled", "six-family-mixed",
                } and all(
                    workload.get("source") == "installed-public-api" and
                    workload.get("driver_private_state_used") is False and
                    workload.get("warmup_ticks") == 600 and
                    workload.get("measured_ticks") == 7200 and
                    workload.get("failed_ticks") == 0
                    for workload in workloads.values()
                ),
                f"F8-78 {mode_name} workload population drift")
        _expect(errors,
                len(mode.get("canonical_negative_controls", [])) == 5,
                f"F8-78 {mode_name} canonical control count drift")
        controls = mode.get("negative_control_differential_diagnostics", {})
        _expect(errors,
                set(controls) == {
                    "candidate_missing_pair", "atomic_failure_violation",
                    "hidden_retry_violation", "timing_stall_retained",
                    "timing_failed_tick",
                } and all(
                    item.get("injected_defect_rejected_independently") is True
                    for item in controls.values()
                ),
                f"F8-78 {mode_name} negative controls are incomplete")
    _expect(errors, evidence.get("final_performance_qualified") is False,
            "F8-78 incorrectly claims final performance qualification")


def validate_candidate(
    candidate: Mapping[str, Any] | None = None,
    *,
    root: Path = ROOT,
) -> list[str]:
    """Return every F8-74 freeze violation; empty means the freeze is exact."""
    errors: list[str] = []
    candidate_path = root / CANDIDATE_PATH.relative_to(ROOT)
    pointer_path = root / ACTIVE_POINTER_PATH.relative_to(ROOT)
    v9_path = root / V9_CANDIDATE_PATH.relative_to(ROOT)
    rejection_path = root / V9_REJECTION_PATH.relative_to(ROOT)
    if candidate is None:
        if not candidate_path.is_file():
            return ["candidate-v10 manifest is missing"]
        candidate = load_json(candidate_path)

    _expect(errors, candidate.get("id") == ACTIVE_ID,
            "active candidate identity drift")
    _expect(errors, candidate.get("task") == "F8-74",
            "active candidate freeze task drift")
    if pointer_path.is_file() and candidate_path.is_file():
        pointer = load_json(pointer_path)
        _expect(errors,
                pointer.get("candidate_id") == ACTIVE_ID and
                pointer.get("manifest") ==
                CANDIDATE_PATH.relative_to(ROOT).as_posix() and
                pointer.get("manifest_sha256") == file_sha256(candidate_path),
                "active candidate pointer drift")
    else:
        errors.append("active candidate pointer is missing")
    source = candidate.get("source", {})
    _expect(errors, source.get("revision") == ACTIVE_SOURCE_REVISION,
            "active source revision drift")
    _expect(errors, source.get("tree") == ACTIVE_SOURCE_TREE,
            "active source tree drift")
    _expect(errors, _revision_tree(root, ACTIVE_SOURCE_REVISION) == ACTIVE_SOURCE_TREE,
            "frozen Git revision no longer resolves to the recorded tree")

    closures = {
        item.get("category"): item
        for item in source.get("protected_closures", [])
    }
    _expect(errors, set(closures) == REQUIRED_CLOSURES,
            "protected closure category set drift")
    requested = tuple(sorted(
        (category, tuple(item.get("patterns", [])))
        for category, item in closures.items()
        if category in REQUIRED_CLOSURES
    ))
    try:
        identities = revision_closure_identities(
            str(root), ACTIVE_SOURCE_REVISION, requested
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        errors.append(f"could not independently hash frozen closures: {exc}")
        identities = {}
    for category in REQUIRED_CLOSURES:
        actual = closures.get(category, {})
        expected = identities.get(category)
        _expect(errors, expected is not None,
                f"{category} closure could not be independently generated")
        if expected is not None:
            _expect(errors,
                    actual.get("file_count") == expected[0] and
                    actual.get("sha256") == expected[1],
                    f"{category} closure hash drift")

    contract = candidate.get("contract_inputs", {})
    expected_contracts = {
        "scenario_fixtures": "conformance/v1/realtime/scenario-fixtures-v1.json",
        "capture_validator": "conformance/v1/realtime_device_suite.py",
        "capture_producer": "conformance/v1/realtime_device_capture.py",
        "capability_obligations": (
            "conformance/v1/qualification/capability-obligations-v1.json"
        ),
        "kernel_device_obligations": (
            "conformance/v1/qualification/kernel-device-obligations-v1.json"
        ),
        "oracle_suite": "conformance/v1/realtime/oracle-suite-v1.json",
        "case_and_golden_manifest": "conformance/v1/manifest.json",
        "budgets_and_tolerances": (
            "conformance/v1/qualification/realtime-profile-v1.json"
        ),
    }
    for key, relative in expected_contracts.items():
        record = contract.get(key, {})
        expected = safe_revision_file_sha256(
            str(root), ACTIVE_SOURCE_REVISION, relative
        )
        _expect(errors, record.get("path") == relative,
                f"{key} contract path drift")
        _expect(errors, record.get("sha256") == expected,
                f"{key.replace('_', ' ')} contract hash drift")

    surface = candidate.get("capability_surface", {})
    _expect(errors,
            surface.get("api_version") == "0.3" and
            surface.get("driver_abi_version") == 2 and
            surface.get("interaction_capability_count") == 49 and
            surface.get("public_case_count") == 30 and
            surface.get("runtime_variant_count") == 5 and
            surface.get("physical_target_count") == 8,
            "frozen public capability surface drift")
    scenarios = contract.get("scenario_fixtures", {})
    _expect(errors,
            scenarios.get("scenario_count") == 12 and
            scenarios.get("measurement_count") == 37,
            "scenario or measurement count drift")

    compile_evidence = candidate.get("compile_evidence", {})
    actions = compile_evidence.get("github_actions", {})
    ci_path = root / CI_EVIDENCE_PATH.relative_to(ROOT)
    _expect(errors,
            ci_path.is_file() and file_sha256(ci_path) == CI_EVIDENCE_SHA256,
            "protected CI evidence hash drift")
    ci = load_json(ci_path) if ci_path.is_file() else {}
    _expect(errors, compile_evidence.get("qualification_claim") is False,
            "protected CI incorrectly claims physical qualification")
    _expect(errors,
            actions.get("run_id") == ACTIVE_CI_RUN_ID and
            actions.get("event") == "workflow_dispatch" and
            actions.get("result") == "success" and
            actions.get("head_revision") == ACTIVE_SOURCE_REVISION and
            actions.get("head_tree") == ACTIVE_SOURCE_TREE,
            "protected exact-source CI identity drift")
    jobs = {item.get("name"): item for item in actions.get("jobs", [])}
    _expect(errors,
            set(jobs) == REQUIRED_CI_JOBS and
            all(item.get("conclusion") == "success" for item in jobs.values()),
            "protected CI job set is incomplete or failed")
    artifacts = {item.get("name"): item
                 for item in actions.get("artifacts", [])}
    _expect(errors, set(artifacts) == REQUIRED_CI_ARTIFACTS,
            "protected CI artifact set drift")
    for name, artifact in artifacts.items():
        _expect(errors,
                isinstance(artifact.get("id"), int) and
                artifact.get("id", 0) > 0 and
                artifact.get("file_count", 0) > 0,
                f"artifact {name} lacks exact identity")
        for key in ("archive_sha256", "extracted_tree_sha256"):
            _expect(errors,
                    bool(re.fullmatch(r"[0-9a-f]{64}",
                                      str(artifact.get(key, "")))),
                    f"artifact {name} lacks {key}")
    _expect(errors,
            actions.get("jobs") == ci.get("jobs") and
            actions.get("artifacts") == ci.get("artifacts") and
            compile_evidence.get("source_clean_evidence") ==
            ci.get("source_clean_evidence") and
            candidate.get("native_binaries") == ci.get("native_binaries") and
            candidate.get("application_artifacts") ==
            ci.get("application_artifacts") and
            candidate.get("toolchains") == ci.get("toolchains"),
            "candidate does not embed the exact protected CI evidence")
    for platform in ("linux", "macos", "windows"):
        report = compile_evidence.get("source_clean_evidence", {}).get(
            f"{platform}_sdk_report", {}
        )
        _expect(errors,
                report.get("source_revision") == ACTIVE_SOURCE_REVISION and
                report.get("source_dirty") is False,
                f"{platform} SDK CI evidence is not exact and clean")
    for binary in candidate.get("native_binaries", []):
        _expect(errors,
                bool(re.fullmatch(r"[0-9a-f]{64}",
                                  str(binary.get("sha256", "")))),
                f"native binary {binary.get('id')} lacks SHA-256")
    _expect(errors,
            {item.get("runtime_variant")
             for item in candidate.get("native_binaries", [])} >= {
                "cpu-runtime", "cuda-runtime", "vulkan-runtime",
                "d3d12-runtime", "metal-runtime",
            },
            "native binary evidence lacks a runtime variant")
    for binary in candidate.get("native_binaries", []):
        if binary.get("candidate_binary") is True:
            _expect(errors,
                    binary.get("source_revision") == ACTIVE_SOURCE_REVISION,
                    f"candidate binary {binary.get('id')} source drift")
        _expect(errors, binary.get("qualification_claim") is False,
                f"binary {binary.get('id')} claims physical qualification")

    toolchain_scopes = {
        item.get("scope") for item in candidate.get("toolchains", [])
    }
    _expect(errors, {
        "portable-sdk-linux", "portable-sdk-macos", "portable-sdk-windows",
        "shared-physics-shaders", "android-gallery", "ipados-gallery",
        "physical-windows-cuda",
    } <= toolchain_scopes, "candidate toolchain identities are incomplete")
    applications = candidate.get("application_artifacts", [])
    _expect(errors,
            {item.get("platform") for item in applications} >= {
                "linux", "windows", "macos", "ipados", "android",
            },
            "candidate application/deployment history lacks a platform")
    for application in applications:
        _expect(errors,
                bool(re.fullmatch(r"[0-9a-f]{64}",
                                  str(application.get("sha256", "")))) and
                application.get("qualification_claim") is False,
                f"application {application.get('id')} identity drift")
        if application.get("candidate_binary") is True:
            _expect(errors,
                    application.get("source_revision") == ACTIVE_SOURCE_REVISION,
                    f"candidate application {application.get('id')} source drift")
    ipad = next(
        (item for item in applications if item.get("id") == "ci-ipados-gallery"),
        {},
    )
    _expect(errors,
            ipad.get("embedded_build_info", {}).get("revision") ==
            ACTIVE_SOURCE_REVISION and
            ipad.get("embedded_build_info", {}).get("dirty") is False,
            "iPad candidate application embeds the wrong source identity")

    _expect(errors, v9_path.is_file() and file_sha256(v9_path) == V9_SHA256,
            "candidate-v9 immutable manifest changed")
    for relative, digest in V9_FAILURES.items():
        path = root / relative
        _expect(errors, path.is_file() and file_sha256(path) == digest,
                f"candidate-v9 failure evidence changed: {relative}")
    _expect(errors, rejection_path.is_file(),
            "candidate-v9 supersession record is missing")
    if rejection_path.is_file():
        rejection = load_json(rejection_path)
        replacement = rejection.get("replacement", {})
        _expect(errors,
                rejection.get("candidate_id") == V9_ID and
                rejection.get("candidate_manifest_sha256") == V9_SHA256 and
                rejection.get("evidence_union_allowed") is False,
                "candidate-v9 supersession identity drift")
        _expect(errors,
                replacement.get("freeze_task") == "F8-74" and
                replacement.get("candidate_id") == ACTIVE_ID and
                replacement.get("source_revision") == ACTIVE_SOURCE_REVISION and
                replacement.get("source_tree") == ACTIVE_SOURCE_TREE and
                replacement.get("ci_run_id") == ACTIVE_CI_RUN_ID,
                "candidate-v9 does not route to exact candidate-v10")
        if candidate_path.is_file():
            _expect(errors,
                    replacement.get("candidate_manifest_sha256") ==
                    file_sha256(candidate_path),
                    "candidate-v9 replacement manifest hash drift")

    supersedes = candidate.get("supersedes", {})
    _expect(errors,
            supersedes.get("candidate_id") == V9_ID and
            supersedes.get("manifest_sha256") == V9_SHA256,
            "candidate-v10 does not supersede exact candidate-v9")
    integration = {
        item.get("task"): item for item in source.get("integration_evidence", [])
    }
    for task, (relative, digest) in INTEGRATION_EVIDENCE.items():
        path = root / relative
        item = integration.get(task, {})
        _expect(errors,
                item.get("path") == relative and item.get("sha256") == digest and
                path.is_file() and file_sha256(path) == digest,
                f"{task} integration evidence drift")
    _validate_windows_integration(errors, root)

    history = {
        item.get("task"): item
        for item in candidate.get("historical_candidate_rows", [])
    }
    for task, (relative, digest) in HISTORICAL_ROWS.items():
        path = root / relative
        item = history.get(task, {})
        _expect(errors,
                item.get("candidate") == V9_ID and
                item.get("path") == relative and
                item.get("sha256") == digest and
                item.get("status") == "completed-history-requalify-v10" and
                path.is_file() and file_sha256(path) == digest,
                f"{task} candidate-v9 history drift")

    debt = candidate.get("timing_debt", {})
    _expect(errors,
            debt.get("final_performance_qualified") is False and
            debt.get("owners") == ["F9-07", "F9-08"] and
            debt.get("source_evidence") ==
            F8_78_PATH.relative_to(ROOT).as_posix() and
            debt.get("windows_six_family_physics_p95_ms") == {
                "plain": 18.5305, "validation": 20.6715,
            } and
            debt.get("windows_six_family_simulated_wall_ratio") == {
                "plain": 0.989253, "validation": 0.837211,
            } and
            debt.get("validation_deformable_physics_p95_ms") == 11.1551,
            "retained Windows timing debt was weakened or misclassified")
    verdicts = candidate.get("verdicts", {})
    _expect(errors,
            verdicts.get("f8_integration_correctness", {}).get("status") ==
            "passed-frozen-source-device-rows-pending" and
            verdicts.get("final_performance", {}).get("status") ==
            "not-qualified" and
            verdicts.get("final_performance", {}).get("owners") ==
            ["F9-07", "F9-08"],
            "integration and final-performance verdicts are not separated")

    inventory = load_json(root / INVENTORY_PATH.relative_to(ROOT))
    targets = {item.get("id"): item
               for item in candidate.get("physical_targets", [])}
    _expect(errors, set(targets) == set(TARGET_OWNERS),
            "physical target coverage drift")
    for target_id, owner in TARGET_OWNERS.items():
        target = targets.get(target_id, {})
        _expect(errors,
                target.get("qualification_owner") == owner and
                target.get("status") == "pending-v10-integration-qualification",
                f"{target_id} replacement routing drift")
    _expect(errors,
            set(TARGET_OWNERS) == {
                item.get("id") for item in inventory.get("physical_targets", [])
            },
            "candidate target set differs from the frozen inventory")
    routing = candidate.get("qualification_routing", {})
    _expect(errors,
            routing.get("accepted_candidate") == ACTIVE_ID and
            routing.get("evidence_union_allowed") is False and
            set(routing.get("tasks", [])) == QUALIFICATION_TASKS,
            "qualification routing drift")

    candidate_sha256 = file_sha256(candidate_path) if candidate_path.is_file() else ""
    for task_id in sorted(QUALIFICATION_TASKS):
        task_path = root / TASK_DIR.relative_to(ROOT) / f"{task_id}.json"
        _expect(errors, task_path.is_file(), f"{task_id} routing task is missing")
        if not task_path.is_file():
            continue
        task = load_json(task_path)
        scope = task.get("scope", "")
        _expect(errors, "F8-74" in task.get("dependencies", []),
                f"{task_id} does not depend on F8-74")
        _expect(errors,
                all(value in scope for value in (
                    ACTIVE_ID, ACTIVE_SOURCE_REVISION, ACTIVE_SOURCE_TREE,
                    candidate_sha256, str(ACTIVE_CI_RUN_ID), V9_ID,
                    "cannot be unioned", "F9-07/F9-08",
                )),
                f"{task_id} does not bind the complete candidate-v10 identity")
    for task_id, historical in (
        ("F8-85", "F8-53"), ("F8-86", "F8-68"), ("F8-87", "F8-71")
    ):
        task = load_json(root / TASK_DIR.relative_to(ROOT) / f"{task_id}.json")
        _expect(errors,
                task.get("status") == "open" and task.get("lease") is None and
                set(task.get("dependencies", [])) == {historical, "F8-74"},
                f"{task_id} is not a bounded replacement leaf")
    for task_id in ("F8-31", "F8-99"):
        dependencies = set(load_json(
            root / TASK_DIR.relative_to(ROOT) / f"{task_id}.json"
        ).get("dependencies", []))
        _expect(errors, {"F8-85", "F8-86", "F8-87"} <= dependencies,
                f"{task_id} lacks candidate-v10 replacement dependencies")

    audit = candidate.get("sole_path_audit", {})
    audit_path = root / str(audit.get("audit_evidence", ""))
    _expect(errors,
            audit.get("production_schedule_count") == 1 and
            audit.get("schedule") == "fixed-tick-60hz-v1" and
            audit.get("legacy_selection_found") is False and
            audit.get("blanket_integration_rejection_found") is False and
            audit_path.is_file(),
            "sole-path audit is incomplete")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = validate_candidate(root=args.root.resolve())
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: {ACTIVE_ID} protected closure and routing are exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
