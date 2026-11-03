#!/usr/bin/env python3
"""Generate and route the F8-74 integration candidate from immutable evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import generate_realtime_candidate as v9


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_PATH = ROOT / "conformance/v1/qualification/realtime-candidate-v9.json"
OUTPUT_PATH = ROOT / "conformance/v1/qualification/realtime-candidate-v10.json"
SOURCE_REVISION = "5bacbe2b1e0fbfad565fb9477bb6873396e31d5d"
SOURCE_TREE = "67f5088c2e9f50938c55856e3683326e42675310"
CANDIDATE_ID = "realtime-single-step-v1-5bacbe2"
PREVIOUS_ID = "realtime-single-step-v1-ba67db3"
PREVIOUS_SHA256 = "4ed4ed7a097928944ac5667e8120e2447affda962a258e56c01dfd0d90d9dd4b"
LEASE_TOKEN = "590b4cc7-7e4d-4dae-a7ae-db76f9e2ea11"
PROTECTED_CI_RUN_ID = 41708083151

EVIDENCE = {
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
        "35d04d6deaa34957d7644b82b4ae18b833ab09f382e990f25a3dbe5fc22aacda",
    ),
}
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
V9_FAILURES = {
    "macos-metal": (
        "docs/roadmap/evidence/F8-28-CANDIDATE-V9-MACOS-METAL-FAILURE.json",
        "e244ba5b67dd7c6584e822b57df26a9b2c870098854c08b7b51efa9400936da7",
    ),
    "windows-cuda": (
        "docs/roadmap/evidence/F8-26-CANDIDATE-V9-WINDOWS-CUDA-FAILURE.json",
        "b0858b7ff180c76bd1ee15af09c3fd3cb59f229d2ff655852773404c7e4b35f8",
    ),
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
REQUIRED_CI_ARTIFACTS = v9.REQUIRED_CI_ARTIFACTS
REQUIRED_CI_JOBS = v9.REQUIRED_CI_JOBS


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision_file_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SOURCE_REVISION}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def revision_file_sha256(relative: str) -> str:
    return hashlib.sha256(revision_file_bytes(relative)).hexdigest()


def revision_json(relative: str) -> dict[str, Any]:
    return json.loads(revision_file_bytes(relative))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_immutable_inputs() -> None:
    require(file_sha256(PREVIOUS_PATH) == PREVIOUS_SHA256,
            "candidate v9 changed; refusing to derive v10")
    for label, (relative, digest) in {**EVIDENCE, **HISTORICAL_ROWS}.items():
        require(file_sha256(ROOT / relative) == digest,
                f"{label} evidence changed; refusing to freeze v10")
    for label, (relative, digest) in V9_FAILURES.items():
        require(file_sha256(ROOT / relative) == digest,
                f"candidate-v9 {label} failure changed")
    actual_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", SOURCE_REVISION],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(actual_tree == SOURCE_TREE, "frozen source tree identity drift")


def verify_ci(ci: dict[str, Any]) -> None:
    run = ci.get("run", {})
    require(run.get("id") == PROTECTED_CI_RUN_ID, "unexpected protected CI run")
    require(run.get("event") == "workflow_dispatch", "protected CI was not manual")
    require(run.get("head_revision") == SOURCE_REVISION,
            "protected CI did not execute frozen source")
    require(run.get("head_tree") == SOURCE_TREE, "protected CI tree drift")
    require(run.get("result") == "success", "protected CI did not pass")
    require({item.get("name") for item in ci.get("artifacts", [])} ==
            REQUIRED_CI_ARTIFACTS, "protected CI artifact set is incomplete")
    jobs = {item.get("name"): item for item in ci.get("jobs", [])}
    require(set(jobs) == REQUIRED_CI_JOBS and
            all(item.get("conclusion") == "success" for item in jobs.values()),
            "protected CI job set is incomplete or failed")


def generate(ci: dict[str, Any]) -> dict[str, Any]:
    verify_immutable_inputs()
    verify_ci(ci)
    candidate = copy.deepcopy(load_json(PREVIOUS_PATH))
    candidate.update({
        "id": CANDIDATE_ID,
        "task": "F8-74",
        "frozen_at": "2026-11-03",
        "status": "frozen-awaiting-integration-qualification",
        "lease_token": LEASE_TOKEN,
        "supersedes": {
            "candidate_id": PREVIOUS_ID,
            "manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": PREVIOUS_SHA256,
            "supersession_record": (
                "conformance/v1/qualification/"
                "realtime-candidate-v9-rejection.json"
            ),
        },
    })

    closures = candidate["source"]["protected_closures"]
    requested = tuple(sorted(
        (item["category"], tuple(item["patterns"])) for item in closures
    ))
    identities = v9.generate_closure_identities(SOURCE_REVISION, requested)
    for closure in closures:
        count, digest = identities[closure["category"]]
        closure["file_count"] = count
        closure["sha256"] = digest
    candidate["source"] = {
        "revision": SOURCE_REVISION,
        "tree": SOURCE_TREE,
        "definition": (
            "Exact clean source physically exercised by F8-78 after the merged "
            "F8-73 Metal/shared-Gallery repair and F8-75/F8-76/F8-77 Windows "
            "ownership, metadata and public-completion repairs. Freeze metadata "
            "and qualification routing are outside this executable identity."
        ),
        "integration_evidence": [
            {"task": task, "path": relative, "sha256": digest}
            for task, (relative, digest) in EVIDENCE.items()
        ],
        "hash_method": (
            "SHA-256 over sorted repository-relative path + NUL + file bytes + NUL"
        ),
        "protected_closures": closures,
    }
    candidate["sole_path_audit"]["audit_evidence"] = (
        "docs/roadmap/evidence/F8-74-SOLE-PATH-AUDIT.json"
    )

    fixture_path = "conformance/v1/realtime/scenario-fixtures-v1.json"
    validator_path = "conformance/v1/realtime_device_suite.py"
    producer_path = "conformance/v1/realtime_device_capture.py"
    profile_path = "conformance/v1/qualification/realtime-profile-v1.json"
    capability_path = "conformance/v1/qualification/capability-obligations-v1.json"
    inventory_path = "conformance/v1/qualification/kernel-device-obligations-v1.json"
    oracle_path = "conformance/v1/realtime/oracle-suite-v1.json"
    manifest_path = "conformance/v1/manifest.json"
    fixtures = revision_json(fixture_path)
    inventory = revision_json(inventory_path)
    kernel_manifest = revision_file_bytes(
        "cmake/SlangKernelManifest.cmake").decode("utf-8")
    modules = v9.cmake_set(kernel_manifest, "PARALLEL_MATER_SLANG_KERNEL_MODULES")
    entry_count = sum(len(v9.cmake_set(
        kernel_manifest, f"PARALLEL_MATER_SLANG_{module}_ENTRIES"
    )) for module in modules)
    candidate["kernel_coverage"].update({
        "canonical_module_count": len(modules),
        "canonical_entry_count": entry_count,
        "kernel_group_count": len(inventory["kernel_groups"]),
    })
    candidate["contract_inputs"] = {
        "scenario_fixtures": {
            "path": fixture_path,
            "sha256": revision_file_sha256(fixture_path),
            "scenario_count": len(fixtures["scenarios"]),
            "measurement_count": sum(
                len(items) for items in fixtures["measurement_algorithms"].values()
            ),
        },
        "capture_validator": {"path": validator_path,
                              "sha256": revision_file_sha256(validator_path),
                              "schema": "parallel-mater-realtime-device-suite-capture/v1"},
        "capture_producer": {"path": producer_path,
                             "sha256": revision_file_sha256(producer_path),
                             "schema": "parallel-mater-realtime-device-suite-capture/v1"},
        "capability_obligations": {"path": capability_path,
                                   "sha256": revision_file_sha256(capability_path),
                                   "rows": 49},
        "kernel_device_obligations": {"path": inventory_path,
                                      "sha256": revision_file_sha256(inventory_path),
                                      "runtime_variants": 5,
                                      "physical_targets": len(inventory["physical_targets"])},
        "oracle_suite": {"path": oracle_path,
                         "sha256": revision_file_sha256(oracle_path),
                         "public_cases": len(inventory["public_case_sets"]["all"])},
        "case_and_golden_manifest": {"path": manifest_path,
                                     "sha256": revision_file_sha256(manifest_path)},
        "budgets_and_tolerances": {"path": profile_path,
                                   "sha256": revision_file_sha256(profile_path)},
    }

    candidate["native_binaries"] = ci["native_binaries"]
    candidate["compile_evidence"] = {
        "qualification_claim": False,
        "rule": (
            "Cross-platform CI proves build, packaging and protected contracts; "
            "physical rows remain separate exact-candidate qualifications."
        ),
        "github_actions": {
            "workflow": ci["run"]["workflow"],
            "run_id": ci["run"]["id"],
            "run_url": ci["run"]["url"],
            "result": ci["run"]["result"],
            "head_revision": ci["run"]["head_revision"],
            "head_tree": ci["run"]["head_tree"],
            "checkout_merge_revision": ci["run"]["head_revision"],
            "jobs": ci["jobs"],
            "artifacts": ci["artifacts"],
            "event": ci["run"]["event"],
            "archive_hash_method": "SHA-256 over downloaded GitHub artifact ZIP bytes",
            "tree_hash_method": (
                "SHA-256 over sorted extracted relative path + NUL + file bytes + NUL"
            ),
        },
        "source_clean_evidence": ci["source_clean_evidence"],
    }
    candidate["application_artifacts"] = ci["application_artifacts"]
    candidate["toolchains"] = ci["toolchains"]
    prerequisites = {item["task"]: item for item in candidate["prerequisite_evidence"]}
    for task, (relative, digest) in EVIDENCE.items():
        prerequisites[task] = {"task": task, "path": relative, "sha256": digest}
    candidate["prerequisite_evidence"] = [
        prerequisites[key] for key in sorted(prerequisites)
    ]
    candidate["execution_evidence"] = EVIDENCE["F8-78"][0]
    candidate["historical_candidate_rows"] = [
        {"task": task, "candidate": PREVIOUS_ID, "path": relative,
         "sha256": digest, "status": "completed-history-requalify-v10"}
        for task, (relative, digest) in HISTORICAL_ROWS.items()
    ]
    candidate["timing_debt"] = {
        "final_performance_qualified": False,
        "policy": "docs/roadmap/PHYSICS-QUALITY-DEVELOPMENT.md",
        "owners": ["F9-07", "F9-08"],
        "source_evidence": EVIDENCE["F8-78"][0],
        "windows_six_family_physics_p95_ms": {"plain": 18.5305, "validation": 20.6715},
        "windows_six_family_simulated_wall_ratio": {"plain": 0.989253, "validation": 0.837211},
        "validation_deformable_physics_p95_ms": 11.1551,
    }
    candidate["verdicts"] = {
        "f8_integration_correctness": {
            "status": "passed-frozen-source-device-rows-pending",
            "basis": (
                "Exact-source protected CI and F8-73/F8-75/F8-76/F8-77/F8-78 "
                "correctness, recovery and independent work-budget evidence."
            ),
        },
        "final_performance": {
            "status": "not-qualified",
            "owners": ["F9-07", "F9-08"],
            "basis": (
                "Retained Windows timing target misses and timing-contaminated "
                "canonical baselines remain explicit F9 debt."
            ),
        },
    }
    for target in candidate["physical_targets"]:
        target["qualification_owner"] = TARGET_OWNERS[target["id"]]
        target["status"] = "pending-v10-integration-qualification"
    for variant in candidate["runtime_variants"]:
        variant["qualification_owners"] = sorted({
            TARGET_OWNERS[target["id"]]
            for target in candidate["physical_targets"]
            if target["variant"] == variant["id"]
        })
    candidate["qualification_routing"].update({
        "accepted_candidate": CANDIDATE_ID,
        "tasks": sorted(QUALIFICATION_TASKS),
        "interim_cpu_evidence": (
            "Candidates v1-v9 and all observations remain immutable history. "
            "F8-53, F8-68 and F8-71 completed candidate-v9 rows; F8-85, F8-86 "
            "and F8-87 independently requalify CPU, CUDA and Vulkan on v10."
        ),
        "rule": (
            "Every F8 integration row names this exact candidate, source, tree, "
            "CI run, manifest hash and protected closures. No evidence union. "
            "Timing-only debt remains F9-07/F9-08."
        ),
    })
    return candidate


def common_scope(candidate_sha256: str) -> str:
    return (
        f"F8 INTEGRATION GATE: qualify only {CANDIDATE_ID}, manifest "
        f"{candidate_sha256}, source {SOURCE_REVISION}, tree {SOURCE_TREE} and "
        f"exact-source CI run {PROTECTED_CI_RUN_ID}. This is the API 0.3/driver "
        "ABI 2 sole fixed-tick SDK/Gallery integration contract with 49 "
        "capabilities, 30 public cases, 12 scenarios and 37 measurements. "
        f"Candidate v9 {PREVIOUS_ID} remains immutable rejected/superseded "
        "history after its Metal and Windows failures; its completed CPU, CUDA "
        "and Vulkan rows cannot be unioned with v10. Preserve one 1/60 step, one "
        "spatial publication, all correctness/recovery/work budgets and the "
        "2026-11-03 timing deferral. Timing-only misses belong to F9-07/F9-08. "
    )


def common_acceptance(candidate_sha256: str) -> list[str]:
    return [
        f"F8-74 freezes exact candidate {CANDIDATE_ID}, manifest {candidate_sha256}, source {SOURCE_REVISION}, tree {SOURCE_TREE} and protected CI run {PROTECTED_CI_RUN_ID}; execute only that identity and never union historical observations.",
        "Run every applicable kernel/variant, domain, coupling, lifecycle and negative case. Complete execution, physical/numerical correctness, capability, independent work-budget and recovery assertions must pass; classify every failure.",
        "Prove one dt=1/60 advance and one spatial publication per tick; verify interpolation, overload slow-motion, no catch-up and fixed dispatch/iteration/submission/readback/synchronization budgets.",
        "Retain all frozen workload timing samples and stalls. Report CPU/GPU/physics/frame p50/p95/p99 and simulated/wall ratio. Timing target misses are disclosed F9-07/F9-08 debt and are not F8 integration failures.",
        "Attach reviewable artifacts identifying exact candidate, binary/deployment, device/OS/GPU/driver/toolchain and manifest hashes. Preserve failures; do not weaken physics, workload, tolerance, timeout or advertised capability.",
        "Use normal public simulation creation with the sole schedule and no execution-profile opt-in. Verify all applicable supported domains/interactions and installed Gallery behavior.",
    ]


def replacement_task(task_id: str, title: str, historical: str,
                     requires: str, target: str, candidate_sha256: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "status": "open",
        "dependencies": [historical, "F8-74"],
        "locks": [f"f8-v10-{target}-qualification"],
        "requires": [requires],
        "scope": common_scope(candidate_sha256) + (
            f"Requalify the {target} row from scratch. {historical} remains "
            "completed candidate-v9 history and supplies no v10 pass."
        ),
        "acceptance": common_acceptance(candidate_sha256),
        "evidence": [],
        "lease": None,
        "history": [],
    }


def route_candidate(candidate: dict[str, Any]) -> None:
    candidate_sha256 = file_sha256(OUTPUT_PATH)
    qualification_dir = ROOT / "conformance/v1/qualification"
    write_json(qualification_dir / "active-realtime-candidate.json", {
        "schema": "parallel-mater-active-realtime-candidate/v1",
        "candidate_id": CANDIDATE_ID,
        "manifest": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": candidate_sha256,
    })
    write_json(qualification_dir / "realtime-candidate-v9-rejection.json", {
        "schema": "parallel-mater-realtime-candidate-rejection/v1",
        "candidate_id": PREVIOUS_ID,
        "candidate_manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
        "candidate_manifest_sha256": PREVIOUS_SHA256,
        "status": "rejected-required-metal-and-windows-integration",
        "failure_evidence": [
            {"label": label, "path": relative, "sha256": digest}
            for label, (relative, digest) in V9_FAILURES.items()
        ],
        "accepted_repairs": [
            {"task": task, "path": relative, "sha256": digest}
            for task, (relative, digest) in EVIDENCE.items()
        ],
        "completed_historical_rows": candidate["historical_candidate_rows"],
        "replacement": {
            "status": "frozen-awaiting-integration-qualification",
            "freeze_task": "F8-74",
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "candidate_manifest_sha256": candidate_sha256,
            "source_revision": SOURCE_REVISION,
            "source_tree": SOURCE_TREE,
            "ci_run_id": PROTECTED_CI_RUN_ID,
        },
        "invalidated_physical_rows": [
            {"target": item["id"],
             "qualification_owner": item["qualification_owner"],
             "status": "invalidated-requalify-replacement"}
            for item in candidate["physical_targets"]
        ],
        "evidence_union_allowed": False,
        "timing_policy": "F9-07/F9-08 retain final performance qualification",
    })

    tasks_dir = ROOT / "docs/roadmap/tasks"
    device_suffix = {
        "F8-26": "Execute every registered physical Windows CUDA tuple.",
        "F8-27": "Execute every registered physical Windows D3D12 tuple.",
        "F8-28": "Execute every registered physical macOS Metal tuple and installed Gallery.",
        "F8-29": "Execute every registered physical iOS/iPadOS Metal tuple; simulator is not a substitute.",
        "F8-30": "Execute every registered physical Android Vulkan tuple and sustained thermal capture.",
    }
    for task_id, suffix in device_suffix.items():
        path = tasks_dir / f"{task_id}.json"
        task = load_json(path)
        require(task["status"] == "open" and task["lease"] is None,
                f"{task_id} must be unleased before routing")
        task["scope"] = common_scope(candidate_sha256) + suffix
        task["acceptance"] = common_acceptance(candidate_sha256)
        write_json(path, task)

    replacements = {
        "F8-85": replacement_task(
            "F8-85", "Requalify candidate v10 on Linux CPU", "F8-53",
            "ThomasLaptop or equivalent registered Linux CPU device", "linux-cpu",
            candidate_sha256),
        "F8-86": replacement_task(
            "F8-86", "Requalify candidate v10 on Linux CUDA", "F8-68",
            "Physical Linux RTX 3050 Ti CUDA device", "linux-cuda",
            candidate_sha256),
        "F8-87": replacement_task(
            "F8-87", "Requalify candidate v10 on Linux Vulkan", "F8-71",
            "Physical Linux RTX 3050 Ti Vulkan device", "linux-vulkan",
            candidate_sha256),
    }
    for task_id, task in replacements.items():
        path = tasks_dir / f"{task_id}.json"
        if path.exists():
            current = load_json(path)
            require(
                current.get("status") == "open" and current.get("lease") is None,
                f"{task_id} is active and cannot be regenerated",
            )
        write_json(path, task)
    for task_id in ("F8-31", "F8-99"):
        path = tasks_dir / f"{task_id}.json"
        task = load_json(path)
        task["dependencies"] = sorted(set(task["dependencies"]) | set(replacements))
        write_json(path, task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-evidence", type=Path, required=True)
    parser.add_argument("--route", action="store_true")
    args = parser.parse_args()
    ci = load_json(args.ci_evidence.resolve())
    candidate = generate(ci)
    write_json(OUTPUT_PATH, candidate)
    if args.route:
        route_candidate(candidate)
    print(f"Generated {CANDIDATE_ID}: {file_sha256(OUTPUT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
