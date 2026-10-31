#!/usr/bin/env python3
"""Generate the F8-67 realtime candidate from immutable source evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v7.json"
)
OUTPUT_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v8.json"
)
SOURCE_REVISION = "c05d3207138c6e969e1347558c08601aeaf1cc07"
SOURCE_TREE = "340ab85997363f328e25f1c549114c5b44571bc3"
CANDIDATE_ID = "realtime-single-step-v1-c05d320"
PREVIOUS_ID = "realtime-single-step-v1-05f8b04"
PREVIOUS_SHA256 = (
    "15f168583cdd16afbf0fa3d7a26a86349dab206e6aab87e9b3f1b18bdcbb8b8f"
)
LEASE_TOKEN = "fb93fd79-4318-410d-8479-2a66c7af9cd6"
F8_66_EVIDENCE = (
    "docs/roadmap/evidence/F8-66-LINUX-VULKAN-REPAIR.json"
)
F8_65_FREEZE_EVIDENCE = (
    "docs/roadmap/evidence/F8-65-CANDIDATE-V7-FREEZE.json"
)
V7_CUDA_EVIDENCE = (
    "docs/roadmap/evidence/F8-24-CANDIDATE-V7-LINUX-CUDA.json"
)
V7_CUDA_EVIDENCE_SHA256 = (
    "49a2e57dd43af1a5735e3b9838d6280de80d4f688ff61ca56035ecc173881764"
)
V7_CUDA_ARCHIVE = (
    "docs/roadmap/evidence/F8-24-CANDIDATE-V7-LINUX-CUDA.tar.zst"
)
V7_CUDA_ARCHIVE_SHA256 = (
    "127b00849c4744faba0543d3eb15971ea6b273e2dd2c0064d99417c538bd1e3c"
)
V7_FAILURE_EVIDENCE = (
    "docs/roadmap/evidence/F8-25-CANDIDATE-V7-LINUX-VULKAN-FAILURE.json"
)
V7_FAILURE_SHA256 = (
    "2adbc20a1ef42d2d6f2c0f3f0e4d977aeeb14d65f72219fafac42cad9efbf6db"
)
V7_FAILURE_ARCHIVE = (
    "docs/roadmap/evidence/F8-25-CANDIDATE-V7-LINUX-VULKAN-FAILURE.tar.zst"
)
V7_FAILURE_ARCHIVE_SHA256 = (
    "7facdbf9d31f9abe4a01a7ade4dc324f76524299c8366fe1b900b53e9c2907c2"
)
PROTECTED_CI_RUN_ID = 41246222115
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def cmake_set(text: str, name: str) -> list[str]:
    match = re.search(rf"set\({re.escape(name)}\s+(.*?)\)", text, re.DOTALL)
    require(match is not None, f"missing CMake set {name}")
    return re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(1))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def generate_closure_identities(
    revision: str,
    requested: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, tuple[int, str]]:
    """Generate identities without using the validation checker's hash code."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="parallel-mater-generate-") as raw:
        extracted = Path(raw)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(extracted)
        result: dict[str, tuple[int, str]] = {}
        for category, patterns in requested:
            files: set[Path] = set()
            for pattern in patterns:
                files.update(
                    path for path in extracted.glob(pattern) if path.is_file()
                )
            digest = hashlib.sha256()
            for path in sorted(
                files, key=lambda item: item.relative_to(extracted).as_posix()
            ):
                digest.update(
                    path.relative_to(extracted).as_posix().encode("utf-8")
                )
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            result[category] = len(files), digest.hexdigest()
        return result


def generate(ci: dict[str, Any]) -> dict[str, Any]:
    require(file_sha256(PREVIOUS_PATH) == PREVIOUS_SHA256,
            "candidate v7 changed; refusing to derive v8")
    require(file_sha256(ROOT / V7_CUDA_EVIDENCE) == V7_CUDA_EVIDENCE_SHA256,
            "candidate-v7 CUDA evidence changed; refusing to derive v8")
    require(file_sha256(ROOT / V7_CUDA_ARCHIVE) == V7_CUDA_ARCHIVE_SHA256,
            "candidate-v7 CUDA archive changed; refusing to derive v8")
    require(file_sha256(ROOT / V7_FAILURE_EVIDENCE) == V7_FAILURE_SHA256,
            "candidate-v7 Vulkan failure evidence changed; refusing to derive v8")
    require(file_sha256(ROOT / V7_FAILURE_ARCHIVE) ==
            V7_FAILURE_ARCHIVE_SHA256,
            "candidate-v7 Vulkan failure archive changed; refusing to derive v8")
    actual_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", SOURCE_REVISION],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(actual_tree == SOURCE_TREE, "frozen source tree identity drift")

    run = ci.get("run", {})
    require(run.get("id") == PROTECTED_CI_RUN_ID,
            "unexpected protected CI run")
    require(run.get("event") == "workflow_dispatch",
            "protected CI was not manually dispatched")
    require(run.get("head_revision") == SOURCE_REVISION,
            "protected CI did not execute the frozen source")
    require(run.get("head_tree") == SOURCE_TREE,
            "protected CI source tree drift")
    require(run.get("result") == "success", "protected CI did not pass")
    require({item.get("name") for item in ci.get("artifacts", [])} ==
            REQUIRED_CI_ARTIFACTS,
            "protected CI artifact set is incomplete")
    require({item.get("name") for item in ci.get("jobs", [])} ==
            REQUIRED_CI_JOBS and
            all(item.get("conclusion") == "success"
                for item in ci.get("jobs", [])),
            "protected CI job set is incomplete or failed")

    candidate = copy.deepcopy(load_json(PREVIOUS_PATH))
    candidate.update({
        "id": CANDIDATE_ID,
        "task": "F8-67",
        "frozen_at": "2026-10-31",
        "lease_token": LEASE_TOKEN,
        "supersedes": {
            "candidate_id": PREVIOUS_ID,
            "manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": PREVIOUS_SHA256,
            "supersession_record": (
                "conformance/v1/qualification/"
                "realtime-candidate-v7-rejection.json"
            ),
        },
    })

    closures = candidate["source"]["protected_closures"]
    for closure in closures:
        if closure["category"] == "api_abi":
            closure["patterns"] = [
                "include/parallel_mater/**/*",
                "include/parallel_mater_scene/**/*",
                "registry/pm.xml",
                "src/loader/**/*",
                "cmake/ParallelMaterLoader.cmake",
                "cmake/ParallelMaterConfig.cmake.in",
                "MIGRATION.md",
            ]
        elif closure["category"] == "adapter_binding":
            closure["patterns"] = [
                "src/api/**/*",
                "src/layers/**/*",
                "src/driver/**/*",
                "src/device/**/*",
                "tools/generate_cpu_dispatch_stubs.py",
                "tools/generate_dxil_dispatch.py",
                "tools/generate_vulkan_dispatch.py",
                "tests/solver/test_dxil_bindings.py",
            ]
        elif closure["category"] == "public_oracle":
            closure["patterns"] = [
                "conformance/v1/**/*",
                "tests/sdk/realtime_profile.c",
                "tests/api/native_realtime_runtime_tests.cpp",
                "tests/solver/check_slang_architecture.py",
            ]
    requested = tuple(sorted(
        (item["category"], tuple(item["patterns"])) for item in closures
    ))
    identities = generate_closure_identities(SOURCE_REVISION, requested)
    for closure in closures:
        count, digest = identities[closure["category"]]
        closure["file_count"] = count
        closure["sha256"] = digest

    candidate["source"] = {
        "revision": SOURCE_REVISION,
        "tree": SOURCE_TREE,
        "definition": (
            "The executable implementation is the exact clean final F8-66 "
            "repair head. Commits c9effb0 through c05d320 repair Vulkan "
            "Gallery resource binding, smoke and paint publication, sustained "
            "translation performance and bounded rope quality. The exact "
            "c05d320 source passes the complete physical Linux Vulkan suite, "
            "pinned validation, relocated installed SDK and installed-public "
            "capture without changing the sole schedule, work budgets or "
            "quality tolerances. F8-67 adds only immutable freeze metadata and "
            "coordination outside this source identity."
        ),
        "repair_evidence": F8_66_EVIDENCE,
        "repair_evidence_sha256": file_sha256(ROOT / F8_66_EVIDENCE),
        "hash_method": (
            "SHA-256 over sorted repository-relative path + NUL + file bytes "
            "+ NUL"
        ),
        "protected_closures": closures,
    }
    candidate["sole_path_audit"]["audit_evidence"] = (
        "docs/roadmap/evidence/F8-67-SOLE-PATH-AUDIT.json"
    )

    fixture_path = "conformance/v1/realtime/scenario-fixtures-v1.json"
    validator_path = "conformance/v1/realtime_device_suite.py"
    producer_path = "conformance/v1/realtime_device_capture.py"
    profile_path = "conformance/v1/qualification/realtime-profile-v1.json"
    capability_path = (
        "conformance/v1/qualification/capability-obligations-v1.json"
    )
    inventory_path = (
        "conformance/v1/qualification/kernel-device-obligations-v1.json"
    )
    oracle_path = "conformance/v1/realtime/oracle-suite-v1.json"
    manifest_path = "conformance/v1/manifest.json"
    fixtures = revision_json(fixture_path)
    inventory = revision_json(inventory_path)
    kernel_manifest = revision_file_bytes(
        "cmake/SlangKernelManifest.cmake").decode("utf-8")
    modules = cmake_set(kernel_manifest, "PARALLEL_MATER_SLANG_KERNEL_MODULES")
    entry_count = sum(len(cmake_set(
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
        "capture_validator": {
            "path": validator_path,
            "sha256": revision_file_sha256(validator_path),
            "schema": "parallel-mater-realtime-device-suite-capture/v1",
        },
        "capture_producer": {
            "path": producer_path,
            "sha256": revision_file_sha256(producer_path),
            "schema": "parallel-mater-realtime-device-suite-capture/v1",
        },
        "capability_obligations": {
            "path": capability_path,
            "sha256": revision_file_sha256(capability_path),
            "rows": 49,
        },
        "kernel_device_obligations": {
            "path": inventory_path,
            "sha256": revision_file_sha256(inventory_path),
            "runtime_variants": 5,
            "physical_targets": len(inventory["physical_targets"]),
        },
        "oracle_suite": {
            "path": oracle_path,
            "sha256": revision_file_sha256(oracle_path),
            "public_cases": len(inventory["public_case_sets"]["all"]),
        },
        "case_and_golden_manifest": {
            "path": manifest_path,
            "sha256": revision_file_sha256(manifest_path),
        },
        "budgets_and_tolerances": {
            "path": profile_path,
            "sha256": revision_file_sha256(profile_path),
        },
    }

    candidate["native_binaries"] = ci["native_binaries"]
    candidate["compile_evidence"] = {
        "qualification_claim": False,
        "rule": (
            "Compilation, reflection, packaging and hosted execution are build "
            "evidence only. Compile-only mirrors and hosted runners qualify no "
            "physical row. F8-66 exact-source Linux Vulkan evidence is separate."
        ),
        "github_actions": {
            "workflow": run["workflow"],
            "run_id": run["id"],
            "run_url": run["url"],
            "result": run["result"],
            "head_revision": run["head_revision"],
            "head_tree": run["head_tree"],
            "checkout_merge_revision": run["head_revision"],
            "jobs": ci["jobs"],
            "artifacts": ci["artifacts"],
            "event": run["event"],
            "archive_hash_method": (
                "GitHub Actions artifact digest returned by the repository API"
            ),
            "tree_hash_method": (
                "SHA-256 over sorted extracted relative path + NUL + file "
                "bytes + NUL"
            ),
        },
        "source_clean_evidence": ci["source_clean_evidence"],
    }
    candidate["application_artifacts"] = ci["application_artifacts"]
    candidate["toolchains"] = ci["toolchains"]

    prerequisites = {
        item["task"]: item for item in candidate["prerequisite_evidence"]
    }
    prerequisites["F8-65"] = {
        "task": "F8-65",
        "path": F8_65_FREEZE_EVIDENCE,
        "sha256": file_sha256(ROOT / F8_65_FREEZE_EVIDENCE),
    }
    prerequisites["F8-66"] = {
        "task": "F8-66",
        "path": F8_66_EVIDENCE,
        "sha256": file_sha256(ROOT / F8_66_EVIDENCE),
    }
    candidate["prerequisite_evidence"] = [
        prerequisites[key] for key in sorted(prerequisites)
    ]
    candidate["execution_evidence"] = F8_66_EVIDENCE
    for target in candidate["physical_targets"]:
        if target["id"] == "linux-rtx3050ti-slang-cuda":
            target["qualification_owner"] = "F8-68"
    candidate["qualification_routing"].update({
        "accepted_candidate": CANDIDATE_ID,
        "tasks": [
            "F8-25", "F8-26", "F8-27", "F8-28", "F8-29", "F8-30",
            "F8-53", "F8-68",
        ],
        "interim_cpu_evidence": (
            "F8-23 and candidates v4-v6 remain immutable rejected or interim "
            "provenance. Candidate v7 is rejected after its F8-24 Linux CUDA "
            "pass and F8-25 Linux Vulkan failure; neither observation supplies "
            "a candidate-v8 pass. F8-66 proves the complete physical Linux "
            "Vulkan regression and installed-public capture on the exact "
            "frozen source. F8-25 through F8-30 and F8-53 own their replacement "
            "rows, while F8-68 owns replacement Linux CUDA requalification."
        ),
        "rule": (
            "Every final physical row must name this exact candidate id, source "
            "revision, tree, CI run, candidate manifest hash and protected "
            "closure hashes. Historical evidence cannot be unioned."
        ),
    })
    return candidate


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def route_candidate(candidate: dict[str, Any], candidate_path: Path) -> None:
    candidate_sha256 = file_sha256(candidate_path)
    active_path = (
        ROOT / "conformance/v1/qualification/active-realtime-candidate.json"
    )
    write_json(active_path, {
        "schema": "parallel-mater-active-realtime-candidate/v1",
        "candidate_id": CANDIDATE_ID,
        "manifest": candidate_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": candidate_sha256,
    })

    rejected_rows = [
        {
            "target": item["id"],
            "qualification_owner": item["qualification_owner"],
            "status": "invalidated-requalify-replacement",
        }
        for item in candidate["physical_targets"]
    ]
    rejection_path = (
        ROOT / "conformance/v1/qualification/realtime-candidate-v7-rejection.json"
    )
    rejection = {
        "schema": "parallel-mater-realtime-candidate-rejection/v1",
        "candidate_id": PREVIOUS_ID,
        "candidate_manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
        "candidate_manifest_sha256": PREVIOUS_SHA256,
        "status": "rejected-required-linux-vulkan-suite",
        "rejected_at": "2026-10-31",
        "rejection_task": "F8-66",
        "historical_qualification_task": "F8-24",
        "historical_pass_evidence": V7_CUDA_EVIDENCE,
        "historical_pass_evidence_sha256": V7_CUDA_EVIDENCE_SHA256,
        "historical_pass_archive": V7_CUDA_ARCHIVE,
        "historical_pass_archive_sha256": V7_CUDA_ARCHIVE_SHA256,
        "failure_evidence": V7_FAILURE_EVIDENCE,
        "failure_evidence_sha256": V7_FAILURE_SHA256,
        "failure_archive": V7_FAILURE_ARCHIVE,
        "failure_archive_sha256": V7_FAILURE_ARCHIVE_SHA256,
        "repair_evidence": F8_66_EVIDENCE,
        "repair_evidence_sha256": file_sha256(ROOT / F8_66_EVIDENCE),
        "reasons": [
            "Candidate v7 passed its F8-24 Linux CUDA row but failed the "
            "mandatory F8-25 physical Linux Vulkan Gallery and installed-public "
            "suite.",
            "F8-66 repairs those failures and proves 157 registered tests, "
            "pinned validation, 27 SPIR-V modules, SDK relocation, 49 "
            "capability rows, 30 public cases, 12 scenarios, 37 named "
            "measurements, five negative controls and five sustained workloads "
            "on exact clean replacement source c05d320.",
            "Candidate-v7 CUDA passes and Vulkan partial observations remain "
            "immutable rejected provenance and cannot qualify or be unioned "
            "with the F8-66 repaired-source evidence.",
        ],
        "superseded_at": "2026-10-31",
        "supersession_task": "F8-67",
        "replacement": {
            "status": "frozen-awaiting-physical-qualification",
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest": candidate_path.relative_to(ROOT).as_posix(),
            "candidate_manifest_sha256": candidate_sha256,
            "freeze_task": "F8-67",
            "source_revision": SOURCE_REVISION,
            "source_tree": SOURCE_TREE,
            "ci_run_id": PROTECTED_CI_RUN_ID,
        },
        "evidence_union_allowed": False,
        "invalidated_physical_rows": rejected_rows,
        "qualification_rule": (
            "No final F8 qualification report may name or reuse candidate v7. "
            "F8-25 through F8-30, F8-53 and F8-68 must execute only "
            f"{CANDIDATE_ID} from F8-67."
        ),
    }
    write_json(rejection_path, rejection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-evidence",
        type=Path,
        required=True,
        help="independently hashed F8-67 protected CI evidence JSON",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--route",
        action="store_true",
        help="also route the active pointer and v7 rejection record",
    )
    args = parser.parse_args()
    candidate = generate(load_json(args.ci_evidence.resolve()))
    write_json(args.output, candidate)
    if args.route:
        route_candidate(candidate, args.output.resolve())
    print(args.output)
    print(file_sha256(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
