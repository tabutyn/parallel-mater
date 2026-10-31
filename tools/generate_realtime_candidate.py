#!/usr/bin/env python3
"""Generate the F8-65 realtime candidate from immutable source evidence."""

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
    ROOT / "conformance/v1/qualification/realtime-candidate-v6.json"
)
OUTPUT_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v7.json"
)
SOURCE_REVISION = "05f8b04a14fc34cfb1ba322f363ea7165fcf3d48"
SOURCE_TREE = "e99f19d6af7ddca416b68b0a9a59755a738345f0"
CANDIDATE_ID = "realtime-single-step-v1-05f8b04"
PREVIOUS_ID = "realtime-single-step-v1-fc2ffba"
PREVIOUS_SHA256 = (
    "6d418e610601337ed4aa25a3c09d9791bcc4ccbb3fa281a69f45e4c7775fe2c9"
)
LEASE_TOKEN = "89aa50b5-7aa2-4043-ae5b-1eee4ea7cc8d"
F8_64_EVIDENCE = (
    "docs/roadmap/evidence/F8-64-INSTALLED-PUBLIC-CUDA-REPAIR.json"
)
F8_65_PHYSICAL_EVIDENCE = (
    "docs/roadmap/evidence/F8-65-EXACT-SOURCE-LINUX-CUDA.json"
)
V6_FAILURE_EVIDENCE = (
    "docs/roadmap/evidence/F8-24-CANDIDATE-V6-LINUX-CUDA-FAILURE.json"
)
V6_FAILURE_SHA256 = (
    "1be323331ac05533be95ec448c5be121604d734c7b3f43185c6c7f58f5465d07"
)
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
            "candidate v6 changed; refusing to derive v7")
    require(file_sha256(ROOT / V6_FAILURE_EVIDENCE) == V6_FAILURE_SHA256,
            "candidate-v6 failure evidence changed; refusing to derive v7")
    actual_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", SOURCE_REVISION],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(actual_tree == SOURCE_TREE, "frozen source tree identity drift")

    run = ci.get("run", {})
    require(run.get("id") == 41213026819, "unexpected protected CI run")
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
        "task": "F8-65",
        "frozen_at": "2026-10-31",
        "lease_token": LEASE_TOKEN,
        "supersedes": {
            "candidate_id": PREVIOUS_ID,
            "manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": PREVIOUS_SHA256,
            "supersession_record": (
                "conformance/v1/qualification/"
                "realtime-candidate-v6-rejection.json"
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
            "The executable implementation is the exact clean final F8-64 "
            "head. Commit 861e8f1 repairs the installed-public CUDA runtime, "
            "cases, capture producer and candidate-consistent fixtures; "
            "commit 05f8b04 repairs only the D3D12 binding generator needed "
            "for green protected CI. The exact 05f8b04 source passes the "
            "complete physical Linux CUDA suite and installed-public capture "
            "without changing the sole schedule, budgets or tolerances. "
            "F8-65 adds only immutable freeze metadata and checks outside this "
            "source identity."
        ),
        "repair_evidence": F8_64_EVIDENCE,
        "repair_evidence_sha256": file_sha256(ROOT / F8_64_EVIDENCE),
        "hash_method": (
            "SHA-256 over sorted repository-relative path + NUL + file bytes "
            "+ NUL"
        ),
        "protected_closures": closures,
    }
    candidate["sole_path_audit"]["audit_evidence"] = (
        "docs/roadmap/evidence/F8-65-SOLE-PATH-AUDIT.json"
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
            "physical row. F8-65 exact-source Linux CUDA evidence is separate."
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
    prerequisites["F8-64"] = {
        "task": "F8-64",
        "path": F8_64_EVIDENCE,
        "sha256": file_sha256(ROOT / F8_64_EVIDENCE),
    }
    candidate["prerequisite_evidence"] = [
        prerequisites[key] for key in sorted(prerequisites)
    ]
    candidate["execution_evidence"] = F8_65_PHYSICAL_EVIDENCE
    candidate["qualification_routing"].update({
        "accepted_candidate": CANDIDATE_ID,
        "interim_cpu_evidence": (
            "F8-23 and candidates v4-v6 remain immutable rejected or interim "
            "provenance and supply no candidate-v7 pass. F8-65 proves the full "
            "physical Linux CUDA regression and installed-public capture on "
            "the exact frozen source; F8-24 through F8-30 and F8-53 still own "
            "final qualification rows."
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
        ROOT / "conformance/v1/qualification/realtime-candidate-v6-rejection.json"
    )
    rejection = load_json(rejection_path)
    require(rejection.get("candidate_id") == PREVIOUS_ID,
            "candidate-v6 rejection identity drift")
    require(rejection.get("candidate_manifest_sha256") == PREVIOUS_SHA256,
            "candidate-v6 rejection manifest drift")
    require(rejection.get("failure_evidence") == V6_FAILURE_EVIDENCE and
            rejection.get("failure_evidence_sha256") == V6_FAILURE_SHA256,
            "candidate-v6 rejection failure evidence drift")
    rejection.update({
        "superseded_at": "2026-10-31",
        "supersession_task": "F8-65",
        "replacement": {
            "status": "frozen-awaiting-physical-qualification",
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest": candidate_path.relative_to(ROOT).as_posix(),
            "candidate_manifest_sha256": candidate_sha256,
            "freeze_task": "F8-65",
            "source_revision": SOURCE_REVISION,
            "source_tree": SOURCE_TREE,
            "ci_run_id": 41213026819,
        },
        "invalidated_physical_rows": rejected_rows,
        "qualification_rule": (
            "No final F8 qualification report may name or reuse candidate v6. "
            "F8-24 through F8-30 and F8-53 must execute only "
            f"{CANDIDATE_ID} from F8-65."
        ),
    })
    write_json(rejection_path, rejection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-evidence",
        type=Path,
        required=True,
        help="independently hashed F8-65 protected CI evidence JSON",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--route",
        action="store_true",
        help="also route the active pointer and v6 rejection record",
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
