#!/usr/bin/env python3
"""Validate the single active realtime qualification candidate."""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v7.json"
)
ACTIVE_POINTER_PATH = (
    ROOT / "conformance/v1/qualification/active-realtime-candidate.json"
)
V1_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v1-rejection.json"
)
V2_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v2-rejection.json"
)
V3_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v3-rejection.json"
)
V4_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v4-rejection.json"
)
V5_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v5-rejection.json"
)
V6_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v6-rejection.json"
)
V1_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v1.json"
)
V2_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v2.json"
)
V3_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v3.json"
)
V4_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v4.json"
)
V5_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v5.json"
)
V6_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v6.json"
)
V5_FAILURE_PATH = (
    ROOT / "docs/roadmap/evidence/"
    "F8-24-CANDIDATE-V5-LINUX-CUDA-FAILURE.json"
)
V6_FAILURE_PATH = (
    ROOT / "docs/roadmap/evidence/"
    "F8-24-CANDIDATE-V6-LINUX-CUDA-FAILURE.json"
)
F8_64_REPAIR_PATH = (
    ROOT / "docs/roadmap/evidence/"
    "F8-64-INSTALLED-PUBLIC-CUDA-REPAIR.json"
)
F8_65_PHYSICAL_PATH = (
    ROOT / "docs/roadmap/evidence/"
    "F8-65-EXACT-SOURCE-LINUX-CUDA.json"
)
INVENTORY_PATH = (
    ROOT / "conformance/v1/qualification/kernel-device-obligations-v1.json"
)
CAPABILITY_PATH = (
    ROOT / "conformance/v1/qualification/capability-obligations-v1.json"
)
PROFILE_PATH = ROOT / "conformance/v1/qualification/realtime-profile-v1.json"
ORACLE_PATH = ROOT / "conformance/v1/realtime/oracle-suite-v1.json"
MANIFEST_PATH = ROOT / "cmake/SlangKernelManifest.cmake"
TASK_DIR = ROOT / "docs/roadmap/tasks"

ACTIVE_ID = "realtime-single-step-v1-05f8b04"
ACTIVE_SOURCE_REVISION = "05f8b04a14fc34cfb1ba322f363ea7165fcf3d48"
ACTIVE_SOURCE_TREE = "e99f19d6af7ddca416b68b0a9a59755a738345f0"
ACTIVE_CI_RUN_ID = 41213026819
V1_REJECTED_ID = "realtime-single-step-v1-dea47c1"
V2_REJECTED_ID = "realtime-single-step-v1-fd78fa8"
V3_REJECTED_ID = "realtime-single-step-v1-3671f18"
V4_REJECTED_ID = "realtime-single-step-v1-5a811c4"
V5_REJECTED_ID = "realtime-single-step-v1-cfb104b"
V6_REJECTED_ID = "realtime-single-step-v1-fc2ffba"
V1_MANIFEST_SHA256 = (
    "c25c010be3a0e908537e3c6542c1759b28b10cfb4a4169b51bb2834233411d43"
)
V2_MANIFEST_SHA256 = (
    "f4f1c7807dd50b07857c568b9df3e8e2ccd6fd3870907238dc9dcfab45b493af"
)
V3_MANIFEST_SHA256 = (
    "d6d84f85a87af71fcdb0ee5c335f6c3979a741a151b3fbb4151096dd588d4a04"
)
V4_MANIFEST_SHA256 = (
    "f0f15a65fed178625b263c6462e7e0d9397fd4880cfc2a3a73582b98584876ae"
)
V5_MANIFEST_SHA256 = (
    "3435be0a4d4caa2d014d82faba7272aaadeba546a555a2e35a77ff18d0d8a000"
)
V6_MANIFEST_SHA256 = (
    "6d418e610601337ed4aa25a3c09d9791bcc4ccbb3fa281a69f45e4c7775fe2c9"
)
V5_FAILURE_SHA256 = (
    "529f43bf26c36cd989c8bd72c936ecdd1306b372d70d194071b71f5c0e9453dd"
)
V6_FAILURE_SHA256 = (
    "1be323331ac05533be95ec448c5be121604d734c7b3f43185c6c7f58f5465d07"
)
F8_64_REPAIR_SHA256 = (
    "27411fa368aa3f08f003b7593ed9be89c5094d4e1cdf68df577038e5ece926cd"
)
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
QUALIFICATION_TASKS = {f"F8-{number}" for number in range(24, 31)} | {"F8-53"}
PREREQUISITE_EVIDENCE = {
    "F8-50": "docs/roadmap/evidence/F8-50-single-execution-path.json",
    "F8-51": "docs/roadmap/evidence/F8-51-gallery-fixed-tick.json",
    "F8-54": "docs/roadmap/evidence/F8-54-metal-rigid-dispatch-batching.json",
    "F8-55": "docs/roadmap/evidence/F8-55-shared-bvh-coupling-batching.json",
    "F8-56": "docs/roadmap/evidence/F8-56-apple-gallery-dispatch/summary.json",
    "F8-60": "docs/roadmap/evidence/F8-60-capability-case-reconciliation.json",
    "F8-62": "docs/roadmap/evidence/F8-62-LINUX-CUDA-GALLERY-REPAIR.json",
    "F8-64": "docs/roadmap/evidence/F8-64-INSTALLED-PUBLIC-CUDA-REPAIR.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmake_set(text: str, name: str) -> list[str]:
    match = re.search(rf"set\({re.escape(name)}\s+(.*?)\)", text, re.DOTALL)
    if not match:
        raise ValueError(f"missing CMake set {name}")
    return re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(1))


def canonical_kernels(root: Path) -> dict[str, list[str]]:
    text = (root / MANIFEST_PATH.relative_to(ROOT)).read_text(encoding="utf-8")
    modules = cmake_set(text, "PARALLEL_MATER_SLANG_KERNEL_MODULES")
    return {
        module: cmake_set(text, f"PARALLEL_MATER_SLANG_{module}_ENTRIES")
        for module in modules
    }


def closure_identity(root: Path, patterns: list[str]) -> tuple[int, str]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


@functools.lru_cache(maxsize=64)
def revision_file_sha256(root_text: str, revision: str, relative: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=Path(root_text),
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def safe_revision_file_sha256(
    root_text: str, revision: str, relative: str
) -> str | None:
    if not relative:
        return None
    try:
        return revision_file_sha256(root_text, revision, relative)
    except (OSError, subprocess.CalledProcessError):
        return None


@functools.lru_cache(maxsize=4)
def revision_closure_identities(
    root_text: str,
    revision: str,
    requested: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, tuple[int, str]]:
    """Hash closures from the candidate's immutable Git tree, not HEAD."""
    root = Path(root_text)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="parallel-mater-candidate-") as raw:
        extracted = Path(raw)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(extracted)
        return {
            category: closure_identity(extracted, list(patterns))
            for category, patterns in requested
        }


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_candidate(
    candidate: Mapping[str, Any] | None = None,
    *,
    root: Path = ROOT,
) -> list[str]:
    """Return every freeze violation; an empty list means the candidate is valid."""
    candidate_path = root / CANDIDATE_PATH.relative_to(ROOT)
    v1_rejection_path = root / V1_REJECTION_PATH.relative_to(ROOT)
    v2_rejection_path = root / V2_REJECTION_PATH.relative_to(ROOT)
    v3_rejection_path = root / V3_REJECTION_PATH.relative_to(ROOT)
    v4_rejection_path = root / V4_REJECTION_PATH.relative_to(ROOT)
    v5_rejection_path = root / V5_REJECTION_PATH.relative_to(ROOT)
    v6_rejection_path = root / V6_REJECTION_PATH.relative_to(ROOT)
    v1_candidate_path = root / V1_CANDIDATE_PATH.relative_to(ROOT)
    v2_candidate_path = root / V2_CANDIDATE_PATH.relative_to(ROOT)
    v3_candidate_path = root / V3_CANDIDATE_PATH.relative_to(ROOT)
    v4_candidate_path = root / V4_CANDIDATE_PATH.relative_to(ROOT)
    v5_candidate_path = root / V5_CANDIDATE_PATH.relative_to(ROOT)
    v6_candidate_path = root / V6_CANDIDATE_PATH.relative_to(ROOT)
    v5_failure_path = root / V5_FAILURE_PATH.relative_to(ROOT)
    v6_failure_path = root / V6_FAILURE_PATH.relative_to(ROOT)
    f8_64_repair_path = root / F8_64_REPAIR_PATH.relative_to(ROOT)
    f8_65_physical_path = root / F8_65_PHYSICAL_PATH.relative_to(ROOT)
    inventory = load_json(root / INVENTORY_PATH.relative_to(ROOT))
    profile = load_json(root / PROFILE_PATH.relative_to(ROOT))
    oracle = load_json(root / ORACLE_PATH.relative_to(ROOT))
    v1_rejection = load_json(v1_rejection_path)
    v2_rejection = load_json(v2_rejection_path)
    v3_rejection = (load_json(v3_rejection_path)
                    if v3_rejection_path.is_file() else {})
    v4_rejection = (load_json(v4_rejection_path)
                    if v4_rejection_path.is_file() else {})
    v5_rejection = (load_json(v5_rejection_path)
                    if v5_rejection_path.is_file() else {})
    v6_rejection = (load_json(v6_rejection_path)
                    if v6_rejection_path.is_file() else {})
    frozen_candidate = load_json(candidate_path)
    if candidate is None:
        candidate = frozen_candidate

    v6_rejected = (
        v6_rejection.get("candidate_id") == V6_REJECTED_ID and
        v6_rejection.get("candidate_manifest_sha256") == V6_MANIFEST_SHA256 and
        v6_rejection.get("status") ==
        "rejected-required-installed-public-cuda-suite"
    )

    errors: list[str] = []
    active_pointer = load_json(root / ACTIVE_POINTER_PATH.relative_to(ROOT))
    _expect(errors,
            active_pointer.get("schema") ==
            "parallel-mater-active-realtime-candidate/v1",
            "active candidate pointer schema drift")
    _expect(errors, active_pointer.get("candidate_id") == ACTIVE_ID,
            "active candidate pointer names the wrong candidate")
    _expect(errors,
            active_pointer.get("manifest") ==
            CANDIDATE_PATH.relative_to(ROOT).as_posix(),
            "active candidate pointer names the wrong manifest")
    _expect(errors,
            active_pointer.get("manifest_sha256") == file_sha256(candidate_path),
            "active candidate pointer manifest hash drift")
    _expect(errors, candidate.get("schema") == "parallel-mater-realtime-candidate/v3",
            "candidate schema is not v3")
    _expect(errors, candidate.get("id") == ACTIVE_ID,
            "active candidate id changed")
    _expect(errors, candidate.get("task") == "F8-65",
            "candidate freeze task is not F8-65")
    _expect(errors, candidate.get("status") == "frozen-awaiting-physical-qualification",
            "candidate status is not frozen")
    source = candidate.get("source", {})
    for key in ("revision", "tree"):
        _expect(errors, bool(re.fullmatch(r"[0-9a-f]{40}", str(source.get(key, "")))),
                f"source {key} is not an exact Git identity")
    _expect(errors, source.get("revision") == ACTIVE_SOURCE_REVISION,
            "active source revision drift")
    _expect(errors, source.get("tree") == ACTIVE_SOURCE_TREE,
            "active source tree drift")
    _expect(errors,
            source.get("repair_evidence") ==
            "docs/roadmap/evidence/F8-64-INSTALLED-PUBLIC-CUDA-REPAIR.json",
            "candidate does not bind the F8-64 repair evidence")
    repair_path = root / str(source.get("repair_evidence", ""))
    _expect(errors, repair_path.is_file() and
            source.get("repair_evidence_sha256") == file_sha256(repair_path),
            "F8-64 repair evidence hash drift")

    closures = {
        item.get("category"): item
        for item in source.get("protected_closures", [])
    }
    _expect(errors, set(closures) == REQUIRED_CLOSURES,
            "protected closure categories are incomplete")
    _expect(errors, len(closures) == len(source.get("protected_closures", [])),
            "protected closure categories are duplicated")
    requested = tuple(sorted(
        (category, tuple(closures[category].get("patterns", [])))
        for category in REQUIRED_CLOSURES & set(closures)
    ))
    frozen_identities: dict[str, tuple[int, str]] = {}
    try:
        frozen_identities = revision_closure_identities(
            str(root.resolve()), ACTIVE_SOURCE_REVISION, requested)
    except (OSError, subprocess.CalledProcessError, tarfile.TarError):
        errors.append("active source tree is unavailable for closure validation")
    for category in sorted(REQUIRED_CLOSURES & set(closures)):
        closure = closures[category]
        patterns = closure.get("patterns", [])
        _expect(errors, bool(patterns), f"{category} closure has no patterns")
        count, digest = frozen_identities.get(category, (0, ""))
        _expect(errors, count == closure.get("file_count"),
                f"{category} closure file count drift")
        _expect(errors, digest == closure.get("sha256"),
                f"{category} closure hash drift")

    frozen_profile = candidate.get("profile", {})
    _expect(errors, frozen_profile.get("id") == profile.get("id"),
            "profile id drift")
    _expect(errors, frozen_profile.get("semantic_name") ==
            profile.get("profile", {}).get("semantic_name"),
            "profile semantic name drift")
    _expect(errors, frozen_profile.get("version") ==
            profile.get("profile", {}).get("version"),
            "profile version drift")
    _expect(errors, frozen_profile.get("timestep") == "1/60",
            "profile timestep drift")

    surface = candidate.get("capability_surface", {})
    _expect(errors, surface.get("api_version") == "0.3",
            "candidate API version is not 0.3")
    _expect(errors, surface.get("public_domains") == [
        "RIGID_BODY", "FLUID", "CLOTH", "SOFT_BODY", "ROPE", "SMOKE"
    ], "candidate public-domain surface drift")
    _expect(errors, set(surface.get("advertised_modes", [])) ==
            set(inventory.get("advertised_modes", [])),
            "candidate advertised-mode surface drift")
    _expect(errors, surface.get("public_case_count") ==
            len(inventory.get("public_case_sets", {}).get("all", [])),
            "candidate public-case surface drift")
    _expect(errors, surface.get("physical_target_count") ==
            len(inventory.get("physical_targets", [])),
            "candidate physical-target surface drift")
    _expect(errors, surface.get("interaction_capability_count") == 49,
            "candidate interaction-capability surface drift")
    _expect(errors, load_json(root / CAPABILITY_PATH.relative_to(ROOT)).get(
        "api_version") == "0.3", "capability obligations are not API 0.3")

    inputs = candidate.get("contract_inputs", {})
    fixture = inputs.get("scenario_fixtures", {})
    fixture_path = root / str(fixture.get("path", ""))
    _expect(errors, fixture_path.is_file(), "scenario fixture contract is missing")
    expected_fixture_sha256 = safe_revision_file_sha256(
        str(root.resolve()), ACTIVE_SOURCE_REVISION, str(fixture.get("path", "")))
    _expect(errors, fixture.get("sha256") == expected_fixture_sha256,
            "scenario fixture contract hash drift")
    _expect(errors, fixture.get("scenario_count") == 12 and
            fixture.get("measurement_count") == 37,
            "scenario fixture surface drift")
    capture_validator = inputs.get("capture_validator", {})
    capture_validator_path = root / str(capture_validator.get("path", ""))
    _expect(errors, capture_validator_path.is_file(),
            "capture validator contract is missing")
    expected_capture_sha256 = safe_revision_file_sha256(
        str(root.resolve()), ACTIVE_SOURCE_REVISION,
        str(capture_validator.get("path", "")))
    _expect(errors, capture_validator.get("sha256") == expected_capture_sha256,
            "capture validator contract hash drift")
    _expect(errors,
            capture_validator.get("schema") ==
            "parallel-mater-realtime-device-suite-capture/v1",
            "capture validator schema drift")
    for key, message in (
        ("capture_producer", "capture producer"),
        ("capability_obligations", "capability obligations"),
        ("kernel_device_obligations", "kernel/device obligations"),
        ("oracle_suite", "oracle suite"),
        ("case_and_golden_manifest", "case/golden manifest"),
        ("budgets_and_tolerances", "budget and tolerance contract"),
    ):
        record = inputs.get(key, {})
        relative = str(record.get("path", ""))
        path = root / relative
        _expect(errors, path.is_file(), f"{message} is missing")
        expected = safe_revision_file_sha256(
            str(root.resolve()), ACTIVE_SOURCE_REVISION, relative)
        _expect(errors, record.get("sha256") == expected,
                f"{message} hash drift")
    producer = inputs.get("capture_producer", {})
    _expect(errors,
            producer.get("schema") ==
            "parallel-mater-realtime-device-suite-capture/v1",
            "capture producer schema drift")
    _expect(errors,
            inputs.get("capability_obligations", {}).get("rows") == 49 and
            inputs.get("oracle_suite", {}).get("public_cases") == 30,
            "candidate contract matrix drift")
    _expect(errors,
            inputs.get("budgets_and_tolerances", {}).get("path") ==
            PROFILE_PATH.relative_to(ROOT).as_posix(),
            "candidate budget and tolerance contract drift")

    audit = candidate.get("sole_path_audit", {})
    _expect(errors, audit.get("public_creation") == "ordinary-api-0.3",
            "ordinary public creation is not the frozen sole path")
    _expect(errors, audit.get("production_schedule_count") == 1,
            "candidate does not freeze exactly one production schedule")
    _expect(errors, audit.get("gallery_session") == "shared",
            "candidate does not freeze the shared Gallery session")
    _expect(errors, audit.get("legacy_selection_found") is False,
            "candidate reports a selectable legacy path")
    audit_path = root / str(audit.get("audit_evidence", ""))
    _expect(errors, audit_path.is_file(), "sole-path audit evidence is missing")
    audit_evidence = load_json(audit_path) if audit_path.is_file() else {}
    _expect(errors, audit_evidence.get("status") == "passed" and
            audit_evidence.get("audited_source_revision") ==
            ACTIVE_SOURCE_REVISION,
            "sole-path audit did not pass for the frozen source")
    negative_findings = audit_evidence.get("negative_findings", {})
    _expect(errors, negative_findings and
            not any(negative_findings.values()),
            "sole-path audit retains a negative finding")
    _expect(errors, not (root / "src/simulation/reference_schedule.hpp").exists(),
            "retired legacy production schedule exists")
    runtime_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "src/simulation").glob("runtime_*.cu.inc"))
    )
    for token in ("PhysicsStepProfile", "simulation::schedule_frame"):
        _expect(errors, token not in runtime_text,
                f"selectable production schedule token remains: {token}")
    public_contract_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in ("include/parallel_mater/parallel_mater.h",
                         "registry/pm.xml")
    )
    for token in ("PM_EXT_simulation_profile", "PmSimulationExecutionProfile"):
        _expect(errors, token not in public_contract_text,
                f"retired public execution selector remains: {token}")

    canonical = canonical_kernels(root)
    covered: defaultdict[str, list[str]] = defaultdict(list)
    realtime_cases = {
        item["id"] for item in oracle.get("public_trajectory_cases", [])
    }
    for group in inventory.get("kernel_groups", []):
        covered[group["module"]].extend(group["entries"])
        _expect(errors, bool(group.get("existing_execution_tests")),
                f"kernel group {group['id']} lacks execution tests")
        _expect(errors, bool(group.get("public_case_sets") or
                             group.get("f8_public_cases")),
                f"kernel group {group['id']} lacks public cases")
        _expect(errors, set(group.get("f8_public_cases", [])) <= realtime_cases,
                f"kernel group {group['id']} names an unknown realtime case")
    _expect(errors, set(covered) == set(canonical),
            "canonical module coverage drift")
    for module, entries in canonical.items():
        _expect(errors, len(covered[module]) == len(set(covered[module])),
                f"canonical entry duplicated in {module}")
        _expect(errors, set(covered[module]) == set(entries),
                f"canonical entry coverage drift in {module}")
    coverage = candidate.get("kernel_coverage", {})
    expected_module_count = len(canonical)
    expected_entry_count = sum(len(entries) for entries in canonical.values())
    expected_group_count = len(inventory.get("kernel_groups", []))
    _expect(errors, coverage.get("canonical_module_count") ==
            expected_module_count, "canonical module count drift")
    _expect(errors, coverage.get("canonical_entry_count") ==
            expected_entry_count, "canonical entry count drift")
    _expect(errors, coverage.get("kernel_group_count") == expected_group_count,
            "kernel group count drift")
    _expect(errors, coverage.get("coverage_gaps") == [],
            "candidate has kernel coverage gaps")

    inventory_variants = {
        item["id"]: item for item in inventory.get("compiled_variants", [])
    }
    runtime_variants = {
        item["id"]: item for item in candidate.get("runtime_variants", [])
    }
    expected_runtime = {
        key for key, item in inventory_variants.items()
        if "compile-only" not in item["kind"]
    }
    _expect(errors, set(runtime_variants) == expected_runtime,
            "runtime variant coverage drift")
    public_execution = set(oracle.get("public_api", {}).get("execution_cases", []))
    for variant_id in expected_runtime & set(runtime_variants):
        actual = runtime_variants[variant_id]
        expected = inventory_variants[variant_id]
        _expect(errors, actual.get("f8_execution_test") ==
                expected.get("f8_execution_test"),
                f"{variant_id} execution test drift")
        _expect(errors, actual.get("f8_execution_test") in public_execution,
                f"{variant_id} execution test is not public")
        _expect(errors, set(actual.get("applicable_targets", [])) ==
                set(expected.get("applicable_targets", [])),
                f"{variant_id} target coverage drift")
        _expect(errors, set(actual.get("qualification_owners", [])) ==
                set(expected.get("qualification_owners", [])),
                f"{variant_id} qualification owner drift")

    compile_only = {
        item["id"]: item for item in candidate.get("compile_only_variants", [])
    }
    _expect(errors, set(compile_only) == set(inventory_variants) - expected_runtime,
            "compile-only variant coverage drift")
    for variant in compile_only.values():
        _expect(errors, variant.get("qualification_allowed") is False,
                f"compile-only variant {variant['id']} permits qualification")

    inventory_targets = {
        item["id"]: item for item in inventory.get("physical_targets", [])
    }
    frozen_targets = {
        item["id"]: item for item in candidate.get("physical_targets", [])
    }
    _expect(errors, set(frozen_targets) == set(inventory_targets),
            "physical target coverage drift")
    for target_id in set(frozen_targets) & set(inventory_targets):
        actual = frozen_targets[target_id]
        expected = inventory_targets[target_id]
        _expect(errors, actual.get("variant") == expected.get("compiled_variant"),
                f"{target_id} runtime variant drift")
        _expect(errors, actual.get("qualification_owner") ==
                expected.get("qualification_owner"),
                f"{target_id} qualification owner drift")
        _expect(errors, actual.get("status") == "pending",
                f"{target_id} is not pending replacement qualification")

    binaries = candidate.get("native_binaries", [])
    binary_variants = {item.get("runtime_variant") for item in binaries}
    _expect(errors, {"cpu-runtime", "cuda-runtime", "vulkan-runtime",
                     "d3d12-runtime", "metal-runtime"}
            <= binary_variants, "native binary hashes lack runtime coverage")
    for binary in binaries:
        _expect(errors, bool(re.fullmatch(r"[0-9a-f]{64}",
                                          str(binary.get("sha256", "")))),
                f"native binary {binary.get('id')} lacks an exact SHA-256")

    evidence = candidate.get("compile_evidence", {})
    _expect(errors, evidence.get("qualification_claim") is False,
            "compile evidence claims physical qualification")
    actions = evidence.get("github_actions", {})
    _expect(errors, actions.get("result") == "success",
            "cross-platform GitHub Actions evidence did not pass")
    _expect(errors, actions.get("head_revision") == ACTIVE_SOURCE_REVISION,
            "GitHub Actions did not run from the frozen source revision")
    _expect(errors, actions.get("head_tree") == ACTIVE_SOURCE_TREE,
            "GitHub Actions did not run from the frozen source tree")
    _expect(errors, actions.get("event") == "workflow_dispatch",
            "GitHub Actions source run was not the immutable manual run")
    _expect(errors, actions.get("run_id") == ACTIVE_CI_RUN_ID,
            "GitHub Actions run identity drift")
    jobs = {item.get("name"): item for item in actions.get("jobs", [])}
    _expect(errors, set(jobs) == REQUIRED_CI_JOBS,
            "protected CI job set drift")
    for name, job in jobs.items():
        _expect(errors, job.get("conclusion") == "success",
                f"protected CI job did not pass: {name}")
    artifact_names = {
        item.get("name") for item in actions.get("artifacts", [])
    }
    _expect(errors, artifact_names == REQUIRED_CI_ARTIFACTS,
            "cross-platform CI artifact set drift")
    for artifact in actions.get("artifacts", []):
        _expect(errors, isinstance(artifact.get("id"), int) and
                artifact.get("id", 0) > 0 and
                artifact.get("file_count", 0) > 0,
                f"artifact {artifact.get('name')} lacks exact identity")
        for key in ("archive_sha256", "extracted_tree_sha256"):
            _expect(errors, bool(re.fullmatch(r"[0-9a-f]{64}",
                                              str(artifact.get(key, "")))),
                    f"artifact {artifact.get('name')} lacks {key}")

    source_clean = evidence.get("source_clean_evidence", {})
    for platform in ("linux", "macos", "windows"):
        report = source_clean.get(f"{platform}_sdk_report", {})
        _expect(errors,
                report.get("source_revision") == ACTIVE_SOURCE_REVISION and
                report.get("source_dirty") is False,
                f"{platform} SDK evidence did not use a clean frozen source")

    toolchain_scopes = {
        item.get("scope") for item in candidate.get("toolchains", [])
    }
    _expect(errors, {
        "portable-sdk-linux", "portable-sdk-macos", "portable-sdk-windows",
        "shared-physics-shaders", "android-gallery", "ipados-gallery",
    } <= toolchain_scopes, "candidate toolchain coverage is incomplete")

    applications = candidate.get("application_artifacts", [])
    application_platforms = {item.get("platform") for item in applications}
    _expect(errors, {"macos", "ipados", "android"} <= application_platforms,
            "candidate application artifacts lack a Gallery platform")
    for artifact in applications:
        _expect(errors, bool(re.fullmatch(r"[0-9a-f]{64}",
                                          str(artifact.get("sha256", "")))),
                f"application artifact {artifact.get('id')} lacks SHA-256")
        _expect(errors, artifact.get("qualification_claim") is False,
                f"application artifact {artifact.get('id')} claims qualification")
    ipad_gallery = next(
        (item for item in applications if item.get("id") == "ci-ipados-gallery"),
        {},
    )
    embedded = ipad_gallery.get("embedded_build_info", {})
    _expect(errors, embedded.get("revision") == ACTIVE_SOURCE_REVISION,
            "iPad compile artifact embeds the wrong source revision")
    _expect(errors, embedded.get("dirty") is False,
            "iPad compile artifact did not use a clean frozen source")
    mobile_workflow = (
        root / ".github/workflows/mobile-gallery.yml"
    ).read_text(encoding="utf-8")
    _expect(errors, '$RUNNER_TEMP/xcode-version.txt' in mobile_workflow and
            "tee xcode-version.txt" not in mobile_workflow,
            "iPad CI still dirties the source checkout before build identity")

    prerequisite_records = {
        item.get("task"): item
        for item in candidate.get("prerequisite_evidence", [])
    }
    _expect(errors, set(prerequisite_records) == set(PREREQUISITE_EVIDENCE),
            "prerequisite evidence set drift")
    for task_id, relative in PREREQUISITE_EVIDENCE.items():
        path = root / relative
        record = prerequisite_records.get(task_id, {})
        _expect(errors, record.get("path") == relative,
                f"{task_id} evidence path drift")
        _expect(errors, path.is_file(), f"{task_id} evidence is missing")
        if path.is_file():
            _expect(errors, record.get("sha256") == file_sha256(path),
                    f"{task_id} evidence hash drift")

    execution_path = root / str(candidate.get("execution_evidence", ""))
    _expect(errors, execution_path == f8_65_physical_path,
            "candidate names the wrong F8-65 execution evidence")
    _expect(errors, execution_path.is_file(),
            "F8-65 exact-source physical evidence is missing")
    execution = load_json(execution_path) if execution_path.is_file() else {}
    repair_source = execution.get("source", {})
    _expect(errors,
            repair_source.get("revision") == ACTIVE_SOURCE_REVISION and
            repair_source.get("tree") == ACTIVE_SOURCE_TREE and
            repair_source.get("working_tree") == "clean",
            "F8-65 did not execute the exact clean frozen source")
    physical = execution.get("results", {}).get("registered_tests", {})
    _expect(errors,
            physical.get("status") == "passed" and
            physical.get("passed") == 131 and
            physical.get("failed") == 0 and
            physical.get("skipped") == 0 and
            physical.get("total") == 131 and
            physical.get("cuda_labeled_tests_executed") == 42,
            "F8-65 complete physical Linux CUDA regression did not pass")
    relocated = execution.get("results", {}).get("installed_sdk_relocation", {})
    _expect(errors,
            relocated.get("status") == "passed" and
            relocated.get("source_revision") == ACTIVE_SOURCE_REVISION and
            relocated.get("source_dirty") is False and
            relocated.get("checks_passed") == 12,
            "F8-65 relocated SDK evidence is incomplete")
    capture = execution.get("results", {}).get("installed_public_capture", {})
    _expect(errors,
            capture.get("status") == "passed" and
            capture.get("failure_count") == 0 and
            capture.get("capability_rows") == 49 and
            capture.get("public_cases_completed") == 30 and
            capture.get("scenarios_passed") == 12 and
            capture.get("unique_named_measurements") == 37 and
            capture.get("negative_controls_passed") == 5 and
            capture.get("workloads_passed") == 5 and
            capture.get("total_measured_workload_ticks") == 72000 and
            capture.get("failed_workload_ticks") == 0,
            "F8-65 installed-public capture is incomplete")
    binding = execution.get("candidate_binding", {})
    raw_binding = binding.get("raw_capture_candidate", {})
    _expect(errors,
            binding.get("candidate_id") == ACTIVE_ID and
            binding.get("manifest_sha256") == file_sha256(candidate_path) and
            raw_binding.get("id") == V6_REJECTED_ID and
            raw_binding.get("manifest_sha256") == V6_MANIFEST_SHA256 and
            raw_binding.get("execution_identity_claimed") is False and
            raw_binding.get("qualification_status") == "remains-rejected",
            "exact-source capture relabels or loses rejected-v6 provenance")
    archive = execution.get("artifacts", {})
    archive_path = root / str(archive.get("tracked_capture_archive", ""))
    _expect(errors,
            archive_path.is_file() and
            archive.get("tracked_capture_archive_sha256") ==
            file_sha256(archive_path),
            "F8-65 tracked capture archive hash drift")

    supersedes = candidate.get("supersedes", {})
    _expect(errors, supersedes.get("candidate_id") == V6_REJECTED_ID and
            supersedes.get("manifest_sha256") == V6_MANIFEST_SHA256,
            "candidate does not supersede exact v6 identity")
    _expect(errors, file_sha256(v1_candidate_path) == V1_MANIFEST_SHA256,
            "rejected v1 candidate manifest changed")
    _expect(errors, file_sha256(v2_candidate_path) == V2_MANIFEST_SHA256,
            "rejected v2 candidate manifest changed")
    _expect(errors, file_sha256(v3_candidate_path) == V3_MANIFEST_SHA256,
            "superseded v3 candidate manifest changed")
    _expect(errors, file_sha256(v4_candidate_path) == V4_MANIFEST_SHA256,
            "superseded v4 candidate manifest changed")
    _expect(errors, file_sha256(v5_candidate_path) == V5_MANIFEST_SHA256,
            "rejected v5 candidate manifest changed")
    _expect(errors, file_sha256(v6_candidate_path) == V6_MANIFEST_SHA256,
            "rejected v6 candidate manifest changed")
    _expect(errors, file_sha256(v5_failure_path) == V5_FAILURE_SHA256,
            "candidate-v5 F8-24 failure evidence changed")
    _expect(errors, v1_rejection.get("candidate_id") == V1_REJECTED_ID and
            v1_rejection.get("candidate_manifest_sha256") == V1_MANIFEST_SHA256,
            "v1 rejection record lost its candidate identity")
    _expect(errors, v1_rejection.get("status") == "rejected-non-executable",
            "v1 candidate is not durably rejected")
    v1_replacement = v1_rejection.get("replacement", {})
    _expect(errors, v1_replacement.get("candidate_id") == V2_REJECTED_ID and
            v1_replacement.get("candidate_manifest_sha256") == V2_MANIFEST_SHA256,
            "v1 rejection history no longer routes to exact v2")
    _expect(errors, v2_rejection.get("candidate_id") == V2_REJECTED_ID and
            v2_rejection.get("candidate_manifest_sha256") == V2_MANIFEST_SHA256,
            "v2 rejection record lost its candidate identity")
    _expect(errors,
            v2_rejection.get("status") == "rejected-telemetry-capacity-overflow",
            "v2 candidate is not durably rejected")
    v2_replacement = v2_rejection.get("replacement", {})
    _expect(errors, v2_replacement.get("candidate_id") == V3_REJECTED_ID and
            v2_replacement.get("candidate_manifest_sha256") ==
            V3_MANIFEST_SHA256,
            "v2 rejection history no longer routes to exact v3")
    _expect(errors, v3_rejection.get("candidate_id") == V3_REJECTED_ID and
            v3_rejection.get("candidate_manifest_sha256") == V3_MANIFEST_SHA256,
            "v3 supersession record lost its candidate identity")
    _expect(errors,
            v3_rejection.get("status") == "superseded-incomplete-single-path",
            "v3 candidate is not durably superseded")
    v3_replacement = v3_rejection.get("replacement", {})
    _expect(errors, v3_replacement.get("candidate_id") == V4_REJECTED_ID and
            v3_replacement.get("candidate_manifest_sha256") ==
            V4_MANIFEST_SHA256,
            "v3 supersession history no longer routes to exact v4")
    _expect(errors, v4_rejection.get("candidate_id") == V4_REJECTED_ID and
            v4_rejection.get("candidate_manifest_sha256") ==
            V4_MANIFEST_SHA256,
            "v4 supersession record lost its candidate identity")
    _expect(errors,
            v4_rejection.get("status") ==
            "superseded-repaired-matrix-refreeze",
            "v4 candidate is not durably superseded")
    v4_replacement = v4_rejection.get("replacement", {})
    _expect(errors, v4_replacement.get("candidate_id") == V5_REJECTED_ID and
            v4_replacement.get("candidate_manifest_sha256") ==
            V5_MANIFEST_SHA256,
            "v4 supersession record no longer routes to exact v5")
    _expect(errors, v5_rejection.get("candidate_id") == V5_REJECTED_ID and
            v5_rejection.get("candidate_manifest_sha256") ==
            V5_MANIFEST_SHA256,
            "v5 rejection record lost its candidate identity")
    _expect(errors,
            v5_rejection.get("status") ==
            "rejected-required-linux-cuda-suite",
            "v5 candidate is not durably rejected")
    _expect(errors,
            v5_rejection.get("failure_evidence") ==
            V5_FAILURE_PATH.relative_to(ROOT).as_posix() and
            v5_rejection.get("failure_evidence_sha256") == V5_FAILURE_SHA256,
            "v5 rejection record lost its exact failure evidence")
    v5_replacement = v5_rejection.get("replacement", {})
    _expect(errors, v5_replacement.get("candidate_id") == V6_REJECTED_ID and
            v5_replacement.get("candidate_manifest_sha256") ==
            V6_MANIFEST_SHA256,
            "v5 rejection record no longer routes to exact v6")
    for label, rejection in (("v1", v1_rejection), ("v2", v2_rejection),
                             ("v3", v3_rejection), ("v4", v4_rejection),
                             ("v5", v5_rejection)):
        _expect(errors, rejection.get("evidence_union_allowed") is False,
                f"{label} rejection record permits evidence union")
        invalidated = {
            item["target"]: item
            for item in rejection.get("invalidated_physical_rows", [])
        }
        _expect(errors, set(invalidated) == set(inventory_targets),
                f"{label} rejection does not invalidate every physical row")
        for target_id, item in invalidated.items():
            _expect(errors,
                    item.get("status") == "invalidated-requalify-replacement",
                    f"{label} rejected row {target_id} is not invalidated")

    _expect(errors, v6_failure_path.is_file() and
            file_sha256(v6_failure_path) == V6_FAILURE_SHA256,
            "candidate-v6 F8-24 failure evidence changed")
    _expect(errors, f8_64_repair_path.is_file() and
            file_sha256(f8_64_repair_path) == F8_64_REPAIR_SHA256,
            "F8-64 repair evidence changed")
    _expect(errors, v6_rejected,
            "v6 candidate is not durably rejected")
    _expect(errors,
            v6_rejection.get("failure_evidence") ==
            V6_FAILURE_PATH.relative_to(ROOT).as_posix() and
            v6_rejection.get("failure_evidence_sha256") == V6_FAILURE_SHA256,
            "v6 rejection record lost its exact failure evidence")
    _expect(errors,
            v6_rejection.get("repair_evidence") ==
            F8_64_REPAIR_PATH.relative_to(ROOT).as_posix() and
            v6_rejection.get("repair_evidence_sha256") ==
            F8_64_REPAIR_SHA256,
            "v6 rejection record lost its exact repair evidence")
    v6_replacement = v6_rejection.get("replacement", {})
    _expect(errors,
            v6_replacement.get("status") ==
            "frozen-awaiting-physical-qualification" and
            v6_replacement.get("freeze_task") == "F8-65" and
            v6_replacement.get("candidate_id") == ACTIVE_ID and
            v6_replacement.get("candidate_manifest_sha256") ==
            file_sha256(candidate_path) and
            v6_replacement.get("source_revision") == ACTIVE_SOURCE_REVISION and
            v6_replacement.get("source_tree") == ACTIVE_SOURCE_TREE and
            v6_replacement.get("ci_run_id") == ACTIVE_CI_RUN_ID,
            "v6 rejection does not route to exact F8-65 replacement")
    _expect(errors, v6_rejection.get("evidence_union_allowed") is False,
            "v6 rejection record permits evidence union")
    v6_invalidated = {
        item["target"]: item
        for item in v6_rejection.get("invalidated_physical_rows", [])
    }
    _expect(errors, set(v6_invalidated) == set(inventory_targets),
            "v6 rejection does not invalidate every physical row")
    for target_id, item in v6_invalidated.items():
        _expect(errors,
                item.get("status") == "invalidated-requalify-replacement",
                f"v6 rejected row {target_id} is not routed to replacement")

    routing = candidate.get("qualification_routing", {})
    _expect(errors, set(routing.get("tasks", [])) == QUALIFICATION_TASKS,
            "qualification routing task set drift")
    _expect(errors, routing.get("accepted_candidate") == ACTIVE_ID,
            "qualification routing names the wrong candidate")
    _expect(errors, routing.get("evidence_union_allowed") is False,
            "qualification routing permits evidence union")
    candidate_sha256 = file_sha256(candidate_path)
    for task_id in sorted(QUALIFICATION_TASKS):
        task = load_json(root / TASK_DIR.relative_to(ROOT) / f"{task_id}.json")
        _expect(errors, "F8-65" in task.get("dependencies", []),
                f"{task_id} does not depend on F8-65")
        scope = task.get("scope", "")
        _expect(errors, ACTIVE_ID in scope,
                f"{task_id} does not route to the exact replacement")
        _expect(errors, ACTIVE_SOURCE_REVISION in scope and
                ACTIVE_SOURCE_TREE in scope,
                f"{task_id} does not bind exact source identity")
        _expect(errors, candidate_sha256 in scope,
                f"{task_id} does not bind exact candidate manifest")
        _expect(errors, str(ACTIVE_CI_RUN_ID) in scope,
                f"{task_id} does not bind exact protected CI run")
        _expect(errors, V6_REJECTED_ID in scope and V5_REJECTED_ID in scope and
                V4_REJECTED_ID in scope and
                V3_REJECTED_ID in scope and
                "interim" in scope and "cannot be unioned" in scope,
                f"{task_id} does not preserve candidate history")
    exit_dependencies = set(load_json(
        root / TASK_DIR.relative_to(ROOT) / "F8-99.json"
    ).get("dependencies", []))
    _expect(errors, QUALIFICATION_TASKS |
            {"F8-33", "F8-34", "F8-35", "F8-36", "F8-37", "F8-38",
             "F8-52", "F8-61", "F8-62", "F8-63", "F8-64", "F8-65"}
            <= exit_dependencies,
            "F8-99 lacks a repair, refreeze or qualification dependency")
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
