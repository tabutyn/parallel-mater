#!/usr/bin/env python3
"""Validate the single active realtime qualification candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v3.json"
)
V1_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v1-rejection.json"
)
V2_REJECTION_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v2-rejection.json"
)
V1_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v1.json"
)
V2_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v2.json"
)
INVENTORY_PATH = (
    ROOT / "conformance/v1/qualification/kernel-device-obligations-v1.json"
)
PROFILE_PATH = ROOT / "conformance/v1/qualification/realtime-profile-v1.json"
ORACLE_PATH = ROOT / "conformance/v1/realtime/oracle-suite-v1.json"
MANIFEST_PATH = ROOT / "cmake/SlangKernelManifest.cmake"
TASK_DIR = ROOT / "docs/roadmap/tasks"

ACTIVE_ID = "realtime-single-step-v1-3671f18"
ACTIVE_SOURCE_REVISION = "3671f180ed951d8a62cb34e5ec4ce1dab52cd560"
ACTIVE_SOURCE_TREE = "785513efa475e8b3b1010a79cb2142e33463089a"
V1_REJECTED_ID = "realtime-single-step-v1-dea47c1"
V2_REJECTED_ID = "realtime-single-step-v1-fd78fa8"
V1_MANIFEST_SHA256 = (
    "c25c010be3a0e908537e3c6542c1759b28b10cfb4a4169b51bb2834233411d43"
)
V2_MANIFEST_SHA256 = (
    "f4f1c7807dd50b07857c568b9df3e8e2ccd6fd3870907238dc9dcfab45b493af"
)
REQUIRED_CLOSURES = {
    "kernel",
    "runtime",
    "adapter_binding",
    "dispatch",
    "schedule",
    "telemetry",
    "profile",
    "public_oracle",
}
QUALIFICATION_TASKS = {f"F8-{number}" for number in range(23, 31)}


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
    v1_candidate_path = root / V1_CANDIDATE_PATH.relative_to(ROOT)
    v2_candidate_path = root / V2_CANDIDATE_PATH.relative_to(ROOT)
    inventory = load_json(root / INVENTORY_PATH.relative_to(ROOT))
    profile = load_json(root / PROFILE_PATH.relative_to(ROOT))
    oracle = load_json(root / ORACLE_PATH.relative_to(ROOT))
    v1_rejection = load_json(v1_rejection_path)
    v2_rejection = load_json(v2_rejection_path)
    if candidate is None:
        candidate = load_json(candidate_path)

    errors: list[str] = []
    _expect(errors, candidate.get("schema") == "parallel-mater-realtime-candidate/v2",
            "candidate schema is not v2")
    _expect(errors, candidate.get("id") == ACTIVE_ID,
            "active candidate id changed")
    _expect(errors, candidate.get("task") == "F8-38",
            "candidate freeze task is not F8-38")
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

    closures = {
        item.get("category"): item
        for item in source.get("protected_closures", [])
    }
    _expect(errors, set(closures) == REQUIRED_CLOSURES,
            "protected closure categories are incomplete")
    _expect(errors, len(closures) == len(source.get("protected_closures", [])),
            "protected closure categories are duplicated")
    for category in sorted(REQUIRED_CLOSURES & set(closures)):
        closure = closures[category]
        patterns = closure.get("patterns", [])
        _expect(errors, bool(patterns), f"{category} closure has no patterns")
        count, digest = closure_identity(root, patterns)
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
    _expect(errors, coverage.get("canonical_module_count") == len(canonical),
            "canonical module count drift")
    _expect(errors, coverage.get("canonical_entry_count") ==
            sum(len(entries) for entries in canonical.values()),
            "canonical entry count drift")
    _expect(errors, coverage.get("kernel_group_count") ==
            len(inventory.get("kernel_groups", [])),
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
    _expect(errors, {"cpu-runtime", "cuda-runtime", "vulkan-runtime", "metal-runtime"}
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
    artifact_text = " ".join(item.get("name", "")
                             for item in actions.get("artifacts", [])).lower()
    for platform in ("ubuntu", "windows", "macos", "android", "ipad"):
        _expect(errors, platform in artifact_text,
                f"cross-platform evidence lacks {platform}")
    for artifact in actions.get("artifacts", []):
        for key in ("archive_sha256", "extracted_tree_sha256"):
            _expect(errors, bool(re.fullmatch(r"[0-9a-f]{64}",
                                              str(artifact.get(key, "")))),
                    f"artifact {artifact.get('name')} lacks {key}")
    execution_path = root / str(candidate.get("execution_evidence", ""))
    _expect(errors, execution_path.is_file(),
            "installed production CPU execution evidence is missing")
    execution = load_json(execution_path) if execution_path.is_file() else {}
    execution_candidate = execution.get("candidate", {})
    _expect(errors, execution_candidate.get("id") == ACTIVE_ID,
            "installed execution evidence names the wrong candidate")
    _expect(errors, execution_candidate.get("manifest_sha256") ==
            file_sha256(candidate_path),
            "installed execution evidence candidate manifest hash drift")
    installed = execution.get("installed_public_profile", {})
    _expect(errors, installed.get("runtime_variant") == "cpu-runtime" and
            installed.get("status") == "passed",
            "installed production CPU public profile did not pass")
    _expect(errors, installed.get("driver_sha256") in {
        item.get("sha256") for item in binaries
    }, "installed execution driver is not frozen in native binary hashes")
    for key in ("plain_observation_sha256", "validation_observation_sha256"):
        _expect(errors, bool(re.fullmatch(r"[0-9a-f]{64}",
                                          str(installed.get(key, "")))),
                f"installed public profile lacks {key}")
    capacity = installed.get("exact_capacity_regression", {})
    _expect(errors, capacity.get("status") == "passed",
            "installed exact-capacity regression did not pass")
    _expect(errors, capacity.get("workload") == {
        "rigid_bodies_live": 8,
        "rigid_bodies_capacity": 8,
        "constraints_live": 8,
        "constraints_capacity": 8,
        "contact_capacity": 2048,
    }, "installed exact-capacity workload drift")
    _expect(errors, capacity.get("work") == {
        "dispatches": 20,
        "workgroups": 1421,
        "rigid_iterations": 8,
        "spatial_builds": 1,
        "spatial_publications": 1,
        "time_advances": 1,
        "queue_submissions": 1,
        "intra_tick_host_readbacks": 0,
    }, "installed exact-capacity work accounting drift")
    _expect(errors, bool(re.fullmatch(r"[0-9a-f]{64}",
                                      str(capacity.get("observation_sha256", "")))),
            "installed exact-capacity observation lacks SHA-256")

    supersedes = candidate.get("supersedes", {})
    _expect(errors, supersedes.get("candidate_id") == V2_REJECTED_ID and
            supersedes.get("manifest_sha256") == V2_MANIFEST_SHA256,
            "candidate does not supersede exact v2 identity")
    _expect(errors, file_sha256(v1_candidate_path) == V1_MANIFEST_SHA256,
            "rejected v1 candidate manifest changed")
    _expect(errors, file_sha256(v2_candidate_path) == V2_MANIFEST_SHA256,
            "rejected v2 candidate manifest changed")
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
    _expect(errors, v2_replacement.get("candidate_id") == ACTIVE_ID and
            v2_replacement.get("candidate_manifest_sha256") ==
            file_sha256(candidate_path),
            "v2 rejection record routes to the wrong replacement")
    for label, rejection in (("v1", v1_rejection), ("v2", v2_rejection)):
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

    routing = candidate.get("qualification_routing", {})
    _expect(errors, set(routing.get("tasks", [])) == QUALIFICATION_TASKS,
            "qualification routing task set drift")
    _expect(errors, routing.get("accepted_candidate") == ACTIVE_ID,
            "qualification routing names the wrong candidate")
    _expect(errors, routing.get("evidence_union_allowed") is False,
            "qualification routing permits evidence union")
    for task_id in sorted(QUALIFICATION_TASKS):
        task = load_json(root / TASK_DIR.relative_to(ROOT) / f"{task_id}.json")
        _expect(errors, "F8-36" in task.get("dependencies", []),
                f"{task_id} does not depend on F8-36")
        _expect(errors, "F8-38" in task.get("dependencies", []),
                f"{task_id} does not depend on F8-38")
        scope = task.get("scope", "")
        _expect(errors, ACTIVE_ID in scope,
                f"{task_id} does not route to the exact replacement")
        _expect(errors, V1_REJECTED_ID not in scope,
                f"{task_id} routes to rejected v1")
        _expect(errors, V2_REJECTED_ID in scope and
                "rejected for qualification" in scope,
                f"{task_id} does not preserve v2 rejection routing")
    exit_dependencies = set(load_json(
        root / TASK_DIR.relative_to(ROOT) / "F8-99.json"
    ).get("dependencies", []))
    _expect(errors, QUALIFICATION_TASKS |
            {"F8-33", "F8-34", "F8-35", "F8-36", "F8-37", "F8-38"}
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
