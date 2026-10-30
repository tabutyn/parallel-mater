#!/usr/bin/env python3
"""Generate the F8-63 realtime candidate from immutable source evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v5.json"
)
OUTPUT_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v6.json"
)
SOURCE_REVISION = "fc2ffbafdb81631c4c7393336970cefc3d65f55e"
SOURCE_TREE = "fccd57eabd07dd4ae40cd38fdce8020d079a09da"
CANDIDATE_ID = "realtime-single-step-v1-fc2ffba"
PREVIOUS_ID = "realtime-single-step-v1-cfb104b"
PREVIOUS_SHA256 = (
    "3435be0a4d4caa2d014d82faba7272aaadeba546a555a2e35a77ff18d0d8a000"
)
LEASE_TOKEN = "978462c7-be4c-4a86-93e3-a46c22791373"
F8_62_EVIDENCE = "docs/roadmap/evidence/F8-62-LINUX-CUDA-GALLERY-REPAIR.json"
V5_FAILURE_EVIDENCE = (
    "docs/roadmap/evidence/F8-24-CANDIDATE-V5-LINUX-CUDA-FAILURE.json"
)
V5_FAILURE_SHA256 = (
    "529f43bf26c36cd989c8bd72c936ecdd1306b372d70d194071b71f5c0e9453dd"
)
QUALIFICATION_INSTRUCTIONS = {
    "F8-24": (
        "Execute every registered Linux CUDA tuple without changing physics, "
        "budgets or advertised capabilities."
    ),
    "F8-25": (
        "Execute every registered Linux Vulkan tuple; software rendering is "
        "not a physical GPU substitute. Do not change physics, budgets or "
        "advertised capabilities."
    ),
    "F8-26": (
        "Execute every registered Windows CUDA tuple with the same cases as "
        "D3D12, without changing physics, budgets or advertised capabilities."
    ),
    "F8-27": (
        "Execute every registered Windows D3D12 tuple after merged P1-59 "
        "parity, without changing physics, budgets or advertised capabilities."
    ),
    "F8-28": (
        "Execute every registered macOS Metal tuple; iOS is not a substitute. "
        "Do not change physics, budgets or advertised capabilities."
    ),
    "F8-29": (
        "Execute every registered iPhone/iPad Metal tuple; simulator or macOS "
        "is not a substitute. Do not change physics, budgets or advertised "
        "capabilities."
    ),
    "F8-30": (
        "Execute every registered Android Vulkan tuple with installed-binary "
        "verification and sustained thermal measurements, without changing "
        "physics, budgets or advertised capabilities."
    ),
    "F8-53": (
        "Requalify CPU and installed public contracts against the exact final "
        "all-supported-domain matrix without assuming unadvertised GPU modes "
        "or changing physics, budgets or capabilities."
    ),
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "candidate v5 changed; refusing to derive v6")
    require(file_sha256(ROOT / V5_FAILURE_EVIDENCE) == V5_FAILURE_SHA256,
            "candidate-v5 failure evidence changed; refusing to derive v6")
    actual_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", SOURCE_REVISION],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(actual_tree == SOURCE_TREE, "frozen source tree identity drift")

    run = ci.get("run", {})
    require(run.get("id") == 40982968850, "unexpected protected CI run")
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
        "task": "F8-63",
        "frozen_at": "2026-10-29",
        "lease_token": LEASE_TOKEN,
        "supersedes": {
            "candidate_id": PREVIOUS_ID,
            "manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": PREVIOUS_SHA256,
            "supersession_record": (
                "conformance/v1/qualification/"
                "realtime-candidate-v5-rejection.json"
            ),
        },
    })

    closures = candidate["source"]["protected_closures"]
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
            "The executable implementation is the exact clean F8-62 repair "
            "commit. It changes only five files in the protected Gallery "
            "closure, repairs the candidate-v5 Linux CUDA failures without "
            "changing physics equations, public capabilities, cases, budgets, "
            "tolerances or schedule, and passes the complete physical Linux "
            "CUDA suite. F8-63 adds only immutable freeze metadata, routing, "
            "evidence and checks outside that source identity."
        ),
        "repair_evidence": F8_62_EVIDENCE,
        "repair_evidence_sha256": file_sha256(ROOT / F8_62_EVIDENCE),
        "hash_method": (
            "SHA-256 over sorted repository-relative path + NUL + file bytes "
            "+ NUL"
        ),
        "protected_closures": closures,
    }
    candidate["sole_path_audit"]["audit_evidence"] = (
        "docs/roadmap/evidence/F8-63-SOLE-PATH-AUDIT.json"
    )

    candidate["native_binaries"] = ci["native_binaries"]
    candidate["compile_evidence"] = {
        "qualification_claim": False,
        "rule": (
            "Compilation, reflection, packaging and hosted execution are build "
            "evidence only. Compile-only mirrors and hosted runners qualify no "
            "physical row. F8-62 physical Linux CUDA evidence is separate."
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
    prerequisites["F8-62"] = {
        "task": "F8-62",
        "path": F8_62_EVIDENCE,
        "sha256": file_sha256(ROOT / F8_62_EVIDENCE),
    }
    candidate["prerequisite_evidence"] = [
        prerequisites[key] for key in sorted(prerequisites)
    ]
    candidate["execution_evidence"] = F8_62_EVIDENCE
    candidate["qualification_routing"].update({
        "accepted_candidate": CANDIDATE_ID,
        "interim_cpu_evidence": (
            "F8-23, candidate-v4 and candidate-v5 preflights remain immutable "
            "provenance and supply no candidate-v6 pass. F8-62 proves the full "
            "physical Linux CUDA regression on the exact repaired source; "
            "F8-24 through F8-30 and F8-53 still own final qualification rows."
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
        ROOT / "conformance/v1/qualification/realtime-candidate-v5-rejection.json"
    )
    write_json(rejection_path, {
        "schema": "parallel-mater-realtime-candidate-rejection/v1",
        "candidate_id": PREVIOUS_ID,
        "candidate_manifest": PREVIOUS_PATH.relative_to(ROOT).as_posix(),
        "candidate_manifest_sha256": PREVIOUS_SHA256,
        "status": "rejected-required-linux-cuda-suite",
        "superseded_at": "2026-10-29",
        "supersession_task": "F8-63",
        "historical_qualification_task": "F8-61",
        "failure_evidence": V5_FAILURE_EVIDENCE,
        "failure_evidence_sha256": V5_FAILURE_SHA256,
        "repair_evidence": F8_62_EVIDENCE,
        "repair_evidence_sha256": file_sha256(ROOT / F8_62_EVIDENCE),
        "reasons": [
            (
                "Candidate v5 failed the required complete physical Linux CUDA "
                "suite: 127/130 tests passed and three required Gallery tests "
                "failed."
            ),
            (
                "F8-62 repairs those failures in the Gallery closure and proves "
                "130/130 tests, including 42 CUDA-labeled tests, on the exact "
                "clean replacement source."
            ),
            (
                "Candidate-v5 passing observations remain immutable provenance "
                "and cannot qualify or be unioned with any candidate-v6 row."
            ),
        ],
        "replacement": {
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest": candidate_path.relative_to(ROOT).as_posix(),
            "candidate_manifest_sha256": candidate_sha256,
            "freeze_task": "F8-63",
        },
        "evidence_union_allowed": False,
        "invalidated_physical_rows": rejected_rows,
        "qualification_rule": (
            "No final F8 qualification report may name or reuse candidate v5. "
            "F8-24 through F8-30 and F8-53 must execute only "
            f"{CANDIDATE_ID} from F8-63."
        ),
    })

    common = (
        f"FINAL RELEASE GATE: qualify only {CANDIDATE_ID}, manifest "
        f"{candidate_sha256}, source {SOURCE_REVISION}, tree {SOURCE_TREE}, "
        "GitHub Actions exact-source run 40982968850, API 0.3/driver ABI 2 "
        "sole fixed-tick SDK/Gallery contract, executable "
        "12-scenario/37-measurement fixture/validator closure, "
        "F8-54/55/56/59/60/62 inputs, 49 capabilities and 30 public cases. "
        f"Candidate {PREVIOUS_ID}, manifest {PREVIOUS_SHA256}, source "
        "cfb104b1c530f859d9a1c3549d2e67c7de31c289, tree "
        "e6838f954384363e79a4485f3c2cfb6ff5e0ea79 failed the required physical "
        f"Linux CUDA suite in {V5_FAILURE_EVIDENCE} (SHA-256 "
        f"{V5_FAILURE_SHA256}) and is rejected historical provenance; its "
        "partial passing observations cannot be unioned with replacement "
        "evidence. Candidate realtime-single-step-v1-5a811c4 remains "
        "superseded historical provenance and cannot be unioned; candidate "
        "realtime-single-step-v1-3671f18 plus F8-23 remain "
        "historical/interim provenance and cannot be unioned; candidates "
        "realtime-single-step-v1-fd78fa8 and "
        "realtime-single-step-v1-dea47c1 remain rejected. "
    )
    suffix = (
        " Follow docs/roadmap/F8-REALTIME-60HZ.md and "
        "docs/roadmap/F8-SINGLE-EXECUTION-PATH.md."
    )
    for task_id, instruction in QUALIFICATION_INSTRUCTIONS.items():
        path = ROOT / "docs/roadmap/tasks" / f"{task_id}.json"
        task = load_json(path)
        require("F8-63" in task.get("dependencies", []),
                f"{task_id} does not depend on F8-63")
        task["scope"] = common + instruction + suffix
        write_json(path, task)

    exit_path = ROOT / "docs/roadmap/tasks/F8-99.json"
    exit_task = load_json(exit_path)
    for dependency in ("F8-62", "F8-63"):
        if dependency not in exit_task["dependencies"]:
            exit_task["dependencies"].append(dependency)
    write_json(exit_path, exit_task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-evidence",
        type=Path,
        required=True,
        help="independently hashed F8-63 protected CI evidence JSON",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--route",
        action="store_true",
        help="also route the active pointer, v5 rejection and task cards",
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
