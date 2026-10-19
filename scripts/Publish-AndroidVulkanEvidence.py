#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Publish a Test-AndroidVulkan.py capture into the evidence registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
KEY_FIELDS = (
    "first_family", "second_family", "scope", "mode", "direction",
    "material_profile")


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".xml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def matching_rule(profile, capability):
    matches = []
    for rule in profile["rules"]:
        match = rule.get("match", {})
        if any(capability.get(field) not in match.get(field, [])
               for field in KEY_FIELDS):
            continue
        pairs = rule.get("mode_directions")
        if pairs is None or [capability["mode"], capability["direction"]] in pairs:
            matches.append(rule)
    if len(matches) != 1:
        raise ValueError("advertised capability does not match exactly one rule")
    return matches[0]


def required_contracts(profile, rule):
    required = {}
    for source in (profile.get("default_required", {}), rule.get("required", {})):
        for scope, contracts in source.items():
            required.setdefault(scope, []).extend(contracts)
    trajectory = rule.get("trajectory_contract")
    if trajectory:
        required.setdefault("execution", []).append(trajectory)
        required.setdefault("reference_regression", []).append(trajectory)
    return required


def log_name(record):
    arguments = record.get("arguments", [])
    suffix = ""
    for selector in ("--contract", "--case", "--quality",
                     "--lifecycle-profile"):
        if selector in arguments:
            suffix = "-" + arguments[arguments.index(selector) + 1]
            break
    if not suffix and arguments:
        suffix = "-" + arguments[0].removeprefix("--")
    return record["test"] + suffix + ".log"


def qualification_module():
    path = ROOT / "conformance/v1/qualification.py"
    spec = importlib.util.spec_from_file_location("android_qualification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path,
                        default=ROOT / "out/android/native-vulkan-tests")
    parser.add_argument("--capture-date", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    capture_root = args.capture.resolve()
    capture = load(capture_root / "capture.json")
    source_commit = capture["source_commit"]
    identity = ("android-pixel7-slang-vulkan-" + source_commit[:7])
    evidence_root = ROOT / "conformance/v1/evidence"
    package = evidence_root / identity
    if package.exists():
        if not args.replace:
            raise SystemExit(f"evidence package exists: {package}; pass --replace")
        shutil.rmtree(package)
    package.mkdir(parents=True)

    payload = capture_root / "payload"
    binary = payload / "libparallel_mater_driver_slang_vulkan.so"
    apk = ROOT / "examples/android/app/build/outputs/apk/debug/app-debug.apk"
    if sha256(binary) != capture["driver_sha256"]:
        raise SystemExit("captured Vulkan driver hash does not match payload")
    if sha256(apk) != capture["apk_sha256"]:
        raise SystemExit("captured APK hash does not match build output")

    shutil.copytree(capture_root / "physics-qualification",
                    package / "physics-qualification")
    shutil.copytree(capture_root / "reports", package / "reports")
    shutil.copy2(capture_root / "results.json", package / "results.json")
    logs = package / "logs"
    logs.mkdir()
    for record in capture["native_runs"]:
        name = log_name(record)
        source = capture_root / name
        if not source.is_file():
            raise SystemExit(f"captured device log is missing: {name}")
        shutil.copy2(source, logs / name)

    canonical_capture = dict(capture)
    canonical_capture["capture_date"] = args.capture_date
    canonical_capture["toolchain"] = dict(capture["toolchain"])
    canonical_capture["toolchain"]["ndk"] = Path(
        str(capture["toolchain"]["ndk"])).name
    write(package / "android-capture.json", canonical_capture)

    relative_package = package.relative_to(ROOT)
    capabilities_path = package / "physics-qualification/capabilities.json"
    capabilities = load(capabilities_path)
    provenance = load(package / "physics-qualification/provenance.json")
    if provenance["source_commit"] != source_commit:
        raise SystemExit("capture and runner provenance revisions differ")
    if capabilities != load(capture_root / "capabilities.json"):
        raise SystemExit("advertised capability captures differ")

    reference_path = package / "reports/reference.json"
    reference = load(reference_path)
    reference_cases = {item["id"]: item for item in reference["cases"]}
    quality_reports = {
        path.stem: load(path)
        for path in (package / "reports/quality").glob("*.json")}
    lifecycle_reports = {
        path.stem: load(path)
        for path in (package / "reports/lifecycle").glob("*.json")}

    profile_path = ROOT / "conformance/v1/qualification/capability-obligations-v1.json"
    profile = load(profile_path)
    contracts = {item["id"]: item for item in profile["contracts"]}

    def artifact(scope, contract_id):
        base = contract_id.split("#", 1)[0]
        if scope == "compilation":
            path, status = capabilities_path, "passed"
        elif base.startswith("case:"):
            case_id = base.split(":", 1)[1]
            path = package / f"physics-qualification/cases/{case_id}.json"
            result = load(path)
            status = ("passed" if result.get("execution_status") == "completed"
                      else result.get("execution_status", "failed"))
            if scope == "reference_regression":
                path = reference_path
                comparison = reference_cases[case_id]
                status = ("passed" if comparison.get("passed") is True else
                          "unsupported" if comparison.get("status") == "unsupported"
                          else "failed")
        elif base.startswith("quality:"):
            profile_id = base.split(":", 1)[1]
            path = package / f"reports/quality/{profile_id}.json"
            report = quality_reports[profile_id]
            status = "passed" if report.get("passed") is True else "failed"
        elif base.startswith("lifecycle:"):
            profile_id = base.split(":", 1)[1]
            path = package / f"reports/lifecycle/{profile_id}.json"
            report = lifecycle_reports[profile_id]
            status = "passed" if report.get("passed") is True else "failed"
        else:
            raise ValueError(f"unsupported evidence contract: {contract_id}")
        return path, status

    evidence = {}
    unavailable_evidence = []
    for capability in capabilities:
        rule = matching_rule(profile, capability)
        missing = rule.get("missing", {})
        for scope, contract_ids in required_contracts(profile, rule).items():
            if scope in missing:
                continue
            for contract_id in contract_ids:
                try:
                    path, status = artifact(scope, contract_id)
                except (FileNotFoundError, KeyError):
                    # The obligations registry can advance after a physical
                    # capture. Omit contracts that were not run so the
                    # qualification evaluator reports them as missing instead
                    # of letting a stale capture claim new coverage.
                    unavailable_evidence.append((scope, contract_id))
                    continue
                contract = contracts[contract_id]
                evidence[(scope, contract_id)] = {
                    "artifact_path": str(path.relative_to(ROOT)),
                    "artifact_sha256": sha256(path),
                    "contract": contract_id,
                    "contract_sha256": contract["sha256"],
                    "scope": scope,
                    "source_commit": source_commit,
                    "status": status,
                }

    driver_info = capture["device"]["vulkan_driver_info"]
    qualification = {
        "schema": "parallel-mater-conformance-qualification-result/v1",
        "obligations_id": profile["id"],
        "obligations_sha256": sha256(profile_path),
        "execution_status": "completed",
        "source": {"commit": source_commit, "dirty": False},
        "driver": {"name": "slang-vulkan", "version": "0.2.0",
                   "binary_sha256": capture["driver_sha256"]},
        "device": {"name": capture["device"]["gpu"],
                   "vendor_id": capture["device"]["vendor_id"],
                   "device_id": capture["device"]["device_id"],
                   "driver_version": driver_info},
        "toolchain": {"compiler": "Android Clang",
                      "version": provenance["host_compiler"],
                      "ndk": canonical_capture["toolchain"]["ndk"],
                      "slang": canonical_capture["toolchain"]["slang"]},
        "advertised_capabilities": capabilities,
        "evidence": [evidence[key] for key in sorted(evidence)],
    }
    qualification_path = evidence_root / (identity + "-qualification.json")
    write(qualification_path, qualification)

    files = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    capture_index = {
        "schema": "parallel-mater-android-device-capture/v1",
        "id": identity,
        "capture_date": args.capture_date,
        "source": qualification["source"],
        "driver": qualification["driver"],
        "device": {**qualification["device"],
                   "model": capture["device"]["model"],
                   "android_version": capture["device"]["android_version"],
                   "vulkan_api_version": capture["device"]["vulkan_api_version"]},
        "toolchain": qualification["toolchain"],
        "claims": {"public_api_physics": True,
                   "analytic_quality": True,
                   "adverse_lifecycle": True,
                   "cuda_reference_regression": False,
                   "gallery_scene_sweep": False,
                   "thermal": False,
                   "timing": False},
        "summary": {"device_runs_passed": sum(
                        item["passed"] for item in capture["native_runs"]),
                    "device_runs_total": len(capture["native_runs"]),
                    "trajectory_passed": reference["passed"],
                    "trajectory_failed": reference["failed"],
                    "trajectory_unsupported": reference["unsupported"],
                    "quality_profiles_passed": sum(
                        item.get("passed") is True for item in quality_reports.values()),
                    "lifecycle_profiles_passed": sum(
                        item.get("passed") is True for item in lifecycle_reports.values())},
        "limitations": [
            "The CUDA golden trajectory comparison has 15 numerical mismatches; exact per-case differences are retained in reports/reference.json.",
            "Gallery rendering, lifecycle recovery, sustained runtime, thermal behavior and timing remain separate F7-18 evidence."],
        "file_count": len(files),
        "files": files,
    }
    write(package / "capture-index.json", capture_index)

    module = qualification_module()
    report = module.evaluate(profile, qualification, sha256(profile_path))
    print(f"Qualification: {report['qualified_count']}/"
          f"{report['capability_count']} capabilities; "
          f"{report['failure_count']} explicit failures")
    if unavailable_evidence:
        print("Capture predates " + str(len(set(unavailable_evidence))) +
              " required evidence contracts; retained as explicit gaps")

    registry_path = evidence_root / "evidence-sources-v1.json"
    registry = load(registry_path)
    registry["obligations"]["sha256"] = sha256(profile_path)
    entry = {"id": identity, "path": str(qualification_path.relative_to(ROOT)),
             "sha256": sha256(qualification_path)}
    values = [item for item in registry["qualification_results"]
              if item.get("id") != identity]
    values.append(entry)
    registry["qualification_results"] = sorted(values, key=lambda item: item["id"])
    write(registry_path, registry)
    print(f"Published {relative_package} and registered {identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
