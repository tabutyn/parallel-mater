#!/usr/bin/env python3
"""Build and check the revisioned per-driver capability evidence index."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SOURCES_SCHEMA = "parallel-mater-evidence-sources/v1"
INDEX_SCHEMA = "parallel-mater-driver-evidence-index/v1"
LEGACY_SCHEMA = "parallel-mater-physics-qualification/v1"


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".xml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _qualification_module(root: Path):
    path = root / "conformance/v1/qualification.py"
    spec = importlib.util.spec_from_file_location("pm_evidence_qualification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _key_text(key: dict[str, Any]) -> str:
    fields = ("first_family", "second_family", "scope", "mode", "direction",
              "material_profile")
    return "/".join(str(key.get(field, "?")) for field in fields)


def _checked_file(root: Path, entry: dict[str, Any], failures: list[dict[str, Any]]) -> Path | None:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        failures.append({"source": entry.get("id"), "message": "source path is missing"})
        return None
    path = root / relative
    if not path.is_file():
        failures.append({"source": entry.get("id"), "message": "source file is missing",
                         "path": relative})
        return None
    actual = _sha256(path)
    if actual != entry.get("sha256"):
        failures.append({"source": entry.get("id"), "message": "source SHA-256 mismatch",
                         "path": relative, "actual": actual,
                         "expected": entry.get("sha256")})
        return None
    return path


def _driver_record(identity: str, labels: dict[str, str],
                   drivers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return drivers.setdefault(identity, {
        "id": identity,
        "label": labels.get(identity, identity),
        "status": "missing",
        "report_ids": [],
        "qualification_result_ids": [],
        "qualified_capability_count": 0,
        "unqualified_capability_count": 0,
    })


def _qualification_results(root: Path, entries: Any, obligations: dict[str, Any],
                           obligations_hash: str, labels: dict[str, str],
                           drivers: dict[str, dict[str, Any]],
                           reports: list[dict[str, Any]],
                           capabilities: list[dict[str, Any]],
                           known_failures: list[dict[str, Any]],
                           failures: list[dict[str, Any]]) -> None:
    if not isinstance(entries, list):
        failures.append({"source": "qualification_results",
                         "message": "expected an array"})
        return
    qualification = _qualification_module(root)
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            failures.append({"source": "qualification_results",
                             "message": "result entry needs an id"})
            continue
        identity = entry["id"]
        if identity in seen:
            failures.append({"source": identity, "message": "duplicate result id"})
            continue
        seen.add(identity)
        path = _checked_file(root, entry, failures)
        if path is None:
            continue
        raw = _load(path)
        report = qualification.evaluate(obligations, raw, obligations_hash)
        driver_value = raw.get("driver")
        driver_id = driver_value.get("name") if isinstance(driver_value, dict) else None
        if not isinstance(driver_id, str) or not driver_id:
            driver_id = "invalid:" + identity
        driver = _driver_record(driver_id, labels, drivers)
        driver["report_ids"].append(identity)
        driver["qualification_result_ids"].append(identity)
        driver["status"] = "qualified" if report["passed"] else "unqualified"
        driver["qualified_capability_count"] += report["qualified_count"]
        driver["unqualified_capability_count"] += (
            report["capability_count"] - report["qualified_count"])
        reports.append({
            "id": identity,
            "kind": "capability-qualification",
            "path": entry["path"],
            "sha256": entry["sha256"],
            "validated": report["passed"],
            "source": raw.get("source"),
            "driver": raw.get("driver"),
            "device": raw.get("device"),
            "toolchain": raw.get("toolchain"),
            "capability_count": report["capability_count"],
            "qualified_count": report["qualified_count"],
            "failure_count": report["failure_count"],
        })
        for capability in report["capabilities"]:
            capabilities.append({
                "driver": driver_id,
                "report": identity,
                "key": capability["key"],
                "key_text": _key_text(capability["key"]),
                "qualified": capability["qualified"],
                "rule_id": capability["rule_id"],
                "scopes": capability["scopes"],
            })
        for failure in report["failures"]:
            known_failures.append({
                "driver": driver_id,
                "report": identity,
                "kind": "qualification_failure",
                "path": failure["path"],
                "message": failure["message"],
                "actual": failure["actual"],
                "expected": failure["expected"],
            })


def _legacy_reports(root: Path, entries: Any, labels: dict[str, str],
                    drivers: dict[str, dict[str, Any]],
                    reports: list[dict[str, Any]], known_failures: list[dict[str, Any]],
                    failures: list[dict[str, Any]]) -> None:
    if not isinstance(entries, list):
        failures.append({"source": "supplemental_reports",
                         "message": "expected an array"})
        return
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kind") != "legacy-physics-v1":
            failures.append({"source": entry.get("id") if isinstance(entry, dict) else None,
                             "message": "unsupported supplemental report kind"})
            continue
        path = _checked_file(root, entry, failures)
        if path is None:
            continue
        raw = _load(path)
        if raw.get("schema") != LEGACY_SCHEMA:
            failures.append({"source": entry.get("id"),
                             "message": "legacy report schema mismatch"})
            continue
        case_failures: list[str] = []
        cases = raw.get("reference_regression", {}).get("cases", [])
        for case in cases if isinstance(cases, list) else []:
            case_id = str(case.get("id"))
            canonical = root / "conformance/v1/cases" / (case_id + ".json")
            if not canonical.is_file():
                case_failures.append(case_id + ": absent case")
            elif case.get("case_sha256") != _sha256(canonical):
                case_failures.append(str(case.get("id")) + ": stale case hash")
        binary_hashes = {
            value.get("path"): value.get("sha256")
            for value in raw.get("binary_hashes", []) if isinstance(value, dict)
        }
        host = raw.get("host", {})
        limitations = raw.get("limitations", [])
        unsupported = raw.get("regression_tests", {}).get("unsupported_reasons", {})
        baseline = raw.get("reference_regression", {}).get("baseline", {}).get("failures", [])
        for driver_entry in entry.get("drivers", []):
            driver_id = driver_entry.get("id")
            binary_path = driver_entry.get("binary_path")
            if not isinstance(driver_id, str) or binary_path not in binary_hashes:
                failures.append({"source": entry.get("id"),
                                 "message": "legacy driver binary is missing",
                                 "driver": driver_id, "path": binary_path})
                continue
            driver = _driver_record(driver_id, labels, drivers)
            driver["report_ids"].append(entry["id"])
            if driver["status"] == "missing":
                driver["status"] = "supplemental_only"
            reports.append({
                "id": entry["id"] + ":" + driver_id,
                "kind": "legacy-physics",
                "path": entry["path"],
                "sha256": entry["sha256"],
                "validated": not case_failures,
                "source": {"commit": raw.get("tested_source"), "dirty": None},
                "driver": {"name": driver_id, "version": None,
                           "binary_sha256": binary_hashes[binary_path]},
                "device": {"name": host.get("gpu"), "vendor_id": None,
                           "device_id": None, "driver_version": host.get("driver")},
                "toolchain": {
                    "compiler": "MSVC", "version": host.get("msvc"),
                    "slang": host.get("slang"), "cuda": host.get("cuda_toolkit"),
                    "dxc": host.get("dxc"),
                },
                "capability_count": 0,
                "qualified_count": 0,
                "failure_count": len(case_failures) + len(limitations) + len(unsupported),
                "scope": raw.get("scope"),
                "case_coverage_count": len(cases),
            })
            for message in case_failures:
                known_failures.append({"driver": driver_id, "report": entry["id"],
                                       "kind": "case_validation", "message": message})
            for message in limitations if isinstance(limitations, list) else []:
                known_failures.append({"driver": driver_id, "report": entry["id"],
                                       "kind": "limitation", "message": message})
            for name, message in unsupported.items() if isinstance(unsupported, dict) else []:
                if driver_id.split("-")[-1] in name:
                    known_failures.append({"driver": driver_id, "report": entry["id"],
                                           "kind": "unsupported", "case": name,
                                           "message": message})
            for item in baseline if isinstance(baseline, list) else []:
                known_failures.append({
                    "driver": driver_id, "report": entry["id"],
                    "kind": "historical_baseline_failure", "case": item.get("id"),
                    "message": item.get("first_difference", {}).get("message"),
                })


def build_index(root: Path, sources_path: Path) -> dict[str, Any]:
    sources = _load(sources_path)
    failures: list[dict[str, Any]] = []
    if sources.get("schema") != SOURCES_SCHEMA:
        failures.append({"source": "sources", "message": "source registry schema mismatch"})
    obligations_entry = sources.get("obligations")
    if not isinstance(obligations_entry, dict):
        raise ValueError("sources obligations entry is missing")
    obligations_path = _checked_file(root, obligations_entry, failures)
    obligations = _load(obligations_path) if obligations_path else {"contracts": []}
    obligations_hash = obligations_entry.get("sha256", "")
    expected = sources.get("expected_drivers", [])
    labels = {item["id"]: item["label"] for item in expected
              if isinstance(item, dict) and isinstance(item.get("id"), str)
              and isinstance(item.get("label"), str)}
    drivers: dict[str, dict[str, Any]] = {}
    for identity in labels:
        _driver_record(identity, labels, drivers)
    reports: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    known_failures: list[dict[str, Any]] = []
    _qualification_results(root, sources.get("qualification_results"), obligations,
                           obligations_hash, labels, drivers, reports, capabilities,
                           known_failures, failures)
    _legacy_reports(root, sources.get("supplemental_reports"), labels,
                    drivers, reports, known_failures, failures)
    for driver in drivers.values():
        if not driver["qualification_result_ids"]:
            known_failures.append({
                "driver": driver["id"], "report": None,
                "kind": "missing_qualification_result",
                "message": "No current capability-qualification result is registered.",
            })
    capabilities.sort(key=lambda item: (item["driver"], item["key_text"], item["report"]))
    reports.sort(key=lambda item: item["id"])
    known_failures.sort(key=lambda item: (
        item["driver"], item["kind"], str(item.get("case", "")),
        str(item.get("path", "")), item["message"]))
    return {
        "schema": INDEX_SCHEMA,
        "sources_id": sources.get("id"),
        "sources_sha256": _sha256(sources_path),
        "obligations_id": obligations.get("id"),
        "obligations_sha256": obligations_hash,
        "valid": not failures,
        "qualification_complete": all(
            driver["qualification_result_ids"] for driver in drivers.values()),
        "driver_count": len(drivers),
        "report_count": len(reports),
        "capability_count": len(capabilities),
        "qualified_capability_count": sum(item["qualified"] for item in capabilities),
        "drivers": list(drivers.values()),
        "reports": reports,
        "capabilities": capabilities,
        "known_failures": known_failures,
        "validation_failures": failures,
    }


def render_document_block(index: dict[str, Any]) -> str:
    lines = [
        "<!-- BEGIN GENERATED DRIVER EVIDENCE -->",
        "| Driver | Current capability result | Qualified records | Open evidence |",
        "|---|---|---:|---|",
    ]
    failures_by_driver: dict[str, list[dict[str, Any]]] = {}
    for failure in index["known_failures"]:
        failures_by_driver.setdefault(failure["driver"], []).append(failure)
    report_paths = {}
    for report in index["reports"]:
        report_paths.setdefault(report["id"].split(":", 1)[0], report["path"])

    def report_link(identity: str) -> str:
        path = report_paths.get(identity)
        return (f"[`{identity}`](../{path})" if path is not None
                else f"`{identity}`")

    for driver in index["drivers"]:
        current = (", ".join(report_link(value)
                             for value in driver["qualification_result_ids"])
                   or "None")
        failures = failures_by_driver.get(driver["id"], [])
        supplemental = [value for value in driver["report_ids"]
                        if value not in driver["qualification_result_ids"]]
        parts = []
        if supplemental:
            parts.append("Supplemental: " + ", ".join(
                report_link(value) for value in supplemental))
        counts: dict[str, int] = {}
        for failure in failures:
            counts[failure["kind"]] = counts.get(failure["kind"], 0) + 1
        if counts:
            parts.append(", ".join(f"{count} {kind.replace('_', ' ')}"
                                   for kind, count in sorted(counts.items())))
        lines.append(f"| {driver['label']} | {current} | "
                     f"{driver['qualified_capability_count']} | "
                     f"{'; '.join(parts) or 'None'} |")
    lines.extend([
        "",
        f"Index status: **{index['qualified_capability_count']} qualified capability "
        f"records from {index['report_count']} revisioned report records**. "
        "Supplemental and historical evidence remains visible but never certifies a capability.",
        "<!-- END GENERATED DRIVER EVIDENCE -->",
    ])
    return "\n".join(lines)


def _document_text(existing: str, begin: str, end: str, block: str) -> str:
    if begin not in existing or end not in existing:
        raise ValueError("documentation evidence markers are missing")
    prefix, remainder = existing.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    return prefix + block + suffix


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path,
                        default=root / "conformance/v1/evidence/evidence-sources-v1.json")
    parser.add_argument("--output", type=Path,
                        default=root / "conformance/v1/evidence/index-v1.json")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        sources_path = arguments.sources.resolve()
        index = build_index(root, sources_path)
        sources = _load(sources_path)
        document = sources["documentation"]
        document_path = root / document["path"]
        block = render_document_block(index)
        expected_document = _document_text(
            document_path.read_text(encoding="utf-8"), document["begin_marker"],
            document["end_marker"], block)
        output_text = json.dumps(index, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n"
        if arguments.check:
            stale = []
            if not arguments.output.is_file() or arguments.output.read_text(
                    encoding="utf-8") != output_text:
                stale.append(str(arguments.output))
            if document_path.read_text(encoding="utf-8") != expected_document:
                stale.append(str(document_path))
            if stale:
                print("evidence index is stale: " + ", ".join(stale))
                return 1
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(output_text, encoding="utf-8")
            document_path.write_text(expected_document, encoding="utf-8")
        if not index["valid"]:
            print(f"evidence index has {len(index['validation_failures'])} validation failures")
            return 1
        print(f"Evidence index: {index['qualified_capability_count']}/"
              f"{index['capability_count']} capabilities, {index['report_count']} reports")
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"evidence index: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
