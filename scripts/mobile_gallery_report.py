"""Validation for the shared Android/iOS Gallery qualification report."""

SCHEMA = "parallel-mater-mobile-gallery-qualification/v2"
EXECUTION_CONTRACT = "fixed-tick-60hz-v1"

LIFECYCLE_CHECKS = {
    "attach",
    "resize",
    "background",
    "foreground",
    "failure_recovery",
    "invalid_scene_recovery",
}
CONTROL_CHECKS = {
    "scene_selection",
    "pause_resume",
    "camera",
    "directional",
    "count",
    "primary_action",
}


def _validate_checks(report, field, required):
    checks = report.get(field)
    if not isinstance(checks, list):
        raise RuntimeError(f"Mobile Gallery report has no {field} check array")
    by_id = {check.get("id"): check for check in checks if isinstance(check, dict)}
    missing = sorted(required - set(by_id))
    if missing:
        raise RuntimeError(f"Mobile Gallery report is missing {field} checks: {', '.join(missing)}")
    failed = sorted(identifier for identifier in required
                    if by_id[identifier].get("applicable", True) and
                    not by_id[identifier].get("passed"))
    if failed:
        raise RuntimeError(f"Mobile Gallery {field} checks failed: {', '.join(failed)}")


def validate_mobile_gallery_report(report, nonce=None, expected_scenes=28):
    """Validate the common schema and return supported/unavailable scene lists."""
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise RuntimeError(f"Expected {SCHEMA}")
    if nonce is not None and report.get("nonce") != nonce:
        raise RuntimeError("Mobile Gallery report nonce does not match this run")
    for field in ("revision", "platform", "backend", "device"):
        if not isinstance(report.get(field), str) or not report[field]:
            raise RuntimeError(f"Mobile Gallery report has no {field}")
    if report.get("execution_contract") != EXECUTION_CONTRACT:
        raise RuntimeError("Mobile Gallery report has the wrong execution contract")
    renderer = report.get("renderer")
    if (not isinstance(renderer, dict) or
            renderer.get("kind") not in {"native-gpu", "software-image"} or
            renderer.get("readback") not in {"diagnostic-only", "host-frame"}):
        raise RuntimeError("Mobile Gallery report has invalid renderer identity")
    if not report.get("completed"):
        raise RuntimeError("Mobile Gallery qualification did not complete")
    if not report.get("passed") or report.get("failures"):
        raise RuntimeError("Mobile Gallery qualification retained failures")
    frames = report.get("frames_per_scene")
    if not isinstance(frames, int) or not 1 <= frames <= 100000:
        raise RuntimeError("Mobile Gallery report has invalid frames_per_scene")
    _validate_checks(report, "lifecycle", LIFECYCLE_CHECKS)
    _validate_checks(report, "controls", CONTROL_CHECKS)

    scenes = report.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != expected_scenes:
        raise RuntimeError("Shared scene catalog is incomplete")
    supported = [scene for scene in scenes if scene.get("available")]
    unavailable = [scene for scene in scenes if not scene.get("available")]
    if not supported:
        raise RuntimeError("Mobile Gallery report contains no supported scenes")
    if not all(scene.get("passed") and scene.get("steps") == frames
               for scene in supported):
        raise RuntimeError("Supported mobile Gallery scene checks failed")
    if not all(scene.get("skipped") and scene.get("reason")
               for scene in unavailable):
        raise RuntimeError("Unavailable mobile Gallery scenes lack exact reasons")
    return supported, unavailable
