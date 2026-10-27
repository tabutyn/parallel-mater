import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mobile_gallery_report import (  # noqa: E402
    CONTROL_CHECKS,
    EXECUTION_CONTRACT,
    LIFECYCLE_CHECKS,
    SCHEMA,
    validate_mobile_gallery_report,
)


def passing_report():
    checks = lambda identifiers: [
        {"id": identifier, "applicable": True, "passed": True}
        for identifier in sorted(identifiers)
    ]
    scenes = [
        {
            "id": f"scene-{index}",
            "scene": f"Scene {index}",
            "available": True,
            "skipped": False,
            "passed": True,
            "steps": 3,
            "timing_ms": {
                phase: {key: 1.0 for key in ("mean", "p50", "p95", "p99")}
                for phase in ("frame", "physics", "render", "present")
            },
            "simulation_to_wall_clock_rate": 1.0,
            "realtime_work": {
                field: 0 for field in (
                    "dispatches", "workgroups", "barriers", "submissions",
                    "bytes_moved", "time_advances", "spatial_builds",
                    "spatial_publications", "intra_tick_host_readbacks")
            },
            "thermal_state_before": "nominal",
            "thermal_state_after": "nominal",
        }
        for index in range(27)
    ]
    scenes.append({
        "id": "unavailable",
        "scene": "Unavailable",
        "available": False,
        "skipped": True,
        "passed": False,
        "steps": 3,
        "reason": "Missing backend feature",
    })
    return {
        "schema": SCHEMA,
        "revision": "abc123",
        "platform": "Android",
        "backend": "vulkan",
        "execution_contract": EXECUTION_CONTRACT,
        "device": "test-device",
        "renderer": {"kind": "native-gpu", "readback": "diagnostic-only"},
        "nonce": "nonce",
        "frames_per_scene": 3,
        "completed": True,
        "passed": True,
        "lifecycle": checks(LIFECYCLE_CHECKS),
        "controls": checks(CONTROL_CHECKS),
        "failures": [],
        "scenes": scenes,
    }


class MobileGalleryReportTests(unittest.TestCase):
    def test_accepts_complete_common_report(self):
        supported, unavailable = validate_mobile_gallery_report(
            passing_report(), nonce="nonce")
        self.assertEqual(len(supported), 27)
        self.assertEqual(len(unavailable), 1)

    def test_rejects_platform_specific_or_incomplete_schema(self):
        for mutation in (
            lambda report: report.pop("schema"),
            lambda report: report.pop("execution_contract"),
            lambda report: report["renderer"].pop("readback"),
            lambda report: report["lifecycle"].pop(),
            lambda report: report["controls"][0].update(passed=False),
            lambda report: report["failures"].append(
                {"stage": "scene", "message": "failed"}),
            lambda report: report["scenes"].pop(),
            lambda report: report["scenes"][0].pop("timing_ms"),
            lambda report: report["scenes"][0].pop("realtime_work"),
            lambda report: report["scenes"][0].pop("thermal_state_after"),
        ):
            with self.subTest(mutation=mutation):
                report = passing_report()
                mutation(report)
                with self.assertRaises(RuntimeError):
                    validate_mobile_gallery_report(report, nonce="nonce")


if __name__ == "__main__":
    unittest.main()
