import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mobile_metal", ROOT / "scripts/Evaluate-MobileMetalQualification.py")
mobile_metal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mobile_metal)


def checks(names):
    return [{"id": name, "applicable": True, "passed": True}
            for name in names]


def available_scene(identifier, offset):
    work = {
        "dispatches": 35 + offset,
        "workgroups": 200 + offset,
        "barriers": 40 + offset,
        "submissions": 1,
        "bytes_moved": 100 + offset,
        "time_advances": 1,
        "spatial_builds": 1,
        "spatial_refits": 0,
        "spatial_publications": 1,
        "numerical_iterations": 8,
        "intra_tick_host_readbacks": 0,
        "intra_tick_host_waits": 0,
        "domain_iterations": {
            "rigid": 8, "fluid": 0, "cloth": 0,
            "soft_body": 0, "rope": 0, "smoke": 0,
        },
        "coupling_edges": 0,
        "coupling_sweeps": 0,
    }
    timing = {
        phase: {field: 1.0 for field in ("mean", "p50", "p95", "p99")}
        for phase in ("frame", "physics", "render", "present")
    }
    return {
        "id": identifier,
        "scene": identifier,
        "scene_asset": identifier + ".glb",
        "requirements": "rigidBodies",
        "workload_profile": "test",
        "available": True,
        "skipped": False,
        "passed": True,
        "steps": 3,
        "timing_ms": timing,
        "simulation_to_wall_clock_rate": 1.0,
        "realtime_work": work,
        "thermal_state_before": "nominal",
        "thermal_state_after": "nominal",
        "memory_footprint_bytes": 1024,
        "workload": {
            "target_updates_per_second": 60,
            "rigid_body_limit": 10,
            "fluid_particle_limit": 100,
            "coupled_body_limit": 10,
            "smoke_particle_limit": 100,
            "render_width": 960,
            "render_height": 720,
            "peak_particle_count": 0,
            "moving_body_count": 1,
        },
    }


def native_profile(context, schedule, work):
    phase = {
        "dispatches": work["dispatches"],
        "workgroups": work["workgroups"],
        "dependency_barriers": work["barriers"],
        "dependency_ranges": 1,
        "transfer_commands": 0,
        "readbacks": 0,
        "host_waits": 0,
        "bytes_moved": work["bytes_moved"],
        "declared_launches": work["dispatches"],
    }
    return {
        "schema": mobile_metal.PROFILE_SCHEMA,
        "context": context,
        "schedule": schedule,
        "final": True,
        "submissions": 1,
        "native_barrier_commands": 0,
        "gpu_timing_status": "supported",
        "instrumentation": {
            "timestamp_queries": 0, "forced_waits": 0, "result_bytes": 0},
        "kernels": [{"name": "kernel", "dispatches": work["dispatches"],
                     "workgroups": work["workgroups"]}],
        "phases": [phase],
        "gpu_ms": 1.0,
        "record_ms": 0.1,
        "submit_ms": 0.01,
        "wait_ms": 0.0,
        "submit_to_fence_ms": 1.1,
        **phase,
    }


class MobileMetalQualificationTests(unittest.TestCase):
    def fixture(self, assets):
        scenes = []
        profiles = []
        for index, identifier in enumerate(mobile_metal.REQUIRED_COVERAGE):
            scene = available_scene(identifier, index)
            scenes.append(scene)
            (assets / scene["scene_asset"]).write_bytes(identifier.encode())
            profiles.extend(native_profile(index + 1, schedule,
                                           scene["realtime_work"])
                            for schedule in range(1, 6))
        for index in range(28 - len(scenes)):
            scenes.append({
                "id": f"unavailable-{index}", "scene": "Unavailable",
                "available": False, "skipped": True, "passed": False,
                "steps": 3, "reason": "unsupported",
            })
        report = {
            "schema": "parallel-mater-mobile-gallery-qualification/v2",
            "revision": "abc", "platform": "iPadOS", "backend": "metal",
            "execution_contract": "fixed-tick-60hz-v1", "device": "Apple M1",
            "renderer": {"kind": "native-gpu", "readback": "diagnostic-only"},
            "nonce": "fresh", "frames_per_scene": 3, "completed": True,
            "passed": True, "failures": [],
            "lifecycle": checks({"attach", "resize", "background", "foreground",
                                 "failure_recovery", "invalid_scene_recovery"}),
            "controls": checks({"scene_selection", "pause_resume", "camera",
                                "directional", "count", "primary_action"}),
            "scenes": scenes,
        }
        return report, profiles

    def test_reconciles_bounded_scene_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            assets = pathlib.Path(temporary)
            report, profiles = self.fixture(assets)
            result = mobile_metal.evaluate(report, profiles, assets)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["scenes"]), len(mobile_metal.REQUIRED_COVERAGE))
            self.assertTrue(result["scenes"][0]["fixed_gate"]["passed"])

    def test_rejects_native_public_counter_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            assets = pathlib.Path(temporary)
            report, profiles = self.fixture(assets)
            profiles[1]["dispatches"] += 1
            with self.assertRaisesRegex(ValueError, "no bounded native context"):
                mobile_metal.evaluate(report, profiles, assets)


if __name__ == "__main__":
    unittest.main()
