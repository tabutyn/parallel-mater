#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "metal_schedule", ROOT / "scripts/Evaluate-MetalSchedule.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def profile(dispatches=35, workgroups=4354, declared=35):
    return {
        "schema": MODULE.PROFILE_SCHEMA,
        "context": 1,
        "schedule": 4,
        "submissions": 1,
        "final": True,
        "dispatches": dispatches,
        "workgroups": workgroups,
        "dependency_barriers": 36,
        "dependency_ranges": 62,
        "native_barrier_commands": 0,
        "transfer_commands": 7,
        "readbacks": 0,
        "host_waits": 0,
        "bytes_moved": 464,
        "declared_launches": declared,
        "kernels": [{
            "name": "prepare_avbd_kernel",
            "dispatches": 1,
            "workgroups": 1,
        }, {
            "name": "solve_avbd_kernel",
            "dispatches": 1,
            "workgroups": 1,
        }, {
            "name": "other_kernel",
            "dispatches": dispatches - 2,
            "workgroups": workgroups - 2,
        }],
        "gpu_ms": 3.2,
        "gpu_timing_status": "supported",
        "record_ms": 0.4,
        "submit_ms": 0.01,
        "wait_ms": 0.1,
        "submit_to_fence_ms": 3.4,
        "instrumentation": {
            "timestamp_queries": 0,
            "forced_waits": 0,
            "result_bytes": 0,
        },
        "phases": [{
            "name": "tick",
            "dispatches": dispatches,
            "workgroups": workgroups,
            "dependency_barriers": 36,
            "dependency_ranges": 62,
            "native_barrier_commands": 0,
            "transfer_commands": 7,
            "readbacks": 0,
            "host_waits": 0,
            "bytes_moved": 464,
            "declared_launches": declared,
        }],
    }


class MetalScheduleProfileTests(unittest.TestCase):
    def test_load_and_evaluate_rigid_profile(self):
        value = profile()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profile.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            loaded = MODULE.load_profiles([path])
        result = MODULE.evaluate(loaded, "rigid-386")
        self.assertTrue(result["passed"])
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(
            result["reconciliation"][0]["native_minus_declared"], 0)

    def test_mixed_reconciliation_preserves_native_delta(self):
        value = profile(dispatches=504, workgroups=520, declared=457)
        value["dependency_barriers"] = 600
        value["phases"][0]["dependency_barriers"] = 600
        result = MODULE.evaluate([value], "mixed")
        self.assertEqual(
            result["reconciliation"][0]["native_minus_declared"], 47)

    def test_independent_dispatch_ceiling_rejects_regression(self):
        with self.assertRaisesRegex(ValueError, "ceiling exceeded"):
            MODULE.evaluate([profile(dispatches=41, declared=40)], "rigid-386")

    def test_readback_and_host_wait_are_rejected(self):
        for field in ("readbacks", "host_waits"):
            with self.subTest(field=field):
                value = profile()
                value[field] = 1
                value["phases"][0][field] = 1
                with self.assertRaisesRegex(ValueError, "readback or wait"):
                    MODULE.evaluate([value], "rigid-386")

    def test_phase_sum_must_match_native_total(self):
        value = profile()
        value["phases"][0]["dispatches"] -= 1
        with self.assertRaisesRegex(ValueError, "phase sum"):
            MODULE.evaluate([value], "rigid-386")

    def test_instrumentation_cannot_force_synchronization(self):
        value = profile()
        value["instrumentation"]["forced_waits"] = 1
        with self.assertRaisesRegex(ValueError, "changed the measured"):
            MODULE.evaluate([value], "rigid-386")

    def test_legacy_staged_solver_is_rejected(self):
        value = profile()
        value["kernels"][2]["name"] = "solve_avbd_stage_kernel"
        with self.assertRaisesRegex(ValueError, "forbidden legacy"):
            MODULE.evaluate([value], "rigid-386")

    def test_cooperative_solve_gate_is_exact(self):
        value = profile()
        value["kernels"][1]["workgroups"] = 2
        value["kernels"][2]["workgroups"] -= 1
        with self.assertRaisesRegex(ValueError, "fixed gate"):
            MODULE.evaluate([value], "rigid-386")

    def test_frame_alignment_reports_frame_and_simulated_second_rates(self):
        frames = [{
            "frame": 1,
            "scene": "Rigid",
            "mode": "interactive",
            "physics_steps": 0,
            "total_ms": 5.0,
            "physics_ms": 0.0,
            "render_ms": 2.0,
            "present_ms": 1.0,
            "other_ms": 2.0,
        }, {
            "frame": 2,
            "scene": "Rigid",
            "mode": "interactive",
            "physics_steps": 2,
            "total_ms": 20.0,
            "physics_ms": 12.0,
            "render_ms": 4.0,
            "present_ms": 2.0,
            "other_ms": 2.0,
        }]
        result = MODULE.evaluate(
            [profile(), profile()], "rigid-386", frames=frames)
        self.assertEqual(result["rendered_frame_count"], 2)
        self.assertEqual(
            result["counters_per_rendered_frame"]["dispatches"]["maximum"], 70)
        self.assertEqual(
            result["counters_per_simulated_second"]["dispatches"], 2100)
        self.assertAlmostEqual(result["simulation_to_wall_clock_rate"], 4 / 3)

    def test_frame_alignment_rejects_missing_native_tick(self):
        frames = [{"physics_steps": 2}]
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.align_frames([profile()], frames, 0)

    def test_priming_sample_is_validated_but_excluded_from_frames(self):
        frames = [{
            "physics_steps": 1,
            "total_ms": 16.0,
            "physics_ms": 10.0,
            "render_ms": 3.0,
            "present_ms": 2.0,
            "other_ms": 1.0,
        }]
        result = MODULE.evaluate(
            [profile(), profile()], "rigid-386", frames=frames,
            priming_samples=1)
        self.assertEqual(result["priming_samples"], 1)
        self.assertEqual(result["sample_count"], 1)
        self.assertIn("cold_priming_timing_ms", result)


if __name__ == "__main__":
    unittest.main()
