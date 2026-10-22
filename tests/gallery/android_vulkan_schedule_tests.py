#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "android_vulkan_schedule", ROOT / "scripts/Test-AndroidVulkanSchedule.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def profile(schedule, submission, final, dispatches, gpu_ms, phases,
            overflow=False):
    return {
        "schema": MODULE.PROFILE_SCHEMA,
        "schedule": schedule,
        "submission": submission,
        "final": final,
        "dispatches": dispatches,
        "gpu_ms": gpu_ms,
        "record_ms": 1.0,
        "submit_to_fence_ms": gpu_ms + 0.5,
        "timestamp_overflow": overflow,
        "phases": phases,
    }


class VulkanScheduleProfileTests(unittest.TestCase):
    def test_parse_and_aggregate_split_submission(self):
        phases = [{"name": "rigid_integration", "gpu_ms": 2.0,
                   "dispatches": 3, "declared_launches": 3,
                   "occurrences": 1}]
        records = [
            profile(7, 1, False, 3, 2.0, phases),
            profile(7, 2, True, 5, 4.0, [{
                "name": "rigid_integration", "gpu_ms": 4.0,
                "dispatches": 5, "declared_launches": 5,
                "occurrences": 2}]),
            profile(8, 1, False, 9, 8.0, phases),
        ]
        log = "\n".join(
            "10-20 I ParallelMater: " + MODULE.PROFILE_PREFIX + json.dumps(item)
            for item in records)
        schedules = MODULE.aggregate_profile_schedules(MODULE.parse_profiles(log))
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0]["submissions"], 2)
        self.assertEqual(schedules[0]["dispatches"], 8)
        self.assertEqual(schedules[0]["gpu_ms"], 6.0)
        self.assertEqual(
            schedules[0]["phases"]["rigid_integration"]["occurrences"], 3)

    def test_summary_uses_zero_for_absent_phase(self):
        first = MODULE.aggregate_profile_schedules([profile(1, 1, True, 2, 2.0, [{
            "name": "fluid_integration", "gpu_ms": 2.0, "dispatches": 2,
            "declared_launches": 2, "occurrences": 1}])])[0]
        second = MODULE.aggregate_profile_schedules([
            profile(2, 1, True, 1, 1.0, [])])[0]
        summary = MODULE.summarize_schedules([first, second])
        self.assertEqual(summary["dispatches_per_tick"]["mean"], 1.5)
        self.assertEqual(summary["phases"][0]["gpu_ms"]["mean"], 1.0)

    def test_chunked_profile_is_reassembled(self):
        value = profile(9, 1, True, 4, 3.0, [])
        payload = json.dumps(value)
        midpoint = len(payload) // 2
        log = "\n".join([
            "I ParallelMater: VulkanScheduleProfileChunk 2 1/2 " + payload[:midpoint],
            "I ParallelMater: VulkanScheduleProfileChunk 2 2/2 " + payload[midpoint:],
        ])
        self.assertEqual(MODULE.parse_profiles(log), [value])

    def test_incomplete_chunked_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            MODULE.parse_profiles(
                "I ParallelMater: VulkanScheduleProfileChunk 2 1/2 {\"x\":")

    def test_overflow_is_rejected(self):
        schedule = MODULE.aggregate_profile_schedules([
            profile(1, 1, True, 1, 1.0, [], overflow=True)])[0]
        with self.assertRaisesRegex(ValueError, "overflowed"):
            MODULE.summarize_schedules([schedule])


if __name__ == "__main__":
    unittest.main()
