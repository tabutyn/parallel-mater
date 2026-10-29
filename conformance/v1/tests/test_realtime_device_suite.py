import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "conformance/v1/realtime_device_suite.py"
SPEC = importlib.util.spec_from_file_location("realtime_device_suite", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RealtimeDeviceSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = MODULE.load_json(MODULE.PROFILE_PATH)
        cls.inventory = MODULE.load_json(MODULE.INVENTORY_PATH)
        cls.obligations = MODULE.load_json(MODULE.CAPABILITY_PATH)
        cls.suite = MODULE.load_json(MODULE.SUITE_PATH)
        cls.fixtures = MODULE.load_json(MODULE.FIXTURE_PATH)
        cls.fixture_sha = hashlib.sha256(
            MODULE.FIXTURE_PATH.read_bytes()).hexdigest()
        cls.candidate = MODULE.load_json(MODULE.CANDIDATE_PATH)
        cls.candidate_sha = hashlib.sha256(
            MODULE.CANDIDATE_PATH.read_bytes()).hexdigest()

    def candidate_identity(self):
        return {
            "id": self.candidate["id"],
            "manifest_sha256": self.candidate_sha,
            "source_revision": self.candidate["source"]["revision"],
        }

    def tick(self, outcome=None):
        profile = self.contract
        iterations = profile["algebraic_iteration_budget"]
        result = {
            "unit": "per-tick",
            "status": "success",
            "result": "PM_SUCCESS",
            "dispatches": 1,
            "phase_dispatches": {
                "boundary": 0, "predict": 0, "index": 1,
                "candidates": 0, "domain": 0, "coupling": 0,
                "commit": 0,
            },
            "workgroups": 1,
            "barriers": 0,
            "queue_submissions": 1,
            "intra_tick_host_readbacks": 0,
            "bytes_moved": 0,
            "spatial_index_updates": 1,
            "spatial_index_publications": 1,
            "time_advances": 1,
            "active_iteration_budgets": [],
            "coupling_edges": 0,
            "iterations": {
                name: value for name, value in iterations.items()
                if name.endswith("sweeps") or name.endswith("cycles")
            },
        }
        if outcome is not None and outcome.get("status") == "failure":
            result.update({
                "status": "failure",
                "result": outcome["result"],
                "spatial_index_updates": 0,
                "spatial_index_publications": 0,
                "time_advances": 0,
                "state_unchanged": True,
                "pending_inputs_preserved": True,
                "hidden_retries": 0,
                "atomic_failure": {
                    "committed_state_preserved": True,
                    "clock_preserved": True,
                    "pending_inputs_preserved": True,
                },
                "retry_substeps": 0,
                "retry_spatial_updates": 0,
            })
        return result

    def scenario(self, definition):
        fixture = next(item for item in self.fixtures["scenarios"]
                       if item["id"] == definition["id"])
        overrides = {
            item["tick"]: item
            for item in fixture["tick_outcomes"]["overrides"]
        }
        case = {"kind": "scenario", **definition}
        measurements = {
            spec.assertion: MODULE.ORACLE.passing_value(spec)
            for spec in MODULE.ORACLE.case_limit_specs(case, self.contract)
        }
        count = definition["ticks"]
        tick_work = [self.tick(overrides.get(index))
                     for index in range(count)]
        executions = [{
            "validation_enabled": mode,
            "simulation_status": "PM_SUCCESS",
            "simulation_instance_count": 1,
            "completed_tick_records": count,
            "measurements_sha256": MODULE.value_sha256(measurements),
            "tick_work_sha256": MODULE.value_sha256(tick_work),
            "final_public_state_sha256": "c" * 64,
        } for mode in (False, True)]
        return {
            "id": definition["id"],
            "source": "installed-public-api",
            "driver_private_state_used": False,
            "public_api_calls": self.fixtures["execution"]["public_api_calls"],
            "runtime_variant": "metal-runtime",
            "fixture_contract": {
                "id": self.fixtures["id"], "sha256": self.fixture_sha,
            },
            "executions": executions,
            "simulation_recreated_during_ticks": False,
            "candidate": self.candidate_identity(),
            "domains": definition["domains"],
            "tags": definition["tags"],
            "simulation_status": "PM_SUCCESS",
            "steps": [{"timestep_seconds": 1.0 / 60.0, "substeps": 1}
                      for _ in range(count)],
            "measurements": measurements,
            "tick_work": tick_work,
        }

    def backend_observation(self, validation):
        observation = {
            "schema": "parallel-mater-realtime-observation/v1",
            "source": "installed-public-api",
            "driver_private_state_used": False,
            "validation_enabled": validation,
            "runtime": {"schedule": "fixed-tick", "version": 1},
            "candidate": {
                "manifest_role": "qualification-input",
                "execution_identity_claimed": False,
                "id": self.candidate["id"],
                "manifest_sha256": self.candidate_sha,
                "source_revision": self.candidate["source"]["revision"],
                "runtime_variant": "metal-runtime",
            },
            "case_id": "f8-kernel-entrypoint-execution/metal",
            "public_api_calls": self.suite["public_api"]["operation_sequence"],
            "steps": [{"timestep_seconds": 1.0 / 60.0, "substeps": 1}],
            "simulation_status": "PM_SUCCESS",
            "statistics": {
                "step_count": 1, "substeps_executed": 1,
                "execution_result": "PM_SUCCESS",
                "telemetry_schema_version": 1,
                "profile_version": 1, "budget_version": 1,
            },
            "copied_public_state": {
                "body_count": 1, "position": [0.0, 1.0, 0.0],
                "linear_velocity": [0.0, -0.1, 0.0],
            },
            "measurements": {},
            "tick_work": [self.tick()],
        }
        return observation

    def capture(self):
        capabilities = []
        for item in MODULE.ORACLE.expand_capabilities(self.obligations):
            capabilities.append({
                key: value for key, value in item.items() if key != "rule_id"
            } | {
                "execution_status": "completed",
                "analytic_status": "passed",
                "lifecycle_status": "passed",
            })
        tier = next(item for item in self.contract["performance"]["tiers"]
                    if item["id"] == "desktop-full-v1")
        distribution = {"p50": 1.0, "p95": 2.0,
                        "p99": 3.0, "maximum": 4.0}
        return {
            "schema": MODULE.CAPTURE_SCHEMA,
            "candidate": self.candidate_identity(),
            "runtime_variant": "metal-runtime",
            "device": {
                "target_id": "macos-m4-slang-metal",
                "name": "test device", "os": "test os", "gpu": "test gpu",
                "driver": "test driver", "toolchain": "test toolchain",
                "driver_binary_sha256": "a" * 64,
                "driver_manifest_sha256": "b" * 64,
            },
            "backend_observations": [
                self.backend_observation(False),
                self.backend_observation(True),
            ],
            "capabilities": capabilities,
            "public_cases": [{
                "id": identity,
                "execution_status": "completed",
                "driver_private_state_used": False,
            } for identity in self.inventory["public_case_sets"]["all"]],
            "workloads": [{
                "id": item["id"], "counts": item["counts"],
                "warmup_ticks": 600, "measured_ticks": 7200,
                "failed_ticks": 0, "simulated_to_wall_time_ratio": 1.0,
                "timings": {
                    "cpu_ms": distribution,
                    "gpu_ms": distribution,
                    "physics_ms": distribution,
                    "frame_ms": distribution,
                },
            } for item in tier["workloads"]],
            "scenarios": [
                self.scenario(item) for item in self.suite["scenario_fixtures"]
            ],
            "negative_controls": [
                {"id": identity, "passed": True}
                for identity in MODULE.REQUIRED_NEGATIVE_CONTROLS
            ],
        }

    def evaluate(self, capture):
        return MODULE.evaluate(
            capture, self.contract, self.inventory, self.obligations,
            self.suite, self.candidate, self.candidate_sha)

    def test_complete_public_device_capture_passes(self):
        report = self.evaluate(self.capture())
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["scenario_count"], 12)

    def test_missing_scenario_fails(self):
        capture = self.capture()
        capture["scenarios"].pop()
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        self.assertTrue(any(item["path"] == "$.scenarios"
                            for item in report["failures"]))

    def test_wrong_timestep_and_extra_substep_fail(self):
        capture = self.capture()
        step = capture["scenarios"][0]["steps"][0]
        step["timestep_seconds"] = 0.01
        step["substeps"] = 2
        report = self.evaluate(capture)
        paths = {item["path"] for item in report["failures"]}
        self.assertTrue(any(path.endswith(".timestep_seconds") for path in paths))
        self.assertTrue(any(path.endswith(".substeps") for path in paths))

    def test_hidden_retry_or_changed_budget_fails(self):
        capture = self.capture()
        work = capture["scenarios"][0]["tick_work"][0]
        work["queue_submissions"] = 2
        work["spatial_index_publications"] = 2
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("queue_submissions", text)
        self.assertIn("spatial_index_publications", text)

    def test_failed_tick_must_be_atomic_and_recovery_succeeds(self):
        capture = self.capture()
        scenario = next(item for item in capture["scenarios"]
                        if item["id"] == "capacity-overflow-atomic-recovery")
        failed = scenario["tick_work"][1]
        self.assertEqual(failed["result"], "PM_ERROR_CAPACITY_EXCEEDED")
        failed["hidden_retries"] = 1
        failed["spatial_index_publications"] = 1
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("hidden_retries", text)
        self.assertIn("spatial_index_publications", text)

    def test_fixture_identity_and_validation_modes_are_exact(self):
        capture = self.capture()
        scenario = capture["scenarios"][0]
        scenario["fixture_contract"]["sha256"] = "0" * 64
        scenario["executions"].pop()
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("exact shared fixture definition", text)
        self.assertIn("validation-enabled", text)

    def test_plain_and_validation_public_states_must_match(self):
        capture = self.capture()
        capture["scenarios"][0]["executions"][1][
            "final_public_state_sha256"] = "d" * 64
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        self.assertIn("public states differ", json.dumps(report["failures"]))

    def test_missing_measurement_and_failed_execution_fail(self):
        capture = self.capture()
        scenario = capture["scenarios"][0]
        scenario["measurements"].pop(next(iter(scenario["measurements"])))
        scenario["simulation_status"] = "PM_ERROR_DEVICE_LOST"
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        self.assertTrue(any("missing" in item["path"]
                            for item in report["failures"]))
        self.assertTrue(any(item["path"].endswith(".simulation_status")
                            for item in report["failures"]))

    def test_candidate_and_runtime_cannot_be_unionized(self):
        capture = self.capture()
        capture["candidate"]["id"] = "historical-candidate"
        capture["backend_observations"][0]["candidate"]["runtime_variant"] = (
            "cpu-runtime")
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("candidate identity is not exact", text)
        self.assertIn("different runtime variant", text)

    def test_private_state_and_negative_control_omission_fail(self):
        capture = self.capture()
        capture["scenarios"][0]["driver_private_state_used"] = True
        capture["negative_controls"].pop()
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("driver-private state", text)
        self.assertIn("negative-control set", text)

    def test_missing_capability_and_unsupported_public_case_fail(self):
        capture = self.capture()
        capture["capabilities"].pop()
        capture["public_cases"][0]["execution_status"] = "unsupported"
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("frozen 49 rows", text)
        self.assertIn("unsupported or failed", text)

    def test_short_or_slow_workload_capture_fails(self):
        capture = self.capture()
        workload = capture["workloads"][0]
        workload["measured_ticks"] = 149
        workload["timings"]["physics_ms"]["p95"] = 20.0
        workload["timings"]["frame_ms"]["p95"] = 20.0
        report = self.evaluate(capture)
        self.assertFalse(report["passed"])
        text = json.dumps(report["failures"])
        self.assertIn("sustained window", text)
        self.assertIn("physics p95", text)
        self.assertIn("frame p95", text)


if __name__ == "__main__":
    unittest.main()
