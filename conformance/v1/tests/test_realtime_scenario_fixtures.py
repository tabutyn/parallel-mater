import copy
import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "conformance/v1/realtime_scenario_fixtures.py"
SPEC = importlib.util.spec_from_file_location(
    "realtime_scenario_fixtures", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RealtimeScenarioFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = MODULE.load_json(MODULE.FIXTURES_PATH)
        cls.suite = MODULE.load_json(MODULE.SUITE_PATH)
        cls.profile = MODULE.load_json(MODULE.PROFILE_PATH)
        cls.inventory = MODULE.load_json(MODULE.INVENTORY_PATH)

    def validate(self, fixtures):
        return MODULE.validate_contract(
            fixtures, self.suite, self.profile, self.inventory)

    def test_shared_fixture_contract_is_complete(self):
        self.assertEqual(self.validate(self.fixtures), [])
        self.assertEqual(len(self.fixtures["scenarios"]), 12)
        self.assertEqual(
            set(self.fixtures["measurement_algorithms"]),
            {"RIGID_BODY", "FLUID", "CLOTH", "SOFT_BODY", "ROPE", "SMOKE"},
        )

    def test_every_backend_consumes_the_same_definitions(self):
        variants = self.fixtures["execution"]["runtime_variants"]
        plans = [MODULE.execution_plan(item, self.fixtures) for item in variants]
        baseline = copy.deepcopy(plans[0])
        baseline.pop("runtime_variant")
        for plan in plans[1:]:
            runtime_variant = plan.pop("runtime_variant")
            self.assertIn(runtime_variant, variants)
            self.assertEqual(plan, baseline)
        self.assertEqual(plans[0]["runtime_variant"], "cpu-runtime")

    def test_native_runner_header_comes_from_the_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "realtime_fixture_contract.hpp"
            MODULE.write_cpp_header(path, self.fixtures)
            header = path.read_text(encoding="utf-8")
        self.assertIn(self.fixtures["id"], header)
        self.assertIn(MODULE.fixture_sha256(), header)
        self.assertIn("std::array<Scenario, 12>", header)
        for scenario in self.fixtures["scenarios"]:
            self.assertIn(f'"{scenario["id"]}"', header)
        for variant in self.fixtures["execution"]["runtime_variants"]:
            self.assertIn(f'"{variant}"', header)

    def test_wrong_timestep_and_extra_substep_are_rejected(self):
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["execution"]["timestep_seconds"] = 0.01
        fixtures["execution"]["substeps"] = 2
        paths = {issue.path for issue in self.validate(fixtures)}
        self.assertIn("$.execution.timestep_seconds", paths)
        self.assertIn("$.execution.substeps", paths)

    def test_missing_public_measurement_is_rejected(self):
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["measurement_algorithms"]["ROPE"].pop()
        issues = self.validate(fixtures)
        self.assertTrue(any(
            issue.path == "$.measurement_algorithms.ROPE"
            for issue in issues))
        self.assertTrue(any(
            issue.path.endswith(".measurements") for issue in issues))

    def test_hidden_retry_or_extra_publication_is_rejected(self):
        fixtures = copy.deepcopy(self.fixtures)
        scenario = next(
            item for item in fixtures["scenarios"]
            if item["id"] == "capacity-overflow-atomic-recovery")
        failed = scenario["tick_outcomes"]["overrides"][0]
        failed["queue_submissions"] = 2
        failed["spatial_index_publications"] = 1
        paths = {issue.path for issue in self.validate(fixtures)}
        self.assertTrue(any(path.endswith(".queue_submissions") for path in paths))
        self.assertTrue(any(
            path.endswith(".spatial_index_publications") for path in paths))

    def test_changed_work_budget_and_fixture_source_are_rejected(self):
        fixtures = copy.deepcopy(self.fixtures)
        fixtures["execution"]["spatial_publications_per_successful_tick"] = 2
        fixtures["scenarios"][0]["fixture"]["definition_sha256"] = "0" * 64
        paths = {issue.path for issue in self.validate(fixtures)}
        self.assertIn(
            "$.execution.spatial_publications_per_successful_tick", paths)
        self.assertTrue(any(path.endswith(".definition_sha256") for path in paths))

    def test_failed_tick_and_recovery_have_exact_public_outcomes(self):
        scenario = next(
            item for item in self.fixtures["scenarios"]
            if item["id"] == "capacity-overflow-atomic-recovery")
        before = MODULE.tick_outcome(scenario, 0)
        failed = MODULE.tick_outcome(scenario, 1)
        recovered = MODULE.tick_outcome(scenario, 2)
        self.assertEqual(before["result"], "PM_SUCCESS")
        self.assertEqual(failed["result"], "PM_ERROR_CAPACITY_EXCEEDED")
        self.assertEqual(failed["spatial_index_publications"], 0)
        self.assertEqual(failed["time_advances"], 0)
        self.assertEqual(recovered["result"], "PM_SUCCESS")
        self.assertEqual(recovered["spatial_index_publications"], 1)


if __name__ == "__main__":
    unittest.main()
