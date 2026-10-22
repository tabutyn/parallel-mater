# SPDX-License-Identifier: MIT
"""Portable tests for the F8-03 independent realtime oracles."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "realtime_oracle.py"
SPEC = importlib.util.spec_from_file_location("realtime_oracle", MODULE_PATH)
assert SPEC and SPEC.loader
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def valid_tick(contract):
    budget = contract["algebraic_iteration_budget"]
    return {
        "unit": "per-tick",
        "status": "success",
        "dispatches": 12,
        "phase_dispatches": {"index": 3, "domain": 5, "coupling": 4},
        "workgroups": 48,
        "barriers": 5,
        "queue_submissions": 1,
        "intra_tick_host_readbacks": 0,
        "bytes_moved": 4096,
        "spatial_index_updates": 1,
        "spatial_index_publications": 1,
        "time_advances": 1,
        "active_iteration_budgets": [
            "rigid_contact_and_joint_sweeps",
            "liquid_density_sweeps",
        ],
        "coupling_edges": 2,
        "iterations": {
            "rigid_contact_and_joint_sweeps":
                budget["rigid_contact_and_joint_sweeps"],
            "liquid_density_sweeps": budget["liquid_density_sweeps"],
            "coupling_sweeps_total":
                2 * budget["coupling_sweeps_per_registered_edge"],
        },
    }


class RealtimeOracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.contract, cls.inventory, cls.obligations,
         cls.suite) = oracle.load_contracts()

    def test_oracle_contract_is_self_consistent(self):
        self.assertEqual(oracle.check_contract(), [])
        self.assertEqual(
            oracle.validate_suite(
                self.suite, self.contract, self.inventory, self.obligations
            ),
            [],
        )

    def test_every_quantitative_contract_leaf_has_one_named_assertion(self):
        specs = oracle.named_limit_specs(self.contract)
        paths = {spec.path for spec in specs}
        assertions = {spec.assertion for spec in specs}
        self.assertEqual(len(paths), len(specs))
        self.assertEqual(len(assertions), len(specs))
        self.assertGreater(len(specs), 150)

        expected_prefixes = {
            "/profile",
            "/clock_and_presentation",
            "/input_envelope",
            "/algebraic_iteration_budget",
            "/quality_measurement",
            "/domain_quality",
            "/coupling_quality_by_mode",
            "/failure_behavior",
            "/performance",
            "/legacy_separation",
        }
        self.assertEqual(
            {next(prefix for prefix in expected_prefixes if path.startswith(prefix))
             for path in paths},
            expected_prefixes,
        )

        domains = {entry["domain"] for entry in self.contract["domain_quality"]}
        for domain in domains:
            self.assertTrue(any(
                spec.path.startswith(f"/domain_quality/{domain}/limits/")
                for spec in specs
            ), domain)
        modes = {
            entry["mode"] for entry in self.contract["coupling_quality_by_mode"]
        }
        for mode in modes:
            self.assertTrue(any(
                spec.path.startswith(f"/coupling_quality_by_mode/{mode}/limits/")
                for spec in specs
            ), mode)

    def test_every_named_limit_has_a_failing_negative_control(self):
        for spec in oracle.named_limit_specs(self.contract):
            with self.subTest(assertion=spec.assertion):
                self.assertIsNone(
                    oracle.evaluate_limit(spec, oracle.passing_value(spec))
                )
                failure = oracle.evaluate_limit(
                    spec, oracle.violating_value(spec)
                )
                self.assertIsNotNone(failure)
                self.assertEqual(failure.assertion, spec.assertion)
                self.assertIsNotNone(oracle.evaluate_limit(spec, math.nan))

    def test_case_plan_is_derived_from_authoritative_sources(self):
        plan = oracle.derived_case_plan(
            self.contract, self.inventory, self.obligations
        )
        self.assertEqual(
            {case["domain"] for case in plan["domains"]},
            {entry["domain"] for entry in self.contract["domain_quality"]},
        )
        self.assertEqual(
            {case["mode"] for case in plan["capabilities"]},
            set(self.inventory["advertised_modes"]),
        )
        self.assertEqual(
            len(plan["capabilities"]),
            len({case["id"] for case in plan["capabilities"]}),
        )
        self.assertEqual(
            {(case["tier"], case["workload"]) for case in plan["workloads"]},
            {
                (tier["id"], workload["id"])
                for tier in self.contract["performance"]["tiers"]
                for workload in tier["workloads"]
            },
        )
        self.assertEqual(
            {case["id"] for case in plan["backends"]},
            set(self.suite["public_api"]["execution_cases"]),
        )
        self.assertEqual(
            {case["id"] for case in plan["scenarios"]},
            {case["id"] for case in self.suite["scenario_fixtures"]},
        )

    def test_registered_public_trajectory_cases_close_inventory_gaps(self):
        registered = {
            case["id"] for case in self.suite["public_trajectory_cases"]
        }
        referenced = {
            case_id
            for group in self.inventory["kernel_groups"]
            for case_id in group.get("f8_public_cases", [])
        } | {
            case_id
            for path in self.inventory["auxiliary_paths"]
            for case_id in path.get("f8_public_cases", [])
        }
        self.assertEqual(registered, referenced)
        self.assertEqual(
            registered,
            {
                "f8-fluid-pair-index",
                "f8-fluid-pair-miscible",
                "f8-fluid-pair-immiscible",
                "f8-paint-contact-stamping",
            },
        )

    def test_adversarial_scenarios_cover_all_domains_and_mixed_families(self):
        scenarios = self.suite["scenario_fixtures"]
        domains = {domain for case in scenarios for domain in case["domains"]}
        self.assertEqual(
            domains,
            {entry["domain"] for entry in self.contract["domain_quality"]},
        )
        self.assertTrue(any(len(set(case["domains"])) >= 3 for case in scenarios))
        self.assertTrue(any(len(set(case["domains"])) == 6 for case in scenarios))
        self.assertTrue(any(case["ticks"] == 3600 for case in scenarios))

    def test_public_case_contract_contains_no_private_entrypoints(self):
        operations = self.suite["public_api"]["operation_sequence"]
        header = (oracle.ROOT / "include/parallel_mater/parallel_mater.h").read_text(
            encoding="utf-8"
        )
        for operation in operations:
            self.assertTrue(operation.startswith("pm"), operation)
            self.assertIn(operation, header, operation)
        self.assertIn("installed public API", self.suite["public_api"]["rule"])

    def test_public_observation_is_case_specific_and_rejects_legacy_fallback(self):
        plan = oracle.derived_case_plan(
            self.contract, self.inventory, self.obligations
        )
        case = next(item for item in plan["domains"]
                    if item["domain"] == "RIGID_BODY")
        specs = oracle.case_limit_specs(case, self.contract)
        observation = {
            "schema": "parallel-mater-realtime-observation/v1",
            "source": "installed-public-api",
            "driver_private_state_used": False,
            "profile": {
                "name": "realtime-single-step",
                "version": {"major": 1, "minor": 0},
            },
            "case_id": case["id"],
            "public_api_calls": [
                "pmEnumerateExecutionProfiles",
                "pmCreateDevice",
                "pmCreateSimulation",
                "pmCmdStepSimulation",
                "pmGetSimulationStatus",
            ],
            "steps": [{"timestep_seconds": 1.0 / 60.0, "substeps": 1}],
            "measurements": {
                spec.assertion: oracle.passing_value(spec) for spec in specs
            },
            "tick_work": [valid_tick(self.contract)],
        }
        self.assertEqual(
            oracle.validate_public_observation(
                observation, self.contract, self.inventory,
                self.obligations, self.suite,
            ),
            [],
        )

        controls = []
        legacy = copy.deepcopy(observation)
        legacy["profile"]["name"] = "legacy-default"
        controls.append(legacy)
        subdivided = copy.deepcopy(observation)
        subdivided["steps"][0]["substeps"] = 4
        controls.append(subdivided)
        private = copy.deepcopy(observation)
        private["driver_private_state_used"] = True
        controls.append(private)
        missing = copy.deepcopy(observation)
        missing["measurements"].pop(next(iter(missing["measurements"])))
        controls.append(missing)
        for control in controls:
            self.assertTrue(oracle.validate_public_observation(
                control, self.contract, self.inventory,
                self.obligations, self.suite,
            ))

    def test_frozen_workload_counts_are_exact(self):
        workload = oracle.derived_case_plan(
            self.contract, self.inventory, self.obligations
        )["workloads"][0]
        self.assertEqual(
            oracle.evaluate_workload_counts(workload["counts"], workload["counts"]),
            [],
        )
        missing = dict(workload["counts"])
        missing.pop(next(iter(missing)))
        self.assertTrue(oracle.evaluate_workload_counts(missing, workload["counts"]))
        reduced = dict(workload["counts"])
        first = next(iter(reduced))
        reduced[first] -= 1
        self.assertTrue(oracle.evaluate_workload_counts(reduced, workload["counts"]))
        expanded = dict(workload["counts"], invented_count=1)
        self.assertTrue(oracle.evaluate_workload_counts(expanded, workload["counts"]))


class NumericalOracleTests(unittest.TestCase):
    def test_raw_sample_state_and_centroid_metrics_are_independent(self):
        statistics = oracle.sample_statistics([0.0, 0.03, 0.04])
        self.assertAlmostEqual(statistics["rms"], 0.05 / math.sqrt(3.0))
        self.assertEqual(statistics["p95"], 0.04)
        self.assertEqual(statistics["maximum"], 0.04)
        self.assertAlmostEqual(
            oracle.normalized_state_error(
                (0.0, 0.0, 0.0), (0.03, 0.04, 0.0), 0.5
            ),
            0.1,
        )
        self.assertEqual(
            oracle.centroid([(0.0, 2.0, 4.0), (2.0, 4.0, 6.0)]),
            (1.0, 3.0, 5.0),
        )
        with self.assertRaises(ValueError):
            oracle.sample_statistics([math.nan])
        with self.assertRaises(ValueError):
            oracle.normalized_state_error((0.0,), (0.0, 1.0), 1.0)
        with self.assertRaises(ValueError):
            oracle.centroid([])

    def test_compliance_conversion_uses_generalized_inverse_mass(self):
        timestep = 1.0 / 60.0
        inverse_mass = 0.5
        for expected_ratio in (0.005025125628140708, 1.0, 19.0):
            compliance = expected_ratio * timestep * timestep * inverse_mass
            ratio, response = oracle.compliance_response(
                compliance, timestep, inverse_mass
            )
            self.assertAlmostEqual(ratio, expected_ratio)
            self.assertAlmostEqual(response, 1.0 / (1.0 + expected_ratio))
        with self.assertRaises(ValueError):
            oracle.compliance_response(-1.0, timestep, inverse_mass)

    def test_rms_and_reciprocal_exchange_use_contract_definitions(self):
        self.assertAlmostEqual(oracle.rms([3.0, 4.0]), math.sqrt(12.5))
        self.assertAlmostEqual(
            oracle.relative_residual((0.01, 0.0, 0.0), (1.0, 0.0, 0.0)),
            0.01,
        )
        self.assertAlmostEqual(
            oracle.relative_residual((5.0e-7, 0.0, 0.0), (0.0, 0.0, 0.0)),
            0.5,
        )
        with self.assertRaises(ValueError):
            oracle.rms([])
        with self.assertRaises(ValueError):
            oracle.rms([math.inf])

    def test_swept_candidate_oracle_allows_false_positives_not_misses(self):
        leaves = [
            oracle.SpatialLeaf(1, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1),
            oracle.SpatialLeaf(2, (0.0, -1.0, 0.0), (0.0, 1.0, 0.0), 0.1),
            oracle.SpatialLeaf(3, (5.0, 0.0, 0.0), (5.0, 0.0, 0.0), 0.1),
        ]
        expected = oracle.brute_force_candidate_pairs(leaves)
        self.assertEqual(expected, {(1, 2)})
        self.assertEqual(
            oracle.validate_candidate_pairs(leaves, expected | {(1, 3)}), []
        )
        failures = oracle.validate_candidate_pairs(leaves, {(1, 3)})
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].actual, [(1, 2)])
        self.assertEqual(oracle.validate_candidate_pairs([], []), [])

    def test_candidate_oracle_handles_correction_bounds_and_random_scenes(self):
        correction_pair = [
            oracle.SpatialLeaf(10, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                               0.1, 0.15),
            oracle.SpatialLeaf(11, (0.45, 0.0, 0.0), (0.45, 0.0, 0.0),
                               0.1, 0.15),
        ]
        self.assertEqual(oracle.brute_force_candidate_pairs(correction_pair),
                         {(10, 11)})

        generator = random.Random(0xF803)
        leaves = []
        for stable_id in range(64):
            start = tuple(generator.uniform(-2.0, 2.0) for _ in range(3))
            end = tuple(value + generator.uniform(-0.5, 0.5) for value in start)
            leaves.append(oracle.SpatialLeaf(
                stable_id, start, end, generator.uniform(0.02, 0.2), 0.05
            ))
        expected = oracle.brute_force_candidate_pairs(leaves)
        self.assertEqual(oracle.validate_candidate_pairs(leaves, expected), [])
        if expected:
            missing = set(expected)
            missing.pop()
            self.assertTrue(oracle.validate_candidate_pairs(leaves, missing))


class BoundaryAndWorkOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = oracle.load_json(oracle.PROFILE_PATH)

    def test_boundary_admission_removal_replacement_and_transfer_are_deterministic(self):
        state = oracle.BoundaryState({"FLUID": {5: 0}, "SMOKE": {}})
        result = state.submit(
            [
                {"operation": "birth", "family": "FLUID", "stable_id": 9},
                {"operation": "birth", "family": "FLUID", "stable_id": 2},
            ],
            {"FLUID": 2, "SMOKE": 1},
        )
        self.assertTrue(result["success"])
        self.assertEqual(state.populations["FLUID"], {5: 0, 2: 0})
        self.assertEqual(result["rejected"][0]["stable_id"], 9)

        state.submit(
            [
                {"operation": "remove", "family": "FLUID", "stable_id": 5},
                {"operation": "replace", "family": "FLUID", "stable_id": 2},
            ],
            {"FLUID": 2, "SMOKE": 1},
        )
        self.assertEqual(state.populations["FLUID"], {2: 1})

        rejected = state.submit(
            [{
                "operation": "phase-transfer", "family": "FLUID",
                "destination_family": "SMOKE", "stable_id": 2,
            }],
            {"FLUID": 2, "SMOKE": 0},
        )
        self.assertEqual(rejected["rejected"][0]["reason"],
                         "destination-capacity")
        self.assertIn(2, state.populations["FLUID"])
        accepted = state.submit(
            [{
                "operation": "phase-transfer", "family": "FLUID",
                "destination_family": "SMOKE", "stable_id": 2,
            }],
            {"FLUID": 2, "SMOKE": 1},
        )
        self.assertEqual(accepted["rejected"], [])
        self.assertNotIn(2, state.populations["FLUID"])
        self.assertEqual(state.populations["SMOKE"][2], 2)

    def test_overflow_failure_is_atomic_and_caller_recovery_is_later(self):
        state = oracle.BoundaryState({"FLUID": {1: 0}})
        mutation = {"operation": "birth", "family": "FLUID", "stable_id": 2}
        before = state.snapshot()
        failed = state.submit(
            [mutation], {"FLUID": 2},
            discovered_error="PM_ERROR_CAPACITY_EXCEEDED",
        )
        self.assertFalse(failed["success"])
        self.assertEqual(failed["before"], failed["after"])
        self.assertEqual(state.snapshot(), before)
        self.assertTrue(failed["pending_inputs_preserved"])
        self.assertEqual(failed["spatial_publications"], 0)
        self.assertEqual(failed["time_advances"], 0)

        recovered = state.submit(state.pending_inputs, {"FLUID": 2})
        self.assertTrue(recovered["success"])
        self.assertEqual(state.clock_ticks, 1)
        self.assertIn(2, state.populations["FLUID"])

    def valid_tick(self):
        return valid_tick(self.contract)

    def test_work_is_counted_per_tick_with_exact_iterations(self):
        self.assertEqual(
            oracle.evaluate_tick_work(self.valid_tick(), self.contract), []
        )
        controls = {
            "second-index-update": ("spatial_index_updates", 2),
            "readback": ("intra_tick_host_readbacks", 1),
            "second-time-advance": ("time_advances", 2),
        }
        for name, (field, value) in controls.items():
            with self.subTest(control=name):
                tick = self.valid_tick()
                tick[field] = value
                self.assertTrue(oracle.evaluate_tick_work(tick, self.contract))

        tick = self.valid_tick()
        tick["iterations"]["liquid_density_sweeps"] += 1
        self.assertTrue(oracle.evaluate_tick_work(tick, self.contract))
        tick = self.valid_tick()
        tick["dispatches"] -= 1
        self.assertTrue(oracle.evaluate_tick_work(tick, self.contract))

    def test_failed_tick_requires_atomic_state_and_no_hidden_retry(self):
        tick = self.valid_tick()
        tick.update({
            "status": "PM_ERROR_CAPACITY_EXCEEDED",
            "spatial_index_publications": 0,
            "time_advances": 0,
            "atomic_failure": {
                "committed_state_preserved": True,
                "clock_preserved": True,
                "pending_inputs_preserved": True,
            },
            "retry_substeps": 0,
            "retry_spatial_updates": 0,
        })
        self.assertEqual(oracle.evaluate_tick_work(tick, self.contract), [])
        tick["retry_substeps"] = 1
        self.assertTrue(oracle.evaluate_tick_work(tick, self.contract))
        tick["retry_substeps"] = 0
        tick["atomic_failure"]["clock_preserved"] = False
        self.assertTrue(oracle.evaluate_tick_work(tick, self.contract))


class TimingOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = oracle.load_json(oracle.PROFILE_PATH)
        cls.suite = oracle.load_json(oracle.SUITE_PATH)
        cls.tier = next(
            tier for tier in cls.contract["performance"]["tiers"]
            if tier["id"] == "desktop-full-v1"
        )

    def collect(self, *, failed_index=None):
        measurement = self.contract["performance"]["measurement"]
        clock = oracle.MockClock()
        collector = oracle.TimingCollector(clock, measurement["warmup_ticks"])
        total = measurement["warmup_ticks"] + measurement["measured_ticks"]
        for index in range(total):
            collector.begin_tick()
            # A retained 250 ms stall verifies that max is not clamped away.
            clock.advance_ms(250.0 if index == total - 1 else 8.0)
            collector.complete_tick(success=index != failed_index)
        return collector

    def test_mock_clock_excludes_warmup_retains_stalls_and_computes_statistics(self):
        collector = self.collect()
        summary = oracle.timing_summary(
            collector, wall_seconds=120.0,
            gpu_timing_available=False,
        )
        self.assertEqual(summary["measured_ticks"], 7200)
        self.assertEqual(summary["maximum_ms"], 250.0)
        self.assertEqual(summary["p50_ms"], 8.0)
        self.assertEqual(summary["p95_ms"], 8.0)
        self.assertEqual(summary["p99_ms"], 8.0)
        self.assertEqual(summary["gpu_timing"], "unavailable")
        self.assertEqual(
            oracle.evaluate_timing_summary(summary, self.tier, self.contract), []
        )

    def test_timing_negative_controls_fail_without_relabeling_cpu_time(self):
        summary = oracle.timing_summary(
            self.collect(), wall_seconds=120.0,
            gpu_timing_available=False,
        )
        controls = {
            "p95": ("p95_ms", 10.001),
            "sample-loss": ("retained_samples", 7199),
            "throughput": ("simulated_to_wall_time_ratio", 0.98),
            "cpu-as-gpu": ("gpu_phase_times", [8.0]),
        }
        for name, (field, value) in controls.items():
            with self.subTest(control=name):
                changed = dict(summary)
                changed[field] = value
                self.assertTrue(
                    oracle.evaluate_timing_summary(
                        changed, self.tier, self.contract
                    )
                )

        failed = oracle.timing_summary(
            self.collect(failed_index=700), wall_seconds=120.0,
            gpu_timing_available=False,
        )
        self.assertTrue(
            oracle.evaluate_timing_summary(failed, self.tier, self.contract)
        )

    def test_legacy_baseline_failure_does_not_mutate_realtime_thresholds(self):
        before = hashlib.sha256(json.dumps(
            self.contract, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        spec = next(
            item for item in oracle.named_limit_specs(self.contract)
            if item.path.endswith("/maximum_contact_penetration_m")
            and "/domain_quality/RIGID_BODY/" in item.path
        )
        failure = oracle.evaluate_limit(spec, oracle.violating_value(spec))
        baseline_record = {
            "profile": "legacy-default",
            "timestep_seconds": 1.0 / 60.0,
            "substeps": 1,
            "status": "failed",
            "failed_assertions": [failure.assertion],
        }
        self.assertEqual(baseline_record["status"], "failed")
        self.assertFalse(
            self.suite["legacy_separation"]["accept_legacy_as_realtime"]
        )
        after = hashlib.sha256(json.dumps(
            self.contract, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
