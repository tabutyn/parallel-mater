# SPDX-License-Identifier: MIT
"""Structural tests for the F8-00 normative design contract.

These checks prevent the specification from drifting before F8-03 turns its
limits into independent physics oracles. They do not execute or qualify a
solver.
"""

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "realtime-profile-v1.json"
)
INVENTORY_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "kernel-device-obligations-v1.json"
)
DOCUMENT_PATH = ROOT / "docs" / "roadmap" / "F8-00-REALTIME-CONTRACT.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_leaves(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from numeric_leaves(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from numeric_leaves(nested)


class F800RealtimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_json(CONTRACT_PATH)
        cls.inventory = load_json(INVENTORY_PATH)

    def test_profile_identity_and_single_step_clock_are_exact(self):
        contract = self.contract
        self.assertEqual(
            contract["schema"], "parallel-mater-realtime-profile-contract/v1"
        )
        self.assertEqual(contract["id"], "realtime-single-step-v1")
        self.assertEqual(contract["task"], "F8-00")
        self.assertIn("not-yet-implemented", contract["status"])

        profile = contract["profile"]
        self.assertEqual(profile["version"], {"major": 1, "minor": 0})
        self.assertEqual(
            profile["timestep_seconds"], {"numerator": 1, "denominator": 60}
        )
        self.assertEqual(profile["time_advances_per_successful_tick"], 1)
        self.assertEqual(profile["dynamic_spatial_updates_per_tick_maximum"], 1)
        self.assertEqual(
            profile["dynamic_spatial_publications_per_successful_tick"], 1
        )
        self.assertEqual(profile["physics_ticks_per_render_callback_maximum"], 1)
        self.assertEqual(profile["catch_up_ticks"], 0)
        self.assertFalse(profile["adaptive_time_subdivision"])
        self.assertFalse(profile["hidden_domain_substeps"])

        clock = contract["clock_and_presentation"]
        self.assertIn("interpolate", clock["high_rate_policy"])
        self.assertIn("slower", clock["overload_policy"])
        self.assertIn("0.5 simulated seconds", clock["sustained_30_hz_result"])
        self.assertIn("never repay", clock["stall_policy"])

    def test_negotiation_never_silently_changes_legacy_behavior(self):
        negotiation = self.contract["negotiation"]
        self.assertEqual(negotiation["scope"], "immutable-per-simulation")
        self.assertIn("legacy", negotiation["omitted_request"])
        self.assertEqual(negotiation["fallback"], "forbidden")
        self.assertEqual(
            negotiation["unsupported_profile"], "PM_ERROR_FEATURE_NOT_PRESENT"
        )
        self.assertEqual(
            negotiation["unsupported_major_or_minor"],
            "PM_ERROR_INCOMPATIBLE_VERSION",
        )
        self.assertIn("validation disabled", negotiation["validation_rule"])
        self.assertIn("substeps must equal 1", negotiation["step_rule"])

        legacy = self.contract["legacy_separation"]
        self.assertEqual(legacy["legacy_profile_id"], "legacy-default")
        self.assertFalse(legacy["legacy_goldens_reused_for_realtime_qualification"])
        self.assertFalse(legacy["realtime_results_relabel_legacy_results"])

    def test_input_envelope_and_iteration_work_are_bounded(self):
        envelope = self.contract["input_envelope"]
        self.assertTrue(envelope["finite_values_required"])
        self.assertLessEqual(
            envelope["interacting_dynamic_mass_ratio_maximum"], 100.0
        )
        self.assertLessEqual(
            envelope["linear_travel_per_tick_characteristic_length_maximum"],
            0.5,
        )
        self.assertLessEqual(
            envelope["total_solver_correction_per_tick_characteristic_length_maximum"],
            0.25,
        )
        self.assertLessEqual(envelope["smoke_grid_cfl_maximum"], 1.0)
        self.assertIn("global and applicable", envelope["rule"])
        stiffness = envelope["dimensionless_constraint_stiffness"]
        compliance = envelope["dimensionless_compliance_ratio"]
        self.assertEqual(stiffness["disabled"], 0.0)
        self.assertLess(stiffness["active_minimum"], stiffness["maximum"])
        self.assertAlmostEqual(
            1.0 / (1.0 + compliance["active_minimum"]),
            stiffness["maximum"],
        )
        self.assertAlmostEqual(
            1.0 / (1.0 + compliance["active_maximum"]),
            stiffness["active_minimum"],
        )

        iterations = self.contract["algebraic_iteration_budget"]
        expected = {
            "rigid_contact_and_joint_sweeps": 8,
            "liquid_density_sweeps": 6,
            "cloth_constraint_sweeps": 8,
            "soft_body_constraint_sweeps": 8,
            "rope_constraint_sweeps": 8,
            "particle_smoke_pressure_sweeps": 6,
            "grid_smoke_projection_cycles": 12,
            "coupling_sweeps_per_registered_edge": 4,
        }
        for field, value in expected.items():
            self.assertEqual(iterations[field], value, field)
        self.assertEqual(iterations["adaptive_extra_iterations"], 0)
        self.assertEqual(iterations["high_strain_recovery_iterations"], 0)
        self.assertIn("exact", iterations["policy"])

    def test_every_domain_and_advertised_mode_has_numeric_quality_limits(self):
        domains = {item["domain"]: item for item in self.contract["domain_quality"]}
        self.assertEqual(
            set(domains),
            {"RIGID_BODY", "FLUID", "CLOTH", "SOFT_BODY", "ROPE", "SMOKE"},
        )
        for domain, item in domains.items():
            self.assertTrue(item["limits"], domain)
            self.assertTrue(
                all(math.isfinite(value) and value >= 0.0
                    for value in numeric_leaves(item["limits"])),
                domain,
            )

        modes = {
            item["mode"]: item
            for item in self.contract["coupling_quality_by_mode"]
        }
        self.assertEqual(set(modes), set(self.inventory["advertised_modes"]))
        for mode, item in modes.items():
            self.assertTrue(item["limits"], mode)
            self.assertTrue(
                all(math.isfinite(value) and value >= 0.0
                    for value in numeric_leaves(item["limits"])),
                mode,
            )
        self.assertIn("no pair-specific", self.contract["coupling_application_rule"])

    def test_failures_are_atomic_and_never_trigger_hidden_recovery(self):
        failure = self.contract["failure_behavior"]
        for field in (
            "atomic_tick",
            "preserve_previous_committed_state",
            "preserve_simulation_clock",
            "preserve_pending_inputs",
        ):
            self.assertTrue(failure[field], field)
        for field in (
            "silent_clamping",
            "partial_commit",
            "retry_with_substeps",
            "retry_with_second_spatial_update",
        ):
            self.assertFalse(failure[field], field)
        self.assertIn("PM_ERROR_INVALID_STATE", failure["gpu_discovered_envelope_escape"])
        self.assertIn("PM_ERROR_CAPACITY_EXCEEDED", failure["capacity_or_candidate_overflow"])

        boundary = self.contract["boundary_semantics"]
        self.assertIn("deterministic", boundary["births"])
        self.assertIn("next boundary", boundary["phase_transfer"])
        self.assertIn("atomic", boundary["phase_transfer"])
        self.assertIn("commit once", boundary["coupling_order"])

    def test_workloads_have_exact_counts_and_leave_frame_time_for_rendering(self):
        performance = self.contract["performance"]
        frame_budget = performance["frame_budget_ms"]
        tiers = {tier["id"]: tier for tier in performance["tiers"]}
        self.assertEqual(
            set(tiers),
            {
                "desktop-full-v1",
                "ipad-gallery-v1",
                "pixel7-gallery-v1",
                "cpu-reference-v1",
            },
        )
        for tier_id, tier in tiers.items():
            self.assertAlmostEqual(
                tier["physics_p95_ms_maximum"]
                + tier["render_input_reserve_ms_minimum"],
                frame_budget,
                places=9,
                msg=tier_id,
            )
            self.assertTrue(tier["workloads"], tier_id)
            self.assertEqual(
                len({workload["id"] for workload in tier["workloads"]}),
                len(tier["workloads"]),
                tier_id,
            )
            for workload in tier["workloads"]:
                counts = workload["counts"]
                self.assertTrue(counts)
                self.assertTrue(
                    all(isinstance(value, int) and value >= 0
                        for value in counts.values())
                )
                for key, live in counts.items():
                    if not key.endswith("_live"):
                        continue
                    capacity_key = key[:-5] + "_capacity"
                    if capacity_key in counts:
                        self.assertGreaterEqual(counts[capacity_key], live)

        measurement = performance["measurement"]
        self.assertEqual(measurement["warmup_ticks"], 600)
        self.assertEqual(measurement["measured_ticks"], 7200)
        self.assertEqual(
            measurement["required_statistics"],
            ["p50", "p95", "p99", "maximum"],
        )
        self.assertGreaterEqual(
            measurement["minimum_simulated_to_wall_time_ratio"], 0.99
        )
        self.assertIn("retain all", measurement["samples"])
        self.assertIn("cannot pass", performance["count_rule"])

    def test_f8_01_device_inventory_has_one_budget_assignment_each(self):
        inventory_targets = {
            target["id"] for target in self.inventory["physical_targets"]
        }
        assignments = self.contract["performance"]["target_assignments"]
        assigned_targets = [assignment["target"] for assignment in assignments]
        self.assertEqual(len(assigned_targets), len(set(assigned_targets)))
        self.assertEqual(set(assigned_targets), inventory_targets)

        tiers = {
            tier["id"] for tier in self.contract["performance"]["tiers"]
        }
        for assignment in assignments:
            self.assertIn(assignment["tier"], tiers)
            self.assertEqual(set(assignment), {"target", "tier"})

    def test_human_contract_names_machine_contract_and_tradeoffs(self):
        text = DOCUMENT_PATH.read_text(encoding="utf-8")
        self.assertIn("realtime-profile-v1.json", text)
        self.assertIn("not a claim", text)
        self.assertIn("Intentional losses", text)
        self.assertIn("No current legacy result", text)
        for phrase in (
            "one time advance",
            "one dynamic spatial epoch",
            "slow motion",
            "no pair-specific",
            "F8-01 confirms eight physical backend tuples",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
