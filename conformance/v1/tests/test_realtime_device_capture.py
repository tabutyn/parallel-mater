import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "conformance/v1/realtime_device_capture.py"
SPEC = importlib.util.spec_from_file_location("realtime_device_capture", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RealtimeDeviceCaptureTests(unittest.TestCase):
    def test_scenario_definitions_retain_authored_measurement_fixture(self):
        suite = {
            "scenario_fixtures": [{
                "id": "example", "domains": ["FLUID"], "ticks": 2,
                "tags": ["containment"],
            }],
        }
        fixtures = {
            "scenarios": [{
                "id": "example", "domains": ["FLUID"], "ticks": 2,
                "tags": ["containment"],
                "fixture": {"closed_container_bounds": {
                    "minimum": [-1, -1, -1], "maximum": [1, 1, 1],
                }},
                "measurement_sets": ["FLUID"],
                "tick_outcomes": {"default": "success", "overrides": []},
                "failure_diagnostics": ["public result"],
            }],
        }
        definitions = MODULE._scenario_definitions(suite, fixtures)
        self.assertEqual(
            definitions["example"]["fixture"]["closed_container_bounds"]
            ["maximum"], [1, 1, 1])

    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(MODULE.PROFILE_PATH.read_text())

    @staticmethod
    def work(status="success", result="PM_SUCCESS"):
        return {
            "unit": "per-tick", "status": status, "result": result,
            "dispatches": 0, "workgroups": 0, "barriers": 0,
            "queue_submissions": 1, "intra_tick_host_readbacks": 0,
            "bytes_moved": 0, "spatial_index_updates": 0,
            "spatial_index_publications": status == "success",
            "time_advances": status == "success", "phase_dispatches": {},
            "active_iteration_budgets": [],
            "iterations": {"coupling_sweeps_total": 0},
            "coupling_edges": 0,
        }

    @staticmethod
    def checkpoint(frame, resources, work=None):
        value = {
            "frame": frame, "resources": resources,
            "contacts": {"rigid_contact_count": 0, "rigid_contacts": [],
                         "fluid_contact_count": 0, "fluid_contacts": []},
            "invariants": {"rigid_quality": {
                "maximum_joint_angle_error_rad": 0.0,
                "maximum_joint_position_error_m": 0.0,
                "mechanical_energy_j": 1.0,
            }},
        }
        if work is not None:
            value["realtime_work"] = work
        return value

    def test_all_37_measurements_come_from_public_checkpoint_fields(self):
        zero = {"count": 1, "maximum": 0.0, "rms": 0.0,
                "squared_sum": 0.0}
        resources = [
            {"id": "rigid", "type": "rigid_body", "position": [0, 1, 0],
             "orientation": [0, 0, 0, 1]},
            {"id": "fluid", "type": "fluid", "count": 1,
             "minimum_clearance": 1.0, "stable_ids": [1],
             "aggregate": {"center_of_mass": [0, 1, 0],
                           "kinetic_energy": 0.0},
             "quality": {"density_relative_error_maximum": 0.0,
                         "density_relative_error_p95": 0.0}},
            {"id": "cloth", "type": "cloth", "count": 1,
             "minimum_clearance": 1.0,
             "quality": {"bend_angle": zero, "stretch": zero,
                         "closed_surface_volume_relative_error": 0.0,
                         "pinned_vertex_drift_m": 0.0}},
            {"id": "soft", "type": "soft_body", "count": 1,
             "minimum_clearance": 1.0, "topology": {"bond_count": 1},
             "quality": {"bond_stretch": zero,
                         "shape_matching_position_error": zero,
                         "inverted_volume_elements": 0,
                         "volume_relative_error": 0.0}},
            {"id": "rope", "type": "rope", "count": 2,
             "minimum_clearance": 1.0,
             "quality": {"bend_angle": zero, "segment_stretch": zero,
                         "attachment_position_error_m": 0.0}},
            {"id": "smoke", "type": "smoke", "count": 1,
             "minimum_clearance": 1.0,
             "quality": {"analytic_transport_centroid_error_grid_cells": 0.0,
                         "number_density_sum": 1.0,
                         "grid_divergence_dt": zero,
                         "maximum_speed_overshoot_m_s": 0.0,
                         "particle_density_relative_error_p95": 0.0}},
        ]
        definition = {
            "id": "all", "ticks": 1,
            "domains": ["RIGID_BODY", "FLUID", "CLOTH", "SOFT_BODY",
                        "ROPE", "SMOKE"],
            "tags": [],
            "fixture": {"characteristic_length_m": 1.0,
                        "support_radius_m": 1.0},
        }
        observed = MODULE.measurements(
            [self.checkpoint(1, resources)], definition, self.contract)
        self.assertEqual(len(observed), 37)
        self.assertTrue(all(value == 0 for value in observed.values()))

    def test_failed_tick_atomic_fields_are_verified_from_adjacent_states(self):
        first = {"id": "fluid", "type": "fluid", "count": 1,
                 "stable_ids": [0], "samples": [{"position": [0, 0, 0]}]}
        recovered = {"id": "fluid", "type": "fluid", "count": 1,
                     "stable_ids": [0],
                     "samples": [{"position": [1, 0, 0]}]}
        checkpoints = [
            self.checkpoint(0, [first]),
            self.checkpoint(1, [first], self.work()),
            self.checkpoint(2, [first], self.work(
                "failure", "PM_ERROR_CAPACITY_EXCEEDED")),
            self.checkpoint(3, [recovered], self.work()),
        ]
        observed = MODULE.tick_work(
            checkpoints, {"id": "capacity", "ticks": 3})
        failed = observed[1]
        self.assertTrue(failed["state_unchanged"])
        self.assertTrue(failed["pending_inputs_preserved"])
        self.assertEqual(failed["hidden_retries"], 0)
        self.assertTrue(failed["atomic_failure"]["clock_preserved"])

    def test_empty_smoke_checkpoint_is_not_a_transport_sample(self):
        empty = {"id": "smoke", "type": "smoke", "count": 0,
                 "quality": {
                     "analytic_transport_centroid_error_grid_cells": 999.0}}
        active = {"id": "smoke", "type": "smoke", "count": 1,
                  "quality": {
                      "analytic_transport_centroid_error_grid_cells": 0.25,
                      "number_density_sum": 1.0,
                      "grid_divergence_dt": {"count": 1, "maximum": 0.0,
                                             "squared_sum": 0.0},
                      "maximum_speed_overshoot_m_s": 0.0,
                      "particle_density_relative_error_p95": 0.0}}
        definition = {"id": "smoke", "ticks": 2, "domains": ["SMOKE"],
                      "tags": [], "fixture": {}}
        observed = MODULE.measurements([
            self.checkpoint(1, [empty]), self.checkpoint(2, [active]),
        ], definition, self.contract)
        key = next(name for name in observed if "analytic-transport" in name)
        self.assertEqual(observed[key], 0.25)

    def test_box_penetration_uses_published_pose_not_ccd_recovery(self):
        first = {"position": [3.0, 2.8, 0.0],
                 "orientation": [0.0, 0.0, 0.84, 0.54]}
        second = {"position": [1.0, 1.5, 0.0],
                  "orientation": [0.0, 0.0, 0.0, 1.0]}
        self.assertEqual(MODULE._box_penetration(
            first, [0.25, 0.25, 0.25], second, [0.2, 0.2, 0.2]), 0.0)

        first = {"position": [0.0, 0.24, 0.0],
                 "orientation": [0.0, 0.0, 0.0, 1.0]}
        second = {"position": [0.0, -0.05, 0.0],
                  "orientation": [0.0, 0.0, 0.0, 1.0]}
        self.assertAlmostEqual(MODULE._box_penetration(
            first, [0.25, 0.25, 0.25], second, [3.0, 0.05, 3.0]), 0.01)

    def test_forced_event_starts_new_energy_interval_at_next_checkpoint(self):
        checkpoints = [self.checkpoint(frame, []) for frame in range(1, 5)]
        for checkpoint, energy in zip(checkpoints, (1.0, 100.0, 90.0, 80.0)):
            checkpoint["invariants"]["rigid_quality"][
                "mechanical_energy_j"] = energy
        definition = {"fixture": {"events": [{"tick": 1}]}}
        self.assertEqual(MODULE._energy_growth(
            checkpoints, definition, "rigid"), 0.0)

    def test_scenario_observation_retains_mode_mismatch_for_reporting(self):
        definition = {
            "id": "mismatch", "ticks": 1, "domains": [], "tags": [],
            "fixture": {},
        }
        fixture = {
            "id": "fixture",
            "execution": {"public_api_calls": ["pmSimulationStep"]},
        }
        plain_resource = {
            "id": "rigid", "type": "rigid_body", "position": [0, 0, 0],
        }
        validation_resource = {
            "id": "rigid", "type": "rigid_body", "position": [1, 0, 0],
        }
        plain = {"checkpoints": [
            self.checkpoint(0, [plain_resource]),
            self.checkpoint(1, [plain_resource], self.work()),
        ]}
        validation = {"checkpoints": [
            self.checkpoint(0, [validation_resource]),
            self.checkpoint(1, [validation_resource], self.work()),
        ]}

        observed = MODULE._scenario_observation(
            definition, fixture, self.contract, {"id": "candidate"},
            "cuda-runtime", plain, validation)

        self.assertNotEqual(
            observed["executions"][0]["final_public_state_sha256"],
            observed["executions"][1]["final_public_state_sha256"])
        self.assertEqual(
            MODULE.value_sha256(observed["executions"][1]["final_public_state"]),
            observed["executions"][1]["final_public_state_sha256"])


if __name__ == "__main__":
    unittest.main()
