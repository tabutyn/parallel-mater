import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "docs/roadmap/evidence/F5-23-qualification-gap-matrix.json"
TASKS_PATH = ROOT / "docs/roadmap/tasks"
KEY_FIELDS = (
    "first_family",
    "second_family",
    "scope",
    "mode",
    "direction",
    "material_profile",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class QualificationGapMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load(MATRIX_PATH)
        cls.tasks = {
            path.stem: load(path)
            for path in TASKS_PATH.glob("*.json")
        }

    def test_matrix_is_complete_and_keyed_by_full_capability_identity(self):
        matrix = self.matrix
        self.assertEqual(
            matrix["schema"],
            "parallel-mater-qualification-gap-matrix/v1",
        )
        self.assertEqual(matrix["task"], "F5-23")
        self.assertEqual(matrix["capability_key_fields"], list(KEY_FIELDS))
        self.assertTrue(
            matrix["decision_basis"]["advertisement_correction"]
            ["all_unique_tuples_have_a_passing_execution_record"]
        )
        self.assertEqual(
            matrix["decision_basis"]["advertisement_correction"]
            ["current_failed_execution_occurrences"],
            11,
        )
        self.assertEqual(len(matrix["gaps"]), 44)
        self.assertEqual(
            sum(len(item["missing_scopes"]) for item in matrix["gaps"].values()),
            76,
        )

        for identity, item in matrix["gaps"].items():
            self.assertEqual(
                identity,
                "/".join(item["capability"][field] for field in KEY_FIELDS),
            )
            self.assertEqual(set(item["capability"]), set(KEY_FIELDS))
            self.assertIn("analytic_invariants", item["missing_scopes"])
            self.assertTrue(item["observed_drivers"])
            self.assertTrue(item["observed_reports"])
            self.assertTrue(item["physical_behavior"])

    def test_every_gap_maps_to_a_bounded_leaf_and_the_f5_gate(self):
        referenced = set()
        assignments = {}
        for identity, item in self.matrix["gaps"].items():
            classification = item["classification"]
            self.assertEqual(classification["kind"], "new_leaf_task")
            task_id = classification["task_id"]
            referenced.add(task_id)
            assignments.setdefault(task_id, []).append(identity)

        self.assertEqual(len(referenced), 31)
        gate_dependencies = set(self.tasks["F5-99"]["dependencies"])
        for task_id in referenced:
            task = self.tasks[task_id]
            self.assertIn(task["status"], {"open", "leased", "done"})
            self.assertEqual(task["dependencies"], ["F5-23"])
            self.assertIn(task_id, gate_dependencies)
            self.assertEqual(len(task["acceptance"]), 3)
            for identity in assignments[task_id]:
                self.assertIn(identity, task["scope"])

    def test_source_pins_and_summary_are_reviewable(self):
        source = self.matrix["source"]
        self.assertRegex(source["repository_revision"], r"^[0-9a-f]{40}$")
        for name in ("evidence_index", "capability_obligations"):
            self.assertTrue((ROOT / source[name]["path"]).is_file())
            self.assertRegex(source[name]["normalized_sha256"], r"^[0-9a-f]{64}$")

        self.assertEqual(
            self.matrix["summary"],
            {
                "advertisement_corrections": 0,
                "existing_contract_bindings": 0,
                "new_leaf_tasks": 31,
                "report_level_missing_contract_results": 393,
                "unique_capability_tuples": 44,
                "unique_tuple_scope_gaps": 76,
            },
        )


if __name__ == "__main__":
    unittest.main()
