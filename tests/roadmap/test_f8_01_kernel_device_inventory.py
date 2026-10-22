import json
import re
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "kernel-device-obligations-v1.json"
)
KERNEL_MANIFEST_PATH = ROOT / "cmake" / "SlangKernelManifest.cmake"
CAPABILITY_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "capability-obligations-v1.json"
)
REALTIME_SUITE_PATH = (
    ROOT / "conformance" / "v1" / "realtime" / "oracle-suite-v1.json"
)
TASK_DIR = ROOT / "docs" / "roadmap" / "tasks"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmake_set(text, name):
    match = re.search(rf"set\({re.escape(name)}\s+(.*?)\)", text, re.DOTALL)
    if not match:
        raise AssertionError(f"missing CMake set {name}")
    return re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(1))


def canonical_kernels():
    text = KERNEL_MANIFEST_PATH.read_text(encoding="utf-8")
    modules = cmake_set(text, "PARALLEL_MATER_SLANG_KERNEL_MODULES")
    return {
        module: cmake_set(text, f"PARALLEL_MATER_SLANG_{module}_ENTRIES")
        for module in modules
    }


class F801KernelDeviceInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load_json(INVENTORY_PATH)
        cls.canonical = canonical_kernels()
        cls.capabilities = load_json(CAPABILITY_PATH)
        cls.realtime_suite = load_json(REALTIME_SUITE_PATH)

    def test_schema_and_canonical_kernel_inventory_are_exact(self):
        self.assertEqual(
            self.inventory["schema"],
            "parallel-mater-f8-kernel-device-obligations/v1",
        )
        self.assertEqual(self.inventory["task"], "F8-01")

        actual = defaultdict(list)
        group_ids = set()
        for group in self.inventory["kernel_groups"]:
            self.assertNotIn(group["id"], group_ids)
            group_ids.add(group["id"])
            self.assertIn(group["module"], self.canonical)
            self.assertTrue(group["domains"])
            self.assertTrue(group["existing_execution_tests"])
            self.assertTrue(group["implementation_owners"])
            for owner in group["implementation_owners"]:
                self.assertTrue((TASK_DIR / f"{owner}.json").is_file())
            actual[group["module"]].extend(group["entries"])

        self.assertEqual(set(actual), set(self.canonical))
        for module, expected_entries in self.canonical.items():
            self.assertEqual(len(actual[module]), len(set(actual[module])))
            self.assertEqual(set(actual[module]), set(expected_entries), module)
            self.assertTrue((ROOT / "src" / "slang" / f"{module}.slang").is_file())
        self.assertEqual(sum(map(len, actual.values())), 172)

    def test_modes_and_public_cases_match_current_contracts(self):
        contract_modes = {
            mode
            for rule in self.capabilities["rules"]
            for mode in rule["match"]["mode"]
        }
        self.assertEqual(set(self.inventory["advertised_modes"]), contract_modes)

        case_ids = {path.stem for path in (ROOT / "conformance" / "v1" / "cases").glob("*.json")}
        case_sets = self.inventory["public_case_sets"]
        self.assertEqual(set(case_sets["all"]), case_ids)
        for name, members in case_sets.items():
            self.assertEqual(len(members), len(set(members)), name)
            self.assertLessEqual(set(members), case_ids, name)

        for group in self.inventory["kernel_groups"]:
            self.assertLessEqual(set(group["advertised_modes"]), contract_modes)
            for case_set in group["public_case_sets"]:
                self.assertIn(case_set, case_sets)
            self.assertTrue(
                group["public_case_sets"] or group.get("f8_public_cases")
                or group.get("coverage_gap")
            )

        realtime_cases = {
            case["id"] for case in self.realtime_suite["public_trajectory_cases"]
        }
        inventory_cases = {
            case_id
            for group in self.inventory["kernel_groups"]
            for case_id in group.get("f8_public_cases", [])
        }
        self.assertLessEqual(inventory_cases, realtime_cases)

    def test_auxiliary_paint_and_test_kernels_are_accounted_for(self):
        auxiliary = {item["id"]: item for item in self.inventory["auxiliary_paths"]}
        self.assertEqual(
            set(auxiliary),
            {"paint-contact-stamping", "avbd-conformance-entry", "metal-device-contract"},
        )
        canonical = {
            f"{module}.{entry}"
            for module, entries in self.canonical.items()
            for entry in entries
        }
        for entry in auxiliary["paint-contact-stamping"]["canonical_entrypoints"]:
            self.assertIn(entry, canonical)
        self.assertEqual(
            auxiliary["paint-contact-stamping"]["f8_public_cases"],
            ["f8-paint-contact-stamping"],
        )

        geometry_constraints = (
            ROOT / "cmake" / "SlangGeometryConstraints.cmake"
        ).read_text(encoding="utf-8")
        self.assertIn("avbdConformanceMain", geometry_constraints)
        metal_build = (ROOT / "cmake" / "SlangMetal.cmake").read_text(encoding="utf-8")
        self.assertIn("metal_device_contract", metal_build)
        self.assertIn("metal_record_contract", metal_build)

    def test_compiled_variants_map_runtime_targets_to_f8_tests(self):
        variants = {item["id"]: item for item in self.inventory["compiled_variants"]}
        self.assertEqual(
            set(variants),
            {
                "cuda-runtime",
                "vulkan-runtime",
                "d3d12-runtime",
                "metal-runtime",
                "cpu-runtime",
                "cuda-source-mirror",
                "ptx-mirror",
                "spirv-mirror",
            },
        )
        for variant_id in (
            "cuda-runtime",
            "vulkan-runtime",
            "d3d12-runtime",
            "metal-runtime",
            "cpu-runtime",
        ):
            variant = variants[variant_id]
            self.assertTrue(variant["applicable_targets"])
            self.assertTrue(variant["existing_execution_test"])
            self.assertTrue(variant["f8_execution_test"].startswith("f8-kernel-entrypoint-execution/"))
            self.assertEqual(variant["test_contract_owner"], "F8-03")
            self.assertTrue(variant["qualification_owners"])
        for variant_id in ("cuda-source-mirror", "ptx-mirror", "spirv-mirror"):
            variant = variants[variant_id]
            self.assertEqual(variant["applicable_targets"], [])
            self.assertIn("compile", variant["kind"])
            self.assertTrue(variant["execution_not_applicable"])

        self.assertEqual(
            variants["cpu-runtime"]["required_entrypoints"],
            [
                "geometry.*",
                "geometry_constraints.prepare_avbd_kernel",
                "geometry_constraints.solve_avbd_kernel",
                "queries.*",
            ],
        )
        self.assertEqual(
            variants["cpu-runtime"]["optional_entrypoints"],
            ["fluid.*", "geometry_fluid.*"],
        )

    def test_registered_physical_targets_are_bounded_and_unqualified(self):
        targets = {item["id"]: item for item in self.inventory["physical_targets"]}
        self.assertEqual(
            set(targets),
            {
                "windows-gtx1080-slang-cuda",
                "windows-gtx1080-slang-d3d12",
                "linux-thomas-laptop-slang-cpu",
                "linux-rtx3050ti-slang-cuda",
                "linux-rtx3050ti-slang-vulkan",
                "macos-m4-slang-metal",
                "ipados-m1-slang-metal",
                "android-pixel7-slang-vulkan",
            },
        )
        variants = {item["id"]: item for item in self.inventory["compiled_variants"]}
        for target_id, target in targets.items():
            self.assertIn(target_id, variants[target["compiled_variant"]]["applicable_targets"])
            self.assertEqual(
                target["f8_execution_test"],
                variants[target["compiled_variant"]]["f8_execution_test"],
            )
            self.assertTrue(target["os"])
            self.assertTrue(target["driver"])
            self.assertTrue(target["toolchain"])
            self.assertTrue(target["availability"])
            self.assertTrue(target["current_exclusions"])
            self.assertNotIn(target["qualification_status"], {"passed", "qualified"})
            for evidence in target["legacy_evidence"]:
                self.assertTrue((ROOT / evidence).is_file(), evidence)

    def test_device_cards_have_correct_dependency_direction(self):
        audit = self.inventory["device_registration_audit"]
        targets = self.inventory["physical_targets"]
        cards = {target["qualification_owner"] for target in targets}
        self.assertEqual(cards, set(audit["covered_qualification_cards"]))
        self.assertEqual(audit["registered_host_count"], 5)
        self.assertEqual(audit["registered_backend_tuple_count"], len(targets))
        self.assertEqual(audit["additional_device_cards_added"], [])

        exit_dependencies = set(load_json(TASK_DIR / "F8-99.json")["dependencies"])
        for card_id in cards:
            card = load_json(TASK_DIR / f"{card_id}.json")
            self.assertIn("F8-22", card["dependencies"], card_id)
            self.assertIn(card_id, exit_dependencies)
        self.assertNotIn("F8-99", load_json(TASK_DIR / "F8-22.json")["dependencies"])

    def test_spatial_handoff_and_legacy_boundaries_are_explicit(self):
        planned = self.inventory["planned_shared_spatial_entrypoints"]
        self.assertEqual({owner for item in planned for owner in item["owners"]}, {"F8-03", "F8-07"})
        statuses = {owner: item["status"] for item in planned
                    for owner in item["owners"]}
        self.assertEqual(statuses["F8-07"],
                         "implemented-canonical-build-manifest")
        self.assertTrue(statuses["F8-03"].startswith("planned"))

        handoffs = {item["task"]: item for item in self.inventory["handoffs"]}
        self.assertEqual(set(handoffs), {"P3-02", "P1-59", "F5-99"})
        self.assertIn("deferred", handoffs["P3-02"]["state_at_inventory"])
        self.assertIn("leased", handoffs["P1-59"]["state_at_inventory"])
        self.assertEqual(handoffs["F5-99"]["state_at_inventory"], "done")
        for handoff in handoffs.values():
            self.assertTrue(handoff["boundary"])
            self.assertTrue(handoff["f8_owners"])


if __name__ == "__main__":
    unittest.main()
