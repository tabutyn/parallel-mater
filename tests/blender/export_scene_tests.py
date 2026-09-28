# SPDX-License-Identifier: MIT
"""Exercise the one exporter in real Blender; all output stays in a temp dir."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import bpy

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "tools/blender/export_parallel_mater_scene.py"
ASSETS = ROOT / "examples/assets"
parser = argparse.ArgumentParser()
parser.add_argument("--loader")
options = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
spec = importlib.util.spec_from_file_location("parallel_mater_exporter", EXPORTER)
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


def read_glb(path):
    with Path(path).open("rb") as stream:
        magic, version, total, size, kind = struct.unpack("<4sIIII", stream.read(20))
        assert magic == b"glTF" and version == 2 and kind == 0x4E4F534A
        assert total == Path(path).stat().st_size
        return json.loads(stream.read(size))


def systems(document):
    return Counter(node.get("extras", {}).get("pm_system")
                   for node in document["nodes"])


def snapshot():
    return {
        "objects": tuple((obj.name, obj.as_pointer(),
                          tuple(value for row in obj.matrix_world for value in row))
                         for obj in bpy.data.objects),
        "meshes": tuple(mesh.name for mesh in bpy.data.meshes),
        "materials": tuple(material.name for material in bpy.data.materials),
        "collections": tuple(collection.name for collection in bpy.data.collections),
        "selected": tuple(obj.name for obj in bpy.context.selected_objects),
        "active": bpy.context.view_layer.objects.active,
        "frame": bpy.context.scene.frame_current,
        "filepath": bpy.data.filepath,
    }


class ExportSceneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="parallel-mater-export-test-")
        self.output = Path(self.directory.name) / "scene.glb"

    def tearDown(self):
        self.directory.cleanup()

    def check_export(self, expected):
        before = snapshot()
        self.assertEqual(exporter.export_scene(self.output), self.output)
        self.assertEqual(snapshot(), before)
        document = read_glb(self.output)
        self.assertEqual(systems(document), expected)
        for node in document["nodes"]:
            self.assertEqual(node["extras"]["pm_schema"], exporter.SCHEMA_VERSION)
        for mesh in document["meshes"]:
            for primitive in mesh["primitives"]:
                self.assertEqual(primitive.get("mode", 4), 4)  # triangles
        if options.loader:
            subprocess.run([options.loader, str(self.output),
                            str(expected["rigid_body"]), str(expected["cloth"]),
                            str(expected["fluid_inflow"]), str(expected["fluid_outflow"]),
                            str(int(expected["fluid_initial_volume"] > 0)),
                            str(expected["soft_body"])],
                           check=True, timeout=60)
        return document

    def test_all_authored_scenes_share_exporter(self):
        for name in ("PassiveActive", "Fluid", "FluidRigid", "Pegs", "Cloth",
                     "ClothTear", "ClothPaint", "ClothWater", "Softbody",
                     "SoftbodyRigidBody", "SoftbodyCloth"):
            with self.subTest(scene=name):
                source = ASSETS / f"{name}.blend"
                digest = hashlib.sha256(source.read_bytes()).digest()
                bpy.ops.wm.open_mainfile(filepath=str(source))
                expected = systems(read_glb(source.with_suffix(".glb")))
                self.check_export(expected)
                self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)

    def test_cloth_without_rigid_bodies(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Cloth.blend"))
        for obj in list(bpy.context.scene.objects):
            if not any(mod.type == "CLOTH" for mod in obj.modifiers):
                bpy.data.objects.remove(obj, do_unlink=True)
        self.check_export(Counter(cloth=1))

    def test_soft_cloth_materials_are_independent(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "SoftbodyCloth.blend"))
        document = self.check_export(Counter(rigid_body=2, cloth=2, soft_body=1))
        sheets = sorted((node["extras"] for node in document["nodes"]
                         if node["extras"].get("pm_system") == "cloth"),
                        key=lambda sheet: sheet["pm_break_strain"])
        self.assertEqual(sheets[0]["pm_break_strain"], 0.0)
        self.assertAlmostEqual(sheets[1]["pm_break_strain"], 0.10)
        self.assertEqual(sheets[0]["pm_solver_iterations"], 48)
        self.assertEqual(sheets[1]["pm_solver_iterations"], 24)
        for sheet in sheets:
            self.assertTrue(sheet["pm_pin_vertices"])
            self.assertEqual(sheet["pm_pin_stiffness"], 1.0)
            self.assertEqual(sheet["pm_fracture_persistence_substeps"], 4)

    def test_flows_without_rigid_bodies(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Fluid.blend"))
        for obj in list(bpy.context.scene.objects):
            if not any(mod.type == "FLUID" for mod in obj.modifiers):
                bpy.data.objects.remove(obj, do_unlink=True)
        self.check_export(Counter(fluid_inflow=1, fluid_outflow=1))

    def test_soft_body_without_rigid_bodies(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Softbody.blend"))
        for obj in list(bpy.context.scene.objects):
            if not any(mod.type == "SOFT_BODY" for mod in obj.modifiers):
                bpy.data.objects.remove(obj, do_unlink=True)
        document = self.check_export(Counter(soft_body=1))
        extras = next(node["extras"] for node in document["nodes"]
                      if node["extras"].get("pm_system") == "soft_body")
        self.assertEqual(extras["pm_maximum_projection_fraction"], 0.20)
        self.assertEqual(extras["pm_constraint_velocity_response"], 0.70)
        self.assertAlmostEqual(extras["pm_shape_matching_stiffness"], 0.35,
                               places=6)
        self.assertEqual(extras["pm_maximum_speed"], 2.0)
        self.assertEqual(extras["pm_solver_iterations"], 16)

    def test_failed_export_cleans_up_and_restores_selection(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Cloth.blend"))
        body = next(obj for obj in bpy.context.scene.objects if obj.rigid_body)
        body["pm_initial_velocity"] = (1.0, 2.0)  # fails after temporary object creation
        bpy.ops.object.select_all(action="DESELECT")
        body.select_set(True)
        bpy.context.view_layer.objects.active = None
        before = snapshot()
        with self.assertRaisesRegex(RuntimeError, "3 components"):
            exporter.export_scene(self.output)
        self.assertEqual(snapshot(), before)
        self.assertFalse(self.output.exists())

    def test_output_guard_and_empty_scene(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with self.assertRaisesRegex(ValueError, ".glb extension"):
            exporter.export_scene(self.output.with_suffix(".blend"))
        with self.assertRaisesRegex(RuntimeError, "save the .blend"):
            exporter.export_scene()
        with self.assertRaisesRegex(RuntimeError, "no rigid bodies, soft bodies, cloth, or liquid flows"):
            exporter.export_scene(self.output)

    def test_blender_menu_operator_uses_same_export(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Cloth.blend"))
        exporter.register()
        try:
            before = snapshot()
            self.assertEqual(bpy.ops.export_scene.parallel_mater(filepath=str(self.output)),
                             {"FINISHED"})
            self.assertEqual(snapshot(), before)
            self.assertEqual(systems(read_glb(self.output)),
                             systems(read_glb(ASSETS / "Cloth.glb")))
        finally:
            exporter.unregister()

    def test_headless_cli_uses_same_export(self):
        subprocess.run([bpy.app.binary_path, "--background", "--threads", "1",
                        str(ASSETS / "Cloth.blend"), "--python-exit-code", "1",
                        "--python", str(EXPORTER), "--", "--output", str(self.output)],
                       check=True, timeout=120)
        self.assertEqual(systems(read_glb(self.output)),
                         systems(read_glb(ASSETS / "Cloth.glb")))


result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(ExportSceneTests))
if not result.wasSuccessful():
    raise RuntimeError("ParallelMater Blender export tests failed")
