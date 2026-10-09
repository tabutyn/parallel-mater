# SPDX-License-Identifier: MIT
"""Exercise the one exporter in real Blender; all output stays in a temp dir."""

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import re
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
        "curves": tuple(curve.name for curve in bpy.data.curves),
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

    def test_property_reference_covers_exporter(self):
        reference = (ROOT / "docs/BLENDER_SCENES.md").read_text()
        documented = set(re.findall(r"`(pm_[A-Za-z0-9_]+)`", reference))
        for name in tuple(documented):
            if "AXIS" in name:
                documented.update(name.replace("AXIS", axis) for axis in "xyz")
            if "END" in name:
                documented.update(name.replace("END", end) for end in ("first", "last"))
        tree = ast.parse(EXPORTER.read_text())
        properties = {node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Constant) and isinstance(node.value, str)
                      and re.fullmatch(r"pm_[a-z0-9_]*[a-z0-9]", node.value)}
        self.assertFalse(properties - documented,
                         f"Undocumented properties: {sorted(properties - documented)}")

    def check_export(self, expected):
        before = snapshot()
        self.assertEqual(exporter.export_scene(self.output), self.output)
        self.assertEqual(snapshot(), before)
        document = read_glb(self.output)
        self.assertEqual(systems(document), expected)
        for node in document["nodes"]:
            self.assertEqual(node["extras"]["pm_schema"], exporter.SCHEMA_VERSION)
            self.assertNotIn("pm_checkerboard", node["extras"])
        for mesh in document.get("meshes", []):
            for primitive in mesh["primitives"]:
                self.assertEqual(primitive.get("mode", 4), 4)  # triangles
        if options.loader:
            subprocess.run([options.loader, str(self.output),
                            # The loader turns each heated surface into a
                            # passive collision body as well as a heat plane.
                            str(expected["rigid_body"] + expected["thermal_surface"]),
                            str(expected["rigid_constraint"]),
                            str(expected["cloth"]),
                            str(expected["fluid_inflow"]), str(expected["fluid_outflow"]),
                            str(int(expected["fluid_initial_volume"] > 0)),
                            str(expected["soft_body"]), str(expected["rope"]),
                            str(expected["hit_box"]), str(expected["sphere_cluster"])],
                           check=True, timeout=60)
        return document

    def test_sphere_cluster_empty_exports_pose_and_shared_template(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.object.empty_add(location=(1, 2, 3), rotation=(0.2, -0.3, 0.4))
        marker = bpy.context.object
        marker.name = "SphereCluster"
        marker.scale = (2, 3, 4)  # Empty display scale must not resize payload.
        bpy.context.view_layer.update()
        document = self.check_export(Counter(sphere_cluster=1))
        node = document["nodes"][0]
        self.assertEqual(node["extras"]["pm_name"], "SphereCluster")
        for actual, expected in zip(node["translation"], (1, 3, -2)):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertIn("rotation", node)
        self.assertNotIn("scale", node)
        primitive = document["meshes"][node["mesh"]]["primitives"][0]
        self.assertEqual(document["accessors"][primitive["indices"]]["count"], 80*3)
        bpy.ops.object.empty_add()
        marker.parent = bpy.context.object
        with self.assertRaisesRegex(RuntimeError, "scene-root Empty"):
            exporter.export_scene(self.output)

    def test_dump_truck_cluster_source(self):
        source = ASSETS / "DumpTruck.blend"
        digest = hashlib.sha256(source.read_bytes()).digest()
        bpy.ops.wm.open_mainfile(filepath=str(source))
        marker = bpy.data.objects["SphereCluster"]
        self.assertEqual(marker.type, "EMPTY")
        position = marker.matrix_world.translation
        document = self.check_export(Counter(rigid_body=11, rigid_constraint=9,
                                              hit_box=1, sphere_cluster=1))
        node = next(node for node in document["nodes"]
                    if node["extras"]["pm_system"] == "sphere_cluster")
        for actual, expected in zip(node["translation"], (position.x, position.z, -position.y)):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)

    def test_canonical_hit_box_preserves_oriented_nonuniform_bounds(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_cube_add(
            location=(1.0, 2.0, 3.0), rotation=(0.2, -0.3, 0.4))
        hit_box = bpy.context.object
        hit_box.name = "GoalVolume"
        hit_box.scale = (1.0, 2.0, 3.0)
        hit_box["pm_hit_box"] = True
        bpy.context.view_layer.update()

        document = self.check_export(Counter(hit_box=1))
        node = next(node for node in document["nodes"]
                    if node.get("extras", {}).get("pm_system") == "hit_box")
        extras = node["extras"]
        self.assertEqual(extras["pm_name"], "GoalVolume")
        self.assertAlmostEqual(extras["pm_half_extent_x"], 1.0, places=5)
        self.assertAlmostEqual(extras["pm_half_extent_y"], 3.0, places=5)
        self.assertAlmostEqual(extras["pm_half_extent_z"], 2.0, places=5)
        self.assertIn("rotation", node)

    def test_all_authored_scenes_share_exporter(self):
        for name in ("PassiveActive", "RigidBody", "Fluid", "FluidRigid", "Pegs", "Cloth",
                     "ClothTear", "ClothPaint", "ClothWater", "Softbody",
                     "SoftbodyRigidBody", "SoftbodyCloth", "SoftbodyFluid", "Rope",
                     "RopeFluid", "RopeCloth", "RopeSoftbody", "Smoke", "SmokeWater",
                     "SmokeRope", "SmokeSoftbody", "SmokeCloth",
                     "ConstraintFixed", "ConstraintPoint", "Celestial", "ConstraintHinge",
                     "ConstraintSlider", "ConstraintPiston", "ConstraintGeneric",
                     "ConstraintGenericSpring", "ConstraintMotor",
                     "ConstraintMotorSpring", "DumpTruck"):
            with self.subTest(scene=name):
                source = ASSETS / f"{name}.blend"
                digest = hashlib.sha256(source.read_bytes()).digest()
                bpy.ops.wm.open_mainfile(filepath=str(source))
                for prop in bpy.data.bl_rna.properties:
                    if prop.type == 'COLLECTION':
                        for block in getattr(bpy.data, prop.identifier):
                            if isinstance(block, bpy.types.ID):
                                self.assertNotIn("pm_checkerboard", block)
                expected = systems(read_glb(source.with_suffix(".glb")))
                for node in read_glb(source.with_suffix(".glb"))["nodes"]:
                    self.assertNotIn("pm_checkerboard", node.get("extras", {}))
                self.check_export(expected)
                self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)

    def test_arrow_force_and_array_inheritance(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_cube_add()
        body = bpy.context.object
        bpy.ops.rigidbody.object_add()
        body["pm_arrow"] = 100.0
        modifier = body.modifiers.new("Copies", "ARRAY")
        modifier.count = 2
        modifier.relative_offset_displace = (2.0, 0.0, 0.0)
        document = self.check_export(Counter(rigid_body=2))
        self.assertTrue(all(node["extras"]["pm_arrow"] == 100.0 for node in document["nodes"]))
        for value in (-1.0, float("inf"), float("nan"), True, "100", 1.0e100):
            with self.subTest(value=value):
                body["pm_arrow"] = value
                with self.assertRaisesRegex(RuntimeError, "pm_arrow must be finite, non-negative newtons"):
                    exporter.export_scene(self.output)
        body["pm_arrow"] = 100.0
        body.rigid_body.type = 'PASSIVE'
        with self.assertRaisesRegex(RuntimeError, "pm_arrow requires an ACTIVE"):
            exporter.export_scene(self.output)
        body.rigid_body.type = 'ACTIVE'
        body.rigid_body.kinematic = True
        with self.assertRaisesRegex(RuntimeError, "pm_arrow requires an ACTIVE"):
            exporter.export_scene(self.output)
        body["pm_arrow"] = 0.0
        self.check_export(Counter(rigid_body=2))

    @unittest.skipUnless(options.loader, "needs gallery scene loader")
    def test_loader_rejects_invalid_arrow_force(self):
        # Validate the receiving side too: another exporter can write these.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_cube_add()
        bpy.ops.rigidbody.object_add()
        bpy.context.object["pm_arrow"] = 100.0
        exporter.export_scene(self.output)
        raw = self.output.read_bytes()
        size = struct.unpack_from('<I', raw, 12)[0]
        tail = raw[20 + size:]
        for value, motion in ((-1, 'dynamic'), ('100', 'dynamic'),
                              (True, 'dynamic'), (None, 'dynamic'),
                              (1e100, 'dynamic'), (100, 'static'),
                              (100, 'kinematic')):
            with self.subTest(value=value, motion=motion):
                document = json.loads(raw[20:20 + size])
                document['nodes'][0]['extras'].update(pm_arrow=value, pm_motion=motion)
                encoded = json.dumps(document, allow_nan=False).encode()
                encoded += b' ' * (-len(encoded) % 4)
                self.output.write_bytes(struct.pack('<4sIIII', b'glTF', 2,
                    20 + len(encoded) + len(tail), len(encoded), 0x4E4F534A) + encoded + tail)
                result = subprocess.run([options.loader, str(self.output), '1', *(['0'] * 8)],
                                        capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('pm_arrow', result.stderr + result.stdout)

    def test_authored_generic_arrow_force(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "ConstraintGeneric.blend"))
        document = self.check_export(Counter(rigid_body=4, rigid_constraint=1))
        driven = [node["extras"] for node in document["nodes"] if node["extras"].get("pm_arrow", 0) > 0]
        self.assertEqual(len(driven), 1)
        self.assertEqual(driven[0]["pm_source_name"], "GenericBlockA")
        self.assertEqual(driven[0]["pm_arrow"], 100.0)

    def test_rigid_body_array_wall_and_hit_box(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "RigidBody.blend"))
        document = self.check_export(Counter(rigid_body=386, hit_box=1))
        bodies = [node["extras"] for node in document["nodes"]
                  if node.get("extras", {}).get("pm_system") == "rigid_body"]
        self.assertEqual(Counter(body["pm_source_name"] for body in bodies),
                         Counter({"Ground": 1, "Icosphere": 1,
                                  "Layer1": 192, "Layer2": 192}))
        self.assertEqual(len({body["pm_name"] for body in bodies}), 386)
        self.assertTrue(all(not body["pm_gravity_tilt"] for body in bodies
                            if body["pm_source_name"] in {"Layer1", "Layer2"}))
        self.assertTrue(all(body["pm_gravity_tilt"] for body in bodies
                            if body["pm_source_name"] in {"Ground", "Icosphere"}))
        # The gallery loads the committed GLB, not this temporary fresh export.
        # Catch stale assets produced by an older Blender script, which can
        # silently omit the false tilt flags and restore global wall steering.
        committed = {
            node["extras"]["pm_name"]: node["extras"]
            for node in read_glb(ASSETS / "RigidBody.glb")["nodes"]
            if node.get("extras", {}).get("pm_system") == "rigid_body"
        }
        self.assertEqual(set(committed), {body["pm_name"] for body in bodies})
        for body in bodies:
            for key in ("pm_gravity_tilt", "pm_mass"):
                self.assertEqual(committed[body["pm_name"]].get(key), body[key],
                                 f"RigidBody.glb is stale: {body['pm_name']} {key}; "
                                 "re-export with the current repository exporter")
        hit_box = next(node["extras"] for node in document["nodes"]
                       if node.get("extras", {}).get("pm_system") == "hit_box")
        self.assertEqual(hit_box["pm_name"], "LoadBox")
        for axis in "xyz":
            self.assertAlmostEqual(hit_box[f"pm_half_extent_{axis}"],
                                   4.398349285125732, places=5)

    def test_smoke_rope_active_panel_attachments(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "SmokeRope.blend"))
        document = self.check_export(Counter(rigid_body=5, rope=4,
                                             smoke_emitter=1))
        ropes = [node["extras"] for node in document["nodes"]
                 if node.get("extras", {}).get("pm_system") == "rope"]
        self.assertEqual(len(ropes), 4)
        self.assertTrue(all(rope["pm_rope_first_body"] == "Plane.001"
                            for rope in ropes))
        self.assertEqual(Counter(rope["pm_rope_last_body"] for rope in ropes),
                         Counter({"Cylinder": 2, "Cylinder.001": 2}))
        panel = next(node["extras"] for node in document["nodes"]
                     if node.get("extras", {}).get("pm_source_name") == "Plane.001")
        self.assertTrue(panel["pm_smoke_collider"])

    def test_rigid_constraint_settings(self):
        expected_types = {
            "ConstraintFixed": ("fixed", 1),
            "ConstraintPoint": ("point", 4),
            "ConstraintHinge": ("hinge", 4),
            "ConstraintSlider": ("slider", 1),
            "ConstraintPiston": ("piston", 1),
            "ConstraintGeneric": ("generic", 1),
            "ConstraintGenericSpring": ("generic_spring", 1),
            "ConstraintMotor": ("motor", 4),
        }
        for name, (kind, count) in expected_types.items():
            with self.subTest(scene=name):
                bpy.ops.wm.open_mainfile(filepath=str(ASSETS / f"{name}.blend"))
                document = self.check_export(Counter(
                    rigid_body=6 if kind in ("motor", "point") else
                               5 if kind == "hinge" else
                               3 if kind == "piston" else
                               51 if kind == "fixed" else 4,
                    rigid_constraint=count,
                    collision_mesh=1 if name == "ConstraintFixed" else 0))
                constraints = [node["extras"] for node in document["nodes"]
                               if node["extras"].get("pm_system") ==
                               "rigid_constraint"]
                self.assertEqual(len(constraints), count)
                self.assertEqual(
                    Counter(item["pm_constraint_type"] for item in constraints),
                    Counter(hinge=3, slider=1) if kind == "hinge" else
                    Counter({kind: count}))
                self.assertTrue(all(item["pm_body_a"] and item["pm_body_b"]
                                    for item in constraints))
                if kind == "fixed":
                    self.assertTrue(constraints[0]["pm_enabled"])
                    self.assertEqual(
                        (constraints[0]["pm_body_a"],
                         constraints[0]["pm_body_b"]),
                        ("Large", "Small.048"))
                    bodies = [node["extras"] for node in document["nodes"]
                              if node.get("extras", {}).get("pm_system") ==
                              "rigid_body"]
                    names = {body["pm_source_name"] for body in bodies}
                    self.assertIn("Ground", names)
                    self.assertIn("Large", names)
                    self.assertEqual(
                        len([name for name in names if name.startswith("Small")]),
                        49)
                    moving = [body for body in bodies
                              if body["pm_source_name"] == "Large" or
                              body["pm_source_name"].startswith("Small")]
                    self.assertTrue(all("pm_initial_velocity_x" not in body
                                        for body in moving))
                if kind == "point":
                    self.assertTrue(all(item["pm_enabled"]
                                        for item in constraints))
                    self.assertEqual({item["pm_body_a"] for item in constraints},
                                     {"PointPost"})
                    self.assertEqual({item["pm_body_b"] for item in constraints},
                                     {"PointSphereA", "PointSphereB",
                                      "PointSphereB.001", "PointSphereB.002"})
                    bodies = {
                        node["extras"]["pm_source_name"]: node["extras"]
                        for node in document["nodes"]
                        if node.get("extras", {}).get("pm_system") ==
                        "rigid_body"
                    }
                    self.assertLess(
                        bodies["PointSphereA"]["pm_initial_velocity_z"] *
                        bodies["PointSphereB"]["pm_initial_velocity_z"], 0.0)
                    self.assertLess(
                        bodies["PointSphereB.001"]["pm_initial_velocity_x"] *
                        bodies["PointSphereB.002"]["pm_initial_velocity_x"],
                        0.0)
                if kind == "hinge":
                    self.assertEqual(
                        {(item["pm_body_a"], item["pm_body_b"])
                         for item in constraints},
                        {("Ground.001", "Gear"),
                         ("Ground.001", "Gear.001"),
                         ("Ground.001", "Gear.002"),
                         ("Ground.001", "Ground.002")})
                    self.assertTrue(all(item["pm_solver_iterations"] == 64
                                        for item in constraints
                                        if item["pm_constraint_type"] == "hinge"))
                    slider = next(item for item in constraints
                                  if item["pm_constraint_type"] == "slider")
                    self.assertTrue(slider["pm_enabled"])
                    self.assertEqual(slider["pm_solver_iterations"], 8)
                    self.assertFalse(slider["pm_use_limit_lin_x"])
                    self.assertEqual(slider["pm_limit_lin_x_lower"], -4.0)
                    self.assertEqual(slider["pm_limit_lin_x_upper"], 2.0)
                    rod = bpy.context.scene.objects["Ground.002"]
                    self.assertIsNotNone(rod.rigid_body)
                    self.assertEqual(rod.rigid_body_constraint.object2, rod)
                    for joint in bpy.context.scene.objects:
                        constraint = joint.rigid_body_constraint
                        if constraint is None or constraint.type != "HINGE":
                            continue
                        gear = constraint.object2
                        joint_z = joint.matrix_world.to_3x3().normalized().col[2]
                        gear_z = gear.matrix_world.to_3x3().normalized().col[2]
                        self.assertAlmostEqual(joint_z.dot(gear_z), 1.0,
                                               places=6)
                    self.assertEqual(
                        {node["extras"]["pm_source_name"]
                         for node in document["nodes"]
                         if node.get("extras", {}).get("pm_system") ==
                         "rigid_body"},
                        {"Ground.001", "Ground.002", "Gear", "Gear.001",
                         "Gear.002"})
                    bodies = {
                        node["extras"]["pm_source_name"]: node["extras"]
                        for node in document["nodes"]
                        if node.get("extras", {}).get("pm_system") ==
                        "rigid_body"
                    }
                    self.assertAlmostEqual(
                        bodies["Ground.001"]["pm_friction"], 4.0, places=5)
                    self.assertAlmostEqual(bodies["Gear"]["pm_friction"],
                                           0.08, places=5)
                    self.assertAlmostEqual(
                        bodies["Gear"]["pm_restitution"], 0.0, places=5)
                    self.assertAlmostEqual(
                        bodies["Gear"]["pm_angular_damping"], 0.03,
                        places=5)
                    self.assertAlmostEqual(bodies["Gear.001"]["pm_mass"],
                                           1.0, places=5)
                    self.assertAlmostEqual(
                        bodies["Gear.001"]["pm_friction"], 0.08, places=5)
                    self.assertAlmostEqual(
                        bodies["Gear.001"]["pm_restitution"], 0.0,
                        places=5)
                    self.assertAlmostEqual(
                        bodies["Gear.001"]["pm_angular_damping"], 0.01,
                        places=5)
                    self.assertAlmostEqual(bodies["Gear.002"]["pm_mass"],
                                           1.0, places=5)
                    self.assertAlmostEqual(
                        bodies["Gear.002"]["pm_friction"], 0.08, places=5)
                    self.assertAlmostEqual(
                        bodies["Gear.002"]["pm_restitution"], 0.0,
                        places=5)
                    self.assertAlmostEqual(
                        bodies["Gear.002"]["pm_angular_damping"], 0.01,
                        places=5)
                if kind in ("slider", "piston"):
                    self.assertEqual(constraints[0]["pm_limit_lin_x_lower"], -1.0)
                    self.assertEqual(constraints[0]["pm_limit_lin_x_upper"], 1.0)
                if kind == "generic_spring":
                    self.assertFalse(constraints[0]["pm_use_spring_x"])
                    for axis in "xyz":
                        self.assertTrue(
                            constraints[0][f"pm_use_spring_ang_{axis}"])
                        self.assertEqual(
                            constraints[0][f"pm_spring_stiffness_ang_{axis}"],
                            80.0)
                        self.assertEqual(
                            constraints[0][f"pm_spring_damping_ang_{axis}"],
                            0.5)
                if kind == "motor":
                    self.assertTrue(all(item["pm_use_motor_ang"]
                                        for item in constraints))

    def test_motor_spring_constraint_settings(self):
        bpy.ops.wm.open_mainfile(
            filepath=str(ASSETS / "ConstraintMotorSpring.blend"))
        document = self.check_export(Counter(rigid_body=10,
                                             rigid_constraint=8))
        constraints = [node["extras"] for node in document["nodes"]
                       if node.get("extras", {}).get("pm_system") ==
                       "rigid_constraint"]
        self.assertEqual(Counter(item["pm_constraint_type"]
                                 for item in constraints),
                         Counter(motor=4, generic_spring=4))
        motors = [item for item in constraints
                  if item["pm_constraint_type"] == "motor"]
        self.assertTrue(all(item["pm_use_motor_ang"] and
                            item["pm_motor_ang_max_impulse"] == 8.0
                            for item in motors))
        springs = [item for item in constraints
                   if item["pm_constraint_type"] == "generic_spring"]
        for spring in springs:
            for axis in "xy":
                self.assertTrue(spring[f"pm_use_limit_lin_{axis}"])
                self.assertEqual(spring[f"pm_limit_lin_{axis}_lower"], 0.0)
                self.assertEqual(spring[f"pm_limit_lin_{axis}_upper"], 0.0)
                self.assertFalse(spring[f"pm_use_spring_{axis}"])
            self.assertTrue(spring["pm_use_limit_lin_z"])
            self.assertAlmostEqual(spring["pm_limit_lin_z_lower"], -0.10)
            self.assertAlmostEqual(spring["pm_limit_lin_z_upper"], 0.10)
            self.assertTrue(spring["pm_use_spring_z"])
            self.assertEqual(spring["pm_spring_stiffness_z"], 500.0)
            self.assertEqual(spring["pm_spring_damping_z"], 8.0)
            for axis in "xyz":
                self.assertTrue(spring[f"pm_use_limit_ang_{axis}"])
                self.assertEqual(spring[f"pm_limit_ang_{axis}_lower"], 0.0)
                self.assertEqual(spring[f"pm_limit_ang_{axis}_upper"], 0.0)
                self.assertFalse(spring[f"pm_use_spring_ang_{axis}"])

    def test_fixed_collector_scene(self):
        bpy.ops.wm.open_mainfile(
            filepath=str(ASSETS / "ConstraintFixed.blend"))
        document = self.check_export(Counter(rigid_body=51,
                                             rigid_constraint=1,
                                             collision_mesh=1))
        constraints = [node["extras"] for node in document["nodes"]
                       if node["extras"].get("pm_system") ==
                       "rigid_constraint"]
        self.assertEqual(len(constraints), 1)
        self.assertTrue(constraints[0]["pm_enabled"])
        self.assertEqual((constraints[0]["pm_body_a"],
                          constraints[0]["pm_body_b"]),
                         ("Large", "Small.048"))
        bodies = [node["extras"] for node in document["nodes"]
                  if node.get("extras", {}).get("pm_system") == "rigid_body"]
        names = {body["pm_source_name"] for body in bodies}
        self.assertEqual(len([name for name in names
                              if name.startswith("Small")]), 49)
        body_by_name = {body["pm_source_name"]: body for body in bodies}
        self.assertAlmostEqual(
            body_by_name["Ground"]["pm_friction"], 4.0, places=5)
        self.assertTrue(all(
            abs(body["pm_friction"] - 16.0) < 1.0e-5
            for body in bodies
            if body["pm_source_name"].startswith("Small")))
        moving = [body for body in bodies
                  if body["pm_source_name"] == "Large" or
                  body["pm_source_name"].startswith("Small")]
        self.assertTrue(all("pm_initial_velocity_x" not in body
                            for body in moving))

        from mathutils.bvhtree import BVHTree

        def geometry(obj):
            evaluated = obj.evaluated_get(
                bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            vertices = [evaluated.matrix_world @ vertex.co
                        for vertex in mesh.vertices]
            polygons = [tuple(polygon.vertices)
                        for polygon in mesh.polygons]
            evaluated.to_mesh_clear()
            return vertices, BVHTree.FromPolygons(
                vertices, polygons, all_triangles=False)

        large = bpy.context.scene.objects["Large"]
        proxy = bpy.context.scene.objects[large["pm_collision_proxy"]]
        self.assertIsNone(proxy.rigid_body)
        self.assertEqual(body_by_name["Large"]["pm_collision_proxy"],
                         "Large__PM_COLLISION")
        self.assertEqual(len(proxy.data.polygons), 5120)
        large_vertices, large_tree = geometry(proxy)
        small_vertices, small_tree = geometry(
            bpy.context.scene.objects["Small.048"])
        self.assertEqual(large_tree.overlap(small_tree), [])
        surface_gap = min(
            min(large_tree.find_nearest(point)[3]
                for point in small_vertices),
            min(small_tree.find_nearest(point)[3]
                for point in large_vertices))
        self.assertLess(surface_gap, 0.001)

    def test_mesh_constraint_requires_both_targets_without_changing_source(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "ConstraintHinge.blend"))
        constraint = bpy.context.scene.objects["Ground.002"].rigid_body_constraint
        for field in ("object1", "object2"):
            with self.subTest(field=field):
                target = getattr(constraint, field)
                setattr(constraint, field, None)
                before = snapshot()
                with self.assertRaisesRegex(
                        RuntimeError, "Ground.002: constraint needs Object 1 and Object 2"):
                    exporter.export_scene(self.output)
                self.assertEqual(snapshot(), before)
                self.assertFalse(self.output.exists())
                setattr(constraint, field, target)

    def test_hinge_gears_remain_engaged_through_tooth_cycle(self):
        from mathutils import Quaternion, Vector
        from mathutils.bvhtree import BVHTree

        bpy.ops.wm.open_mainfile(
            filepath=str(ASSETS / "ConstraintHinge.blend"))
        gears = [bpy.context.scene.objects[name]
                 for name in ("Gear", "Gear.001", "Gear.002")]
        bases = [gear.matrix_world.to_quaternion() for gear in gears]

        def geometry(obj):
            evaluated = obj.evaluated_get(
                bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            vertices = [evaluated.matrix_world @ vertex.co
                        for vertex in mesh.vertices]
            polygons = [tuple(polygon.vertices)
                        for polygon in mesh.polygons]
            evaluated.to_mesh_clear()
            return vertices, BVHTree.FromPolygons(
                vertices, polygons, all_triangles=False)

        maximum_clearance = [0.0, 0.0]
        maximum_overlap = [0, 0]
        for half_degree in range(61):
            angle = half_degree * 0.5
            gear_angles = (angle, -angle * 0.5, angle * 0.5)
            for gear, base, gear_angle in zip(gears, bases, gear_angles):
                gear.rotation_mode = "QUATERNION"
                gear.rotation_quaternion = base @ Quaternion(
                    Vector((0.0, 0.0, 1.0)),
                    gear_angle * 3.141592653589793 / 180.0)
            bpy.context.view_layer.update()
            geometry_by_gear = [geometry(gear) for gear in gears]
            for pair in range(2):
                first_vertices, first_tree = geometry_by_gear[pair]
                second_vertices, second_tree = geometry_by_gear[pair + 1]
                maximum_overlap[pair] = max(
                    maximum_overlap[pair],
                    len(first_tree.overlap(second_tree)))
                clearance = min(
                    min(first_tree.find_nearest(point)[3]
                        for point in second_vertices),
                    min(second_tree.find_nearest(point)[3]
                        for point in first_vertices))
                maximum_clearance[pair] = max(maximum_clearance[pair],
                                               clearance)

        for pair in range(2):
            combined_margin = (gears[pair].rigid_body.collision_margin +
                               gears[pair + 1].rigid_body.collision_margin)
            self.assertLess(maximum_clearance[pair], combined_margin * 1.10)
            # A small number of coplanar tooth-face intersections is expected
            # in the authored contact pose. Runtime regression coverage bounds
            # the actual recovered penetration by the collision margins.
            self.assertLess(maximum_overlap[pair], 32)

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

    def test_nonrectangular_inflow_exports_surface_and_velocity(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "SoftbodyFluid.blend"))
        obj = next(obj for obj in bpy.context.scene.objects if any(
            m.type == "FLUID" and m.fluid_type == "FLOW" and
            m.flow_settings.flow_behavior == "INFLOW" for m in obj.modifiers))
        mesh = bpy.data.meshes.new("TiltedEmissionTriangle")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0.5), (0, 1, 0)], [], [(0, 1, 2)])
        obj.data = mesh
        obj["pm_source_spacing"] = 0.25
        obj["pm_particles_per_second"] = 999999
        flow = next(m.flow_settings for m in obj.modifiers if m.type == "FLUID")
        flow.use_initial_velocity = True
        flow.velocity_coord = (0, 0, -1)
        document = self.check_export(Counter(rigid_body=2, soft_body=1,
                                            fluid_inflow=1, fluid_outflow=1))
        extras = next(node["extras"] for node in document["nodes"]
                      if node["extras"].get("pm_system") == "fluid_inflow")
        self.assertEqual(extras["pm_source_spacing"], 0.25)
        self.assertEqual(extras["pm_velocity_y"], -1)
        self.assertNotIn("pm_particles_per_second", extras)

    def test_rope_uses_evaluated_hook_positions(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Rope.blend"))
        rope = next(obj for obj in bpy.context.scene.objects if obj.type == "CURVE")
        hooks = [m for m in rope.modifiers if m.type == "HOOK"]
        for shift in (0.0, 0.1):
            hooks[0].object.location.x += shift
            bpy.context.view_layer.update()
            evaluated = rope.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            expected = [evaluated.matrix_world @ mesh.vertices[i].co for i in (0, len(mesh.vertices)-1)]
            evaluated.to_mesh_clear()
            document = self.check_export(Counter(rigid_body=4, rope=1))
            extras = next(node["extras"] for node in document["nodes"] if node["extras"].get("pm_system") == "rope")
            points = extras["pm_rope_points"].split(";")
            for encoded, point in zip((points[0], points[-1]), expected):
                for actual, wanted in zip(map(float, encoded.split(",")), (point.x, point.z, -point.y)):
                    self.assertAlmostEqual(actual, wanted, places=5)
        hooks[0].object = None
        with self.assertRaisesRegex(RuntimeError, "Hook target must be a rigid or soft body"):
            exporter.export_scene(self.output)

    def test_rope_hook_can_target_soft_body(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "RopeSoftbody.blend"))
        document = self.check_export(Counter(rigid_body=3, soft_body=1, rope=1))
        extras = next(node["extras"] for node in document["nodes"]
                      if node["extras"].get("pm_system") == "rope")
        self.assertEqual(extras["pm_rope_first_body"], "Icosphere")
        self.assertEqual(extras["pm_rope_last_soft_body"], "Cylinder")
        post = next(node["extras"] for node in document["nodes"]
                    if node["extras"].get("pm_system") == "soft_body" and
                    node["extras"].get("pm_pin_group") == "PostBase")
        self.assertEqual(post["pm_pin_group"], "PostBase")
        self.assertEqual(len(post["pm_pin_vertices"].split(";")), 32)

    def test_rope_cloth_geometry_infers_four_joints(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "RopeCloth.blend"))
        document = self.check_export(Counter(rigid_body=7, cloth=1, rope=4))
        rope_nodes = [node["extras"] for node in document["nodes"]
                      if node["extras"].get("pm_system") == "rope"]
        self.assertEqual({node["pm_rope_first_body"] for node in rope_nodes},
                         {"Cylinder", "Cylinder.001", "Cylinder.002", "Cylinder.003"})
        self.assertTrue(all(node["pm_rope_last_cloth"] == "Plane.009"
                            for node in rope_nodes))
        cloth = next(node for node in document["nodes"]
                     if node["extras"].get("pm_system") == "cloth")
        self.assertTrue(cloth["extras"]["pm_weld_vertices"])
        mesh = document["meshes"][cloth["mesh"]]
        accessor = document["accessors"][mesh["primitives"][0]["attributes"]["POSITION"]]
        self.assertGreaterEqual(accessor["count"], 289)

    def test_smoke_flow_exports_separate_gas_system(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "Smoke.blend"))
        document = self.check_export(Counter(rigid_body=1, smoke_emitter=1))
        emitter = next(node["extras"] for node in document["nodes"]
                       if node["extras"].get("pm_system") == "smoke_emitter")
        self.assertEqual(emitter["pm_smoke_obstacle"], "VortexSphere")
        self.assertAlmostEqual(emitter["pm_velocity_x"], 1.6)
        self.assertEqual(emitter["pm_smoke_model_version"], 3)
        self.assertEqual(emitter["pm_smoke_grid_resolution"], 128)
        self.assertEqual(emitter["pm_smoke_grid_vertical_resolution"], 32)
        self.assertEqual(emitter["pm_smoke_grid_pressure_iterations"], 24)
        self.assertAlmostEqual(
            emitter["pm_smoke_grid_kinematic_viscosity"], 1.5e-5)
        self.assertAlmostEqual(emitter["pm_smoke_grid_les_coefficient"], 0.12)
        self.assertAlmostEqual(
            emitter["pm_smoke_grid_pressure_tolerance"], 1.0e-3)
        self.assertEqual(emitter["pm_smoke_wind_response"], 0.5)
        self.assertEqual(emitter["pm_smoke_pressure_stiffness"], 2.0)
        self.assertEqual(emitter["pm_smoke_rest_number_density"], 12.0)
        self.assertEqual(emitter["pm_smoke_vorticity_confinement"], 0.1)
        self.assertNotIn("pm_smoke_wake_strength", emitter)

    def test_smoke_water_temperature_and_heater(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "SmokeWater.blend"))
        document = self.check_export(Counter(
            rigid_body=3, smoke_emitter=1, fluid_initial_volume=1,
            thermal_surface=1))
        extras = [node["extras"] for node in document["nodes"]]
        liquid = next(item for item in extras
                      if item.get("pm_system") == "fluid_initial_volume")
        heater = next(item for item in extras
                      if item.get("pm_system") == "thermal_surface")
        self.assertEqual(liquid["pm_temperature"], 80.0)
        self.assertEqual(heater["pm_temperature"], 500.0)
        sphere = next(node["extras"] for node in document["nodes"]
                      if node["extras"].get("pm_system") == "rigid_body")
        self.assertNotIn("pm_checkerboard", sphere)

    def test_rope_fluid_authored_mass_and_geometry(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "RopeFluid.blend"))
        self.assertAlmostEqual(bpy.data.objects["Icosphere"].rigid_body.mass, 20.0)
        self.assertAlmostEqual(bpy.data.objects["Cylinder"].location.z, 0.30825454)
        self.check_export(Counter(rigid_body=4, rope=1, fluid_initial_volume=1))

    def test_soft_goal_group_exports_only_full_weight_pins(self):
        bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "SoftbodyFluid.blend"))
        document = self.check_export(Counter(rigid_body=2, soft_body=1,
                                            fluid_inflow=1, fluid_outflow=1))
        extras = next(node["extras"] for node in document["nodes"]
                      if node["extras"].get("pm_system") == "soft_body")
        self.assertEqual(extras["pm_pin_group"], "Goal")
        pins = extras["pm_pin_vertices"].split(";")
        self.assertEqual(len(pins), 4)
        self.assertTrue(all(pin.endswith(",1") for pin in pins))
        self.assertEqual(extras["pm_node_spacing"], 0.12)
        self.assertEqual(extras["pm_shape_matching_stiffness"], 1.0)
        self.assertEqual(extras["pm_solver_iterations"], 64)
        self.assertEqual(extras["pm_maximum_projection_fraction"], 0.5)
        self.assertEqual(extras["pm_stretch_compliance"], 0.0)
        obj = next(obj for obj in bpy.context.scene.objects
                   if any(m.type == "SOFT_BODY" for m in obj.modifiers))
        # Explicit material recovery can coexist with exact Goal pins, but a
        # Goal group alone must not impose whole-body shape matching.
        del obj["pm_shape_matching_stiffness"]
        without_override = self.check_export(Counter(rigid_body=2, soft_body=1,
                                                     fluid_inflow=1, fluid_outflow=1))
        default_extras = next(node["extras"] for node in without_override["nodes"]
                              if node["extras"].get("pm_system") == "soft_body")
        self.assertEqual(default_extras["pm_shape_matching_stiffness"], 0.0)
        settings = next(m.settings for m in obj.modifiers if m.type == "SOFT_BODY")
        from mathutils import Matrix
        # Min/max remapping, partial weights, disabled Goal, and missing groups.
        obj.vertex_groups["Goal"].add([0], 0.5, "REPLACE")
        self.assertEqual(len(exporter.soft_body_goal_pins(obj, settings, Matrix.Identity(4))), 4)
        settings.goal_max = 0.8
        self.assertEqual(exporter.soft_body_goal_pins(obj, settings, Matrix.Identity(4)), [])
        settings.use_goal = False
        self.assertEqual(exporter.soft_body_goal_pins(obj, settings, Matrix.Identity(4)), [])

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
        with self.assertRaisesRegex(RuntimeError, "no supported physics objects"):
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


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ExportSceneTests))
    if not result.wasSuccessful():
        raise RuntimeError("ParallelMater Blender export tests failed")
