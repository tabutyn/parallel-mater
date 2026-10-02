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
                            # The loader turns each heated surface into a
                            # passive collision body as well as a heat plane.
                            str(expected["rigid_body"] + expected["thermal_surface"]),
                            str(expected["rigid_constraint"]),
                            str(expected["cloth"]),
                            str(expected["fluid_inflow"]), str(expected["fluid_outflow"]),
                            str(int(expected["fluid_initial_volume"] > 0)),
                            str(expected["soft_body"]), str(expected["rope"])],
                           check=True, timeout=60)
        return document

    def test_all_authored_scenes_share_exporter(self):
        for name in ("PassiveActive", "Fluid", "FluidRigid", "Pegs", "Cloth",
                     "ClothTear", "ClothPaint", "ClothWater", "Softbody",
                     "SoftbodyRigidBody", "SoftbodyCloth", "SoftbodyFluid", "Rope",
                     "RopeFluid", "RopeCloth", "Smoke", "SmokeWater",
                     "SmokeRope", "SmokeSoftbody", "SmokeCloth",
                     "ConstraintFixed", "ConstraintPoint", "ConstraintHinge",
                     "ConstraintSlider", "ConstraintPiston", "ConstraintGeneric",
                     "ConstraintGenericSpring", "ConstraintMotor"):
            with self.subTest(scene=name):
                source = ASSETS / f"{name}.blend"
                digest = hashlib.sha256(source.read_bytes()).digest()
                bpy.ops.wm.open_mainfile(filepath=str(source))
                expected = systems(read_glb(source.with_suffix(".glb")))
                self.check_export(expected)
                self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)

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
            "ConstraintPoint": ("point", 2),
            "ConstraintHinge": ("hinge", 2),
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
                    rigid_body=6 if kind == "motor" else
                               3 if kind == "fixed" else 4,
                    rigid_constraint=count))
                constraints = [node["extras"] for node in document["nodes"]
                               if node["extras"].get("pm_system") ==
                               "rigid_constraint"]
                self.assertEqual(len(constraints), count)
                self.assertTrue(all(item["pm_constraint_type"] == kind
                                    for item in constraints))
                self.assertTrue(all(item["pm_body_a"] and item["pm_body_b"]
                                    for item in constraints))
                if kind == "fixed":
                    self.assertFalse(constraints[0]["pm_enabled"])
                if kind == "point":
                    self.assertTrue(all(item["pm_enabled"]
                                        for item in constraints))
                    self.assertEqual({item["pm_body_a"] for item in constraints},
                                     {"PointPost"})
                    self.assertEqual({item["pm_body_b"] for item in constraints},
                                     {"PointSphereA", "PointSphereB"})
                if kind == "hinge":
                    self.assertEqual(
                        {(item["pm_body_a"], item["pm_body_b"])
                         for item in constraints},
                        {("Ground", "Gear"),
                         ("Ground", "Gear.001")})
                    self.assertTrue(all(item["pm_solver_iterations"] == 64
                                        for item in constraints))
                    for joint_name, gear_name in (
                            ("SmallGearHinge", "Gear"),
                            ("LargeGearHinge", "Gear.001")):
                        joint = bpy.context.scene.objects[joint_name]
                        gear = bpy.context.scene.objects[gear_name]
                        joint_z = joint.matrix_world.to_3x3().normalized().col[2]
                        gear_z = gear.matrix_world.to_3x3().normalized().col[2]
                        self.assertAlmostEqual(joint_z.dot(gear_z), 1.0,
                                               places=6)
                    self.assertEqual(
                        {node["extras"]["pm_source_name"]
                         for node in document["nodes"]
                         if node.get("extras", {}).get("pm_system") ==
                         "rigid_body"},
                        {"Ground", "HingeSphere", "Gear", "Gear.001"})
                    bodies = {
                        node["extras"]["pm_source_name"]: node["extras"]
                        for node in document["nodes"]
                        if node.get("extras", {}).get("pm_system") ==
                        "rigid_body"
                    }
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
                if kind in ("slider", "piston"):
                    self.assertEqual(constraints[0]["pm_limit_lin_x_lower"], -1.0)
                    self.assertEqual(constraints[0]["pm_limit_lin_x_upper"], 1.0)
                if kind == "generic_spring":
                    self.assertTrue(constraints[0]["pm_use_spring_x"])
                    self.assertTrue(constraints[0]["pm_use_spring_ang_z"])
                if kind == "motor":
                    self.assertTrue(all(item["pm_use_motor_ang"]
                                        for item in constraints))

    def test_hinge_gears_clear_through_tooth_cycle(self):
        from mathutils import Quaternion, Vector
        from mathutils.bvhtree import BVHTree

        bpy.ops.wm.open_mainfile(
            filepath=str(ASSETS / "ConstraintHinge.blend"))
        small = bpy.context.scene.objects["Gear"]
        large = bpy.context.scene.objects["Gear.001"]
        small_base = small.rotation_euler.to_quaternion()
        large_base = large.rotation_quaternion.copy()

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

        maximum_clearance = 0.0
        for half_degree in range(61):
            angle = half_degree * 0.5
            small.rotation_mode = "QUATERNION"
            small.rotation_quaternion = small_base @ Quaternion(
                Vector((0.0, 0.0, 1.0)), angle * 3.141592653589793 / 180.0)
            large.rotation_quaternion = large_base @ Quaternion(
                Vector((0.0, 0.0, 1.0)),
                -angle * 0.5 * 3.141592653589793 / 180.0)
            bpy.context.view_layer.update()
            small_vertices, small_tree = geometry(small)
            large_vertices, large_tree = geometry(large)
            self.assertEqual(small_tree.overlap(large_tree), [])
            clearance = min(
                min(small_tree.find_nearest(point)[3]
                    for point in large_vertices),
                min(large_tree.find_nearest(point)[3]
                    for point in small_vertices))
            maximum_clearance = max(maximum_clearance, clearance)

        combined_margin = (small.rigid_body.collision_margin +
                           large.rigid_body.collision_margin)
        self.assertLess(maximum_clearance, combined_margin * 1.10)

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
        self.assertFalse(sphere["pm_checkerboard"])

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
