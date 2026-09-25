"""Derive the tear and wet-cloth gallery sources from the authored scenes.

Run with `blender --background --python tools/blender/make_cloth_variants.py`.
This never saves over Cloth.blend or ClothFluid.blend.
"""

from pathlib import Path

import bpy


ASSETS = Path(__file__).resolve().parents[2] / "examples" / "assets"
bpy.context.preferences.filepaths.save_version = 0


def open_source(name: str) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(ASSETS / name))


open_source("Cloth.blend")
sheet = next(obj for obj in bpy.data.objects
             if any(mod.type == "CLOTH" for mod in obj.modifiers))
sheet["pm_tear_ratio"] = 1.11
sheet["pm_tear_requires_contact"] = True
sheet["pm_stretch_compliance"] = 0.0
sheet["pm_solver_iterations"] = 20
ball = next(obj for obj in bpy.data.objects
            if obj.rigid_body and obj.rigid_body.type == "ACTIVE")
ball.rigid_body.mass = 35.0
bpy.ops.wm.save_as_mainfile(filepath=str(ASSETS / "ClothTear.blend"),
                            compress=True)

open_source("ClothFluid.blend")
cloth = next(obj for obj in bpy.data.objects
             if any(mod.type == "CLOTH" for mod in obj.modifiers))
pins = cloth.vertex_groups.new(name="FixedVertices")
highest = max(vertex.co.z for vertex in cloth.data.vertices)
pin_indices = [vertex.index for vertex in cloth.data.vertices
               if vertex.co.z >= highest - 0.07]
if not pin_indices:
    raise RuntimeError("ClothFluid has no top vertices to pin")
pins.add(pin_indices, 1.0, "REPLACE")
cloth_modifier = next(mod for mod in cloth.modifiers if mod.type == "CLOTH")
cloth_modifier.settings.vertex_group_mass = pins.name
cloth_modifier.settings.pin_stiffness = 1.0
cloth["pm_paintable"] = True
cloth["pm_paint_resolution"] = 128

water = bpy.data.objects["Water"]
flow = water.modifiers.new(name="ParallelMater Geometry Flow", type="FLUID")
flow.fluid_type = "FLOW"
flow.flow_settings.flow_type = "LIQUID"
flow.flow_settings.flow_behavior = "GEOMETRY"
water["pm_particle_spacing"] = 0.075
bpy.ops.wm.save_as_mainfile(filepath=str(ASSETS / "ClothPaint.blend"),
                            compress=True)
