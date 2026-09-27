"""Derive the tear and rigid-paint gallery sources from Cloth.blend.

Run with `blender --background --python tools/blender/make_cloth_variants.py`.
This never saves over Cloth.blend.
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
sheet["pm_break_strain"] = 0.96
sheet["pm_fracture_persistence_substeps"] = 16
sheet["pm_impact_break_impulse"] = 0.0010
sheet["pm_stretch_compliance"] = 0.0
sheet["pm_solver_iterations"] = 24
ball = next(obj for obj in bpy.data.objects
            if obj.rigid_body and obj.rigid_body.type == "ACTIVE")
ball.rigid_body.mass = 35.0
bpy.ops.wm.save_as_mainfile(filepath=str(ASSETS / "ClothTear.blend"),
                            compress=True)

open_source("Cloth.blend")
sheet = next(obj for obj in bpy.data.objects
             if any(mod.type == "CLOTH" for mod in obj.modifiers))
ball = next(obj for obj in bpy.data.objects
            if obj.rigid_body and obj.rigid_body.type == "ACTIVE")
ball.location.z += 0.5
ball.location.y += 1.75
ball["pm_initial_velocity"] = (0.0, 2.5, 0.0)
sheet["pm_paintable"] = True
sheet["pm_paint_resolution"] = 128
sheet["pm_paint_source"] = ball.name
sheet["pm_paint_brush_radius"] = 0.18
# A neutral dry cloth makes the blue contact paint visible in the gallery.
dry_cloth = bpy.data.materials.new("DryClothPaint")
dry_cloth.diffuse_color = (0.8, 0.82, 0.78, 1.0)
dry_cloth.use_nodes = True
dry_cloth.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0.8, 0.82, 0.78, 1.0)
if len(sheet.data.materials) == 0:
    sheet.data.materials.append(dry_cloth)
else:
    for index in range(len(sheet.data.materials)):
        sheet.data.materials[index] = dry_cloth
bpy.ops.wm.save_as_mainfile(filepath=str(ASSETS / "ClothPaint.blend"),
                            compress=True)
