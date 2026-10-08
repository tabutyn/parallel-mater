#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create the authored Smoke Flow / passive-sphere gallery scene."""

from pathlib import Path

import bpy


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=3, radius=0.48, location=(0.0, 0.0, 1.5)
    )
    sphere = bpy.context.object
    sphere.name = "VortexSphere"
    bpy.ops.rigidbody.object_add()
    sphere.rigid_body.type = "PASSIVE"
    material = bpy.data.materials.new("Warm sphere")
    material.diffuse_color = (0.88, 0.42, 0.16, 1.0)
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (
        0.88, 0.42, 0.16, 1.0
    )
    sphere.data.materials.append(material)

    mesh = bpy.data.meshes.new("Smoke inlet plane")
    mesh.from_pydata(
        [(0, -0.24, -0.24), (0, 0.24, -0.24),
         (0, 0.24, 0.24), (0, -0.24, 0.24)],
        [], [(0, 1, 2, 3)],
    )
    mesh.update()
    emitter = bpy.data.objects.new("SmokeFlow", mesh)
    bpy.context.scene.collection.objects.link(emitter)
    emitter.location = (-2.0, 0.0, 1.5)
    bpy.ops.object.select_all(action="DESELECT")
    emitter.select_set(True)
    bpy.context.view_layer.objects.active = emitter
    flow = emitter.modifiers.new("Smoke Flow", "FLUID")
    flow.fluid_type = "FLOW"
    flow.flow_settings.flow_type = "SMOKE"
    flow.flow_settings.flow_behavior = "INFLOW"
    flow.flow_settings.use_initial_velocity = True
    flow.flow_settings.velocity_coord = (1.6, 0.0, 0.0)
    emitter["pm_smoke_obstacle"] = sphere.name
    emitter["pm_smoke_capacity"] = 4500
    emitter["pm_smoke_rate"] = 900.0
    emitter["pm_smoke_lifetime"] = 5.0
    emitter["pm_smoke_radius"] = 0.085
    emitter["pm_smoke_buoyancy"] = 0.12
    emitter["pm_smoke_wind_response"] = 0.5
    emitter["pm_smoke_rest_number_density"] = 12.0
    emitter["pm_smoke_pressure_stiffness"] = 2.0
    emitter["pm_smoke_viscosity"] = 0.02
    emitter["pm_smoke_vorticity_confinement"] = 0.1
    emitter["pm_smoke_grid_resolution"] = 128
    emitter["pm_smoke_grid_vertical_resolution"] = 32
    emitter["pm_smoke_grid_pressure_iterations"] = 24
    emitter["pm_smoke_grid_kinematic_viscosity"] = 1.5e-5
    emitter["pm_smoke_grid_les_coefficient"] = 0.12
    emitter["pm_smoke_grid_pressure_tolerance"] = 1.0e-3

    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    bpy.context.scene.frame_set(1)
    output = Path(__file__).resolve().parents[1] / "Smoke.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
