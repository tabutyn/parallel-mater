#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export Blender rigid bodies as a triangle-only ParallelMater GLB scene.

The source .blend is never modified. Evaluated mesh copies are triangulated,
object scale is baked into their vertices, and Blender ACTIVE/PASSIVE settings
are written as glTF extras consumed by the gallery loader.
"""

import argparse
import pathlib
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Output .glb (defaults beside the open .blend with the same stem)",
    )
    return parser.parse_args(argv)


def output_path(requested: str | None) -> pathlib.Path:
    if requested:
        return pathlib.Path(requested).expanduser().resolve()
    if not bpy.data.filepath:
        raise RuntimeError("save the .blend or pass --output before exporting")
    return pathlib.Path(bpy.data.filepath).with_suffix(".glb").resolve()


def fallback_material(index: int, passive: bool) -> bpy.types.Material:
    palette = (
        (0.10, 0.36, 0.92, 1.0),
        (0.92, 0.24, 0.10, 1.0),
        (0.95, 0.58, 0.08, 1.0),
        (0.18, 0.72, 0.32, 1.0),
    )
    color = (0.68, 0.70, 0.74, 1.0) if passive else palette[index % len(palette)]
    material = bpy.data.materials.new(f"PMExportMaterial{index}")
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = 0.48
    return material


def copy_for_export(
    source: bpy.types.Object,
    index: int,
    collection: bpy.types.Collection,
    depsgraph: bpy.types.Depsgraph,
    created_meshes: list[bpy.types.Mesh],
    created_materials: list[bpy.types.Material],
) -> bpy.types.Object:
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
    )
    created_meshes.append(mesh)

    location, rotation, scale = source.matrix_world.decompose()
    scale_matrix = Matrix.Diagonal(Vector((scale.x, scale.y, scale.z, 1.0)))
    geometry = bmesh.new()
    geometry.from_mesh(mesh)
    bmesh.ops.transform(geometry, matrix=scale_matrix, verts=geometry.verts)
    bmesh.ops.triangulate(geometry, faces=list(geometry.faces))
    geometry.normal_update()
    geometry.to_mesh(mesh)
    geometry.free()
    mesh.validate(clean_customdata=False)
    mesh.update()

    exported = bpy.data.objects.new(source.name, mesh)
    collection.objects.link(exported)
    exported.matrix_world = Matrix.LocRotScale(location, rotation, None)

    rigid = source.rigid_body
    passive = rigid.type == "PASSIVE"
    motion = "static" if passive else "dynamic"
    if not passive and getattr(rigid, "kinematic", False):
        motion = "kinematic"
    exported["pm_schema"] = 2
    exported["pm_system"] = "rigid_body"
    exported["pm_name"] = source.name
    exported["pm_motion"] = motion
    exported["pm_mass"] = float(rigid.mass)
    exported["pm_friction"] = float(rigid.friction)
    exported["pm_restitution"] = float(rigid.restitution)
    exported["pm_linear_damping"] = float(rigid.linear_damping)
    exported["pm_angular_damping"] = float(rigid.angular_damping)
    exported["pm_collision_margin"] = (
        float(rigid.collision_margin) if rigid.use_margin else 0.005
    )
    exported["pm_checkerboard"] = bool(
        source.get("pm_checkerboard", passive)
    )

    if len(mesh.materials) == 0:
        material = fallback_material(index, passive)
        created_materials.append(material)
        mesh.materials.append(material)
    return exported


def export(output: pathlib.Path) -> None:
    sources = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.rigid_body is not None
    ]
    if not sources:
        raise RuntimeError("the scene contains no mesh objects with Rigid Body enabled")
    if any(obj.parent is not None for obj in sources):
        raise RuntimeError("rigid-body export currently requires scene-root objects")

    previous_selection = list(bpy.context.selected_objects)
    previous_active = bpy.context.view_layer.objects.active
    collection = bpy.data.collections.new("ParallelMaterExportTemporary")
    bpy.context.scene.collection.children.link(collection)
    created_objects: list[bpy.types.Object] = []
    created_meshes: list[bpy.types.Mesh] = []
    created_materials: list[bpy.types.Material] = []
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for index, source in enumerate(sources):
            created_objects.append(
                copy_for_export(
                    source,
                    index,
                    collection,
                    depsgraph,
                    created_meshes,
                    created_materials,
                )
            )

        bpy.ops.object.select_all(action="DESELECT")
        for obj in created_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = created_objects[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(
            filepath=str(output),
            export_format="GLB",
            export_extras=True,
            use_selection=True,
            export_cameras=False,
            export_lights=False,
            export_yup=True,
        )
        print(f"ParallelMater scene exported to {output}")
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in created_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)
        for mesh in created_meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for material in created_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        for obj in previous_selection:
            if obj.name in bpy.context.scene.objects:
                obj.select_set(True)
        if previous_active is not None and previous_active.name in bpy.context.scene.objects:
            bpy.context.view_layer.objects.active = previous_active


def main() -> None:
    options = arguments()
    export(output_path(options.output))


if __name__ == "__main__":
    main()
