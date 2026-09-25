#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export Blender rigid bodies, cloth, and liquid flow as ParallelMater GLB.

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


def rigid_metadata(
    source: bpy.types.Object,
    exported: bpy.types.Object,
    collision_proxy_name: str | None,
) -> None:
    rigid = source.rigid_body
    passive = rigid.type == "PASSIVE"
    motion = "static" if passive else "dynamic"
    if not passive and getattr(rigid, "kinematic", False):
        motion = "kinematic"
    exported["pm_schema"] = 2
    exported["pm_system"] = "rigid_body"
    exported["pm_name"] = exported.name
    exported["pm_source_name"] = source.name
    exported["pm_motion"] = motion
    exported["pm_mass"] = float(rigid.mass)
    exported["pm_friction"] = float(rigid.friction)
    exported["pm_restitution"] = float(rigid.restitution)
    exported["pm_linear_damping"] = float(rigid.linear_damping)
    exported["pm_angular_damping"] = float(rigid.angular_damping)
    exported["pm_collision_margin"] = (
        float(rigid.collision_margin) if rigid.use_margin else 0.005
    )
    exported["pm_checkerboard"] = bool(source.get("pm_checkerboard", passive))
    exported["pm_paintable"] = bool(source.get("pm_paintable", False))
    if "pm_paint_resolution" in source:
        exported["pm_paint_resolution"] = float(source["pm_paint_resolution"])
    if collision_proxy_name is not None:
        exported["pm_collision_proxy"] = collision_proxy_name


def copy_for_export(
    source: bpy.types.Object,
    index: int,
    collision_proxy_name: str | None,
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

    rigid_metadata(source, exported, collision_proxy_name)

    if len(mesh.materials) == 0:
        material = fallback_material(index, source.rigid_body.type == "PASSIVE")
        created_materials.append(material)
        mesh.materials.append(material)
    return exported


def copy_array_rigid_for_export(
    source: bpy.types.Object,
    index: int,
    collection: bpy.types.Collection,
    depsgraph: bpy.types.Depsgraph,
    created_meshes: list[bpy.types.Mesh],
    created_materials: list[bpy.types.Material],
) -> list[bpy.types.Object]:
    """Turn disconnected Array copies into independently simulated bodies."""
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
    )
    created_meshes.append(mesh)
    parent = list(range(len(mesh.vertices)))

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for edge in mesh.edges:
        first, second = edge.vertices
        parent[root(second)] = root(first)
    components: dict[int, list[int]] = {}
    for vertex in range(len(mesh.vertices)):
        components.setdefault(root(vertex), []).append(vertex)
    groups = sorted(components.values(), key=lambda group: group[0])
    expected = 1
    for modifier in source.modifiers:
        if modifier.type == "ARRAY":
            expected *= modifier.count
    if len(groups) != expected or expected < 2:
        raise RuntimeError(
            f"{source.name}: Array rigid body needs {expected} disconnected "
            f"copies, found {len(groups)}"
        )

    centers = []
    for group in groups:
        coordinates = [mesh.vertices[vertex].co for vertex in group]
        centers.append((
            Vector(tuple(min(point[axis] for point in coordinates) for axis in range(3)))
            + Vector(tuple(max(point[axis] for point in coordinates) for axis in range(3)))
        ) * 0.5)
    first = groups[0]
    reference = [mesh.vertices[vertex].co - centers[0] for vertex in first]
    for group, center in zip(groups[1:], centers[1:]):
        if len(group) != len(first) or any(
            (mesh.vertices[vertex].co - center - shape).length > 1.0e-4
            for vertex, shape in zip(group, reference)
        ):
            raise RuntimeError(f"{source.name}: Array copies have different geometry")

    vertex_map = {vertex: local for local, vertex in enumerate(first)}
    faces = [polygon for polygon in mesh.polygons
             if polygon.vertices[0] in vertex_map]
    copy_mesh = bpy.data.meshes.new(f"{source.name}_ArrayBody")
    created_meshes.append(copy_mesh)
    copy_mesh.from_pydata(reference, [], [
        tuple(vertex_map[vertex] for vertex in polygon.vertices)
        for polygon in faces
    ])
    for material in mesh.materials:
        copy_mesh.materials.append(material)
    for face, polygon in zip(copy_mesh.polygons, faces):
        face.material_index = polygon.material_index
    if len(copy_mesh.materials) == 0:
        material = fallback_material(index, False)
        created_materials.append(material)
        copy_mesh.materials.append(material)
    _, rotation, scale = source.matrix_world.decompose()
    geometry = bmesh.new()
    geometry.from_mesh(copy_mesh)
    bmesh.ops.transform(
        geometry,
        matrix=Matrix.Diagonal(Vector((scale.x, scale.y, scale.z, 1.0))),
        verts=geometry.verts,
    )
    bmesh.ops.triangulate(geometry, faces=list(geometry.faces))
    geometry.to_mesh(copy_mesh)
    geometry.free()
    copy_mesh.validate(clean_customdata=False)
    copy_mesh.update()

    result = []
    for number, center in enumerate(centers):
        exported = bpy.data.objects.new(f"{source.name}_{number:03d}", copy_mesh)
        collection.objects.link(exported)
        exported.matrix_world = Matrix.LocRotScale(
            source.matrix_world @ center, rotation, None
        )
        rigid_metadata(source, exported, None)
        result.append(exported)
    return result


def copy_collision_for_export(
    source: bpy.types.Object,
    proxy: bpy.types.Object,
    exported_name: str,
    collection: bpy.types.Collection,
    depsgraph: bpy.types.Depsgraph,
    created_meshes: list[bpy.types.Mesh],
) -> bpy.types.Object:
    evaluated = proxy.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=False, depsgraph=depsgraph
    )
    created_meshes.append(mesh)

    # Store proxy vertices in the rigid body's local frame. This permits an
    # independently positioned Blender proxy while exporting both nodes with
    # exactly the same world-space rigid transform.
    location, rotation, _ = source.matrix_world.decompose()
    rigid_frame = Matrix.LocRotScale(location, rotation, None)
    relative = rigid_frame.inverted_safe() @ proxy.matrix_world
    geometry = bmesh.new()
    geometry.from_mesh(mesh)
    bmesh.ops.transform(geometry, matrix=relative, verts=geometry.verts)
    bmesh.ops.triangulate(geometry, faces=list(geometry.faces))
    geometry.normal_update()
    geometry.to_mesh(mesh)
    geometry.free()
    mesh.validate(clean_customdata=False)
    mesh.update()

    exported = bpy.data.objects.new(exported_name, mesh)
    collection.objects.link(exported)
    exported.matrix_world = Matrix.LocRotScale(location, rotation, None)
    exported["pm_schema"] = 2
    exported["pm_system"] = "collision_mesh"
    exported["pm_name"] = exported_name
    return exported


def copy_flow_for_export(
    source: bpy.types.Object,
    collection: bpy.types.Collection,
    created_meshes: list[bpy.types.Mesh],
) -> bpy.types.Object:
    fluid_modifiers = [m for m in source.modifiers if m.type == "FLUID"]
    if len(fluid_modifiers) != 1 or fluid_modifiers[0].fluid_type != "FLOW":
        raise RuntimeError(f"{source.name}: expected one Fluid Flow modifier")
    flow = fluid_modifiers[0].flow_settings
    if flow.flow_type != "LIQUID" or flow.flow_behavior not in {
        "INFLOW", "OUTFLOW", "GEOMETRY"
    }:
        raise RuntimeError(
            f"{source.name}: only Liquid Inflow/Outflow/Geometry is supported"
        )
    if source.parent is not None:
        raise RuntimeError(f"{source.name}: fluid flow plane must be a scene-root object")

    mesh = source.data.copy()
    created_meshes.append(mesh)
    location, rotation, scale = source.matrix_world.decompose()
    geometry = bmesh.new()
    geometry.from_mesh(mesh)
    geometry.transform(Matrix.Diagonal(Vector((scale.x, scale.y, scale.z, 1.0))))
    bmesh.ops.triangulate(geometry, faces=list(geometry.faces))
    geometry.to_mesh(mesh)
    geometry.free()
    mesh.validate(clean_customdata=False)
    mesh.update()
    exported = bpy.data.objects.new(source.name, mesh)
    collection.objects.link(exported)
    exported.matrix_world = Matrix.LocRotScale(location, rotation, None)
    exported["pm_schema"] = 2
    exported["pm_system"] = {
        "INFLOW": "fluid_inflow",
        "OUTFLOW": "fluid_outflow",
        "GEOMETRY": "fluid_initial_volume",
    }[flow.flow_behavior]
    if flow.flow_behavior in {"INFLOW", "GEOMETRY"}:
        velocity = flow.velocity_coord if flow.use_initial_velocity else Vector((0, 0, 0))
        # glTF export changes Blender Z-up into Y-up.
        exported["pm_velocity_x"] = float(velocity.x)
        exported["pm_velocity_y"] = float(velocity.z)
        exported["pm_velocity_z"] = float(-velocity.y)
    if flow.flow_behavior == "GEOMETRY":
        for name in ("pm_particle_spacing", "pm_gravity_scale"):
            if name in source:
                exported[name] = float(source[name])
    if flow.flow_behavior == "INFLOW":
        exported["pm_particles_per_second"] = float(
            source.get("pm_particles_per_second", 2400.0)
        )
    return exported


def copy_cloth_for_export(
    source: bpy.types.Object,
    index: int,
    collection: bpy.types.Collection,
    created_meshes: list[bpy.types.Mesh],
    created_materials: list[bpy.types.Material],
) -> bpy.types.Object:
    modifiers = [modifier for modifier in source.modifiers if modifier.type == "CLOTH"]
    if len(modifiers) != 1 or source.parent is not None or source.rigid_body is not None:
        raise RuntimeError(f"{source.name}: expected one scene-root Cloth modifier")
    settings = modifiers[0].settings
    group_name = settings.vertex_group_mass
    group = source.vertex_groups.get(group_name) if group_name else None
    if group is None:
        raise RuntimeError(f"{source.name}: Cloth Shape Pin Group is required")
    if abs(float(settings.pin_stiffness) - 1.0) > 1.0e-5:
        raise RuntimeError(f"{source.name}: only Cloth pin stiffness 1.0 is supported")

    mesh = source.data.copy()
    created_meshes.append(mesh)
    location, rotation, scale = source.matrix_world.decompose()
    scale_matrix = Matrix.Diagonal(Vector((scale.x, scale.y, scale.z, 1.0)))
    pins = []
    for vertex in source.data.vertices:
        weight = next((assignment.weight for assignment in vertex.groups
                       if assignment.group == group.index), 0.0)
        if weight <= 0.0:
            continue
        position = scale_matrix @ vertex.co
        # glTF uses Y up: Blender (x,y,z) -> glTF (x,z,-y).
        pins.append(f"{position.x:.9g},{position.z:.9g},{-position.y:.9g},{weight:.9g}")
    if not pins:
        raise RuntimeError(f"{source.name}: Cloth pin group is empty")
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
    exported["pm_schema"] = 2
    exported["pm_system"] = "cloth"
    exported["pm_name"] = source.name
    exported["pm_pin_group"] = group_name
    exported["pm_pin_vertices"] = ";".join(pins)
    exported["pm_pin_stiffness"] = float(settings.pin_stiffness)
    exported["pm_vertex_mass"] = float(source.get("pm_vertex_mass", 0.001))
    exported["pm_thickness"] = float(source.get("pm_thickness", 0.025))
    exported["pm_tear_ratio"] = float(source.get("pm_tear_ratio", 0.0))
    exported["pm_tear_requires_contact"] = bool(
        source.get("pm_tear_requires_contact", False))
    exported["pm_stretch_compliance"] = float(source.get("pm_stretch_compliance", 1.0e-6))
    exported["pm_solver_iterations"] = int(source.get("pm_solver_iterations", 8))
    exported["pm_paintable"] = bool(source.get("pm_paintable", False))
    exported["pm_paint_resolution"] = int(source.get("pm_paint_resolution", 512))
    exported["pm_paint_source"] = str(source.get("pm_paint_source", ""))
    if len(mesh.materials) == 0:
        material = fallback_material(index, False)
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
    flows = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and any(mod.type == "FLUID" for mod in obj.modifiers)
    ]
    cloths = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and any(mod.type == "CLOTH" for mod in obj.modifiers)
    ]

    previous_selection = list(bpy.context.selected_objects)
    previous_active = bpy.context.view_layer.objects.active
    collection = bpy.data.collections.new("ParallelMaterExportTemporary")
    bpy.context.scene.collection.children.link(collection)
    created_objects: list[bpy.types.Object] = []
    created_meshes: list[bpy.types.Mesh] = []
    created_materials: list[bpy.types.Material] = []
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        used_proxies: set[str] = set()
        for index, source in enumerate(sources):
            if source.rigid_body.type == "ACTIVE" and any(
                modifier.type == "ARRAY" for modifier in source.modifiers
            ):
                if source.get("pm_collision_proxy") is not None:
                    raise RuntimeError(
                        f"{source.name}: Array bodies cannot share a collision proxy"
                    )
                created_objects.extend(copy_array_rigid_for_export(
                    source, index, collection, depsgraph,
                    created_meshes, created_materials,
                ))
                continue
            proxy = None
            proxy_export_name = None
            requested_proxy = source.get("pm_collision_proxy")
            if requested_proxy is not None:
                if not isinstance(requested_proxy, str) or not requested_proxy:
                    raise RuntimeError(
                        f"{source.name}: pm_collision_proxy must be an object name"
                    )
                proxy = bpy.data.objects.get(requested_proxy)
                if proxy is None or proxy.type != "MESH":
                    raise RuntimeError(
                        f"{source.name}: collision proxy '{requested_proxy}' "
                        "is not a mesh object"
                    )
                if proxy.rigid_body is not None:
                    raise RuntimeError(
                        f"{source.name}: collision proxy must not be a rigid body"
                    )
                if proxy.name in used_proxies:
                    raise RuntimeError(
                        f"{source.name}: collision proxy is already in use"
                    )
                used_proxies.add(proxy.name)
                proxy_export_name = f"{source.name}__PM_COLLISION"
            created_objects.append(
                copy_for_export(
                    source,
                    index,
                    proxy_export_name,
                    collection,
                    depsgraph,
                    created_meshes,
                    created_materials,
                )
            )
            if proxy is not None and proxy_export_name is not None:
                created_objects.append(
                    copy_collision_for_export(
                        source,
                        proxy,
                        proxy_export_name,
                        collection,
                        depsgraph,
                        created_meshes,
                    )
                )
        for source in flows:
            created_objects.append(
                copy_flow_for_export(source, collection, created_meshes)
            )
        for index, source in enumerate(cloths):
            created_objects.append(copy_cloth_for_export(
                source, index, collection, created_meshes, created_materials
            ))

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
        if (
            previous_active is not None
            and previous_active.name in bpy.context.scene.objects
        ):
            bpy.context.view_layer.objects.active = previous_active


def main() -> None:
    options = arguments()
    export(output_path(options.output))


if __name__ == "__main__":
    main()
