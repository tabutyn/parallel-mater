#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single Blender interface for exporting ParallelMater schema-2 GLB scenes.

The source .blend is never modified. Evaluated mesh copies are triangulated,
object scale is baked into their vertices, and Blender ACTIVE/PASSIVE settings
are written as glTF extras. Install this file as an add-on, run it in Blender's
Scripting workspace, or use it with Blender's --background --python flags.
"""

import argparse
import pathlib
import sys

import bmesh
import bpy
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector

bl_info = {
    "name": "ParallelMater Scene",
    "author": "ParallelMater contributors",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "File > Export > ParallelMater Scene (.glb)",
    "description": "Export rigid, soft, cloth, liquid, smoke, and paint metadata",
    "category": "Import-Export",
}

SCHEMA_VERSION = 2


def arguments() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Output .glb (defaults beside the open .blend with the same stem)",
    )
    return parser.parse_args(argv)


def output_path(requested: str | pathlib.Path | None) -> pathlib.Path:
    if requested:
        output = pathlib.Path(bpy.path.abspath(str(requested))).expanduser().resolve()
    else:
        if not bpy.data.filepath:
            raise RuntimeError("save the .blend or pass --output before exporting")
        output = pathlib.Path(bpy.data.filepath).with_suffix(".glb").resolve()
    if output.suffix.lower() != ".glb":
        raise ValueError("ParallelMater scene output must have a .glb extension")
    return output


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
    exported["pm_schema"] = SCHEMA_VERSION
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
    if "pm_initial_velocity" in source:
        velocity = source["pm_initial_velocity"]
        if len(velocity) != 3:
            raise RuntimeError(f"{source.name}: pm_initial_velocity needs 3 components")
        exported["pm_initial_velocity_x"] = float(velocity[0])
        exported["pm_initial_velocity_y"] = float(velocity[2])
        exported["pm_initial_velocity_z"] = -float(velocity[1])
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
    exported["pm_schema"] = SCHEMA_VERSION
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
    smoke = flow.flow_type == "SMOKE" and flow.flow_behavior == "INFLOW"
    liquid = flow.flow_type == "LIQUID" and flow.flow_behavior in {
        "INFLOW", "OUTFLOW", "GEOMETRY"}
    if not (smoke or liquid):
        raise RuntimeError(
            f"{source.name}: use Liquid Inflow/Outflow/Geometry or Smoke Inflow"
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
    exported["pm_schema"] = SCHEMA_VERSION
    exported["pm_system"] = "smoke_emitter" if smoke else {
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
        exported["pm_source_spacing"] = float(source.get("pm_source_spacing", 0.0))
    if smoke:
        obstacle = source.get("pm_smoke_obstacle")
        if not isinstance(obstacle, str) or not obstacle:
            raise RuntimeError(f"{source.name}: pm_smoke_obstacle must name a rigid sphere")
        exported["pm_smoke_obstacle"] = obstacle
        for key, default in (
            ("pm_smoke_capacity", 4500),
            ("pm_smoke_rate", 900.0),
            ("pm_smoke_lifetime", 5.0),
            ("pm_smoke_radius", 0.085),
            ("pm_smoke_buoyancy", 0.12),
            ("pm_smoke_response", 6.0),
            ("pm_smoke_wake_strength", 4.0),
        ):
            exported[key] = source.get(key, default)
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
    if group is not None and abs(float(settings.pin_stiffness) - 1.0) > 1.0e-5:
        raise RuntimeError(f"{source.name}: only Cloth pin stiffness 1.0 is supported")

    mesh = source.data.copy()
    created_meshes.append(mesh)
    location, rotation, scale = source.matrix_world.decompose()
    scale_matrix = Matrix.Diagonal(Vector((scale.x, scale.y, scale.z, 1.0)))
    pins = []
    for vertex in source.data.vertices:
        weight = next((assignment.weight for assignment in vertex.groups
                       if group is not None and assignment.group == group.index), 0.0)
        if weight <= 0.0:
            continue
        position = scale_matrix @ vertex.co
        # glTF uses Y up: Blender (x,y,z) -> glTF (x,z,-y).
        pins.append(f"{position.x:.9g},{position.z:.9g},{-position.y:.9g},{weight:.9g}")
    if group is not None and not pins:
        raise RuntimeError(f"{source.name}: Cloth pin group is empty")
    subdivision_applied = False
    geometry = bmesh.new()
    geometry.from_mesh(mesh)
    for modifier in source.modifiers:
        if modifier.type == "CLOTH":
            break
        if modifier.type != "SUBSURF" or not modifier.show_viewport:
            continue
        if modifier.subdivision_type != "SIMPLE" or modifier.levels > 5:
            raise RuntimeError(
                f"{source.name}: pre-Cloth subdivision needs Simple, at most 5 levels"
            )
        if modifier.levels:
            subdivision_applied = True
            bmesh.ops.subdivide_edges(
                geometry, edges=list(geometry.edges),
                cuts=(1 << modifier.levels) - 1, use_grid_fill=True,
            )
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
    exported["pm_schema"] = SCHEMA_VERSION
    exported["pm_system"] = "cloth"
    exported["pm_weld_vertices"] = subdivision_applied
    exported["pm_name"] = source.name
    exported["pm_pin_group"] = group_name if group is not None else ""
    exported["pm_pin_vertices"] = ";".join(pins)
    exported["pm_pin_stiffness"] = float(settings.pin_stiffness)
    exported["pm_vertex_mass"] = float(source.get("pm_vertex_mass", 0.001))
    exported["pm_thickness"] = float(source.get("pm_thickness", 0.025))
    exported["pm_break_strain"] = float(source.get("pm_break_strain", 0.0))
    exported["pm_fracture_persistence_substeps"] = int(
        source.get("pm_fracture_persistence_substeps", 4))
    exported["pm_impact_break_impulse"] = float(
        source.get("pm_impact_break_impulse", 0.0))
    exported["pm_stretch_compliance"] = float(source.get("pm_stretch_compliance", 1.0e-6))
    exported["pm_solver_iterations"] = int(source.get("pm_solver_iterations", 8))
    exported["pm_velocity_damping"] = float(
        source.get("pm_velocity_damping", settings.air_damping))
    exported["pm_contact_friction"] = float(
        source.get("pm_contact_friction", 0.4))
    exported["pm_pressure_enabled"] = bool(settings.use_pressure)
    exported["pm_uniform_pressure"] = float(settings.uniform_pressure_force)
    exported["pm_pressure_scale"] = float(settings.pressure_factor)
    exported["pm_pressure_custom_volume"] = bool(settings.use_pressure_volume)
    exported["pm_pressure_target_volume"] = float(settings.target_volume)
    exported["pm_pressure_fluid_density"] = float(settings.fluid_density)
    exported["pm_contains_fluid"] = bool(source.get("pm_contains_fluid", False))
    exported["pm_paintable"] = bool(source.get("pm_paintable", False))
    exported["pm_paint_resolution"] = int(source.get("pm_paint_resolution", 512))
    exported["pm_paint_source"] = str(source.get("pm_paint_source", ""))
    exported["pm_paint_brush_radius"] = float(
        source.get("pm_paint_brush_radius", 0.15))
    if len(mesh.materials) == 0:
        material = fallback_material(index, False)
        created_materials.append(material)
        mesh.materials.append(material)
    return exported


def soft_body_goal_pins(source, settings, scale_matrix):
    """Full effective Goal weight is a fixed node, not rigid shape matching."""
    if not settings.use_goal or not settings.vertex_group_goal:
        return []
    group = source.vertex_groups.get(settings.vertex_group_goal)
    if group is None:
        raise RuntimeError(f"{source.name}: Soft Body Goal group is missing")
    pins = []
    for vertex in source.data.vertices:
        weight = next((g.weight for g in vertex.groups if g.group == group.index), 0.0)
        effective = settings.goal_min + weight * (settings.goal_max - settings.goal_min)
        if effective < 1.0 - 1.0e-6:
            continue
        position = scale_matrix @ vertex.co
        pins.append(f"{position.x:.9g},{position.z:.9g},{-position.y:.9g},1")
    return pins


def _rope_endpoint_targets(point, radius, cloths, rigid_bodies):
    """Infer only unambiguous cloth-vertex and passive-mesh endpoint joints."""
    matches = []
    tolerance = max(1.0e-4, radius * 0.05)
    for cloth in cloths:
        for vertex in cloth.data.vertices:
            distance = ((cloth.matrix_world @ vertex.co) - point).length
            if distance <= tolerance:
                matches.append(("cloth", cloth.name, distance))
                break
    for body in rigid_bodies:
        if body.rigid_body.type != "PASSIVE":
            continue
        inverse = body.matrix_world.inverted_safe()
        local = inverse @ point
        hit, nearest, _, _ = body.closest_point_on_mesh(local)
        if not hit:
            continue
        distance = ((body.matrix_world @ nearest) - point).length
        on_surface = distance <= tolerance
        # Count ray crossings in mesh-local space, so an endpoint inside a
        # closed post is a joint even when its center is far from the skin.
        direction = (inverse.to_3x3() @ Vector((0.593, 0.714, 0.365))).normalized()
        origin = local + direction * 1.0e-6
        crossings = 0
        for _ in range(128):
            crossed, location, _, _ = body.ray_cast(origin, direction)
            if not crossed:
                break
            crossings += 1
            origin = location + direction * 1.0e-5
        if on_surface or crossings % 2:
            matches.append(("body", body.name, distance))
    matches.sort(key=lambda match: match[2])
    if len(matches) > 1 and matches[1][2] <= max(
        tolerance, matches[0][2] * 1.5
    ):
        raise RuntimeError(
            f"rope endpoint intersects multiple attachment targets: {matches}"
        )
    return matches[0][:2] if matches else None


def copy_rope_for_export(source, collection, cloths, rigid_bodies):
    """Curve shape + Hook references only; rope physics/sampling lives in API."""
    if source.parent is not None or len(source.data.splines) != 1:
        raise RuntimeError(f"{source.name}: rope needs one scene-root open spline")
    spline = source.data.splines[0]
    controls = spline.bezier_points if spline.type == "BEZIER" else spline.points
    if spline.type not in {"BEZIER", "POLY"} or spline.use_cyclic_u or len(controls) < 2:
        raise RuntimeError(f"{source.name}: rope needs one open Bezier or Poly spline")
    hooks = [m for m in source.modifiers if m.type == "HOOK" and m.show_viewport]
    if spline.type == "POLY" and hooks:
        raise RuntimeError(f"{source.name}: Poly rope Hooks are not supported; use Bezier")
    anchors = [None, None]
    for hook in hooks:
        if hook.object is None:
            raise RuntimeError(f"{source.name}: Hook target must be a rigid or soft body")
        rigid = hook.object.rigid_body is not None
        soft = any(mod.type == "SOFT_BODY" for mod in hook.object.modifiers)
        if rigid == soft:
            raise RuntimeError(f"{source.name}: Hook target must be a rigid or soft body")
        if hook.strength != 1.0 or (hook.falloff_type != "NONE" and hook.falloff_radius != 0):
            raise RuntimeError(f"{source.name}: rope Hook needs strength 1 and no distance falloff")
        indices = set(hook.vertex_indices)
        ends = [end for end, control in enumerate((0, len(controls)-1)) if 3*control+1 in indices]
        if len(ends) != 1 or any(i//3 not in (0, len(controls)-1) for i in indices):
            raise RuntimeError(f"{source.name}: each Hook must select exactly one endpoint control point")
        if anchors[ends[0]] is not None:
            raise RuntimeError(f"{source.name}: duplicate endpoint Hook")
        anchors[ends[0]] = ("body" if rigid else "soft_body", hook.object.name)
    if spline.type == "POLY" and not hooks:
        world = [source.matrix_world @ point.co.xyz for point in controls]
    else:
        # evaluated.data.splines still exposes undeformed controls in Blender.
        # Evaluate a private, unbevelled copy to include native Hook bindings.
        temporary = source.copy()
        temporary.data = source.data.copy()
        curve = temporary.data
        collection.objects.link(temporary)
        try:
            for modifier in list(temporary.modifiers):
                if modifier.type == "SOFT_BODY":
                    temporary.modifiers.remove(modifier)
                elif modifier.type != "HOOK":
                    raise RuntimeError(f"{source.name}: rope supports Hook and Soft Body modifiers")
            curve.bevel_depth = 0
            curve.extrude = 0
            curve.resolution_u = max(64, curve.resolution_u)
            curve.splines[0].resolution_u = curve.resolution_u
            bpy.context.view_layer.update()
            evaluated = temporary.evaluated_get(bpy.context.evaluated_depsgraph_get())
            polyline = evaluated.to_mesh()
            try:
                adjacent = [[] for _ in polyline.vertices]
                for edge in polyline.edges:
                    a, b = edge.vertices
                    adjacent[a].append(b)
                    adjacent[b].append(a)
                ends = [i for i, links in enumerate(adjacent) if len(links) == 1]
                if len(ends) != 2 or any(len(links) not in (1, 2) for links in adjacent):
                    raise RuntimeError(f"{source.name}: evaluated rope must be one open polyline")
                world = []
                previous, current = -1, min(ends)
                while True:
                    world.append(evaluated.matrix_world @ polyline.vertices[current].co)
                    following = [index for index in adjacent[current] if index != previous]
                    if not following:
                        break
                    previous, current = current, following[0]
                if len(world) != len(polyline.vertices):
                    raise RuntimeError(f"{source.name}: disconnected rope geometry")
            finally:
                evaluated.to_mesh_clear()
        finally:
            bpy.data.objects.remove(temporary, do_unlink=True)
            if curve.users == 0:
                bpy.data.curves.remove(curve)
    radius = float(source.get("pm_rope_radius", source.data.bevel_depth or 0.01))
    for end, point in enumerate((world[0], world[-1])):
        if anchors[end] is None:
            anchors[end] = _rope_endpoint_targets(point, radius, cloths, rigid_bodies)
    exported = bpy.data.objects.new(source.name, None)
    collection.objects.link(exported)
    exported["pm_schema"] = SCHEMA_VERSION
    exported["pm_system"] = "rope"
    exported["pm_rope_points"] = ";".join(f"{p.x:.9g},{p.z:.9g},{-p.y:.9g}" for p in world)
    for end, label in enumerate(("first", "last")):
        anchor = anchors[end]
        exported[f"pm_rope_{label}_body"] = anchor[1] if anchor and anchor[0] == "body" else ""
        exported[f"pm_rope_{label}_soft_body"] = anchor[1] if anchor and anchor[0] == "soft_body" else ""
        exported[f"pm_rope_{label}_cloth"] = anchor[1] if anchor and anchor[0] == "cloth" else ""
    settings = next((m.settings for m in source.modifiers if m.type == "SOFT_BODY"), None)
    exported["pm_rope_mass"] = float(source.get("pm_rope_mass", settings.mass if settings else 0.1))
    exported["pm_rope_radius"] = radius
    exported["pm_rope_spacing"] = float(source.get("pm_rope_spacing", 2 * exported["pm_rope_radius"]))
    for name, default in (("pm_rope_compliance", 0.0), ("pm_rope_friction", 0.4),
                          ("pm_rope_damping", 0.1),
                          ("pm_rope_maximum_substep_timestep", 1.0 / 480),
                          ("pm_rope_iterations", 24)):
        exported[name] = float(source.get(name, default))
    return exported


def copy_soft_body_for_export(
    source: bpy.types.Object,
    index: int,
    collection: bpy.types.Collection,
    created_meshes: list[bpy.types.Mesh],
    created_materials: list[bpy.types.Material],
) -> bpy.types.Object:
    modifiers = [modifier for modifier in source.modifiers
                 if modifier.type == "SOFT_BODY"]
    if len(modifiers) != 1 or source.parent is not None or source.rigid_body is not None:
        raise RuntimeError(
            f"{source.name}: expected one scene-root Soft Body modifier")
    settings = modifiers[0].settings
    mesh = source.data.copy()
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
    if not mesh.vertices or not mesh.polygons:
        raise RuntimeError(f"{source.name}: Soft Body requires a closed mesh")
    minimum = Vector(tuple(min(vertex.co[axis] for vertex in mesh.vertices)
                           for axis in range(3)))
    maximum = Vector(tuple(max(vertex.co[axis] for vertex in mesh.vertices)
                           for axis in range(3)))
    extent = maximum - minimum
    # Nine samples through the thinnest axis grossly oversamples slabs: tens
    # of thousands of nodes whose support cannot propagate across one solve.
    # Keep at least three layers, while targeting 18 along the longest axis.
    default_spacing = max(1.0e-4, min(extent) / 9.0,
                          min(max(extent) / 18.0, min(extent) / 3.0))

    exported = bpy.data.objects.new(source.name, mesh)
    collection.objects.link(exported)
    exported.matrix_world = Matrix.LocRotScale(location, rotation, None)
    exported["pm_schema"] = SCHEMA_VERSION
    exported["pm_system"] = "soft_body"
    exported["pm_name"] = source.name
    exported["pm_total_mass"] = float(source.get("pm_total_mass", settings.mass))
    exported["pm_node_spacing"] = float(
        source.get("pm_node_spacing", default_spacing))
    exported["pm_node_radius"] = float(source.get(
        "pm_node_radius", exported["pm_node_spacing"] * 0.35))
    exported["pm_stretch_compliance"] = float(
        source.get("pm_stretch_compliance", 1.0e-7))
    exported["pm_velocity_damping"] = float(
        source.get("pm_velocity_damping", max(0.0, settings.damping)))
    exported["pm_spring_damping"] = float(
        source.get("pm_spring_damping", 0.85))
    exported["pm_contact_friction"] = float(
        source.get("pm_contact_friction", settings.friction))
    exported["pm_pin_group"] = settings.vertex_group_goal if settings.use_goal else ""
    exported["pm_pin_vertices"] = ";".join(
        soft_body_goal_pins(source, settings, scale_matrix))
    # A selected Goal group anchors authored vertices. It must not also impose
    # a whole-body rest-shape projection, which prevents a cantilever bending.
    default_shape_stiffness = (settings.goal_default * settings.goal_spring
                               if settings.use_goal and not settings.vertex_group_goal
                               else 0.0)
    exported["pm_shape_matching_stiffness"] = float(source.get(
        "pm_shape_matching_stiffness", default_shape_stiffness))
    exported["pm_maximum_projection_fraction"] = float(
        source.get("pm_maximum_projection_fraction", 0.20))
    exported["pm_constraint_velocity_response"] = float(
        source.get("pm_constraint_velocity_response", 0.70))
    exported["pm_maximum_speed"] = float(
        source.get("pm_maximum_speed", 2.0))
    exported["pm_solver_iterations"] = int(
        source.get("pm_solver_iterations", 16))
    if len(mesh.materials) == 0:
        material = fallback_material(index, False)
        created_materials.append(material)
        mesh.materials.append(material)
    return exported


def export_scene(filepath: str | pathlib.Path | None = None) -> pathlib.Path:
    """Export the current scene through the same path used by CLI and UI.

    Returns the written path. No source objects, materials, selection, active
    object, or .blend file are changed. Object mode is required.
    """
    output = output_path(filepath)
    if bpy.context.mode != "OBJECT":
        raise RuntimeError("switch to Object Mode before exporting a ParallelMater scene")
    sources = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.rigid_body is not None
    ]
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
    soft_bodies = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and any(mod.type == "SOFT_BODY" for mod in obj.modifiers)
    ]
    ropes = [obj for obj in bpy.context.scene.objects if obj.type == "CURVE" and
             any(m.type in {"HOOK", "SOFT_BODY"} for m in obj.modifiers)]
    if not (sources or flows or cloths or soft_bodies or ropes):
        raise RuntimeError(
            "the scene contains no rigid bodies, soft bodies, cloth, or liquid flows")

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
        for index, source in enumerate(soft_bodies):
            created_objects.append(copy_soft_body_for_export(
                source, index, collection, created_meshes, created_materials
            ))
        for source in ropes:
            created_objects.append(copy_rope_for_export(
                source, collection, cloths, sources))

        bpy.ops.object.select_all(action="DESELECT")
        for obj in created_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = created_objects[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        result = bpy.ops.export_scene.gltf(
            filepath=str(output),
            export_format="GLB",
            export_extras=True,
            use_selection=True,
            export_cameras=False,
            export_lights=False,
            export_yup=True,
        )
        if "FINISHED" not in result:
            raise RuntimeError("Blender glTF export did not finish")
        print(f"ParallelMater scene exported to {output}")
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        # Helpers may fail after linking an object but before returning it.
        # The temporary collection is the authoritative cleanup list.
        for obj in list(collection.objects):
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
        bpy.context.view_layer.objects.active = previous_active
    return output


class EXPORT_SCENE_OT_parallel_mater(bpy.types.Operator, ExportHelper):
    """Export the current scene for ParallelMater without changing its source"""

    bl_idname = "export_scene.parallel_mater"
    bl_label = "Export ParallelMater Scene"
    filename_ext = ".glb"
    filter_glob: bpy.props.StringProperty(default="*.glb", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        try:
            output = export_scene(self.filepath)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {output.name}")
        return {"FINISHED"}


def export_menu(self, context):
    self.layout.operator(EXPORT_SCENE_OT_parallel_mater.bl_idname,
                         text="ParallelMater Scene (.glb)")


def register() -> None:
    bpy.utils.register_class(EXPORT_SCENE_OT_parallel_mater)
    bpy.types.TOPBAR_MT_file_export.append(export_menu)


def unregister() -> None:
    bpy.types.TOPBAR_MT_file_export.remove(export_menu)
    bpy.utils.unregister_class(EXPORT_SCENE_OT_parallel_mater)


def main() -> None:
    options = arguments()
    export_scene(options.output)


if __name__ == "__main__":
    if bpy.app.background:
        main()
    else:
        register()
