#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create the canonical ParallelMater rigid-body gallery scene in Blender."""

import argparse
import math
import pathlib
import sys

import bpy


def arguments() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .glb path")
    parser.add_argument("--blend-output", help="Optional editable .blend path")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                       bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def material(name: str, color: tuple[float, float, float, float]):
    result = bpy.data.materials.new(name)
    result.diffuse_color = color
    result.use_nodes = True
    principled = result.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.48
    return result


def rigid_metadata(obj, motion: str, collider: str, mass: float | None = None,
                   friction: float = 0.6, restitution: float = 0.05,
                   checkerboard: bool = False) -> None:
    obj["pm_schema"] = 1
    obj["pm_system"] = "rigid_body"
    obj["pm_motion"] = motion
    obj["pm_collider"] = collider
    obj["pm_friction"] = friction
    obj["pm_restitution"] = restitution
    obj["pm_checkerboard"] = checkerboard
    if mass is not None:
        obj["pm_mass"] = mass


def apply_scale(obj) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)


def create_capsule(name: str, radius: float, half_height: float,
                   segments: int = 32, hemisphere_rings: int = 8):
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    vertices.append((0.0, 0.0, -half_height - radius))
    rings: list[list[int]] = []
    for ring in range(1, hemisphere_rings + 1):
        angle = -math.pi / 2.0 + ring * math.pi / (2.0 * hemisphere_rings)
        ring_radius = radius * math.cos(angle)
        z = -half_height + radius * math.sin(angle)
        indices = []
        for segment in range(segments):
            azimuth = 2.0 * math.pi * segment / segments
            indices.append(len(vertices))
            vertices.append((ring_radius * math.cos(azimuth),
                             ring_radius * math.sin(azimuth), z))
        rings.append(indices)

    top_equator = []
    for segment in range(segments):
        azimuth = 2.0 * math.pi * segment / segments
        top_equator.append(len(vertices))
        vertices.append((radius * math.cos(azimuth),
                         radius * math.sin(azimuth), half_height))
    rings.append(top_equator)

    for ring in range(1, hemisphere_rings):
        angle = ring * math.pi / (2.0 * hemisphere_rings)
        ring_radius = radius * math.cos(angle)
        z = half_height + radius * math.sin(angle)
        indices = []
        for segment in range(segments):
            azimuth = 2.0 * math.pi * segment / segments
            indices.append(len(vertices))
            vertices.append((ring_radius * math.cos(azimuth),
                             ring_radius * math.sin(azimuth), z))
        rings.append(indices)

    top_pole = len(vertices)
    vertices.append((0.0, 0.0, half_height + radius))
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((0, rings[0][following], rings[0][segment]))
    for lower, upper in zip(rings, rings[1:]):
        for segment in range(segments):
            following = (segment + 1) % segments
            faces.append((lower[segment], lower[following],
                          upper[following], upper[segment]))
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((rings[-1][segment], rings[-1][following], top_pole))

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_scene() -> None:
    floor_material = material("CheckerFloor", (0.66, 0.68, 0.72, 1.0))
    red = material("SphereRed", (0.8, 0.12, 0.08, 1.0))
    blue = material("BoxBlue", (0.08, 0.3, 0.85, 1.0))
    green = material("CapsuleGreen", (0.1, 0.68, 0.3, 1.0))

    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "StaticPlane"
    floor.data.materials.append(floor_material)
    rigid_metadata(floor, "static", "plane", friction=0.85,
                   restitution=0.05, checkerboard=True)

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=0.7,
                                          location=(-2.4, 0.0, 3.4))
    sphere = bpy.context.object
    sphere.name = "DynamicSphere"
    sphere.data.materials.append(red)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    rigid_metadata(sphere, "dynamic", "sphere", mass=2.0,
                   friction=0.72, restitution=0.25)

    bpy.ops.mesh.primitive_cube_add(size=1.4, location=(0.0, 0.0, 4.6),
                                    rotation=(0.18, 0.22, 0.1))
    box = bpy.context.object
    box.name = "DynamicBox"
    box.data.materials.append(blue)
    rigid_metadata(box, "dynamic", "box", mass=3.0,
                   friction=0.62, restitution=0.12)

    capsule = create_capsule("DynamicCapsule", 0.5, 0.75)
    capsule.location = (2.5, 0.0, 5.5)
    capsule.rotation_euler = (0.0, 0.22, -0.12)
    capsule.data.materials.append(green)
    rigid_metadata(capsule, "dynamic", "capsule", mass=2.5,
                   friction=0.68, restitution=0.18)


def export_scene(output: pathlib.Path, blend_output: pathlib.Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_extras=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True,
    )
    if blend_output is not None:
        blend_output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_output))


def main() -> None:
    options = arguments()
    clear_scene()
    build_scene()
    export_scene(pathlib.Path(options.output).resolve(),
                 pathlib.Path(options.blend_output).resolve()
                 if options.blend_output else None)


if __name__ == "__main__":
    main()
