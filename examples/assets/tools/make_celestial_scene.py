"""Author Celestial.blend; export separately through the shared scene exporter.

Run: blender --background --python examples/assets/tools/make_celestial_scene.py
Four unmotorized point joints carry plain spheres on different-length arms. Motion
comes from native rigid-body properties, initial velocity, gravity and contacts.
"""

from math import cos, sin, pi
from pathlib import Path

import bpy
from mathutils import Vector


ASSETS = Path(__file__).resolve().parents[1]
PIVOT = Vector((0, 0, 6.3))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.frame_end = 1200
scene.render.fps = 60


def material(name, color):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = 0.55
    return result


base_color = material("Base gray", (0.12, 0.15, 0.19))
arm_color = material("Arm gray", (0.65, 0.68, 0.72))
cyan = material("Cyan", (0.05, 0.68, 0.80))
red = material("Red", (0.85, 0.12, 0.07))
violet = material("Violet", (0.46, 0.16, 0.83))
green = material("Green", (0.14, 0.72, 0.37))


def finish(name, mat):
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def sphere(name, center, radius, mat, detail=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=detail, radius=radius,
                                        location=center)
    obj = finish(name, mat)
    if detail >= 3:
        for face in obj.data.polygons:
            face.use_smooth = True
    return obj


def rod(name, start, end, radius, mat, sides=16):
    start, end = Vector(start), Vector(end)
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(vertices=sides, radius=radius,
        depth=direction.length,
        location=(start + end) / 2)
    obj = finish(name, mat)
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    return obj


def join(parts, name):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def rigid(obj, active=False):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    rb = obj.rigid_body
    rb.type = "ACTIVE" if active else "PASSIVE"
    rb.collision_shape = "MESH"
    rb.mass = 2.0
    rb.friction = 0.35
    rb.restitution = 0.32
    rb.linear_damping = 0.004
    rb.angular_damping = 0.004
    rb.use_margin = True
    rb.collision_margin = 0.004


def proxy_for(obj, parts):
    proxy = join(parts, obj.name + "_Collision")
    obj["pm_collision_proxy"] = proxy.name
    proxy.hide_render = True
    proxy.hide_set(True)
    return proxy


# Plain base and post keep attention on the spheres and unequal arm lengths.
floor = rod("CelestialDais", (0, 0, -0.35), (0, 0, 0), 6.48, base_color, 96)
rigid(floor)
proxy_for(floor, [rod("Base collision", (0, 0, -0.35), (0, 0, 0), 6.48, base_color, 64)])

# All four constraints share this body's one world-space pivot.
support = [rod("Post", (0, 0, 0), PIVOT, 0.095, arm_color, 16),
           sphere("Pivot", PIVOT, 0.16, arm_color)]
support = join(support, "CelestialAnchor")
rigid(support)
proxy_for(support, [rod("Support collision", (0, 0, 0), PIVOT, 0.095, arm_color, 8)])

specs = [
    ("Azure", 2.0, 20, 0.85, cyan, 8.5),
    ("Forge", 2.8, 115, 1.10, red, -10.5),
    ("Amethyst", 3.6, 205, 0.75, violet, 12.0),
    ("Jade", 4.4, 295, 1.25, green, -14.0),
]
for name, radius, degrees, drop, color, speed in specs:
    angle = degrees*pi/180
    radial = Vector((cos(angle), sin(angle), 0))
    tangent = Vector((-sin(angle), cos(angle), 0))
    center = PIVOT + radial*radius - Vector((0, 0, drop))
    axis = (center - PIVOT).normalized()
    parts = [rod("Arm", PIVOT, center, 0.055, arm_color, 12),
             sphere(name, center, 0.48, color)]
    proxies = [rod("Arm collision", PIVOT + axis*0.52, center, 0.055, arm_color, 8),
               sphere("Sphere collision", center, 0.48, color, 2)]
    assembly = join(parts, "Celestial" + name)
    rigid(assembly, active=True)
    assembly["pm_initial_velocity"] = tuple(tangent*speed)
    proxy_for(assembly, proxies)
    bpy.ops.object.empty_add(type="SPHERE", radius=0.18, location=PIVOT)
    joint = bpy.context.object
    joint.name = "CelestialPoint" + name
    bpy.ops.rigidbody.constraint_add()
    constraint = joint.rigid_body_constraint
    constraint.type = "POINT"
    constraint.object1 = support
    constraint.object2 = assembly
    constraint.disable_collisions = True
    constraint.use_override_solver_iterations = True
    constraint.solver_iterations = 32

scene.rigidbody_world.substeps_per_frame = 4
scene.rigidbody_world.solver_iterations = 32
scene.rigidbody_world.point_cache.frame_end = scene.frame_end

# Useful editable Blender presentation; runtime uses its own camera and lights.
bpy.ops.object.camera_add(location=(10, -14, 10))
camera = bpy.context.object
camera.name = "CelestialPresentationCamera"
camera.rotation_euler = (Vector((0, 0, 3.1))-camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.type = "PERSP"
camera.data.lens = 43
scene.camera = camera
for name, location, power, size in (("Key", (1, -6, 12), 2300, 8),
                                     ("Rim", (-6, 3, 9), 1800, 6),
                                     ("Fill", (6, 5, 5), 1000, 5)):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = power
    light.data.shape = "DISK"
    light.data.size = size
    light.rotation_euler = (Vector((0, 0, 3))-light.location).to_track_quat("-Z", "Y").to_euler()
scene.world = bpy.data.worlds.new("Night studio")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.08, 0.10, 0.16, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.render.resolution_x = 1280
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "AgX"
scene["Celestial_notes"] = "Four rigid arms; four POINT constraints at one anchor. ParallelMater initial velocities are pm_initial_velocity; Blender playback does not apply these custom velocities. Export with the shared ParallelMater exporter."
bpy.ops.object.select_all(action="DESELECT")
support.select_set(True)
bpy.context.view_layer.objects.active = support
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
            area.spaces.active.shading.color_type = "MATERIAL"
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(ASSETS / "Celestial.blend"), compress=True)
print("Celestial: 4 point joints, 4 moving assemblies, separate source preserved.")
