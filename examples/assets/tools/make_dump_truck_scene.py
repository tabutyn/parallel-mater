"""Build DumpTruck.blend from Motor + Spring's four-wheel suspension rig.

The bucket is a five-sided box. DumpLift is a fully constrained generic joint;
the gallery rotates its chassis-side frame smoothly from 0 to 100 degrees.
Export separately with tools/blender/export_parallel_mater_scene.py.
"""
from math import pi
from pathlib import Path

import bpy
from mathutils import Vector

ASSETS = Path(__file__).resolve().parents[1]
bpy.ops.wm.open_mainfile(filepath=str(ASSETS / "ConstraintMotorSpring.blend"))
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
scene.frame_set(1)
scene.render.fps = 60
scene.frame_end = 1200

# Keep only the vehicle rig and its ground. In-progress obstacles in the source
# scene are unrelated to this new level; the source file itself is never saved.
keep = {"MotorChassis", "Ground"}
for obj in list(scene.objects):
    if obj.rigid_body_constraint:
        keep.add(obj.name)
        keep.add(obj.rigid_body_constraint.object1.name)
        keep.add(obj.rigid_body_constraint.object2.name)
for obj in list(bpy.data.objects):
    if obj.name not in keep:
        bpy.data.objects.remove(obj, do_unlink=True)


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.65
    return mat


frame_mat = material("Truck frame", (0.19, 0.22, 0.27))
cab_mat = material("Truck cab", (0.90, 0.48, 0.07))
window_mat = material("Cab windows", (0.08, 0.22, 0.31))
bed_mat = material("Bucket", (0.43, 0.49, 0.55))


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def box(name, center, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign(obj, mat)
    return obj


def join(parts, name):
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    parts[0].name = name
    return parts[0]


def active(obj, mass, friction=0.6):
    bpy.context.view_layer.objects.active = obj
    if obj.rigid_body is None:
        bpy.ops.rigidbody.object_add()
    body = obj.rigid_body
    body.type = "ACTIVE"
    body.collision_shape = "MESH"
    body.mass = mass
    body.friction = friction
    body.restitution = 0.02
    body.linear_damping = body.angular_damping = 0.02
    body.use_margin = True
    body.collision_margin = 0.005


ground = bpy.data.objects["Ground"]
ground.dimensions = (30, 24, 0.3)
bpy.context.view_layer.objects.active = ground
ground.select_set(True)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
ground.select_set(False)

chassis = bpy.data.objects["MotorChassis"]
chassis.location = (0, 0, 1.0)
chassis.dimensions = (5.0, 2.15, 0.32)
bpy.context.view_layer.objects.active = chassis
chassis.select_set(True)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
chassis.select_set(False)
assign(chassis, frame_mat)
cab = box("Cab", (1.73, 0, 1.97), (1.42, 2.05, 1.62), cab_mat)
windshield = box("Windshield", (2.45, 0, 2.25), (0.018, 1.74, 0.65), window_mat)
left_window = box("Left window", (1.78, 1.035, 2.25), (0.99, 0.018, 0.65), window_mat)
right_window = box("Right window", (1.78, -1.035, 2.25), (0.99, 0.018, 0.65), window_mat)
chassis = join([chassis, cab, windshield, left_window, right_window], "MotorChassis")
active(chassis, 50)

for joint in [o for o in scene.objects if o.rigid_body_constraint and o.rigid_body_constraint.type == "MOTOR"]:
    constraint = joint.rigid_body_constraint
    front = "Front" in joint.name
    left = "Left" in joint.name
    x, y = (1.78 if front else -1.80), (1.47 if left else -1.47)
    wheel, hub = constraint.object2, constraint.object1
    wheel.location = (x, y, 0.64)
    wheel.scale *= 1.25
    active(wheel, 2.0, 1.0)
    hub.location = (x, 0.99 if left else -0.99, 0.64)
    active(hub, 1.0)
    joint.location = wheel.location
    constraint.motor_ang_max_impulse = 24.0
    constraint.motor_ang_target_velocity = 0.0
    constraint.use_override_solver_iterations = True
    constraint.solver_iterations = 24
    spring = next(o for o in scene.objects if o.rigid_body_constraint and
                  o.rigid_body_constraint.type == "GENERIC_SPRING" and
                  o.rigid_body_constraint.object2 == hub)
    spring.location = hub.location
    suspension = spring.rigid_body_constraint
    suspension.limit_lin_z_lower = -0.18
    suspension.limit_lin_z_upper = 0.18
    suspension.spring_stiffness_z = 5000
    suspension.spring_damping_z = 80
    suspension.use_override_solver_iterations = True
    suspension.solver_iterations = 24

# Bed extends behind the cab. Exactly five solid walls, no top or tailgate.
bucket = join([
    box("Bucket floor", (-0.85, 0, 1.32), (3.10, 2.60, 0.14), bed_mat),
    box("Bucket rear", (-2.33, 0, 2.61), (0.14, 2.60, 2.44), bed_mat),
    box("Bucket front", (0.63, 0, 2.61), (0.14, 2.60, 2.44), bed_mat),
    box("Bucket left", (-0.85, 1.23, 2.61), (2.82, 0.14, 2.44), bed_mat),
    box("Bucket right", (-0.85, -1.23, 2.61), (2.82, 0.14, 2.44), bed_mat),
], "DumpBucket")
# At 100 degrees the rear wall slopes only ten degrees toward its open rim.
# A low-friction liner lets the final spheres slide out under gravity.
active(bucket, 10, 0.02)
bpy.ops.object.empty_add(type="ARROWS", location=(-2.4, 0, 1.32), rotation=(pi/2, 0, 0))
lift = bpy.context.object
lift.name = "DumpLift"
bpy.ops.rigidbody.constraint_add()
joint = lift.rigid_body_constraint
joint.type = "GENERIC"
joint.object1 = chassis
joint.object2 = bucket
joint.disable_collisions = True
joint.use_override_solver_iterations = True
joint.solver_iterations = 32
for axis in "xyz":
    setattr(joint, "use_limit_lin_" + axis, True)
    setattr(joint, "limit_lin_" + axis + "_lower", 0)
    setattr(joint, "limit_lin_" + axis + "_upper", 0)
    setattr(joint, "use_limit_ang_" + axis, True)
    setattr(joint, "limit_ang_" + axis + "_lower", 0)
    setattr(joint, "limit_ang_" + axis + "_upper", 0)

load_volume = box("DumpLoadVolume", (-0.85, 0, 2.60), (2.80, 2.28, 2.36), bed_mat)
load_volume["pm_hit_box"] = True
load_volume.hide_render = True
load_volume.hide_set(True)

# Runtime P/count controls center a grid on this Empty, independent of any
# load-volume helper. The shared exporter supplies the sphere mesh template.
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(-0.85, 0, 2.80))
bpy.context.object.name = "SphereCluster"

scene.rigidbody_world.substeps_per_frame = 8
scene.rigidbody_world.solver_iterations = 32
scene.rigidbody_world.point_cache.frame_end = scene.frame_end
bpy.ops.object.camera_add(location=(8, -10, 16))
camera = bpy.context.object
camera.rotation_euler = (Vector((0, 0, 1.6))-camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 48
scene.camera = camera
for name, location, energy in (("Key", (2, -5, 10), 1800), ("Fill", (-4, 5, 7), 1000)):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.size = 7
    light.rotation_euler = (Vector((0, 0, 1))-light.location).to_track_quat("-Z", "Y").to_euler()
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
            area.spaces.active.shading.color_type = "MATERIAL"
scene["Dump_controls"] = "Up/Down drive, Left/Right steer, Space toggles bucket between upright and 100 degrees; P edits payload count; R resets."
bpy.ops.wm.save_as_mainfile(filepath=str(ASSETS / "DumpTruck.blend"), compress=True)
