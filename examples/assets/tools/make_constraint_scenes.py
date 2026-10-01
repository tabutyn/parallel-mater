#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build Blender-authored rigid-constraint gallery scenes."""

import math
import pathlib

import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ConstraintWorld")
    scene.world.color = (0.025, 0.035, 0.055)
    scene.gravity = (0.0, 0.0, -9.81)


def material(name: str, color: tuple[float, float, float, float]):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = 0.15
    shader.inputs["Roughness"].default_value = 0.36
    return value


def finish_object(obj, name: str, color, rigid: str, mass: float = 1.0):
    obj.name = name
    obj.data.name = f"{name}Mesh"
    obj.data.materials.append(material(f"{name}Material", color))
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = rigid
    obj.rigid_body.mass = mass
    obj.rigid_body.friction = 0.8
    obj.rigid_body.restitution = 0.08
    obj.rigid_body.linear_damping = 0.08
    obj.rigid_body.angular_damping = 0.12
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.006
    obj["pm_checkerboard"] = rigid == "PASSIVE"
    obj.select_set(False)
    return obj


def box(name: str, location, scale, color, rigid="ACTIVE", mass=1.0):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.scale = scale
    return finish_object(obj, name, color, rigid, mass)


def sphere(name: str, location, radius=0.45, color=(0.92, 0.20, 0.10, 1.0),
           mass=1.0, velocity=None):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius,
                                         location=location)
    obj = finish_object(bpy.context.object, name, color, "ACTIVE", mass)
    if velocity is not None:
        obj["pm_initial_velocity"] = velocity
    return obj


def cylinder(name: str, location, radius, depth, color, rigid="ACTIVE",
             mass=1.0, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=depth,
                                       location=location, rotation=rotation)
    return finish_object(bpy.context.object, name, color, rigid, mass)


def add_floor(size=6.0):
    return box("Ground", (0.0, 0.0, -0.15), (size, size, 0.15),
               (0.16, 0.19, 0.24, 1.0), "PASSIVE")


def constraint(name: str, kind: str, first, second, location,
               rotation=(0.0, 0.0, 0.0), enabled=True):
    bpy.ops.object.empty_add(type="ARROWS", location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.empty_display_size = 0.45
    bpy.ops.rigidbody.constraint_add()
    settings = obj.rigid_body_constraint
    settings.type = kind
    settings.object1 = first
    settings.object2 = second
    settings.enabled = enabled
    settings.disable_collisions = True
    settings.use_override_solver_iterations = True
    settings.solver_iterations = 16
    return obj


def fixed_scene():
    reset()
    add_floor()
    left = sphere("FixedSphereA", (-2.2, 0.0, 0.5), color=(0.95, 0.26, 0.12, 1.0),
                  velocity=(2.2, 0.0, 0.0))
    right = sphere("FixedSphereB", (2.2, 0.0, 0.5), color=(0.12, 0.48, 0.95, 1.0),
                   velocity=(-2.2, 0.0, 0.0))
    constraint("FixedJoint", "FIXED", left, right, (0.0, 0.0, 0.5), enabled=False)


def point_scene():
    reset()
    add_floor()
    post = cylinder("PointPost", (0.0, 0.0, 1.25), 0.28, 2.5,
                    (0.70, 0.73, 0.78, 1.0), "PASSIVE")
    ball = sphere("PointSphereA", (-2.6, 0.0, 0.55),
                  color=(0.95, 0.45, 0.08, 1.0), velocity=(2.0, 0.0, 0.0))
    sphere("PointSphereB", (2.6, 0.0, 0.55),
           color=(0.12, 0.48, 0.95, 1.0), velocity=(-1.2, 0.0, 0.0))
    constraint("PointJoint", "POINT", post, ball, (-0.28, 0.0, 0.65), enabled=False)


def hinge_scene():
    reset()
    add_floor()
    frame = box("HingePost", (0.0, -1.25, 1.1), (0.18, 0.18, 1.1),
                (0.65, 0.68, 0.72, 1.0), "PASSIVE")
    door = box("HingePanel", (0.0, 0.0, 1.1), (0.12, 1.25, 1.1),
               (0.16, 0.58, 0.86, 1.0), mass=4.0)
    joint = constraint("HingeJoint", "HINGE", frame, door,
                       (0.0, -1.25, 1.1))
    joint.rigid_body_constraint.use_limit_ang_z = True
    joint.rigid_body_constraint.limit_ang_z_lower = -math.radians(45.0)
    joint.rigid_body_constraint.limit_ang_z_upper = math.radians(45.0)
    sphere("HingeSphere", (-3.2, 0.0, 0.55), radius=0.5,
           color=(0.95, 0.30, 0.10, 1.0), mass=3.0,
           velocity=(4.5, 0.0, 0.0))


def slider_scene():
    reset()
    add_floor()
    rail = box("SliderRail", (0.0, 0.0, 0.22), (2.0, 0.16, 0.12),
               (0.58, 0.62, 0.68, 1.0), "PASSIVE")
    block = box("SliderBlock", (0.0, 0.0, 0.62), (0.65, 0.65, 0.40),
                (0.15, 0.68, 0.80, 1.0), mass=3.0)
    joint = constraint("SliderJoint", "SLIDER", rail, block, (0.0, 0.0, 0.62))
    joint.rigid_body_constraint.use_limit_lin_x = True
    joint.rigid_body_constraint.limit_lin_x_lower = -1.0
    joint.rigid_body_constraint.limit_lin_x_upper = 1.0
    sphere("SliderSphere", (-3.0, 0.0, 0.55), radius=0.5,
           color=(0.94, 0.30, 0.10, 1.0), mass=2.5,
           velocity=(4.5, 0.0, 0.0))


def piston_scene():
    reset()
    add_floor()
    pole = cylinder("PistonPole", (0.0, 0.0, 1.65), 0.20, 3.3,
                    (0.62, 0.66, 0.72, 1.0), "PASSIVE")
    flag = box("PistonFlag", (1.0, 0.0, 1.65), (1.0, 0.12, 0.55),
               (0.12, 0.62, 0.92, 1.0), mass=2.5)
    joint = constraint("PistonJoint", "PISTON", pole, flag,
                       (0.0, 0.0, 1.65), rotation=(0.0, -math.pi / 2.0, 0.0))
    joint.rigid_body_constraint.use_limit_lin_x = True
    joint.rigid_body_constraint.limit_lin_x_lower = -1.0
    joint.rigid_body_constraint.limit_lin_x_upper = 1.0
    sphere("PistonSphere", (1.1, -3.0, 1.55), radius=0.5,
           color=(0.95, 0.30, 0.10, 1.0), mass=3.0,
           velocity=(0.0, 4.5, 0.0))


def generic_scene(spring=False):
    reset()
    add_floor()
    fixed = box("GenericBlockA", (-0.8, 0.0, 0.7), (0.55, 0.55, 0.55),
                (0.55, 0.58, 0.64, 1.0), "PASSIVE")
    moving = box("GenericBlockB", (0.8, 0.0, 0.7), (0.55, 0.55, 0.55),
                 (0.16, 0.68, 0.82, 1.0), mass=3.0)
    kind = "GENERIC_SPRING" if spring else "GENERIC"
    joint = constraint("GenericSpringJoint" if spring else "GenericJoint",
                       kind, fixed, moving, (0.0, 0.0, 0.7))
    settings = joint.rigid_body_constraint
    for axis in "xyz":
        setattr(settings, f"use_limit_lin_{axis}", True)
        setattr(settings, f"limit_lin_{axis}_lower", -0.15)
        setattr(settings, f"limit_lin_{axis}_upper", 0.15)
        setattr(settings, f"use_limit_ang_{axis}", True)
        setattr(settings, f"limit_ang_{axis}_lower", -math.radians(10.0))
        setattr(settings, f"limit_ang_{axis}_upper", math.radians(10.0))
        if spring:
            setattr(settings, f"use_spring_{axis}", True)
            setattr(settings, f"spring_stiffness_{axis}", 80.0)
            setattr(settings, f"spring_damping_{axis}", 6.0)
            setattr(settings, f"use_spring_ang_{axis}", True)
            setattr(settings, f"spring_stiffness_ang_{axis}", 30.0)
            setattr(settings, f"spring_damping_ang_{axis}", 3.0)
    sphere("GenericSphere", (3.2, 0.0, 0.55), radius=0.5,
           color=(0.95, 0.30, 0.10, 1.0), mass=3.0,
           velocity=(-4.5, 0.0, 0.0))


def motor_scene():
    reset()
    add_floor(8.0)
    chassis = box("MotorChassis", (0.0, 0.0, 0.9), (1.5, 0.7, 0.28),
                  (0.10, 0.52, 0.82, 1.0), mass=8.0)
    for side, y in (("Left", 0.88), ("Right", -0.88)):
        for axle, x in (("Front", 1.05), ("Rear", -1.05)):
            name = f"MotorWheel{side}{axle}"
            wheel = cylinder(name, (x, y, 0.52), 0.48, 0.34,
                             (0.08, 0.09, 0.11, 1.0), mass=1.0,
                             rotation=(math.pi / 2.0, 0.0, 0.0))
            joint = constraint(f"MotorJoint{side}{axle}", "MOTOR",
                               chassis, wheel, (x, y, 0.52),
                               rotation=(0.0, 0.0, math.pi / 2.0))
            settings = joint.rigid_body_constraint
            settings.use_motor_ang = True
            settings.motor_ang_target_velocity = 0.0
            settings.motor_ang_max_impulse = 8.0


SCENES = {
    "ConstraintFixed": fixed_scene,
    "ConstraintPoint": point_scene,
    "ConstraintHinge": hinge_scene,
    "ConstraintSlider": slider_scene,
    "ConstraintPiston": piston_scene,
    "ConstraintGeneric": lambda: generic_scene(False),
    "ConstraintGenericSpring": lambda: generic_scene(True),
    "ConstraintMotor": motor_scene,
}


for name, build in SCENES.items():
    build()
    bpy.context.scene["pm_gravity_scale"] = 1.0
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / f"{name}.blend"))
    print(f"Saved {name}.blend")
