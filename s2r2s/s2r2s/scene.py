"""MuJoCo scene construction for SO-101 planar pushing.

Scene = MuJoCo Menagerie SO-101 (unchanged kinematics, inertias and servo
models) + a cylindrical pusher rod held by the closed gripper + a table plane +
one free object + a visual-only goal marker + cameras.

Two builds share identical dynamics:
  * ``visual=True``  : full meshes, for the viewer and video rendering.
  * ``visual=False`` : meshes stripped, for fast batched training (each
    environment owns a small model copy so physical parameters can be
    randomised per episode).
Every robot body gets an explicit inertial taken from the compiled full
model, so stripping visual geoms cannot change masses or inertias.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math

import mujoco
import numpy as np

from .objects import OBJECTS, ObjectShape

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_SO101 = ROOT / "assets" / "menagerie" / "robotstudio_so101" / "so101.xml"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Collision bit masks: table / objects / pusher / static obstacles (gate walls).
_TABLE, _OBJECT, _PUSHER, _STATIC = 1, 2, 4, 8
# Gate walls: 10 mm thick, 25 mm tall (taller than the 20 mm objects, below the 35 mm jaw tips).
GATE_WALL_HALF_THICKNESS = 0.005
GATE_WALL_HALF_HEIGHT = 0.0125
GATE_WALL_LENGTH = (0.09, 0.11)     # inner / outer wall length from the opening edge (m)


@dataclass
class SceneConfig:
    timestep: float = 0.002
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 10.0
    iterations: int = 20
    ls_iterations: int = 20
    # Pusher rod in the gripper body frame (gripper -z points down when the tool is vertical).
    # Default: a 12 mm rod clamped against the fixed jaw, ending 30 mm below the jaw tip.
    pusher_radius: float = 0.006
    pusher_xy: tuple[float, float] = (-0.0014, 0.0)
    pusher_top_z: float = -0.075
    pusher_tip_z: float = -0.131
    gripper_hold: float = 0.10          # gripper joint target while holding the rod
    table_friction: float = 0.35        # object-table sliding friction (nominal)
    pusher_friction: float = 0.30       # object-pusher sliding friction (nominal)
    # Planned real camera (RealSense D435i colour stream: 69 x 42 deg FOV). Opposite the robot and fairly
    # low: the camera study (camera_study.py) found 14 % mean arm occlusion here vs 67 % straight overhead.
    camera_pos: tuple[float, float, float] = (0.62, 0.0, 0.38)
    camera_target: tuple[float, float, float] = (0.20, 0.0, 0.0)
    camera_fovy: float = 42.5
    # Level 1 puzzle: a wall with an opening across the work band (placed per episode, see set_gate).
    gate: bool = False
    gate_friction: float = 0.35

    def to_dict(self):
        return asdict(self)


def _look_at_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion for a MuJoCo camera at ``pos`` looking at ``target`` (camera looks along -z)."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    y = up - up.dot(z) * z
    if np.linalg.norm(y) < 1e-6:          # looking straight down: image up = -x (towards robot)
        y = np.array([-1.0, 0.0, 0.0]) - z * (-z[0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([x, y, z], axis=1).ravel())
    return quat


def _robot_inertials():
    """Compiled inertial properties of every robot body in the unmodified Menagerie model."""
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE_SO101))
    return {m.body(i).name: (float(m.body_mass[i]), m.body_ipos[i].copy(), m.body_iquat[i].copy(),
                             m.body_inertia[i].copy()) for i in range(1, m.nbody)}


_INERTIALS = None


def build_spec(cfg: SceneConfig | None = None, object_name: str = "tee", visual: bool = True) -> mujoco.MjSpec:
    global _INERTIALS
    cfg = cfg or SceneConfig()
    shape: ObjectShape = OBJECTS[object_name]
    if _INERTIALS is None:
        _INERTIALS = _robot_inertials()

    spec = mujoco.MjSpec.from_file(str(MENAGERIE_SO101))
    spec.modelname = f"so101_push_{object_name}"
    opt = spec.option
    opt.timestep = cfg.timestep
    opt.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                      "euler": mujoco.mjtIntegrator.mjINT_EULER,
                      "rk4": mujoco.mjtIntegrator.mjINT_RK4}[cfg.integrator]
    opt.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
                "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[cfg.cone]
    opt.impratio = cfg.impratio
    opt.iterations = cfg.iterations
    opt.ls_iterations = cfg.ls_iterations

    # Freeze robot inertials, then disable all robot collisions (the pusher is added below).
    for body in spec.bodies:
        if body.name in _INERTIALS:
            mass, ipos, iquat, inertia = _INERTIALS[body.name]
            body.explicitinertial = True
            body.mass, body.ipos, body.iquat, body.inertia = mass, ipos, iquat, inertia
            body.fullinertia = [math.nan] * 6
    for geom in list(spec.geoms):
        geom.contype = 0
        geom.conaffinity = 0
        if not visual and (geom.type == mujoco.mjtGeom.mjGEOM_MESH or geom.group >= 2):
            spec.delete(geom)
    if not visual:
        for mesh in list(spec.meshes):
            spec.delete(mesh)

    # Pusher rod and tool-tip site on the gripper (fixed-jaw) body.
    grip = spec.body("gripper")
    px, py = cfg.pusher_xy
    r = cfg.pusher_radius
    grip.add_geom(name="pusher", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, 0, 0],
                  fromto=[px, py, cfg.pusher_top_z, px, py, cfg.pusher_tip_z + r],
                  contype=_PUSHER, conaffinity=_TABLE | _STATIC, condim=3, priority=1,
                  friction=[cfg.pusher_friction, 0.005, 0.0001], mass=0, rgba=[0.15, 0.15, 0.15, 1], group=1)
    grip.add_site(name="tool_tip", pos=[px, py, cfg.pusher_tip_z], size=[0.002, 0, 0], rgba=[1, 0, 0, 1], group=3)

    world = spec.worldbody
    world.add_light(pos=[0.2, -0.3, 1.0], dir=[0, 0.3, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_light(pos=[0.4, 0.4, 0.8], dir=[-0.3, -0.4, -1], diffuse=[0.4, 0.4, 0.4], castshadow=False)
    # Table: an infinite collision plane plus a finite visual slab.
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.6, 0.6, 0.01],
                   contype=_TABLE, conaffinity=0, condim=3, priority=1,
                   friction=[cfg.table_friction, 0.005, 0.0001], rgba=[0, 0, 0, 0], group=3)
    if visual:
        tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                               rgb1=[0.82, 0.82, 0.80], rgb2=[0.76, 0.76, 0.74], width=512, height=512)
        spec.add_material(name="table_mat", textures=["", "grid"], texrepeat=[12, 12], reflectance=0.05)
        world.add_geom(name="table_visual", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.32, 0.40, 0.01],
                       pos=[0.18, 0, -0.0101], contype=0, conaffinity=0, material="table_mat", group=0)
        # Faint guide showing the reachable pushing annulus (r = 0.12 .. 0.28 m).
        for rad in (0.12, 0.28):
            n = 36
            for k in range(n):
                a0, a1 = math.radians(-60 + 120 * k / n), math.radians(-60 + 120 * (k + 1) / n)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0006, 0, 0],
                               fromto=[rad * math.cos(a0), rad * math.sin(a0), 0.0002,
                                       rad * math.cos(a1), rad * math.sin(a1), 0.0002],
                               contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.3, 0.35], group=0)

    # Free object, flat on the table; origin = footprint centroid = centre of mass.
    obj = world.add_body(name="object", pos=[0.2, 0, shape.height / 2])
    obj.simple = False      # its centre of mass is randomised at run time
    obj.add_freejoint(name="object_free")
    h = shape.height / 2
    if shape.disk_radius > 0:
        obj.add_geom(name="object_0", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[shape.disk_radius, h, 0],
                     mass=shape.mass, contype=_OBJECT, conaffinity=_TABLE | _OBJECT | _PUSHER | _STATIC, condim=3,
                     friction=[0.1, 0.005, 0.0001], rgba=shape.rgba)
    else:
        boxes = shape.centred_boxes
        areas = np.array([hx * hy for _, _, hx, hy in boxes])
        for i, (cx, cy, hx, hy) in enumerate(boxes):
            obj.add_geom(name=f"object_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, h], pos=[cx, cy, 0],
                         mass=float(shape.mass * areas[i] / areas.sum()), contype=_OBJECT,
                         conaffinity=_TABLE | _OBJECT | _PUSHER | _STATIC, condim=3,
                         friction=[0.1, 0.005, 0.0001], rgba=shape.rgba)
    obj.add_site(name="object_top", pos=[0, 0, h], size=[0.003, 0, 0], rgba=[1, 1, 1, 0], group=3)

    if cfg.gate:
        gate = world.add_body(name="gate", pos=[0.2, 0.0, 0.0])
        for name, length in zip(("gate_inner", "gate_outer"), GATE_WALL_LENGTH):
            gate.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2, GATE_WALL_HALF_THICKNESS, GATE_WALL_HALF_HEIGHT],
                          pos=[0, 0, GATE_WALL_HALF_HEIGHT], contype=_STATIC, conaffinity=0, condim=3, priority=1,
                          friction=[cfg.gate_friction, 0.005, 0.0001], rgba=[0.80, 0.55, 0.30, 1], group=0)

    # Goal marker: mocap body with a thin, non-colliding copy of the footprint.
    goal = world.add_body(name="goal", mocap=True, pos=[0.2, 0.05, 0.0])
    if shape.disk_radius > 0:
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[shape.disk_radius, 0.0004, 0],
                      pos=[0, 0, 0.0004], contype=0, conaffinity=0, rgba=[0.2, 0.85, 0.3, 0.45], group=0)
    else:
        for cx, cy, hx, hy in shape.centred_boxes:
            goal.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.0004], pos=[cx, cy, 0.0004],
                          contype=0, conaffinity=0, rgba=[0.2, 0.85, 0.3, 0.45], group=0)

    if visual:
        # The policy's camera estimate of the object pose (delayed + noisy), drawn as a thin outline.
        est = world.add_body(name="estimate", mocap=True, pos=[0.2, -0.05, 0.0])
        if shape.disk_radius > 0:
            est.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[shape.disk_radius, 0.0003, 0],
                         pos=[0, 0, shape.height + 0.002], contype=0, conaffinity=0,
                         rgba=[1.0, 0.85, 0.1, 0.35], group=1)
        else:
            for cx, cy, hx, hy in shape.centred_boxes:
                est.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.0003],
                             pos=[cx, cy, shape.height + 0.002], contype=0, conaffinity=0,
                             rgba=[1.0, 0.85, 0.1, 0.35], group=1)

    # Cameras: the planned real camera and two viewing cameras for videos.
    world.add_camera(name="d435i", pos=list(cfg.camera_pos),
                     quat=list(_look_at_quat(cfg.camera_pos, cfg.camera_target)), fovy=cfg.camera_fovy)
    world.add_camera(name="front", pos=[0.58, -0.30, 0.32], quat=list(_look_at_quat([0.58, -0.30, 0.32], [0.18, 0.0, 0.02])), fovy=45)
    world.add_camera(name="top", pos=[0.20, 0.0, 0.75], quat=list(_look_at_quat([0.20, 0.0, 0.75], [0.20, 0.0, 0.0])), fovy=40)
    if visual:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
    return spec


def build_model(cfg: SceneConfig | None = None, object_name: str = "tee", visual: bool = True) -> mujoco.MjModel:
    model = build_spec(cfg, object_name, visual).compile()
    # Single-geom objects compile with body frame == inertial frame, and MuJoCo then
    # ignores body_ipos. Clear the flag so centre-of-mass randomisation takes effect.
    model.body_sameframe[model.body("object").id] = 0
    return model


class SceneIndex:
    """Address book for a compiled scene (joint/qpos/qvel/actuator indices)."""

    def __init__(self, model: mujoco.MjModel):
        self.arm_qpos = np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS])
        self.arm_qvel = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_JOINTS])
        self.gripper_qpos = model.jnt_qposadr[model.joint("gripper").id]
        free = model.joint("object_free").id
        self.obj_qpos = model.jnt_qposadr[free]       # 7: pos(3), quat(4)
        self.obj_qvel = model.jnt_dofadr[free]        # 6: linvel(3), angvel(3)
        self.arm_act = np.array([model.actuator(n).id for n in ARM_JOINTS])
        self.gripper_act = model.actuator("gripper").id
        self.object_body = model.body("object").id
        self.goal_mocap = model.body_mocapid[model.body("goal").id]
        self.tool_site = model.site("tool_tip").id
        self.pusher_geom = model.geom("pusher").id
        self.table_geom = model.geom("table").id
        self.object_geoms = np.array([i for i in range(model.ngeom) if model.geom(i).name.startswith("object_")])
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nu


def set_gate(model: mujoco.MjModel, centre, angle: float, width: float):
    """Place the gate (in a model copy): opening centre (x, y), wall direction ``angle`` (rad), opening width (m).

    The walls run along the gate's local x axis on either side of the opening; the passage is local y.
    Only positions change (sizes are fixed at build time), so MuJoCo's cached bounding volumes stay valid;
    world poses of static bodies are recomputed every step.
    """
    b = model.body("gate").id
    model.body_pos[b] = [centre[0], centre[1], 0.0]
    model.body_quat[b] = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    for name, sign, length in (("gate_inner", -1, GATE_WALL_LENGTH[0]), ("gate_outer", 1, GATE_WALL_LENGTH[1])):
        model.geom_pos[model.geom(name).id, 0] = sign * (width / 2 + length / 2)


def set_marker(model: mujoco.MjModel, data: mujoco.MjData, body: str, pose):
    """Place a mocap marker body ("goal" or "estimate") at planar pose (x, y, yaw)."""
    mid = model.body_mocapid[model.body(body).id]
    data.mocap_pos[mid] = [pose[0], pose[1], 0.0]
    data.mocap_quat[mid] = [math.cos(pose[2] / 2), 0, 0, math.sin(pose[2] / 2)]
