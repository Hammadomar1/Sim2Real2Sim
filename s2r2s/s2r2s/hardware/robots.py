"""Robots behind one interface: joint readings and commands in LeRobot degrees.

    read_joints() -> (5,) arm joints, degrees
    send_joints(deg)   arm joint targets, degrees (the gripper keeps holding the rod)
    set_torque(on)     torque off lets you move the arm by hand (support it first: it falls!)
    close()
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..kinematics import PusherKinematics
from ..objects import OBJECTS
from ..scene import ARM_JOINTS, SceneConfig, SceneIndex, build_model, set_gate
from .joint_map import JointMap


class SimRobot:
    """The simulated SO-101 behind the real robot's interface.

    ``hidden`` is the robot's *true* joint map, unknown to the software under test: give it offsets
    or flipped signs to mimic an arm before calibration. Physics advances only when a command is
    sent (one control period per command), so tests run faster than real time.
    """

    def __init__(self, scene: SceneConfig | None = None, object_name: str = "tee", hidden: JointMap | None = None,
                 control_dt: float = 0.05, encoder_noise_deg: float = 0.0, seed: int = 0):
        self.scene = scene or SceneConfig(timestep=0.005)
        self.model = build_model(self.scene, object_name, visual=False)
        self.object_height = OBJECTS[object_name].height
        self.data = mujoco.MjData(self.model)
        self.idx = SceneIndex(self.model)
        self.kin = PusherKinematics(self.model)
        self.hidden = hidden or JointMap()
        self.nsub = round(control_dt / self.scene.timestep)
        self.noise = encoder_noise_deg
        self.rng = np.random.default_rng(seed)
        self.torque = True
        self.block_height = OBJECTS[self.scene.clutter].height if self.scene.clutter else 0.0
        self.place((0.15, -0.10), (0.2, 0.0, 0.0))

    # -- scene setup (things a person does by hand on the real table)
    def place(self, tool_xy, object_pose, tool_z: float = 0.005):
        q, err, _ = self.kin.solve(np.array([[*tool_xy, tool_z]]), self.kin.seed(np.array([tool_xy])), iterations=30)
        if err[0] > 1e-3:
            raise ValueError(f"tool position {tool_xy} is not reachable")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.idx.arm_qpos] = q[0]
        self.data.qpos[self.idx.gripper_qpos] = self.scene.gripper_hold
        o = self.idx.obj_qpos
        x, y, yaw = object_pose
        self.data.qpos[o:o + 7] = [x, y, self.object_height / 2, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.ctrl[self.idx.arm_act] = q[0]
        self.data.ctrl[self.idx.gripper_act] = self.scene.gripper_hold
        if self.scene.clutter:          # a clutter scene: the block waits out of reach until place_block()
            self.place_block((0.31, -0.22, 0.0))
        mujoco.mj_forward(self.model, self.data)

    def place_block(self, pose):
        """Clutter scenes: put the block that must not be disturbed at (x, y, yaw)."""
        o = self.idx.clutter_qpos
        x, y, yaw = pose
        self.data.qpos[o:o + 7] = [x, y, self.block_height / 2, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[self.idx.clutter_qvel:self.idx.clutter_qvel + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def place_gate(self, centre, angle, width):
        set_gate(self.model, centre, angle, width)
        mujoco.mj_forward(self.model, self.data)

    def move_by_hand(self, q_rad):
        """The simulated person: put the arm in a configuration (torque off on a real arm)."""
        self.data.qpos[self.idx.arm_qpos] = q_rad
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.idx.arm_act] = q_rad
        mujoco.mj_forward(self.model, self.data)

    # -- robot interface
    def read_joints(self) -> np.ndarray:
        deg = self.hidden.to_robot(self.data.qpos[self.idx.arm_qpos])
        return deg + (self.rng.normal(0, self.noise, 5) if self.noise else 0.0)

    def send_joints(self, deg):
        if self.torque:
            lo = self.model.actuator_ctrlrange[self.idx.arm_act, 0]
            hi = self.model.actuator_ctrlrange[self.idx.arm_act, 1]
            self.data.ctrl[self.idx.arm_act] = np.clip(self.hidden.to_sim(deg), lo, hi)
        mujoco.mj_step(self.model, self.data, nstep=self.nsub)

    def set_torque(self, on: bool):
        self.torque = on

    def close(self):
        pass

    # -- ground truth (only the simulation has it)
    def object_pose(self) -> np.ndarray:
        return self._planar_pose(self.idx.obj_qpos)

    def block_pose(self) -> np.ndarray:
        return self._planar_pose(self.idx.clutter_qpos)

    def _planar_pose(self, o):
        q = self.data.qpos[o + 3:o + 7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return np.array([self.data.qpos[o], self.data.qpos[o + 1], yaw])

    def tool_tip(self) -> np.ndarray:
        return self.data.site_xpos[self.idx.tool_site].copy()


class SO101Robot:
    """The real SO-101 follower through LeRobot (``uv pip install "lerobot[feetech]"``).

    Uses LeRobot's own calibration (run once with LeRobot first) and ``use_degrees=True``. LeRobot's
    ``max_relative_target`` caps every command to ``max_step_deg`` from the present position. The
    gripper keeps the position it had at connection: it is clamping the pusher rod. Torque stays on
    after close(); support the arm before powering it down.
    """

    def __init__(self, port: str, robot_id: str = "s2r2s_follower", max_step_deg: float = 5.0,
                 calibrate: bool = True):
        try:
            from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
        except ImportError:
            try:   # older LeRobot releases
                from lerobot.robots.so101_follower import SO101Follower as SOFollower
                from lerobot.robots.so101_follower import SO101FollowerConfig as SOFollowerRobotConfig
            except ImportError as error:
                raise ImportError('LeRobot is not installed: uv pip install "lerobot[feetech]" '
                                  "(see HARDWARE.md)") from error
        # Keep torque on at disconnect: by default LeRobot relaxes the servos and the arm drops.
        config = SOFollowerRobotConfig(port=port, id=robot_id, use_degrees=True, max_relative_target=max_step_deg,
                                       disable_torque_on_disconnect=False)
        self.robot = SOFollower(config)
        self.robot.connect(calibrate=calibrate)
        self.gripper = float(self.robot.get_observation()["gripper.pos"])

    def read_joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{name}.pos"] for name in ARM_JOINTS], dtype=float)

    def send_joints(self, deg):
        action = {f"{name}.pos": float(v) for name, v in zip(ARM_JOINTS, deg)}
        action["gripper.pos"] = self.gripper
        self.robot.send_action(action)

    def set_torque(self, on: bool):
        bus = self.robot.bus
        if on:   # hold where the arm is now instead of jumping to an old goal
            self.send_joints(self.read_joints())
            bus.enable_torque()
        else:
            bus.disable_torque()

    def close(self):
        self.robot.disconnect()
