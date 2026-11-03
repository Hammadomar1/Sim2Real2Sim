"""Robots behind one interface: joint readings and commands in LeRobot degrees.

    read_joints() -> (5,) arm joints, degrees
    send_joints(deg)   arm joint targets, degrees (the gripper keeps holding the rod), through the safety guard
    health()           servo load, temperature, voltage and error flags (None if unreadable)
    hold() / resume()  stop where the arm is / clear a safety stop
    set_torque(on)     torque off lets you move the arm by hand (support it first: it falls!)
    close()

Every command to the real arm goes through a ``SafetyGuard`` (safety.py): joint limits, a speed limit and a
collision check of the target and the path to it, then a servo monitor. A fault holds the arm and raises
``SafetyStop``; later commands raise it again until ``resume()``.
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from ..kinematics import PusherKinematics
from ..objects import OBJECTS
from ..scene import ARM_JOINTS, SceneConfig, SceneIndex, build_model, set_gate
from .joint_map import JointMap
from .safety import SafetyGuard, SafetyStop, ServoMonitor


class SimRobot:
    """The simulated SO-101 behind the real robot's interface.

    ``hidden`` is the robot's *true* joint map, unknown to the software under test: give it offsets
    or flipped signs to mimic an arm before calibration. Physics advances only when a command is
    sent (one control period per command, or ``seconds`` of it), so tests run faster than real time.

    Play, per joint (pan, lift, elbow, wrist flex[, roll]; degrees, full width): ``backlash_deg`` is a dead band
    between the command and the link that the encoder sees (it reads the link), the simulator's
    ``rand.backlash_deg``; ``slack_deg`` is play between the encoder's shaft and the link that the encoder cannot
    see (a loose horn or coupling): the shaft goes where commanded, the link stops up to half the slack short.

    ``guard``: the same safety guard as on the real arm (commands limited and checked, servo monitor with load
    from the actuator forces); None: commands go straight to the actuators, clamped to their ranges.
    ``rod=False``: no pusher rod (it neither touches nor is touched: teleoperation rehearsals without one).
    """

    def __init__(self, scene: SceneConfig | None = None, object_name: str = "tee", hidden: JointMap | None = None,
                 control_dt: float = 0.05, encoder_noise_deg: float = 0.0, seed: int = 0,
                 backlash_deg=(0.0,) * 5, slack_deg=(0.0,) * 5, guard: SafetyGuard | None = None, rod: bool = True):
        self.scene = scene or SceneConfig(timestep=0.005)
        self.model = build_model(self.scene, object_name, visual=False)
        if not rod:
            g = self.model.geom("pusher").id
            self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 0
        self.object_height = OBJECTS[object_name].height
        self.data = mujoco.MjData(self.model)
        self.idx = SceneIndex(self.model)
        self.kin = PusherKinematics(self.model)
        self.hidden = hidden or JointMap()
        self.nsub = round(control_dt / self.scene.timestep)
        self.noise = encoder_noise_deg
        self.rng = np.random.default_rng(seed)
        self.torque = True
        pad = lambda v: np.radians(np.r_[np.asarray(v, float), np.zeros(5)][:5])
        self.play, self.slack = pad(backlash_deg), pad(slack_deg)
        self.block_height = OBJECTS[self.scene.clutter].height if self.scene.clutter else 0.0
        self.guard = guard
        self.monitor = ServoMonitor(guard.limits) if guard is not None else None
        self.stopped, self.events, self.time, self.last_goal = None, [], 0.0, None
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
        self._settle_play(q[0])
        self._forget_motion()
        if self.scene.clutter:          # a clutter scene: the block waits out of reach until place_block()
            self.place_block((0.31, -0.22, 0.0))
        mujoco.mj_forward(self.model, self.data)

    def _settle_play(self, q):
        """Play taken up nowhere in particular: shaft and link where the arm is."""
        self.shaft, self.link = np.array(q, float), np.array(q, float)

    def _forget_motion(self):
        """The arm is somewhere new (put there, moved by hand, held): no previous command to compare with."""
        self.last_goal = None
        if self.monitor is not None:
            self.monitor.reset_motion()

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
        self._settle_play(q_rad)
        self._forget_motion()
        mujoco.mj_forward(self.model, self.data)

    # -- robot interface
    def read_joints(self) -> np.ndarray:
        q = self.data.qpos[self.idx.arm_qpos] + (self.shaft - self.link)     # the encoder sits on the shaft
        deg = self.hidden.to_robot(q)
        return deg + (self.rng.normal(0, self.noise, 5) if self.noise else 0.0)

    def send_joints(self, deg, seconds: float | None = None, gripper: float | None = None):
        """Set the servo targets (degrees, robot frame) and advance the physics by ``seconds`` (default: one
        control period). With a guard, the targets are limited and checked first and the monitor watches the
        move (``events``: what the guard limited; ``SafetyStop`` on a fault). ``gripper``: 0-100 (LeRobot's
        gripper units), None: keep holding the rod."""
        if self.stopped:
            raise SafetyStop(self.stopped)
        nstep = self.nsub if seconds is None else max(1, round(seconds / self.scene.timestep))
        if self.guard is not None:
            deg, self.events = self.guard.filter(self.read_joints(), deg, dt=nstep * self.scene.timestep,
                                                 previous=self.last_goal)
            self.last_goal = deg
        if self.torque:
            lo = self.model.actuator_ctrlrange[self.idx.arm_act, 0]
            hi = self.model.actuator_ctrlrange[self.idx.arm_act, 1]
            q = np.clip(self.hidden.to_sim(deg), lo, hi)
            self.shaft = np.clip(self.shaft, q - self.play / 2, q + self.play / 2)            # dead band, seen
            self.link = np.clip(self.link, self.shaft - self.slack / 2, self.shaft + self.slack / 2)   # unseen
            self.data.ctrl[self.idx.arm_act] = self.link
            if gripper is not None:
                g_lo, g_hi = self.model.actuator_ctrlrange[self.idx.gripper_act]
                self.data.ctrl[self.idx.gripper_act] = g_lo + np.clip(gripper, 0, 100) / 100 * (g_hi - g_lo)
        mujoco.mj_step(self.model, self.data, nstep=nstep)
        self.time += nstep * self.scene.timestep
        if self.monitor is not None:
            reason = self.monitor.update(self.time, self.read_joints(), deg, self.health())
            if reason:
                self.hold(reason)
                raise SafetyStop(reason)

    def health(self) -> dict:
        """Servo state as the real arm reports it: load from the actuator forces; nothing overheats here."""
        load = self.data.actuator_force[self.idx.arm_act] / self.model.actuator_forcerange[self.idx.arm_act, 1]
        return dict(load=load, temperature=np.full(5, 35.0), status=np.zeros(5, int), voltage=np.full(5, 12.0))

    def hold(self, reason: str | None = None):
        """Stay where the arm is (with ``reason``: a safety stop; commands raise SafetyStop until resume())."""
        self.data.ctrl[self.idx.arm_act] = self.data.qpos[self.idx.arm_qpos]
        self.shaft, self.link = self.data.qpos[self.idx.arm_qpos].copy(), self.data.qpos[self.idx.arm_qpos].copy()
        self._forget_motion()
        if reason:
            self.stopped = reason

    def resume(self):
        self.stopped = None
        if self.guard is not None:
            self.monitor = ServoMonitor(self.guard.limits)

    def jam(self, joint: int):
        """Test hook: something the safety model does not know about blocks ``joint`` (0-4) where it is."""
        jid = self.model.joint(ARM_JOINTS[joint]).id
        q = self.data.qpos[self.idx.arm_qpos[joint]]
        self.model.jnt_limited[jid] = 1
        self.model.jnt_range[jid] = [q - 1e-3, q + 1e-3]

    def set_torque(self, on: bool):
        self.torque = on
        self._forget_motion()

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

    Uses LeRobot's own calibration (run once with LeRobot first) and ``use_degrees=True``. Every command goes
    through ``guard`` (required: build it with ``safety.guard_for``), which also gets each servo's calibrated
    range here; the guard limits each command to its step from the joints read just before (LeRobot's own
    ``max_relative_target`` would read them a second time). The servo monitor reads load, temperature, voltage
    and error flags ``guard.limits.health_hz`` times a second. The gripper keeps the position it had at
    connection (it clamps the pusher rod) unless ``hold_gripper=False`` (teleoperation). Torque stays on after
    close() and after a safety stop; support the arm before powering it down.
    """

    def __init__(self, port: str, guard: SafetyGuard, robot_id: str = "s2r2s_follower", calibrate: bool = True,
                 hold_gripper: bool = True):
        if not isinstance(guard, SafetyGuard):
            raise TypeError("SO101Robot needs a SafetyGuard (safety.guard_for(scene, joint_map))")
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
        config = SOFollowerRobotConfig(port=port, id=robot_id, use_degrees=True, max_relative_target=None,
                                       disable_torque_on_disconnect=False)
        self.robot = SOFollower(config)
        self.robot.connect(calibrate=calibrate)
        self.guard = guard
        guard.set_calibrated_range(self.calibrated_range_deg())
        self.monitor = ServoMonitor(guard.limits)
        self.gripper = float(self.robot.get_observation()["gripper.pos"])
        self.hold_gripper = hold_gripper
        self.stopped, self.events, self.event_log = None, [], []
        self.last_send, self.last_goal, self.last_health, self.bus_errors = None, None, -math.inf, 0

    def calibrated_range_deg(self) -> np.ndarray:
        """Each arm joint's calibrated range in LeRobot degrees, (5, 2): the servos' own position limits."""
        cal = self.robot.calibration
        half = np.array([(cal[n].range_max - cal[n].range_min) / 2 * 360 / 4095 for n in ARM_JOINTS])
        return np.c_[-half, half]

    def read_joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{name}.pos"] for name in ARM_JOINTS], dtype=float)

    def send_joints(self, deg, seconds: float | None = None, gripper: float | None = None):
        """Write the goal positions (degrees) through the guard; ``seconds`` is for the simulated arm (real time
        passes by itself). ``gripper``: 0-100 when not holding the rod."""
        if self.stopped:
            raise SafetyStop(self.stopped)
        if self.bus_errors >= 3:
            self.stopped = "the servos stopped answering (3 health reads in a row failed)"
            raise SafetyStop(self.stopped)
        now = time.monotonic()
        dt = None if self.last_send is None else now - self.last_send
        self.last_send = now
        present = self.read_joints()
        deg, self.events = self.guard.filter(present, deg, dt=dt, previous=self.last_goal)
        self.last_goal = deg
        if self.events:
            self.event_log.append((time.time(), self.events))
        action = {f"{name}.pos": float(v) for name, v in zip(ARM_JOINTS, deg)}
        action["gripper.pos"] = self.gripper if gripper is None or self.hold_gripper else float(np.clip(gripper, 0, 100))
        self.robot.send_action(action)
        health = None
        if now - self.last_health >= 1.0 / self.guard.limits.health_hz:
            self.last_health = now
            health = self.health()
        reason = self.monitor.update(now, present, deg, health)
        if reason:
            self.hold(reason)
            raise SafetyStop(reason)

    def health(self) -> dict | None:
        """Load (share of maximum torque), temperature (C), voltage (V) and error flags per arm servo."""
        bus, names = self.robot.bus, list(ARM_JOINTS)
        try:
            read = {reg: bus.sync_read(reg, names, normalize=False)
                    for reg in ("Present_Load", "Present_Temperature", "Present_Voltage", "Status")}
        except Exception:        # a dropped packet now and then is normal; send_joints stops after three in a row
            self.bus_errors += 1
            return None
        self.bus_errors = 0
        col = lambda reg: np.array([read[reg][n] for n in names], dtype=float)
        return dict(load=col("Present_Load") / 1000, temperature=col("Present_Temperature"),
                    voltage=col("Present_Voltage") / 10, status=col("Status").astype(int))

    def limit_report(self) -> list[str]:
        """The servos' own limits against LeRobot's calibration, one line per motor (PASS/FAIL)."""
        bus, cal = self.robot.bus, self.robot.calibration
        lo, hi = bus.sync_read("Min_Position_Limit", normalize=False), bus.sync_read("Max_Position_Limit", normalize=False)
        torque = bus.sync_read("Max_Torque_Limit", normalize=False)
        lines = []
        for n in bus.motors:
            ok = lo[n] == cal[n].range_min and hi[n] == cal[n].range_max
            lines.append(f"{'PASS' if ok else 'FAIL'}  {n:13s} servo limits {lo[n]}..{hi[n]} "
                         f"(calibration {cal[n].range_min}..{cal[n].range_max}), max torque {torque[n] / 10:.0f} %")
        return lines

    def hold(self, reason: str | None = None):
        """Stay where the arm is (with ``reason``: a safety stop; commands raise SafetyStop until resume())."""
        if reason:
            self.stopped = reason
        self.last_goal = None
        self.monitor.reset_motion()
        try:
            present = self.read_joints()
            action = {f"{name}.pos": float(v) for name, v in zip(ARM_JOINTS, present)}
            action["gripper.pos"] = self.gripper
            self.robot.send_action(action)
        except Exception as error:      # the servos still hold their last goal
            print(f"could not write the hold position: {error}")

    def resume(self):
        """Clear a safety stop (after checking the arm)."""
        self.stopped, self.bus_errors = None, 0
        self.monitor = ServoMonitor(self.guard.limits)

    def set_torque(self, on: bool):
        bus = self.robot.bus
        if on:   # hold where the arm is now instead of jumping to an old goal
            self.hold()
            bus.enable_torque()
        else:
            bus.disable_torque()
            self.last_goal = None
            self.monitor.reset_motion()

    def close(self):
        self.robot.disconnect()
