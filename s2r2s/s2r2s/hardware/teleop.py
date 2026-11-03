"""Teleoperate the SO-101 follower from the SO-101 leader arm, through the safety guard.

LeRobot's ``lerobot-teleoperate`` writes the leader's angles straight to the follower, so the follower goes
wherever the leader goes: into the mat, the base, a wall or itself. This loop sends them through the same
``SafetyGuard`` as every other tool (safety.py): the follower stops at its joint limits and at a clearance from
everything it could hit, moves no faster than ``max_speed_deg_s``, and a servo fault (overload, stall, overheating)
holds it where it is. When the leader goes where the follower may not, the follower waits at the boundary and
follows again as soon as the leader comes back. At the start it moves to the leader's pose slowly.

    python -m s2r2s.hardware.teleop --leader-port COM6 --port COM5 --map calibration\\joint_map.json --lightbox
    python -m s2r2s.hardware.teleop --robot sim --lightbox      # rehearsal: a scripted leader tries to hit things

By default the pusher rod stays clamped (the follower's gripper holds it whatever the leader's gripper does) and
is part of the collision model; ``--no-rod``: no rod, the follower's gripper follows the leader's. Ctrl+C stops;
the follower holds its last position (support it before turning the power off).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from ..scene import ARM_JOINTS, SceneConfig
from .joint_map import JointMap
from .safety import SafetyGuard, SafetyStop, guard_for


def follow(leader, robot, guard: SafetyGuard, seconds: float = math.inf, fps: float = 30.0,
           catch_up_deg_s: float = 30.0, accel_deg_s2: float = 600.0, realtime: bool = True, report=print) -> dict:
    """Follow ``leader() -> (arm degrees (5,), gripper 0-100 or None)`` with ``robot`` at ``fps`` for ``seconds``.
    Until the follower first reaches the leader's pose it moves at ``catch_up_deg_s`` at most. The follower's
    commands speed up and slow down at ``accel_deg_s2`` at most (a leader yanked from rest does not throw the
    follower). What the guard held back is reported at most once a second. Returns counts; a safety stop raises
    SafetyStop (the arm holds)."""
    dt = 1.0 / fps
    stats = dict(ticks=0, limited=0)
    caught_up, last_report, t = False, -math.inf, 0.0
    velocity = np.zeros(5)
    while t < seconds:
        t0 = time.monotonic()
        arm, gripper = leader()
        arm = np.asarray(arm, float)
        present = robot.read_joints()
        sent = present if robot.last_goal is None else np.asarray(robot.last_goal, float)
        target = arm
        if not caught_up:
            if np.max(np.abs(arm - present)) < 3.0:
                caught_up = True
            else:
                v = catch_up_deg_s * dt
                target = present + np.clip(arm - present, -v, v)
        # Accelerate toward the target, and brake in time to stop on it (per joint).
        gap = target - sent
        want = np.clip(gap / dt, -np.sqrt(2 * accel_deg_s2 * np.abs(gap)), np.sqrt(2 * accel_deg_s2 * np.abs(gap)))
        velocity = velocity + np.clip(want - velocity, -accel_deg_s2 * dt, accel_deg_s2 * dt)
        command = sent + velocity * dt
        robot.send_joints(command, seconds=dt, gripper=gripper)
        velocity = (np.asarray(robot.last_goal, float) - sent) / dt if robot.last_goal is not None else velocity * 0
        stats["ticks"] += 1
        if robot.events:
            stats["limited"] += 1
            if t - last_report >= 1.0:
                report("follower held back: " + "; ".join(dict.fromkeys(robot.events)))
                last_report = t
        t += dt
        if realtime:
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    return stats


def scripted_leader(guard: SafetyGuard, seconds_per_move: float = 2.0, fps: float = 30.0):
    """A leader for rehearsals: from a safe hover it drives the rod into the mat, the arm into the base, into
    itself and (lightbox) into a wall, going back to the hover between them. Returns (leader, total seconds, plan)."""
    ex = guard.examples()
    plan = ["hover"] + [p for k in ("mat", "base", "self", "wall", "link_mat") if k in ex for p in (k, "hover")]
    n = round(seconds_per_move * fps)
    path = [ex["hover"]]
    for a, b in zip(plan, plan[1:]):
        path += list(np.linspace(ex[a], ex[b], n)[1:]) + [ex[b]] * (n // 2)    # move there, then insist
    it = iter(path)
    last = [path[0]]

    def leader():
        last[0] = next(it, last[0])
        return last[0], None
    return leader, len(path) / fps, plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["sim", "so101"], default="so101")
    ap.add_argument("--port", default="", help="follower serial port, e.g. COM5")
    ap.add_argument("--leader-port", default="", help="leader serial port, e.g. COM6")
    ap.add_argument("--leader-id", default="s2r2s_leader", help="LeRobot id of the leader's calibration")
    ap.add_argument("--map", default="", help="joint map from calibrate.py (required on the real arm)")
    ap.add_argument("--lightbox", action="store_true", help="the follower stands in the lightbox")
    ap.add_argument("--no-rod", action="store_true", help="no pusher rod: the follower's gripper follows the leader's")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-speed", type=float, default=None, help="joint speed limit (deg/s; default SafetyLimits)")
    args = ap.parse_args(argv)
    scene = SceneConfig(timestep=0.005, lightbox=args.lightbox)
    rod = not args.no_rod
    if args.robot == "so101" and not args.map:
        ap.error("--map is required on the real arm (run calibrate.py first)")
    jmap = JointMap.load(args.map) if args.map else JointMap()
    try:
        guard = guard_for(scene, jmap, rod=rod)
    except SafetyStop as stop:
        raise SystemExit(f"not starting: {stop}")
    if args.max_speed:
        guard.limits.max_speed_deg_s = args.max_speed

    if args.robot == "sim":
        from .robots import SimRobot
        robot = SimRobot(scene, guard=guard, rod=rod)
        robot.move_by_hand(jmap.to_sim(guard.examples()["hover"]))
        leader, seconds, plan = scripted_leader(guard, fps=args.fps)
        print(f"rehearsal: the scripted leader goes {' -> '.join(plan)}")
        stats = follow(leader, robot, guard, seconds, args.fps, realtime=False)
        worst = guard.check(robot.read_joints())
        print(f"{stats['ticks']} ticks, the guard held the follower back in {stats['limited']}; "
              f"follower at the end: {'clear of everything' if worst.ok else '; '.join(worst.text.values())}")
        return

    from .robots import SO101Robot
    try:
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
    except ImportError as error:
        raise SystemExit(f'LeRobot is not installed: uv pip install "lerobot[feetech]" ({error})')
    leader_arm = SOLeader(SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
    leader_arm.connect(calibrate=True)
    robot = SO101Robot(args.port, guard=guard, hold_gripper=rod)

    def leader():
        a = leader_arm.get_action()
        return np.array([a[f"{n}.pos"] for n in ARM_JOINTS]), a.get("gripper.pos")

    print("Following the leader; Ctrl+C to stop. The follower first moves to the leader's pose, slowly.")
    try:
        stats = follow(leader, robot, guard, fps=args.fps)
    except KeyboardInterrupt:
        stats = None
        robot.hold()
        print("stopped by user: the follower holds its position")
    except SafetyStop as stop:
        stats = None
        print(f"SAFETY STOP: {stop}\nThe follower holds where it is. Support it before turning the power off.")
    finally:
        leader_arm.disconnect()
        robot.close()
    if robot.event_log:
        print(f"the guard held the follower back {len(robot.event_log)} times")


if __name__ == "__main__":
    main()
