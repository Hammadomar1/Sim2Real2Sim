"""The safety layer (hardware/safety.py): every command is limited and checked before it reaches the motors, the
servo monitor stops the arm on a fault, and the tools and teleoperation go through it."""
import math
from pathlib import Path

import numpy as np
import pytest

from s2r2s.hardware.joint_map import JointMap
from s2r2s.hardware.poses import SimPoseSource
from s2r2s.hardware.robots import SimRobot
from s2r2s.hardware.runner import PolicyRunner
from s2r2s.hardware.safety import SafetyGuard, SafetyLimits, SafetyStop, ServoMonitor, guard_for
from s2r2s.hardware.teleop import follow, scripted_leader
from s2r2s.scene import SceneConfig

ROOT = Path(__file__).resolve().parents[1]
FLAT, LIGHTBOX = SceneConfig(timestep=0.005), SceneConfig(timestep=0.005, lightbox=True)


@pytest.fixture(scope="module")
def guards():
    return {"flat": SafetyGuard(FLAT), "lightbox": SafetyGuard(LIGHTBOX)}


@pytest.mark.parametrize("scene,rod", [(FLAT, True), (LIGHTBOX, True), (LIGHTBOX, False)])
def test_selftest_passes(scene, rod):
    results = SafetyGuard(scene, rod=rod).selftest()
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]
    names = {name for name, _, _ in results}
    assert ("rod into the mat refused" in names) == rod
    assert ("arm into a lightbox wall refused" in names) == scene.lightbox


def test_pushing_poses_are_never_limited(guards):
    """The whole band the policies push in, rod 5 to 40 mm over the mat: nothing for the guard to refuse."""
    g = guards["lightbox"]
    rng = np.random.default_rng(1)
    reached = 0
    for _ in range(300):
        r, a = rng.uniform(0.12, 0.275), math.radians(rng.uniform(-55, 55))
        xyz = (0.0388 + r * math.cos(a), r * math.sin(a), rng.uniform(0.005, 0.04))
        q = g.ik(xyz)
        if np.linalg.norm(g.tip(q) - xyz) > 1e-3:        # out of reach for a vertical rod: no pushing pose
            continue
        reached += 1
        assert g.check(q).ok, g.check(q).text
    assert reached > 200


def test_rod_never_commanded_into_the_mat_and_never_creeps(guards):
    g = guards["lightbox"]
    hover, down = g.ik((0.20, 0.0, 0.02)), g.ik((0.20, 0.0, -0.03))
    q = hover.copy()
    lowest = np.inf
    for _ in range(300):               # a controller insisting, from wherever the last command left the arm
        q, events = g.filter(q, down, 0.05, previous=q)
        lowest = min(lowest, g.tip(q)[2])
    assert lowest >= -1e-4 and "rod" in " ".join(events)
    assert g.tip(q)[2] < 0.002         # it got all the way down to the mat, not stopped short


def test_out_of_a_bad_pose_but_not_deeper(guards):
    g = guards["flat"]
    pressed = g.ik((0.20, 0.0, -0.003))
    z0 = g.tip(pressed)[2]
    assert g.drive(pressed, g.ik((0.20, 0.0, -0.01)), 50) is not None
    assert g.tip(g.drive(pressed, g.ik((0.20, 0.0, -0.01)), 50))[2] >= z0 - 1e-6
    assert g.tip(g.drive(pressed, g.ik((0.20, 0.0, 0.03)), 50))[2] > 0.025


def test_holding_keeps_the_previous_command_not_the_sagged_reading(guards):
    """A refused command must not let gravity win: the guard holds the previous command, not the measured pose."""
    g = guards["flat"]
    previous = g.ik((0.20, 0.0, 0.0005))            # the rod just over the mat
    sagged = g.ik((0.20, 0.0, -0.002))              # the arm sagged 2.5 mm below it
    got, _ = g.filter(sagged, g.ik((0.20, 0.0, -0.02)), 0.05, previous=previous)
    assert g.tip(got)[2] >= -1e-4                   # commanded back to the mat, not left where it sagged to


def test_joint_limits_calibration_margin_step_and_speed():
    g = SafetyGuard(FLAT, joint_range_deg=[[-90, 90]] * 5)
    assert np.allclose(g.lo, [-87, -87, -87, -87, -87]) and np.isclose(g.hi[3], 87)    # 3 deg from the stops
    hover = g.ik((0.20, 0.0, 0.03))
    past = hover.copy()
    past[4] = 120.0
    assert g.drive(hover, past, 100)[4] == pytest.approx(87.0)
    jump = hover + 30.0
    got, _ = g.filter(hover, jump)
    assert np.max(np.abs(got - hover)) <= g.limits.max_step_deg + 1e-9
    got, _ = g.filter(hover, hover + 1.0, dt=0.005, previous=hover)          # 120 deg/s x 5 ms = 0.6 deg
    assert np.max(np.abs(got - hover)) == pytest.approx(0.6)
    got, events = g.filter(hover, [np.nan] * 5)
    assert np.allclose(got, hover) and events


def test_walls_only_in_the_lightbox(guards):
    target = guards["lightbox"].examples()["wall"]
    assert any("wall" in " ".join(k) for k in guards["lightbox"].check(target).bad if isinstance(k, tuple))
    assert not any("wall" in " ".join(k) for k in guards["flat"].check(target).bad if isinstance(k, tuple))


def test_map_calibrated_elsewhere_is_refused():
    with pytest.raises(SafetyStop, match="lightbox"):
        guard_for(LIGHTBOX, JointMap(lightbox=False))
    with pytest.raises(SafetyStop, match="lightbox"):
        guard_for(FLAT, JointMap(lightbox=True))
    assert guard_for(FLAT, JointMap(lightbox=False), selftest=False)


def test_joint_map_records_the_workspace(tmp_path):
    JointMap(lightbox=True).save(tmp_path / "map.json")
    assert JointMap.load(tmp_path / "map.json").lightbox is True
    (tmp_path / "old.json").write_text('{"sign": [1, 1, 1, 1, 1], "offset_deg": [0, 0, 0, 0, 0]}')
    assert JointMap.load(tmp_path / "old.json").lightbox is None         # maps from before: not checked


def test_monitor_faults():
    lim = SafetyLimits()
    q = np.zeros(5)
    healthy = dict(load=np.zeros(5), temperature=np.full(5, 40.0), status=np.zeros(5, int))
    m = ServoMonitor(lim)
    assert m.update(0.0, q, q, healthy) is None
    assert "overheat" in m.update(0.1, q, q, {**healthy, "status": np.array([0, 0, 4, 0, 0])})
    assert "C" in ServoMonitor(lim).update(0.0, q, q, {**healthy, "temperature": np.full(5, 70.0)})
    # Full load while moving is a fast move; full load without moving is something in the way.
    m = ServoMonitor(lim)
    loaded = {**healthy, "load": np.array([0, 0.95, 0, 0, 0])}
    for k in range(10):
        moving = np.array([0, 3.0 * k, 0, 0, 0])
        assert m.update(0.05 * k, moving, moving + 2, loaded) is None
    m = ServoMonitor(lim)
    reasons = [m.update(0.05 * k, q, q + 2, loaded) for k in range(10)]
    assert any(r and "resisting" in r for r in reasons)
    # Commanded to move, not moving: blocked.
    m = ServoMonitor(lim)
    reasons = [m.update(0.05 * k, q, q + np.array([0, 0, 4.0, 0, 0])) for k in range(30)]
    assert any(r and "blocked" in r for r in reasons)
    # Sent somewhere, found 12 deg away: falling or pushed.
    m = ServoMonitor(lim)
    m.update(0.0, q, q)
    assert "falling" in m.update(0.05, q + np.array([0, -12.0, 0, 0, 0]), q)


def test_sim_arm_stops_and_holds_when_something_blocks_it():
    guard = SafetyGuard(FLAT)
    robot = SimRobot(FLAT, guard=guard)
    robot.move_by_hand(JointMap().to_sim(guard.ik((0.20, 0.0, 0.03))))
    robot.jam(0)                                   # an obstacle the model does not know stops the pan joint
    target = robot.read_joints() + np.array([20.0, 0, 0, 0, 0])
    with pytest.raises(SafetyStop, match="shoulder_pan"):
        for _ in range(100):
            robot.send_joints(target)
    held = robot.read_joints()
    with pytest.raises(SafetyStop):               # stopped until someone clears it
        robot.send_joints(held)
    robot.resume()
    robot.send_joints(held)


@pytest.mark.parametrize("scene,rod", [(FLAT, True), (LIGHTBOX, True), (LIGHTBOX, False)])
def test_teleop_leader_cannot_drive_the_follower_into_anything(scene, rod):
    """A scripted leader drives into the mat, the base, the arm itself and the walls: the follower stops short of
    each (at most a few mm into its clearance margin while braking), follows again, and is never stopped."""
    guard = SafetyGuard(scene, rod=rod)
    robot = SimRobot(scene, guard=guard, rod=rod)
    robot.move_by_hand(JointMap().to_sim(guard.examples()["hover"]))
    leader, seconds, plan = scripted_leader(guard, seconds_per_move=1.0)
    worst = {}

    def watched():
        c = guard.check(robot.read_joints())
        for k, v in c.bad.items():
            if isinstance(k, tuple):
                worst[k] = (max(worst.get(k, (0.0,))[0], v), c.kinds[k])
        return leader()
    stats = follow(watched, robot, guard, seconds, realtime=False, report=lambda _: None)
    assert stats["limited"] > 0
    lim = guard.limits
    for pair, (mm, kinds) in worst.items():
        if kinds == ("rod", "surface"):            # the rod may touch the mat; the simulated one is rigid, so a
            assert mm < 5.0, (pair, mm)            # stop from speed presses it in a little (foam gives on the real)
        else:                                      # braking eats into the clearance, never through it
            need = lim.link_clearance if "surface" in kinds or "wall" in kinds else lim.self_clearance
            assert mm < need * 1e3, (pair, mm)
    assert guard.check(robot.read_joints()).ok


@pytest.mark.skipif(not (ROOT / "runs" / "tee_lightbox_v2" / "best.pt").exists(), reason="needs the trained policy")
def test_guard_never_limits_a_policy_at_work():
    guard = guard_for(LIGHTBOX, JointMap())
    robot = SimRobot(LIGHTBOX, guard=guard)
    robot.place((0.15, -0.10), (0.20, -0.02, 0.3))
    runner = PolicyRunner(ROOT / "runs" / "tee_lightbox_v2" / "best.pt", robot, SimPoseSource(robot, seed=1), JointMap(),
                          realtime=False)
    runner.start(np.array([0.21, 0.02, 1.0]))
    limited = 0
    for _ in range(300):
        runner.tick()
        limited += bool(robot.events)
    pos, yaw = runner.errors(robot.object_pose())
    assert pos < 0.01 and yaw < math.radians(10)
    assert limited == 0


def test_filter_is_fast(guards):
    import time
    g = guards["lightbox"]
    q = g.ik((0.20, 0.0, 0.01))
    t0 = time.perf_counter()
    for _ in range(200):
        g.filter(q, q + 0.8, dt=0.01, previous=q)
    assert (time.perf_counter() - t0) / 200 < 0.003       # well inside a 100 Hz ramp step
