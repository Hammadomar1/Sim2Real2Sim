"""Real-arm bring-up tools on the simulated arm: the runner's lost-object recovery, backlash and camera latency
(hardware/runner.py, backlash.py, latency.py, rod_marker.py)."""
import json
import math

import numpy as np
import pytest

from s2r2s.hardware import backlash, latency
from s2r2s.hardware.joint_map import JointMap
from s2r2s.hardware.poses import SimPoseSource
from s2r2s.hardware.robots import SimRobot
from s2r2s.hardware.runner import PolicyRunner
from s2r2s.kinematics import PusherKinematics
from s2r2s.scene import SceneConfig, build_model


def _untrained(path, task=None):
    """A policy checkpoint (untrained: the recovery does not depend on what the policy does), and its scene."""
    import torch
    from s2r2s.env import EnvConfig
    from s2r2s.ppo import ActorCritic, PPOConfig
    from s2r2s.tasks import make_env
    cfg = EnvConfig(num_envs=1, num_threads=1) if task is None else EnvConfig(num_envs=1, num_threads=1, task=task)
    env = make_env(cfg)
    dims = (env.num_obs, env.num_critic_obs, env.num_actions)
    torch.manual_seed(0)
    torch.save({"ppo": {"policy": ActorCritic(*dims, PPOConfig()).state_dict()}, "env_cfg": cfg.to_dict(),
                "ppo_cfg": PPOConfig().to_dict(), "iteration": 0, "samples": 0, "dims": dims}, path)
    return path, cfg.scene


@pytest.fixture(scope="module")
def untrained(tmp_path_factory):
    return _untrained(tmp_path_factory.mktemp("policy") / "untrained.pt")[0]


@pytest.fixture(scope="module")
def clutter_untrained(tmp_path_factory):
    """A clutter (box) checkpoint and its scene."""
    from s2r2s.env import TaskConfig
    return _untrained(tmp_path_factory.mktemp("policy") / "clutter.pt", TaskConfig(clutter="box"))


class Blinkered(SimPoseSource):
    """The training camera model, blind while the rod tip is within ``radius`` of ``spot`` (the arm hides the
    object there; with ``block``, the clutter block instead)."""

    def __init__(self, robot, spot=None, radius=0.02, block=False, **kw):
        super().__init__(robot, dropout=0.0, **kw)
        self.spot, self.radius, self.block = spot, radius, block

    def _hidden(self):
        return self.spot is not None and np.linalg.norm(self.robot.tool_tip()[:2] - self.spot) < self.radius

    def read(self):
        pose, t = super().read()
        return (None, t) if self._hidden() and not self.block else (pose, t)

    def read_block(self):
        pose, t = super().read_block()
        return (None, t) if self._hidden() and self.block else (pose, t)


def _runner(untrained, **kw):
    robot = SimRobot()
    robot.place((0.15, -0.10), (0.20, -0.02, 0.3))
    poses = Blinkered(robot, seed=1)
    return robot, poses, PolicyRunner(untrained, robot, poses, JointMap(), realtime=False, **kw)


def test_runner_backs_off_until_it_sees_the_object_then_resumes(untrained):
    robot, poses, runner = _runner(untrained)
    runner.start(np.array([0.21, 0.02, 1.0]))
    for _ in range(10):
        runner.tick()
    poses.spot = robot.tool_tip()[:2].copy()               # the arm now hides the object
    modes = [runner.tick() for _ in range(120)]
    first = runner.events[0]
    assert first["result"] == "resumed" and first["seconds"] < 3.0
    assert modes.index("retreat") == 40                     # lost after max_pose_age (2 s = 40 ticks)
    assert "lower" in modes and modes[modes.index("lower"):].count("policy") > 0
    tips = np.array([r["tip"] for r in runner.log if r["mode"] == "retreat"])
    assert tips[:, 2].max() > 0.02                           # lifted
    assert np.linalg.norm(np.array(first["at"]) - poses.spot) >= 0.02      # handed back where the camera sees
    pose = np.array(first["last_pose"])
    gap = runner.twin._surface_distance(np.array([first["at"]]), pose[None], runner.twin.obj_id[:1])[0]
    assert gap >= 0.009                                      # lowered clear of the object
    assert max(abs(np.array(r["target"]) - np.array(r["q"])).max() for r in runner.log) <= math.radians(4.0) + 1e-9
    assert runner.fresh_age == 0.5        # lowering and handing back need a pose this recent, whatever the trigger


def _clutter_runner(clutter_untrained, **kw):
    path, scene = clutter_untrained
    robot = SimRobot(scene)
    robot.place((0.18, -0.06), (0.21, -0.02, 0.3))
    robot.place_block((0.15, -0.10, 0.2))                   # right behind the rod, toward the robot
    poses = Blinkered(robot, seed=1, **kw)
    return robot, poses, PolicyRunner(path, robot, poses, JointMap(), realtime=False)


def test_clutter_runner_backs_off_clear_of_the_block(clutter_untrained):
    robot, poses, runner = _clutter_runner(clutter_untrained)
    runner.start(np.array([0.21, 0.04, 1.0]))
    poses.spot = robot.tool_tip()[:2].copy()
    for _ in range(120):
        runner.tick()
    first = runner.events[0]
    assert first["result"] == "resumed"
    at = np.array([first["at"]])
    assert runner.twin._block_clearance(at, runner.twin.clutter_obs[:1])[0] - 0.006 >= 0.009
    assert runner.block_moved(robot.block_pose(), (0.15, -0.10, 0.2))[0] < 0.002


def test_clutter_runner_starts_clear_when_the_arm_hides_the_block(clutter_untrained):
    """The camera sees the T but not the block at the start (rendered rehearsal, 1 scene in 8): the trial starts
    from a clear start, with the block's start pose from what the camera saw on the way."""
    robot, poses, runner = _clutter_runner(clutter_untrained, block=True)
    poses.spot, poses.radius = robot.tool_tip()[:2].copy(), 0.03
    runner.start(np.array([0.21, 0.04, 1.0]))
    assert runner.log and runner.log[0]["mode"] == "start"
    assert runner.twin.clutter_home[0, :2] == pytest.approx((0.15, -0.10), abs=0.004)
    assert runner.tick() == "policy"
    assert runner.block_moved(robot.block_pose(), (0.15, -0.10, 0.2))[0] < 0.002


def test_runner_gives_up_with_the_rod_lifted(untrained):
    robot, poses, runner = _runner(untrained, recover_seconds=2.0)
    runner.start(np.array([0.21, 0.02, 1.0]))
    poses.spot, poses.radius = robot.tool_tip()[:2].copy(), 1.0            # blind everywhere
    modes = [runner.tick() for _ in range(100)]
    assert "lost" in modes and modes.index("lost") <= 40 + 41 + 1
    assert runner.events[0]["result"] == "lost"
    for _ in range(10):
        assert runner.tick() == "lost"
    assert robot.tool_tip()[2] > 0.015


def test_runner_holds_still_when_asked(untrained):
    robot, poses, runner = _runner(untrained, on_lost="hold")
    runner.start(np.array([0.21, 0.02, 1.0]))
    poses.spot, poses.radius = robot.tool_tip()[:2].copy(), 1.0
    for _ in range(60):
        assert runner.tick() == "policy"
    assert all(np.allclose(r["target"], r["q"]) for r in runner.log[41:])
    assert not runner.events


def test_runner_starts_clear_of_a_start_that_hides_the_object(untrained):
    robot, poses, runner = _runner(untrained)
    poses.spot, poses.radius = robot.tool_tip()[:2].copy(), 0.03
    runner.start(np.array([0.21, 0.02, 1.0]))
    assert runner.start_shift >= 0.03
    assert runner.log and runner.log[0]["mode"] == "start"
    tip = robot.tool_tip()
    assert np.linalg.norm(tip[:2] - poses.spot) >= 0.03 and tip[2] < 0.01      # down at the new start
    assert runner.tick() == "policy"


def test_retreat_point_follows_the_shadow_predictor(untrained):
    """With a pose source that predicts the arm's shadow, the rod goes to the nearest point the camera would see
    the object from (here: only points to the object's right, y below it)."""
    robot, poses, runner = _runner(untrained)
    runner.start(np.array([0.21, 0.02, 1.0]))
    kin = runner.twin.kin

    class View:
        def visible(self, q, name, pose):
            return 1.0 if kin.forward(q[None])[0][0, 1] < pose[1] - 0.03 else 0.0
    poses.view = View()
    q = JointMap().to_sim(robot.read_joints())
    pose = robot.object_pose()
    tip = kin.forward(q[None])[0][0, :2]
    point = runner._retreat_point(q, tip, pose, [])
    assert point is not None and point[1] < pose[1] - 0.03
    assert runner._gaps(point[None], pose)[0] >= 0.01
    assert runner._liftable(point)[0]


def test_view_predictor_sees_what_the_arm_hides():
    import mujoco
    from s2r2s.hardware.arm_mask import ViewPredictor
    from s2r2s.hardware.vision import CameraModel
    scene = SceneConfig(timestep=0.005, lightbox=True)
    model = build_model(scene, visual=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = ViewPredictor(CameraModel.from_mujoco(model, data, "c920", 1920, 1080), scene)
    kin = PusherKinematics(build_model(scene, visual=False))
    q, _, _ = kin.solve(np.array([[0.20, 0.0, 0.005]]), kin.seed(np.array([[0.20, 0.0]])), iterations=30)
    # The left-wall camera looks across the arm: a T beyond the arm (robot's right) is hidden, one on the camera's
    # side is not.
    assert view.visible(q[0], "tee", (0.17, -0.05, 0.0)) < 0.5
    assert view.visible(q[0], "tee", (0.20, 0.10, 0.0)) > 0.95
    q_far, _, _ = kin.solve(np.array([[0.14, 0.15, 0.005]]), kin.seed(np.array([[0.14, 0.15]])), iterations=30)
    assert view.band_hidden(q_far[0]) < view.band_hidden(q[0])


# ---------------------------------------------------------------------------------------------- backlash
def test_sim_robot_play_seen_and_unseen():
    """The dead band shows in the encoders; the slack between encoder and link does not."""
    robot = SimRobot(backlash_deg=(1.0, 0, 0, 0), slack_deg=(0.6, 0, 0, 0))
    jmap = JointMap()
    q0 = jmap.to_sim(robot.read_joints())
    out = {}
    for side in (-1, 1):
        backlash.approach(robot, jmap, q0, 0, side * math.radians(3), settle=0.5)
        out[side] = (jmap.to_sim(robot.read_joints())[0], robot.data.qpos[robot.idx.arm_qpos][0])
    assert math.degrees(out[1][0] - out[-1][0]) == pytest.approx(1.0, abs=0.02)          # encoders
    assert math.degrees(out[1][1] - out[-1][1]) == pytest.approx(1.6, abs=0.02)          # the link


def _sim_arm(**kw):
    scene = SceneConfig(timestep=0.005, lightbox=True)
    robot = SimRobot(scene, encoder_noise_deg=0.05, **kw)
    return robot, PusherKinematics(build_model(scene, visual=False)), scene


def test_backlash_encoders_recover_the_dead_band():
    robot, kin, _ = _sim_arm(backlash_deg=(0.6, 1.0, 0.8, 1.2), slack_deg=(0.4, 0, 0, 0))
    rows = backlash.measure(robot, JointMap(), kin, [0.20], list(backlash.JOINTS), cycles=2, log=lambda *a: None)
    got = {r["joint"]: r["encoders_deg"] for r in rows}
    assert got == pytest.approx({"pan": 0.6, "lift": 1.0, "elbow": 0.8, "wrist": 1.2}, abs=0.06)
    (only,) = backlash.suggestion(backlash.summarise(rows))           # the slack is invisible to the encoders
    key, values = only.split("=")
    assert key == "rand.backlash_deg"
    assert [float(v) for v in values.split(",")] == pytest.approx([0.6, 1.0, 0.8, 1.2], abs=0.051)


def test_backlash_mark_and_measure_finds_play_the_encoders_miss():
    robot, kin, _ = _sim_arm(backlash_deg=(0.6, 0, 0, 0.5), slack_deg=(0.4, 0, 0, 0.6))
    probe = backlash.ManualProbe(backlash.SimPerson(robot, noise_mm=0.05))
    rows = backlash.measure(robot, JointMap(), kin, [0.20], ["pan", "wrist"], probe, cycles=1, log=lambda *a: None)
    got = {r["joint"]: (r["encoders_deg"], r["probe_deg"]) for r in rows}
    assert got["pan"] == pytest.approx((0.6, 1.0), abs=0.06)
    assert got["wrist"] == pytest.approx((0.5, 1.1), abs=0.06)
    s = backlash.summarise(rows)
    assert s["pan"]["hidden_deg"] == pytest.approx(0.4, abs=0.08)
    assert backlash.suggestion(s)[1] == "rand.joint_offset_deg=0.3"


def test_backlash_camera_probe_on_rendered_frames():
    from s2r2s.hardware.camera_sim import SimCamera
    from s2r2s.hardware.rod_marker import ROD_MARKER, ArmPoints
    robot, kin, scene = _sim_arm(backlash_deg=(0.6, 0, 0, 0), slack_deg=(0.4, 0, 0, 0))
    try:
        cam = SimCamera(scene, rod_marker=ROD_MARKER)
    except Exception as error:           # no OpenGL context (headless machine)
        pytest.skip(f"cannot render: {error}")
    rng = np.random.default_rng(0)

    def grab(n):
        cam.set_qpos(robot.data.qpos)
        return [cam.render(rng) for _ in range(n)]
    probe = backlash.CameraProbe(grab, cam.camera(), ArmPoints(scene), frames=5)
    rows = backlash.measure(robot, JointMap(), kin, [0.20], ["pan"], probe, cycles=2, log=lambda *a: None)
    assert rows[0]["encoders_deg"] == pytest.approx(0.6, abs=0.06)
    assert rows[0]["probe_deg"] == pytest.approx(1.0, abs=0.08)


# ---------------------------------------------------------------------------------------------- latency
def test_fit_delay_recovers_a_known_delay():
    rng = np.random.default_rng(0)
    t_enc = np.arange(0, 10, 0.05) + rng.uniform(-0.003, 0.003, 200)
    pan = 0.14 * np.sin(2 * np.pi * 0.5 * t_enc) + rng.normal(0, 0.001, 200)
    t_cam = np.arange(0.3, 10, 1 / 30)
    seen = -0.98 * 0.14 * np.sin(2 * np.pi * 0.5 * (t_cam - 0.087)) + 0.4 + rng.normal(0, 0.002, len(t_cam))
    fit = latency.fit_delay(t_cam, seen, t_enc, pan)
    assert fit["delay"] == pytest.approx(0.087, abs=0.003)
    assert fit["scale"] == pytest.approx(-0.98, abs=0.02)


def test_latency_steps_cover_the_pose_age():
    assert latency.latency_steps(120.0) == (2, 4)       # ages 120-153 ms
    assert latency.latency_steps(60.0) == (1, 2)


def test_latency_rehearsal_finds_the_injected_delay(tmp_path):
    from s2r2s.hardware.camera_sim import SimCamera
    try:
        SimCamera(SceneConfig(timestep=0.005, lightbox=True), width=64, height=36)
    except Exception as error:
        pytest.skip(f"cannot render: {error}")
    out = tmp_path / "latency.json"
    latency.main(["--robot", "sim", "--seconds", "5", "--sim-delay", "0.09", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["camera_ms"] == pytest.approx(90.0, abs=4.0)
    assert r["marker_frames"] >= 0.9 * r["frames"]


def test_runner_ramps_the_servo_targets_over_each_tick(untrained):
    """Each tick's joint target goes out in equal steps from the last one (about 100 Hz), as in training
    (TaskConfig.servo_ramp); the simulated arm advances 10 ms per step, 50 ms per tick."""
    robot, poses, runner = _runner(untrained)
    sent = []
    send = robot.send_joints
    robot.send_joints = lambda deg, seconds=None: (sent.append((np.array(deg, float), seconds)), send(deg, seconds))
    runner.start(np.array([0.21, 0.02, 1.0]))
    t0 = robot.data.time
    for _ in range(4):
        before = runner.last_target.copy()
        sent.clear()
        runner.tick()
        steps = np.array([np.radians(d) for d, _ in sent])          # JointMap() is the identity, in degrees
        assert len(sent) == runner.ramp_steps == 5 and all(s == pytest.approx(0.01) for _, s in sent)
        assert np.allclose(steps[-1], runner.last_target)
        assert np.allclose(np.diff(np.vstack([before, steps]), axis=0), (runner.last_target - before) / 5)
    assert robot.data.time - t0 == pytest.approx(4 * 0.05)
    _, _, jumpy = _runner(untrained, ramp=False)
    assert jumpy.ramp_steps == 1
