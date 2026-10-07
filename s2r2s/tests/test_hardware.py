import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from s2r2s.hardware.calibrate import default_points, fit, simulate_touches
from s2r2s.hardware.joint_map import JointMap
from s2r2s.hardware.poses import SimPoseSource
from s2r2s.hardware.robots import SimRobot, SO101Robot
from s2r2s.hardware.runner import PolicyRunner
from s2r2s.kinematics import PusherKinematics
from s2r2s.scene import SceneConfig, build_model

ROOT = Path(__file__).resolve().parents[1]


def test_joint_map_round_trip(tmp_path):
    m = JointMap(sign=[1, -1, 1, -1, 1], offset_deg=[3, -2, 7, 0.5, 1])
    deg = np.array([10.0, -20.0, 35.0, 80.0, 5.0])
    assert np.allclose(m.to_robot(m.to_sim(deg)), deg)
    m.save(tmp_path / "map.json")
    assert JointMap.load(tmp_path / "map.json") == m


def test_sim_robot_reads_through_its_hidden_map():
    hidden = JointMap(sign=[1, 1, -1, 1, 1], offset_deg=[4, -6, 3, 8, 0])
    robot = SimRobot(hidden=hidden)
    q = robot.data.qpos[robot.idx.arm_qpos].copy()
    assert np.allclose(hidden.to_sim(robot.read_joints()), q)


def test_calibration_recovers_signs_and_offsets():
    hidden = JointMap(sign=[1, 1, -1, 1, 1], offset_deg=[4, -6, 3, 8, 0])
    robot = SimRobot(hidden=hidden, encoder_noise_deg=0.05)
    points = default_points()
    readings = simulate_touches(robot, points)
    kin = PusherKinematics(build_model(SceneConfig(timestep=0.005), visual=False))
    jmap, errors, ranking = fit(readings, points, kin, fit_rod=False)
    assert jmap.sign[:4] == [1.0, 1.0, -1.0, 1.0]
    assert np.allclose(jmap.offset_deg[:4], [4, -6, 3, 8], atol=0.2)
    assert errors.max() < 1e-3
    assert ranking[1][0] > 10 * ranking[0][0]          # the right directions win clearly


def test_so101_robot_talks_lerobot_degrees(monkeypatch):
    sent = {}

    class Config:
        def __init__(self, **kw):
            sent["config"] = kw

    class Follower:
        def __init__(self, config):
            self.bus = types.SimpleNamespace(enable_torque=lambda: None, disable_torque=lambda: None)

        def connect(self, calibrate=True):
            sent["calibrate"] = calibrate

        def get_observation(self):
            return {"shoulder_pan.pos": 1.0, "shoulder_lift.pos": 2.0, "elbow_flex.pos": 3.0,
                    "wrist_flex.pos": 4.0, "wrist_roll.pos": 5.0, "gripper.pos": 42.0}

        def send_action(self, action):
            sent["action"] = action
            return action

        def disconnect(self):
            sent["closed"] = True

    module = types.ModuleType("lerobot.robots.so_follower")
    module.SOFollower, module.SOFollowerRobotConfig = Follower, Config
    for name in ("lerobot", "lerobot.robots"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", module)
    robot = SO101Robot("COM9", max_step_deg=3.0)
    assert sent["config"]["use_degrees"] is True and sent["config"]["max_relative_target"] == 3.0
    assert sent["config"]["disable_torque_on_disconnect"] is False      # the arm must not drop at exit
    assert np.allclose(robot.read_joints(), [1, 2, 3, 4, 5])
    robot.send_joints([10, 20, 30, 40, 50])
    assert sent["action"]["elbow_flex.pos"] == 30 and sent["action"]["gripper.pos"] == 42.0   # gripper holds
    robot.close()
    assert sent["closed"]


@pytest.mark.skipif(not (ROOT / "runs" / "tee_v1" / "best.pt").exists(), reason="needs the trained policy")
def test_policy_runs_through_the_hardware_path():
    robot = SimRobot()
    robot.place((0.15, -0.10), (0.20, -0.02, 0.3))
    runner = PolicyRunner(ROOT / "runs" / "tee_v1" / "best.pt", robot, SimPoseSource(robot, seed=1), JointMap(),
                          realtime=False)
    goal = np.array([0.21, 0.02, 1.0])
    runner.start(goal)
    for _ in range(300):
        runner.tick()
    pos, yaw = runner.errors(robot.object_pose())
    assert pos < 0.01 and yaw < math.radians(10)
    assert max(abs(np.array(r["target"]) - np.array(r["q"])).max() for r in runner.log) <= math.radians(4.0) + 1e-9


def test_clutter_policy_gets_the_block_through_the_hardware_path(tmp_path):
    import torch
    from s2r2s.env import EnvConfig, TaskConfig
    from s2r2s.ppo import ActorCritic, PPOConfig
    from s2r2s.tasks import make_env
    cfg = EnvConfig(num_envs=1, num_threads=1, task=TaskConfig(clutter="box"))
    env = make_env(cfg)                       # also records the block in cfg.scene, as training does
    dims = (env.num_obs, env.num_critic_obs, env.num_actions)
    torch.manual_seed(0)
    policy = ActorCritic(*dims, PPOConfig())
    torch.save({"ppo": {"policy": policy.state_dict()}, "env_cfg": cfg.to_dict(), "ppo_cfg": PPOConfig().to_dict(),
                "iteration": 0, "samples": 0, "dims": dims}, tmp_path / "untrained.pt")
    robot = SimRobot(cfg.scene)
    robot.place((0.15, -0.10), (0.20, -0.02, 0.3))
    robot.place_block((0.21, 0.06, 0.4))
    runner = PolicyRunner(tmp_path / "untrained.pt", robot, SimPoseSource(robot, dropout=0.0, seed=1), JointMap(),
                          realtime=False)
    assert runner.clutter
    runner.start(np.array([0.21, 0.0, 1.0]))
    assert np.allclose(runner.twin.clutter_home[0], [0.21, 0.06, 0.4], atol=0.06)        # camera noise: 1 mm, 1 deg
    assert np.allclose(runner.twin.clutter_home[0, :2], [0.21, 0.06], atol=0.005)
    robot.place_block((0.23, 0.07, 0.4))      # someone nudges the block: the policy sees it move
    for _ in range(3):
        runner.tick()
    assert np.allclose(runner.twin.clutter_obs[0, :2], [0.23, 0.07], atol=0.005)
    assert runner.block_moved(robot.block_pose())[0] > 0.015
    assert max(abs(np.array(r["target"]) - np.array(r["q"])).max() for r in runner.log) <= math.radians(4.0) + 1e-9


@pytest.mark.skipif(not (ROOT / "runs" / "clutter_v7" / "best.pt").exists(), reason="needs the trained policy")
def test_clutter_policy_runs_through_the_hardware_path():
    from s2r2s.train import load_policy
    checkpoint = ROOT / "runs" / "clutter_v7" / "best.pt"
    _, cfg, _ = load_policy(checkpoint)
    robot = SimRobot(cfg.scene)
    # A held-out scene: the T moves 106 mm and turns 141 deg; the block sits 17.5 mm from its straight path.
    robot.place((0.2359, 0.0297), (0.1813, -0.0637, 2.3041))
    block = (0.1778, 0.1064, -1.3348)
    robot.place_block(block)
    runner = PolicyRunner(checkpoint, robot, SimPoseSource(robot, seed=1), JointMap(), realtime=False)
    runner.start(np.array([0.1957, 0.0409, -0.1613]))
    for _ in range(400):
        runner.tick()
    pos, yaw = runner.errors(robot.object_pose())
    assert pos < 0.01 and yaw < math.radians(10)
    assert runner.block_moved(robot.block_pose(), block)[0] < 0.002                 # not even touched
