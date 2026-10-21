"""Levels 2 and 3 through the hardware path (hardware/levels.py): the planner from camera poses, push switching,
docking, recovery with the blocks' roles swapped."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from s2r2s.env import EnvConfig, TaskConfig
from s2r2s.hardware.joint_map import JointMap
from s2r2s.hardware.levels import LevelRunner
from s2r2s.hardware.poses import SimPoseSource
from s2r2s.hardware.robots import SimRobot
from s2r2s.level2 import BOX, DOCK, TEE
from s2r2s.level3 import Level3Env
from s2r2s.ppo import ActorCritic, PPOConfig
from s2r2s.tasks import make_env

ROOT = Path(__file__).resolve().parents[1]
TRAINED = {TEE: ROOT / "runs" / "clutter_lightbox_v2" / "best.pt", BOX: ROOT / "runs" / "clutter_box_lightbox_v2" / "best.pt",
           DOCK: ROOT / "runs" / "dock_lightbox_v3" / "best.pt"}


def _skill(path, **task):
    """An untrained skill checkpoint with the layout of ``task``; returns the path and its scene."""
    cfg = EnvConfig(num_envs=1, num_threads=1, task=TaskConfig(**task))
    env = make_env(cfg)                       # also records the block in cfg.scene, as training does
    dims = (env.num_obs, env.num_critic_obs, env.num_actions)
    torch.manual_seed(0)
    torch.save({"ppo": {"policy": ActorCritic(*dims, PPOConfig()).state_dict()}, "env_cfg": cfg.to_dict(),
                "ppo_cfg": PPOConfig().to_dict(), "iteration": 0, "samples": 0, "dims": dims}, path)
    return path, cfg.scene


@pytest.fixture(scope="module")
def skills(tmp_path_factory):
    d = tmp_path_factory.mktemp("skills")
    tee, scene = _skill(d / "tee.pt", clutter="box")
    box, _ = _skill(d / "box.pt", objects=("box",), clutter="tee")
    dock, _ = _skill(d / "dock.pt", objects=("box",), clutter="tee", dock=True, episode_seconds=10.0)
    return {TEE: tee, BOX: box, DOCK: dock}, scene


class Still:
    """A policy that never moves the rod: pushes then end only by the sequencer's clock or by settling."""

    def act_deterministic(self, x):
        return torch.zeros(len(x), 2)


class HideBox(SimPoseSource):
    """The training camera model without noise or drops; blind to the box while ``hide`` is set."""

    hide = False

    def __init__(self, robot, **kw):
        super().__init__(robot, latency_steps=0, pos_noise=0.0, yaw_noise=0.0, dropout=0.0, **kw)

    def _seen(self, name, truth):
        pose, t = super()._seen(name, truth)
        return (None, t) if self.hide and name == "block" else (pose, t)


def _runner(skills, level, tee, box, tool=(0.15, -0.12), still=True):
    paths, scene = skills
    robot = SimRobot(scene)
    robot.place(tool, tee)
    robot.place_block(box)
    poses = HideBox(robot)
    runner = LevelRunner(level, paths, robot, poses, JointMap(), realtime=False)
    if still:
        runner.skills = {w: (Still(), c, tw) for w, (_, c, tw) in runner.skills.items()}
    return robot, poses, runner


def _move_tee(robot, pose):
    o = robot.idx.obj_qpos
    robot.data.qpos[o:o + 7] = [pose[0], pose[1], robot.object_height / 2, math.cos(pose[2] / 2), 0, 0, math.sin(pose[2] / 2)]


def test_corner_plans_choose_a_fitting_corner_with_the_fewest_pushes():
    w = Level3Env(EnvConfig(num_envs=256, seed=3, num_threads=4, task=TaskConfig(clutter="box", episode_seconds=80.0)),
                  parking=True)
    tee, box = w.object_pose(), w.clutter_pose()
    side, length, who, goal, clear = w.corner_plans(tee, w.goal, box)
    assert (length > 0).mean() > 0.97 and w._fits(w.goal, side)[length > 0].all()
    two = w.plan_len == 3                       # the scene's corner has a two-push plan (then docking) ...
    assert (length[two] == 3).all()             # ... so the robot's choice has one too, with at least as much room
    assert (clear[two] >= w.plan_clearance[two] - 1e-12).all()
    r = np.flatnonzero(length > 0)
    last = length[r] - 1
    assert (who[r, last] == DOCK).all() and np.allclose(goal[r, last], w.docked(w.goal[r], side[r]))
    box_last = np.array([max(k for k in range(n) if who[i, k] == BOX) for i, n in zip(r, last)])
    assert np.allclose(goal[r, box_last], w.predock(w.goal[r], side[r])[0])       # the box's last move: pre-dock
    assert 0.2 < (side[r] == 1).mean() < 0.8


# Both blocks within the settle tolerance of their goals, the T's 5 mm away from the box and the box's 5 mm toward
# the T: the T first leaves 5 mm more room.
TEE_AT, BOX_AT, TEE_GOAL, BOX_GOAL = (0.20, 0.05, 0.3), (0.20, -0.06, 0.5), (0.20, 0.055, 0.3), (0.20, -0.055, 0.5)


def test_level2_pushes_switch_once_the_block_has_settled(skills):
    tee, box, box_goal = TEE_AT, BOX_AT, BOX_GOAL
    robot, poses, runner = _runner(skills, 2, tee, box)
    runner.start_level(np.array(TEE_GOAL), np.array(box_goal))
    assert [w for w, _ in runner.plan] == [TEE, BOX] and runner.push == 0 and not poses.swap
    assert runner.twin is runner.skills[TEE][2]
    modes = [runner.tick() for _ in range(20)]                 # 1 s within the tolerance ends the first push
    assert modes == ["policy"] * 20 and runner.push == 1
    tw = runner.twin
    assert tw is runner.skills[BOX][2] and poses.swap
    assert np.allclose(tw.obs_pose[0], box, atol=1e-4) and np.allclose(tw.clutter_home[0], tee, atol=1e-4)  # keeps the T
    assert np.allclose(tw.goal[0], box_goal) and np.allclose(tw.cmd_xy[0], robot.tool_tip()[:2], atol=1e-3)
    modes = [runner.tick() for _ in range(20)]
    assert modes[-1] == "done" and modes[:-1] == ["policy"] * 19 and runner.tick() == "done"
    log = runner.log
    assert [e["skill"] for e in log] == ["tee"] * 20 + ["box"] * 20 and [e["push"] for e in log] == [0] * 20 + [1] * 20
    assert np.allclose(log[25]["pose"], box, atol=1e-4) and np.allclose(log[25]["block"], tee, atol=1e-4)   # as it sees them
    s = runner.summary()
    assert s["pushes_done"] == 2 and s["all_settled"] and s["time_s"] == pytest.approx(2.0)
    assert runner.outcome(robot.object_pose(), robot.block_pose())["success"]


def test_level3_docks_for_10_s_with_the_target_fixed_when_docking_starts(skills):
    tee = np.array([0.20, 0.0, math.radians(120)])
    robot, poses, runner = _runner(skills, 3, tee, (0.31, -0.22, 0.0))
    P = runner.planner
    side = np.ones(1)
    assert P._fits(tee[None], side)[0]
    pre = P.predock(tee[None], side)[0][0]
    robot.place_block(pre)                                     # the box waits at the corner's pre-dock spot
    runner.start_level(tee)
    assert [w for w, _ in runner.plan] == [TEE, BOX, DOCK] and runner.side == 1.0
    assert np.allclose(runner.plan[1][1], pre)
    for _ in range(40):
        runner.tick()
    assert runner.push == 2 and runner.twin is runner.skills[DOCK][2] and poses.swap
    target = runner.twin.goal[0].copy()                        # the corner where the camera sees the T now
    assert np.allclose(target, P.docked(tee[None], side)[0], atol=1e-4)
    _move_tee(robot, tee + [0.004, 0.0, 0.0])                  # the T moves while docking: the target stays put
    modes = [runner.tick() for _ in range(200)]                # the box never gets there: 10 s, then done
    assert modes[-1] == "done" and "done" not in modes[:-1]
    assert np.array_equal(runner.twin.goal[0], target)
    assert runner.pushes[2]["seconds"] == pytest.approx(10.0) and not runner.pushes[2]["settled"]
    assert runner.summary()["pushes_done"] == 2 and math.isnan(runner.summary()["time_s"])
    assert not runner.outcome(robot.object_pose(), robot.block_pose())["box_ok"]


def test_a_lost_box_is_recovered_as_the_box_skills_object(skills):
    tee, box = TEE_AT, BOX_AT
    robot, poses, runner = _runner(skills, 2, tee, box)
    runner.start_level(np.array(TEE_GOAL), np.array(BOX_GOAL))
    for _ in range(20):
        runner.tick()
    assert runner.push == 1
    poses.hide = True                                          # the arm hides the box the second push moves
    modes = [runner.tick() for _ in range(41)]
    assert modes[-1] == "retreat" and modes[:-1] == ["policy"] * 40       # lost after 2 s, as with one policy
    assert runner.push == 1             # its held pose was at the target, but unseen for 0.5 s it does not settle
    poses.hide = False
    while runner.tick() != "done":
        assert len(runner.log) < 400
    event = runner.events[0]
    assert event["result"] == "resumed" and np.allclose(event["last_pose"], box, atol=1e-4)   # it backed off the box
    assert runner.pushes[1]["settled"] and runner.push == 1                 # no switch while recovering
    recovering = [e for e in runner.log if e["mode"] != "policy"]
    assert recovering and {e["skill"] for e in recovering} == {"box"}
    gap = runner.twin._surface_distance(np.array([event["at"]]), np.array([box]), runner.twin.obj_id[:1])[0]
    assert gap >= 0.009                                        # lowered clear of the box
    assert max(abs(np.array(e["target"]) - np.array(e["q"])).max() for e in runner.log) <= math.radians(4.0) + 1e-9


def test_no_plan_when_the_blocks_break_the_15_mm_rule(skills):
    robot, poses, runner = _runner(skills, 2, (0.20, 0.0, 0.0), (0.20, 0.052, 0.0))
    gap = runner.planner._gap_to_object(np.array([robot.block_pose()]), np.array([robot.object_pose()]), np.zeros(1, int))
    assert 0.0 < gap[0] < 0.015
    with pytest.raises(RuntimeError, match="move a block"):
        runner.start_level(np.array([0.20, 0.0, 0.0]), np.array([0.20, 0.052, 0.0]))
    with pytest.raises(RuntimeError, match="outside the work band"):
        runner.start_level(np.array([0.30, 0.0, 0.0]), np.array([0.20, -0.06, 0.0]))


def test_skills_must_match_their_roles(skills):
    paths, scene = skills
    robot = SimRobot(scene)
    with pytest.raises(ValueError, match="box skill"):
        LevelRunner(2, {TEE: paths[TEE], BOX: paths[TEE]}, robot, None, JointMap(), realtime=False)
    with pytest.raises(ValueError, match="dock skill"):
        LevelRunner(3, {**paths, DOCK: paths[BOX]}, robot, None, JointMap(), realtime=False)


def test_mount_reaches_the_rendered_camera_in_every_dry_run(skills, monkeypatch):
    from s2r2s.hardware import camera_sim, runner

    class Built(Exception):
        pass

    mounts = []

    def rendered(robot, scene, object_name="tee", **kw):
        mounts.append(kw.get("mount"))
        raise Built                            # the rendered camera is built before the first episode
    monkeypatch.setattr(camera_sim, "RenderedPoseSource", rendered)
    paths, _ = skills
    for skill_args in (["--checkpoint", str(paths[TEE])],
                       ["--level", "2", "--tee", str(paths[TEE]), "--box", str(paths[BOX])],
                       ["--level", "3", "--tee", str(paths[TEE]), "--box", str(paths[BOX]), "--dock", str(paths[DOCK])]):
        with pytest.raises(Built):
            runner.main(skill_args + ["--robot", "sim", "--camera", "render", "--mount", "back", "--episodes", "1"])
    assert mounts == ["back"] * 3


@pytest.mark.skipif(not all(p.exists() for p in TRAINED.values()), reason="needs the lightbox skills")
@pytest.mark.parametrize("level, tool, tee, box, tee_goal, box_goal, plan", [
    # Held-out scenes (seed 20000003): Level 2 box first, Level 3 box to the pre-dock spot, the T, then docking.
    (2, (0.155, 0.1425), (0.1765, 0.0786, -1.6795), (0.177, -0.0114, 0.0781), (0.2016, 0.0674, -1.7053),
     (0.2086, -0.044, 3.1313), [BOX, TEE]),
    (3, (0.2526, -0.0369), (0.1948, 0.0436, 1.1142), (0.1693, -0.078, -2.0716), (0.1834, 0.0087, 1.3751), None,
     [BOX, TEE, DOCK])])
def test_trained_skills_do_a_level_through_the_hardware_path(level, tool, tee, box, tee_goal, box_goal, plan):
    from s2r2s.train import load_policy
    _, cfg, _ = load_policy(TRAINED[TEE])
    robot = SimRobot(cfg.scene, encoder_noise_deg=0.05)
    robot.place(tool, tee)
    robot.place_block(box)
    runner = LevelRunner(level, TRAINED, robot, SimPoseSource(robot, seed=1), JointMap(), realtime=False)
    runner.start_level(np.array(tee_goal), None if box_goal is None else np.array(box_goal))
    assert [w for w, _ in runner.plan] == plan
    for _ in range(round(runner.episode_seconds / runner.dt)):
        if runner.tick() == "done":
            break
    assert runner.done and runner.summary()["all_settled"]
    assert runner.outcome(robot.object_pose(), robot.block_pose())["success"]
