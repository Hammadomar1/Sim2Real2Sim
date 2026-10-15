import math

import mujoco
import numpy as np

from s2r2s.env import EnvConfig, PushEnv, RandomizationConfig, TaskConfig
from s2r2s.kinematics import PusherKinematics
from s2r2s.scene import LIGHTBOX_MAT_X, SceneConfig, build_model


def make(n=64, seed=0, randomize=True, **rand):
    cfg = EnvConfig(num_envs=n, seed=seed, num_threads=4, task=TaskConfig(easy_fraction=0.0),
                    rand=RandomizationConfig(enabled=randomize, **rand))
    cfg.scene.lightbox = True
    return PushEnv(cfg)


def test_the_mat_is_the_work_surface_and_the_robot_stands_below_it():
    scene = SceneConfig(timestep=0.005, lightbox=True)
    m = build_model(scene, visual=False)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    top = lambda g: m.geom_pos[m.geom(g).id, 2] + m.geom_size[m.geom(g).id, 2]
    assert abs(top("table")) < 1e-12 and abs(top("floor") + scene.mat_height) < 1e-12
    assert abs(d.body("base").xpos[2] + scene.mat_height) < 1e-12
    near_edge = m.geom_pos[m.geom("table").id, 0] - m.geom_size[m.geom("table").id, 0]
    assert abs(near_edge - LIGHTBOX_MAT_X[0]) < 1e-12
    assert abs(d.cam("c920").xpos[2] - (scene.c920_pos[2] - scene.mat_height)) < 1e-12
    # Same joints, same tip relative to the robot: the lightbox only moves the work surface.
    flat = build_model(SceneConfig(timestep=0.005), visual=False)
    q = np.array([[0.1, -0.2, 0.5, 1.2, 0.0]])
    tip_box, _ = PusherKinematics(m).forward(q)
    tip_flat, _ = PusherKinematics(flat).forward(q)
    assert np.allclose(tip_box - tip_flat, [0, 0, -scene.mat_height], atol=1e-12)


def test_the_rod_reaches_the_pushing_band_above_the_mat():
    # Standing 9.8 mm lower, the arm's vertical-rod reach shrinks by about 7 mm: the ring's outermost corners
    # (r > 27.4 cm beyond +-53 deg, 0.025 % of it) are just out of reach; the controller clamps there.
    m = build_model(SceneConfig(timestep=0.005, lightbox=True), visual=False)
    kin = PusherKinematics(m)
    t = TaskConfig()
    rng = np.random.default_rng(1)
    n = 20000
    r, a = rng.uniform(*t.tool_r, n), rng.uniform(-t.tool_az, t.tool_az, n)
    tgt = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, t.tool_height)], -1)
    _, err, axis_z = kin.solve(tgt, kin.seed(tgt[:, :2]), iterations=30)
    assert (err < 1e-4).mean() > 0.999 and (axis_z > 0.9999).mean() > 0.999
    inner = r < 0.27                                    # everything the objects' failure band (r <= 26.5 cm) needs
    assert err[inner].max() < 1e-4


def test_objects_rest_on_the_mat_and_a_push_off_its_edge_is_a_failure():
    env = make(8, randomize=False)
    pose0 = env.object_pose().copy()
    for _ in range(20):
        _, _, done, _ = env.step(np.zeros((env.n, 2)))
        assert not done.any()
    assert np.abs(env.object_pose() - pose0).max() < 1e-4
    # World 0: the T's centre inside the pushing band (r 14 cm at 40 deg) but 11 mm beyond the mat's near edge,
    # bar towards the robot. It falls to the floor board, 9.8 mm down.
    o = env.qpos_adr + env.idx.obj_qpos
    x, y = 0.14 * math.cos(math.radians(40)), 0.14 * math.sin(math.radians(40))
    assert x < LIGHTBOX_MAT_X[0] - 0.01
    env.state[0, o:o + 3] = [x, y, env.obj_height / 2]
    env.state[0, o + 3:o + 7] = [0, 0, 0, 1]
    for _ in range(10):
        _, _, done, info = env.step(np.zeros((env.n, 2)))
        if done[0]:
            break
    k = list(info["terminal_ids"]).index(0)
    assert done[0] and info["episode"]["dropped"][k] and not info["episode"]["outside"][k]


def test_joint_calibration_error_moves_the_arm_not_the_reading():
    env = make(256, joint_offset_deg=2.0)
    assert (np.abs(env.joint_offset[:, :4]) <= math.radians(2.0) + 1e-12).all()
    assert (env.joint_offset[:, 4] == 0).all() and np.abs(env.joint_offset).max() > math.radians(1.5)
    # What the encoders read puts the rod where IK sent it; the arm itself is elsewhere.
    true_q = env.state[:, env.qpos_adr + env.idx.arm_qpos]
    true_tip, _ = env.kin.forward(true_q)
    assert np.allclose(env.tool_position()[:, 2], env.tool_z_cmd, atol=1e-4)
    miss = np.linalg.norm(true_tip[:, :2] - env.tool_position()[:, :2], axis=-1)
    assert np.median(miss) > 1e-3
    for _ in range(10):
        env.step(np.zeros((env.n, 2)))
    assert np.allclose(env.qarm, env.state[:, env.qpos_adr + env.idx.arm_qpos] + env.joint_offset)


def test_hardware_errors_do_not_change_the_scenes():
    plain, hard = make(64, seed=5), make(64, seed=5, joint_offset_deg=1.0, backlash_deg=(1.0, 1.0, 1.0, 1.0))
    assert np.allclose(plain.object_pose(), hard.object_pose()) and np.allclose(plain.goal, hard.goal)


def test_backlash_holds_the_link_inside_the_play():
    play = 2.0
    env = make(16, backlash_deg=(play, 0.0, 0.0, 0.0))
    half = math.radians(play) / 2
    for k in range(30):
        action = np.tile([0.0, 1.0 if k < 15 else -1.0], (env.n, 1))     # sweep left, then reverse
        env.step(action)
        cmd = env.q_cmd - env.joint_offset
        lag = env.link_target - cmd
        assert (np.abs(lag[:, 0]) <= half + 1e-12).all() and np.allclose(lag[:, 1:], 0)
        if k in (14, 29):          # while the command keeps going one way, the link trails it by the full half play
            assert np.isclose(np.abs(lag[:, 0]), half).mean() > 0.8      # (a world that just reset starts at 0)
