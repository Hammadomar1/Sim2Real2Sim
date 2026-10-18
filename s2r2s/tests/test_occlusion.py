import numpy as np
import pytest

from s2r2s.env import EnvConfig, PushEnv, RandomizationConfig, TaskConfig
from s2r2s.occlusion import LIGHTBOX_MOUNTS, FacePoints, OcclusionCamera, mount_position
from s2r2s.scene import SceneConfig
from s2r2s.tasks import make_env


def make(n=32, seed=0, randomize=True, clutter="", **rand):
    cfg = EnvConfig(num_envs=n, seed=seed, num_threads=4, task=TaskConfig(easy_fraction=0.0, clutter=clutter),
                    rand=RandomizationConfig(enabled=randomize, **rand))
    cfg.scene.lightbox = True
    return make_env(cfg)


def test_mounts():
    scene = SceneConfig(lightbox=True)
    assert np.allclose(mount_position("c920", scene), LIGHTBOX_MOUNTS["c920_left_wall"], atol=1e-9)
    assert np.allclose(mount_position("0.4,-0.1,0.3", scene), [0.4, -0.1, 0.3])
    with pytest.raises(ValueError):
        mount_position("ceiling", scene)


def test_the_arm_hides_what_is_under_it_and_nothing_else():
    env = make(8, randomize=False)
    tip = env.tool_position()
    qpos = env.state[:, env.qpos_adr:env.qpos_adr + env.idx.nq]
    kind = np.zeros(env.n, dtype=int)
    face = FacePoints(["tee"], 3)
    for i in range(env.n):
        # A camera straight above the rod: a T centred under it is hidden by the rod, the gripper and the wrist.
        cam = OcclusionCamera(env.cfg.scene, f"{tip[i, 0]},{tip[i, 1]},0.45", nthread=1)
        under = cam.view(qpos[i:i + 1], [(np.r_[tip[i, :2], 0.0][None], kind[:1])], face)[0, 0]
        assert under[0] < 0.25
        # The same T 15 cm to the other side of the robot's centre line is in full view.
        far = np.array([[0.2, -0.15 * np.sign(tip[i, 1]), 0.0]])
        assert cam.view(qpos[i:i + 1], [(far, kind[:1])], face)[0, 0, 0] == 1.0


def test_the_shift_points_to_the_visible_part():
    face = FacePoints(["tee"], 3)
    cam = OcclusionCamera(SceneConfig(lightbox=True), "back_wall_low", nthread=1)
    pts = face.world(np.array([[0.2, 0.0, 0.0]]), np.array([0]))
    bar = pts[0, :9]                               # the first rectangle's points (the T's bar)
    cam.hidden = lambda qpos, p: np.arange(p.shape[1])[None] >= 9    # hide everything but the bar
    v = cam.view(np.zeros((1, 1)), [(np.array([[0.2, 0.0, 0.0]]), np.array([0]))], face)[0, 0]
    assert np.isclose(v[0], 0.5)
    assert np.allclose(v[1:], bar[:, :2].mean(0) - pts[0, :, :2].mean(0))


def test_occlusion_is_off_by_default_and_draws_no_randomness():
    plain = make(32, seed=3)
    assert plain.occluder is None
    # A threshold of 0 drops nothing: every observation matches the camera without occlusion, step for step.
    seen = make(32, seed=3, occlusion_camera="c920")
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = rng.uniform(-1, 1, (32, 2))
        o1, *_ = plain.step(a)
        seen.cfg.rand.occlusion_threshold = 0.0
        o2, *_ = seen.step(a)
        assert np.array_equal(o1["actor"], o2["actor"])
    assert (seen.view_hist[:, :, 0, 0] <= 1).all() and (seen.view_hist[:, 0, 0, 0] < 1).any()


def test_hidden_frames_hold_the_last_estimate():
    # A threshold above 1 hides every frame: the estimate stays at the start pose while the arm pushes.
    env = make(16, randomize=False, occlusion_camera="c920", occlusion_threshold=1.01)
    start = env.obs_pose.copy()
    for k in range(60):
        _, _, done, info = env.step(np.tile([1.0, 0.0], (env.n, 1)))
        live = env.step_count == k + 1
        assert np.array_equal(env.obs_pose[live], start[live])
        if "episode" in info:
            assert (info["episode"]["occluded"] == 1).all()
    assert (env.occluded_longest == env.step_count).all()


def test_the_clutter_block_is_tracked_too():
    env = make(8, clutter="box", occlusion_camera="c920", occlusion_threshold=1.01, randomize=False)
    assert env.view_hist.shape[2] == 2
    home = env.clutter_obs.copy()
    for _ in range(10):
        env.step(np.zeros((env.n, 2)))
    assert np.array_equal(env.clutter_obs, home)
