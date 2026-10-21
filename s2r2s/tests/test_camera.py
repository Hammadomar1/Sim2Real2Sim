"""The camera pose estimator on rendered C920 frames (hardware/vision.py, camera_sim.py, camera_calibrate.py)."""
import math
import time

import mujoco
import numpy as np
import pytest

from s2r2s.hardware.arm_mask import ArmMask
from s2r2s.hardware.camera import ReplaySource
from s2r2s.hardware.camera_calibrate import (board, calibrate_intrinsics, default_layout, detect_board, detect_markers,
                                             learn_colours, merge_detections, solve_extrinsics)
from s2r2s.hardware.camera_sim import Degradation, SimCamera, band_poses
from s2r2s.hardware.poses import CameraPoseSource
from s2r2s.hardware.vision import DEFAULT_COLOURS, PoseEstimator, region_integral
from s2r2s.kinematics import PusherKinematics
from s2r2s.objects import OBJECTS
from s2r2s.scene import SceneConfig


def _yaw_err(name, a, b):
    s = OBJECTS[name].symmetry
    period = 2 * math.pi / s if s > 0 else 2 * math.pi
    return abs(math.degrees((a - b + period / 2) % period - period / 2))


@pytest.fixture(scope="module")
def sim():
    try:
        return SimCamera()
    except Exception as error:           # no OpenGL context (headless machine)
        pytest.skip(f"cannot render: {error}")


@pytest.fixture(scope="module")
def estimator(sim):
    return PoseEstimator(sim.camera(), DEFAULT_COLOURS, ("tee", "box"))


def test_camera_model_projects_where_mujoco_renders(sim):
    sim.place(tee=(0.22, -0.04, 0.6), box=(0.30, 0.10, 0.3))
    cam = sim.true_camera                  # the pinhole renders, before any added lens distortion
    r = mujoco.Renderer(sim.model, cam.height, cam.width)
    r.enable_segmentation_rendering()
    r.update_scene(sim.data, camera="c920")
    seg = r.render()
    g = sim.model.geom("object_0").id
    ys, xs = np.nonzero((seg[..., 0] == g) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
    pos, rot, half = sim.data.geom_xpos[g], sim.data.geom_xmat[g].reshape(3, 3), sim.model.geom_size[g]
    corners = [pos + rot @ (np.array(s) * half) for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
    uv = cam.project(np.array(corners))
    # Pixel centres on integers: the outermost covered pixels lie within half a pixel of the projected box.
    assert abs(uv[:, 0].min() - xs.min()) < 1.0 and abs(uv[:, 0].max() - xs.max()) < 1.0
    assert abs(uv[:, 1].min() - ys.min()) < 1.0 and abs(uv[:, 1].max() - ys.max()) < 1.0
    # Back-projection inverts projection (with lens distortion).
    dist_cam = sim.camera()
    pts = np.array([[0.2, -0.1, 0.02], [0.35, 0.15, 0.0]])
    assert np.allclose(dist_cam.backproject(dist_cam.project(pts), pts[0, 2])[0], pts[0], atol=1e-6)


def test_region_integral_is_exact_and_continuous():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(40, 50))
    C = np.zeros((40, 51))
    C[:, 1:] = np.cumsum(W, 1)
    p = np.array([[10.3, 5.7], [40.1, 12.2], [33.3, 38.9], [8.8, 30.4]])
    ss = 20                                # brute force: supersampled point-in-polygon
    ys, xs = np.mgrid[-0.5:39.5:1 / ss, -0.5:49.5:1 / ss] + 0.5 / ss
    from matplotlib.path import Path as MplPath
    inside = MplPath(p).contains_points(np.c_[xs.ravel(), ys.ravel()]).reshape(xs.shape)
    brute = (W[np.floor(ys + 0.5).astype(int), np.floor(xs + 0.5).astype(int)] * inside).sum() / ss ** 2
    exact = region_integral([p], [1.0], C, W)
    assert abs(exact - brute) < 0.05 and abs(region_integral([p[::-1].copy()], [1.0], C, W) - exact) < 1e-9
    ones = np.ones((40, 50))
    assert abs(region_integral([p], [1.0], np.c_[np.zeros(40), np.cumsum(ones, 1)], ones) - 728.88) < 1e-9
    # Moving a vertex by a hair changes the integral by a hair: no pixel steps for the optimiser.
    q = p.copy()
    q[1, 0] += 1e-4
    assert abs(region_integral([q], [1.0], C, W) - exact) < 1e-2


def test_poses_from_rendered_frames_are_accurate_and_never_swapped(sim, estimator):
    rng = np.random.default_rng(3)
    for _ in range(4):
        tee, box = band_poses(rng, np.array([0.13, -0.18, 0.0]), OBJECTS["tee"].radius, OBJECTS["box"].radius)
        sim.place(tee=tee, box=box)
        estimator.last.clear()
        found = estimator.estimate(sim.render(rng))
        truth = sim.truth()
        for name in ("tee", "box"):
            pose = found[name].pose
            assert pose is not None, (name, found[name].status)
            assert np.linalg.norm(pose[:2] - truth[name][:2]) < 1.5e-3, (name, pose, truth[name])
            assert _yaw_err(name, pose[2], truth[name][2]) < 1.5
        assert found["tee"].visible > 0.9 and found["box"].visible > 0.9


def _arm_around_tee(sim, tee):
    """Arm configurations (rod tip around the T) and how much of the T each leaves visible to the camera."""
    kin = PusherKinematics(sim.model)
    out = []
    for dx in (-0.05, -0.03, 0.0, 0.03, 0.05):
        for dy in (-0.05, -0.03, 0.0, 0.03, 0.05):
            if dx == dy == 0.0:
                continue
            target = np.array([[tee[0] + dx, tee[1] + dy, 0.005]])
            q, err, _ = kin.solve(target, kin.seed(target[:, :2]), iterations=30)
            if err[0] > 1e-3:
                continue
            sim.data.qpos[sim.idx.arm_qpos] = q[0]
            mujoco.mj_kinematics(sim.model, sim.data)
            out.append((q[0].copy(), sim.visible_share()["tee"]))
    return out


def test_arm_over_the_tee_with_the_arm_mask(sim, estimator):
    rng = np.random.default_rng(5)
    tee = np.array([0.20, 0.06, 0.8])
    sim.place(tee=tee, box=(0.33, -0.12, 0.0))
    configs = _arm_around_tee(sim, tee)
    clear = [c for c in configs if c[1] > 0.98]
    partly = [c for c in configs if 0.3 < c[1] < 0.85]
    assert clear and partly, [round(c[1], 2) for c in configs]
    arm = ArmMask(sim.camera(), estimator.grid, sim.scene, margin=0.006)

    def look(q, tracked):
        sim.data.qpos[sim.idx.arm_qpos] = q
        mujoco.mj_kinematics(sim.model, sim.data)
        estimator.last.clear()
        if tracked is not None:                  # seen a moment ago, before the arm came over
            estimator.last["tee"] = (tracked, time.monotonic())
        return estimator.estimate(sim.render(rng), hidden=arm(q + np.radians(rng.uniform(-0.5, 0.5, 5))))["tee"]

    seen = look(clear[0][0], None)
    assert seen.pose is not None
    for q, share in partly:
        for tracked in (seen.pose, None):
            det = look(q, tracked)
            if tracked is not None:              # tracking: a pose down to a quarter visible
                assert det.pose is not None, (det.status, det.visible, share)
            if det.pose is not None:             # accepted means right, however much is hidden
                assert np.linalg.norm(det.pose[:2] - tee[:2]) < 2e-3 and _yaw_err("tee", det.pose[2], tee[2]) < 2.0
                assert abs(det.visible - share) < 0.15
    for q, share in [c for c in configs if c[1] < 0.15]:     # nearly all hidden: no pose rather than a guess
        assert look(q, seen.pose).pose is None


def test_calibration_chain_on_rendered_targets():
    rng = np.random.default_rng(7)
    scene = SceneConfig(lightbox=True, clutter="box")
    brd = board()
    try:
        sim = SimCamera(scene, board=brd)
    except Exception as error:
        pytest.skip(f"cannot render: {error}")
    true = sim.camera()
    views = []
    for k in range(40):
        if len(views) >= 12:
            break
        u, v = 300 + (k % 4) * 440, 220 + (k // 4 % 3) * 320
        ray = true.R.T @ np.linalg.solve(true.K, [u, v, 1.0])
        ray /= np.linalg.norm(ray)
        z = -ray
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        import cv2
        rot = np.stack([x, np.cross(z, x), z], 1) @ cv2.Rodrigues(np.r_[rng.uniform(-0.6, 0.6, 2), 0.0])[0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        sim.place_board(true.position + ray * rng.uniform(0.3, 0.5), quat)
        view = detect_board(sim.render(rng), brd)
        if view is not None:
            views.append(view)
    intr = calibrate_intrinsics(views, (true.width, true.height))
    assert abs(intr["fovy_deg"] - true.fovy_deg) < 0.2 and intr["rms_px"] < 0.5
    assert abs(intr["dist"][0] - true.dist[0]) < 0.03
    layout = default_layout()
    sim = SimCamera(scene, markers=layout)
    sim.place(tee=(0.27, -0.06, 0.4), box=(0.30, 0.09, 0.2))
    detected = merge_detections([detect_markers(sim.render(rng)) for _ in range(3)])
    assert len(detected) >= 5
    extr = solve_extrinsics(detected, layout, intr["K"], intr["dist"], (true.width, true.height))
    assert np.linalg.norm(np.array(extr["camera_position_m"]) - true.position) < 3e-3
    assert max(extr["held_out_mm"].values()) < 1.0


def test_learned_colours_tell_the_tee_from_the_box(sim):
    sim.data.qpos[sim.idx.arm_qpos] = sim.model.qpos0[sim.idx.arm_qpos]       # the arm parked off the mat
    sim.place(tee=(0.22, -0.08, 2.0), box=(0.27, 0.10, 0.4))
    img = sim.render(np.random.default_rng(9))
    colours, report = learn_colours(img, sim.camera())
    assert colours.objects["tee"].hue[0] <= 110 <= colours.objects["tee"].hue[1]          # blue
    assert report["tee"]["catches_box"] < 0.01 and report["box"]["catches_tee"] < 0.01
    found = PoseEstimator(sim.camera(), colours, ("tee", "box")).estimate(img)
    for name, truth in sim.truth().items():
        assert np.linalg.norm(found[name].pose[:2] - truth[:2]) < 1.5e-3


def test_camera_pose_source_on_replayed_frames(sim):
    rng = np.random.default_rng(11)
    sim.place(tee=(0.21, -0.03, -0.7), box=(0.31, 0.11, 0.2))
    truth = sim.truth()
    frames = [sim.render(rng) for _ in range(3)]
    poses = CameraPoseSource(source=ReplaySource(frames, fps=30, loop=True), camera=sim.camera(),
                             colours=DEFAULT_COLOURS, scene=sim.scene)
    try:
        poses.set_arm(sim.data.qpos[sim.idx.arm_qpos])
        deadline = time.monotonic() + 10
        pose, block = None, None
        while (pose is None or block is None) and time.monotonic() < deadline:
            time.sleep(0.05)
            pose, stamp = poses.read()
            block, _ = poses.read_block()
        assert pose is not None and block is not None
        assert np.linalg.norm(pose[:2] - truth["tee"][:2]) < 1.5e-3
        assert np.linalg.norm(block[:2] - truth["box"][:2]) < 1.5e-3
        assert time.monotonic() - stamp < 1.0
        assert "frames" in poses.summary()
    finally:
        poses.close()


def test_rendered_pose_source_renders_estimates_and_masks_from_its_mount(sim):
    from s2r2s.hardware.camera_sim import MOUNTS, RenderedPoseSource
    from s2r2s.hardware.robots import SimRobot
    scene = SceneConfig(lightbox=True, clutter="box")
    robot = SimRobot(scene)
    robot.place((0.15, -0.12), (0.22, 0.05, 0.3))
    robot.place_block((0.25, -0.08, 0.0))
    left = RenderedPoseSource(robot, scene)
    assert np.allclose(left.sim.camera().position, sim.camera().position)       # default: the scene's slot
    back = RenderedPoseSource(robot, scene, mount="back")
    assert back.mount == "back" and scene.c920_pos == SceneConfig().c920_pos   # the caller's scene is kept
    pos, target = MOUNTS["back"]
    lens = np.array(pos) - (0.0, 0.0, scene.mat_height)
    cam = back.sim.camera()
    assert np.allclose(cam.position, lens, atol=1e-6)
    assert np.allclose(cam.project(np.array([target]))[0], (cam.width / 2, cam.height / 2), atol=1.0)
    # The estimator, the arm mask and the shadow predictor see from the same lens.
    assert np.allclose(back.estimator.camera.position, lens, atol=1e-6)
    for mask in (back.arm_mask, back.view.arm):
        assert np.allclose([*mask.foot, mask.cz], lens, atol=1e-6)
    back.set_arm(robot.data.qpos[robot.idx.arm_qpos])
    tee, _ = back.read()
    box, _ = back.read_block()
    assert np.linalg.norm(tee[:2] - robot.object_pose()[:2]) < 1.5e-3
    assert np.linalg.norm(box[:2] - robot.block_pose()[:2]) < 1.5e-3
