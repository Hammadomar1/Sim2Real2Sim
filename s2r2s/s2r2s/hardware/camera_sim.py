"""The camera pose estimator on simulated C920 images: accuracy and dropped frames before the real camera.

    python -m s2r2s.hardware.camera_sim                          # 300 scenes x 3 views, true camera model
    python -m s2r2s.hardware.camera_sim --calibrated             # camera model from simulated ChArUco + ArUco
    python -m s2r2s.hardware.camera_sim --scenes 50 --out runs/camera_sim/quick.json

``SimCamera`` renders the lightbox scene's ``c920`` camera at 1920x1080 the way the webcam would deliver it
(lens distortion, blur, sensor noise, exposure and white-balance error, MJPG compression, varied lighting),
with the T and the box in distinct colours and the simulator's goal and estimate markers removed. The frames
go through the same ``PoseEstimator`` that ``CameraPoseSource`` runs on the real camera.

Frames come from pushing episodes of a clutter policy in the lightbox (``--checkpoint``, default
runs/clutter_v7): the first frame of each episode with the T and the box moved anywhere in the pushing band
(``band``: the arm mostly out of the way), and frames while the policy pushes (``pushing``: the arm next to
and over the T), and whole episodes at 20 Hz (``--sequences``: tracking, and how long the drops last). The
estimator gets the arm's mask from joint angles with up to +-0.5 deg error per joint, two frames (100 ms) old
in sequences (``--no-arm-mask`` to go without). The true visible share of each object (segmentation renders
with and without the arm) says how much the arm hides. Reported: position and yaw error percentiles of accepted estimates, the dropped-frame
rate by visible share, wrong estimates that were accepted, and processing time.

``--calibrated`` replaces the true camera model with the calibration the real procedure would produce:
intrinsics from rendered ChArUco views (``camera_calibrate.calibrate_intrinsics``) and extrinsics from
rendered ArUco markers on the mat (``camera_calibrate.calibrate_extrinsics``), so the numbers include
calibration error.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ..objects import OBJECTS
from ..scene import LIGHTBOX_MAT_HALF_WIDTH, LIGHTBOX_MAT_X, SceneConfig, SceneIndex, build_spec
from .arm_mask import ArmMask, ViewPredictor
from .vision import CameraModel, DEFAULT_COLOURS, EstimatorConfig, PoseEstimator, draw

PROJECT = Path(__file__).resolve().parents[2]
# Distinct, saturated object colours (DEFAULT_COLOURS segments them); the arm stays Menagerie yellow and black.
SIM_RGBA = {"tee": (0.15, 0.35, 0.90, 1.0), "box": (0.85, 0.15, 0.45, 1.0)}
# Where the C920 can go (camera_study.LIGHTBOX_CANDIDATES; SceneConfig convention: z above the floor board, so
# the mat height is added): lens position and the point it looks at.
MOUNTS = {"left": ((0.2022, 0.3662, 0.40), (0.2703, 0.0, 0.0)),            # the slot as built
          "right": ((0.2022, -0.3662, 0.40), (0.2703, 0.0, 0.0)),          # its mirror in the right wall
          "back": ((0.465, 0.0, 0.25 + 0.0098), (0.20, 0.0, 0.0))}         # back wall, centre, 25 cm above the mat
# Typical C920 radial distortion at 1080p (published calibrations: k1 0.05-0.15, k2 -0.1 to -0.3).
C920_DISTORTION = (0.10, -0.20, 0.0, 0.0, 0.0)


@dataclass
class Degradation:
    """What a real webcam adds to a perfect render. Ranges are drawn per frame."""
    noise: float = 2.0             # sensor noise, standard deviation in 8-bit levels
    jpeg: int = 85                 # MJPG quality (0: no compression)
    blur: float = 0.7              # Gaussian blur sigma (px): focus and demosaicing
    gain: float = 0.20             # exposure error: brightness scaled by up to +-this
    white_balance: float = 0.06    # per-channel gain error
    light: float = 0.30            # scene lights scaled by up to +-this
    distortion: tuple = C920_DISTORTION

    @classmethod
    def none(cls):
        return cls(0.0, 0, 0.0, 0.0, 0.0, 0.0, (0.0,) * 5)


class SimCamera:
    """The lightbox C920 in the simulator.

    ``markers``: ``camera_calibrate.Marker`` s lying on the mat (extrinsics);
    ``board``: a ChArUco board (``camera_calibrate.board()``) on a movable body (intrinsics);
    ``rod_marker``: the ArUco marker taped to the rod (``rod_marker.RodMarker``: backlash and latency measurements).
    """

    def __init__(self, scene: SceneConfig | None = None, object_name: str = "tee", width: int = 1920, height: int = 1080,
                 degradation: Degradation | None = None, markers=(), board=None, texture_dir: Path | None = None,
                 rod_marker=None):
        scene = copy.deepcopy(scene) if scene is not None else SceneConfig(lightbox=True, clutter="box")
        scene.lightbox = True
        self.scene, self.object_name, self.width, self.height = scene, object_name, width, height
        self.deg = degradation if degradation is not None else Degradation()
        spec = build_spec(scene, object_name, visual=True)
        # The real mat has no goal, estimate or home markers and no guide lines.
        for body in ("goal", "estimate", "clutter_home"):
            try:
                geoms = list(spec.body(body).geoms)
            except Exception:
                continue
            for g in geoms:
                spec.delete(g)
        for g in list(spec.worldbody.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE and g.name == "":
                spec.delete(g)
        # Matte printed parts in distinct colours (MuJoCo's default material is fairly glossy).
        for prefix, name in (("object_", object_name), ("clutter_", scene.clutter)):
            if not name:
                continue
            mat = spec.add_material(name=f"{prefix}matte", rgba=list(SIM_RGBA[name]))
            mat.specular, mat.shininess = 0.15, 0.2
            for g in spec.geoms:
                if g.name.startswith(prefix):
                    g.material, g.rgba = f"{prefix}matte", list(SIM_RGBA[name])
        if markers or board is not None or rod_marker is not None:
            from .camera_calibrate import marker_image
            tex_dir = Path(texture_dir or PROJECT / "runs" / "camera_sim" / "textures")
            tex_dir.mkdir(parents=True, exist_ok=True)
        if rod_marker is not None:         # a card on the rod, flat (gripper +z is up when the rod is vertical)
            mk = rod_marker
            png = tex_dir / f"rod_marker_{mk.id}_{mk.size * 1e3:.0f}_{mk.margin * 1e3:.0f}.png"
            cv2.imwrite(str(png), marker_image(mk.id, mk.size, mk.margin, px_per_mm=8))
            spec.add_texture(name="rod_marker", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
            mat = spec.add_material(name="rod_marker", textures=["", "rod_marker"], texuniform=False, texrepeat=[1, 1])
            mat.specular, mat.shininess = 0.0, 0.0
            half = mk.size / 2 + mk.margin
            spec.body("gripper").add_geom(name="rod_marker", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.0005],
                                          pos=list(mk.sim_pos), material="rod_marker", contype=0, conaffinity=0)
        for mk in markers:
            png = tex_dir / f"marker_{mk.id}_{mk.size * 1e3:.0f}_{mk.margin * 1e3:.0f}.png"
            cv2.imwrite(str(png), marker_image(mk.id, mk.size, mk.margin, px_per_mm=8))
            spec.add_texture(name=f"marker{mk.id}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
            mat = spec.add_material(name=f"marker{mk.id}", textures=["", f"marker{mk.id}"], texuniform=False,
                                    texrepeat=[1, 1])
            mat.specular, mat.shininess = 0.0, 0.0
            half, turn = mk.size / 2 + mk.margin, mk.yaw - math.pi / 2     # texture right/up = geom x/y
            spec.worldbody.add_geom(name=f"marker{mk.id}", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[half, half, 0.01],
                                    pos=[mk.centre[0], mk.centre[1], mk.z], quat=[math.cos(turn / 2), 0, 0, math.sin(turn / 2)],
                                    material=f"marker{mk.id}", contype=0, conaffinity=0)
        self.board = board
        if board is not None:
            png = tex_dir / "charuco.png"
            sx, sy = board.getChessboardSize()
            sq = board.getSquareLength()
            cv2.imwrite(str(png), board.generateImage((int(sx * sq * 4000), int(sy * sq * 4000)), marginSize=0))
            spec.add_texture(name="charuco", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
            mat = spec.add_material(name="charuco", textures=["", "charuco"], texuniform=False, texrepeat=[1, 1])
            mat.specular, mat.shininess = 0.0, 0.0
            body = spec.worldbody.add_body(name="board", mocap=True, pos=[5.0, 5.0, -1.0])
            body.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[sx * sq / 2, sy * sq / 2, 0.01],
                          material="charuco", contype=0, conaffinity=0)
        self.model = spec.compile()
        m = self.model
        self.data = mujoco.MjData(m)
        self.idx = SceneIndex(m)
        self.renderer = mujoco.Renderer(m, height, width)
        self.seg = mujoco.Renderer(m, height // 4, width // 4)
        self.light0 = (m.light_diffuse.copy(), m.light_ambient.copy(), m.vis.headlight.diffuse.copy(),
                       m.vis.headlight.ambient.copy())
        mujoco.mj_forward(m, self.data)
        self.true_camera = CameraModel.from_mujoco(m, self.data, "c920", width, height)
        self._distortion_maps = None
        if any(self.deg.distortion):
            # Distorted pixel p_d shows what the pinhole render has at undistort(p_d).
            u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
            pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
            K = self.true_camera.K
            und = cv2.undistortPoints(pts, K, np.asarray(self.deg.distortion, float), P=K).reshape(height, width, 2)
            self._distortion_maps = cv2.convertMaps(und[..., 0].astype(np.float32), und[..., 1].astype(np.float32),
                                                    cv2.CV_16SC2)

    def camera(self) -> CameraModel:
        """The exact camera behind the rendered images (with the distortion the renders were given)."""
        cam = copy.deepcopy(self.true_camera)
        cam.dist = np.asarray(self.deg.distortion, float)
        return cam

    # -- the scene
    def set_qpos(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)

    def place(self, tee=None, box=None):
        """Move the object (T) and the clutter block to planar poses (x, y, yaw), keeping the arm."""
        for pose, adr, shape in ((tee, self.idx.obj_qpos, OBJECTS[self.object_name]),
                                 (box, getattr(self.idx, "clutter_qpos", None), OBJECTS.get(self.scene.clutter))):
            if pose is None or adr is None:
                continue
            x, y, yaw = pose
            self.data.qpos[adr:adr + 7] = [x, y, shape.height / 2, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_kinematics(self.model, self.data)

    def truth(self) -> dict[str, np.ndarray]:
        def planar(o):
            q = self.data.qpos[o + 3:o + 7]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            return np.array([self.data.qpos[o], self.data.qpos[o + 1], yaw])
        out = {self.object_name: planar(self.idx.obj_qpos)}
        if self.scene.clutter:
            out[self.scene.clutter] = planar(self.idx.clutter_qpos)
        return out

    def on_mat(self) -> dict[str, bool]:
        """Whether each object rests on the mat (not fallen 9.8 mm onto the floor board past its edge, nor tilted)."""
        out = {}
        for name, adr in ((self.object_name, self.idx.obj_qpos), (self.scene.clutter, getattr(self.idx, "clutter_qpos", None))):
            if name and adr is not None:
                q = self.data.qpos[adr + 3:adr + 7]
                upright = q[1] ** 2 + q[2] ** 2 < 1e-4                  # no roll or pitch beyond about 1 deg
                out[name] = bool(upright and abs(self.data.qpos[adr + 2] - OBJECTS[name].height / 2) < 0.002)
        return out

    def place_board(self, pos, quat):
        mid = self.model.body_mocapid[self.model.body("board").id]
        self.data.mocap_pos[mid], self.data.mocap_quat[mid] = pos, quat
        mujoco.mj_kinematics(self.model, self.data)

    # -- images
    def render(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """A BGR frame as cv2.VideoCapture would deliver it."""
        rng = rng or np.random.default_rng()
        d, m = self.deg, self.model
        diffuse, ambient, head_d, head_a = self.light0
        m.light_diffuse[:] = diffuse * rng.uniform(1 - d.light, 1 + d.light, (len(diffuse), 1))
        m.vis.headlight.diffuse[:] = head_d * rng.uniform(1 - d.light, 1 + d.light)
        m.vis.headlight.ambient[:] = head_a * rng.uniform(1 - d.light, 1 + d.light)
        self.renderer.update_scene(self.data, camera="c920")
        img = self.renderer.render()[:, :, ::-1].astype(np.float32)
        m.light_diffuse[:], m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = diffuse, head_d, head_a
        if self._distortion_maps is not None:
            img = cv2.remap(img, *self._distortion_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if d.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), d.blur)
        img *= rng.uniform(1 - d.gain, 1 + d.gain) * rng.uniform(1 - d.white_balance, 1 + d.white_balance, 3)
        if d.noise > 0:
            img += rng.normal(0, d.noise, img.shape).astype(np.float32)
        img = np.clip(img + 0.5, 0, 255).astype(np.uint8)
        if d.jpeg:
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, d.jpeg])[1], cv2.IMREAD_COLOR)
        return img

    def visible_share(self) -> dict[str, float]:
        """True share of each object's silhouette the arm (and rod) leaves visible: segmentation renders at
        quarter resolution with and without the robot (geom groups 1 and 2)."""
        counts = []
        for show_robot in (True, False):
            opt = mujoco.MjvOption()
            opt.geomgroup[1] = opt.geomgroup[2] = show_robot
            self.seg.enable_segmentation_rendering()
            self.seg.update_scene(self.data, camera="c920", scene_option=opt)
            seg = self.seg.render()
            self.seg.disable_segmentation_rendering()
            ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
            counts.append({name: int(np.isin(ids, geoms).sum()) for name, geoms in self._object_geoms().items()})
        return {n: counts[0][n] / max(counts[1][n], 1) for n in counts[0]}

    def _object_geoms(self):
        m = self.model
        out = {self.object_name: [i for i in range(m.ngeom) if m.geom(i).name.startswith("object_")]}
        if self.scene.clutter:
            out[self.scene.clutter] = [i for i in range(m.ngeom) if m.geom(i).name.startswith("clutter_")]
        return out


class RenderedPoseSource:
    """The C920 pipeline on the simulated arm, for ``runner --robot sim --camera render``: every tick the SimRobot's
    scene is rendered as the C920 sees it, from ``latency_steps`` ticks ago, and goes through the pose estimator with
    the arm masked out at the joint angles the runner passes (``set_arm``), as ``CameraPoseSource`` does on the real
    camera. Synchronous and on simulated time, so a dry run gives the same result however fast it runs."""

    def __init__(self, robot, scene: SceneConfig, object_name: str = "tee", latency_steps: int = 2, dt: float = 0.05,
                 seed: int = 0):
        if not scene.lightbox:
            raise ValueError("the C920 is in the lightbox: use --lightbox (or a policy trained there)")
        self.robot, self.latency, self.dt = robot, latency_steps, dt
        self.sim = SimCamera(scene, object_name)
        if self.sim.model.nq != robot.model.nq:
            raise RuntimeError("the rendering model's state layout differs from the simulated robot's")
        cam = self.sim.camera()
        names = tuple(n for n in (self.sim.object_name, scene.clutter) if n)
        self.estimator = PoseEstimator(cam, DEFAULT_COLOURS, names)
        self.arm_mask = ArmMask(cam, self.estimator.grid, scene, margin=0.006)
        self.view = ViewPredictor(cam, scene, names, margin=0.006)
        self.object_name, self.block_name = self.sim.object_name, scene.clutter
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.history, self.q_arm, self.found, self.t = [], None, {}, 0.0
        self.estimator.last.clear()
        self.counts = {"frames": 0, "dropped": {}}

    def place_gate(self, centre, angle, width):
        from ..scene import set_gate
        set_gate(self.sim.model, centre, angle, width)

    def set_arm(self, q_sim, stamp=None):
        self.q_arm = np.array(q_sim, float)

    def read(self):
        """A new frame (call once per tick): the scene ``latency_steps`` ticks ago, estimated."""
        self.history.insert(0, self.robot.data.qpos.copy())
        del self.history[self.latency + 1:]
        self.sim.set_qpos(self.history[-1])
        self.t += self.dt
        hidden = None if self.q_arm is None else self.arm_mask(self.q_arm)
        self.found = self.estimator.estimate(self.sim.render(self.rng), self.t, hidden)
        self.counts["frames"] += 1
        for name, det in self.found.items():
            self.counts["dropped"][name] = self.counts["dropped"].get(name, 0) + (det.pose is None)
        return self._get(self.object_name)

    def read_block(self):
        """The block from the newest frame (read() renders it), at 60 % visible as CameraPoseSource has it."""
        return self._get(self.block_name, 0.6)

    def _get(self, name, min_visible: float = 0.0):
        det = self.found.get(name)
        return (None if det is None or det.pose is None or det.visible < min_visible else det.pose.copy()), self.t


# ---------------------------------------------------------------------------------------------- evaluation
def calibrated_camera(scene: SceneConfig, rng: np.random.Generator, degradation: Degradation | None = None,
                      views: int = 30, frames: int = 10) -> tuple[CameraModel, dict]:
    """The camera model the real procedure would produce here: intrinsics from rendered ChArUco views at
    random poses in front of the lens, extrinsics from the default mat markers (both through the functions
    camera_calibrate uses on the real camera)."""
    from .camera_calibrate import (board, calibrate_intrinsics, default_layout, detect_board, detect_markers,
                                   merge_detections, solve_extrinsics)
    brd = board()
    sim = SimCamera(scene, degradation=degradation, board=brd)
    true = sim.camera()
    found = []
    for _ in range(views * 3):
        if len(found) >= views:
            break
        u, v = rng.uniform(250, true.width - 250), rng.uniform(180, true.height - 180)
        ray = true.R.T @ np.linalg.solve(true.K, [u, v, 1.0])
        ray /= np.linalg.norm(ray)
        z = -ray                                                # the board faces the camera, then tilts and turns
        x0 = np.cross([0.0, 0.0, 1.0], z)
        x0 /= np.linalg.norm(x0)
        roll = rng.uniform(-math.pi, math.pi)
        x = math.cos(roll) * x0 + math.sin(roll) * np.cross(z, x0)
        rot = np.stack([x, np.cross(z, x), z], 1) @ cv2.Rodrigues(np.r_[rng.uniform(-0.7, 0.7, 2), 0.0])[0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        sim.place_board(true.position + ray * rng.uniform(0.28, 0.55), quat)
        view = detect_board(sim.render(rng), brd)
        if view is not None:
            found.append(view)
    intr = calibrate_intrinsics(found, (true.width, true.height))
    layout = default_layout()
    sim = SimCamera(scene, degradation=degradation, markers=layout)
    sim.place(tee=(0.27, -0.06, 0.4), box=(0.30, 0.09, 0.2))
    detected = merge_detections([detect_markers(sim.render(rng)) for _ in range(frames)])
    extr = solve_extrinsics(detected, layout, intr["K"], intr["dist"], (true.width, true.height))
    cam = CameraModel(intr["K"], intr["dist"], extr["R"], extr["t"], true.width, true.height)
    turn = cv2.Rodrigues(cam.R @ true.R.T)[0]
    info = {"views": intr["views"], "rms_px": intr["rms_px"], "fovy_deg": intr["fovy_deg"], "true_fovy_deg": true.fovy_deg,
            "dist": intr["dist"], "markers": extr["markers"], "mat_error_mm": extr["mat_error_mm"],
            "held_out_mm": extr["held_out_mm"], "position_error_mm": float(np.linalg.norm(cam.position - true.position) * 1e3),
            "rotation_error_deg": float(np.degrees(np.linalg.norm(turn)))}
    return cam, info


def _on_mat(pose, margin):
    return (LIGHTBOX_MAT_X[0] + margin <= pose[0] <= LIGHTBOX_MAT_X[1] - margin
            and abs(pose[1]) <= LIGHTBOX_MAT_HALF_WIDTH - margin)


def band_poses(rng: np.random.Generator, tip, tee_r: float, box_r: float):
    """The T and the box anywhere in the pushing band (r 13-26.5 cm, +-43 deg), wholly on the mat (up to its
    edges), apart from each other and from the rod."""
    def one():
        r, a = rng.uniform(0.13, 0.265), rng.uniform(-0.75, 0.75)
        return np.array([r * math.cos(a), r * math.sin(a), rng.uniform(-math.pi, math.pi)])
    while True:
        tee, box = one(), one()
        if (_on_mat(tee, tee_r + 0.002) and _on_mat(box, box_r + 0.002)
                and np.linalg.norm(tee[:2] - box[:2]) > tee_r + box_r + 0.005
                and np.linalg.norm(tee[:2] - tip[:2]) > tee_r + 0.01 and np.linalg.norm(box[:2] - tip[:2]) > box_r + 0.01):
            return tee, box


def rollout_states(checkpoint: str, episodes: int, record_steps, seed: int):
    """Arm and object states (qpos) from pushing episodes of a policy in the lightbox: every episode's first
    state, and every world's state at the given control steps. Returns (scene config, first, {step: states})."""
    import torch
    from ..evaluation import policy_agent
    from ..tasks import make_env
    from ..train import load_policy
    torch.set_num_threads(4)
    policy, cfg, _ = load_policy(checkpoint, "cpu")
    cfg.num_envs, cfg.num_threads, cfg.seed = episodes, 8, seed
    cfg.scene.lightbox = True
    cfg.task.difficulty, cfg.task.easy_fraction = 1.0, 0.0
    env = make_env(cfg)
    act = policy_agent(policy, "cpu")(env)
    nq = env.idx.nq
    first = [env.state[i, 1:1 + nq].copy() for i in range(env.n)]
    later, record = {}, set(record_steps)
    obs = env.obs
    for step in range(max(record) + 1):
        obs, _, _, _ = env.step(act(obs))
        if step in record:
            later[step] = [env.state[i, 1:1 + nq].copy() for i in range(env.n)]
    return env.cfg.scene, first, later


def _errors(name, est, truth):
    shape = OBJECTS[name]
    period = 2 * math.pi / shape.symmetry if shape.symmetry > 0 else 2 * math.pi
    d = (est[2] - truth[2] + period / 2) % period - period / 2
    return float(np.linalg.norm(est[:2] - truth[:2]) * 1e3), float(abs(math.degrees(d)))


def measure(sim: SimCamera, est: PoseEstimator, rng, kind: str, keep: list | None = None, arm=None,
            q_arm=None, joint_error_deg: float = 0.5) -> list[dict]:
    """Render the current state, estimate, and score every object against the truth. ``arm``: an ArmMask, given
    the arm's joint angles ``q_arm`` (default: the true ones now) plus up to +-joint_error_deg per joint."""
    truth, share, on_mat = sim.truth(), sim.visible_share(), sim.on_mat()
    img = sim.render(rng)
    hidden = None
    if arm is not None:
        q = sim.data.qpos[sim.idx.arm_qpos] if q_arm is None else q_arm
        hidden = arm(q + np.radians(rng.uniform(-joint_error_deg, joint_error_deg, len(q))))
    t0 = time.perf_counter()
    found = est.estimate(img, hidden=hidden)
    ms = (time.perf_counter() - t0) * 1e3
    rows = []
    for name, det in found.items():
        row = {"kind": kind, "object": name, "truth": truth[name].tolist(), "true_visible": share[name], "on_mat": on_mat[name],
               "status": det.status, "visible": det.visible, "coverage": det.coverage, "ms": ms,
               "pose": None if det.pose is None else det.pose.tolist(), "fit": None if det.fit is None else det.fit.tolist()}
        if det.fit is not None:
            row["fit_mm"], row["fit_deg"] = _errors(name, det.fit, truth[name])
        rows.append(row)
    if keep is not None:
        keep.append((img, found, truth, rows))
    return rows


VIS_BINS = [(0.95, 1.01), (0.8, 0.95), (0.6, 0.8), (0.4, 0.6), (0.25, 0.4), (0.0, 0.25)]


def summarise(rows: list[dict]) -> dict:
    """Per object: share dropped and why, error percentiles of accepted estimates, and both by true visible share.
    Frames where the object is off the mat (pushed past its edge onto the floor board: a failed trial, and the
    estimator assumes the mat) are only counted."""
    out = {}
    off = {n: sum(1 for r in rows if r["object"] == n and not r.get("on_mat", True)) for n in {r["object"] for r in rows}}
    rows = [r for r in rows if r.get("on_mat", True)]
    pct = lambda a: {p: float(np.percentile(a, p)) for p in (50, 90, 95, 99, 100)} if len(a) else {}
    for name in sorted({r["object"] for r in rows}):
        rs = [r for r in rows if r["object"] == name]
        acc = [r for r in rs if r["pose"] is not None]
        mm, deg = np.array([r["fit_mm"] for r in acc]), np.array([r["fit_deg"] for r in acc])
        s = {"frames": len(rs), "off_mat_frames": off[name], "accepted": len(acc), "dropped": 1 - len(acc) / max(len(rs), 1),
             "drop_reasons": {k: sum(r["status"] == k for r in rs) for k in sorted({r["status"] for r in rs}) if k != "ok"},
             "pos_mm": pct(mm), "yaw_deg": pct(deg),
             "accepted_over_2mm_or_2deg": int(((mm > 2) | (deg > 2)).sum()),
             "accepted_wrong_over_10mm_or_20deg": int(((mm > 10) | (deg > 20)).sum()), "by_true_visible": []}
        for lo, hi in VIS_BINS:
            b = [r for r in rs if lo <= r["true_visible"] < hi]
            ba, bf = [r for r in b if r["pose"] is not None], [r for r in b if "fit_mm" in r]
            p95 = lambda rr, k: float(np.percentile([r[k] for r in rr], 95)) if rr else None
            s["by_true_visible"].append({
                "bin": [lo, min(hi, 1.0)], "frames": len(b), "dropped": 1 - len(ba) / len(b) if b else None,
                "accepted_pos_mm_p95": p95(ba, "fit_mm"), "accepted_yaw_deg_p95": p95(ba, "fit_deg"),
                "all_fits_pos_mm_p95": p95(bf, "fit_mm"), "all_fits_yaw_deg_p95": p95(bf, "fit_deg")})
        out[name] = s
    ms = np.array([r["ms"] for r in rows[::max(1, len({r['object'] for r in rows}))]] or [0.0])
    out["ms_per_frame"] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)), "max": float(ms.max())}
    return out


def drop_bursts(rows: list[dict], name: str) -> dict:
    """Runs of consecutive dropped frames for one object in a sequence (rows in frame order)."""
    runs, n = [], 0
    for r in (r for r in rows if r["object"] == name):
        if r["pose"] is None:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return {"bursts": len(runs), "longest_frames": max(runs, default=0), "dropped_frames": int(sum(runs))}


def print_summary(title: str, s: dict):
    print(f"\n{title}")
    for name, v in s.items():
        if name == "ms_per_frame":
            continue
        p, y = v["pos_mm"], v["yaw_deg"]
        print(f"  {name}: {v['accepted']}/{v['frames']} accepted ({v['dropped']:.1%} dropped: {v['drop_reasons']})"
              + (f"; {v['off_mat_frames']} more frames with it off the mat, not counted" if v["off_mat_frames"] else ""))
        if p:
            print(f"    position error mm  p50 {p[50]:.2f}  p90 {p[90]:.2f}  p95 {p[95]:.2f}  p99 {p[99]:.2f}  max {p[100]:.2f}")
            print(f"    yaw error deg      p50 {y[50]:.2f}  p90 {y[90]:.2f}  p95 {y[95]:.2f}  p99 {y[99]:.2f}  max {y[100]:.2f}")
        print(f"    accepted but over 2 mm or 2 deg: {v['accepted_over_2mm_or_2deg']}; wrong (over 10 mm or 20 deg): "
              f"{v['accepted_wrong_over_10mm_or_20deg']}")
        print("    true visible share   frames  dropped   accepted p95 mm / deg   every fit p95 mm / deg")
        for b in v["by_true_visible"]:
            if not b["frames"]:
                continue
            acc = (f"{b['accepted_pos_mm_p95']:6.2f} / {b['accepted_yaw_deg_p95']:5.2f}"
                   if b["accepted_pos_mm_p95"] is not None else "      -       ")
            fit = (f"{b['all_fits_pos_mm_p95']:6.2f} / {b['all_fits_yaw_deg_p95']:5.2f}"
                   if b["all_fits_pos_mm_p95"] is not None else "      -")
            print(f"    {b['bin'][0]:.2f}-{b['bin'][1]:.2f}          {b['frames']:6d}  {b['dropped']:7.1%}   {acc}        {fit}")
    m = s["ms_per_frame"]
    print(f"  processing per frame: median {m['median']:.0f} ms, p95 {m['p95']:.0f} ms, max {m['max']:.0f} ms")


def _badness(kept_frame) -> float:
    """Largest accepted error (in units of 2 mm / 2 deg), or else how much of an object is hidden."""
    rows = kept_frame[3]
    return max([r.get("fit_mm", 0) / 2 + r.get("fit_deg", 0) / 2 if r["pose"] is not None else 0 for r in rows]
               + [1 - r["true_visible"] for r in rows])


def contact_sheet(kept, path: Path, camera: CameraModel, n: int = 6):
    """The worst frames (largest accepted error, then the most hidden), fitted outlines in green (accepted) or
    red (dropped) and the truth in white, cropped around the objects."""
    tiles = []
    for img, found, truth, rows in sorted(kept, key=_badness, reverse=True)[:n]:
        over = draw(img, camera, found, truth)
        uv = camera.project(np.array([[p[0], p[1], 0.01] for p in truth.values()]))
        c = np.clip(uv.mean(0), (480, 270), (1920 - 480, 1080 - 270)).astype(int)
        tile = over[c[1] - 270:c[1] + 270, c[0] - 480:c[0] + 480].copy()
        label = "   ".join(f"{r['object']} {r['status']} visible {r['true_visible']:.2f}"
                           + (f" {r['fit_mm']:.1f} mm {r['fit_deg']:.1f} deg" if "fit_mm" in r else "") for r in rows)
        cv2.rectangle(tile, (0, 0), (960, 30), (25, 25, 25), -1)
        cv2.putText(tile, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)], 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(PROJECT / "runs" / "clutter_v7" / "best.pt"))
    p.add_argument("--mount", choices=sorted(MOUNTS), default="left", help="where the C920 is (default: as built)")
    p.add_argument("--scenes", type=int, default=300, help="episodes: one band frame and --views pushing frames each")
    p.add_argument("--views", type=int, default=2, help="pushing frames per episode (at random control steps)")
    p.add_argument("--sequences", type=int, default=4, help="episodes rendered at every control step (20 Hz)")
    p.add_argument("--sequence-steps", type=int, default=200)
    p.add_argument("--calibrated", action="store_true", help="camera model from simulated ChArUco and ArUco calibration")
    p.add_argument("--no-degrade", action="store_true", help="perfect renders (no noise, JPEG, distortion ...)")
    p.add_argument("--no-arm-mask", action="store_true", help="estimate without the arm's mask (no joint angles)")
    p.add_argument("--joint-error-deg", type=float, default=0.5, help="arm mask: joint angle error, per joint and frame")
    p.add_argument("--arm-lag-steps", type=int, default=2, help="sequences: arm mask from joints this many frames old")
    p.add_argument("--min-visible", type=float, default=None, help="override EstimatorConfig.min_visible")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(PROJECT / "runs" / "camera_sim" / "eval.json"))
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    deg = Degradation.none() if args.no_degrade else Degradation()
    cfg = EstimatorConfig() if args.min_visible is None else EstimatorConfig(min_visible=args.min_visible)

    steps = sorted(set(rng.integers(10, 400, args.views).tolist()))
    t0 = time.time()
    scene, first, later = rollout_states(args.checkpoint, args.scenes, steps, args.seed)
    scene.c920_pos, scene.c920_target = MOUNTS[args.mount]
    print(f"{args.scenes} pushing episodes of {args.checkpoint} in the lightbox ({time.time() - t0:.0f} s)")
    sim = SimCamera(scene, degradation=deg)
    if sim.model.nq != len(first[0]):
        raise RuntimeError("the rendering model's state layout differs from the policy environment's")
    camera, calib = calibrated_camera(scene, rng, deg) if args.calibrated else (sim.camera(), None)
    if calib:
        print(f"simulated calibration: {calib['views']} board views, {calib['rms_px']:.2f} px, vertical FOV "
              f"{calib['fovy_deg']:.2f} deg (true {calib['true_fovy_deg']:.2f}); markers {calib['markers']}; camera "
              f"{calib['position_error_mm']:.2f} mm / {calib['rotation_error_deg']:.3f} deg from the truth")
    est = PoseEstimator(camera, DEFAULT_COLOURS, (sim.object_name, sim.scene.clutter), cfg)
    arm = None if args.no_arm_mask else ArmMask(camera, est.grid, scene, margin=0.006)
    tee_r, box_r = OBJECTS[sim.object_name].radius, OBJECTS[sim.scene.clutter].radius

    rows, kept, t0 = [], [], time.time()
    for i in range(args.scenes):
        sim.set_qpos(first[i])
        sim.place(*band_poses(rng, sim.data.site_xpos[sim.idx.tool_site], tee_r, box_r))
        est.last.clear()
        rows += measure(sim, est, rng, "band", kept, arm, joint_error_deg=args.joint_error_deg)
        for s in steps:
            sim.set_qpos(later[s][i])
            est.last.clear()
            rows += measure(sim, est, rng, "pushing", kept, arm, joint_error_deg=args.joint_error_deg)
        if len(kept) > 30:                     # frames are 6 MB: keep the worst for the contact sheet
            kept = sorted(kept, key=_badness, reverse=True)[:12]
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.scenes} scenes ({time.time() - t0:.0f} s)", flush=True)
    single = summarise(rows)
    print_summary(f"Independent frames, no tracking ({args.scenes} band, {args.scenes * len(steps)} pushing)", single)

    seq_rows, bursts = [], {}
    if args.sequences:
        _, _, seq = rollout_states(args.checkpoint, args.sequences, range(args.sequence_steps), args.seed + 1)
        for i in range(args.sequences):
            est.last.clear()
            ep = []
            for s in range(args.sequence_steps):
                sim.set_qpos(seq[s][i])
                q_old = seq[max(s - args.arm_lag_steps, 0)][i][sim.idx.arm_qpos]
                ep += measure(sim, est, rng, "sequence", None, arm, q_old, args.joint_error_deg)
            seq_rows += ep
            for name in est.templates:
                bursts.setdefault(name, []).append(drop_bursts(ep, name))
        print_summary(f"Sequences: {args.sequences} episodes x {args.sequence_steps} frames at 20 Hz, tracking on",
                      summarise(seq_rows))
        for name, bs in bursts.items():
            print(f"  {name}: dropped frames per episode {[b['dropped_frames'] for b in bs]}, in "
                  f"{[b['bursts'] for b in bs]} bursts, longest {[b['longest_frames'] for b in bs]} frames (50 ms each)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "mount": args.mount, "scenes": args.scenes, "pushing_steps": steps,
                               "calibrated": args.calibrated, "calibration": calib, "degradation": vars(deg),
                               "estimator": vars(cfg), "independent": single,
                               "sequences": summarise(seq_rows) if seq_rows else None, "bursts": bursts}, indent=1))
    sheet = out.with_name(out.stem + "_worst.png")
    contact_sheet(kept, sheet, camera)
    print(f"\nsaved {out} and {sheet}")


if __name__ == "__main__":
    main()
