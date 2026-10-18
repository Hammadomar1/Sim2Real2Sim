"""Planar poses of the pushing objects from one calibrated colour camera (the lightbox C920).

    est = PoseEstimator(CameraModel.load(), Colours.load())
    found = est.estimate(bgr_image, hidden=arm_mask(q))   # {"tee": Detection, "box": Detection}
    found["tee"].pose                                     # (x, y, yaw), robot frame (z = 0 the mat top), or None

No camera I/O here (``camera.py`` grabs frames, ``poses.CameraPoseSource`` ties the two together), so the
same code runs on real frames and on rendered ones (``camera_sim.py``). Per frame:

 1. Rectify. The image is resampled onto a top-down grid of the plane z = h, the objects' top faces (20 mm
    above the mat), at 0.5 mm per pixel: one cv2.remap through the calibrated intrinsics (lens distortion
    included) and extrinsics, with the maps computed once.
 2. Segment. Each object has its own colour (HSV ranges, ``Colours``). Along its edge a pixel counts by its
    brightness between the mat's and the object's (blur and MJPG smear the colour about a pixel outwards).
    Pixels behind the arm (``hidden``, from the joint angles: ``arm_mask.py``), in an occluder colour, in the
    other object's colour or outside the image carry no evidence; everything else (the black mat, shadows,
    walls) says "not this object".
 3. Fit. Seen from the camera and back-projected onto z = h, a prism's silhouette is its top face plus its
    visible sides: per convex part, the hull of the top face and of the bottom face pulled towards the
    camera's foot point by (Cz - h) / Cz. The fit maximises the integral of W over the silhouette, W = +1
    object, -1 not the object, 0 no evidence, over (x, y, yaw); the integral is exact (``region_integral``),
    so the objective has no pixel steps. Tracking starts from the previous estimate, with a weak bounded pull
    towards it (a partly hidden object does not slide along a direction its visible part leaves free);
    otherwise the fit starts from the blob's centroid and principal axis taken every way round. A coarse fit
    at 2 mm per pixel, then Nelder-Mead at 0.5 mm. The T turned round is fitted too: when it explains what is seen about as well
    (part of the T hidden), the previous estimate decides, or without one the frame is dropped.
 4. Check. ``visible`` is the share of the fitted silhouette seen in the object's colour. The frame is
    dropped for that object (pose None) when less than a quarter is visible with the arm masked (60 %
    without the mask, 45 % to start a track), when object-coloured pixels are left unexplained, or when much of
    the fitted silhouette lies on pixels that say "not this object" (part of it out of its colour range).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..objects import OBJECTS, ObjectShape
from ..scene import LIGHTBOX_MAT_HALF_WIDTH, LIGHTBOX_MAT_X

PROJECT = Path(__file__).resolve().parents[2]
CALIBRATION = PROJECT / "calibration"
INTRINSICS = CALIBRATION / "c920_intrinsics.json"
EXTRINSICS = CALIBRATION / "c920_extrinsics.json"
COLOURS = CALIBRATION / "c920_colours.json"
# Where objects are looked for: the mat plus 10 mm all round (robot frame, m).
MAT_REGION = ((LIGHTBOX_MAT_X[0] - 0.01, LIGHTBOX_MAT_X[1] + 0.01),
              (-LIGHTBOX_MAT_HALF_WIDTH - 0.01, LIGHTBOX_MAT_HALF_WIDTH + 0.01))


# ---------------------------------------------------------------------------------------------- camera
@dataclass
class CameraModel:
    """Pinhole camera with lens distortion, OpenCV conventions (x right, y down, z forward, pixel centres on
    integers). ``R``, ``t`` take robot-frame points into the camera frame: p_cam = R p + t."""
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int = 1920
    height: int = 1080

    def __post_init__(self):
        self.K = np.asarray(self.K, float).reshape(3, 3)
        self.dist = np.asarray(self.dist, float).ravel()
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.t = np.asarray(self.t, float).ravel()

    @property
    def position(self) -> np.ndarray:
        """Camera centre in the robot frame."""
        return -self.R.T @ self.t

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2 / self.K[1, 1]))

    def project(self, points) -> np.ndarray:
        """Robot-frame points (N, 3) -> pixels (N, 2), lens distortion included."""
        pts = np.asarray(points, np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(self.R)[0], self.t, self.K, self.dist)
        return uv.reshape(-1, 2)

    def backproject(self, uv, z: float = 0.0) -> np.ndarray:
        """Pixels (N, 2) -> robot-frame points (N, 3) on the horizontal plane at height z."""
        rays = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        d = np.c_[rays, np.ones(len(rays))] @ self.R           # ray directions in the robot frame (R^T d)
        c = self.position
        return c + ((z - c[2]) / d[:, 2])[:, None] * d

    @classmethod
    def from_mujoco(cls, model, data, camera: str, width: int, height: int) -> "CameraModel":
        """The exact camera of a MuJoCo scene (square pixels, no distortion): ground truth for rendered images."""
        cam = model.camera(camera).id
        f = height / 2 / math.tan(math.radians(model.cam_fovy[cam]) / 2)
        K = [[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1]]
        R = (data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])).T   # MuJoCo cameras look along -z, y up
        return cls(K, np.zeros(5), R, -R @ data.cam_xpos[cam], width, height)

    @classmethod
    def load(cls, intrinsics=INTRINSICS, extrinsics=EXTRINSICS) -> "CameraModel":
        i, e = json.loads(Path(intrinsics).read_text()), json.loads(Path(extrinsics).read_text())
        if (i["width"], i["height"]) != tuple(e.get("image_size", (i["width"], i["height"]))):
            raise ValueError("intrinsics and extrinsics were measured at different resolutions")
        return cls(i["K"], i["dist"], e["R"], e["t"], i["width"], i["height"])


# ---------------------------------------------------------------------------------------------- colours
@dataclass
class ColourRange:
    """An HSV box in OpenCV units (hue 0-180, saturation and value 0-255). hue[0] > hue[1] wraps through red."""
    hue: tuple[int, int]
    sat: tuple[int, int] = (90, 255)
    val: tuple[int, int] = (30, 255)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        (h0, h1), (s0, s1), (v0, v1) = self.hue, self.sat, self.val
        if h0 <= h1:
            return cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
        return cv2.inRange(hsv, (h0, s0, v0), (180, s1, v1)) | cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))


@dataclass
class Colours:
    """Each object's colour, and the colours of things that may hide them (the arm)."""
    objects: dict[str, ColourRange]
    occluders: list[ColourRange] = field(default_factory=list)

    def to_dict(self):
        return {"objects": {k: asdict(v) for k, v in self.objects.items()}, "occluders": [asdict(c) for c in self.occluders]}

    @classmethod
    def from_dict(cls, d):
        rng = lambda c: ColourRange(tuple(c["hue"]), tuple(c["sat"]), tuple(c["val"]))
        return cls({k: rng(v) for k, v in d["objects"].items()}, [rng(c) for c in d.get("occluders", [])])

    def save(self, path=COLOURS):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path=COLOURS) -> "Colours":
        return cls.from_dict(json.loads(Path(path).read_text()))


# The simulator's colours (camera_sim.SIM_RGBA): a blue T, a magenta box and the yellow arm. They are also the
# suggested colours for the printed parts: far from each other, from the arm, the black mat and the white walls.
DEFAULT_COLOURS = Colours({"tee": ColourRange((85, 130)), "box": ColourRange((140, 10))},
                          [ColourRange((15, 40), (80, 255), (60, 255))])


# ---------------------------------------------------------------------------------------------- shapes
@dataclass
class Template:
    """An object as the camera fits it: convex parts of its footprint (object frame, m) and its height."""
    name: str
    parts: list
    height: float
    symmetry: int
    area: float
    radius: float
    axis: float          # major principal axis of the footprint (rad, object frame); nan when isotropic

    @classmethod
    def from_shape(cls, shape: ObjectShape) -> "Template":
        if shape.disk_radius > 0:
            a = np.linspace(0, 2 * math.pi, 32, endpoint=False)
            parts = [shape.disk_radius * np.c_[np.cos(a), np.sin(a)]]
            return cls(shape.name, parts, shape.height, shape.symmetry, math.pi * shape.disk_radius ** 2,
                       shape.disk_radius, math.nan)
        parts, cov, area = [], np.zeros((2, 2)), 0.0
        for cx, cy, hx, hy in shape.centred_boxes:
            parts.append(np.array([(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]))
            a = 4 * hx * hy
            cov += a * (np.diag([hx ** 2 / 3, hy ** 2 / 3]) + np.outer((cx, cy), (cx, cy)))   # centroid at 0
            area += a
        w, v = np.linalg.eigh(cov / area)
        axis = math.nan if w[1] - w[0] < 0.05 * w[1] else math.atan2(v[1, 1], v[0, 1])
        return cls(shape.name, parts, shape.height, shape.symmetry, area, shape.radius, axis)

    def silhouette(self, pose, foot, cz: float, z: float) -> list[np.ndarray]:
        """Polygons (plane coordinates, m) the object covers when seen from a camera at height ``cz`` above the
        point ``foot`` and back-projected onto the plane at height ``z``: per part, the hull of its top and
        bottom faces, each pulled towards the foot point."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rot = np.array([[c, -s], [s, c]])
        k_top, k_bottom = (cz - z) / (cz - self.height), (cz - z) / cz
        out = []
        for part in self.parts:
            p = part @ rot.T + pose[:2] - foot
            pts = np.vstack([foot + k_top * p, foot + k_bottom * p]).astype(np.float32)
            out.append(cv2.convexHull(pts).reshape(-1, 2).astype(np.float64))
        return out

    def wrap_yaw(self, yaw: float) -> float:
        """Yaw in the object's symmetry period: (-pi, pi] for the T, [-45, 45) deg for the box."""
        period = 2 * math.pi / self.symmetry if self.symmetry > 0 else 2 * math.pi
        return (yaw + period / 2) % period - period / 2


def _centroid(polys) -> np.ndarray:
    """Area-weighted centroid of convex polygons (overlaps counted twice: good enough for a start)."""
    total, acc = 0.0, np.zeros(2)
    for p in polys:
        q = np.roll(p, -1, 0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        a = cross.sum() / 2
        if abs(a) > 1e-12:
            acc += ((p + q) * cross[:, None]).sum(0) / 6
            total += a
    return acc / total


def _nelder_mead(f, x0, step, tol: float, max_evals: int):
    """Minimise f from x0 with an initial simplex of the given steps; stop when the simplex is smaller than tol."""
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if max(np.abs(p - pts[0]).max() for p in pts[1:]) < tol:
            break
        c = np.mean(pts[:-1], 0)
        xr = 2 * c - pts[-1]
        fr = f(xr)
        evals += 1
        if fr < vals[0]:
            xe = 3 * c - 2 * pts[-1]
            fe = f(xe)
            evals += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = (c + pts[-1]) / 2 if fr >= vals[-1] else (c + xr) / 2
            fc = f(xc)
            evals += 1
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [(p + pts[0]) / 2 for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
                evals += n
    i = int(np.argmin(vals))
    return pts[i], vals[i], evals


# ---------------------------------------------------------------------------------------------- the plane
class PlaneGrid:
    """Top-down pixel grid on the plane at height z. Column j <-> y = y_max - (j + 0.5) res (image right is the
    robot's right), row i <-> x = x_max - (i + 0.5) res (image up is away from the robot)."""

    def __init__(self, region=MAT_REGION, res: float = 0.0005, z: float = 0.02):
        (x0, x1), (y0, y1) = region
        self.x_max, self.y_max, self.res, self.z = x1, y1, res, z
        self.rows, self.cols = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))

    def to_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([(self.y_max - xy[..., 1]) / self.res - 0.5, (self.x_max - xy[..., 0]) / self.res - 0.5], -1)

    def to_xy(self, px) -> np.ndarray:
        px = np.asarray(px, float)
        return np.stack([self.x_max - (px[..., 1] + 0.5) * self.res, self.y_max - (px[..., 0] + 0.5) * self.res], -1)

    def points(self) -> np.ndarray:
        """(rows, cols, 3) robot-frame centres of every pixel."""
        cols, rows = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        xy = self.to_xy(np.stack([cols, rows], -1))
        return np.concatenate([xy, np.full(xy.shape[:2] + (1,), self.z)], -1)


class Rectifier:
    """Camera image -> top-down view of a PlaneGrid (one cv2.remap; lens distortion included)."""

    def __init__(self, camera: CameraModel, grid: PlaneGrid):
        self.camera, self.grid = camera, grid
        pts = grid.points().reshape(-1, 3)
        uv = camera.project(pts).reshape(grid.rows, grid.cols, 2).astype(np.float32)
        depth = (pts @ camera.R.T + camera.t)[:, 2].reshape(grid.rows, grid.cols)
        self.valid = ((uv[..., 0] >= 0) & (uv[..., 0] <= camera.width - 1) & (uv[..., 1] >= 0)
                      & (uv[..., 1] <= camera.height - 1) & (depth > 0))
        self.maps = cv2.convertMaps(uv[..., 0], uv[..., 1], cv2.CV_16SC2)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.camera.height, self.camera.width):
            raise ValueError(f"frame is {image.shape[1]}x{image.shape[0]}, "
                             f"calibration is {self.camera.width}x{self.camera.height}")
        return cv2.remap(image, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _membership(core: np.ndarray, hsv: np.ndarray, colour: ColourRange, ignore: np.ndarray) -> np.ndarray:
    """How much of each pixel is the object (0-1). Blur and MJPG compression mix the object's colour into the
    background around it, and against the dark mat with hue and saturation intact, so the colour test alone makes
    the object about a pixel too big, and more so where it is bright (the top) than where it is dark (the sides):
    a bias in the fit. Along the edge, a pixel's share is its brightness between the background's and the
    object's nearby (the background is the mat, or the white floor board past the mat's edge)."""
    V = hsv[..., 2].astype(np.float32)
    colour_ok = ColourRange(colour.hue, colour.sat, (8, 255)).mask(hsv) > 0
    core8 = core.astype(np.uint8)
    near = cv2.dilate(core8, np.ones((5, 5), np.uint8)) > 0
    inner = cv2.erode(core8, np.ones((3, 3), np.uint8)) > 0
    background = (~near & ~ignore).astype(np.float32)
    # Local background brightness: the mean over background pixels within 15 mm (normalised box filter).
    total = cv2.boxFilter(V * background, -1, (31, 31), normalize=False)
    count = cv2.boxFilter(background, -1, (31, 31), normalize=False)
    v_bg = np.where(count > 20, total / np.maximum(count, 1.0), 20.0)
    v_obj = cv2.dilate(np.where(core, V, 0.0).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    span = v_obj - v_bg
    share = np.where(np.abs(span) < 15.0, 1.0, np.clip((V - v_bg) / np.where(np.abs(span) < 15.0, 1.0, span), 0.0, 1.0))
    return np.where(inner, 1.0, np.where(near & colour_ok, share, 0.0)).astype(np.float32)


def region_integral(polys, signs, C: np.ndarray, W: np.ndarray) -> float:
    """Sum over polygons of sign * (integral of W over the polygon), exact for a pixel-constant W (pixel (r, c)
    covers [c - 0.5, c + 0.5) x [r - 0.5, r + 0.5); zero outside the array). ``C`` is W's row-wise prefix sum
    with a leading zero column.

    Green's theorem: the area integral is the boundary integral of F dy, F(x, y) the integral of W along the
    row up to x. Cut at every row and column crossing, F is linear on each piece of an edge, so the midpoint
    rule is exact there. Unlike drawing the polygon, this is continuous in the vertices: the fit's objective
    has no pixel steps."""
    starts = np.concatenate(polys)
    ends, orient, off = np.empty_like(starts), np.empty(len(starts)), 0
    for p, sg in zip(polys, signs):
        q = ends[off:off + len(p)]
        q[:-1], q[-1] = p[1:], p[0]
        # positive orientation (shoelace area > 0) makes the boundary integral the area integral
        orient[off:off + len(p)] = sg if float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum()) > 0 else -sg
        off += len(p)
    n = len(starts)
    v0 = np.concatenate([starts[:, 0], starts[:, 1]])        # x then y of every edge start
    dv = np.concatenate([ends[:, 0], ends[:, 1]]) - v0
    lo = np.minimum(v0, v0 + dv)
    ka = np.ceil(lo - 0.5)
    cnt = np.floor(np.maximum(v0, v0 + dv) - 0.5) - ka + 1
    cnt = np.where((cnt > 0) & (np.abs(dv) > 1e-12), cnt, 0).astype(np.int64)
    idx = np.repeat(np.arange(2 * n), cnt)                     # pixel-boundary crossings k + 0.5
    k = np.repeat(ka, cnt) + (np.arange(len(idx)) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    t = (k + 0.5 - v0[idx]) / dv[idx]
    edge = idx % n
    key = np.concatenate([2.0 * np.arange(n), 2.0 * np.arange(n) + 1.0, 2.0 * edge + np.minimum(np.maximum(t, 0.0), 1.0)])
    key.sort()                                                 # by edge, then along it
    e = (key * 0.5).astype(np.int64)
    same = e[1:] == e[:-1]
    e, ta, tb = e[:-1][same], key[:-1][same], key[1:][same]
    tm = (ta + tb) * 0.5 - 2 * e
    dx, dy = dv[:n], dv[n:]
    xm, ym = v0[:n][e] + tm * dx[e], v0[n:][e] + tm * dy[e]
    h, w = W.shape
    r = np.floor(ym + 0.5).astype(np.int64)
    keep = (r >= 0) & (r < h)
    xc = np.minimum(np.maximum(xm[keep], -0.5), w - 0.5)
    j = np.minimum((xc + 0.5).astype(np.int64), w - 1)         # xc + 0.5 >= 0: truncation is floor
    r = r[keep]
    F = C[r, j] + (xc - j + 0.5) * W[r, j]
    e = e[keep]
    return float(np.dot(orient[e] * dy[e] * (tb[keep] - ta[keep]), F))


class _Window:
    """One object's silhouette against the evidence W around it, at one scale (every k-th grid pixel).

    The template's corners go straight from the object frame to window pixels: the plane -> grid -> window
    maps are folded into one affine map, since the score is evaluated a few hundred times per object and frame.
    """

    def __init__(self, W: np.ndarray, r0: int, c0: int, k: int, tpl: Template, grid: PlaneGrid, foot, cz: float,
                 exact: bool = True):
        self.exact, self.k = exact, k
        h, w = W.shape[0] // k * k, W.shape[1] // k * k
        self.W = W if k == 1 else cv2.resize(W[:h, :w], (w // k, h // k), interpolation=cv2.INTER_AREA)
        self.Wd = self.W.astype(np.float64)
        self.Wabs = np.abs(self.W)
        self.C = np.zeros((self.W.shape[0], self.W.shape[1] + 1))
        self.C[:, 1:] = np.cumsum(self.Wd, axis=1)
        self.canvas = np.zeros(self.W.shape, np.uint8)
        self.parts = np.array(tpl.parts)                                  # (parts, corners, 2)
        self.k_top, self.k_bottom = (cz - grid.z) / (cz - tpl.height), (cz - grid.z) / cz
        self.foot = np.asarray(foot, float)
        s = 1 / (grid.res * k)
        self.A = np.array([[0.0, -s], [-s, 0.0]])                        # (x, y) -> (col, row), transposed
        self.b = np.array([(grid.y_max / grid.res - 0.5 - c0 - (k - 1) / 2) / k,
                           (grid.x_max / grid.res - 0.5 - r0 - (k - 1) / 2) / k])

    def hulls(self, pose) -> list[np.ndarray]:
        """The silhouette's convex parts at ``pose``, in window pixels (float32)."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        p = self.parts @ np.array([[c, s], [-s, c]]) + (pose[0] - self.foot[0], pose[1] - self.foot[1])
        px = ((np.concatenate([self.k_top * p, self.k_bottom * p], 1) + self.foot) @ self.A + self.b).astype(np.float32)
        return [cv2.convexHull(q).reshape(-1, 2) for q in px]

    def score(self, pose) -> float:
        """Integral of W over the silhouette (window pixel units); overlapping parts counted once. Exact, or
        (``exact=False``, 3-4 times faster) from the drawn silhouette, whose pixel steps are fine for a coarse search."""
        if not self.exact:
            sub, at = self.raster(pose)
            return 0.0 if sub is None else float((sub * self.W[at]).sum()) / 255
        hulls = self.hulls(pose)
        polys, signs = [h.astype(np.float64) for h in hulls], [1.0] * len(hulls)
        for a in range(len(hulls)):
            for b in range(a + 1, len(hulls)):
                area, inter = cv2.intersectConvexConvex(hulls[a], hulls[b])
                if area > 0 and inter is not None and len(inter) >= 3:
                    polys.append(inter.reshape(-1, 2).astype(np.float64))
                    signs.append(-1.0)
        return region_integral(polys, signs, self.C, self.Wd)

    def quality(self, pose) -> tuple[float, float]:
        """(score / evidence, evidence): how well the silhouette at ``pose`` agrees with the pixels that say
        anything (1 = perfectly, whatever share is hidden), and how many such pixels it covers."""
        sub, at = self.raster(pose)
        if sub is None:
            return 0.0, 0.0
        evidence = float((sub * self.Wabs[at]).sum()) / 255
        return float((sub * self.W[at]).sum()) / 255 / max(evidence, 1e-9), evidence

    def raster(self, pose):
        """Anti-aliased silhouette at ``pose``: (coverage 0-255 over its bounding box, that window slice)."""
        hulls = [np.round(h * 16).astype(np.int32) for h in self.hulls(pose)]
        pts = np.vstack(hulls)
        h, w = self.W.shape
        x0, y0 = max(int(pts[:, 0].min()) // 16 - 1, 0), max(int(pts[:, 1].min()) // 16 - 1, 0)
        x1, y1 = min(int(pts[:, 0].max()) // 16 + 3, w), min(int(pts[:, 1].max()) // 16 + 3, h)
        if x1 <= x0 or y1 <= y0:
            return None, None
        sub = self.canvas[y0:y1, x0:x1]
        sub[:] = 0
        off = np.array([x0 * 16, y0 * 16], np.int32)
        for hull in hulls:       # one call per part: fillPoly fills several polygons by the even-odd rule
            cv2.fillPoly(sub, [hull - off], 255, cv2.LINE_AA, shift=4)
        return sub, (slice(y0, y1), slice(x0, x1))


# ---------------------------------------------------------------------------------------------- estimator
@dataclass
class EstimatorConfig:
    resolution: float = 0.0005      # rectified grid (m per pixel)
    coarse: int = 4                 # the coarse fit runs at resolution * coarse
    background_weight: float = 1.0  # cost of silhouette over not-the-object pixels (object pixels earn 1)
    # Drop the frame when less of the fitted silhouette is seen in the object's colour. With the arm's mask
    # (estimate(hidden=...)) the fit stays within a millimetre down to a quarter visible; without it, a dark servo
    # in front looks like the mat and fits go wrong below about 60 % (camera_sim). The policies lose far more
    # from missing frames than from small errors, so drop as little as the fit allows.
    min_visible: float = 0.25
    min_visible_unmasked: float = 0.60
    min_visible_acquire: float = 0.45   # ... and to start tracking (no recent estimate), at least this much
    min_coverage: float = 0.85      # drop when the fit explains less of the object-coloured pixels
    max_contradicted: float = 0.20  # drop when more of the fitted silhouette lies on "not this object" pixels
    min_blob: float = 0.20          # object colour needed to try at all, as a share of the top face
    flip_margin: float = 0.10       # T: turned round, it fits as well when within this share of the evidence ...
    track_seconds: float = 2.0      # start from the previous estimate when it is this recent ...
    track_score: float = 0.80       # ... and search afresh when, refitted, it agrees less with what is seen
    prior_depth: float = 16.0       # tracking: pull towards the previous estimate, at most this many evidence pixels ...
    prior_radius: float = 4.0       # ... fading out beyond this many fine pixels (2 mm; yaw: arc at the object's radius)


@dataclass
class Detection:
    pose: np.ndarray | None         # (x, y, yaw) in the robot frame; None: no reliable estimate in this frame
    status: str                     # ok | missing | occluded | unexplained | contradicted | ambiguous
    visible: float = 0.0            # share of the fitted silhouette seen in the object's colour
    coverage: float = 0.0           # share of the object-coloured pixels the fit explains
    contradicted: float = 0.0       # share of the fitted silhouette on pixels that say "not this object"
    fit: np.ndarray | None = None   # the best fit, also when the frame is dropped (diagnostics)


class PoseEstimator:
    def __init__(self, camera: CameraModel, colours: Colours = DEFAULT_COLOURS, objects=("tee", "box"),
                 config: EstimatorConfig | None = None, region=MAT_REGION):
        self.camera, self.colours, self.cfg = camera, colours, config or EstimatorConfig()
        self.templates = {name: Template.from_shape(OBJECTS[name]) for name in objects}
        missing = [n for n in objects if n not in colours.objects]
        if missing:
            raise ValueError(f"no colour for {missing}: run camera_calibrate colours")
        heights = {t.height for t in self.templates.values()}
        self.grid = PlaneGrid(region, self.cfg.resolution, max(heights))
        self.rectify = Rectifier(camera, self.grid)
        self.foot, self.cz = camera.position[:2], float(camera.position[2])
        self.last: dict[str, tuple[np.ndarray, float]] = {}
        self.timing = {}

    def estimate(self, image: np.ndarray, stamp: float | None = None, hidden: np.ndarray | None = None
                 ) -> dict[str, Detection]:
        """Every object's pose in one BGR frame (the camera's resolution). ``hidden``: pixels of the grid known to
        be behind the arm (``arm_mask.ArmMask``), which then carry no evidence."""
        stamp = time.monotonic() if stamp is None else stamp
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(self.rectify(image), cv2.COLOR_BGR2HSV)
        masks = {n: self.colours.objects[n].mask(hsv) for n in self.templates}
        unknown = ~self.rectify.valid
        if hidden is not None:
            unknown |= hidden
        for occ in self.colours.occluders:
            unknown |= occ.mask(hsv) > 0
        t1 = time.perf_counter()
        out = {}
        for name, tpl in self.templates.items():
            others = [m for n, m in masks.items() if n != name]
            prev = self.last.get(name)
            prev = prev[0] if prev is not None and stamp - prev[1] <= self.cfg.track_seconds else None
            out[name] = self._detect(tpl, masks[name], others, unknown, prev, hsv, self.colours.objects[name],
                                     masked=hidden is not None)
            if out[name].pose is not None:
                self.last[name] = (out[name].pose, stamp)
        self.timing = {"segment_ms": (t1 - t0) * 1e3, "fit_ms": (time.perf_counter() - t1) * 1e3}
        return out

    # -- one object
    def _detect(self, tpl: Template, mask, others, unknown, prev, hsv=None, colour: ColourRange | None = None,
                masked: bool = False) -> Detection:
        cfg, res, k = self.cfg, self.grid.res, self.cfg.coarse
        # Blobs of the object's colour, found on the coarse grid (a cell counts when a quarter of it has the colour).
        h, w = mask.shape[0] // k * k, mask.shape[1] // k * k
        small = (cv2.resize(mask[:h, :w], (w // k, h // k), interpolation=cv2.INTER_AREA) >= 64).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(small, connectivity=8)
        top = tpl.area / (res * k) ** 2                       # the top face, in coarse cells
        blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.02 * top]
        if not blobs:
            return Detection(None, "missing")
        main = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        reach = (2 * tpl.radius + 0.01) / (res * k)          # pieces of one object split by an occluder
        group = [i for i in blobs if np.linalg.norm(cents[i] - cents[main]) <= reach]
        if sum(stats[i, cv2.CC_STAT_AREA] for i in group) < cfg.min_blob * top:
            return Detection(None, "missing")
        x0 = k * min(stats[i, cv2.CC_STAT_LEFT] for i in group)
        y0 = k * min(stats[i, cv2.CC_STAT_TOP] for i in group)
        x1 = k * max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in group)
        y1 = k * max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in group)
        band = tpl.height * np.linalg.norm(self.grid.to_xy(cents[main] * k + (k - 1) / 2) - self.foot) / self.cz
        pad = int(math.ceil((2 * tpl.radius + band + 0.005) / res))
        r0, c0 = max(y0 - pad, 0), max(x0 - pad, 0)
        r1, c1 = min(y1 + pad, self.grid.rows), min(x1 + pad, self.grid.cols)
        sl = (slice(r0, r1), slice(c0, c1))
        # The object's pixels: its colour, within a coarse cell of the group's blobs.
        region = np.isin(labels[r0 // k:-(-r1 // k), c0 // k:-(-c1 // k)], group).astype(np.uint8)
        region = cv2.dilate(np.repeat(np.repeat(region, k, 0), k, 1), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        region = region[r0 % k:r0 % k + r1 - r0, c0 % k:c0 % k + c1 - c0]
        full = np.zeros((r1 - r0, c1 - c0), np.uint8)
        full[:region.shape[0], :region.shape[1]] = region
        obj = (mask[sl] > 0) & (full > 0)
        ignore = unknown[sl] | ((mask[sl] > 0) & ~obj)       # other blobs of this colour carry no evidence either
        for m in others:
            ignore |= m[sl] > 0
        member = obj.astype(np.float32) if hsv is None else _membership(obj, hsv[sl], colour, ignore)
        lam = cfg.background_weight
        W = np.where(ignore & (member == 0), 0.0, (1 + lam) * member - lam).astype(np.float32)

        coarse = _Window(W, r0, c0, k, tpl, self.grid, self.foot, self.cz, exact=False)
        fine = _Window(W, r0, c0, 1, tpl, self.grid, self.foot, self.cz)
        scale = np.array([res, res, res / tpl.radius])      # optimiser units: fine pixels (yaw: arc at the radius)

        def fit(win, start, step, tol, evals, prior=None):
            # Tracking: a weak, bounded pull towards the previous estimate. Where the visible part of an object leaves
            # a direction free (the rest behind the arm), the fit stays where the object was instead of sliding; the
            # pull is at most cfg.prior_depth evidence pixels, so it never holds an estimate against what is seen.
            u0 = None if prior is None else np.asarray(prior, float) / scale
            depth, r2 = cfg.prior_depth / win.k ** 2, cfg.prior_radius ** 2

            def cost(u):
                pull = 0.0 if u0 is None else depth * (1.0 - math.exp(-float(np.sum((u - u0) ** 2)) / r2))
                return -win.score(u * scale) + pull
            u, f, _ = _nelder_mead(cost, np.asarray(start, float) / scale, step, tol, evals)
            return u * scale, f

        def coarse_fit(start, prior=None):
            return fit(coarse, start, (2.0 * k,) * 3, 0.5 * k, 60, prior)

        rows, cols = np.nonzero(obj)
        xy = self.grid.to_xy(np.c_[cols + c0, rows + r0])
        fits, tracking = [], False
        if prev is not None:                                  # tracking: refit from the previous estimate. At the
            prev = np.asarray(prev, float)                    # coarse scale first: started at full resolution, the
            fits.append(coarse_fit(prev, prev))               # fit can stay in a small local optimum frame after frame
            tracking = coarse.quality(fits[0][0])[0] >= cfg.track_score
        if not tracking:                                      # otherwise search from the blob
            fits += [coarse_fit(s) for s in self._starts(tpl, xy)]
        best = min(fits, key=lambda f: f[1])
        ambiguous = False
        if tpl.symmetry == 1 and not tracking:
            # The T turned round: the best coarse fit more than 90 degrees from the winner (or a fresh one). When it
            # agrees with the evidence about as well, the previous estimate decides; without one the frame is dropped.
            flipped = [f for f in fits if abs(tpl.wrap_yaw(f[0][2] - best[0][2])) > math.pi / 2]
            rival = min(flipped, key=lambda f: f[1]) if flipped else coarse_fit(best[0] + (0.0, 0.0, math.pi))
            if rival[1] - best[1] < cfg.flip_margin * coarse.quality(best[0])[1]:
                if prev is None:
                    ambiguous = True
                else:
                    best = min((best, rival), key=lambda f: abs(tpl.wrap_yaw(f[0][2] - prev[2])))
        pose, _ = fit(fine, best[0], (1.5, 1.5, 1.5), 0.05, 150, prev if tracking else None)
        pose[2] = tpl.wrap_yaw(pose[2])

        # Checks at the fitted pose.
        S, at = fine.raster(pose)
        if S is None:
            return Detection(None, "missing", fit=pose)
        S = S.astype(np.float32) / 255
        seen = float((S * member[at]).sum())
        visible, coverage = seen / max(float(S.sum()), 1e-9), seen / max(float(member.sum()), 1e-9)
        # The share of the fitted silhouette on pixels that say "not this object" (not merely hidden): small for a
        # right fit however much is hidden; large when part of the object is out of its colour range (lighting
        # drift) or a dark occluder is not masked.
        contradicted = float((S * np.maximum(-W[at], 0.0)).sum()) / lam / max(float(S.sum()), 1e-9)
        det = Detection(pose, "ok", visible, coverage, contradicted=contradicted, fit=pose.copy())
        if ambiguous:
            det.pose, det.status = None, "ambiguous"
        need = cfg.min_visible if masked else cfg.min_visible_unmasked
        if visible < (need if prev is not None else max(need, cfg.min_visible_acquire)):
            det.pose, det.status = None, "occluded"
        elif coverage < cfg.min_coverage:
            det.pose, det.status = None, "unexplained"
        elif contradicted > cfg.max_contradicted:
            det.pose, det.status = None, "contradicted"
        return det

    def _starts(self, tpl: Template, xy: np.ndarray) -> list[np.ndarray]:
        """Starting poses from the object-coloured pixels ``xy`` (plane, m): their principal axis taken every
        way the shape allows, and their centroid moved back by the side faces' share of the silhouette."""
        centre = xy.mean(0)
        if math.isnan(tpl.axis):
            yaws = [j * math.pi / 6 for j in range(3)] if tpl.symmetry == 4 else [0.0]
        else:
            _, v = np.linalg.eigh(np.cov(xy.T))
            phi = math.atan2(v[1, 1], v[0, 1]) - tpl.axis
            yaws = [phi + j * math.pi / 2 for j in range(4 if tpl.symmetry == 1 else 2)]
        starts = []
        for yaw in yaws:
            pose = np.array([*centre, yaw])
            shift = _centroid(tpl.silhouette(pose, self.foot, self.cz, self.grid.z)) - centre
            starts.append(np.array([*(centre - shift), yaw]))
        return starts


def draw(image: np.ndarray, camera: CameraModel, detections: dict[str, Detection], truth: dict | None = None) -> np.ndarray:
    """Overlay the fitted prisms (green: accepted, red: dropped) and, when given, the true ones (white)."""
    out = image.copy()

    def prism(name, pose, colour):
        tpl = Template.from_shape(OBJECTS[name])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        for part in tpl.parts:
            xy = part @ np.array([[c, -s], [s, c]]).T + pose[:2]
            for z in (0.0, tpl.height):
                uv = camera.project(np.c_[xy, np.full(len(xy), z)])
                cv2.polylines(out, [np.round(uv * 4).astype(np.int32)], True, colour, 2, cv2.LINE_AA, shift=2)

    for name, pose in (truth or {}).items():
        prism(name, np.asarray(pose), (255, 255, 255))
    for name, det in detections.items():
        if det.fit is not None:
            prism(name, det.fit, (60, 220, 60) if det.pose is not None else (60, 60, 230))
    return out
