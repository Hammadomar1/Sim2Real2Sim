"""Calibrate the lightbox C920 for the pose estimator, then check it live (LIGHTBOX.md phase C).

    python -m s2r2s.hardware.camera_calibrate boards       # print these: ChArUco board, mat markers (calibration/print/)
    python -m s2r2s.hardware.camera_calibrate settings     # fix focus, exposure, white balance -> c920_settings.json
    python -m s2r2s.hardware.camera_calibrate intrinsics   # wave the ChArUco board in view -> c920_intrinsics.json
    python -m s2r2s.hardware.camera_calibrate extrinsics   # markers at known places on the mat -> c920_extrinsics.json
    python -m s2r2s.hardware.camera_calibrate colours      # the T and the box anywhere on the mat -> c920_colours.json
    python -m s2r2s.hardware.camera_calibrate check        # live poses, dropped frames and timing

All files go to calibration/. ``c920_markers.json`` says where the mat markers lie (written with the default
layout the first time; edit it if you place them elsewhere). The steps that look at the camera also accept
saved images (``--images``); ``camera_sim`` runs the same functions on rendered frames.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..scene import LIGHTBOX_MAT_HALF_WIDTH, LIGHTBOX_MAT_X, SceneConfig
from .vision import (CALIBRATION, COLOURS, DEFAULT_COLOURS, EXTRINSICS, INTRINSICS, MAT_REGION, CameraModel, ColourRange,
                     Colours, PlaneGrid, PoseEstimator, Rectifier, draw)

MARKERS = CALIBRATION / "c920_markers.json"
MAT_DICT = cv2.aruco.DICT_4X4_50        # markers on the mat: big cells, robust at a slant
BOARD_DICT = cv2.aruco.DICT_5X5_100     # ChArUco board (a different dictionary: never mistaken for a mat marker)
MARKER_SIZE, MARKER_MARGIN = 0.050, 0.010
NOMINAL_FOVY = 43.3                     # C920 at 16:9 (LIGHTBOX.md)


# ---------------------------------------------------------------------------------------------- targets
def board(squares=(7, 5), square: float = 0.035, marker: float = 0.026) -> cv2.aruco.CharucoBoard:
    """The intrinsics board: 7 x 5 squares of 35 mm (fits A4 landscape). Its printed scale does not matter for
    the intrinsics; it must be flat (glue it to card or foam board)."""
    return cv2.aruco.CharucoBoard(squares, square, marker, cv2.aruco.getPredefinedDictionary(BOARD_DICT))


def marker_image(mid: int, size: float = MARKER_SIZE, margin: float = MARKER_MARGIN, px_per_mm: float = 20) -> np.ndarray:
    """A mat marker with its white border (grey levels). The border's outer edge is what lies against the mat edge;
    the small grey triangle in it marks the marker's top, which points away from the robot."""
    side, pad = round(size * 1e3 * px_per_mm), round(margin * 1e3 * px_per_mm)
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(MAT_DICT), mid, side, borderBits=1)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    c, w = img.shape[1] / 2, 2.5 * px_per_mm
    cv2.fillPoly(img, [np.round([(c - w, 2 * w), (c + w, 2 * w), (c, w)]).astype(np.int32)], 170)
    return img


@dataclass
class Marker:
    """A printed ArUco marker lying on the mat. Robot frame, metres; ``yaw`` is the direction of the printed
    marker's top edge (0: away from the robot, +x)."""
    id: int
    centre: tuple[float, float]
    yaw: float = 0.0
    size: float = MARKER_SIZE
    margin: float = MARKER_MARGIN
    z: float = 0.0001                   # paper on the mat
    where: str = ""

    def corners(self) -> np.ndarray:
        """(4, 3) corners of the black square in ArUco order: top-left, top-right, bottom-right, bottom-left."""
        up = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        right = np.array([math.sin(self.yaw), -math.cos(self.yaw)])
        h, c = self.size / 2, np.asarray(self.centre, float)
        xy = [c - h * right + h * up, c + h * right + h * up, c + h * right - h * up, c - h * right - h * up]
        return np.c_[np.array(xy), np.full(4, self.z)]


def default_layout() -> list[Marker]:
    """Eight markers pushed into the mat's corners and against its edges (their white border flush with the
    edges). Print and place all eight: the extrinsics use whichever the camera sees (at least three, better
    four or more). Seen from the left-wall slot as built: 0-5 (the left edge runs along the bottom of the image);
    from the back wall: 3-7; from the right wall: all but 3 and 4. The side markers sit 8 cm along the edge from
    the near corners' markers."""
    c = MARKER_SIZE / 2 + MARKER_MARGIN
    (xn, xf), yl = LIGHTBOX_MAT_X, LIGHTBOX_MAT_HALF_WIDTH
    xm = xn + c + 0.08
    spots = [((xf - c, yl - c), "far-left corner"), ((xf - c, 0.0), "far edge, centre"),
             ((xf - c, -yl + c), "far-right corner"), ((xm, -yl + c), "right edge, 2nd from robot"),
             ((xn + c, -yl + c), "near-right corner"), ((xn + c, 0.0), "near edge, centre"),
             ((xn + c, yl - c), "near-left corner"), ((xm, yl - c), "left edge, 2nd from robot")]
    return [Marker(i, (round(x, 5), round(y, 5)), where=w) for i, ((x, y), w) in enumerate(spots)]


def load_layout(path=MARKERS) -> list[Marker]:
    path = Path(path)
    if not path.exists():
        save_layout(default_layout(), path)
        print(f"wrote the default marker layout to {path}")
    return [Marker(**{**m, "centre": tuple(m["centre"])}) for m in json.loads(path.read_text())["markers"]]


def save_layout(layout: list[Marker], path=MARKERS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"note": "robot frame (m): x forward, y left, origin 38.8 mm behind the pan axis; "
                                              "yaw = direction of the printed marker's top edge",
                                      "markers": [asdict(m) for m in layout]}, indent=1))


# ---------------------------------------------------------------------------------------------- detection and fits
def _gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_markers(image) -> dict[int, np.ndarray]:
    """Mat markers in an image: {id: (4, 2) corners, sub-pixel}."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MAT_DICT), params)
    corners, ids, _ = det.detectMarkers(_gray(image))
    return {} if ids is None else {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.ravel())}


def merge_detections(frames: list[dict]) -> dict[int, np.ndarray]:
    """Median corners of markers seen in at least half the frames (less noise than one frame)."""
    ids = {i for f in frames for i in f}
    return {i: np.median([f[i] for f in frames if i in f], 0) for i in sorted(ids)
            if sum(i in f for f in frames) >= max(1, len(frames) // 2)}


def detect_board(image, brd: cv2.aruco.CharucoBoard, min_corners: int = 8):
    """ChArUco corners: (object points (N, 3), image points (N, 2)) or None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(brd).detectBoard(_gray(image))
    if ids is None or len(ids) < min_corners:
        return None
    obj, img = brd.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def calibrate_intrinsics(views, image_size, fix_k3: bool = True) -> dict:
    """Camera matrix and distortion from ChArUco views [(object points, image points), ...]."""
    if len(views) < 5:
        raise RuntimeError(f"only {len(views)} usable board views; need at least 5 (better 20)")
    w, h = image_size
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([v[0] for v in views], [v[1] for v in views], (w, h), None, None,
                                                     flags=flags)
    per_view = []
    for (obj, img), r, t in zip(views, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1)))))
    return {"width": w, "height": h, "K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms),
            "per_view_rms_px": per_view, "views": len(views),
            "fovy_deg": math.degrees(2 * math.atan(h / 2 / K[1, 1])), "fovx_deg": math.degrees(2 * math.atan(w / 2 / K[0, 0])),
            "date": time.strftime("%Y-%m-%d %H:%M")}


def solve_extrinsics(detected: dict[int, np.ndarray], layout: list[Marker], K, dist, image_size) -> dict:
    """Camera pose in the robot frame from mat markers (solvePnP on their corners), with its errors on the mat.

    ``mat_error_mm`` is each marker's corner residual back-projected onto the mat; ``held_out_mm`` is the same
    for each marker when the pose is solved without it (a fairer measure of what the camera gets wrong)."""
    use = [m for m in layout if m.id in detected]
    if len(use) < 3:
        raise RuntimeError(f"saw markers {sorted(detected)}; need at least 3 of the layout's {[m.id for m in layout]}")
    K, dist = np.asarray(K, float), np.asarray(dist, float)

    def solve(markers):
        obj = np.vstack([m.corners() for m in markers])
        img = np.vstack([detected[m.id] for m in markers])
        _, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        return CameraModel(K, dist, cv2.Rodrigues(rvec)[0], tvec.ravel(), *image_size)

    def mat_error(cam, m):
        on_mat = cam.backproject(detected[m.id], m.z)
        return float(np.sqrt(np.mean(np.sum((on_mat[:, :2] - m.corners()[:, :2]) ** 2, 1)))) * 1e3

    cam = solve(use)
    obj = np.vstack([m.corners() for m in use])
    img = np.vstack([detected[m.id] for m in use])
    reproj = np.linalg.norm(cam.project(obj) - img, axis=1)
    errors = {m.id: mat_error(cam, m) for m in use}
    held_out = {m.id: mat_error(solve([o for o in use if o is not m]), m) for m in use} if len(use) >= 4 else {}
    nominal = np.array(SceneConfig().c920_pos) - (0, 0, SceneConfig().mat_height)
    return {"R": cam.R.tolist(), "t": cam.t.tolist(), "image_size": list(image_size),
            "camera_position_m": cam.position.tolist(), "offset_from_nominal_mm": ((cam.position - nominal) * 1e3).tolist(),
            "markers": [m.id for m in use], "reprojection_px": {"rms": float(np.sqrt(np.mean(reproj ** 2))), "max": float(reproj.max())},
            "mat_error_mm": errors, "held_out_mm": held_out, "date": time.strftime("%Y-%m-%d %H:%M")}


def learn_colours(image, camera: CameraModel, names=("tee", "box"), occluders=None) -> tuple[Colours, dict]:
    """Each object's colour from one frame with the objects anywhere on the mat (and the arm off it).

    Saturated blobs on the mat are fitted with every object's shape; each object takes the blob its shape fits
    best, and its HSV ranges cover that blob's pixels (top and sides) with some margin."""
    occluders = list(DEFAULT_COLOURS.occluders if occluders is None else occluders)
    est = PoseEstimator(camera, Colours({n: ColourRange((0, 180)) for n in names}), names)
    hsv = cv2.cvtColor(est.rectify(image), cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] >= 60) & (hsv[..., 2] >= 40) & est.rectify.valid
    for occ in occluders:                                # the arm's colour is never an object's
        sat &= occ.mask(hsv) == 0
    sat = sat.astype(np.uint8)
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    px = est.grid.res ** 2
    areas = [t.area / px for t in est.templates.values()]
    blobs = [i for i in range(1, n) if 0.5 * min(areas) <= stats[i, cv2.CC_STAT_AREA] <= 4 * max(areas)]
    if len(blobs) < len(names):
        raise RuntimeError(f"found {len(blobs)} coloured objects on the mat, expected {len(names)}: are they in view "
                           f"and saturated colours? (blob areas {sorted(stats[1:, cv2.CC_STAT_AREA].tolist())[-6:]} px)")
    unknown = ~est.rectify.valid
    quality = {}
    for b in blobs:
        mask = (labels == b).astype(np.uint8) * 255
        for name, tpl in est.templates.items():
            d = est._detect(tpl, mask, [], unknown, None)
            quality[b, name] = d.visible * d.coverage if d.fit is not None else 0.0
    best, best_q = None, -1.0
    for perm in itertools.permutations(blobs, len(names)):
        q = sum(quality[b, name] for b, name in zip(perm, names))
        if q > best_q:
            best, best_q = perm, q
    objects, report = {}, {}
    for b, name in zip(best, names):
        pix = hsv[labels == b].astype(float)
        h = pix[:, 0] * (2 * math.pi / 180)
        mean = math.atan2(np.sin(h).mean(), np.cos(h).mean()) * 180 / (2 * math.pi) % 180
        d = (pix[:, 0] - mean + 90) % 180 - 90
        lo, hi = np.percentile(d, [0.5, 99.5]) + (-5, 5)
        objects[name] = ColourRange((int(round(mean + lo)) % 180, int(round(mean + hi)) % 180),
                                    (int(max(50, np.percentile(pix[:, 1], 1) - 25)), 255),
                                    (int(max(30, np.percentile(pix[:, 2], 1) - 15)), 255))
        report[name] = {"fit_quality": quality[b, name], "pixels": int(len(pix)), "hue_mean": mean}
    colours = Colours(objects, occluders)
    # Each range must not catch the other object, nor the mat.
    blob_px = np.isin(labels, best)
    for name, rng in objects.items():
        m = rng.mask(hsv) > 0
        report[name]["mat_false_positive"] = float((m & ~blob_px & est.rectify.valid).mean())
        for other, b in zip(names, best):
            if other != name:
                report[name][f"catches_{other}"] = float(m[labels == b].mean())
    return colours, report


# ---------------------------------------------------------------------------------------------- commands
def _images(paths) -> list[np.ndarray]:
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")) if p.is_dir() else [p]
    out = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files]
    if not out or any(i is None for i in out):
        raise SystemExit(f"cannot read images from {paths}")
    return out


def _camera(args):
    from .camera import C920, CameraSettings
    s = CameraSettings.load(args.settings)
    if args.index is not None:
        s.index = args.index
    return C920(s)


def cmd_boards(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.out or CALIBRATION / "print")
    out.mkdir(parents=True, exist_ok=True)

    def page(img, size_mm, name, place_mm=None):
        """A page of exact physical size: print at 100 % ("actual size"), never "fit to page"."""
        w, h = size_mm
        fig = plt.figure(figsize=(w / 25.4, h / 25.4))
        iw, ih = (img.shape[1] / 20, img.shape[0] / 20) if place_mm is None else place_mm
        ax = fig.add_axes([(w - iw) / 2 / w, (h - ih) / 2 / h, iw / w, ih / h])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axis("off")
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
        cv2.imwrite(str(out / f"{name}.png"), img)

    brd = board()
    sx, sy = brd.getChessboardSize()
    sq = brd.getSquareLength() * 1e3
    page(brd.generateImage((round(sx * sq * 20), round(sy * sq * 20)), marginSize=0), (297, 210), "charuco_board_A4_landscape")
    layout = load_layout(args.layout)
    tile = round((MARKER_SIZE + 2 * MARKER_MARGIN) * 1e3 * 20)
    for p in range(0, len(layout), 6):
        sheet = np.full((3 * tile + 4 * 300, 2 * tile + 3 * 300), 255, np.uint8)
        for k, m in enumerate(layout[p:p + 6]):
            r, c = divmod(k, 2)
            y, x = 300 + r * (tile + 300), 300 + c * (tile + 300)
            img = marker_image(m.id, m.size, m.margin)
            sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
            cv2.rectangle(sheet, (x - 1, y - 1), (x + img.shape[1], y + img.shape[0]), 170, 2)   # cut line
            cv2.putText(sheet, f"id {m.id}: {m.where}", (x, y + img.shape[0] + 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 4)
        page(sheet, (210, 297), f"mat_markers_A4_{p // 6 + 1}")
    print(f"wrote {out}: print the PDFs at 100 % scale. Cut each marker along its grey line and lay it with the small\n"
          f"triangle pointing away from the robot, its white border flush with the mat edges (positions: {args.layout}).\n"
          f"Measure the printed board squares and marker black squares;\n"
          f"if a marker is not {MARKER_SIZE * 1e3:.0f} mm, put its size in {args.layout}.\n"
          f"Board squares: {sq:.0f} mm, markers {brd.getMarkerLength() * 1e3:.0f} mm (only their ratio matters).")


def cmd_settings(args):
    from .camera import C920, CameraSettings, describe, read_back, sharpness
    s = CameraSettings.load(args.settings)
    if args.index is not None:
        s.index = args.index
    if args.backend:
        s.backend = args.backend
    for k in ("focus", "exposure", "gain", "white_balance"):
        if getattr(args, k) is not None:
            setattr(s, k, getattr(args, k))
    s.autofocus = s.auto_exposure = s.auto_white_balance = False
    if args.auto:
        # Let the camera's automatics settle on the scene, then lock what they chose.
        auto = CameraSettings(**{**asdict(s), "autofocus": True, "auto_exposure": True, "auto_white_balance": True})
        cam = C920(auto)
        time.sleep(3.0)
        d = read_back(cam.cap)
        cam.close()
        print("automatics chose: " + describe(d))
        for key, val in (("focus", d["focus"]), ("exposure", d["exposure"]), ("white_balance", d["white_balance"])):
            if getattr(args, key) is None and val not in (-1, 0):
                setattr(s, key, val)
    if args.sweep_focus and args.focus is None:
        cam = C920(s)
        scores = []
        for f in range(0, 105, 5):
            cam.cap.set(cv2.CAP_PROP_FOCUS, f)
            time.sleep(0.4)
            frame, _, _ = cam.wait(cam.wait(0)[2])
            scores.append((sharpness(frame), f))
            print(f"  focus {f:3d}: sharpness {scores[-1][0]:8.1f}")
        cam.close()
        s.focus = max(scores)[1]
    s.save(args.settings)
    cam = C920(s)
    time.sleep(1.0)
    frame, _, _ = cam.wait(cam.wait(0)[2])
    print("driver reports: " + describe(read_back(cam.cap)))
    cam.close()
    cv2.imwrite(str(CALIBRATION / "c920_frame.png"), frame)
    print(f"saved {args.settings} and a frame, {CALIBRATION / 'c920_frame.png'}: check it is sharp, not clipped "
          f"(the white walls below 250) and the mat black but not crushed. Re-run with --exposure / --focus to adjust.")


def cmd_intrinsics(args):
    brd = board(square=args.square_mm / 1e3, marker=args.marker_mm / 1e3)
    frames_dir = CALIBRATION / "c920_intrinsics_frames"
    if args.images:
        images = _images(args.images)
    else:
        frames_dir.mkdir(parents=True, exist_ok=True)
        cam = _camera(args)
        print(f"Hold the ChArUco board in front of the camera: fill the corners and edges of the image, tilt it up to "
              f"about 45 degrees each way, near and far. A view is kept when it differs from the others ({args.frames} needed).")
        images, kept, n, last = [], [], 0, 0.0
        try:
            while len(images) < args.frames:
                frame, _, n = cam.wait(n)
                found = detect_board(frame, brd, min_corners=12)
                if found is None or time.monotonic() - last < 0.7:
                    continue
                pts = found[1]
                centre, spread = pts.mean(0), float(np.sqrt(cv2.contourArea(cv2.convexHull(pts))))
                if any(np.linalg.norm(centre - c) < 0.08 * frame.shape[1] and abs(math.log(spread / s)) < 0.25
                       for c, s in kept):
                    continue
                kept.append((centre, spread))
                images.append(frame)
                last = time.monotonic()
                cv2.imwrite(str(frames_dir / f"{len(images):03d}.png"), frame)
                print(f"  view {len(images)}/{args.frames}: {len(pts)} corners at ({centre[0]:.0f}, {centre[1]:.0f})")
        finally:
            cam.close()
    views = [v for v in (detect_board(i, brd) for i in images) if v is not None]
    h, w = images[0].shape[:2]
    result = calibrate_intrinsics(views, (w, h))
    result["board"] = {"squares": list(brd.getChessboardSize()), "square_m": brd.getSquareLength(),
                       "marker_m": brd.getMarkerLength(), "dictionary": "DICT_5X5_100"}
    Path(args.out or INTRINSICS).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out or INTRINSICS).write_text(json.dumps(result, indent=1))
    K = np.array(result["K"])
    print(f"{result['views']} views, RMS reprojection {result['rms_px']:.2f} px (good: below 0.5)")
    print(f"focal {K[0, 0]:.1f} / {K[1, 1]:.1f} px, principal point ({K[0, 2]:.1f}, {K[1, 2]:.1f}), "
          f"distortion {np.round(result['dist'], 4).tolist()}")
    print(f"vertical field of view {result['fovy_deg']:.2f} deg (the scene assumes {NOMINAL_FOVY}), "
          f"horizontal {result['fovx_deg']:.2f} deg; saved {args.out or INTRINSICS}")


def cmd_extrinsics(args):
    intr = json.loads(Path(args.intrinsics).read_text())
    layout = load_layout(args.layout)
    if args.touch:
        layout = touch_markers(layout, args)
        save_layout(layout, args.layout)
        print(f"saved the touched marker positions to {args.layout}")
    if args.images:
        frames = [detect_markers(i) for i in _images(args.images)]
    else:
        cam = _camera(args)
        frames, n = [], 0
        try:
            for _ in range(args.frames):
                frame, _, n = cam.wait(n)
                frames.append(detect_markers(frame))
        finally:
            cam.close()
        cv2.imwrite(str(CALIBRATION / "c920_extrinsics_frame.png"), frame)
    detected = merge_detections(frames)
    result = solve_extrinsics(detected, layout, intr["K"], intr["dist"], (intr["width"], intr["height"]))
    Path(args.out or EXTRINSICS).write_text(json.dumps(result, indent=1))
    missing = [m.id for m in layout if m.id not in detected]
    print(f"markers used {result['markers']}" + (f", not seen {missing}" if missing else ""))
    print(f"reprojection {result['reprojection_px']['rms']:.2f} px RMS ({result['reprojection_px']['max']:.2f} max)")
    print("error on the mat per marker (mm): " + ", ".join(f"{i}: {e:.2f}" for i, e in result["mat_error_mm"].items()))
    if result["held_out_mm"]:
        print("... each solved without it (mm):  " + ", ".join(f"{i}: {e:.2f}" for i, e in result["held_out_mm"].items()))
    p, off = result["camera_position_m"], result["offset_from_nominal_mm"]
    print(f"camera at ({p[0] * 1e3:.1f}, {p[1] * 1e3:.1f}, {p[2] * 1e3:.1f}) mm above the mat; "
          f"{np.linalg.norm(off):.0f} mm from the scene's c920 position")
    worst = max((result["held_out_mm"] or result["mat_error_mm"]).values())
    print("PASS" if worst <= 1.0 else "CHECK: above 1 mm on the mat; re-measure the marker positions (or use --touch)",
          f"(worst {worst:.2f} mm); saved {args.out or EXTRINSICS}")


def touch_markers(layout: list[Marker], args) -> list[Marker]:
    """Measure each marker's centre with the rod tip (robot frame by forward kinematics through the joint map)."""
    from ..kinematics import PusherKinematics
    from ..scene import build_model
    from .joint_map import JointMap
    from .robots import SO101Robot
    jmap = JointMap.load(args.map)
    kin = PusherKinematics(build_model(SceneConfig(timestep=0.005, lightbox=True), visual=False))
    robot = SO101Robot(args.port)
    out = []
    try:
        input("Torque will turn OFF. Support the arm with one hand, then press Enter...")
        robot.set_torque(False)
        for m in layout:
            if input(f"Rest the rod tip on the centre of marker {m.id} ({m.where}), rod vertical, then Enter "
                     f"(s + Enter: skip it)...").strip().lower() == "s":
                continue
            q = np.mean([jmap.to_sim(robot.read_joints()) for _ in range(10)], 0)
            tip, _ = kin.forward(q[None])
            print(f"  marker {m.id}: ({tip[0, 0] * 1e3:.1f}, {tip[0, 1] * 1e3:.1f}) mm, "
                  f"{np.linalg.norm(tip[0, :2] - m.centre) * 1e3:.1f} mm from the layout")
            out.append(Marker(**{**asdict(m), "centre": (float(tip[0, 0]), float(tip[0, 1]))}))
        input("Hold the arm in a safe pose; torque turns back ON when you press Enter...")
        robot.set_torque(True)
    finally:
        robot.close()
    return out


def cmd_colours(args):
    camera = CameraModel.load(args.intrinsics, args.extrinsics)
    if args.images:
        image = _images(args.images)[0]
    else:
        cam = _camera(args)
        image, _, _ = cam.wait(cam.wait(0)[2])
        cam.close()
    colours, report = learn_colours(image, camera)
    colours.save(args.out or COLOURS)
    for name, r in report.items():
        c = colours.objects[name]
        print(f"{name}: hue {c.hue}, saturation >= {c.sat[0]}, value >= {c.val[0]} "
              f"(fit quality {r['fit_quality']:.2f}, mat pixels caught {r['mat_false_positive']:.4%}, "
              + ", ".join(f"{k} {v:.1%}" for k, v in r.items() if k.startswith("catches_")) + ")")
    bad = [n for n, r in report.items() if r["mat_false_positive"] > 1e-3 or any(v > 0.01 for k, v in r.items() if k.startswith("catches_"))]
    est = PoseEstimator(camera, colours, tuple(colours.objects))
    overlay = Path(args.out or COLOURS).with_suffix(".png")
    cv2.imwrite(str(overlay), draw(image, camera, est.estimate(image)))
    print(("CHECK: " + ", ".join(bad) + " overlap other things; use more distinct colours. " if bad else "")
          + f"saved {args.out or COLOURS} and {overlay} (fitted outlines)")


def cmd_check(args):
    from .camera import ReplaySource
    from .poses import CameraPoseSource
    source = ReplaySource(_images(args.images), fps=30, loop=True) if args.images else None
    if source is None:
        from .camera import C920, CameraSettings
        s = CameraSettings.load(args.settings)
        if args.index is not None:
            s.index = args.index
        source = C920(s)
    for path, step in ((args.intrinsics, "intrinsics"), (args.extrinsics, "extrinsics"), (args.colours, "colours")):
        if not Path(path).exists():
            source.close()
            raise SystemExit(f"{path} is missing: run the {step} step first")
    # No joint angles here, so no arm mask: keep the arm off the mat (poses need 60 % of an object visible).
    poses = CameraPoseSource(source=source, camera=CameraModel.load(args.intrinsics, args.extrinsics),
                             colours=Colours.load(args.colours))
    expect = None if args.expect is None else np.array([args.expect[0], args.expect[1], math.radians(args.expect[2])])
    errors = []
    t_end, frame = time.monotonic() + args.seconds, None
    try:
        while time.monotonic() < t_end:
            time.sleep(0.5)
            found, stamp = poses.detections()
            if found is None:
                continue
            line = []
            for name, d in found.items():
                if d.pose is None:
                    line.append(f"{name}: dropped ({d.status}, visible {d.visible:.2f})")
                else:
                    line.append(f"{name}: ({d.pose[0] * 1e3:6.1f}, {d.pose[1] * 1e3:6.1f}) mm {math.degrees(d.pose[2]):7.1f} deg"
                                f" visible {d.visible:.2f}")
            if expect is not None and found.get(poses.object_name) is not None and found[poses.object_name].pose is not None:
                p = found[poses.object_name].pose
                e = (np.linalg.norm(p[:2] - expect[:2]) * 1e3, abs(math.degrees((p[2] - expect[2] + math.pi) % (2 * math.pi) - math.pi)))
                errors.append(e)
                line.append(f"error {e[0]:.1f} mm {e[1]:.1f} deg")
            print("  " + " | ".join(line))
        print(poses.summary())
        if errors:
            e = np.array(errors)
            print(f"against --expect: median {np.median(e[:, 0]):.2f} mm / {np.median(e[:, 1]):.2f} deg "
                  f"(pass: 2 mm / 2 deg)")
        if not args.images:
            frame, _, _ = source.wait(source.wait(0)[2])
            found, _ = poses.detections()
            cv2.imwrite(str(CALIBRATION / "c920_check.png"), draw(frame, poses.estimator.camera, found or {}))
            print(f"saved {CALIBRATION / 'c920_check.png'} (fitted outlines on the last frame)")
    finally:
        poses.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index", type=int, default=None, help="camera index (default: from the settings file)")
    common.add_argument("--settings", default=str(CALIBRATION / "c920_settings.json"))
    common.add_argument("--images", nargs="*", default=None, help="saved images (files or folders) instead of the camera")
    common.add_argument("--out", default="")
    p = sub.add_parser("boards", parents=[common], help="printable ChArUco board and mat markers")
    p.add_argument("--layout", default=str(MARKERS))
    p = sub.add_parser("settings", parents=[common], help="fix focus, exposure and white balance")
    p.add_argument("--backend", default="", choices=["", "dshow", "msmf", "v4l2", "any"])
    p.add_argument("--focus", type=float, default=None)
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--white-balance", type=float, default=None)
    p.add_argument("--auto", action="store_true", help="lock the values the camera's automatics settle on")
    p.add_argument("--sweep-focus", action="store_true", help="choose the sharpest focus (unless --focus)")
    p = sub.add_parser("intrinsics", parents=[common], help="ChArUco board views -> camera matrix and distortion")
    p.add_argument("--frames", type=int, default=25)
    p.add_argument("--square-mm", type=float, default=35.0)
    p.add_argument("--marker-mm", type=float, default=26.0)
    p = sub.add_parser("extrinsics", parents=[common], help="mat markers -> camera pose in the robot frame")
    p.add_argument("--intrinsics", default=str(INTRINSICS))
    p.add_argument("--layout", default=str(MARKERS))
    p.add_argument("--frames", type=int, default=30, help="frames averaged")
    p.add_argument("--touch", action="store_true", help="first measure the marker centres with the rod tip")
    p.add_argument("--port", default="")
    p.add_argument("--map", default=str(CALIBRATION / "joint_map.json"))
    p = sub.add_parser("colours", parents=[common], help="learn the T's and the box's colours")
    p.add_argument("--intrinsics", default=str(INTRINSICS))
    p.add_argument("--extrinsics", default=str(EXTRINSICS))
    p = sub.add_parser("check", parents=[common], help="live poses, dropped frames and timing")
    p.add_argument("--intrinsics", default=str(INTRINSICS))
    p.add_argument("--extrinsics", default=str(EXTRINSICS))
    p.add_argument("--colours", default=str(COLOURS))
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--expect", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW_DEG"),
                   help="the T's true pose (m, m, deg): print the error")
    args = ap.parse_args(argv)
    {"boards": cmd_boards, "settings": cmd_settings, "intrinsics": cmd_intrinsics, "extrinsics": cmd_extrinsics,
     "colours": cmd_colours, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
