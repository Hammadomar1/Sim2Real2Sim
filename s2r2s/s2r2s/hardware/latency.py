"""Camera latency: how old the C920's view is when it reaches the computer, plus the pose estimator's time (LIGHTBOX.md C5).

    python -m s2r2s.hardware.latency --robot sim                      # rehearsal: rendered C920, 120 ms injected delay
    python -m s2r2s.hardware.latency --robot sim --sim-delay 0.07
    python -m s2r2s.hardware.latency --robot so101 --port COM5 --map calibration\\joint_map.json --lightbox

The rod, with the rod marker taped to it (``rod_marker.py``), hovers --hover above the mat at --reach straight
ahead, and the pan joint swings it +-amplitude at --freq Hz (the peak rod speed is printed; the policies push at
up to 80 mm/s). Every control tick the joints are read and stamped (time.monotonic() halfway through the read);
the camera thread stamps every frame as it arrives (``camera.C920``: the stamp ``CameraPoseSource`` uses) and the
marker is found in each frame. The marker's azimuth about the pan axis (its centre back-projected onto its own
height) follows the pan angle the encoders read, late by the camera's latency. The delay that explains one by the
other best (least squares with free scale and offset, -0.1 to 0.5 s in 1 ms steps, refined between steps) is the
camera's latency, from exposure to arrival on the computer.

The pose estimator adds its own time per frame: it is timed on frames from the run (``vision.PoseEstimator`` with
the arm mask, as ``CameraPoseSource`` runs it; put the T on the mat in view, out of the rod's way; without
calibrated colours the 33 ms measured in simulation is assumed). End-to-end = camera + estimator. Pass: <= 150 ms
(LIGHTBOX.md: 100-150 ms costs tee_v1 1 point, 150-200 ms 5, 200-300 ms 18). Prints the matching
``--set rand.obs_latency_steps=lo,hi`` for ``python -m s2r2s.lightbox``.

--robot sim: the simulated arm at 100 Hz (its joints read at 20 Hz, as on the real arm), rendered at 30 fps by the
lightbox's C920 (``camera_sim.SimCamera`` with the rod marker and the T) and each frame stamped --sim-delay
(+- --sim-jitter) after it was taken; the tool should find that delay.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from ..kinematics import PusherKinematics
from ..scene import PAN_AXIS_X, SceneConfig, build_model
from .joint_map import JointMap
from .rod_marker import ROD_MARKER, find, locate
from .runner import move_to

PROJECT = Path(__file__).resolve().parents[2]
SIM_ESTIMATOR_MS = 33.0          # HARDWARE.md section 4: median per frame on this laptop, rendered C920 frames
PASS_MS = 150.0


# ---------------------------------------------------------------------------------------------- analysis
def azimuth(uv, camera, z: float) -> np.ndarray:
    """Angle (rad) about the pan axis of image points ``uv`` (N, 2) back-projected onto the plane at height z."""
    p = camera.backproject(np.asarray(uv, float), z)
    return np.unwrap(np.arctan2(p[:, 1], p[:, 0] - PAN_AXIS_X))


def fit_delay(t_cam, sig_cam, t_enc, sig_enc, lo: float = -0.1, hi: float = 0.5, step: float = 0.001) -> dict:
    """The delay d that best explains ``sig_cam(t) = a * sig_enc(t - d) + b`` (encoder signal interpolated in
    time), least squares over the camera samples for which every candidate delay has encoder data."""
    t_cam, sig_cam = np.asarray(t_cam, float), np.asarray(sig_cam, float)
    t_enc, sig_enc = np.asarray(t_enc, float), np.asarray(sig_enc, float)
    keep = (t_cam - hi >= t_enc[0]) & (t_cam - lo <= t_enc[-1])
    t_cam, sig_cam = t_cam[keep], sig_cam[keep]
    if len(t_cam) < 20:
        raise ValueError(f"only {len(t_cam)} camera samples overlap the joint readings")
    delays = np.arange(lo, hi + step / 2, step)
    rms, coef = [], []
    for d in delays:
        A = np.c_[np.interp(t_cam - d, t_enc, sig_enc), np.ones(len(t_cam))]
        c, *_ = np.linalg.lstsq(A, sig_cam, rcond=None)
        rms.append(float(np.sqrt(np.mean((A @ c - sig_cam) ** 2))))
        coef.append(c)
    rms = np.array(rms)
    k = int(np.argmin(rms))
    best = float(delays[k])
    if 0 < k < len(rms) - 1:                                  # parabola through the minimum and its neighbours
        den = rms[k - 1] - 2 * rms[k] + rms[k + 1]
        if den > 0:
            best += 0.5 * step * (rms[k - 1] - rms[k + 1]) / den
    return dict(delay=best, rms=float(rms[k]), scale=float(coef[k][0]), offset=float(coef[k][1]), samples=int(len(t_cam)),
                delays=delays, curve=rms)


def latency_steps(e2e_ms: float, frame_ms: float = 1000 / 30, tick_ms: float = 50.0) -> tuple[int, int]:
    """``rand.obs_latency_steps`` (control ticks) for a pose whose age when the runner reads it is e2e to e2e + one
    frame interval (the newest result is up to a frame old)."""
    return int(math.floor(e2e_ms / tick_ms)), int(math.ceil((e2e_ms + frame_ms) / tick_ms))


# ---------------------------------------------------------------------------------------------- the motion
def pan_target(q0, t: float, amplitude: float, freq: float, ramp: float = 1.0) -> np.ndarray:
    """q0 with the pan swinging +-amplitude (rad) at freq (Hz), the swing growing over the first ``ramp`` s."""
    q = q0.copy()
    q[0] += amplitude * min(1.0, t / ramp) * math.sin(2 * math.pi * freq * t)
    return q


class MarkerTrack:
    """Finds the rod marker in every frame a camera (``wait(after, timeout)``) delivers, in a thread: (stamp,
    corners or None) per frame, plus every ``keep_every``-th frame for timing the pose estimator."""

    def __init__(self, source, keep_every: int = 15, keep: int = 20):
        self.source, self.rows, self.kept = source, [], []
        self.keep_every, self.keep = keep_every, keep
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="marker-track", daemon=True)
        self._thread.start()

    def _run(self):
        count = 0
        while not self._stop:
            got = self.source.wait(count, timeout=0.5)
            if got is None:
                continue
            frame, stamp, count = got
            self.rows.append((stamp, find(frame)))
            if count % self.keep_every == 0 and len(self.kept) < self.keep:
                self.kept.append((frame, stamp))

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)


def run_real(robot, jmap, kin, camera_source, args, q0) -> tuple[list, list, list, list]:
    """The swing on the real arm: joint stamps and angles, the camera's (stamp, corners) rows and kept frames."""
    track = MarkerTrack(camera_source)
    t_enc, q_enc = [], []
    amp, t_start = math.radians(args.amplitude), time.monotonic()
    try:
        while (t := time.monotonic() - t_start) < args.seconds:
            t0 = time.monotonic()
            reading = robot.read_joints()
            t1 = time.monotonic()
            t_enc.append((t0 + t1) / 2)
            q_enc.append(jmap.to_sim(reading))
            robot.send_joints(jmap.to_robot(pan_target(q0, t, amp, args.freq)))
            time.sleep(max(0.0, 0.05 - (time.monotonic() - t0)))
        for _ in range(20):                                     # back to the middle, gently
            robot.send_joints(jmap.to_robot(q0))
            time.sleep(0.05)
    finally:
        track.stop()
    return t_enc, q_enc, track.rows, track.kept


def run_sim(robot, sim_cam, args, q0, rng) -> tuple[list, list, list, list]:
    """The swing on the simulated arm, in simulated time: the robot at 100 Hz, joints read at 20 Hz, frames
    rendered at 30 fps and stamped ``sim_delay`` (+- ``sim_jitter``) after they were taken."""
    dt = 0.01
    t_enc, q_enc, rows, kept = [], [], [], []
    amp, next_frame, frame_dt = math.radians(args.amplitude), 0.0, 1 / 30
    jmap = JointMap()
    for k in range(round(args.seconds / dt)):
        t = k * dt
        if k % 5 == 0:
            t_enc.append(t)
            q_enc.append(jmap.to_sim(robot.read_joints()))
        if t >= next_frame - 1e-9:                              # the frame exposed now
            sim_cam.set_qpos(robot.data.qpos)
            frame = sim_cam.render(rng)
            stamp = t + args.sim_delay + rng.uniform(-args.sim_jitter, args.sim_jitter)
            rows.append((stamp, find(frame)))
            if len(rows) % 15 == 0 and len(kept) < 20:
                kept.append((frame, stamp))
            next_frame += frame_dt
        robot.send_joints(jmap.to_robot(pan_target(q0, t, amp, args.freq)))
    return t_enc, q_enc, rows, kept


def estimator_ms(kept, q_at, camera, scene, colours) -> float | None:
    """Median time (ms) the pose estimator takes per frame with the arm mask, on frames from the run."""
    from .arm_mask import ArmMask
    from .vision import PoseEstimator
    names = tuple(n for n in ("tee", "box") if n in colours.objects)
    if not kept or not names:
        return None
    est = PoseEstimator(camera, colours, names)
    mask = ArmMask(camera, est.grid, scene, margin=0.006)
    times = []
    for frame, stamp in kept:
        t0 = time.perf_counter()
        est.estimate(frame, stamp, mask(q_at(stamp)))
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times[1:] if len(times) > 2 else times))


def analyse(t_enc, q_enc, rows, camera, kin, q0) -> dict:
    """Camera latency from the marker's azimuth against the encoders' pan angle."""
    seen = [(s, c) for s, c in rows if c is not None]
    if len(seen) < 30:
        raise RuntimeError(f"the rod marker was found in only {len(seen)} of {len(rows)} frames: check that the camera "
                           f"sees it (python -m s2r2s.hardware.backlash --check-marker)")
    stamps = np.array([s for s, _ in seen])
    located = [locate(c, camera) for _, c in seen]
    uv = np.array([u for u, _ in located])
    z = float(np.median([p[2] for _, p in located]))           # the marker's height (solvePnP), for back-projection
    az = azimuth(uv, camera, z)
    pan = np.array(q_enc)[:, 0] * kin.pan_sign                  # the arm's azimuth turns with the pan joint
    fit = fit_delay(stamps, az, np.array(t_enc), pan)
    tip = kin.forward(q0[None])[0][0]
    return dict(fit, frames=len(rows), marker_frames=len(seen), marker_height_m=z,
                residual_deg=math.degrees(fit["rms"]), reach_m=float(math.hypot(tip[0] - PAN_AXIS_X, tip[1])))


# ---------------------------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["sim", "so101"], default="sim")
    ap.add_argument("--port", default="")
    ap.add_argument("--map", default="", help="joint map from calibrate.py (required on the real arm)")
    ap.add_argument("--lightbox", action="store_true", help="heights above the lightbox mat (SceneConfig.lightbox)")
    ap.add_argument("--reach", type=float, default=0.20, help="rod position straight ahead (m)")
    ap.add_argument("--hover", type=float, default=0.025, help="rod tip above the mat (m)")
    ap.add_argument("--amplitude", type=float, default=8.0, help="pan swing, +- deg")
    ap.add_argument("--freq", type=float, default=0.5, help="Hz")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--max-step-deg", type=float, default=5.0)
    ap.add_argument("--sim-delay", type=float, default=0.120, help="sim: frame arrival after exposure (s)")
    ap.add_argument("--sim-jitter", type=float, default=0.005, help="sim: +- uniform (s)")
    ap.add_argument("--mount", default="left", help="sim camera: left | right | back (camera_sim.MOUNTS)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    sim = args.robot == "sim"
    scene = SceneConfig(timestep=0.005, lightbox=args.lightbox or sim)
    kin = PusherKinematics(build_model(scene, visual=False))
    q0, err, _ = kin.solve(np.array([[args.reach, 0.0, args.hover]]), kin.seed(np.array([[args.reach, 0.0]])), iterations=40)
    if err[0] > 1e-3:
        raise SystemExit(f"the rod cannot hover at {args.hover * 1e3:.0f} mm at {args.reach * 100:.0f} cm")
    q0 = q0[0]
    lever = math.hypot(args.reach - PAN_AXIS_X, 0.0)
    peak = math.radians(args.amplitude) * 2 * math.pi * args.freq * lever
    print(f"pan swing +-{args.amplitude:g} deg at {args.freq:g} Hz for {args.seconds:g} s, rod {args.reach * 100:.0f} cm out, "
          f"{args.hover * 1e3:.0f} mm above the mat: peak rod speed {peak * 1e3:.0f} mm/s")

    if sim:
        from .camera_sim import MOUNTS, SimCamera
        from .robots import SimRobot
        from .vision import DEFAULT_COLOURS
        scene.c920_pos, scene.c920_target = MOUNTS[args.mount]
        robot = SimRobot(scene, control_dt=0.01, encoder_noise_deg=0.05)
        robot.place((args.reach, 0.0), (0.27, -0.12, 0.6), tool_z=args.hover)    # the T in view, out of the way
        robot.move_by_hand(q0)
        sim_cam = SimCamera(scene, rod_marker=ROD_MARKER)
        camera, colours = sim_cam.camera(), DEFAULT_COLOURS
        t_enc, q_enc, rows, kept = run_sim(robot, sim_cam, args, q0, np.random.default_rng(3))
    else:
        if not args.map:
            ap.error("--map is required on the real arm (run calibrate.py first)")
        from .camera import C920
        from .robots import SO101Robot
        from .vision import COLOURS, CameraModel, Colours
        camera = CameraModel.load()
        colours = Colours.load() if COLOURS.exists() else None
        jmap = JointMap.load(args.map)
        robot = SO101Robot(args.port, max_step_deg=args.max_step_deg)
        source = None
        try:
            input(f"Tape the rod marker to the rod (rod_marker.py) and put the T on the mat in view, away from the rod. "
                  f"The pan joint will swing the rod +-{args.amplitude:g} deg. Press Enter ")
            move_to(robot, jmap, kin, (args.reach, 0.0, args.hover), seconds=3.0)
            source = C920()
            t_enc, q_enc, rows, kept = run_real(robot, jmap, kin, source, args, q0)
        finally:
            if source is not None:
                source.close()
            robot.close()

    r = analyse(t_enc, q_enc, rows, camera, kin, q0)
    t_arr, q_arr = np.array(t_enc), np.array(q_enc)
    q_at = lambda s: np.array([np.interp(s, t_arr, q_arr[:, k]) for k in range(q_arr.shape[1])])
    est_ms = estimator_ms(kept, q_at, camera, scene, colours) if colours is not None else None
    cam_ms = float(r["delay"] * 1e3)
    proc = est_ms if est_ms is not None else SIM_ESTIMATOR_MS
    e2e = float(cam_ms + proc)
    lo, hi = latency_steps(e2e)
    print(f"\nmarker found in {r['marker_frames']} of {r['frames']} frames ({r['marker_height_m'] * 1e3:.0f} mm above the mat); "
          f"fit over {r['samples']}: residual {r['residual_deg']:.3f} deg, scale {r['scale']:+.3f} (expected +-1)")
    if abs(abs(r["scale"]) - 1) > 0.15:
        print("  CHECK: the marker does not turn with the pan joint as expected (extrinsics? marker moved?)")
    print(f"camera latency (exposure to arrival): {cam_ms:.0f} ms")
    print(f"pose estimator: {proc:.0f} ms per frame" + ("" if est_ms is not None else
          " (assumed: no calibrated colours; run camera_calibrate colours to time it here)"))
    print(f"end-to-end: {e2e:.0f} ms -> {'PASS' if e2e <= PASS_MS else 'FAIL'} (<= {PASS_MS:.0f} ms)")
    print(f"\nSimulate it:  python -m s2r2s.lightbox runs\\tee_lightbox_v1\\best.pt --set rand.obs_latency_steps={lo},{hi}")
    print("(tee_lightbox_v1 was trained with rand.obs_latency_steps=0,3: 0-150 ms)")
    if sim:
        print(f"\nrehearsal: injected {args.sim_delay * 1e3:.0f} ms (+-{args.sim_jitter * 1e3:.0f}), found {cam_ms:.1f} ms "
              f"({cam_ms - args.sim_delay * 1e3:+.1f})")
    out = Path(args.out or (PROJECT / ("runs/rehearsal/latency_sim.json" if sim else "calibration/latency.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"robot": args.robot, "camera_ms": cam_ms, "estimator_ms": proc,
                               "estimator_measured": est_ms is not None, "end_to_end_ms": e2e, "pass": e2e <= PASS_MS,
                               "obs_latency_steps": [lo, hi], "residual_deg": r["residual_deg"], "scale": r["scale"],
                               "frames": r["frames"], "marker_frames": r["marker_frames"], "peak_speed_mm_s": peak * 1e3,
                               **({"sim_delay_ms": args.sim_delay * 1e3} if sim else {})}, indent=1))
    np.savez(out.with_suffix(".npz"), t_enc=t_arr, q_enc=q_arr, t_cam=np.array([s for s, _ in rows]),
             corners=np.array([np.full((4, 2), np.nan) if c is None else c for _, c in rows]),
             delays=r["delays"], curve=r["curve"])
    print(f"saved {out} and .npz")


if __name__ == "__main__":
    main()
