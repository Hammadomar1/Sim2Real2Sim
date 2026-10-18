"""Backlash: per joint, how far apart the arm stops when it comes to the same target from either side (LIGHTBOX.md B4).

    python -m s2r2s.hardware.backlash --robot sim                    # rehearsal: SimRobot with a known dead band
    python -m s2r2s.hardware.backlash --robot sim --probe camera     # ... watched by the rendered C920 (rod marker)
    python -m s2r2s.hardware.backlash --robot sim --probe manual     # ... mark-and-measure, the simulator answering
    python -m s2r2s.hardware.backlash --robot so101 --port COM5 --map calibration\\joint_map.json --lightbox

At each reach (--reaches, straight ahead, the rod tip --hover above the mat) and for each joint (pan, lift, elbow,
wrist flex) the joint swings --swing degrees to one side and comes back to the target, settles and is measured;
then the same from the other side; --cycles times. The difference between the two sides is the play:

- encoders (always): the play the joint readings show, a dead band between command and link. This is what the
  simulator's ``rand.backlash_deg`` models (the full width, ``backlash_deg=1`` is +-0.5 deg); the policy sees it
  in its joint inputs.
- camera (``--probe camera``, the default when the C920 is calibrated, HARDWARE.md section 4): the rod's actual
  displacement, from an ArUco marker taped to the rod (``rod_marker.py``; print it with ``camera_calibrate boards``).
  Its image motion, divided by how far one degree of the joint moves it (from its position, solvePnP, and the arm's
  kinematics), is the joint's actual play. What exceeds the encoders' share is play they cannot see (a loose horn,
  a printed coupling): to the policy it looks like joint calibration error (``rand.joint_offset_deg``).
- manual (``--probe manual``, the default without a calibrated camera): after each approach the tool waits while
  you mark where the rod is (a sharp pencil line on paper under the tip, or a dial gauge against the rod), then asks
  the distance between the two marks. Divided by the joint's lever arm at the tip, that is the actual play.
- none: encoders only.

Prints the play per joint in degrees and at the rod tip (mm) and the matching ``--set rand.backlash_deg=...`` (and
``rand.joint_offset_deg`` for play the encoders cannot see) for ``python -m s2r2s.lightbox``; saves the rows to
--out. With --robot sim the simulated arm has a dead band the encoders see (--sim-backlash) and slack they cannot
(--sim-slack), and the tool prints how well it recovered them.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..kinematics import PusherKinematics
from ..scene import SceneConfig, build_model
from .joint_map import JointMap
from .robots import SimRobot, SO101Robot
from .runner import move_to

PROJECT = Path(__file__).resolve().parents[2]
JOINTS = {"pan": 0, "lift": 1, "elbow": 2, "wrist": 3}
TRAINED_BACKLASH_DEG = 1.0          # tee_lightbox_v1 and the lightbox skills: rand.backlash_deg=1,1,1,1 (+-0.5 deg)


# ---------------------------------------------------------------------------------------------- probes
class Probe:
    """Measures nothing beyond the encoders."""
    name = "none"
    every_cycle = True

    def capture(self, label: str):
        return None

    def play(self, below, above, q_below, q_above, j: int, lever: float) -> float | None:
        return None


class CameraProbe(Probe):
    """The C920 (or the rendered one) watching the rod marker. ``grab(n)``: n frames taken now (BGR)."""
    name = "camera"

    def __init__(self, grab, camera, arm, frames: int = 5):
        self.grab, self.camera, self.arm, self.frames = grab, camera, arm, frames

    def capture(self, label: str):
        from .rod_marker import average, locate
        corners = average(self.grab(self.frames))
        return None if corners is None else locate(corners, self.camera)

    def play(self, below, above, q_below, q_above, j, lever):
        """The marker's image motion between the two sides as an angle of joint ``j``, after taking out what the
        other joints' measured differences explain."""
        if below is None or above is None:
            return None
        q, xyz = (q_below + q_above) / 2, (below[1] + above[1]) / 2
        g = self.arm.image_jacobian(q, xyz, self.camera)
        duv = above[0] - below[0] - sum(g[:, k] * (q_above[k] - q_below[k]) for k in range(len(q)) if k != j)
        return float(g[:, j] @ duv / (g[:, j] @ g[:, j]))


class ManualProbe(Probe):
    """Mark and measure: ``ask(prompt) -> str`` is input() on the real arm (or a simulated person). Once per joint
    and reach (the first cycle): two marks and a distance."""
    name = "manual"
    every_cycle = False

    def __init__(self, ask=input):
        self.ask = ask

    def capture(self, label):
        self.ask(f"    rod came {label}: mark where it is, then press Enter ")
        return label

    def play(self, below, above, q_below, q_above, j, lever):
        while True:
            answer = self.ask("    distance between the two marks, along the rod's motion (mm): ")
            try:
                return float(answer) / 1e3 / lever
            except ValueError:
                print("    a number, please (0 if the marks coincide)")


class SimPerson:
    """The person with the pencil, simulated: notes the true rod tip at each mark and measures the distance
    between marks to ``resolution`` (a ruler: 0.1 mm noise)."""

    def __init__(self, robot: SimRobot, noise_mm: float = 0.1, seed: int = 0):
        self.robot, self.noise, self.rng, self.marks = robot, noise_mm, np.random.default_rng(seed), []

    def __call__(self, prompt: str) -> str:
        if "distance" in prompt:
            a, b = self.marks[-2:]
            mm = float(np.linalg.norm(b - a) * 1e3 + self.rng.normal(0, self.noise))
            print(f"{prompt}{mm:.2f}")
            return f"{mm:.3f}"
        self.marks.append(self.robot.tool_tip()[:2].copy())
        return ""


# ---------------------------------------------------------------------------------------------- the measurement
def _send(robot, jmap, q, dt, realtime):
    t0 = time.monotonic()
    robot.send_joints(jmap.to_robot(q))
    if realtime:
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))


def safe_swing(kin: PusherKinematics, q0, j: int, swing: float, min_z: float = 0.005) -> float:
    """``swing`` (rad), shrunk until the rod tip stays above ``min_z`` on both sides (lift and elbow lower it)."""
    while swing > math.radians(0.5):
        q = np.tile(q0, (2, 1))
        q[:, j] += (-swing, swing)
        if kin.forward(q)[0][:, 2].min() >= min_z:
            return swing
        swing *= 0.8
    return swing


def approach(robot, jmap, q0, j: int, offset: float, settle: float, dt: float = 0.05, realtime: bool = False,
             step: float = math.radians(0.5)):
    """Joint ``j`` out to ``q0 + offset`` and back to ``q0`` in steps of ``step``, then hold ``q0`` ``settle`` s."""
    n = max(2, math.ceil(abs(offset) / step))
    out = q0.copy()
    out[j] += offset
    for q in list(np.linspace(q0, out, n + 1)[1:]) + list(np.linspace(out, q0, n + 1)[1:]):
        _send(robot, jmap, q, dt, realtime)
    for _ in range(max(1, round(settle / dt))):
        _send(robot, jmap, q0, dt, realtime)


def hold_and_read(robot, jmap, q0, reads: int, dt: float = 0.05, realtime: bool = False) -> np.ndarray:
    """Mean of ``reads`` joint readings (simulator radians) while holding ``q0``."""
    out = []
    for _ in range(reads):
        out.append(jmap.to_sim(robot.read_joints()))
        _send(robot, jmap, q0, dt, realtime)
    return np.mean(out, 0)


def measure(robot, jmap: JointMap, kin: PusherKinematics, reaches, joints, probe: Probe | None = None,
            swing_deg: float = 3.0, cycles: int = 3, settle: float = 0.5, hover: float = 0.025, reads: int = 5,
            dt: float = 0.05, realtime: bool = False, log=print) -> list[dict]:
    """The play of each joint in ``joints`` at each reach: one row per (reach, joint), angles in degrees."""
    probe = probe or Probe()
    rows = []
    for r in reaches:
        move_to(robot, jmap, kin, (r, 0.0, hover), seconds=3.0, realtime=realtime)
        q_now = jmap.to_sim(robot.read_joints())
        q0, err, _ = kin.solve(np.array([[r, 0.0, hover]]), q_now[None], iterations=30)
        if err[0] > 1e-3:
            raise ValueError(f"reach {r} m is out of the vertical rod's range at {hover * 1e3:.0f} mm")
        q0 = q0[0]
        jpos = kin.forward(q0[None], with_jacobian=True)[2][0]
        log(f"reach {r * 100:.0f} cm, rod tip {hover * 1e3:.0f} mm above the mat")
        for name in joints:
            j = JOINTS[name]
            swing = safe_swing(kin, q0, j, math.radians(swing_deg))
            lever = float(np.linalg.norm(jpos[:2, j]))          # m of horizontal tip motion per radian
            enc, ext = [], []
            for c in range(cycles):
                got = {}
                for side, label in ((-1, f"back down after {name} -"), (1, f"back down after {name} +")):
                    approach(robot, jmap, q0, j, side * swing, settle, dt, realtime)
                    q = hold_and_read(robot, jmap, q0, reads, dt, realtime)
                    got[side] = (q, probe.capture(label) if (probe.every_cycle or c == 0) else None)
                enc.append(got[1][0][j] - got[-1][0][j])
                if probe.every_cycle or c == 0:
                    p = probe.play(got[-1][1], got[1][1], got[-1][0], got[1][0], j, lever)
                    if p is not None:
                        ext.append(p)
            row = dict(reach=r, joint=name, swing_deg=math.degrees(swing), lever_mm_per_deg=lever * 1e3 * math.pi / 180,
                       encoders_deg=math.degrees(float(np.mean(enc))), encoders_sd_deg=math.degrees(float(np.std(enc))),
                       probe=probe.name, probe_deg=math.degrees(float(np.mean(ext))) if ext else None,
                       probe_sd_deg=math.degrees(float(np.std(ext))) if len(ext) > 1 else None)
            rows.append(row)
            total = row["probe_deg"] if row["probe_deg"] is not None else row["encoders_deg"]
            log(f"  {name:6s} encoders {row['encoders_deg']:5.2f} deg"
                + (f"   {probe.name} {row['probe_deg']:5.2f} deg" if row["probe_deg"] is not None else
                   ("" if probe.name == "none" else f"   {probe.name}: no measurement (marker not seen?)"))
                + f"   = {total * row['lever_mm_per_deg']:.2f} mm at the tip")
    return rows


def summarise(rows: list[dict]) -> dict:
    """Per joint, over reaches: the largest play the encoders see, the largest actual play (probe) and the
    largest share of it the encoders miss, in degrees, and the largest at the rod tip (mm)."""
    out = {}
    for name in JOINTS:
        rs = [r for r in rows if r["joint"] == name]
        if not rs:
            continue
        probe = [r["probe_deg"] for r in rs if r["probe_deg"] is not None]
        hidden = [max(0.0, r["probe_deg"] - r["encoders_deg"]) for r in rs if r["probe_deg"] is not None]
        out[name] = dict(encoders_deg=max(r["encoders_deg"] for r in rs),
                         probe_deg=max(probe) if probe else None, hidden_deg=max(hidden) if hidden else None,
                         tip_mm=max((r["probe_deg"] if r["probe_deg"] is not None else r["encoders_deg"])
                                    * r["lever_mm_per_deg"] for r in rs))
    return out


def suggestion(summary: dict) -> list[str]:
    """``--set`` values for ``python -m s2r2s.lightbox`` (to 0.05 deg; unmeasured joints 0)."""
    up = lambda x: round(max(x, 0.0) * 20) / 20
    seen = [up(summary[n]["encoders_deg"]) if n in summary else 0.0 for n in JOINTS]
    out = ["rand.backlash_deg=" + ",".join(f"{v:g}" for v in seen)]
    hidden = [s["hidden_deg"] for s in summary.values() if s["hidden_deg"] is not None]
    if hidden and max(hidden) >= 0.1:
        out.append(f"rand.joint_offset_deg={up(max(hidden) / 2):g}")
    return out


# ---------------------------------------------------------------------------------------------- command line
def _floats(text: str, n: int = 4) -> tuple:
    v = [float(x) for x in text.split(",")]
    return tuple(v + [0.0] * (n - len(v)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["sim", "so101"], default="sim")
    ap.add_argument("--port", default="")
    ap.add_argument("--map", default="", help="joint map from calibrate.py (required on the real arm)")
    ap.add_argument("--lightbox", action="store_true", help="heights above the lightbox mat (SceneConfig.lightbox)")
    ap.add_argument("--probe", choices=["auto", "camera", "manual", "none"], default="auto",
                    help="what measures the rod's actual motion (auto: the camera when calibrated, else manual)")
    ap.add_argument("--joints", nargs="+", choices=list(JOINTS), default=list(JOINTS))
    ap.add_argument("--reaches", type=float, nargs="+", default=[0.15, 0.20, 0.25], help="rod positions straight ahead (m)")
    ap.add_argument("--hover", type=float, default=0.025, help="rod tip above the mat (m)")
    ap.add_argument("--swing", type=float, default=3.0, help="how far each joint backs off before an approach (deg)")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--settle", type=float, default=0.5, help="seconds held at the target before measuring")
    ap.add_argument("--max-step-deg", type=float, default=5.0)
    ap.add_argument("--check-marker", action="store_true",
                    help="camera: only check that the rod marker is seen at every test pose (saves the views)")
    ap.add_argument("--sim-backlash", default="0.6,1.0,0.8,1.2",
                    help="sim: the simulated arm's dead band per joint (deg, full width), which the encoders see")
    ap.add_argument("--sim-slack", default="0.4,0,0,0.3",
                    help="sim: play between encoder and link per joint (deg), which they cannot see")
    ap.add_argument("--mount", default="left", help="sim camera: left | right | back (camera_sim.MOUNTS)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    sim = args.robot == "sim"
    scene = SceneConfig(timestep=0.005, lightbox=args.lightbox or sim)
    kin = PusherKinematics(build_model(scene, visual=False))
    probe_name = args.probe
    if probe_name == "auto":
        from .vision import EXTRINSICS, INTRINSICS
        probe_name = "camera" if sim or (INTRINSICS.exists() and EXTRINSICS.exists()) else "manual"

    if sim:
        robot = SimRobot(scene, encoder_noise_deg=0.05, backlash_deg=_floats(args.sim_backlash),
                         slack_deg=_floats(args.sim_slack))
        jmap = JointMap()
    else:
        if not args.map:
            ap.error("--map is required on the real arm (run calibrate.py first)")
        jmap = JointMap.load(args.map)
        robot = SO101Robot(args.port, max_step_deg=args.max_step_deg)

    probe, cam = Probe(), None
    if probe_name == "camera":
        from .rod_marker import ROD_MARKER, ArmPoints
        if sim:
            from .camera_sim import MOUNTS, SimCamera
            scene.c920_pos, scene.c920_target = MOUNTS[args.mount]
            sim_cam = SimCamera(scene, rod_marker=ROD_MARKER)
            rng = np.random.default_rng(1)

            def grab(n):
                sim_cam.set_qpos(robot.data.qpos)
                return [sim_cam.render(rng) for _ in range(n)]
            camera = sim_cam.camera()
        else:
            from .camera import C920
            from .vision import CameraModel
            camera, cam = CameraModel.load(), C920()
            seen = [0]

            def grab(n):
                frames = []
                for _ in range(n):
                    got = cam.wait(seen[0], timeout=1.0)
                    if got is not None:
                        frames.append(got[0])
                        seen[0] = got[2]
                return frames
        probe = CameraProbe(grab, camera, ArmPoints(scene))
    elif probe_name == "manual":
        probe = ManualProbe(SimPerson(robot) if sim else input)

    try:
        if not sim:
            input(f"The rod will hover {args.hover * 1e3:.0f} mm above the mat at {', '.join(f'{r * 100:.0f}' for r in args.reaches)} cm "
                  f"straight ahead, and each joint will swing {args.swing:g} deg either way. Clear the mat, keep a hand "
                  f"near the power switch, then press Enter ")
        if args.check_marker:
            return check_marker(robot, jmap, kin, args, probe, sim)
        t0 = time.time()
        rows = measure(robot, jmap, kin, args.reaches, args.joints, probe, args.swing, args.cycles, args.settle,
                       args.hover, realtime=not sim)
    finally:
        if cam is not None:
            cam.close()
        robot.close()
    summary = summarise(rows)
    print(f"\nplay per joint, the largest over {len(args.reaches)} reaches ({time.time() - t0:.0f} s):")
    for name, s in summary.items():
        line = f"  {name:6s} encoders see {s['encoders_deg']:.2f} deg"
        if s["probe_deg"] is not None:
            line += f"; {probe.name}: {s['probe_deg']:.2f} deg actual, {s['hidden_deg']:.2f} deg the encoders miss"
        print(line + f"; up to {s['tip_mm']:.2f} mm at the rod tip")
    sets = suggestion(summary)
    print(f"\nSimulate it:  python -m s2r2s.lightbox runs\\tee_lightbox_v1\\best.pt --set {' '.join(sets)}")
    print(f"(the lightbox policies were trained with rand.backlash_deg={','.join([f'{TRAINED_BACKLASH_DEG:g}'] * 4)}, "
          f"+-{TRAINED_BACKLASH_DEG / 2:g} deg; joint_offset_deg also stands for the calibration residual: add B2's)")
    if sim:
        true_b, true_s = _floats(args.sim_backlash), _floats(args.sim_slack)
        print("\nrehearsal: the simulated arm's true play, and what was measured:")
        for name, s in summary.items():
            j = JOINTS[name]
            line = (f"  {name:6s} dead band {true_b[j]:.2f} deg -> encoders {s['encoders_deg']:.2f} "
                    f"({s['encoders_deg'] - true_b[j]:+.2f})")
            if s["probe_deg"] is not None:
                line += (f";  dead band + slack {true_b[j] + true_s[j]:.2f} deg -> {probe.name} {s['probe_deg']:.2f} "
                         f"({s['probe_deg'] - true_b[j] - true_s[j]:+.2f})")
            print(line)
    out = Path(args.out or (PROJECT / ("runs/rehearsal/backlash_sim.json" if sim else "calibration/backlash.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"robot": args.robot, "probe": probe.name, "rows": rows, "summary": summary,
                               "set": sets, **({"sim_backlash_deg": _floats(args.sim_backlash),
                                                "sim_slack_deg": _floats(args.sim_slack)} if sim else {})}, indent=1))
    print(f"saved {out}")


def check_marker(robot, jmap, kin, args, probe, sim):
    """Move to each test pose and say whether the camera sees the rod marker (views saved next to --out)."""
    import cv2
    from .rod_marker import find
    if not isinstance(probe, CameraProbe):
        raise SystemExit("--check-marker needs the camera (--probe camera)")
    out = Path(args.out or (PROJECT / ("runs/rehearsal" if sim else "calibration"))) / "rod_marker_views"
    out.mkdir(parents=True, exist_ok=True)
    for r in args.reaches:
        move_to(robot, jmap, kin, (r, 0.0, args.hover), seconds=3.0, realtime=not sim)
        frames = probe.grab(1)
        if not frames:
            raise SystemExit("no frame from the camera")
        frame = frames[0]
        corners = find(frame)
        if corners is not None:
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.imwrite(str(out / f"reach_{r * 100:.0f}cm.png"), frame)
        print(f"reach {r * 100:.0f} cm: marker {'seen' if corners is not None else 'NOT SEEN: move it, or the camera'}")
    print(f"views in {out}")


if __name__ == "__main__":
    main()
