# Bringing up the real SO-101 (Milestone 2)

The bridge in `s2r2s/hardware/` runs the trained policy on the real arm with the *same* controller and
observation code as training. Every step below was rehearsed on a deliberately miscalibrated simulated
arm (`--robot sim`). The real-arm commands are identical with `--robot so101 --port COMx`.

| Rehearsal on the simulated arm (offsets 4/−6/3/8°, elbow reversed) | Result |
|---|---|
| Policy without calibration | 0 % success (50 mm / 113° error) |
| Touch-point calibration | Signs found, offsets within 0.03° (rod length known), 0.2 mm residual |
| Policy with the calibrated map | 100 % success (10/10), 1.2 mm / 2.3° |
| Full hardware path, correct map (20 scenes) | 95 % success, matching the 96.6 % in simulation |

**In the lightbox** (our real setup: the mat is 9.8 mm above the surface the robot stands on), add `--lightbox`
to `calibrate`, `tracking` and `runner` so the kinematics put z = 0 on the mat top. Without it the rod is
commanded 4.8 mm into the mat. Policies trained in the lightbox (`runs/tee_lightbox_v1`) carry the setting.
[LIGHTBOX.md](LIGHTBOX.md) has the measured geometry, the simulator's predictions and the full bring-up order.

## 0. Install (once)

```powershell
cd D:\Sim2Real2Sim\s2r2s
uv sync --extra hardware          # adds LeRobot 0.6 (SO-101 driver) and the RealSense SDK
```

## 1. LeRobot setup (once per arm)

```powershell
.\.venv\Scripts\lerobot-find-port.exe          # which COM port is the follower arm
.\.venv\Scripts\lerobot-calibrate.exe --robot.type=so101_follower --robot.port=COM5 --robot.id=s2r2s_follower
```

Use the id `s2r2s_follower`: the bridge connects with it. During LeRobot's calibration, move every joint
through its **full** range. LeRobot puts 0° at the middle of that range, so a partial sweep shifts the zero.
Step 2 measures whatever offset remains.

Then mount the pusher rod: hold it between the jaws and close the gripper on it. The bridge holds the gripper
wherever it is when it connects. Measure the rod (length below the jaw tips, diameter); if it differs from
the model (30 mm, Ø12 mm), set `pusher_tip_z` / `pusher_radius` in `SceneConfig` (`s2r2s/scene.py`).

## 2. Joint mapping: touch nine points

```powershell
python -m s2r2s.hardware.calibrate --robot so101 --port COM5 --verify --no-fit-rod             # flat table
python -m s2r2s.hardware.calibrate --robot so101 --port COM5 --verify --no-fit-rod --lightbox  # points on the mat
```

- Mark the nine points the tool prints on the table. They are in the robot base frame: x forward, y to the robot's
  left, origin **38.8 mm behind the shoulder-pan axis** (the centre of the rotating base). Measure from that axis.
- Torque turns off, so **support the arm**. For each point, rest the rod tip on the mark (rod roughly vertical) and
  press Enter. The fit tries all 16 joint-direction combinations and solves the offsets.
- **Accept** if the RMS is below 2 mm and no point is above 3 mm. `--verify` then hovers 15 mm above each mark
  so you can check by eye. The result is `calibration/joint_map.json`; commit it, one per arm.
- If you didn't measure the rod, drop `--no-fit-rod`: the fit then estimates the rod length too
  (tip positions stay accurate; individual elbow and wrist offsets become less certain).

## 3. Tracking test

```powershell
python -m s2r2s.hardware.tracking --robot so101 --port COM5 --map calibration/joint_map.json   # add --lightbox in the lightbox
```

The rod draws an 80 mm square 30 mm above the table at 40 mm/s. Ideal servos (simulation): lag 0 ms,
0.7 mm RMS. **On the real arm, record the lag and the error after removing it.** If the lag is above
~100 ms, or the joints lag by more than ~1°, the simulator's servo model (gains, delay randomisation in
`RandomizationConfig`) must be updated and the policy retrained before trusting results.

## 4. Camera: the C920 pose estimator

`CameraPoseSource` (`s2r2s/hardware/poses.py`) gives the runner the T's (x, y, yaw) in the robot frame (z = 0 the
mat top) from the lightbox's Logitech C920 at 1920 × 1080, and through `read_block()` the 40 mm box's. Targets
(`python -m s2r2s.sensitivity`, LIGHTBOX.md): **2 mm / 2°**, under **150 ms** delay, about 30 Hz. Clutter policies
need the box as good as the T: where it is when a trial starts is where it must still be at the end, and the
controller's rod guard keeps the rod 3 mm from where the camera says it is. The two must never be confused.

Print the T and the box in **saturated, matte, different colours**. The simulator tests a blue T and a magenta box.
Avoid yellow and orange (the arm), black (the mat and the servos), white and grey (the walls).

**Calibrate** (once per camera mount; the files go to `calibration/`):

```powershell
python -m s2r2s.hardware.camera_calibrate boards                     # calibration\print\*.pdf: print at 100 %
python -m s2r2s.hardware.camera_calibrate settings --sweep-focus     # fixed focus, exposure, white balance
python -m s2r2s.hardware.camera_calibrate intrinsics                 # wave the ChArUco board: 25 views
python -m s2r2s.hardware.camera_calibrate extrinsics                 # the eight markers on the mat
python -m s2r2s.hardware.camera_calibrate colours                    # the T and the box on the mat, the arm off it
python -m s2r2s.hardware.camera_calibrate check --expect 0.20 0.05 90   # the T at a known pose (m, m, deg)
```

| step | what it does | pass |
|---|---|---|
| `settings` | Fixes focus (the sharpest of a sweep), exposure and white balance; `--auto` locks what the camera's automatics chose, `--focus` / `--exposure` / `--white-balance` set them. Prints what the driver accepted and saves `c920_frame.png`. | Sharp; the white walls not clipped; the mat black but not crushed. |
| `intrinsics` | ChArUco board (glued flat) in 25 places: corners and edges of the image, tilted, near and far. Kept views are saved, so `--images calibration\c920_intrinsics_frames` re-runs it. | RMS ≤ 0.5 px; vertical field of view ≈ 43.3° (the scene's value). |
| `extrinsics` | Eight ArUco markers in the mat's corners and against its edges (`calibration\c920_markers.json`), 30 frames averaged, `solvePnP` on their corners. | Each marker, with the camera solved without it, within 1 mm on the mat. |
| `colours` | Finds the two coloured objects, tells the T from the box by shape, sets each one's HSV ranges (`c920_colours.json`) and saves the fitted outlines (`c920_colours.png`). | No overlap with the other object or the mat (it prints CHECK otherwise). |
| `check` | Live poses, dropped frames and processing time; with `--expect`, the T's error. | ≤ 2 mm / 2° (LIGHTBOX.md C4: ten poses across the band). |

Cut the markers along their grey line and lay each one with its small triangle pointing away from the robot and
its white border flush with the mat's edges. The layout assumes the mat where LIGHTBOX.md has it: near edge 7.9 cm in
front of the pan axis, centred. If it sits elsewhere, or for the best accuracy, measure the marker centres with the
rod tip after the joint calibration (section 2). That ties the camera to the robot frame directly:
`python -m s2r2s.hardware.camera_calibrate extrinsics --touch --port COM5 --map calibration\joint_map.json`.
From the left-wall slot the camera sees markers 0–5, from the back wall 3–7; any three or more will do.

**How it works** (`hardware/vision.py`):
- Each frame is resampled onto a top-down grid of the plane 20 mm above the mat (0.5 mm per pixel, lens distortion
  included) and segmented by colour. Along an object's edge a pixel counts by how bright it is between the
  background and the object; otherwise blur and MJPG smear the colour outwards and bias the fit.
- The fit compares the object's silhouette as the camera sees it (top face plus visible sides) with the coloured
  pixels. The objective is an exact area integral, so it has no pixel steps: a coarse search, then Nelder–Mead.
  Between frames it tracks from the previous estimate, with a weak, bounded pull towards it, so a partly hidden
  object does not slide along a direction its visible part leaves free.
- **The arm is masked out.** The runner passes the joint angles every tick (`set_arm`); the arm's links are
  projected into the grid (`hardware/arm_mask.py`, 6 mm margin) and those pixels count neither way. Without the
  mask a dark servo or the rod in front of the T looks like the mat. Under heavy occlusion the T then gets fitted
  turned round.
- The T's 180° ambiguity: both ways round are fitted. If part of the T is hidden and both explain what is seen,
  the previous estimate decides; without one, the frame is dropped.
- A frame is reported missing (pose None) only when less than **25 %** of the object is visible. Starting a track
  needs 45 %. Without joint angles (no arm mask, as in `check`) every frame needs 60 %. Missing frames cost the
  policies far more than sub-millimetre errors (LIGHTBOX.md C6). `read_block()` needs 60 % of the block visible:
  a missing block estimate costs nothing (the runner keeps the last one), a wrong one moves the rod guard.
- A frame is also dropped when more than a fifth of the fitted silhouette lies on pixels that say "not this
  object": part of it has left its colour range. In simulation that happened when the lighting made the box's top
  face bluer than its range. If `check` shows `contradicted` drops on the real table, the lighting has changed
  since `colours`: run it again.
- Objects are assumed to rest on the mat. One pushed off its edge onto the floor board (a failed trial) is
  estimated several millimetres off.

**Tested on simulated C920 images** (`python -m s2r2s.hardware.camera_sim`). The renders of the lightbox's `c920`
camera have lens distortion, blur, sensor noise, exposure and white-balance error, varied lighting and MJPG
compression. Frames come from `clutter_v7` pushing in the lightbox. The arm mask uses joint angles up to 0.5° off,
from 100 ms before the frame. Accepted estimates (`runs/camera_sim/eval*.json`; the left-wall slot as built and
the true camera model unless a row says otherwise):

| frames | T accepted | T error p50 / p95 / max | box accepted | box error p50 / p95 / max |
|---|---|---|---|---|
| tracked at 20 Hz (6 pushing episodes × 200 frames) | 86.8 % | 0.22 / 0.41 / 1.06 mm, 0.09 / 0.35 / 2.26° | 88.1 % | 0.17 / 0.45 / 0.65 mm, 0.10 / 0.63 / 1.44° |
| single frames, no tracking (300 scenes anywhere in the band, 600 mid-push) | 70.9 % | 0.24 / 0.58 / 1.01 mm, 0.12 / 0.48 / 1.13° | 76.0 % | 0.22 / 0.56 / 1.04 mm, 0.12 / 0.52 / 1.38° |
| tracked, camera model from the simulated calibration (`--calibrated`) | 92.3 % | 0.29 / 0.56 / 0.86 mm, 0.10 / 0.40 / 1.28° | 100 % | 0.14 / 0.22 / 0.36 mm, 0.10 / 0.33 / 0.49° |
| tracked, C920 on the back wall, 25 cm up (`--mount back`) | 98.2 % | 0.11 / 0.21 / 1.62 mm, 0.06 / 0.19 / 2.13° | 90.8 % | 0.16 / 0.27 / 0.39 mm, 0.11 / 0.29 / 0.99° |

Each row draws its own scenes, so the accepted shares differ mostly with how much the arm hides. Of the 11,057
accepted estimates in these four runs (`runs/camera_sim/eval*.json`), none is off by more than 2 mm. Three are off
by more than 2° (2.1–2.3°, the T while tracking), and none by more than 10 mm or 20°. Tracking from the left wall,
the T comes back in every frame with at least 60 % of it visible, in 95 % of frames 40–60 % visible and in 92 % of
frames 25–40 % visible. Frames under 25 % are dropped. The simulated calibration (ChArUco intrinsics, mat markers)
recovered the camera within 0.41 mm / 0.04° and the vertical field of view within 0.01°.

**Delay.** On this laptop (idle) a frame takes 33 ms median (39 ms p95) from its arrival: the arm mask and both
objects. The capture thread decodes the MJPG frame in another 5 ms, in parallel, so the estimator keeps up with
30 fps. The trial logs record the processing time (`camera` in each file). The C920's own delay (exposure,
compression, USB) comes on top: measure it (LIGHTBOX.md C5).

## 5. First policy runs

```powershell
python -m s2r2s.hardware.runner --checkpoint runs\tee_v1\best.pt --robot sim --episodes 20      # rehearsal
python -m s2r2s.hardware.runner --checkpoint runs\clutter_v7\best.pt --robot sim --episodes 20  # with the block
python -m s2r2s.hardware.runner --checkpoint runs\tee_lightbox_v1\best.pt --robot sim --camera render --episodes 5
python -m s2r2s.hardware.runner --checkpoint runs\tee_v1\best.pt --robot so101 --port COM5 `
    --map calibration\joint_map.json --goal 0.20 0.05 90 --max-step-deg 2
```

Safety built into the runner:
- **Speed:** every tick, joint targets move at most `--max-step-deg` from the measured joints. LeRobot also caps them.
- **Workspace:** the rod stays inside the reachable ring, at its fixed height.
- **Lost object:** the arm holds still when the object hasn't been seen for `--max-pose-age` seconds (0.5 s).
  With the C920 in the left-wall slot this can deadlock: the arm stops where it hides the T and keeps hiding it.
  In a rendered rehearsal of `clutter_v7` (`--camera render`, below) the arm froze over the T for the last 11 s;
  the same scene with `--max-pose-age 5` was solved. `tee_lightbox_v1` solved 4 of 5 rendered episodes at 0.5 s
  and 5 of 5 at 5 s. Three more of the 8 scenes could not start at all: the arm hid the T at its start position.
  Move the camera (LIGHTBOX.md C6) or allow a longer gap, with a hand near the power switch.
- **Stopping:** Ctrl+C lifts the rod 4 cm and exits.

`--camera render` rehearses the camera path too: every tick, the simulated arm's scene is rendered as the C920 sees
it (100 ms late) and goes through the same estimator and arm mask as `CameraPoseSource`. It takes about 0.25 s per
tick, so 100 s per episode.

**Torque stays on after the program ends** (LeRobot's default would drop the arm). Support the arm before
powering it down. Keep a hand near the power switch during first runs, and start with `--max-step-deg 2`.

## 6. Trial logs (for Milestone 3)

Every real run writes `trials/<date_time>.json`: the goal, the final pose and errors, and per tick the joints,
joint targets, rod position, commanded position, camera pose (and block pose), action and gate stage. The failure-guided
sampler (Milestone 3) will read these files.
