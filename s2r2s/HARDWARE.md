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

### Backlash (LIGHTBOX.md B4)

```powershell
python -m s2r2s.hardware.backlash --robot sim                     # rehearsal: a known dead band and slack
python -m s2r2s.hardware.backlash --robot so101 --port COM5 --map calibration\joint_map.json --lightbox
python -m s2r2s.hardware.backlash --robot so101 --port COM5 --map calibration\joint_map.json --lightbox --check-marker
```

At 15, 20 and 25 cm straight ahead, with the rod 25 mm above the mat, each joint (pan, lift, elbow, wrist flex)
swings 3° to one side and comes back to the same target. It settles, is measured, and then does the same from the
other side, three times. The difference between the two sides is the joint's play, measured two ways:
- **Encoders** (always): what the joint readings show, a dead band between command and link. This is what the
  simulator's `rand.backlash_deg` models (full width: `1` is ±0.5°), and the policy sees it in its joint inputs.
- **The rod's actual motion**: play the encoders cannot see (a loose horn, a printed coupling) acts like joint
  calibration error. With the C920 calibrated (§4) the camera measures it (`--probe camera`, the default then): print
  `calibration\print\rod_marker.pdf` (`camera_calibrate boards`; ArUco id 20, 30 mm), glue it to card and tape it
  flat to the rod 5–10 mm above the tip, sticking out towards the camera. Take it off before pushing anything.
  `--check-marker` first: it saves the camera's view at each test pose and says whether the marker is seen. The
  marker's image motion, divided by how far one degree of the joint moves it (from its solvePnP position and the
  arm's kinematics), is the joint's play. Without the camera (`--probe manual`, the default then), the tool waits
  after each approach while you mark the rod's position (a sharp pencil line on paper under the tip, or a dial
  gauge), then asks for the distance between the two marks and divides it by the joint's lever arm.

It prints the play per joint in degrees and at the rod tip, and the `--set` for `python -m s2r2s.lightbox`:
`rand.backlash_deg=` the encoders' share, and `rand.joint_offset_deg=` half the largest share they miss (add the
calibration residual from §2). The rows go to `calibration\backlash.json`. On the real arm a run takes about two
minutes (72 approaches), plus the prompts with `--probe manual`.

Rehearsal on the simulated arm, with a dead band of 0.6 / 1.0 / 0.8 / 1.2° and slack the encoders cannot see of
0.4 / 0 / 0 / 0.3° (pan / lift / elbow / wrist; largest over the three reaches; manual at 20 cm only;
`runs\rehearsal\backlash_sim*.json`):

| joint | dead band | encoders | dead band + slack | camera (rendered, left-wall slot) | manual (simulated pencil) |
|---|---|---|---|---|---|
| pan | 0.60° | 0.62° | 1.00° | 1.04° | 1.02° |
| lift | 1.00° | 1.01° | 1.00° | 0.99° | 1.00° |
| elbow | 0.80° | 0.83° | 0.80° | 0.83° | 0.82° |
| wrist | 1.20° | 1.19° | 1.50° | 1.50° | 1.50° |

The encoders' errors are their noise (0.05° per reading in the simulation; without it they are exact). The simulated
marker sticks out 30 mm to the gripper's left, 7 mm above the tip. From the left-wall slot it is seen at all three
test poses. Over the wider band it is seen at 6 of 7 poses, and at 5 of 7 from the back wall. Where the gripper or
its wrist-camera mount hangs over the marker, it is not seen: check with `--check-marker`.

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
compression, USB) comes on top: measure it (LIGHTBOX.md C5), as follows.

### Latency (LIGHTBOX.md C5)

```powershell
python -m s2r2s.hardware.latency --robot sim                      # rehearsal: frames stamped 120 ms late
python -m s2r2s.hardware.latency --robot so101 --port COM5 --map calibration\joint_map.json --lightbox
```

Tape the rod marker to the rod (backlash, §3) and put the T on the mat in view, away from the rod. The pan joint
swings the rod ±8° at 0.5 Hz for 12 s, 20 cm out and 25 mm above the mat: 71 mm/s at the peak (the policies push at
up to 80 mm/s). The joints are read at 20 Hz and stamped halfway through each read. The camera thread stamps every
frame as it arrives (the stamp `CameraPoseSource` uses), and the marker is found in every frame. The marker's azimuth
about the pan axis is the encoders' pan angle, late by the camera's latency. The tool fits the delay that explains one
by the other best (least squares, scale and offset free, −100 to 500 ms): that is the time from exposure to arrival.
It then times the pose estimator on frames from the run (arm mask on, as `CameraPoseSource` runs it; 33 ms is assumed
without calibrated colours) and adds the two: **end-to-end, pass ≤ 150 ms**. It also prints
`--set rand.obs_latency_steps=lo,hi` for `python -m s2r2s.lightbox`: when the runner reads a pose, it is end-to-end
to end-to-end + one frame old. Results go to `calibration\latency.json`, the raw tracks to `.npz`.

Rehearsal (`runs\rehearsal\latency_sim*.json`): rendered frames stamped 120 ± 5 ms after they were taken, from the
left-wall slot. The tool finds 120.0 ms (fit residual 0.11°, scale 0.994). The estimator adds 29 ms: 149 ms, just
inside the limit. With another rehearsal running alongside, the estimator took 42 ms and the total failed at
162 ms, so measure with nothing else running. With 70 ± 10 ms from the back wall it finds 70.1 ms: 109 ms
end-to-end. A C920 at 120 ms leaves almost no margin for the estimator.

## 5. First policy runs

```powershell
python -m s2r2s.hardware.runner --checkpoint runs\tee_v1\best.pt --robot sim --episodes 20      # rehearsal
python -m s2r2s.hardware.runner --checkpoint runs\clutter_v7\best.pt --robot sim --episodes 20  # with the block
python -m s2r2s.hardware.runner --checkpoint runs\tee_lightbox_v1\best.pt --robot sim --camera render --episodes 8
python -m s2r2s.hardware.runner --checkpoint runs\tee_v1\best.pt --robot so101 --port COM5 `
    --map calibration\joint_map.json --goal 0.20 0.05 90 --max-step-deg 2
```

Safety built into the runner:
- **Speed:** every tick, joint targets move at most `--max-step-deg` from the measured joints. LeRobot also caps them.
- **Smooth targets:** each tick's target goes to the servos in five equal steps from the last one, 10 ms apart
  (`--ramp-hz 100`), as the simulator does in training (`task.servo_ramp`). Sent in one jump (`--no-ramp`), a target
  is reached at the servo's full speed and the arm lurches 20 times a second. If the bus cannot keep up (the trial
  log's tick times run late), lower `--ramp-hz` to 50.
- **Workspace:** the rod stays inside the reachable ring, at its fixed height.
- **Lost object:** when the object hasn't been seen for `--max-pose-age` seconds (2 s), the runner takes over the
  rod. Until then the policy pushes on with the last pose it had: it closes most short losses itself (below). The
  runner then lifts the rod 25 mm (5 mm above the objects) and moves it a few centimetres away from the last known pose, at
  the policy's speed limit. The new point is inside the reach ring, at least 1 cm from the object, the block and the
  gate's walls, and the lifted path never passes over them. There the runner waits for the camera (a pose at most
  0.5 s old, whatever `--max-pose-age`), lowers the rod and hands back to the policy, restarting the controller from
  fresh measurements. With the C920 (`CameraPoseSource`,
  or `--camera render`), the point is the nearest one from which the arm's predicted shadow leaves at least 60 % of
  the object in view (`arm_mask.ViewPredictor`, the estimator's own arm mask). Without the camera, it is away from
  the object and toward the robot. When the object stays unseen, the runner tries further points. After
  `--recover-seconds` (8 s) it gives up and ends the trial with the rod lifted. `--on-lost hold` holds the arm
  still instead, as before. With the C920 in the left-wall slot, holding can deadlock: the arm stops where it hides
  the T and keeps hiding it.
- **Hidden start:** when the arm hides the object at its start position, or a clutter policy's block (whose start
  pose is where it must stay), the rod first goes (lifted) to where the arm hides least of the band. Once both are
  found, the rod comes down at the nearest clear point from which the object should stay in view. The trial log
  records how far that is from the planned start (`start_shift`).
- **Logs:** every tick has its `mode` (`policy`, `retreat`, `lower`, `lost`, `start`); every trial has its
  `recoveries`: where the object was lost, how long it took and how it ended.
- **Stopping:** Ctrl+C lifts the rod (up to 4 cm, as high as a vertical rod reaches there) and exits.

Rendered rehearsal, C920 in the left-wall slot, the same 8 held-out scenes (`runs\camera_sim\rehearsal_*.log`),
`tee_lightbox_v1` unless stated:

| runner | started | solved | frames without a pose | object lost (in episodes) | resumed / gave up |
|---|---|---|---|---|---|
| before this change: hold still after 0.5 s | 5 | 4 | 18.2 % | | |
| before this change: hold still after 5 s | 5 | 5 | 9.5 % | | |
| hold still after 0.5 s (`--on-lost hold --max-pose-age 0.5`), clear start | 8 | 5 | 37.8 % | | |
| hold still after 5 s (`--on-lost hold --max-pose-age 5`), clear start | 8 | 8 | 8.1 % | | |
| recover after 0.5 s (`--max-pose-age 0.5`) | 8 | 6 | 17.3 % | 19 (5) | 18 / 0 |
| **recover after 2 s (default)** | 8 | **8** | 7.7 % | 1 (1) | 1 / 0 |
| `tee_occlusion_v1`, recover after 0.5 s | 8 | 7 | 12.4 % | 12 (3) | 12 / 0 |

(A loss that is neither resumed nor given up was still in progress when the episode ended.) What this says:
1. **The clear start fixes the scenes that could not start.** In 3 of the 8 the arm hid the T at its start position.
   They now start 20–60 mm from the planned start, and most are solved.
2. **Recovery never deadlocked and never gave up,** but after 0.5 s it steps in too often. Once it hands back, the
   policy pushes from the same place again and the arm hides the T again: in one scene 7 times, and it missed. Given
   2 s, the policy mostly uncovers the T itself: the losses while pushing are short. With a 2 s trigger the runner
   stepped in once in all 8 episodes and every scene was solved. Holding still for 5 s also solved all 8 here, but
   when the T is still hidden after 5 s, that arm stays frozen where it hides it.
3. `tee_occlusion_v1`, fine-tuned with the left wall's occlusion, loses the T less often.

Eight scenes are few (one episode is 12 points), but the direction is clear, so **the default is 2 s**. With
`--max-pose-age 0.5` the policy acts on a stale pose for at most half a second, at the cost of the interruptions
above. Moving the camera (LIGHTBOX.md C6) removes most of the losses either way.

The clutter policy `clutter_lightbox_v1` (push the T, keep the box in place), same camera and 8 scenes
(`runs\camera_sim\rehearsal_clutter_*.log`). The trial logs (`--log-dir`) were checked tick by tick against the
box's true start pose:

| runner | started | solved, box kept | frames without a pose | object lost (in episodes) | resumed / gave up | runner moves: closest to the box |
|---|---|---|---|---|---|---|
| recover after 0.5 s, before the hidden-block start | 7 | 7 | 5.8 % | 5 (3) | 5 / 0 | 34 mm |
| hold still after 0.5 s (`--on-lost hold --max-pose-age 0.5`), clear start | 8 | 7 | 26.2 % | | | 34 mm |
| recover after 0.5 s (`--max-pose-age 0.5`) | 8 | **8** | 14.1 % | 15 (4) | 14 / 0 | 34 mm |
| recover after 2 s (`--max-pose-age 2`) | 8 | **8** | 6.7 % | 1 (1) | 1 / 0 | 34 mm |

1. **A hidden block now gets a clear start too.** In one scene the arm hid the box, not the T, at the planned
   start. Only a hidden T triggered the clear start, so that trial stopped ("no block pose from the camera"). Now
   it starts 20 mm away and is solved in every run.
2. **No runner move came near the box.** Across 28 retreats and clear starts, the rod stayed at least 34 mm from
   the box (a clear start; retreats 36 mm), never flew over it and came down at least 36 mm from it. The box moved
   in one episode, 1.4 mm (the limit is 10 mm), and before the runner stepped in: the policy pushed the T against it.
3. The rest is as with `tee_lightbox_v1`. After 0.5 s, two scenes lost the T 6 times each: the policy pushed back to
   where the arm hides it. Both were still solved. After 2 s the runner stepped in once. Holding still missed one scene.
4. The clear start's first move, lifted and straight to the park point, is not checked against the box or the T:
   while the arm hides them, their poses are not known. In 400 sampled scenes, the shadow predictor has the arm
   hiding one of them at the planned start in 142. In 6 % of those, that move passes over the box 4–5 mm above it.
   In 6 %, over the T.

`--camera render` rehearses the camera path too: every tick, the simulated arm's scene is rendered as the C920 sees
it (100 ms late) and goes through the same estimator and arm mask as `CameraPoseSource`. It takes about 0.25 s per
tick, so 100 s per episode. It renders from the left-wall slot as built; `--mount back` puts the C920 on the back
wall, 25 cm above the mat (LIGHTBOX.md C6), and the arm mask and the shadow predictor move with it.

**Torque stays on after the program ends** (LeRobot's default would drop the arm). Support the arm before
powering it down. Keep a hand near the power switch during first runs, and start with `--max-step-deg 2`.

### Levels 2 and 3 (`--level`)

```powershell
python -m s2r2s.hardware.runner --level 2 --robot sim --episodes 200                 # rehearsal, held-out scenes
python -m s2r2s.hardware.runner --level 3 --robot sim --camera render --episodes 8   # through the C920 pipeline
python -m s2r2s.hardware.runner --level 3 --robot sim --camera render --mount back --episodes 8   # from the back wall
python -m s2r2s.hardware.runner --level 2 --robot so101 --port COM5 --map calibration\joint_map.json `
    --goal 0.20 0.05 90 --box-goal 0.20 -0.06 0 --max-step-deg 2
python -m s2r2s.hardware.runner --level 3 --robot so101 --port COM5 --map calibration\joint_map.json `
    --goal 0.20 0.0 120 --max-step-deg 2
```

The skills default to the lightbox ones with ramped targets: `--tee runs\clutter_lightbox_v2\best.pt` (push the T,
keep the box), `--box runs\clutter_box_lightbox_v2\best.pt` (push the box, keep the T) and, for Level 3,
`--dock runs\dock_lightbox_v3\best.pt`. `--goal` is the T's goal (x, y in m, yaw in degrees) and `--box-goal` the
box's (Level 2); Level 3 nests the box in a corner of the T at its goal. Put both blocks, and both goals, in the
work band (17.5–22.5 cm from the pan axis, within 30° of straight ahead), at least 15 mm apart. A trial
(`hardware/levels.py`, `LevelRunner`) does what `python -m s2r2s.level2` / `level3` do in simulation:

1. **It plans from the camera's first view.** The T skill finds both blocks, with the clear start if the arm hides
   one. Level 2's planner orders the pushes so that each keeps the skills' 15 mm rule: two pushes in the order with
   more room, or three when a block must be parked first (the runner tries 1,000 parking spots; the simulator's
   batched planner tries 24). Level 3 brings the T to its goal and the box to the pre-dock spot of a corner, then
   docks. Of the T's two corners that fit the band it takes the one whose plan has fewer pushes, then more room
   (the simulator draws one at random). Without a plan the trial does not start, and says how far apart the
   blocks are: move one. Camera noise can do this when the blocks are within about a millimetre of the rule.
2. **Each push runs its skill through that skill's own twin**, as in simulation. The skill that pushes the T keeps
   the box in place; the box and dock skills keep the T. Each twin's rod guard keeps the rod clear of the block it
   keeps, at the pose the camera saw when the push began. The pose source reports the pushed block as the object
   (`swap`): the box down to a quarter visible, the T then at 60 %, as the roles have it with one policy.
3. **Pushes switch on the camera.** The next push starts once the pushed block has stayed within 8 mm / 8° of its
   target for 1 s, each of those ticks with a pose at most 0.5 s old, or after 20 s. Docking gets 10 s (the dock
   skill's episode). Its target is the corner where the camera sees the T when docking starts, fixed from then
   on: a target that follows the T lets the skill shove both blocks along. The trial ends when the last push has
   settled (Level 3: also when docking's 10 s are up; the simulator then holds the arm still), or after 60 s
   (Level 2) / 80 s (Level 3), the simulator's episode lengths. Pushes switch only while the policy has the rod.
4. **Safety, the servo ramp, the lost-object recovery and the logs are the single-policy runner's**, for the block
   the current push moves. If the camera loses the box for 2 s while the box skill pushes it, the runner lifts the
   rod out of the arm's shadow, clear of both blocks, and hands back. A recovery restarts the 1 s settle clock.
   The trial log (`trials\level2_<date_time>.json`) adds the plan, the corner, and per push the skill, target,
   start and end ticks, whether it settled, its time and how far the kept block moved (by the camera). Each tick
   carries `push` and `skill`; its `pose` and `block` are as that skill sees them: the block it pushes and the one
   it keeps. The final poses and errors come from the camera, with the rod lifted.

Rehearsal: the held-out scenes `python -m s2r2s.level2` / `level3` draw for the same seed and `--episodes`, on the
simulated arm. The simulator runs in the world the simulated arm has: nominal physics, the training camera model with
50 ms delay, no joint errors (`--set scene.lightbox=true rand.table_friction=0.35,0.35 rand.pusher_friction=0.3,0.3
rand.mass_scale=1,1 rand.com_offset=0 rand.servo_kp_scale=1,1 rand.damping_scale=1,1 rand.tool_height_error=0
rand.max_action_delay=0 rand.obs_latency_steps=1,1 rand.joint_offset_deg=0 rand.backlash_deg=0,0,0,0
rand.tool_noise=0`). Every command and its output: `runs\rehearsal\levels_*.log`, `runs\camera_sim\rehearsal_level*_render*.log`.

| 1,000 held-out scenes each | Level 2, simulator | Level 2, hardware path | Level 3, simulator | Level 3, hardware path |
|---|---|---|---|---|
| **success** (a trial that did not start is a miss) | 95.6 % | **95.3 %** | 91.0 % | **91.0 %** |
| not started: no plan from the camera's poses | | 0.8 % | | 0.8 % |
| T at its goal | 98.1 % | 98.5 % | 94.2 % | 94.7 % |
| box at its goal / nested in the corner (3 mm, 5°) | 96.9 % | 97.3 % | 94.7 % | 95.3 % |
| every push settled at its target | 93.1 % | 93.6 % | 80.9 % | 82.2 % |
| a block that had to stay moved >10 mm | 0.5 % | 0.4 % | 3.1 % | 3.0 % |
| a block left the band | 0.3 % | 0.2 % | 0.8 % | 0.9 % |
| plan as the simulator's (order, parking, corner) | | 97.6 % | | 90.8 % |
| time to finish, median | 11.0 s | 10.8 s | 17.1 s | 17.4 s |

Rows below the first two are over the 992 trials that started. What this says:
1. **The hardware path reproduces the simulator.** Success agrees within 0.3 points on 1,000 scenes (the noise of
   1,000 scenes is about ±0.7 points for Level 2, ±0.9 for Level 3), and so does every part of it. Single scenes
   do not: in Level 3, 62 scenes are assembled only in the simulator and 62 only through the hardware path, from
   different camera noise and encoder noise. Compare rates, not single trials.
2. **8 scenes in 1,000 do not start.** The training camera model's 1 mm noise puts blocks that the scene placed just
   over 15 mm apart just under it, and the runner refuses rather than break the skills' rule. The C920 estimator is
   more precise (p95 0.4 mm in simulation): with the rendered camera, a Level 2 scene this camera model refused
   started (below). On the real table, move a block a few millimetres.
3. **Level 3 plans differ from the simulator's in 9 % of scenes**, at no cost (95 % of them succeed). In 79 of the
   91 the parking push moves the other block: the runner tries 1,000 spots against the simulator's 24 and keeps the
   one with most room. In 9 the number of pushes differs (6 fewer, 3 more: the corner choice, or camera noise at
   the 15 mm limit), in 3 the order.

Through the rendered C920, 8 held-out scenes per level, from the left-wall slot as built (`--camera render
--episodes 8`, about 2 minutes per trial) and from the back wall, 25 cm above the mat (`--mount back`, LIGHTBOX.md
C6; about 1 minute per trial):

| the same 8 scenes | Level 2 | Level 3 |
|---|---|---|
| simulator, same world | 7 | 7 |
| hardware path, training camera model | 7 (one not started) | 6 |
| **hardware path, rendered C920, left wall** | **6** | **5** |
| ... frames without the T / the box | 16.1 / 23.0 % | 12.2 / 20.5 % |
| ... lost-object recoveries, all resumed (trials) | 3 (3) | 6 (3) |
| ... trials from a clear start | 1 | 2 |
| **hardware path, rendered C920, back wall** | **7** | **7** |
| ... frames without the T / the box | 1.2 / 0.8 % | 1.4 / 1.1 % |
| ... lost-object recoveries (trials) | 0 | 0 |
| ... trials from a clear start | 0 | 0 |

Eight scenes are few (one is 12.5 points), but the camera path works end to end. Every trial started: three from a
clear start 20–60 mm away, and the Level 2 scene the training camera model refused (the rendered estimate put the
blocks over 15 mm apart). All 9 losses of the pushed block were recovered; no trial gave up. The left wall hides the
block that must *stay* far more often than the pushed one: it needs 60 % in view and the arm leans over it (the box
was unseen in up to 94 % of a T push's ticks). One Level 2 miss is exactly that. The T skill guarded a box pose it
could not refresh and pushed the T into the box (22 mm). The other misses are the skills': the box skill stalled
45 mm short while seen every tick, twice the T push in Level 3 ran out of its 20 s, and once the arm hid both blocks
during the box's last push. The clutter, box and dock skills were not fine-tuned for the left wall's occlusion.

From the back wall the arm hardly hides either block: about 1 % of frames lack the T or the box, every trial
started where the scene put the rod, and no trial lost a block. Both levels match the simulator, 7 of 8. Gone are
the T pushed into the unseen box, the Level 3 trial where the arm hid both blocks, and one of the two Level 3 T
pushes that ran out of time. The two misses left are not the camera's: in Level 2 the box skill stalls 45.8 mm
short, the same to the tenth of a millimetre as from the left wall (the training camera model's run solves that
scene), and in Level 3 the T ends 8.8 mm / 14.8° off in the scene that the training camera model and the left wall
miss too (14.6 and 40.5 mm). As for Push-T, move the camera to the back wall (LIGHTBOX.md C6) before judging Levels
2 and 3 on the real table.

## 6. Trial logs (for Milestone 3)

Every real run writes `trials/<date_time>.json`: the goal, the final pose and errors, and per tick the joints,
joint targets, rod position, commanded position, camera pose (and block pose), action and gate stage. Level 2 and 3
trials (`trials/level2_<date_time>.json`) add the plan and its pushes, and per tick the push and its skill (§5).
The failure-guided sampler (Milestone 3) will read these files.
