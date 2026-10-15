# The lightbox: from simulation to the real table

Our real setup is NVIDIA's Sim-to-Real SO-101 lightbox: a white foam-board box (76.2 × 50.8 × 50.8 cm, open at
the front where the robot stands), a black EVA mat (45.7 × 30.5 × 0.5 cm) on a 0.48 cm foam board, and a Logitech
C920 pushed through a slot in the left wall. This page records what that changes for the policies, what was
fixed, what the simulator predicts, and the order in which to test the real thing.

`SceneConfig(lightbox=True)` (or `--set scene.lightbox=true` on any command that takes overrides) builds this
workspace. It reproduces the measured scene file `so101_lightbox.py` exactly (with the arm at its HOME pose the
gripper frame is 11.67 cm above the mat in both) with one change of frame: **z = 0 is the mat top**, so the
robot base sits 9.8 mm below it. Objects, goals, the rod height and every task file keep their coordinates.

## What the lightbox changes

| | measured | before (flat table) | consequence |
|---|---|---|---|
| Work surface height | mat top 9.8 mm above the robot's standing surface | level with the robot | the kinematics must know it: told nothing, IK drives the rod 4.8 mm *into* the mat |
| Mat near edge | 7.9 cm in front of the pan axis (11.8 cm from the base origin) | endless table | objects can be pushed off; the band's failure boundary reaches 10.0 cm |
| Joint angles for a given rod position | lift −1.2°, elbow −3°, wrist +4.3° (mean, while pushing) | | the policy reads 4 joint angles: a shifted input |
| Camera | C920 on the left wall, (20.2, 36.6, 40) cm, 46° down at the mat centre | D435i opposite the robot (outside the box now) | the arm hides more than half the T in 29 % of views (C6); webcam latency unknown |
| Mat friction (EVA foam) | not measured | trained on 0.20–0.50 | rubbery mats can be grippier |
| Walls | side walls 15 cm, back wall 20 cm beyond the farthest rod position | none | no effect on pushing |
| Vertical-rod reach | 7 mm shorter standing below the mat: 29.2 cm straight ahead, 27.3 cm at 55° | 29.9 / 28.0 cm | 0.025 % of the reach ring (its outer corners) is lost; the controller clamps there |

The camera's field of view in `so101_lightbox.py` (`CAM_FOVY = 52` at 4:3) assumes the 78° diagonal applies at
4:3. Logitech's 78° is the 16:9 figure (70.4° × 43.3°), so the vertical field is 43.3° at 1920 × 1080; the
scene uses that. Confirm it with the intrinsic calibration (step C2 below). At 1080p the whole pushing band is in
view (`c920` camera; only the mat's near-left corner, under the camera, is cut off).

## What the simulator predicts

`python -m s2r2s.lightbox <checkpoint>` runs 1,000 held-out scenes per row, the same scenes in every row.

| condition (lightbox unless stated) | `tee_v1` (flat table) | `tee_lightbox_v1` |
|---|---|---|
| training scene (flat table, robot standing on it) | 96.6 % | **98.9 %** |
| lightbox, measured geometry | 93.8 % | **98.5 %** |
| lightbox, controller not told about the mat (rod 4.8 mm into it) | 23.5 % | **69.4 %** |
| mat 4 mm lower than modelled (rod 9 mm above it) | 92.0 % | **98.8 %** |
| mat 2 mm higher (rod 3 mm above) | 95.3 % | **99.1 %** |
| mat 4 mm higher (rod 1 mm above) | 96.6 % | **98.5 %** |
| mat 6 mm higher (rod 1 mm into it) | 88.4 % | **93.2 %** |
| mat friction 0.20 | 95.9 % | **98.1 %** |
| mat friction 0.35 | 95.3 % | **99.0 %** |
| mat friction 0.50 | 94.6 % | **98.9 %** |
| mat friction 0.70 | 91.6 % | **98.5 %** |
| mat friction 0.90 | 90.8 % | **98.1 %** |
| camera latency 100–150 ms | 92.8 % | **98.4 %** |
| camera latency 150–200 ms | 89.0 % | **98.1 %** |
| camera latency 200–300 ms | 75.8 % | **89.9 %** |
| camera noise 2 mm / 2° | 92.1 % | **98.2 %** |
| joint calibration error up to 0.5° per joint | 93.5 % | **98.9 %** |
| ... up to 1° | 87.6 % | **97.2 %** |
| ... up to 2° | 74.3 % | **95.2 %** |
| backlash ±0.5°, pan only | 88.2 % | **98.5 %** |
| backlash ±0.5°, every joint (Menagerie's figure) | 79.8 % | **98.8 %** |
| backlash ±1°, every joint | 30.9 % | **98.5 %** |
| all at once: calibration ≤1°, backlash ±0.5°, μ 0.2–0.9, latency 0–150 ms | 71.3 % | **97.7 %** |

Other skills in the lightbox (no retraining): gate 95.3 → 90.6 % (0.4 % pushed off the mat), clutter 94.9 → 93.5 %,
Level 2 92.4 → 92.4 %, Level 3 82.5 → 80.9 % (`--set scene.lightbox=true`).

What this says:
1. **The height must be in the model.** Without it, Push-T falls to 23.5 %, the rod dragging through the mat.
2. **The flat-table policy loses 2.8 points in the lightbox, all from its joint-angle inputs.** Given the joint angles
   a robot standing level with the mat would have for the same rod position, it scores 97.6 %. On the flat table,
   the rod's height (1–9 mm above the table) makes no difference (96.0–96.6 %). The joint angles add no
   information beyond the rod position, so any error in them (height, calibration, backlash) only costs.
3. **Hardware errors matter more than any lightbox measurement.** Backlash of ±0.5° on every joint (the figure in the
   Menagerie model) takes the flat-table Push-T policy to 79.8 %, ±1° to 30.9 %. Joint calibration error up to 1°
   per joint takes it to 87.6 %. Neither was in the simulator before; both are now (`rand.joint_offset_deg`,
   `rand.backlash_deg`, off by default).
4. **Mat friction, camera latency and camera noise are tolerable but need measuring.** μ 0.7–0.9 costs 2–3 points,
   latency 150–200 ms costs 5, 200–300 ms costs 18.
5. **Train in the lightbox with the hardware errors.** `tee_lightbox_v1` (below) is the Push-T policy for the real
   table: 98.5 % in the lightbox, 97.7 % with every error above at once.
6. **Move the camera if you can** (C6): the as-built left-wall slot is the second-worst mount for arm occlusion.

## The lightbox Push-T policy (`runs/tee_lightbox_v1`)

`tee_v1` fine-tuned for 30 minutes (best checkpoint after 14 minutes, 54 M samples) in the lightbox, with joint
calibration error up to 1° per joint, backlash ±0.5° on every joint, mat friction 0.2–0.9 and camera latency 0–150 ms:

```powershell
python -m s2r2s.train --run tee_lightbox_v1 --init runs	ee_v1est.pt --no-curriculum --minutes 30 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3
```

It scores 98.5 % in the lightbox and 97.7 % with all of those errors at once (the flat-table policy: 71.3 %), and it
is better on the flat table too (98.9 % against 96.6 %): the extra randomisation cost nothing. Through the hardware
path on the simulated arm (`hardware.runner --robot sim`, 20 episodes) it solves 20 of 20, median 1.7 mm / 1.5°.
Its weak spots are a rod pressed into the mat (93.2 % at 1 mm into it, 69.4 % at 4.8 mm) and latency above 200 ms (89.9 %).

## Bring-up, in order

Every step names its command, what passes, and the simulation number that justifies the threshold. Do not skip
ahead: each step removes one suspect from the list when a real push goes wrong.

**A. Check the givens (no motion)**
1. *Mat height.* Calipers from the floor board to the mat top at the four corners and the centre. Model: 9.8 mm.
   Pass: within ±2 mm everywhere (rod 3–9 mm above the mat keeps the policy within a point or two). Otherwise set
   `scene.mat_height` to the measured value. Keep the mat flat (no bubbles, taped edges).
2. *Mat and robot placement.* Mat near edge 7.9 cm from the pan axis, both centred left to right; base clamped.
   A mat further from the robot puts start poses over its near edge.
3. *Rod.* Ø12 mm, 30 mm below the jaw tips (calipers). Otherwise set `pusher_radius` / `pusher_tip_z`.
4. *Friction.* Tilt test: a printed, ballasted T on a strip of the mat on a board; raise one end until it slides;
   μ = tan(angle), five repeats, also rod against T. Pass: inside the policy's training range (0.20–0.90 for
   `tee_lightbox_v1`). Then re-run `python -m s2r2s.lightbox runs\tee_lightbox_v1\best.pt --set rand.table_friction=μ,μ`.
5. *Objects.* Print the T and the box to `objects.py`; colour them for the camera: bright, matte and different
   from each other (the mat is black, the walls white).

**B. The arm, no objects**
1. LeRobot calibration through every joint's full range (HARDWARE.md §1).
2. *Touch-point calibration on the mat:* `python -m s2r2s.hardware.calibrate --robot so101 --port COM5 --verify --lightbox`.
   Pass: RMS ≤ 1.5 mm, no point above 3 mm. (Simulated calibration error ≤0.5° per joint costs the flat-table
   policy 0.3 points, ≤1° costs 6.) The residual cannot catch a missing `--lightbox`: in simulation, a fit on the
   mat with the flat-table model still reaches 0.56 mm RMS, hiding the 9.8 mm in elbow and wrist offsets 4–5° off,
   and when it also fits the rod length (no `--no-fit-rod`) the rod ends 1 mm into the mat. Step 3 catches that.
3. *Rod height.* Hover at commanded heights +2, +1, 0 and −1 mm over the nine points (`--verify` shows the
   points); paper should slide under at +1 mm and catch at −1 mm. This checks mat height and calibration together.
4. *Backlash.* At reaches of 15, 20 and 25 cm, bring the rod onto a mark from the left, then from the right (and
   from the front and back), with a pen or the camera. The gap ÷ reach is the play. ±0.5° ≈ ±1.7 mm at 20 cm.
   Re-run `python -m s2r2s.lightbox` with `--set rand.backlash_deg=<play>,...` to see what it costs.
5. *Tracking:* `python -m s2r2s.hardware.tracking --robot so101 --port COM5 --map calibration\joint_map.json --lightbox`.
   Pass: lag ≤ 100 ms, joint error after the lag ≤ 1°.

**C. The camera**
1. 1920 × 1080 MJPG at 30 fps; autofocus, auto-exposure and auto white balance off (fixed values, saved).
2. Intrinsics from a ChArUco board (OpenCV). Check the vertical field of view against 43.3°.
3. Extrinsics: four or more ArUco markers at known mat positions, or touch board corners with the rod tip;
   `solvePnP`; reprojection error on the mat plane ≤ 1 mm.
4. Pose estimator (`CameraPoseSource` in `hardware/poses.py`): segment the coloured top faces, back-project onto the
   plane 20 mm above the mat, fit the template. Pass: ≤ 2 mm / 2° on a printed template at ten poses across the band
   (2 mm / 2° noise costs the flat-table Push-T policy about 2 points).
5. Latency: move the rod at a known speed and cross-correlate its camera track with FK. Pass: ≤ 150 ms
   (100–150 ms: −1 point for `tee_v1`; 150–200 ms: −5; 200–300 ms: −18). Keep the capture buffer at one frame.
6. Occlusion: `python -m s2r2s.camera_study --checkpoint runs	ee_lightbox_v1est.pt --lightbox` casts rays from mounts inside the
   box to the T's top face (12,800 views each). Mean share hidden by the arm / share of views more than half hidden:
   back wall, centre, 25 cm above the mat **9.6 % / 1.9 %**; right-wall slot 17.2 % / 4.8 %; back-left corner 18.8 % / 6.8 %;
   back wall, 45 cm up 22.8 % / 15.5 %; **left-wall slot as built 34.2 % / 29.3 %**; overhead 62.0 % / 68.9 %. The arm's links
   sit to one side of its base and pass between the left-wall camera and the T. Move the C920 to the middle of the
   back wall, low, if the build allows; otherwise the pose estimator must treat heavily hidden views as missing frames
   (the policy's tolerance to dropped frames was tested with scattered drops, not with the bursts an occluding arm causes).

**D. Pushing without the policy**
1. Open-loop replay: a straight push through the T's centre and an off-centre push, five times each, in simulation
   (`SimRobot`, lightbox) and on the table. Compare the final T poses; adjust the friction range if they disagree.
2. Repeatability: the same scene three times, same homing move (the supervisor's backlash question: spread within a
   scene vs between scenes).

**E. Policy trials**
1. Rehearse on the simulated arm: `python -m s2r2s.hardware.runner --checkpoint runs\tee_lightbox_v1\best.pt --robot sim --episodes 20`.
2. Real: `python -m s2r2s.hardware.runner --checkpoint runs\tee_lightbox_v1\best.pt --robot so101 --port COM5 --map calibration\joint_map.json --max-step-deg 2`,
   20–30 logged trials from held-out scenes. Compare with `python -m s2r2s.lightbox` re-run at the *measured*
   friction, latency, noise and backlash: that is the simulator's prediction for this table.
3. Then clutter, the gate (build the wall), and Levels 2 and 3 (they need the planner in the runner).

Policies trained on the flat table run in the lightbox only with `--lightbox` on the hardware tools (the kinematics
must know the mat height); policies trained in the lightbox carry it in their config.

## Still open

- The clutter, box and dock skills are still flat-table policies; retrain them like `tee_lightbox_v1` before Levels 2 and 3 on hardware.
- Backlash here is a dead band between servo command and link, the same on every joint, with the encoder reading
  the link (the STS3215's encoder sits on its output shaft), so the policy can see the play and correct it. Gravity
  holds the lift, elbow and wrist against one side of their play, so this overstates them; the pan joint carries no
  such load. Play the encoders cannot see (a loose horn, a printed coupling) acts like the calibration-error rows.
  Measure (B4) and set per-joint values.
- The camera model drops frames at random; an occluding arm drops them in bursts. Model occlusion from the real
  camera pose before trusting the left-wall slot.
- The camera pose estimator (`CameraPoseSource`) is still to be written: phases C and E wait for it.
- The arm has no collision geometry except the rod. The walls are out of reach, but a wrist camera mount (if the
  real arm carries one) is not checked against the forearm.
