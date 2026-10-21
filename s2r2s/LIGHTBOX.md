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
| Camera | C920 on the left wall, (20.2, 36.6, 40) cm, 46° down at the mat centre | D435i opposite the robot (outside the box now) | the arm hides more than half the T in 31 % of frames, Push-T 98.5 → 69.2 % (C6); webcam latency unknown |
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
Level 2 92.4 → 92.4 %, Level 3 82.5 → 80.9 % (`--set scene.lightbox=true`). With every hardware error above at once,
Level 2 falls to 78.8 % and Level 3 to 50.4 %; the clutter, box and dock skills retrained in the lightbox (below) hold
them at 89.6 % and 74.5 %.

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
   table: 98.5 % in the lightbox, 97.7 % with every error above at once. The clutter, box and dock skills retrained
   the same way take Level 3 with every error from 50.4 % to 74.5 % (Level 2: 78.8 → 89.6 %).
6. **Move the camera to the back wall** (C6): from the as-built left-wall slot the arm hides more than half the T in
   31 % of frames, for seconds at a time, and Push-T falls to 69.2 % (clutter to 63.4 %). A fine-tune with the
   occlusion modelled recovers Push-T to 95.8 %. The back wall, 25 cm up, costs nothing.

## The lightbox Push-T policy (`runs/tee_lightbox_v1`)

`tee_v1` fine-tuned for 30 minutes (best checkpoint after 14 minutes, 54 M samples) in the lightbox, with joint
calibration error up to 1° per joint, backlash ±0.5° on every joint, mat friction 0.2–0.9 and camera latency 0–150 ms:

```powershell
python -m s2r2s.train --run tee_lightbox_v1 --init runs\tee_v1\best.pt --no-curriculum --minutes 30 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3
```

It scores 98.5 % in the lightbox and 97.7 % with all of those errors at once (the flat-table policy: 71.3 %), and it
is better on the flat table too (98.9 % against 96.6 %): the extra randomisation cost nothing. Through the hardware
path on the simulated arm (`hardware.runner --robot sim`, 20 episodes) it solves 20 of 20, median 1.7 mm / 1.5°.
Its weak spots are a rod pressed into the mat (93.2 % at 1 mm into it, 69.4 % at 4.8 mm) and latency above 200 ms (89.9 %).

## The lightbox skills for Levels 2 and 3

The clutter skill (push the T, keep the box in place), the box skill (push the box, keep the T in place) and the dock
skill were fine-tuned like `tee_lightbox_v1`: from the flat-table skill, with the same lightbox and hardware errors,
each with its source run's task flags and starting action noise. Best checkpoints after 28.5, 5.7 and 36.7 minutes
(74, 15 and 84 M samples):

```powershell
python -m s2r2s.train --run clutter_lightbox_v1 --clutter box --init runs\clutter_v7\best.pt --no-curriculum --minutes 35 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3
python -m s2r2s.train --run clutter_box_lightbox_v1 --objects box --clutter tee --init runs\clutter_box_v2\best.pt --init-std 0.1 --no-curriculum --minutes 40 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3 task.tool_near_block=0.3
python -m s2r2s.train --run dock_lightbox_v1 --objects box --clutter tee --dock --episode-seconds 10 --init runs\dock_v4\best.pt --init-std 0.12 --no-curriculum --minutes 40 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3 task.dock_start_bank=runs/dock_v4/start_bank.npz task.dock_bank_share=0.5
```

`python -m s2r2s.lightbox`, flat-table skill → lightbox skill (1,000 held-out scenes per row; every row in
`runs/<run>/lightbox_best.json`):

| condition (lightbox unless stated) | clutter: `clutter_v7` → `clutter_lightbox_v1` | box: `clutter_box_v2` → `clutter_box_lightbox_v1` | dock: `dock_v4` → `dock_lightbox_v1` |
|---|---|---|---|
| training scene (flat table, robot standing on it) | 94.9 → 95.3 % | 96.5 → 96.3 % | 95.1 → 93.7 % |
| lightbox, measured geometry | 93.5 → **95.3 %** | 95.8 → **96.7 %** | 93.4 → **92.5 %** |
| lightbox, controller not told about the mat | 35.6 → 64.9 % | 55.0 → 69.7 % | 41.8 → 69.0 % |
| mat friction 0.20 | 93.8 → 95.3 % | 94.5 → 96.1 % | 91.1 → 88.2 % |
| mat friction 0.90 | 88.8 → 92.0 % | 95.8 → 96.8 % | 94.1 → 93.5 % |
| camera latency 150–200 ms | 87.7 → 91.8 % | 94.2 → 95.8 % | 86.8 → 89.9 % |
| camera latency 200–300 ms | 61.1 → 84.9 % | 91.3 → 92.9 % | 70.9 → 76.7 % |
| joint calibration error up to 1° per joint | 89.8 → 92.8 % | 88.9 → 93.3 % | 77.1 → 89.9 % |
| ... up to 2° | 72.0 → 85.1 % | 70.2 → 79.1 % | 46.6 → 71.5 % |
| backlash ±0.5°, every joint | 90.4 → 95.4 % | 93.1 → 96.5 % | 81.3 → 94.1 % |
| backlash ±1°, every joint | 72.5 → 93.5 % | 80.5 → 92.5 % | 49.0 → 86.7 % |
| **all at once: calibration ≤1°, backlash ±0.5°, μ 0.2–0.9, latency 0–150 ms** | 80.7 → **91.0 %** | 87.1 → **94.2 %** | 63.1 → **88.3 %** |
| arm occlusion, C920 in the left-wall slot (as built) | 63.4 → 67.0 % | 74.6 → 76.2 % | 67.3 → 64.1 % |
| arm occlusion, C920 on the back wall, 25 cm up | 92.5 → 95.6 % | 96.2 → 96.5 % | 93.4 → 93.7 % |

Levels 2 and 3 with these skills, 1,000 held-out scenes each, the same worlds for both sets of skills:

| world (lightbox) | Level 2, flat-table skills | Level 2, lightbox skills | Level 3, flat-table skills | Level 3, lightbox skills |
|---|---|---|---|---|
| measured geometry | 92.4 % | **93.7 %** | 80.9 % | **87.2 %** |
| + joint calibration error ≤1°, backlash ±0.5° | 83.1 % | **89.9 %** | 53.4 % | **77.3 %** |
| + mat friction 0.2–0.9, camera latency 0–150 ms (every error) | 78.8 % | **89.6 %** | 50.4 % | **74.5 %** |

What this says:
1. **The retrained skills gain most where the hardware errors are.** With every error at once the clutter skill goes
   from 80.7 to 91.0 %, the box skill from 87.1 to 94.2 % and the dock skill from 63.1 to 88.3 %. Without errors
   the gains are small (clutter +1.8, box +0.9) and the dock skill is level (−0.9, within the ±0.8-point noise of
   1,000 scenes).
2. **Level 3 is the weak link once the joints are imperfect.** With calibration error and backlash, the flat-table
   skills assemble 53.4 % (the box docked in 63.9 %, a block that had to stay moved >10 mm in 16.7 %); the lightbox
   skills 77.3 % (docked 86.3 %, moved 9.5 %). The dock skill is most of the gain: the lightbox T and box skills
   with `dock_v4` reach only 61.1 % (docked 69.7 %). Docking places the box to 3 mm / 5°, and backlash ±0.5° alone
   takes `dock_v4` to 81.3 %. With every error, Level 3 still loses 12.7 points to the error-free lightbox (87.2 →
   74.5 %): T at its goal 91.0 → 86.2 %, box docked 93.1 → 81.3 %, a kept block moved 3.1 → 9.4 %.
3. **The dock start bank recorded on the flat table is not the problem.** Docking alone with every error,
   `dock_lightbox_v1` docks 88.6 % of the docking starts that Level 3 produces in the lightbox with the lightbox skills
   and every error (979 starts on the test seed, `runs/dock_lightbox_v2/test_starts.npz`), 88.7 % of the flat-table
   Level 3 starts and 90.1 % of sampled starts (`dock_v4`: 63.5, 63.1 and 60.5 %). Fine-tuning 40 more minutes with
   half the episodes from 3,912 lightbox starts on training seeds 1–4 (`dock_lightbox_v2`) did not help: 86.1 % on the
   lightbox starts and 86.2 / 76.0 / 75.2 % in Level 3 (rows of the table above), so `dock_lightbox_v1` stays.

**Levels 2 and 3 take the world's randomisation from the T skill's checkpoint.** With `--tee
runs\clutter_lightbox_v1\best.pt`, `--set scene.lightbox=true` alone puts every hardware error into the world. The
rows above set the world explicitly (world 1: no errors, as the flat-table runs; world 2 adds the joint errors;
world 3 every error):

```powershell
python -m s2r2s.level2 --tee runs\clutter_lightbox_v1\best.pt --box runs\clutter_box_lightbox_v1\best.pt --set scene.lightbox=true rand.backlash_deg=0,0,0,0 rand.joint_offset_deg=0 rand.table_friction=0.2,0.5 rand.obs_latency_steps=0,2
python -m s2r2s.level3 --tee runs\clutter_lightbox_v1\best.pt --box runs\clutter_box_lightbox_v1\best.pt --dock runs\dock_lightbox_v1\best.pt --set scene.lightbox=true rand.backlash_deg=1,1,1,1 rand.joint_offset_deg=1.0 rand.table_friction=0.2,0.5 rand.obs_latency_steps=0,2
python -m s2r2s.level3 --tee runs\clutter_lightbox_v1\best.pt --box runs\clutter_box_lightbox_v1\best.pt --dock runs\dock_lightbox_v1\best.pt --set scene.lightbox=true rand.backlash_deg=1,1,1,1 rand.joint_offset_deg=1.0 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3
python -m s2r2s.level3 --tee runs\clutter_lightbox_v1\best.pt --box runs\clutter_box_lightbox_v1\best.pt --dock runs\dock_lightbox_v1\best.pt --record-dock-starts runs\dock_lightbox_v2\test_starts.npz --record-seeds 20000003 --set scene.lightbox=true rand.backlash_deg=1,1,1,1 rand.joint_offset_deg=1.0 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3
python -m s2r2s.evaluate runs\dock_lightbox_v1\best.pt --set task.dock_start_bank=runs/dock_lightbox_v2/test_starts.npz task.dock_bank_share=1   # docking alone, every error (the checkpoint's own)
```

Every command and its output are in `runs/level2/lightbox_evals.log` and `runs/level3/lightbox_evals.log`, the
summaries in `runs/level2/lightbox_*eval.json` and `runs/level3/lightbox_*eval.json`.

## Smooth motion: ramped servo targets

In the videos the arm shook. Measured at every 5 ms physics step, the rod tip moved in a 20 Hz sawtooth: up to
360 mm/s at the 99th percentile (640 mm/s at most) although the policy never asks for more than 80 mm/s, and about
1 mm of wobble around its path. The policy was not the cause: a perfect camera, no randomisation and the scripted
pusher all showed it. The controller sent each joint a new position target every 50 ms, and the simulated servos
reached each one within a few milliseconds and overshot, so the arm lurched 20 times a second. On the real arm the
same stepped targets would hit the gearboxes the same way, and the camera noise the policy reacts to adds to it.

The fix is in the controller, shared with the hardware: the joint targets now move from the previous command to the
new one in equal steps over the control period, `task.servo_ramp`, on for every run trained from now on (checkpoints
saved before it load with it off, as they were trained). The hardware runner sends the same ramp at about 100 Hz
(HARDWARE.md §5). The simulated servo command delay (0–25 ms, drawn per episode) shifts the stream of ramped targets
later in time: during the delay the servos finish the previous ramp, then start the new one at the same even pace.
(The first version of the ramp held the old target during the delay and then rushed through the whole ramp in the
time left, so with that delay the arm still pulsed 20 times a second: p99 rod speed 266 mm/s with every hardware
error. Corrected on 20 Oct; the numbers below are with the corrected model.)

The four lightbox skills were fine-tuned for 20 minutes each from their earlier versions, with the ramp on and the
same hardware errors as before: `tee_lightbox_v2`, `clutter_lightbox_v2`, `clutter_box_lightbox_v2` and
`dock_lightbox_v3` (`dock_lightbox_v2` is the earlier start-bank experiment). They were trained under the first ramp
model and were not retrained after the correction: their success is unchanged within the noise of 1,000 scenes.
*Wobble* is the RMS distance of the rod tip from the straight line between where it was at the start and at the end
of each 50 ms period, *speed* the 99th percentile of its speed over every 5 ms physics step
(`python -m s2r2s.lightbox <checkpoint> --smoothness`); both in the lightbox row. 1,000 held-out scenes per cell:

| skill, stepped → ramped | lightbox | every hardware error | wobble | rod speed, p99 |
|---|---|---|---|---|
| Push-T, `tee_lightbox_v1` → `tee_lightbox_v2` | 98.5 → **99.0 %** | 97.7 → **98.1 %** | 1.08 → **0.11 mm** | 331 → **114 mm/s** |
| clutter, `clutter_lightbox_v1` → `clutter_lightbox_v2` | 95.3 → **95.0 %** | 91.0 → **94.3 %** | 1.09 → **0.13 mm** | 338 → **114 mm/s** |
| box, `clutter_box_lightbox_v1` → `clutter_box_lightbox_v2` | 96.7 → **97.3 %** | 94.2 → **96.2 %** | 1.02 → **0.13 mm** | 322 → **114 mm/s** |
| dock, `dock_lightbox_v1` → `dock_lightbox_v3` | 92.5 → **97.0 %** | 88.3 → **92.3 %** | 0.96 → **0.16 mm** | 323 → **115 mm/s** |

Levels 2 and 3, stepped skills → ramped skills, in the same three lightbox worlds as above
(`runs\level*\lightbox_skills_v2*_eval.json`):

| world | Level 2 | Level 3 |
|---|---|---|
| measured geometry, no hardware errors | 93.7 → **95.5 %** | 87.2 → **90.0 %** |
| + joint calibration error ≤ 1°, backlash ±0.5° | 89.9 → **92.1 %** | 77.3 → **83.7 %** |
| + mat friction 0.2–0.9, camera latency 0–150 ms (every error) | 89.6 → **91.1 %** | 74.5 → **82.6 %** |

- **The shaking is gone in simulation.** Wobble drops about eight-fold (≈1.0 → 0.11–0.16 mm) and the rod's p99
  speed from about 330 to 115 mm/s, with every hardware error on. Traced at 5 ms over 64 Push-T scenes with every
  error, the rod never exceeds twice the commanded maximum (160 mm/s); before, it did in 26 % of steps.
  The rod still runs a little above 80 mm/s when it catches up with a command that ran ahead (up to 15 mm, the
  anti-windup limit).
- **Success holds or improves.** With every hardware error, clutter goes from 91.0 to 94.3 % and docking from 88.3 to
  92.3 %; Level 3 from 74.5 to 82.6 %. Clutter in the error-free lightbox moves 95.3 → 95.0 %, within the ±0.7-point
  noise of 1,000 scenes.
- **Ramping alone is safe for the stepped policies.** Without retraining, `tee_lightbox_v1` with ramped targets scores
  98.8 % in the lightbox and 98.1 % with every error (98.5 and 97.7 % stepped; measured under the first ramp model).
- **Hardware path:** `tee_lightbox_v2` through the runner on the simulated arm, ramped at 100 Hz: 20 of 20,
  median 1.2 mm / 1.2°. The runner's ramp is unaffected by the correction (the real latency shifts it by itself).
- **On the real arm**, watch the tracking test (B5) and the first trials: if the arm still buzzes or lurches, the
  simulator's servo model needs a measured step response.
- Retrain: `python -m s2r2s.train --run tee_lightbox_v2 --init runs\tee_lightbox_v1\best.pt --init-std 0.1
  --no-curriculum --minutes 20 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1
  rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3 task.servo_ramp=true`, and the same for the others with
  their task flags (each run's `config.json`).

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
4. *Backlash:* `python -m s2r2s.hardware.backlash --robot so101 --port COM5 --map calibration\joint_map.json --lightbox`
   (HARDWARE.md §3). At reaches of 15, 20 and 25 cm each joint comes to the same target from both sides, three
   times. The encoders give the play they see. The rod's actual motion gives the play they miss: from the C920 and
   a marker on the rod once C1–C3 are done (re-run it then), or from pencil marks you measure (`--probe manual`)
   before that. Pass: every joint's play, encoders and actual, within the 1° (±0.5°) that `tee_lightbox_v1` and the
   lightbox skills were trained with. Then run the `python -m s2r2s.lightbox ... --set rand.backlash_deg=...` it
   prints to see what the measured play costs. In rehearsal (simulated arm, known play) it finds the dead band to
   0.03° and the total to 0.04°. ±0.5° ≈ ±1.7 mm at 20 cm for the pan joint.
5. *Tracking:* `python -m s2r2s.hardware.tracking --robot so101 --port COM5 --map calibration\joint_map.json --lightbox`.
   Pass: lag ≤ 100 ms, joint error after the lag ≤ 1°. Watch the arm too: during policy runs the runner ramps its
   targets at 100 Hz (*Smooth motion* above). If the real servos still buzz or lurch, the simulator's servo model is
   off and needs a step-response measurement before more training.

**C. The camera**
All of it: `python -m s2r2s.hardware.camera_calibrate <step>` (HARDWARE.md §4); files in `calibration/`.
1. `settings --sweep-focus`: 1920 × 1080 MJPG at 30 fps; autofocus, auto-exposure and auto white balance off (fixed
   values, saved). Check the saved frame: sharp, white walls not clipped.
2. `intrinsics`: 25 views of a ChArUco board. Pass: RMS ≤ 0.5 px; vertical field of view ≈ 43.3° (the scene's value).
3. `extrinsics`: eight ArUco markers in the mat's corners and against its edges (`boards` prints them), or their
   centres touched with the rod tip (`--touch`); `solvePnP`. Pass: every marker, solved without it, within 1 mm on
   the mat. In simulation (rendered markers, the left-wall view): 0.4 mm, camera within 0.5 mm.
4. Pose estimator (`CameraPoseSource` in `hardware/poses.py`, HARDWARE.md §4): `colours` learns the T's and the box's
   colours, `check --expect X Y YAW` compares with a known pose. Pass: ≤ 2 mm / 2° on a printed template at ten poses
   across the band (2 mm / 2° noise costs the flat-table Push-T policy about 2 points). On simulated C920 images
   (`python -m s2r2s.hardware.camera_sim`, `clutter_v7` pushing in the lightbox, the arm masked from joint angles
   0.5° off and 100 ms old), tracking at 20 Hz from the left-wall slot, the error is p95 0.41 mm / 0.35° for the T
   and 0.45 mm / 0.63° for the box. The T comes back in 87 % of frames, down to a quarter of it visible. Of 11,057
   accepted estimates over four runs (true or simulated calibration, left or back wall, with and without tracking),
   none is off by more than 2 mm, and three by more than 2° (at most 2.3°). About 33 ms per frame on this laptop.
5. Latency: `python -m s2r2s.hardware.latency --robot so101 --port COM5 --map calibration\joint_map.json --lightbox`
   (HARDWARE.md §4). The pan joint swings the rod, with the rod marker, at up to 71 mm/s. The marker's track in the
   camera, against the encoders, gives the camera's latency (exposure to arrival). The pose estimator's time per
   frame is added to it. Pass: end-to-end ≤ 150 ms (100–150 ms: −1 point for `tee_v1`; 150–200 ms: −5; 200–300 ms:
   −18). Then run the `--set rand.obs_latency_steps=...` row it prints. In rehearsal (rendered frames stamped 120 ms
   and 70 ms late) it finds 120.0 and 70.1 ms. Keep the capture buffer at one frame.
6. Occlusion: `python -m s2r2s.camera_study --checkpoint runs\tee_lightbox_v1\best.pt --lightbox` casts rays from mounts inside the
   box to the T's top face (12,800 views each). Mean share hidden by the arm / share of views more than half hidden:
   back wall, centre, 25 cm above the mat **9.6 % / 1.9 %**; right-wall slot 17.2 % / 4.8 %; back-left corner 18.8 % / 6.8 %;
   back wall, 45 cm up 22.8 % / 15.5 %; **left-wall slot as built 34.2 % / 29.3 %**; overhead 62.0 % / 68.9 %. The arm's links
   sit to one side of its base and pass between the left-wall camera and the T.

   The camera model now includes this (`rand.occlusion_camera`, `occlusion.py`). Each step it casts rays from the mount
   to 18 points on the T's top face. Frames with less than `rand.occlusion_threshold` of the face visible (default 0.5)
   are dropped, and the actor keeps its last estimate. `rand.occlusion_bias=1` moves the estimate to the centroid of the
   visible part. Rows 23–33 of `python -m s2r2s.lightbox`, 1,000 held-out lightbox scenes each:

   | condition | `tee_lightbox_v1` | `tee_occlusion_v1` | `clutter_v7` |
   |---|---|---|---|
   | lightbox, 3 % of frames dropped at random (baseline) | 98.5 % | 97.8 % | 93.5 % |
   | 30 % dropped at random | 98.6 % | 98.8 % | 93.2 % |
   | **left-wall slot (as built)**: 31 % of frames hidden, longest gap p90 10.6 s | **69.2 %** | **95.8 %** (21.5 % hidden) | **63.4 %** |
   | right-wall slot: 2.9 % hidden | 98.7 % | 98.2 % | 92.9 % |
   | **back wall, 25 cm up**: 1.7 % hidden | **99.0 %** | 97.8 % | **92.5 %** |
   | left wall, estimator returns a pose down to 25 % visible | 96.2 % | 97.3 % | 90.2 % |
   | left wall, estimator needs 75 % visible | 28.1 % | 75.5 % | 25.5 % |
   | left wall, estimate = centroid of the visible part | 61.1 % | 78.2 % | 32.6 % |
   | back wall, needs 75 % visible / centroid of the visible part | 93.0 / 93.4 % | 96.2 / 88.7 % | 85.1 / 77.0 % |
   | right wall, needs 75 % visible / centroid of the visible part | 86.9 / 93.2 % | 87.1 / 94.2 % | 75.0 / 78.3 % |

   The policies tolerate dropped frames when the drops are scattered (30 % at random costs nothing), but not long gaps.
   With the left-wall camera, success falls with the longest blind gap in an episode: 92 % below 0.5 s, 76 % at 1–2 s,
   42 % above 5 s.

   `tee_occlusion_v1` is `tee_lightbox_v1` fine-tuned for 35 minutes (62 M samples; occlusion halves the training
   speed) with the left-wall occlusion on, plus the same hardware errors:

   ```powershell
   python -m s2r2s.train --run tee_occlusion_v1 --init runs\tee_lightbox_v1\best.pt --no-curriculum --minutes 35 --set scene.lightbox=true rand.joint_offset_deg=1.0 rand.backlash_deg=1,1,1,1 rand.table_friction=0.2,0.9 rand.obs_latency_steps=0,3 rand.occlusion_camera=c920
   ```

   It recovers most of the left wall's loss (69.2 → 95.8 %) by keeping the T in view more often (21.5 % of frames
   hidden instead of 31 %). It gives back about a point everywhere else: lightbox 97.8 %, all hardware errors at once
   96.5 % (against 97.7 %), flat table 96.7 % (against 98.9 %). On the left wall it still depends on the estimator
   (75.5 % if the estimator needs 75 % of the T visible).

   **Recommendation: move the C920 to the middle of the back wall, 25 cm above the mat, and keep `tee_lightbox_v1`.**
   That mount hides the least (1.7 % of frames), costs nothing (99.0 %), and is the most forgiving of a strict
   estimator. If the back wall cannot take it, use the mirror of the existing slot in the right wall. If the camera
   must stay in the left-wall slot, run `tee_occlusion_v1` there (95.8 %, about 3 points below the back wall) and
   make sure the estimator returns poses from partial views. Whatever the mount, the pose estimator
   should fit the T template to partial views rather than take the centroid of the visible pixels, and report a
   frame as missing only when little of the T is visible. `CameraPoseSource` does both: it masks the arm out using
   the joint angles and returns poses down to a quarter of the T visible (C4, HARDWARE.md §4). The calibration and the
   estimator work from any mount (`camera_sim --mount back` tests the recommended one).

   Cost: with the occlusion model on, a 1,000-scene evaluation takes 16 s instead of 6.6 s, and a 4,096-world step
   takes 145–150 ms instead of 62 ms. `rand.occlusion_fast=true` rays the collision boxes instead of the meshes. It is
   6× cheaper per ray but only about 18 % faster per step (125 ms), because the per-world Python loop
   (about 10 µs per world, holding the GIL) dominates. The boxes also hide a little less: 73.8 % instead of 69.2 %
   at the left wall.

**D. Pushing without the policy**
1. Open-loop replay: a straight push through the T's centre and an off-centre push, five times each, in simulation
   (`SimRobot`, lightbox) and on the table. Compare the final T poses; adjust the friction range if they disagree.
2. Repeatability: the same scene three times, same homing move (the supervisor's backlash question: spread within a
   scene vs between scenes).

**E. Policy trials**
1. Rehearse on the simulated arm: `python -m s2r2s.hardware.runner --checkpoint runs\tee_lightbox_v2\best.pt --robot sim --episodes 20`,
   then through the camera path: `... --camera render --episodes 8` (the C920 rendered from its mount, 100 s per
   episode). When the camera loses the T for 2 s (`--max-pose-age`), the runner lifts the rod out of the arm's
   shadow, lowers it once the T is back in view and hands back to the policy (HARDWARE.md §5). A start where the arm
   hides the T begins from a clear point nearby. Rendered from the left-wall slot (8 held-out scenes),
   `tee_lightbox_v1` solves all 8 with the 2 s default and 6 with `--max-pose-age 0.5`, which steps in too
   often. The runner before this change started only 5 of them and solved 4.
2. Real: `python -m s2r2s.hardware.runner --checkpoint runs\tee_lightbox_v2\best.pt --robot so101 --port COM5 --map calibration\joint_map.json --max-step-deg 2`,
   20–30 logged trials from held-out scenes. Compare with `python -m s2r2s.lightbox` re-run at the *measured*
   friction, latency, noise and backlash: that is the simulator's prediction for this table.
3. Then clutter (`--checkpoint runs\clutter_lightbox_v2\best.pt`), the gate (build the wall), and Levels 2 and 3
   with the lightbox skills `clutter_lightbox_v2`, `clutter_box_lightbox_v2` and `dock_lightbox_v3` (ramped targets):
   `python -m s2r2s.hardware.runner --level 2 --robot so101 --port COM5 --map calibration\joint_map.json --goal X Y YAW --box-goal X Y YAW --max-step-deg 2`,
   and `--level 3 ... --goal X Y YAW` (the box then nests in the T's corner). The runner plans the pushes from the
   camera's first view, runs each with its skill through that skill's twin and starts the next once the camera
   sees the pushed block settled (HARDWARE.md §5, "Levels 2 and 3"). Rehearse first: `--level 2 --robot sim
   --episodes 200`, then `--camera render --episodes 8`. On the simulated arm, 1,000 held-out scenes each, the
   hardware path matches the simulator in the same world: Level 2 95.3 % against 95.6 %, Level 3 91.0 % against
   91.0 %. Through the rendered C920 in the left-wall slot, 8 scenes per level, 6 and 5 succeed (simulator: 7 and
   7). Every trial started and every lost block was recovered. But the arm hides the block that must stay (the box
   in up to 94 % of a T push), and one miss was the T pushed into a box the camera could not see: move the camera
   (C6) first. Compare real trials with `python -m s2r2s.level2` / `level3 --set ...` at the *measured* errors;
   with every error at once they are at 91.5 / 81.9 % (*Smooth motion* above).

Policies trained on the flat table run in the lightbox only with `--lightbox` on the hardware tools (the kinematics
must know the mat height); policies trained in the lightbox carry it in their config.

## Still open

- The gate skill is still a flat-table policy (the clutter, box and dock skills have lightbox versions).
- Level 3 with every hardware error is at 74.5 %: docking (81.3 %) and a kept block moved by the rod (9.4 %) are the
  losses. Measure the real backlash and calibration residual (B2, B4) before training against them further.
- Backlash here is a dead band between servo command and link, the same on every joint, with the encoder reading
  the link (the STS3215's encoder sits on its output shaft), so the policy can see the play and correct it. Gravity
  holds the lift, elbow and wrist against one side of their play, so this overstates them; the pan joint carries no
  such load. Play the encoders cannot see (a loose horn, a printed coupling) acts like the calibration-error rows.
  `hardware.backlash` (B4) measures both per joint, at three reaches, under the arm's own gravity load, and prints
  the per-joint `rand.backlash_deg` and the `rand.joint_offset_deg` they amount to.
- Arm occlusion is modelled (C6): the left-wall slot costs the current policies 29–30 points. A 35-minute fine-tune
  (`tee_occlusion_v1`) recovers most of it for Push-T (95.8 %). The clutter, box and dock skills have no such
  fine-tune: move the camera, or fine-tune them the same way. The actor is not told that its estimate is stale.
  A "frames since the last fresh pose" input (the runner knows it) might close the rest of the gap.
- Levels 2 and 3 run through the hardware runner (E3) but are rehearsed on the simulated arm only. With the
  left-wall camera the block that must stay is often hidden, and the rod guard then works from its last pose. The
  rendered rehearsal covers only that mount (the runner's rendered camera has no mount option yet). Blocks placed
  within about a millimetre of the 15 mm rule can be refused: camera noise decides.
- The camera pose estimator, the backlash tool's camera probe and the latency tool are tested on rendered images
  only: C4 and C5 on the real camera are next. The latency tool cannot see delay in the joint readings themselves
  (LeRobot's read is stamped halfway through).
- The runner no longer deadlocks when the arm hides the T: it backs the rod out of the arm's shadow, waits for the
  camera and hands back (HARDWARE.md §5), and trials whose start hides the T begin from a clear point. After it
  hands back, though, the policy can push from the same place and lose the T again. In one rendered scene,
  `tee_lightbox_v1` lost it 7 times when the runner stepped in after 0.5 s. That is why the default waits 2 s
  (`--max-pose-age`): the policy closes most short losses itself. The actor is still not told that it acts on a
  stale pose. Moving the camera (C6) is the real fix.
- The arm has no collision geometry except the rod. The walls are out of reach, but a wrist camera mount (if the
  real arm carries one) is not checked against the forearm.
