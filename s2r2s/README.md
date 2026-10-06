# s2r2s: Failure-Guided Sim2Real2Sim on the SO-101

Milestone 1 (simulation): **SO-101 Push-T, learned from scratch with reinforcement learning in MuJoCo.**
The arm pushes a half-scale Push-T block (the Diffusion Policy benchmark object) from a random pose to a
random goal pose: translation up to 12 cm and **any rotation up to 180°**. There are no demonstrations.
Training runs under domain randomization and a camera sensor model, so the policy can be
carried to the real arm and camera in Milestone 2.

This is the proposal's own task: planar pushing to a target SE(2) pose with a state-based policy. The
same code already supports the proposal's six-object set (see *Objects*). The next steps are in [ROADMAP.md](ROADMAP.md).

> Legacy work (the *Push-Turn-Park* gate puzzle in `../src/so101_m1`) is untouched. This project is
> self-contained and runs natively on Windows (no WSL).

## Results (Milestone 1)

Training took 45 minutes on this laptop: 171M samples at about 68k samples/s, physics on the Core Ultra 9 and PPO on
the RTX 5080. The curriculum reached full difficulty after 21 minutes. The scores below are on **1,000 held-out scenes**,
drawn from a seed never used for training or checkpoint selection, at full goal difficulty, with domain randomisation and the camera
model on. The scripted baseline runs on the *same* scenes but with the *true* object pose.

| | RL policy (`runs/tee_v1/best.pt`) | scripted baseline |
|---|---|---|
| success (≤10 mm and ≤10° at the end of the episode) | **96.6 %** | 88.7 % |
| final position error, median / p90 | **1.4 / 3.9 mm** | 4.5 / 6.7 mm |
| final orientation error, median / p90 | **2.0 / 5.9°** | 3.9 / 9.2° |
| proposal metric E = e_pos/20 mm + e_yaw/10°, mean | **0.51** | 1.33 |
| time to reach tolerance, median | **6.0 s** | 8.6 s |
| failures (object escaped the reachable band) | 0.5 % | 0.1 % |
| action rate (smoothness, lower is smoother) | **0.027** | 0.125 |

- **Large rotations:** success by required rotation, policy vs baseline: 0–60° 97 % vs 95 %, 60–120° 97 % vs 99 %,
  120–150° 95 % vs 84 %, **150–180° 96 % vs 59 %**. RL found rotation strategies the heuristic lacks.
- **Robustness:** with 2 ms physics instead of the 5 ms used in training, success is 97.1 %, so the policy does not exploit the integrator.
  With nominal physics and a perfect camera it is 94.5 % (same scenes). Likely cause: a deterministic policy can
  stall when nothing perturbs it. Real sensors always add noise, but check this if it ever matters.
- **Camera requirements** (`python -m s2r2s.sensitivity`, 512 scenes per row; trained at 1 mm / 1°, 0–100 ms, 3 % dropout):

  | pose noise (1σ) | success | latency | success | dropped frames | success |
  |---|---|---|---|---|---|
  | 0 mm / 0° | 96.1 % | 0 ms | 97.5 % | 0 % | 98.6 % |
  | 1 mm / 1° | 98.0 % | 0–100 ms | 98.0 % | 10 % | 95.9 % |
  | 2 mm / 2° | 96.1 % | 100–200 ms | 94.9 % | 25 % | 96.1 % |
  | 4 mm / 4° | 76.0 % | 200–300 ms | 81.1 % | 50 % | 94.3 % |
  | 8 mm / 8° | 22.7 % | 300–400 ms | 50.8 % | | |

  Target for the real pipeline: **≤2 mm / 2° noise and ≤150 ms latency**. Brief occlusions are harmless.
- **Videos** (`videos/`): `tee_policy_grid.mp4` (four random scenes), `tee_policy_kicks.mp4` (the object is knocked away
  every 5 s and the policy recovers), `tee_d435i_view.mp4` (the view from the planned real camera).

## Quick start (Windows, PowerShell)

```powershell
cd D:\Sim2Real2Sim\s2r2s
.\scripts\setup.ps1                                   # once: fetches the SO-101 model, .venv + MuJoCo 3.11 + PyTorch cu128, runs the tests
.\scripts\play.ps1 -Scripted                          # watch the scripted baseline in the MuJoCo viewer
.\scripts\play.ps1 runs\tee_v1\best.pt                # watch the trained policy (keys: P pause, N next, K kick the object)
.\scripts\train.ps1 -Run tee_v2 -Minutes 90           # train a new policy (about 70k samples/s on this laptop)
.\scripts\tensorboard.ps1                             # live curves at http://localhost:6006
.\scripts\evaluate.ps1 runs\tee_v2\best.pt            # 1000 held-out scenes, side by side with the scripted baseline
.\scripts\record.ps1 runs\tee_v2\best.pt videos\tee.mp4   # 2x2 grid video
```

Everything is also available as `python -m s2r2s.<module> --help` (`train`, `play`, `evaluate`, `record`, `sensitivity`, `camera_study`).
Resume training with `python -m s2r2s.train --run tee_v2b --resume runs\tee_v2\latest.pt --minutes 60`.
Multi-object: `.\scripts\train.ps1 -Run six_v1 -Objects tee,ell,box,rect,disk,plus -Minutes 180`.

**Laptop graphics note.** On hybrid-graphics laptops, Windows runs MuJoCo's OpenGL viewer and renderer on the
*integrated* GPU by default. Here, that measured 65 ms per frame on "Intel(R) Graphics", so the viewer is choppy.
To use the RTX GPU: Settings → System → Display → Graphics → add
`%APPDATA%\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe` (the interpreter behind `.venv`) and
`.venv\Scripts\python.exe` → *High performance*. Training is unaffected because PyTorch already uses CUDA.

## The task

| | |
|---|---|
| Robot | MuJoCo Menagerie SO-101 (kinematics, inertias and STS3215 servo models unchanged) |
| Tool | 12 mm rod clamped in the closed gripper, tip 5 mm above the table, kept vertical |
| Object | Push-T at half scale: bar 60x15 mm, stem 15x45 mm, 20 mm tall, 30 g |
| Workspace | Vertical-tool reach is an annulus, r = 0.12–0.275 m, azimuth ±55° (measured by IK). Object start and goal centres are kept in r = 0.175–0.225 m, ±30°, so the rod can always get behind the object |
| Goal | Random pose. Curriculum from easy (≤3 cm, ≤30°) to full (≤12 cm, ≤180°) |
| Episode | 20 s at 20 Hz, fixed length. The policy must reach the goal **and stay there**, matching the proposal's final-pose metric |
| Success | Final position error ≤10 mm and orientation error ≤10° (symmetry-aware). The proposal's E = e_pos/20 mm + e_yaw/10° is reported too |

## Design, and why

**Physics.** Batched CPU MuJoCo through `mujoco.rollout`: one small model copy per world, stepped in C on all 24
threads at 5 ms. The previous attempt lost days to GPU-physics jitter and NaNs; this setup avoids that class of problem.
Measured on identical scripted off-centre pushes against a 1 ms reference: the 5 ms result differs by a median
1.1 mm and 1.3°. A 0.2 mm change in the start pose at 1 ms already moves the result by 0.9 mm and 1.0°, so the
discretisation error is below pushing's own sensitivity. No tipping, jitter or penetration issues were seen.
The robot has no collision geometry except the rod, because the arm cannot reach object height.

**Controller (identical in simulation and on hardware).** action ∈ [-1,1]² → tool velocity (≤8 cm/s) → commanded
tool position (kept within 15 mm of the measured tip: anti-windup) → projected into the reachable annulus →
one warm-started damped-least-squares IK step (vertical tool) → joint position targets → servos.
`kinematics.py` reads the chain from the MuJoCo model, matches MuJoCo FK to 1e-15, and tracks to about 0.1 mm.

**Observations.** The *actor* gets only what the real system measures: tool position from joint encoders
(FK), commanded-position error, the **camera estimate** of the object pose, the goal, keypoints relative to the
tool and to the goal, the previous action and joint angles. The *critic* also gets the true state and the
randomised physical parameters (asymmetric actor-critic).

**Camera model.** The policy never sees the true object pose. It sees a camera estimate with
0–100 ms latency, 1 mm / 1° noise and 3 % dropped frames. In the viewer, the yellow outline is what the
policy sees and the solid block is the truth. On the real robot, the camera pipeline produces this same
(x, y, yaw); see ROADMAP.md.

**Reward.** A keypoint distance D (three keypoints on the T, symmetry-aware for other shapes) couples position
and orientation in one metric. reward = exp(-D/5 cm) + exp(-D/1 cm) + 0.5·(progress per cm)
+ 0.2·exp(-gap/2 cm) − action and action-rate costs, with −10 on failure (tipping, object escapes the
reachable band). Kernel terms are bounded, so idling next to the object never beats moving it.

**Domain randomisation (per episode).** Table friction 0.2–0.5, rod friction 0.15–0.5, object mass ×0.7–1.5,
centre of mass ±3 mm, servo gains ×0.8–1.2, joint damping ×0.8–1.2, tool-height calibration error ±2 mm,
servo command latency 0–25 ms, plus the camera model above.

**Training.** PPO (`ppo.py`, about 200 lines): 4096 worlds × 24 steps per update, GAE (γ 0.99, λ 0.95), observation
and value normalisation, time-outs bootstrapped from the true terminal state, KL-adaptive learning rate.
The automatic curriculum raises goal difficulty by 0.1 whenever 80 % of episodes at the current level succeed.
Every 50 updates, the policy is scored on 512 fixed held-out scenes at full difficulty, and `best.pt` is kept.

## What was verified before any RL (`pytest`, 12 tests)

- Stripped training model has exactly the Menagerie masses, inertias and servo gains
- FK and Jacobian match MuJoCo; IK solves >99.9 % of the workspace; one IK step per control tick tracks to <0.3 mm (p99)
- Physics state layout, reset validity (object, goal and tool placement), zero-action stillness, stability under random actions,
  and scenes that depend only on the seed (so evaluations are paired across settings)
- Symmetry-aware pose error; the reward prefers progress over idling
- **Scripted keypoint pusher** (`scripted.py`) solves easy goals (>90 %). At full difficulty it solves 86–100 % across all six
  objects, so every task variant is physically feasible before learning starts

## Objects (`objects.py`)

| name | shape | symmetry | scripted baseline, full difficulty (nominal physics, 192 scenes) |
|---|---|---|---|
| `tee` | T, bar 60x15 + stem 15x45 mm | none | 90 % |
| `ell` | L, 15x60 + 30x15 mm | none | 86 % |
| `box` | 40x40 mm | 90° | 98 % |
| `rect` | 60x25 mm | 180° | 96 % |
| `disk` | Ø44 mm | continuous | 100 % |
| `plus` | 50x50 mm cross, 15 mm arms | 90° | 100 % |

All are 20 mm tall and 30 g nominal. To change dimensions after measuring printed parts, edit `objects.py`.
To change the rod after building the tool, edit `SceneConfig` in `scene.py`.

## Files

| file | purpose |
|---|---|
| `s2r2s/scene.py` | MJCF construction (robot, rod, table, object, goal and estimate markers, cameras) |
| `s2r2s/objects.py` | Object library: footprints, keypoints, symmetry |
| `s2r2s/kinematics.py` | Batched FK/IK, shared by simulation and hardware |
| `s2r2s/env.py` | Batched environment: control, observations, camera model, reward, randomisation, resets |
| `s2r2s/ppo.py`, `train.py` | PPO and the training loop (logs, curriculum, evaluation, checkpoints) |
| `s2r2s/evaluation.py`, `evaluate.py` | Held-out evaluation (policy and baseline on identical scenes) |
| `s2r2s/scripted.py` | Heuristic pusher: feasibility check and baseline |
| `s2r2s/play.py`, `record.py`, `visual.py` | Viewer, videos |
| `s2r2s/sensitivity.py` | How good the camera pipeline must be: degrade the camera model, re-evaluate |
| `s2r2s/camera_study.py` | Arm-occlusion study for choosing the real camera mount |
| `tests/` | Regression tests |
