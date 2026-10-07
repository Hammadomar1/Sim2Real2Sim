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
| success (≤10 mm and ≤10° at the end of the episode) | **96.6 %** | 88.8 % |
| final position error, median / p90 | **1.4 / 3.9 mm** | 4.5 / 6.6 mm |
| final orientation error, median / p90 | **2.0 / 5.9°** | 3.8 / 10.3° |
| proposal metric E = e_pos/20 mm + e_yaw/10°, mean | **0.51** | 1.33 |
| time to reach tolerance, median | **6.0 s** | 8.6 s |
| failures (object escaped the reachable band) | 0.5 % | 0.0 % |
| action rate (smoothness, lower is smoother) | **0.027** | 0.124 |

- **Large rotations:** success by required rotation, policy vs baseline: 0–60° 97 % vs 96 %, 60–120° 97 % vs 99 %,
  120–150° 95 % vs 84 %, **150–180° 96 % vs 58 %**. RL found rotation strategies the heuristic lacks.
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

## Level 1: through the gate (`gate.py`)

A wall crosses the work band with an opening **64–70 mm** wide. The T starts on one side, and its goal (a random
pose) is on the other side. The wall reaches beyond the workspace, so the opening is the only way through. The T is 56–71 mm
wide depending on its orientation, so it only fits when turned so the bar runs along the wall (about ±25° of
slack) and centred to within a few millimetres. The policy must line it up, push it through straight, and then
place it.

- **Guidance, not a script.** Three waypoints: an aligned pose before the wall, the same pose past it, then the goal.
  The reward is the remaining path length *through the opening*, so pushing straight at the wall never pays.
  The observation shows the current waypoint in the same slots as the Milestone 1 goal, plus 17 gate features.
- **Warm start.** The Milestone 1 network is copied, and the new inputs get zero weights, so training starts from a
  policy that already follows waypoints. The curriculum narrows the opening from 80 mm to 64–70 mm.
- **Feasibility first.** A scripted controller with a dedicated "push straight through" mode gets about 40 % of T's
  through the narrowest openings, with no failures. Its jams are steering errors, not physics problems.

Held-out results: 1,000 unseen scenes at full difficulty, with randomisation and the camera model on
(`runs/gate_pilot/final_eval.json`):

| | trained (`runs/gate_pilot/best.pt`) | Milestone 1 policy, untrained on gates | scripted (true pose) |
|---|---|---|---|
| passed the gate | **98.5 %** | 46.5 % | 30.7 % |
| solved (≤10 mm, ≤10° at the goal) | **95.3 %** | 39.9 % | 12.0 % |
| final error, median | **1.4 mm / 2.1°** | 72 mm / 31° | 99 mm / 73° |
| failures (object left the band) | 1.9 % | 5.6 % | 0.0 % |
| time to the goal, median | 8.7 s | 13.4 s | 16.6 s |

Training time on this laptop: the best checkpoint came after **9 minutes** (29M samples) of a 30-minute run. Longer
training did not help: the final checkpoint scores 93.8 %. Success is flat across opening widths (94.8 % at 64–66 mm,
95.8 % at 68–70 mm). With 2 ms physics instead of 5 ms it scores 93.1 %. That small drop, which Milestone 1 did not
show, suggests wall contacts are somewhat sensitive to the physics step. Check this on hardware.

```powershell
python -m s2r2s.play --checkpoint runs\gate_pilot\best.pt          # watch it (or: --scripted --gate)
python -m s2r2s.evaluate runs\gate_pilot\best.pt --baseline         # held-out evaluation
python -m s2r2s.train --run gate_v2 --gate --init runs\tee_v1\best.pt --minutes 30   # retrain (warm start)
```

Videos: `videos/gate_policy_grid.mp4` (four scenes) and `videos/gate_policy_top.mp4` (top view).

## Clutter: place the T without disturbing another block (`clutter.py`)

A 40 × 40 mm block shares the work band. In about three quarters of the scenes it sits close beside the T's path:
15–30 mm from the area the T sweeps when moved straight (and turned) from its start to its goal. In the rest it is
anywhere in the band. It is never closer than 15 mm, the rod's 12 mm plus a 3 mm safety margin, so the rod can always
pass between the block and the T's path. **Success** means the T at its goal (≤10 mm, ≤10°) *and* the block within
10 mm / 10° of where it started. This is the proposal's *clutter* factor, and the "don't knock the other piece" skill
that the block-connecting puzzle needs.

- **The policy sees the block the way it sees the T**: through the camera model (latency, noise, dropped frames),
  plus three inputs computed from those estimates: T-to-block clearance, rod-to-block clearance, and the block's
  nearest point to the rod.
- **Rod guard, in the controller** (the same code runs on the real arm): the rod is never commanded closer than
  3 mm to the block's estimated outline; it slides along the block instead of shoving it.
- **Reward**: Milestone 1's, plus costs for moving the block (3 per cm, and 1 per step while it is out of place),
  for coming within 10 mm of it with the rod or the T (up to 0.5 per step), and for leaning on the guard (up to
  0.5 per step, in proportion to the part of a command the guard had to remove), so the policy plans around the
  block instead of pushing against the guard.
- **Training**: warm start from the Milestone 1 policy. Its inputs keep their weights; the 17 block inputs start at
  zero. `best.pt` came after 24 minutes (69M samples) of a 60-minute run at about 50k samples/s.

Held-out results, 1,000 unseen scenes, every agent on the same scenes, rod guard on
(`runs/clutter_v7/final_eval.json`):

| | trained (`runs/clutter_v7/best.pt`) | Milestone 1 policy, blind to the block | scripted (true pose, blind to the block) |
|---|---|---|---|
| solved (T at its goal and the block undisturbed) | **94.9 %** | 86.2 % | 85.3 % |
| T at its goal at the end, block or not | **96.7 %** | 89.6 % | 85.3 % |
| block disturbed (>10 mm or >10°) | 1.9 % | 3.9 % | 0.1 % |
| block touched (moved >2 mm) | 3.8 % | 7.4 % | 1.8 % |
| T final error, median | **1.3 mm / 1.9°** | 1.5 mm / 2.3° | 4.5 mm / 4.0° |
| failures (T left the band) | 0.7 % | 1.2 % | 0.0 % |
| time to the goal, median | **4.8 s** | 6.1 s | 8.8 s |

By the block's clearance from the T's straight path (solved, trained vs the blind Milestone 1 policy): 15–20 mm
(274 scenes) **92.0 %** vs 79.6 %, 20–25 mm **95.5 %** vs 82.7 %, 25–30 mm **93.2 %** vs 87.9 %, over 30 mm
**98.6 %** vs 94.6 %.

What each part contributes (solved, same 1,000 scenes):

| | rod guard off | rod guard on |
|---|---|---|
| Milestone 1 policy, blind to the block | 68.3 % | 86.2 % |
| trained with the block inputs, without the guard (`clutter_v5`) | 83.0 % | 92.2 % |
| trained with the guard, no cost for leaning on it (`clutter_v6`) | – | 92.8 % |
| **final** (`clutter_v7`) | 83.6 % | **94.9 %** |
| scripted, blind to the block | 74.4 % | 85.3 % |

`clutter_v5` and `clutter_v6` were trained under the earlier 12 mm gap rule and are not in git. Reproduce them with the
retrain command below plus `--set task.rod_guard=0 task.w_guard=0 task.clutter_min_gap=0.012` (v5) or
`--set task.w_guard=0 task.clutter_min_gap=0.012` (v6).

**Robustness** (trained policy): 2 ms physics instead of 5 ms, 95.0 %. Camera noise doubled to 2 mm / 2°, 94.1 %
(the block is disturbed more often, 3.7 %: the guard trusts the camera). Latency 100–150 ms, 93.4 %. Nominal
physics and a perfect camera, 93.3 %. Through the hardware path (simulated arm and camera, joint map, safety limits;
`hardware.runner --robot sim`, 200 episodes), 92.5 % with the block kept in place in 96.5 %. The final checkpoint of
the run scores 94.3 %.

How we got here, from 66.6 % (`clutter_v3`):
1. **Measure before tuning.** In four touches out of five, the rod, not the T, moved the block, and half of those
   came in the first second: the rod drove straight at the T through a block in the way. Proximity costs with 4 mm
   and then 10 mm margins did not teach it to go around (about 81 %).
2. **A guard in the controller fixed that at once.** Switching it on for a policy trained without it lifts it from
   83.0 % to 92.2 %. On its own, though, it let the rod get stuck leaning on the block in half of the remaining misses,
   hence the cost on the removed command, and a 15 mm gap rule so the guarded rod always fits.
3. **A geometry bug.** `objects.py` lost two corners of the T's outline to a floating-point sliver, so the first
   version's T-to-block clearance input was off by up to 12 mm. Clearances are now exact (separating axes and
   corners, checked against brute force to 0.004 mm) and 3× faster, which made long training affordable.
4. **What still fails** (`runs/clutter_v7/stuck_cases.png`): the rod ends up on the block's far side, away from the
   side of the T it has to push, and must detour around the block, sometimes near the edge of the reachable ring.
   That needs a path planner for the rod, or more training on exactly these scenes: what the failure-guided sampler
   (Milestone 3) is for.

```powershell
python -m s2r2s.play --checkpoint runs\clutter_v7\best.pt                       # watch it; the grey outline is the block's place
python -m s2r2s.evaluate runs\clutter_v7\best.pt --baseline --compare runs\tee_v1\best.pt   # held-out evaluation
python -m s2r2s.evaluate runs\clutter_v7\best.pt --set task.rod_guard=0          # the same without the rod guard
python -m s2r2s.train --run clutter_v8 --clutter box --init runs\tee_v1\best.pt --no-curriculum --minutes 60   # retrain
python -m s2r2s.hardware.runner --checkpoint runs\clutter_v7\best.pt --robot sim --episodes 20   # hardware-path rehearsal
```

Video: `videos/clutter_policy_grid.mp4` (four scenes). The status line shows how far the block has moved.

## Level 2: two blocks, each to its own goal (`level2.py`)

The T and the 40 mm box each start in the work band with their own goal pose (the Milestone 1 rules: up to 12 cm
away, any rotation), at least 15 mm apart at the start and at the goal. **Success** means both blocks at their goals
(≤10 mm, ≤10°) at the end.

Instead of one big new policy, Level 2 uses two learned skills and a small planner:
- **Skills.** Each skill is a clutter policy: push one block to a target while the other stays where it is. The T
  skill is the clutter policy above (`clutter_v7`, which keeps the box in place). The box skill is new
  (`clutter_box_v2`, which keeps the T in place). It is warm-started from a plain box pusher (`box_v1`: 98.9 % of
  held-out scenes after 20 minutes; scripted 98.6 %, but 4.0 mm vs 0.7 mm median error), the same way the T skill
  started from Milestone 1.
- **Planner.** It picks the order in which both pushes keep the skills' 15 mm rule (the one with more room). Two
  pushes in some order work for **83 %** of random scenes. Otherwise it looks for a spot to park one block first
  (park A, push B, then bring A back). That rescues only about 1 % more: in a 5 cm-deep band the two paths cross
  whichever block goes first, and no spot is within the skills' 12 cm reach. The remaining 16 % have no plan and
  are left out.
- **Execution.** One simulated world holds both blocks. Each skill sees it through its own copy of the
  environment, exactly as on the real arm: measured joints and camera estimates go in, joint targets come out. The
  rod guard protects whichever block must stay. The next push starts once the current block has stayed at its
  target for 1 s, judged by the camera.

Held-out results, 1,000 unseen two-block scenes, every configuration on the same scenes
(`runs/level2/final_eval.json`):

| | solved (both at their goals) | T at goal | box at goal | a block that should stay moved >10 mm | time, median |
|---|---|---|---|---|---|
| **planner + both skills** | **92.4 %** | 95.2 % | 95.7 % | 1.2 % | 11.7 s |
| always the T first | 82.0 % | 89.7 % | 88.1 % | 9.1 % | 12.0 s |
| always the box first | 87.8 % | 91.1 % | 95.4 % | 6.4 % | 11.3 s |
| planner, skills blind to the other block (`tee_v1`, `box_v1`) | 75.6 % | 87.4 % | 82.3 % | 6.7 % | 14.2 s |
| planner + skills, rod guard off | 62.2 % | 74.6 % | 77.0 % | 35.4 % | 11.8 s |

Final errors, median: T 1.7 mm / 2.3°, box 1.0 mm / 1.7°. Failures (a block left the band): 1.1 %.

The box skill on its own (1,000 held-out scenes, `runs/clutter_box_v2/final_eval.json`):

| solved | `clutter_box_v2` | `clutter_box_v1` | box pusher blind to the T | scripted |
|---|---|---|---|---|
| standard scenes | **98.6 %** | 97.5 % | 91.5 % | 92.6 % |
| rod starting 4–12 mm from the T | **93.6 %** | 87.6 % | 79.9 % | |
| standard scenes, rod guard off | 93.3 % | | 81.5 % | |

How we got here:
1. **Order matters.** Scenes where the T goes first were the weak spot. There, the box skill must work around a
   placed T, which is much bigger than the box. With the first box skill, those scenes succeeded 82.9 % of the time
   against 93.3 % for box-first scenes (87.9 % overall).
2. **Train the skills for how they are used.** A second push starts with the rod right next to the block that was
   just placed, and the skills had never started that way. In the failed T-first scenes, the rod spent a median
   53 s pressing against the guard around the T. Fine-tuning the box skill with the rod starting beside the T in
   30 % of episodes (`task.tool_near_block=0.3`, 40 minutes) raised T-first scenes to 90.3 % and Level 2 to
   92.4 %. The same fine-tune for the T skill (`clutter_v8`) gave 93.0 %, within noise, so `clutter_v7` stays.
   Pushing around the small box was not the problem.

```powershell
python -m s2r2s.level2 --view                                   # watch scenes one after another in the viewer
python -m s2r2s.level2                                          # 1,000 held-out scenes (add --stats for the scene census)
python -m s2r2s.level2 --order tee                              # without the planner: always the T first
python -m s2r2s.level2 --episodes 4 --video videos\level2.mp4   # a 2x2 video
python -m s2r2s.train --run box_v2 --objects box --minutes 20   # retrain: the box pusher, then the box skill
python -m s2r2s.train --run clutter_box_v3 --objects box --clutter tee --init runs\box_v2\best.pt --no-curriculum --minutes 75
python -m s2r2s.train --run clutter_box_v4 --objects box --clutter tee --init runs\clutter_box_v3\best.pt --init-std 0.1 --no-curriculum --minutes 40 --set task.tool_near_block=0.3
```

Video: `videos/level2_grid.mp4` (four scenes; the box's goal is the faint orange outline, the T's is green).

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

**Real arm:** follow [HARDWARE.md](HARDWARE.md) (install `uv sync --extra hardware`, calibrate, test, run).

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

## What was verified before any RL (`pytest`, 35 tests)

- Stripped training model has exactly the Menagerie masses, inertias and servo gains
- FK and Jacobian match MuJoCo; IK solves >99.9 % of the workspace; one IK step per control tick tracks to <0.3 mm (p99)
- Physics state layout, reset validity (object, goal and tool placement), zero-action stillness, stability under random actions,
  and scenes that depend only on the seed (so evaluations are paired across settings)
- Symmetry-aware pose error; the reward prefers progress over idling
- Gate (Level 1): walls block where they are placed, scenes start and end on opposite sides, the object cannot pass
  through a wall, the waypoint stage advances and falls back, and warm start reproduces the old policy exactly
- Hardware bridge: joint-map round trip, the simulated arm reads through its hidden map, calibration recovers
  joint directions and offsets, the LeRobot driver sends degrees and holds the gripper (checked with a fake LeRobot
  and against the real LeRobot 0.6 classes), and policy episodes through the full hardware path succeed
  (Milestone 1, and clutter with the block left untouched) without ever exceeding the joint-step limit
- Clutter: footprint clearances are exact (against brute force, including bars crossing like a plus sign); the guarded
  rod always fits between the block and the T's path; the block stays still at rest; moving it, or coming close,
  costs reward (and moving it, success); the rod guard slides the rod along the block instead of shoving it; and the
  block's camera pose reaches a policy through the hardware path; the rod can start beside the block (as after
  placing it)
- Level 2: every two-block scene has a plan whose pushes all keep the skills' 15 mm rule (also with parking), the
  clearance sweep matches point checks at its ends, and each skill's twin guards the block that must stay
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
| `s2r2s/gate.py`, `tasks.py` | Level 1 gate puzzle (waypoints, path-length reward, gate features); environment factory |
| `s2r2s/clutter.py` | Clutter task: block placement, exact footprint clearances, rod guard, block features and costs |
| `s2r2s/level2.py` | Level 2: two-block scenes, push planner (order, parking), skills run through twins, viewer and videos |
| `s2r2s/ppo.py`, `train.py` | PPO and the training loop (logs, curriculum, evaluation, checkpoints) |
| `s2r2s/evaluation.py`, `evaluate.py` | Held-out evaluation (policy and baseline on identical scenes) |
| `s2r2s/scripted.py` | Heuristic pusher: feasibility check and baseline |
| `s2r2s/play.py`, `record.py`, `visual.py` | Viewer, videos |
| `s2r2s/sensitivity.py` | How good the camera pipeline must be: degrade the camera model, re-evaluate |
| `s2r2s/camera_study.py` | Arm-occlusion study for choosing the real camera mount |
| `s2r2s/hardware/` | Real-arm bridge (LeRobot), joint-map calibration, tracking test, policy runner: see [HARDWARE.md](HARDWARE.md) |
| `tests/` | Regression tests |
