# Roadmap: from Milestone 1 to the proposal's Sim2Real2Sim study

## 1. What the proposal needs

The research claim: **at matched simulation compute, choosing training scenes around failures observed on
the real arm (C) beats choosing them around failures observed in simulation (B) and uniform randomisation (A)**.
It is tested on a hobby arm (SO-101) with about 150 real trials, using continuous final-pose error.

That requires five working parts, in this order:

1. A state-based simulation policy for planar SE(2) pushing. It must be good in simulation and transfer well
   enough that real failures carry information. A policy that always fails or always succeeds on hardware
   gives the sampler nothing to learn from. **Milestone 1 delivers this (README).**
2. A perception pipeline that turns camera images into the policy's inputs (object pose) and also logs
   each trial's scene parameters automatically.
3. A hardware interface that runs the same controller code as simulation, with trial logging.
4. A failure-conditioned scene sampler and a retraining loop with matched compute across A, B and C.
5. An evaluation protocol: held-out scenes, continuous error, randomised interleaved blocks, confidence intervals.

**Proposal fixes worth making now**
- **Reference [2].** The text says "Fail2Progress approach [2]", but [2] (arXiv:1810.05687) is Chebotar et al.,
  *Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience*. Fail2Progress is
  Y. Huang, N. Alvina, M. Devendran Shanthi, T. Hermans, *Fail2Progress: Learning from Real-World Robot Failures
  with Stein Variational Inference*, arXiv:2509.01746 (2025). Both are relevant: cite both and fix the sentence.
- **Trial budget.** The proposal implies about 330 real trials (150 for collection plus 60 × 3 for evaluation). The earlier
  `PLAN.md` proposes 750 (every condition evaluated after every cycle). Choose one and state it.
- **Undefined scene factors.** Define "approach direction" and "clutter" precisely in the scene specification.
  Milestone 1 has no clutter yet; one anchored obstacle is the natural first version.

## 2. Why the earlier attempt stalled, and what changed

| Issue | Evidence in the old docs | What s2r2s does instead |
|---|---|---|
| The task was much harder than the proposal's: Push-Turn-Park through a 48 mm gate, 5 stages, 1 s stationary hold | Never got past stage 2 of 5; best fresh success about 11 % | The proposal's task (push to an SE(2) goal pose); the curriculum varies goal difficulty only |
| Pusher was the gripper's jaw meshes; extreme solver settings (1 ms, 100 iterations, very stiff contacts); GPU physics | Many checkpoints spent on jitter, NaNs and native-vs-GPU divergence | Dedicated rod, Menagerie defaults at 5 ms, CPU rollout. Deterministic, with fidelity checked against 1 ms |
| Runs too short to show learning | Most experiments were 3–6.5M transitions (50–100 updates); the longest was 29.6M | For comparison, our policy needed 26M samples just to master the easiest goals, and 157M for the final 96.6 % (45 min here) |
| Reward patches (idle proximity bonus, conflicting terms); strict promotion gate (3 × 90 %) | `STAGE2_REWARD_DIAGNOSIS.md`, `STAGE2_DISTANCE_CURRICULUM.md` | Bounded kernels + progress on one keypoint metric; automatic 80 % gate |
| Exploration collapse (action std about 0.005) | `STAGE2_REWARD_DIAGNOSIS.md` | Std floor, KL-adaptive learning rate |
| No feasibility check before RL | — | Scripted pusher + regression tests ran *before* training |
| No camera model; the policy saw exact simulator state | — | Actor gets a delayed, noisy, sometimes-missing pose estimate; critic gets the truth |

The underlying lesson: **prove the task is solvable (scripted controller), make simulation cheap (about 70k
samples/s), then train long enough.** Diagnosing a 2-minute run is not informative.

## 3. How the arm will "see" the objects

**Architecture (unchanged from the proposal: state-based).**
`camera → object pose estimator → (x, y, yaw) in robot frame → same observation vector as in simulation → policy → IK → servos`

The policy does not consume pixels. The camera's only job is to supply the object's planar pose, about 30 times per second.
In simulation, the camera is modelled as a *sensor* with latency (0–100 ms), noise (1 mm / 1°) and dropouts (3 %),
so the policy is already robust to an imperfect estimator. In the viewer, the yellow outline is exactly what the policy sees.
When real measurements arrive, replace these numbers with the measured ones.

Why not an end-to-end vision policy? It makes RL far more expensive, opens a large visual sim-to-real gap,
and the proposal is state-based. The sampler also needs scene parameters (poses) that this pipeline logs for free.
If a vision policy is wanted later, distil the state-based teacher into an image-based student.

**Which camera?** Use the **RealSense D435i** as the primary sensor:
- Its colour stream (1280×720 or 1920×1080 at 30 fps) has factory-calibrated intrinsics and fixed focus.
- Its depth stream makes segmentation easy: anything about 20 mm above the table plane is an object, regardless of colour or lighting.
- Depth also gives the table plane and lets you detect tipping or lifting.

The Logitech also works (similar field of view). It needs ChArUco intrinsic calibration and locked focus,
exposure and white balance. Use it as a second camera for recording demos.

**Pose estimation.**
- *Recommended: markerless.* Segment the object's top face (depth height ≈ 20 mm, plus colour), back-project
  to the table plane, and fit the known 2D template: centroid and second moments for initialisation, then IoU or ICP
  refinement. The T's silhouette is unambiguous. This scales to all six objects from `objects.py`.
- *Alternative: AprilTag 36h11 / ArUco on each object.* Simple and gives the object ID for free. But the half-scale
  T's bars are only 15 mm wide, so tags would be ≤12 mm. Validate the detection rate *during pushing* before relying on them.
- *Calibration:* a ChArUco board placed at known robot-frame positions. Register the table frame by touching
  marked points with the rod tip. Acceptance: **≤2 mm and ≤2°** on a measured grid; latency measured; dropout rate during pushing.

**Perception spec, measured on the trained policy** (`python -m s2r2s.sensitivity`):
- Pose noise ≤2 mm / 2° (1σ): 96 %. At 4 mm / 4°: 76 %. At 8 mm / 8°: 23 %.
- End-to-end latency ≤150 ms: 100–200 ms gives 95 %, 200–300 ms gives 81 %.
- Dropped frames barely matter: 50 % dropped still gives 94 %.

If the real pipeline turns out worse, retrain with its *measured* noise and latency rather than tuning the estimator forever.

**Where to mount it.** `python -m s2r2s.camera_study` casts rays from candidate camera mounts to the object's top face
during pushing episodes driven by the trained policy (64 episodes, 12,800 views per mount):

| mount (robot frame, m) | mean occlusion | >50 % hidden | >90 % hidden | mm/pixel at 720p |
|---|---|---|---|---|
| **front_low (0.62, 0, 0.38): opposite the robot, ~40° elevation** | **14 %** | **2 %** | **0 %** | 0.61 |
| corner (0.50, 0.32, 0.55) | 34 % | 25 % | 0.3 % | 0.76 |
| front_high (0.46, 0, 0.62) | 35 % | 26 % | 0.6 % | 0.73 |
| side_left (0.20, 0.42, 0.55) | 43 % | 42 % | 1.3 % | 0.75 |
| overhead (0.20, 0, 0.70) | 67 % | 73 % | 26 % | 0.76 |
| behind_robot (−0.15, 0, 0.60) | 69 % | 80 % | 17 % | 0.75 |

Mount the camera **opposite the robot and fairly low**. A camera straight overhead is almost the worst
choice: whenever the arm pushes from the far side, the forearm reaches over the object. The scene's `d435i`
camera now sits at the recommended pose. `videos/camera_study/camera_views.png` shows all six views.

## 4. Milestone 2: first real pushes (exact steps)

1. **Station.** Clamp the SO-101 to the table. 3D-print the pusher: a Ø12 mm rod clamped in the closed gripper,
   ending 30 mm below the jaw tips. If you build something different, update `SceneConfig`. Print the T and
   the other five objects to `objects.py` dimensions, ballasted to about 30 g. Use a matte table mat.
2. **Joint mapping, the most important interface.** Map LeRobot `SO101Follower` joint readings to MuJoCo radians
   (sign and offset per joint). Verify by commanding poses and checking `kinematics.py` FK against a ruler.
   Then run `kinematics.py` IK unchanged and send goal positions at 20 Hz, with a workspace clamp, speed limit and e-stop.
3. **Tool calibration.** Command the rod over a grid. Measure the real tip positions to find tool-length and height offsets, backlash and sag.
4. **Camera pipeline and calibration** (section 3).
5. **System identification.** Servo step responses (gain, latency), push tests (friction range), and *replay*: send identical
   command sequences in simulation and on hardware and compare object trajectories. Set the randomisation ranges so they cover the real values.
6. **Zero-shot deployment** of the Milestone 1 policy, about 20–30 logged trials (scene spec, trajectories, video, final error).
   Diagnose the gap by component: perception, servo tracking, friction, geometry. Retrain with corrected ranges if needed.

## 5. Milestone 3: the Sim2Real2Sim loop

1. **One scene specification** c = (object id, start pose, goal pose, approach direction, clutter). It is shared by the
   simulation sampler, the real-trial generator and the logs. Today `env.reset` *samples* (start pose, goal pose)
   internally per world, and the object is fixed per world. Milestone 3 needs a small change: reset from an explicit
   list of scenes supplied by the sampler.
2. **Pre-registered failure definition**, for example final E > 1 or outside tolerance.
3. **Sampler:** p_{k+1}(c) = 0.5 p_uniform(c) + 0.5 p_failure,k(c), a kernel mixture around failed scenes
   (10 mm / 10° perturbations, categorical factors kept), rejecting invalid scenes. Fall back to uniform when there are no failures.
4. **Conditions A/B/C with matched compute**: the same number of PPO samples per cycle. B's failures come from simulated
   trials using C's scene assignments.
5. **Three cycles × 50 real trials.** A fixed held-out set of 60 scenes never enters adaptation. Run conditions in randomised,
   interleaved blocks. Report E with bootstrap CIs over scene blocks.

## 6. Immediate next actions

- [ ] Run `.\scripts\setup.ps1`, then `.\scripts\play.ps1 runs\tee_v1\best.pt`. Press K to kick the T mid-push and watch it recover.
- [ ] (Simulation) Train the six-object generalist; add one anchored obstacle (clutter) to the scene and the sampler.
- [ ] (Hardware) Print the rod and the T; build the LeRobot ↔ MuJoCo joint mapping (Milestone 2, step 2).
- [ ] (Perception) Mount the D435i where the occlusion study suggests; prototype the markerless T pose estimator.
- [ ] (Paper) Fix reference [2] and the trial budget.
