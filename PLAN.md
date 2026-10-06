# MuJoCo + SO-101: a practical approach to the proposal

## 1. Assessment and research objective

**This is feasible as a 10–12-week project, provided calibration and reliable pushing come before the adaptation experiments.** Your RGB camera, printed markers, and SO-101 are sufficient for the main study. Depth can support diagnostics.

I reviewed the complete proposal: :codex-file-citation{path="D:/Sim2Real2Sim/Project_Proposal_Doc.pdf" purpose="source"}.

The central question should be:

> With the same simulation training budget, does using real-world failures to choose subsequent training scenes improve physical pushing performance more than uniform sampling or simulation-failure sampling?

A functioning robot demonstration establishes the engineering result. **The comparison between the three methods establishes the research result.** Improvement from real failures remains a hypothesis to test.

Four points need correction or clarification:

- **Trial budget:** use your selected **750 task trials**, plus calibration and pilot trials. This supports the complete learning curve.
- **Literature:** reference [2] is Chebotar et al.’s *Closing the Sim-to-Real Loop*. The actual *Fail2Progress* paper is by Huang et al. and uses Stein variational inference to improve skill effect models. Your proposed density sampler is a simpler, related approach; describe that distinction explicitly. [Chebotar et al.](https://arxiv.org/abs/1810.05687), [Fail2Progress](https://arxiv.org/abs/2509.01746)
- **Implementation status:** the inspected project directory contains the proposal only. Treat model conversion, inertia correction, training, and validation as planned work.
- **Dynamics identification:** lacking force sensing limits what can be inferred from task failures, but encoders and camera trajectories still support preliminary calibration. Keep dynamics adaptation outside the main experiment to preserve its focus.

## 2. Physical setup and simulation

**Build a repeatable planar pushing station.**

Clamp the arm to a rigid table. Attach a lightweight, rounded pushing tool to the fixed part of the gripper assembly so pushing does not depend on grasp reliability. Model its geometry and attachment in MuJoCo.

Use six lightweight, flat-bottomed objects with measurable orientation. Record their dimensions, mass, geometry, marker offset, and rotational symmetry. Start development with one rectangular object, then expand to all six.

For clutter, use either an empty workspace or **one anchored obstacle of known shape**, with its location varied between trials. Include both cases in training and evaluation.

**Use RGB to turn images into numerical state.**

Mount the camera overhead so it sees the complete working area, object markers, and workspace reference markers. The pipeline will be:

`RGB image → marker detection → object pose in robot coordinates → policy → arm motion`

Calibrate:

- Camera intrinsics and lens distortion.
- Camera-to-table and table-to-robot transforms.
- Marker-to-object-center offsets and marker height above the table.
- Pushing-tool position relative to the arm.

A table-plane homography alone can introduce errors for markers mounted above the table. Use calibrated pose estimation with the known marker geometry and height. OpenCV documents the required calibration and marker-pose workflow. [OpenCV marker pose estimation](https://docs.opencv.org/4.7.0/d5/dae/tutorial_aruco_detection.html)

Depth is optional for checking table height, detecting tipping, and inspecting unexpected contact. Keep it outside the policy inputs for the primary experiment unless the RGB pipeline fails its validation.

**Start from an existing SO-101 MuJoCo model.**

MuJoCo Menagerie already provides an SO-101 MJCF model, including collision geometry. This removes the need to begin with a fresh URDF conversion. Its current XML defines a backlash class but explicitly says it is not applied to joints; using the model does not establish that backlash is represented. [Model documentation](https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotstudio_so101/README.md), [Model XML](https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotstudio_so101/so101.xml)

Add the measured table, tool, objects, and obstacle. Use primitive or compound collision shapes wherever they adequately represent contact. Keep objects dynamically free in 3D so tipping and lifting remain possible failures, even though the task is planar.

Before training, measure and reproduce:

| Measurement on the real arm | Purpose in simulation |
|---|---|
| Joint direction, zero position, and range | Correct coordinate mapping |
| Small steps and slow trajectories | Approximate servo response and delay |
| Direction reversals | Bound backlash and hysteresis |
| Repeated simple pushes | Check translation, rotation, and friction variability |
| Camera timing and pose noise | Match the policy’s observation quality |

Freeze the resulting dynamics-randomization ranges before the comparative study. Apply the same ranges to all methods. MuJoCo provides an explicit mechanism for modeling backlash; its parameters still need measurement on your arm. [MuJoCo backlash modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#backlash)

## 3. Policy, software interfaces, and adaptation loop

**Use a compact state-based reinforcement-learning policy.**

Use Python 3.12, MuJoCo, Gymnasium, Stable-Baselines3 PPO, OpenCV, and LeRobot’s SO-101 hardware interface. Pin the tested package versions and model revision. PPO supports continuous actions and CPU-based parallel environments, making it a reasonable starting point for this small state space. [PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

The policy receives:

- Measured joint positions and recent observation history.
- Estimated pushing-tool position.
- Object position and orientation.
- Target position and orientation.
- Object identity.
- Obstacle presence, pose, and dimensions.
- Previous action.

Represent angles with sine and cosine. Simulated observations must include measured noise and latency; the deployed policy must never depend on simulator-only quantities such as contact forces or exact hidden dynamics.

The policy outputs **two-dimensional tool displacement**, converted by a shared inverse-kinematics controller into joint-position targets. Use a fixed pushing height and a feasible tool orientation. This keeps the action space manageable while retaining the arm’s physical constraints.

Initial engineering settings, subject to pilot validation:

- Policy frequency: **20 Hz**.
- MuJoCo timestep: **2 ms**, giving 25 physics steps per action.
- Maximum tool displacement: **2 mm per action**.
- Episode duration: **20 seconds**.
- Three recent observations supplied to the policy.

Use a reward based on improvement in position and orientation error, with additional terms for approaching the object, smooth motion, successful completion, and invalid contact or workspace exit. Freeze the reward before comparing methods.

LeRobot exposes joint-position observations and commands. Implement and test an explicit mapping between its configured units, joint offsets, and MuJoCo radians. Log the commands actually sent after clipping. [SO follower implementation](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py)

**Keep the simulation and hardware contracts identical.**

| Interface | Required content |
|---|---|
| Scene specification | Object, start pose, goal pose, initial approach, obstacle configuration |
| Observation | Timestamped measured state, pose validity, recent history |
| Action | Tool displacement and resulting bounded joint command |
| Trial record | Scene, method, cycle, seed, checkpoint, trajectories, video, errors, termination reason |

Use metres, radians, and a documented robot-base coordinate frame internally. Each trial must be replayable from its scene specification and recorded commands.

**Implement a simple failure-conditioned sampler.**

Let \(c\) represent a scene configuration. After each collection cycle, update the simulation sampler to:

\[
p_{k+1}(c)=0.5\,p_{\mathrm{uniform}}(c)+0.5\,p_{\mathrm{failure},k}(c).
\]

Half the training scenes retain broad coverage; half concentrate around observed failures.

For the failure component:

1. Select a failed trial from the accumulated adaptation log.
2. Retain categorical values such as object identity and obstacle presence.
3. Perturb its start pose, relative goal, initial approach, and obstacle pose.
4. Reject overlapping, unreachable, or otherwise invalid scenes.

This is an empirical kernel mixture—a simple density model suited to a small dataset. Start with **10 mm position and 10° angular perturbations**, validate them during the pilot, and then freeze them. Fall back to uniform sampling when no eligible failures exist.

Keep the physical collection scenes drawn from a fixed, object-balanced distribution. This prevents repeated sampling of one difficult region from being mistaken for evidence that it fails more frequently.

The three conditions are:

| Condition | How subsequent simulation scenes are selected |
|---|---|
| **A: Uniform** | Continue uniform domain randomization |
| **B: Simulation failures** | Mixture sampler fitted to failed simulated trials |
| **C: Real failures** | Identical sampler fitted to failed physical trials |

For each cycle, B receives 50 simulated diagnostic trials matching C’s planned scene and seed assignments. Use identical failure criteria and sampler settings. Give all conditions equal simulation interaction and optimization budgets, including diagnostic simulation.

Train three initial seeds and fork A, B, and C from the corresponding initial checkpoints. Pool failure data within each adaptive condition across those seeds. For C, divide each 50-trial collection approximately equally across seeds, rotating the remainder between cycles. This produces three policy replicas under a shared adaptation history; it does **not** constitute three independent real-world adaptation studies.

**Check whether simulation can expose the real failures.**

Replay physical failures in MuJoCo, using the recorded scene and commands across the fixed dynamics ranges. Inspect whether the simulated trajectories show comparable difficulty.

If real failures are absent throughout simulation, investigate missing actuator behavior, sensing errors, geometry, or contact modeling before starting the formal comparison. Once the study starts, any necessary model revision requires rerunning affected conditions consistently.

## 4. Evaluation and acceptance criteria

Your selected full learning curve requires:

| Trial group | Physical trials |
|---|---:|
| Shared initial baseline | 60 |
| Three adaptation collections | 150 |
| A, B, and C after cycle 1 | 180 |
| A, B, and C after cycle 2 | 180 |
| A, B, and C after cycle 3 | 180 |
| **Total** | **750** |

Calibration and pilot trials are additional. At an assumed 3–5 minutes per trial including reset, reserve approximately **38–63 hours of trial time**, plus setup and maintenance.

Create **60 held-out scenario blocks**: ten per object, split equally between empty and cluttered scenes. Assign 20 blocks to each seed. Reuse each block’s scene and seed across methods and checkpoints for paired comparisons.

Save every checkpoint, then evaluate the frozen checkpoints after adaptation is complete. Randomize and interleave method and checkpoint order within sessions. Evaluation outcomes must never update the sampler, reward, model, or checkpoint selection.

The learning-curve horizontal axis is **real adaptation trials used: 0, 50, 100, 150**. Report the separate 600-trial evaluation cost explicitly.

**Measure both translation and rotation.**

Report final position error in millimetres and symmetry-aware orientation error in degrees. Measure the settled object after the tool withdraws.

Use the preregistered normalized score

\[
E=\frac{e_{\mathrm{position}}}{20\ \mathrm{mm}}
+\frac{e_{\mathrm{orientation}}}{10^\circ}
\]

for the primary comparison, while always reporting its two components separately. Use reaching both tolerances as a secondary success measure. These are proposed task tolerances, not measured capabilities.

The primary hypothesis is lower final-cycle \(E\) for C than B; C versus A is secondary. Report paired effect sizes and 95% confidence intervals by resampling whole scenario blocks, preserving method/checkpoint pairing. Report seed and object breakdowns descriptively; ten scenarios per object will not support strong object-specific conclusions.

Retain aborted trials. Recover their final pose from stationary images when possible; report missing measurements and sensitivity bounds when recovery fails. Distinguish task failure, tracking loss, communication failure, and setup error using rules fixed before evaluation.

**Required validation before the formal study:**

- Camera pose accuracy target: **≤2 mm position and ≤2° yaw** on an independently measured grid.
- Repeatable tool trajectories across the permitted workspace, initially targeting **≤5 mm tracking error**.
- Correct units, joint signs, angle wrapping, symmetry handling, and scene replay.
- Stable contacts and similar outcomes when halving the simulation timestep.
- Verified handling of stale images, unreachable actions, workspace limits, and interrupted communication.
- Automatic, reproducible sampler updates with no held-out trial IDs entering adaptation.
- A working uniform-randomization policy on all six objects before collecting comparative evidence.

A null result remains a valid project outcome if the experiment is controlled. Claims should remain specific to the tested arm, objects, workspace, and shared adaptation protocol.

## 5. Schedule and deliverables

| Weeks | Work and completion gate |
|---|---|
| **1–2** | Assemble the station, tool, and marker tracking; validate pose and coordinate accuracy |
| **3–4** | Validate the MuJoCo model, controller, actuator behavior, and simple push replay |
| **5–6** | Train the uniform baseline; expand to six objects and simple clutter; finish pilot and freeze protocol |
| **7–9** | Run three 50-trial adaptation cycles; retrain A/B/C with matched budgets |
| **10–11** | Run the 600 held-out evaluations using frozen checkpoints |
| **12** | Analyze results, produce figures, document limitations, and prepare the demonstration |

The deliverables are a calibrated physical station, validated MuJoCo environment, shared simulation/hardware interface, reproducible training and sampling code, complete trial logs, frozen checkpoints, and an evidence-backed comparison of all three methods.

The immediate implementation milestone is **one tagged object pushed reliably in both MuJoCo and reality, with synchronized trajectories and measured final pose error**.

- :codex-followup[Build the starter project]{prompt="Implement the starter project for this agreed MuJoCo and SO-101 plan, including the environment, shared interfaces, configuration, and trial logging."}
- :codex-followup[Develop the calibration workflow]{prompt="Develop the RGB marker calibration and SO-101 coordinate validation workflow for this project, including procedures and acceptance checks."}
- :codex-followup[Prepare the experiment protocol]{prompt="Turn the agreed 750-trial study into a preregistered protocol with trial allocation, metrics, exclusion rules, statistical analysis, and a revised proposal."}
