# Gripper validation and starting RL

The environment is ready to **start curriculum RL**, using the actual closed SO-101 gripper. There is no cyan pushing attachment. This completes the pre-training feasibility and engineering checks, not the learned-policy milestone.

## Evidence, 4 October 2026

| Check | Result | Evidence |
|---|---|---|
| Native full path, independently audited every physics step | Pass: 1.887 mm, 0.483 degrees; success hold complete at 27.967 s; final stationary hold 3.183 s | `artifacts/gripper_precision_independent.json` |
| GPU full path with object feedback | Pass: 2.273 mm, 1.246 degrees; success at 28.45 s; zero failure flags | `artifacts/gripper_mpc_precision.json` |
| Contact stability and timestep sensitivity | 22 trials pass; representative native/GPU pushes agree within 2 mm and 2 degrees | `artifacts/physics_validation.json` |
| Randomized geometric resets | 5,620 pass; no initial overlap; subset reset and state restoration checks pass | `artifacts/gripper_reset_validation.json` |
| Success and collision rules | 41 tests pass, including CUDA contact classification, shortcuts, tipping, joint limits, and continuous stationary hold | `artifacts/gripper_final_rules_tests.xml` |
| Table settling and boundaries | 216/216 settling poses pass; all eight boundary cases rejected on both backends | `artifacts/table_candidate_validation.json` |
| Numerical stress test | 460,800 transitions across all stages plus randomized stage 5; no non-finite states or contact-capacity overflow | `artifacts/gripper_numerical_soak.json` |
| PPO save, resume, validation | Iteration 2 to 4; 1,024 to 2,048 transitions; optimizer and normalization continue; 64 validation episodes complete | `artifacts/gripper_ppo_resume_smoke.json` |
| Long-training preflight | Pass; source and evidence hashes checked before training | `artifacts/training_readiness.json` |
| Selected batch PPO smoke | 2,048 environments, two updates and 131,072 transitions complete | `runs/gripper_batch_smoke/status.json` |

Current zero-action physics benchmark: 256 / 512 / 1,024 / 2,048 environments achieved approximately 10,863 / 20,460 / 37,556 / 58,494 transitions per second. The selected 2,048-world configuration used 2.97 GiB with 81.4% VRAM headroom. These are physics-only measurements; contact-rich learning and validation can be slower. The separate two-update PPO test completed successfully, but is too short for a reliable overnight estimate. See `artifacts/benchmark.json`.

The native demonstration is diagnostic control. The GPU demonstration uses offline receding-horizon planning with object feedback and the native reference. Candidate actions are tested in scratch worlds; the execution world moves continuously through physical gripper contact. Neither demonstration is a learned policy, and neither controller supplies a scripted solution inside the RL environment.

The stable table uses plane contact with the finite visual slab retained. The task rejects every block corner outside the bounded workspace, well before the visible table edge. It does not simulate falling off a physical table. Original gripper collisions and the free three-dimensional block remain active. Physics runs at 1 ms; actions run at 20 Hz with 40 mm/s speed and 200 mm/s² acceleration limits.

Earlier fixed-action sensitivity tests failed and remain historical evidence of open-loop fragility. The new successful GPU feedback path does not establish a randomized policy success rate. The stress test verifies numerical behavior under random actions, not task-solving robustness. The short PPO validation has zero success, as expected for this tiny installation test; its checkpoints are not demonstration policies.

## Commands in VS Code PowerShell

Open a PowerShell terminal and run:

```powershell
Set-Location D:\Sim2Real2Sim
.\scripts\verify-gripper.ps1
.\scripts\view-gripper.ps1 -Backend warp
```

This opens the MuJoCo GUI playing the recorded, physically executed GPU gripper sequence. Press **P** to pause/resume and **R** to restart. Orbit/zoom with the viewer controls. For the independently audited native recording:

```powershell
.\scripts\view-gripper.ps1 -Backend native
```

To rerun the full pre-training suite, including both full paths and small PPO tests:

```powershell
.\scripts\verify-gripper.ps1 -Full
```

Start a new nominal curriculum run (do not resume the tiny smoke-test policies):

```powershell
.\scripts\run.ps1 train --num-envs 2048 --seed 0 --stage 1 --hours 10 --run-dir runs/seed0
```

The curriculum advances after three successive validations at 90% success, and retains 20% easier-stage resets. Training is headless for throughput. In a second terminal, start the learning dashboard:

```powershell
Set-Location D:\Sim2Real2Sim
.\scripts\tensorboard.ps1
```

Open <http://localhost:6006>. Logs appear under `runs/seed0/tensorboard`. A normal stop writes `latest.pt`; periodic checkpoints are written every 25 updates. Ctrl+C requests a stop at an update boundary. Avoid closing the terminal or shutting down WSL while it is saving. Numerical failures stop training and preserve diagnostic state and the previous valid checkpoint.

Resume the same run, preserving its environment count and source version:

```powershell
.\scripts\run.ps1 train --num-envs 2048 --resume runs/seed0/latest.pt --hours 10 --run-dir runs/seed0
```

The 10-hour limit applies to this invocation; the default 100-million-transition budget is cumulative for the checkpoint. Use a separate run directory and seeds 1 and 2 for independent training runs. Run them sequentially on this laptop.

## What remains after starting RL

Learning the five stages, proving 95% nominal success and 85% perturbed success, three-seed comparisons, smoothness ablation, actual motion metrics, standard-MuJoCo policy replay, and policy demonstration videos remain future milestone work. No such results are claimed here. Physics randomization ranges remain uncalibrated simulation tests, and native randomized evaluation does not yet mirror all GPU physics randomization.

Only after a stage-5 `best.pt` exists, evaluate an untouched nominal test set:

```powershell
.\scripts\run.ps1 evaluate runs/seed0/best.pt --episodes 500 --num-envs 128 --stage 5 --seed 2000000 --output artifacts/seed0_nominal_test.json
.\scripts\run.ps1 evaluate runs/seed0/best.pt --episodes 500 --num-envs 128 --stage 5 --seed 2000000 --randomized --output artifacts/seed0_randomized_test.json
```

Do not tune against these final test results. Use the training validation logs for development. If learning stalls, inspect stage success, failures and videos before spending the whole overnight budget.
