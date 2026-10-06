**Failed-push inspection (2026-10-05):** Replayed the unchanged filtered policy on the same 64 initial scenes: 7 successes, 57 timeouts, zero rule/numerical failures. All timeouts stall in block progress while the gripper keeps making small motions. Traces support a failure to regain useful contact; they do not establish a universal need to switch to the opposite side. Exact failure recordings and GUI commands are in [PUSH_FAILURE_DIAGNOSIS.md](PUSH_FAILURE_DIAGNOSIS.md). No policy, physics, or reward changes and no training in this checkpoint.

**Gripper efficiency update (2026-10-05):** Implemented jaw-aware approach shaping, stronger progress/time incentives, renewed-contact and command-change costs, and a larger failure penalty. A matched 50-update comparison and command-filter tests are complete. Selected experimental replay: `runs/push_efficiency/efficient/filtered.pt`. The unchanged demonstration improves from 18.6 s to 7.3 s with 9.59 mm error. On four matched fresh successes, time improves from 10.23 to 6.74 s and contact bouts from 2.25 to 1.25; however fresh success is still only 7/64. Mild filtering halves the corrected policy's duration-normalized jerk on the fresh panel; it remains rougher than the slow early policy. No overnight run or full reliability claim. See [PUSH_EFFICIENCY.md](PUSH_EFFICIENCY.md).

**Completed learning-rate comparison (2026-10-05):** Two 25-update branches from the same early policy are complete. Fixed-panel successes: early 24/128, adaptive 4/128, fixed 12/128. Fresh-panel successes: early 27/128, adaptive 3/128, fixed 23/128. Fixed 0.0003 beats adaptive here but does not improve the starting policy and increases duration-normalized joint jerk about 2.24 times on fresh scenes. Actual jaw-contact monitoring is implemented. All 768 evaluation episodes have zero rule failures; stochastic training did include rule failures. Preserve the early checkpoint; no overnight run. See [STAGE2_LR_COMPARISON.md](STAGE2_LR_COMPARISON.md) for graphs, evidence and GUI commands.

**Matched stage-2 diagnosis (2026-10-05):** The apparent large success decline was not reproduced on the same 128 scenes: early 23/128 versus later 22/128. Later joint jerk is 2.47 times higher. Input-scaling swaps do not rescue the later policy. A disposable same-data PPO update comparison supports testing a smaller learning rate, but does not prove improved task success. No extended training or checkpoint replacement occurred. See [STAGE2_DECLINE_DIAGNOSIS.md](STAGE2_DECLINE_DIAGNOSIS.md). Earlier interpretation of a success decline is superseded by this matched comparison.

**Short-push curriculum checkpoint (2026-10-05):** Implemented 15-20, 20-25, 25-30 mm goals, then the original 30-40 mm forward-offset distribution. Advancement requires three consecutive 64-episode validations at >=90%. Reset, success, native/GPU, and PPO resume checks pass. A bounded 100-update pilot completed; best validation was 18/64, latest 8/64. The best checkpoint achieved 22/128 (17.2%) on separate development scenes, with zero rule failures and 106 timeouts. No advancement occurred; an overnight run is not recommended. See [STAGE2_DISTANCE_CURRICULUM.md](STAGE2_DISTANCE_CURRICULUM.md) for graphs and the dedicated resume/GUI commands. Production physics and original checkpoints are preserved. Earlier entries below are historical.

**Stage 2 learning audit (2026-10-05):** The original policy learned contact but not directed pushing. Reward fixes and a matched short comparison are documented in [STAGE2_REWARD_DIAGNOSIS.md](STAGE2_REWARD_DIAGNOSIS.md). No overnight run is recommended yet. This learning-quality finding does not retract the physical-feasibility checks below.

**Current gripper validation (2026-10-04):** Precision native and GPU full paths pass (1.89 mm / 0.48 deg native; 2.27 mm / 1.25 deg GPU). The 460,800-transition numerical soak, 216 settling poses, 5,620 resets, and 41 rule tests pass. PPO save/resume and validation also pass; the hash-checked training gate is enabled. See [GRIPPER_READY_FOR_RL.md](GRIPPER_READY_FOR_RL.md) for evidence, limitations and tested PowerShell commands. No long training run has started. Earlier entries below are historical.

# Milestone 1 progress

Updated 2026-10-04. Work proceeds one reviewable task at a time.

**Table-contact checkpoint:** the opt-in plane-contact/visible-slab candidate passes **216/216 GPU settling cases** and **8/8 boundary probes on each backend**. Finite task boundaries are enforced; physical edge-fall dynamics are not modeled by the plane. Millisecond traces with original jaw geometry locate a gripper lower-limit violation at **13.618 s** under gate wedging and a fixed-jaw/south-gate collision at **21.031 s**. Five instrumented three-world runs stayed finite but did not complete the full task; prior intermittent NaNs remain unresolved. See [TABLE_CONTACT_CHECKPOINT.md](TABLE_CONTACT_CHECKPOINT.md). Candidate is not yet the production default; RL remains off. Next: integrate candidate contact with boundary enforcement, improve gate feedback/clearance and gripper margin, then repeat full-path and reset-soak validation.

**GPU isolation update:** the parking jitter is reproducible with frozen motors and the arm away; native MuJoCo settles the same pose. A diagnostic plane table removes the jitter in all three tested GPU worlds without changing friction or stationary thresholds. Shared-stream execution and clearing solver warmstart do not eliminate moving-path divergence. Detailed jaw collisions remain a candidate for investigation; the full-path nonfinite cause is unresolved. Production geometry/controller are unchanged and RL remains off. See [GPU_ISOLATION_CHECKPOINT.md](GPU_ISOLATION_CHECKPOINT.md). Next: validate a stable finite-workspace table contact representation and trace moving-contact numerical failures before feedback improvements.

**GPU replay and sensitivity update:** testing is complete, but this checkpoint **does not pass readiness for RL**. Native nominal still succeeds at 27.25 s. Single-world GPU nominal reaches the goal but times out without the required stationary hold. Of three parallel nominal GPU worlds, one succeeds, one becomes nonfinite, and one tips. All 12 small perturbations fail to complete in each backend; the native 0.5 ms timestep control also times out. These are fixed-command diagnostics, not trained-policy success rates. See [GPU_REPLAY_SENSITIVITY.md](GPU_REPLAY_SENSITIVITY.md). Next: isolate GPU settling/numerical consistency, then improve feedback and parking margin. RL has not started. Earlier updates below describe prior checkpoints.

**Complete-path update:** the **closed-gripper push–turn–park sequence passes** independent native MuJoCo replay and production environment rules. Completion is **27.246 s** (environment success at **27.25 s**), with **9.316 mm / 7.049 degrees** final error and zero failure flags. See [GRIPPER_COMPLETE_CHECKPOINT.md](GRIPPER_COMPLETE_CHECKPOINT.md). This is one nominal diagnostic, not RL or a randomized success rate. Full-path GPU replay and sensitivity checks remain. Earlier checkpoint text below is historical.

**Gate update:** closed-gripper repositioning and complete gate passage now pass independent replay and the actual environment rules. Entire block clears at **14.376 s**, with zero failure flags and a stationary finish beyond the gate. See [GRIPPER_GATE_CHECKPOINT.md](GRIPPER_GATE_CHECKPOINT.md). Final rotation/parking is next; RL remains off. The stalled attempt described below is retained as history.

**Latest checkpoint:** the **closed-gripper contact suite passes all 22 trials**, including native/GPU and 1/0.5/0.25 ms comparisons. **37 rule tests, 5,620 resets and 148 settling episodes** also pass on the current settings. See [GRIPPER_PHYSICS.md](GRIPPER_PHYSICS.md). The gripper-only path attempt turns the block and reaches the gate but stalls; independent replay also rejects brief jaw–gate contact. Complete valid passage/parking and shortening the path remain pending. Old attachment-based full-path results below remain historical only. RL has not started.

## Current checkpoint: visible scene inspection

- Installed isolated WSL Python 3.12 training environment; dependency lock is `uv.lock`.
- GPU MuJoCo Warp smoke test passed. Short zero-action physics benchmarks ran at 256, 512, 1024 and 2048 environments. These are not PPO training throughput measurements.
- Installed a separate native Windows MuJoCo 3.11.0 viewer environment, pinned in `requirements-viewer.txt`. WSL rendering produced corrupted images; Windows rendering was visually verified.
- Built the SO-101 scene using the pinned Menagerie model, fixed base, rounded pusher, free L-shaped block, gate and parking outline.
- Initial tool IK position error is checked below 1 mm when the viewer starts. Full path feasibility has not yet been demonstrated.
- RL environment and training code are drafts requiring further validation. No long training, trained policy, final evaluation or milestone acceptance result exists yet.

## Open the GUI

From PowerShell in `D:\Sim2Real2Sim`:

```powershell
.\scripts\view.ps1
```

The viewer starts paused with no policy loaded. Drag the left mouse button to orbit, use the wheel to zoom, and drag the right mouse button to pan. Press **P** to run/pause physics with joint-position hold, and **R** to reset and pause. Close the window to exit. Running physics does not execute the puzzle.

The visually checked screenshot is `artifacts/scene.png`. GUI diagnostics are in `artifacts/viewer.log` and `artifacts/viewer.error.log`. Reinstall viewer dependencies with `scripts/setup-viewer.ps1` (requires the existing managed Python 3.12 or a pre-created `.venv-viewer`). The viewer does not require PyTorch.

## Next tasks, separately

1. Review scene appearance and layout with the user.
2. Validate controller, contacts, reachable paths, reset logic and independent success detection; demonstrate a feasible full solution using diagnostic control.
3. Run a short PPO training smoke test and validate checkpoint/resume behavior.
4. Train the curriculum, measure performance, and only then conduct held-out evaluation and prepare demonstrations.

The 95% nominal and 85% randomized success rates remain acceptance targets. The scene's appearance and initial IK do not establish that the entire puzzle is feasible.

## Controller/contact checkpoint (2026-10-03)

Executed `scripts/controller_audit.py` in the WSL training environment using standard MuJoCo and the actual environment's `_control` implementation. The trial holds for five seconds, follows three diagnostic tool waypoints, and settles for two seconds. These diagnostic waypoints are not part of the RL policy/controller. Recorded states are replayed by the Windows GUI; replay itself does not re-simulate physics.

| Check | Measured result | Status |
|---|---|---|
| Five-second zero-command tool hold | 0.103 mm position error | Pass for this pose |
| Commanded planar speed limit | 40 mm/s maximum | Pass |
| Commanded planar acceleration limit | 200 mm/s² maximum | Pass |
| Physical push and rotation | Block moved and rotated | Observed |
| Deepest contact penetration at 2 ms | 1.936 mm, table/block | Needs correction |
| 1 ms versus 2 ms final object position | 9.495 mm difference | Not sufficiently stable |
| 1 ms versus 2 ms final object orientation | 23.569 degrees difference | Not sufficiently stable |
| Full puzzle completion | Not demonstrated | Pending |

Command limits are not measurements of actual tool speed/acceleration. This trial is one canonical scene, not comprehensive workspace validation. The physics feasibility gate remains open; no training was started. Next task: investigate block/table contact parameters and pushing geometry, repeat timestep comparison, then attempt full-path control.

Reopen this original diagnostic with `scripts/view-audit.ps1 -Trial initial`. The default `scripts/view-audit.ps1` now shows the corrected contact trial. **P** pauses/resumes, **R** returns to the beginning and pauses. Original evidence: `artifacts/controller_audit.json`, `artifacts/controller_audit_1ms.json`, and their `.npz` trajectories.
