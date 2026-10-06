# Stage-2 failed-push diagnosis

Inspected the selected `runs/push_efficiency/efficient/filtered.pt` policy without changing weights, rewards, physics, or controller. No training was run. Repeated the 64-scene panel with RNG seed 6600000 and recorded full trajectories. Its initial-scene hash matches the earlier evaluation: `e04184708a9f0230580eb4f19a13799962122cccf41a7d6068c096fbcb1537d8`.

## Finding

The block stops making useful progress, but the robot does not simply stop commanding movement. After an initial push, the gripper settles into small repetitive motions and fails to obtain another useful push. Repositioning and reacquiring productive contact is the appropriate next skill to address.

The user's suggestion to approach from a different side is plausible for some configurations, but these measurements do not prove that every failure requires the opposite side. None of the failed scenes had a mean final-five-second jaw/goal side cosine above +0.3 (positive means the jaw is on the goal-facing side). This center-based geometry is only a clue: the L shape, contact normals, rotation, and reachability matter. A collision-checked alternative push has not been demonstrated by this diagnostic.

## Observations

- 7/64 successes; 57/64 timeouts; zero rule failures or numerical failures.
- All 57 timeouts have less than 0.1 mm absolute net goal progress in the final five seconds. Median actual tool speed during that interval is 2.51 mm/s (range 1.16-5.01), closely matching the commanded speed. This points to the policy's small repeated commands rather than a frozen simulator or a controller ignoring large commands.
- All 57 finish closer to the goal than they started (0.36-9.06 mm improvement). Net pushing in the wrong direction is not the dominant failure in this panel; small temporary backward movements can still occur.
- 48 failed episodes have sampled jaw contact followed by stalled progress. Nine have no contact at the sampled instants. **Those nine cannot be called proven missed-contact episodes:** sampling at 20 Hz misses brief contacts between frames, and their blocks did move.
- Median nearest jaw-tip sphere gap over the final five seconds is 6.20 mm (range 2.49-11.78). This is a six-sphere geometry proxy, not the clearance of every gripper surface. The median final-five-second sampled contact duration is zero.
- These are heuristic trajectory categories, not causal proof of which reward or training setting produced the behavior. No separate training seed or new test panel was added.

## Representative exact batch recordings

Scene 56 illustrates repeated ineffective motion. Goal error falls from 17.14 to 13.48 mm, then becomes flat. In the last five seconds the block has zero measured motion while the tool moves at 5.01 mm/s, with no sampled contact. The final nearest-tip gap averages 7.74 mm. The required position tolerance is 10 mm, so this is an unfinished push, not a successful early stop.

Scene 39 improves from 18.37 to 15.37 mm, largely in the first three seconds. The tool continues oscillating at an average 2.51 mm/s over the final five seconds, while the block remains stationary. No contact is captured at the 20 Hz sample instants; brief earlier contact remains possible. This episode must not be labeled 'never touched' from these traces.

The plot shows block/jaw paths, goal error with sampled contacts, and actual versus commanded tool speed: [failure_traces.png](artifacts/push_failure_diagnosis/failure_traces.png). Green circles represent the stage-2 block-center position tolerance, not the complete parking footprint or orientation requirement.

```powershell
Set-Location D:\Sim2Real2Sim
.\scripts\watch-push-failure.ps1 -Scene 56
# Second representative:
.\scripts\watch-push-failure.ps1 -Scene 39
```

These replay the exact recorded batch trajectories, avoiding the different random draws that can occur when a single environment is initialized separately. P pauses; R restarts. Stage 2 is position-only and has no gate. These recordings end on timeout at 30 seconds.

## Recommended next checkpoint

Teach and validate recovery after an initial push: move to a useful new contact position, re-establish contact, and continue toward the remaining goal. First demonstrate a feasible recovery from a representative stalled state, then use a bounded learning experiment with the smoothing filter active during training. Evaluate whether the policy learns recovery across starts, rather than inserting a mandatory switch-side script into its actions. Preserve the existing policy as the comparison baseline.

The current recontact penalty charges 0.1 after a contact gap of at least 250 ms, including potentially useful repositioning. Its effect deserves a controlled ablation; these traces alone do not establish that it caused the failure. Do not solve this by simply increasing command gain: that previously increased invalid collisions.

## Reproduction and evidence

- `scripts/diagnose_push_failures.py`: unchanged-policy GPU trace collection, terminal states captured before autoreset, numeric and geometric metrics.
- `scripts/report_push_failures.py`: figures and native-viewer episode exports.
- `artifacts/push_failure_diagnosis/report.json`: checkpoint hash, matching initial-scene hash, 64 episode outcomes and diagnostic thresholds.
- `artifacts/push_failure_diagnosis/traces.npz`: qpos, block/tool/jaw positions, raw/applied actions, command velocities, controller targets, sampled contact, hold counter, goals, and mocap state.
- `artifacts/push_failure_diagnosis/representatives.json`: exact metrics for scenes 56 and 39.

As before, GPU repetitions can differ slightly near contact thresholds. This repeat again gives 7/64 success; it does not establish bit-identical trajectories or robust task completion.
