# Stage 2 reward diagnosis

The original PPO run learned controlled contact but failed nearby pushing. Its last two stage-2 validations were 0/64 successes. This investigation preserves `runs/seed0` and compares old and corrected rewards before further overnight training.

## Confirmed issues

1. **Idle proximity pays repeatedly.** At contact, the old proximity bonus is +0.02 per decision, versus a -0.002 time cost. An otherwise stationary policy can earn +0.018 per decision without approaching the goal.
2. **The reward and success rule disagree about rotation.** Stage 2 requires position accuracy, low translation speed, and an upright block. Its old progress reward also penalizes changes in orientation. Rotation may be necessary during pushing and belongs to the later orientation stage.
3. **Exploration collapsed.** The original latest actor's two action standard deviations are approximately 0.00488 and 0.00346. These are normalized action values, not metres. A near-deterministic actor has little opportunity to discover a different contact strategy.

On the recorded stalled rollout (`artifacts/seed0/policy_stage2_seed3000000.npz`), old reward contributions sum to:

| Component | Total |
|---|---:|
| Position progress | -1.024 |
| Orientation progress | -1.202 |
| Approach progress | +0.143 |
| Repeated proximity bonus | +8.647 |
| Time cost | -1.200 |
| Subtotal | **+5.364** |

The actual recorded return is about +5.360 after motion penalties. Thus this failed behavior earned a positive return. A positive failed return alone does not prove the global optimum is wrong, but these terms provide a concrete incentive for the observed stall.

## Implemented correction

`scripts/stage2_rewards.py` wraps the existing environment and changes **only stage 2 reward**. It does not change physics, observations, control limits, resets, or success rules.

| Term | Old | Corrected |
|---|---|---|
| Position progress | 100 times distance reduction in metres | 300 times distance reduction |
| Orientation progress | Included | Removed in stage 2 |
| Approach progress | 4.5 times reduction in gripper-to-block distance | Unchanged |
| Reward just for proximity | Up to +0.02 every decision | Removed |
| Time cost | -0.002 per decision | -0.01 per decision |
| Success/failure, motion penalties | Existing values | Unchanged |

The same stalled trajectory's corrected subtotal becomes **-8.930**, before the unchanged motion penalties. Position progress is more important; waiting near the block is costly. These coefficients are a tested candidate, not a claim of optimal tuning.

Regression checks verify that idle contact loses reward, useful progress outranks idling, moving away is penalized, rotation alone has no stage-2 cost, terminal returns include the correction, and other stages retain the original reward. Native paired steps confirm identical physics and done flags. Evidence: `artifacts/stage2_reward_audit/reward_diagnosis.json`.

## Paired short experiment

Both arms begin with an identical copy of the original iteration-452 actor and normalization. Both use a fresh critic and optimizer, initial action standard deviation 0.15, minimum standard deviation 0.05, seed 71, 2,048 worlds, and 100 updates (6,553,600 transitions). The critic/optimizer restart is intentional: this is a warm start, not an exact resume.

Both retain 20% easier-stage resets. Every evaluation uses the same 64 stage-2 scenes, seed 1700001. Stage-specific validation, rather than mixed training success or incomparable reward totals, determines the comparison. This is one-seed development evidence, not the final 500-episode milestone test.

- Old reward: `runs/stage2_baseline_audit`.
- Corrected reward: `runs/stage2_corrected_audit`.
- Original immutable comparison inputs: `artifacts/stage2_reward_audit/baseline_latest.pt` and `baseline_stage1.pt`.

The original production trainer remains unchanged for reproducibility. **An ordinary `run.ps1 train --resume ...` does not enable the corrected reward.** Use the separate bounded experiment entry point; do not restart the original overnight command expecting this correction.

```powershell
Set-Location D:\Sim2Real2Sim
.\scripts\train-stage2.ps1 -Variant corrected -Updates 100
```

This creates a new timestamped run and deliberately caps experiments at 150 updates. It never overwrites `runs/seed0`. Review validation before any larger training commitment.

## First comparison result and directional refinement

At 100 updates, the old reward achieved 0/64 successes, 28.78 mm mean error, and zero rule failures. The first correction also achieved 0/64 successes, with 31.77 mm mean error and seven rule failures. Removing the known incentives was insufficient to teach pushing in this pilot; the first correction is not a demonstrated improvement.

The remaining approach term rewards getting near the block from any side. In the stalled recording, the vector from the gripper toward the block points partly away from the goal after contact. A second candidate, `scripts/stage2_directional.py`, replaces this term with progress toward a contact approach point on the side opposite the goal. The point uses the L-block's projected outline and 5 mm clearance, bounded to the controller workspace. Its reward is 30 times the reduction in distance to this point. Position progress remains 300 times distance reduction; idle proximity and orientation costs remain removed.

This is a geometric reward hint, not a scripted controller: it does not write joint targets, choose policy actions, teleport objects, or solve the gate sequence. It is applied only to stage 2. Its physics, stage isolation, terminal accounting, useful-side preference, and zero-goal-distance numerical checks pass. The same short experiment is being evaluated in `runs/stage2_directional_audit`.

Use `-Variant directional` for this refinement. `-Variant corrected` reproduces the first, unsuccessful reward correction. Experiment checkpoints carry a different source signature so an ordinary production resume rejects them instead of silently reverting to the old reward. The exact executed source for the first two runs is archived in `artifacts/stage2_reward_audit/*_executed.py`.

The scenarios and starting weights are matched, but GPU contact replay is not bitwise deterministic. Initial mean error varies by about 0.22 mm between runs. Small differences should not be overinterpreted, and one training seed cannot establish a reliable general improvement.

## Final 100-update validation results

| Reward | Stage-2 success | Mean final error | Rule failures | Timeouts |
|---|---:|---:|---:|---:|
| Original, with restored exploration | 0/64 | 28.78 mm | 0 | 64 |
| Remove idle bonus and orientation term | 0/64 | 31.77 mm | 7 | 57 |
| Also reward approaching the useful side | 0/64 | 23.35 mm | 0 | 64 |

The directional candidate briefly achieved 1/64 successes at update 50, with eight rule failures. That success did not persist. Its final error is lower, but this does **not** satisfy a successful-pushing acceptance gate. The three runs together collected 19,660,800 additional transitions. All finished their bounded budgets; no overnight run was launched.

Plots: [comparison.png](artifacts/stage2_reward_audit/comparison.png), [comparison.pdf](artifacts/stage2_reward_audit/comparison.pdf). Raw stage-specific validation and per-episode results are retained in each run folder. Validation returns use the original environment scorer for consistency; success/error/failure measurements, not reward totals, determine the comparison.

**Recommendation: do not start an overnight run yet.** The reward incentive errors are addressed in the candidates, but reliable stage-2 learning remains unresolved. The next controlled experiment should shorten the initial push distance and gradually increase it, or warm-start from valid stage-2 pushing demonstrations. That is a curriculum/initialization change, beyond the reward-only comparison reported here. Exploration-floor and update settings also remain provisional.

To inspect the final directional candidate (expect incomplete pushing, not a solved task):

```powershell
.\scripts\watch-policy.ps1 -Checkpoint runs/stage2_directional_audit/latest.pt -Stage 2
```

This uses the candidate's reward variant when recording and preserves earlier recordings in separate run folders. The original checkpoint and original stalled rollout remain unchanged.

## Separate development-set check

After checkpoint selection, seed 1705001 supplied 128 additional matched stage-2 episodes, outside the fixed validation set and the final milestone test seeds.

| Checkpoint | Success | Mean error | Invalid episodes | Timeouts |
|---|---:|---:|---:|---:|
| Old-reward best, update 100 | 0/128 | 27.48 mm | 0 | 128 |
| Directional validation-selected best, update 50 | 0/128 | 26.79 mm | 14 | 114 |
| Directional final, update 100 | 0/128 | 22.28 mm | 0 | 128 |

The selected update-50 checkpoint's isolated validation success did not generalize to this set. Its failures include invalid robot contact (14 episodes) and excessive penetration (one overlapping episode). The final checkpoint is more accurate and has no invalid episodes on this set, but still cannot meet the success condition. The overnight recommendation remains **no**.

Raw results: `artifacts/stage2_reward_audit/baseline_development.json`, `directional_best_development.json`, and `directional_final_development.json`. These are diagnostic development results, not a completed robustness benchmark.
