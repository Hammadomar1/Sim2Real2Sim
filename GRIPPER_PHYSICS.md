# Closed-gripper contact validation — 2026-10-04

**The full contact regression suite passes on the gripper model. RL has not run.**

The original jaw collision meshes and primitives remain active; no cyan attachment or replacement pushing shape is present. The gripper holds its closed command. The layout, friction coefficients, mass, controller speed/acceleration limits and acceptance thresholds are unchanged.

Two settings changed: physics now runs at **1 ms**, with 50 physics steps per 20 Hz command; jaw contacts use **three force components** (normal and two sliding directions, MuJoCo `condim=3`). Multiple contacts on the actual jaw surfaces generate torque. The extra point-level rolling/torsional friction terms from the imported grasping model are omitted. This is a simulation contact-model choice, not measured hardware calibration.

## Evidence

The original 2 ms / six-component jaw contacts failed timestep and native/GPU agreement. Increasing solver iterations alone did not help. Softer contacts and three-component contacts at 2 ms helped individual examples but did not pass the broader checks. The combined 1 ms / three-component setting passed the original limits.

`artifacts/physics_validation.json` contains **22 trials**: six contact scenes at 1 / 0.5 / 0.25 ms, 30-second zero/random-action trials, and two GPU comparisons. The finer timestep and GPU runs receive the same actions as their 1 ms reference.

| Comparison | Maximum position difference | Maximum yaw difference |
|---|---:|---:|
| Native timestep comparisons | 0.768 mm | 1.209 degrees |
| Native vs GPU, simple push | 1.449 mm | 1.434 degrees |
| Native vs GPU, blocked gate | 1.684 mm | 1.170 degrees |
| Required limit | 2 mm | 2 degrees |

Finite-state, fixed-base, joint-limit, object-height/uprightness, penetration (at most 1 mm), hold-drift and command-limit checks also pass. GPU penetration is sampled at control steps in this regression report; native contact checks sample every physics step. These stress rollouts are not full-path success demonstrations. The task's separate failure monitor operates at physics frequency.

Exploratory reports are retained in `gripper_physics_baseline.json`, `gripper_physics_1ms_6d.json`, `gripper_contact_sweep.json`, `gripper_contact_sweep_extended.json` and `gripper_timestep_sweep.json` under `artifacts`. The passing report hashes the production scene, environment, rule monitor and validation script.

```powershell
wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_physics.py --gpu
.\scripts\view-audit.ps1 -Trial contact
```

Full push–turn–park feasibility, shortening the gripper-only path and full-path CPU/GPU behavior remain separate checks. The passing contact suite alone does not authorize a claim of training readiness.

## Follow-up checks and current path status

The current settings also pass **37 automated rule/preflight tests**, **5,620 geometric reset checks**, and **148 one-second settling episodes**. Evidence is in `gripper_rules_tests.xml` and `gripper_reset_validation.json` under `artifacts`.

The diagnostic planner now uses the low jaw-mesh footprint to choose contact offsets and searches short withdrawal moves when it cannot change contact side. The latest 1 ms attempt (`gripper_feedback_1ms.json/.npz`) aligns the block and reaches the gate, but stalls after 18.9 simulated seconds, with its center near x=216.36 mm and y=-6.47 mm. It does not achieve complete passage or parking, and must not be presented as a successful full demonstration. Next work is to correct repositioning and centering at the narrow gate, then independently replay a complete trajectory and shorten it. No policy learning was used.

Independent replay (`gripper_partial_path_audit.json`) also detects brief jaw–gate contact up to **0.094 mm**, beyond the 0.05 mm invalid-contact tolerance. This attempted path is therefore **invalid as well as incomplete**. The diagnostic planner has been corrected to check post-integration contacts, matching the independent audit, and to stop execution immediately on an invalid selected move. That planner correction still needs a new complete-path search. It does not change the production environment or invalidate the passing contact regression suite.
