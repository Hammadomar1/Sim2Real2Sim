# Bringing up the real SO-101 (Milestone 2)

The bridge in `s2r2s/hardware/` runs the trained policy on the real arm with the *same* controller and
observation code as training. Every step below was rehearsed on a deliberately miscalibrated simulated
arm (`--robot sim`). The real-arm commands are identical with `--robot so101 --port COMx`.

| Rehearsal on the simulated arm (offsets 4/−6/3/8°, elbow reversed) | Result |
|---|---|
| Policy without calibration | 0 % success (50 mm / 113° error) |
| Touch-point calibration | Signs found, offsets within 0.03° (rod length known), 0.2 mm residual |
| Policy with the calibrated map | 100 % success (10/10), 1.2 mm / 2.3° |
| Full hardware path, correct map (20 scenes) | 95 % success, matching the 96.6 % in simulation |

## 0. Install (once)

```powershell
cd D:\Sim2Real2Sim\s2r2s
uv sync --extra hardware          # adds LeRobot 0.6 (SO-101 driver) and the RealSense SDK
```

## 1. LeRobot setup (once per arm)

```powershell
.\.venv\Scripts\lerobot-find-port.exe          # which COM port is the follower arm
.\.venv\Scripts\lerobot-calibrate.exe --robot.type=so101_follower --robot.port=COM5 --robot.id=s2r2s_follower
```

Use the id `s2r2s_follower`: the bridge connects with it. During LeRobot's calibration, move every joint
through its **full** range. LeRobot puts 0° at the middle of that range, so a partial sweep shifts the zero.
Step 2 measures whatever offset remains.

Then mount the pusher rod: hold it between the jaws and close the gripper on it. The bridge holds the gripper
wherever it is when it connects. Measure the rod (length below the jaw tips, diameter); if it differs from
the model (30 mm, Ø12 mm), set `pusher_tip_z` / `pusher_radius` in `SceneConfig` (`s2r2s/scene.py`).

## 2. Joint mapping: touch nine points

```powershell
python -m s2r2s.hardware.calibrate --robot so101 --port COM5 --verify --no-fit-rod
```

- Mark the nine points the tool prints on the table. They are in the robot base frame: x forward, y to the robot's
  left, origin **38.8 mm behind the shoulder-pan axis** (the centre of the rotating base). Measure from that axis.
- Torque turns off, so **support the arm**. For each point, rest the rod tip on the mark (rod roughly vertical) and
  press Enter. The fit tries all 16 joint-direction combinations and solves the offsets.
- **Accept** if the RMS is below 2 mm and no point is above 3 mm. `--verify` then hovers 15 mm above each mark
  so you can check by eye. The result is `calibration/joint_map.json`; commit it, one per arm.
- If you didn't measure the rod, drop `--no-fit-rod`: the fit then estimates the rod length too
  (tip positions stay accurate; individual elbow and wrist offsets become less certain).

## 3. Tracking test

```powershell
python -m s2r2s.hardware.tracking --robot so101 --port COM5 --map calibration/joint_map.json
```

The rod draws an 80 mm square 30 mm above the table at 40 mm/s. Ideal servos (simulation): lag 0 ms,
0.7 mm RMS. **On the real arm, record the lag and the error after removing it.** If the lag is above
~100 ms, or the joints lag by more than ~1°, the simulator's servo model (gains, delay randomisation in
`RandomizationConfig`) must be updated and the policy retrained before trusting results.

## 4. Camera (the Milestone 2 perception task)

Mount the D435i opposite the robot, about 40° down, near (0.62, 0, 0.38) m in the robot frame (the
`d435i` camera in the scene). Implement `CameraPoseSource` in `s2r2s/hardware/poses.py`. It must return the
object's (x, y, yaw) in the robot frame at about 30 Hz, accurate to **2 mm / 2°** with under **150 ms** delay
(`python -m s2r2s.sensitivity`). See ROADMAP.md, section 3, for the recommended pipeline. Clutter policies also
call `read_block()`: the same estimate for the 40 mm block that must stay where it is. Where the block is when a
trial starts is where it must still be at the end. The block's estimate also drives the controller's rod guard
(the rod is never commanded within 3 mm of the block), so it must be as good as the T's, and the two must never
be confused.

## 5. First policy runs

```powershell
python -m s2r2s.hardware.runner --checkpoint runs\tee_v1\best.pt --robot sim --episodes 20      # rehearsal
python -m s2r2s.hardware.runner --checkpoint runs\clutter_v7\best.pt --robot sim --episodes 20  # with the block
python -m s2r2s.hardware.runner --checkpoint runs\tee_v1\best.pt --robot so101 --port COM5 `
    --map calibration\joint_map.json --goal 0.20 0.05 90 --max-step-deg 2
```

Safety built into the runner:
- **Speed:** every tick, joint targets move at most `--max-step-deg` from the measured joints. LeRobot also caps them.
- **Workspace:** the rod stays inside the reachable ring, at its fixed height.
- **Lost object:** the arm holds still when the object hasn't been seen for 0.5 s.
- **Stopping:** Ctrl+C lifts the rod 4 cm and exits.

**Torque stays on after the program ends** (LeRobot's default would drop the arm). Support the arm before
powering it down. Keep a hand near the power switch during first runs, and start with `--max-step-deg 2`.

## 6. Trial logs (for Milestone 3)

Every real run writes `trials/<date_time>.json`: the goal, the final pose and errors, and per tick the joints,
joint targets, rod position, commanded position, camera pose (and block pose), action and gate stage. The failure-guided
sampler (Milestone 3) will read these files.
