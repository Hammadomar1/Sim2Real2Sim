"""Real-robot bridge for Milestone 2.

Everything here runs the *same* controller and observation code as training; only the
sources of measurements differ:

    robots.py      SO101Robot (LeRobot, real arm) and SimRobot (MuJoCo, same interface)
    joint_map.py   LeRobot degrees  <->  MuJoCo radians (sign and offset per joint)
    poses.py       where the object pose comes from (simulated camera now, RealSense later)
    calibrate.py   touch known points with the rod tip -> fitted joint offsets/signs
    tracking.py    command a path, measure how well the rod tip follows it
    runner.py      run a trained policy in a 20 Hz loop, with safety limits and trial logs

Every tool works against SimRobot today, so the pipeline is tested before the arm is ready.
"""
