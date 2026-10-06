"""Native Windows scene inspection. No trained policy is loaded."""
from pathlib import Path
import argparse
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import mujoco.viewer
import numpy as np
from so101_m1.scene import build_scene, solve_ik, RESET_XY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--replay", type=Path, help="Display recorded physics states; this is not a live policy")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--label", default="RECORDED CONTROLLER TRIAL")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(build_scene(ROOT / "artifacts/scene_windows.xml")))
    data = mujoco.MjData(model)
    q, error, _ = solve_ik(model, RESET_XY)
    if error > .001:
        raise RuntimeError(f"Initial tool IK error: {error}")

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = q
        data.ctrl[:] = q
        # Both initial and parking orientations require a turn at the gate.
        data.qpos[6:13] = [.150, 0, .009, np.sqrt(.5), 0, 0, np.sqrt(.5)]
        data.mocap_quat[model.body("goal").mocapid[0]] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        mujoco.mj_forward(model, data)

    def camera(cam):
        cam.lookat[:] = [.14, 0, .06]
        cam.distance = .62
        cam.azimuth = 135
        cam.elevation = -40

    reset()
    replay = np.load(args.replay) if args.replay else None
    if replay is not None and ('end_effector' not in replay or str(replay['end_effector'])!='closed_gripper'):
        raise RuntimeError('This recording used the retired pushing attachment. It cannot be replayed on the gripper-only model.')
    frame = args.frame
    if replay is not None:
        data.mocap_pos[:] = replay['mocap_pos']
        data.mocap_quat[:] = replay['mocap_quat']
        data.qpos[:] = replay['qpos'][frame % len(replay['qpos'])]
        mujoco.mj_forward(model, data)
    if args.screenshot:
        from PIL import Image
        cam = mujoco.MjvCamera()
        camera(cam)
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            renderer.update_scene(data, camera=cam)
            Image.fromarray(renderer.render()).save(args.screenshot)
        print(f"Rendered {args.screenshot}", flush=True)
        return

    events = queue.SimpleQueue()
    paused = replay is None
    print("SO-101 scene inspection: starts PAUSED; P runs/pauses physics; R resets. No trained policy.", flush=True)
    if replay is not None:
        print("RECORDED TRAJECTORY: loops measured physics states; no live physics during replay. Source: "+args.label, flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put) as viewer:
        camera(viewer.cam)
        if replay is not None:
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             args.label+'\nRecorded physics states\nP: pause/resume  R: reset',
                             args.replay.stem))
        viewer.sync()
        print("GUI READY", flush=True)
        while viewer.is_running():
            start = time.perf_counter()
            while not events.empty():
                key = events.get()
                if key == ord("P"):
                    paused = not paused
                    print("PAUSED" if paused else "Physics running: joint hold only", flush=True)
                elif key == ord("R"):
                    reset()
                    frame = 0
                    if replay is not None:
                        data.qpos[:] = replay['qpos'][0]
                        data.mocap_pos[:] = replay['mocap_pos']
                        data.mocap_quat[:] = replay['mocap_quat']
                        mujoco.mj_forward(model, data)
                    paused = True
            if not paused:
                if replay is not None:
                    data.qpos[:] = replay['qpos'][frame]
                    data.mocap_pos[:] = replay['mocap_pos']
                    data.mocap_quat[:] = replay['mocap_quat']
                    mujoco.mj_forward(model, data)
                    frame = (frame + 1) % len(replay['qpos'])
                else:
                    for _ in range(round(.01/model.opt.timestep)):
                        mujoco.mj_step(model, data)
            viewer.sync()
            interval = float(replay['dt']) if replay is not None else .01
            time.sleep(max(0, interval - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
