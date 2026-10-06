import json,time
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from .env import EnvConfig,PushTurnParkEnv
from .scene import ARTIFACTS

def camera():
    c=mujoco.MjvCamera(); c.lookat[:]=[.16,0,.06]; c.distance=.65; c.azimuth=135; c.elevation=-45
    return c

def scene_image():
    env=PushTurnParkEnv(EnvConfig(num_envs=1,backend="native",stage=5,training=False))
    with mujoco.Renderer(env.model,height=720,width=1280) as renderer:
        renderer.update_scene(env.snapshot(),camera=camera())
        path=ARTIFACTS/"scene.png"; imageio.imwrite(path,renderer.render())
    return path

def play(args):
    env=PushTurnParkEnv(EnvConfig(num_envs=1,backend="native",stage=args.stage,seed=args.seed,training=False))
    policy=None
    if args.checkpoint:
        from .training import load_policy
        policy=load_policy(args.checkpoint,env)
    renderer=mujoco.Renderer(env.model,height=720,width=1280) if args.video else None
    writer=imageio.get_writer(args.video,fps=20) if args.video else None
    handle=None
    if args.viewer:
        import mujoco.viewer
        handle=mujoco.viewer.launch_passive(env.model,env.sim.mj_data)
        handle.cam.lookat[:]=[.16,0,.06]; handle.cam.distance=.65; handle.cam.azimuth=135; handle.cam.elevation=-45
    records=[]
    try:
        while len(records)<args.episodes:
            start=time.monotonic()
            with torch.inference_mode(): action=policy(env.get_observations()) if policy else torch.zeros((1,2))
            env.step(action)
            records.extend(env.last_terminal)
            if renderer:
                renderer.update_scene(env.snapshot(),camera=camera()); writer.append_data(renderer.render())
            if handle:
                if not handle.is_running(): break
                handle.sync(); time.sleep(max(0,env.cfg.control_dt-(time.monotonic()-start)))
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        if handle: handle.close()
    if args.video: Path(args.video+".json").write_text(json.dumps({"checkpoint":args.checkpoint,"seed":args.seed,"episodes":records},indent=2))
    print(json.dumps(records,indent=2))
