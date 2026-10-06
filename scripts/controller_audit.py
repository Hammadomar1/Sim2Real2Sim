"""Record real controller/physics trajectories, without reward or policy training."""
import json
import argparse
import numpy as np
import torch
import mujoco
from so101_m1.env import PushTurnParkEnv, EnvConfig
from so101_m1.scene import ARTIFACTS

parser = argparse.ArgumentParser()
parser.add_argument('--timestep', type=float, default=.002)
parser.add_argument('--contact', choices=['model','original','firm','firm4','firm12','ref8','moderate','slow'], default='model')
parser.add_argument('--tag', default=None)
parser.add_argument('--impratio', type=float, default=10)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--ls-iterations', type=int, default=50)
args = parser.parse_args()
env = PushTurnParkEnv(EnvConfig(num_envs=1, backend="native", stage=5, training=False, timestep=args.timestep))
d = env.sim.mj_data
env.model.opt.impratio=args.impratio
env.model.opt.iterations=args.iterations
env.model.opt.ls_iterations=args.ls_iterations
if args.contact == 'original':
    for name in ['table','block_0','block_1','pusher','gate_north','gate_south']:
        env.model.geom(name).solref[:] = [.01 if name.startswith('block') else .02,1]
        env.model.geom(name).solimp[:] = [.9,.95,.001,.5,2]
elif args.contact != 'model':
    for name in ['table','block_0','block_1','pusher','gate_north','gate_south']:
        geom = env.model.geom(name)
        geom.solref[:] = [dict(firm=.008,firm4=.004,firm12=.012,ref8=.008,moderate=.01,slow=.02)[args.contact],1]
        geom.solimp[:] = ([.9,.95,.001,.5,2] if args.contact=='ref8' else [.95,.99,.001,.5,2] if args.contact in ['moderate','slow'] else [.99,.999,.001,.5,2])
d.qpos[6:13] = [.150, 0, .0092, np.sqrt(.5), 0, 0, np.sqrt(.5)]
env.sim.forward()
frames, records = [], []
base_start = d.xpos[env.model.body("base").id].copy() if "base" in [env.model.body(i).name for i in range(env.model.nbody)] else d.xpos[1].copy()
tip_start = d.site_xpos[env.sid].copy()
max_penetration = 0.
worst_contact = None
contacts = set()

def step(action, label):
    global max_penetration, worst_contact
    env._control(torch.tensor([action], dtype=torch.float64))
    for _ in range(env.decimation):
        env.sim.step()
        for c in d.contact:
            names = (env.model.geom(c.geom1).name, env.model.geom(c.geom2).name)
            if c.dist < 0:
                contacts.add(tuple(sorted(names)))
                if -float(c.dist) > max_penetration:
                    max_penetration = -float(c.dist)
                    worst_contact = dict(geoms=names, time=float(d.time))
    env.sim.forward()
    frames.append(d.qpos.copy())
    xy, yaw, tip = env.state()
    records.append(dict(time=float(d.time), segment=label, tip=tip[0].tolist(), block=xy[0].tolist(), yaw=float(yaw[0]), upright=float(d.xmat[env.block].reshape(3,3)[2,2]), command_velocity=env.velocity[0].tolist()))

for _ in range(100): step([0, 0], "hold")
hold_error = float(np.linalg.norm(d.site_xpos[env.sid]-tip_start))
waypoints = [[.120,.020],[.145,.020],[.175,.020]]
for i, target in enumerate(waypoints):
    for _ in range(100):
        delta = np.array(target)-d.site_xpos[env.sid,:2]
        action = delta / .01
        step(action.tolist(), f"waypoint_{i}")
        if np.linalg.norm(delta)<.001: break
for _ in range(40): step([0, 0], "settle")
vel = np.array([r["command_velocity"] for r in records])
report = dict(timestep=args.timestep, contact=args.contact, solver_iterations=args.iterations, solver_ls_iterations=args.ls_iterations, impratio=args.impratio, hold_error_m=hold_error, max_command_speed=float(np.linalg.norm(vel,axis=1).max()), max_command_acceleration=float(np.linalg.norm(np.diff(vel,axis=0)/.05,axis=1).max()), max_penetration_m=max_penetration, worst_contact=worst_contact, contacts=sorted(contacts), final=records[-1], full_puzzle_demonstrated=False, trajectory=records)
stem = args.tag or ('controller_audit' if args.timestep == .002 else 'controller_audit_1ms')
(ARTIFACTS / f"{stem}.json").write_text(json.dumps(report,indent=2))
np.savez(ARTIFACTS / f"{stem}.npz", qpos=np.array(frames), mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(), dt=.05)
print(json.dumps({k:v for k,v in report.items() if k!="trajectory"},indent=2))
