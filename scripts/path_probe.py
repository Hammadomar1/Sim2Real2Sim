"""Diagnostic contact waypoint experiments; never used by the learned policy."""
import argparse,json,math
import numpy as np
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS

parser=argparse.ArgumentParser();parser.add_argument('--points',default='[[0.12,0.02],[0.145,0.02],[0.195,0.02]]');parser.add_argument('--load');parser.add_argument('--tag',default='path_probe');args=parser.parse_args()
torch.set_num_threads(1)
e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=5,training=False));d=e.sim.mj_data
if args.load:
    z=np.load(ARTIFACTS/(args.load+'.npz'));d.qpos[:]=z['qpos'][-1];d.qvel[:]=0;d.ctrl[:]=d.qpos[:6]
else:d.qpos[6:13]=[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]
e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2]
frames=[];trace=[];contacts=set();penetration=0
def step(a):
    global penetration
    e._control(torch.tensor([a],dtype=torch.float64))
    for _ in range(e.decimation):
        e.sim.step()
        for c in d.contact:
            if c.dist<0:
                contacts.add(tuple(sorted([e.model.geom(c.geom1).name,e.model.geom(c.geom2).name])))
                penetration=max(penetration,-float(c.dist))
    e.sim.forward();frames.append(d.qpos.copy())
    trace.append(dict(t=d.time,tip=d.site_xpos[e.sid].tolist(),block=d.qpos[6:9].tolist(),yaw=float(e.state()[1][0]),upright=float(d.xmat[e.block].reshape(3,3)[2,2])))
for _ in range(10):step([0,0])
for point in json.loads(args.points):
    for _ in range(150):
        delta=np.array(point)-d.site_xpos[e.sid,:2]
        step(np.clip(delta/.01,-1,1).tolist())
        if np.linalg.norm(delta)<.001:break
    for _ in range(10):step([0,0])
    print('WAYPOINT',point,trace[-1],flush=True)
np.savez(ARTIFACTS/(args.tag+'.npz'),qpos=frames,mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy(),dt=.05)
(ARTIFACTS/(args.tag+'.json')).write_text(json.dumps(dict(trace=trace,contacts=sorted(contacts),max_penetration=penetration),indent=2))
print('CONTACTS',contacts,'PENETRATION',penetration,flush=True)
