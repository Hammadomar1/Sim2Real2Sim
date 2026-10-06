"""Gripper-only contact checkpoint; no policy learning or object edits during motion."""
import json,hashlib,math
import numpy as np
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ROOT,ARTIFACTS,GRIPPER_CLOSED,solve_ik

def trial(dt):
    e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5,timestep=dt))
    d=e.sim.mj_data;d.qpos[6:13]=[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]
    e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.tensor([0]))
    frames=[];maxpen=0.;pairs=set();initial=d.qpos[6:8].copy();initialtip=d.site_xpos[e.sid].copy();hold_error=0.
    # Approach from the left and push with the real fingertip surfaces.
    points=[np.array([.12,.020]),np.array([.18,.020])];index=0
    for frame in range(320):
        a=np.zeros(2)
        if 60<=frame<260 and index<len(points):
            delta=points[index]-d.site_xpos[e.sid,:2];a=np.clip(delta/.01,-1,1)
            if np.linalg.norm(delta)<.001:index+=1
        e._control(torch.tensor([a.tolist()],dtype=e.dtype))
        for _ in range(e.decimation):
            e.sim.step();e.rules.native()
            for c in d.contact:
                if c.dist<0:
                    maxpen=max(maxpen,-float(c.dist));pairs.add(tuple(sorted([e.model.geom(c.geom1).name,e.model.geom(c.geom2).name])))
        e.sim.forward();e.rules.native(False);frames.append(d.qpos.copy())
        if frame==59:hold_error=float(np.linalg.norm(d.site_xpos[e.sid]-initialtip))
    r=dict(timestep=dt,seconds=float(d.time),failure_flags=int(e.rules.flags[0]),max_penetration_m=maxpen,hold_error_m=hold_error,block_displacement_m=float(np.linalg.norm(d.qpos[6:8]-initial)),final_block=d.qpos[6:8].tolist(),contacts=sorted(pairs),gripper_angle=float(d.qpos[5]),gripper_target=GRIPPER_CLOSED)
    r['pass']=r['failure_flags']==0 and maxpen<.001 and hold_error<.0005 and r['block_displacement_m']>.01
    if dt==.002:np.savez(ARTIFACTS/'gripper_push.npz',qpos=frames,mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy(),dt=.05,end_effector='closed_gripper')
    return r

def main():
    torch.set_num_threads(1)
    report={'trials':[trial(.002),trial(.001)]}
    report['pass']=all(r['pass'] for r in report['trials'])
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py',ROOT/'scripts/validate_gripper.py']}
    (ARTIFACTS/'gripper_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['pass']:raise SystemExit(2)
if __name__=='__main__':main()
