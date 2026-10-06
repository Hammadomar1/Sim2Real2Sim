"""Replay the gate checkpoint through production step(), including failure rules."""
import json,hashlib,math
import numpy as np
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT,BLOCK_CORNERS,GATE_X,GOAL_X

def main():
    torch.set_num_threads(1)
    e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5))
    source=ARTIFACTS/'gripper_gate_reposition.npz';z=np.load(source);d=e.sim.mj_data
    d.qpos[6:13]=[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]
    d.mocap_pos[:]=z['mocap_pos'];d.mocap_quat[:]=z['mocap_quat'];e.goal[0]=torch.tensor([GOAL_X,0.,math.pi/2],dtype=e.dtype)
    e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.tensor([0]))
    terminal=[]
    for a in np.vstack([z['actions'],np.zeros((40,2))]):
        _,_,done,_=e.step(torch.tensor([a.tolist()],dtype=e.dtype))
        if done.any():terminal=e.last_terminal;break
    vertices=BLOCK_CORNERS@d.xmat[e.block].reshape(3,3).T+d.qpos[6:9]
    result=dict(backend='native',timestep=e.cfg.timestep,seconds=float(d.time),terminal=terminal,passed_gate=bool(e.passed[0]),failure_flags=int(e.rules.flags[0]),gate_clearance_m=float(vertices[:,0].min()-GATE_X-.006),final_pose=[*e.state()[0][0].tolist(),float(e.state()[1][0])],input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    result['pass']=not terminal and result['passed_gate'] and result['failure_flags']==0 and result['gate_clearance_m']>0
    result['full_task_success']=False
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py',ROOT/'scripts/validate_gate_env.py']}
    (ARTIFACTS/'gripper_gate_env_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['pass']:raise SystemExit(2)
if __name__=='__main__':main()
