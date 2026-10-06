"""Verify diagnostic full-task actions with production rules and episode timing."""
import json,hashlib,math
import numpy as np
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT,GOAL_X

def replay(z,limit):
    e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5,episode_seconds=limit))
    d=e.sim.mj_data
    d.qpos[6:13]=[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]
    d.mocap_pos[:]=z['mocap_pos'];d.mocap_quat[:]=z['mocap_quat']
    e.goal[0]=torch.tensor([GOAL_X,0.,math.pi/2],dtype=e.dtype)
    e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.tensor([0]))
    for a in np.vstack([z['actions'],np.zeros((40,2))]):
        _,_,done,_=e.step(torch.tensor([a.tolist()],dtype=e.dtype))
        if done.any():return dict(episode_limit=limit,**e.last_terminal[0])
    return dict(episode_limit=limit,success=False,reason='No terminal event')

def main():
    torch.set_num_threads(1)
    source=ARTIFACTS/'gripper_complete_path.npz';z=np.load(source)
    nominal=replay(z,30.)
    out=dict(backend='native',nominal=nominal,within_30_seconds=bool(nominal['success']))
    if not nominal['success'] and nominal.get('timeout'):
        out['extended_diagnostic']=replay(z,max(60.,len(z['actions'])*.05+3))
    out['physically_valid_complete_path']=bool(out.get('extended_diagnostic',nominal)['success'])
    out['input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py',ROOT/'scripts/validate_complete_env.py']}
    (ARTIFACTS/'gripper_complete_env_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
    if not out['physically_valid_complete_path']:raise SystemExit(2)
if __name__=='__main__':main()
