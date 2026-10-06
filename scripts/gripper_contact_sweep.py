"""Diagnose timestep sensitivity without changing production scene settings."""
import json
import numpy as np
import torch
from validate_physics import trial,compare
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS
torch.set_num_threads(1)
results=[]
for name,condim,timeconst,iters,ls in [('one_ms',6,.008,100,50),('half_ms',6,.008,100,50)]:
    dts=[.001,.0005] if name=='one_ms' else [.0005,.00025]
    worlds=[PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=5,training=False,timestep=dt)) for dt in dts]
    for e in worlds:
        e.model.opt.iterations=iters;e.model.opt.ls_iterations=ls
        for i in range(e.model.ngeom):
            if e.model.geom(i).name.startswith(('fixed_jaw_','moving_jaw_')):
                e.model.geom_condim[i]=condim;e.model.geom_solref[i,0]=timeconst
    for offset in [(0.,0.,0.),(0.,-.003,.04),(0.,.003,-.04),(.005,.055,-np.pi/2)]:
        kind='gate' if offset[1]==.055 else 'push'
        a,actions,_,_=trial(worlds[0],offset=offset,kind=kind);b,_,_,_=trial(worlds[1],offset=offset,actions=actions,kind=kind)
        r=dict(profile=name,offset=offset,comparison=compare(a,b),penetration=max(a['max_penetration_m'],b['max_penetration_m']),upright=min(a['min_upright'],b['min_upright']))
        results.append(r);print(json.dumps(r),flush=True)
        (ARTIFACTS/'gripper_timestep_sweep.json').write_text(json.dumps(results,indent=2))
