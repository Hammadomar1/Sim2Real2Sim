import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from .env import PushTurnParkEnv,EnvConfig
from .scene import ARTIFACTS

def diagnose(kind):
    env=PushTurnParkEnv(EnvConfig(num_envs=1,backend="native",stage=5,training=False))
    start=env.state()[2].clone()
    trace=[]
    for i in range(200):
        a=torch.zeros((1,2))
        if kind=="push": a[0,0]=.5
        obs,reward,done,_=env.step(a)
        contacts=[(env.model.geom(c.geom1).name,env.model.geom(c.geom2).name,float(c.dist)) for c in env.sim.mj_data.contact if c.dist<-.0005]
        if i%20==0: trace.append({"step":i,"tip":env.state()[2][0].tolist(),"block":env.d.qpos[0,6:13].tolist(),"contacts":contacts,"terminal":env.last_terminal})
    result={"kind":kind,"start":start.tolist(),"trace":trace}
    (ARTIFACTS/f"diagnostic_{kind}.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
