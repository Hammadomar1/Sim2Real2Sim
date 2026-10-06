"""Pose-grid settling and finite-workspace rejection for the plane candidate."""
import json,math,hashlib
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from table_candidate import enable
from isolate_gpu import record
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT

def main():
    enable();torch.set_num_threads(1);wp.init()
    # Stage 3 removes the gate; isolate table contact without gate collisions.
    poses=[(x,y,yaw,tilt,z) for x,y in [(.16,.06),(.25,.07),(.25,-.07)] for yaw in np.arange(0,360,15) for tilt,z in [(0,.0092),(2,.011),(-2,.011)]]
    with wp.ScopedDevice('cuda:0'),torch.cuda.stream(wp.stream_to_torch('cuda:0')):
        e=PushTurnParkEnv(EnvConfig(num_envs=len(poses),stage=3,training=False))
        for i,(x,y,yaw,tilt,z) in enumerate(poses):
            a=math.radians(yaw)/2;b=math.radians(tilt)/2
            e.d.qpos[i,6:]=torch.tensor([x,y,z,math.cos(a)*math.cos(b),math.cos(a)*math.sin(b),math.sin(a)*math.sin(b),math.sin(a)*math.cos(b)],device=e.device)
        e.d.qvel[:]=0;e.d.qacc_warmstart[:]=0;e.d.ctrl[:]=e.d.qpos[:,:6]
        e.sim.forward();e.rules.reset(torch.arange(e.num_envs,device=e.device))
        out=wp.zeros((100,e.num_envs,25),dtype=float,device=e.device)
        with wp.ScopedCapture() as capture:
            for j in range(100):
                mjw.step(e.sim.wp_model,e.sim.wp_data);e.rules.launch()
                wp.launch(record,dim=(e.num_envs,13),inputs=[e.sim.wp_data.qpos,e.sim.wp_data.qvel,out,j],device=e.device)
        tail=[];finite=True
        for chunk in range(30):
            wp.capture_launch(capture.graph);wp.synchronize();values=out.numpy()
            finite &= bool(np.isfinite(values).all())
            if chunk>=20:tail.append(values.copy())
        tail=np.concatenate(tail);lin=np.linalg.norm(tail[:,:,19:22],axis=-1);ang=np.linalg.norm(tail[:,:,22:25],axis=-1)
        passed=(lin.max(0)<=.005)&(ang.max(0)<=math.radians(5))&(e.rules.flags.cpu().numpy()==0)
        settling=dict(cases=len(poses),passed=int(passed.sum()),finite=finite,max_last_second_linear=float(lin.max()),max_last_second_angular=float(ang.max()),failures=[dict(pose=poses[i],flags=int(e.rules.flags[i]),linear=float(lin[:,i].max()),angular=float(ang[:,i].max())) for i in range(len(poses)) if not passed[i]])
        # Original task bounds sit strictly inside the physical visual slab.
        # Include four workspace violations and all four physical table edges.
        boundary_poses=[(.10,0),(.30,0),(.20,-.13),(.20,.13),(-.20,0),(.48,0),(.20,-.29),(.20,.29)]
        b=PushTurnParkEnv(EnvConfig(num_envs=len(boundary_poses),stage=3,training=False))
        for i,(x,y) in enumerate(boundary_poses):b.d.qpos[i,6:]=torch.tensor([x,y,.0092,1,0,0,0],device=b.device)
        b.sim.forward();b.rules.reset(torch.arange(b.num_envs,device=b.device));b.rules.launch();wp.synchronize()
        boundary_flags=b.rules.flags.cpu().tolist();boundary_pass=all(f&4 for f in boundary_flags)
        native=[]
        for x,y in boundary_poses:
            n=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=3,training=False));n.d.qpos[0,6:]=torch.tensor([x,y,.0092,1,0,0,0],dtype=n.dtype)
            n.sim.forward();n.rules.reset(torch.tensor([0]));n.rules.native();native.append(int(n.rules.flags[0]))
        report=dict(settling=settling,boundary_poses=boundary_poses,gpu_boundary_flags=boundary_flags,native_boundary_flags=native,boundary_pass=boundary_pass and all(f&4 for f in native),scope='Candidate only: finite visual slab, infinite contact plane, existing stricter finite task boundary enforced. No physical edge-fall model.',ready_for_rl=False)
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/table_candidate.py',ROOT/'scripts/validate_table_candidate.py',ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/rules.py']}
        (ARTIFACTS/'table_candidate_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
