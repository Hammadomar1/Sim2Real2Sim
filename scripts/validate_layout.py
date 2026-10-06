"""Geometric reset audit and demonstrated action replay through actual task step()."""
import argparse,hashlib,json,math
import numpy as np
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig,block_vertices
from so101_m1.scene import ARTIFACTS,ROOT,GATE_X,GOAL_X,solve_ik

def audit(backend,n,batches):
    e=PushTurnParkEnv(EnvConfig(num_envs=n,backend=backend,training=False,seed=20261004))
    count=0;max_overlap=0.;ik_error=0.
    for stage in range(1,6):
        e.cfg.stage=stage
        for batch in range(batches):
            e.rules.flags[:]=127;e.rules.hold[:]=600;e.rules.mask[:]=65535;e.rules.passed[:]=1
            e.velocity[:]=1;e.previous_action[:]=1;e.d.qvel[:]=1
            obs=e.reset()['actor'];count+=n
            assert torch.isfinite(obs).all()
            assert torch.allclose(torch.linalg.vector_norm(e.d.qpos[:,9:13],dim=-1),torch.ones(n,device=e.device,dtype=e.dtype),atol=1e-6)
            for t in [e.d.qvel,e.velocity,e.previous_action,e.rules.flags,e.rules.hold,e.rules.mask,e.rules.passed]:assert not t.any()
            assert torch.allclose(e.target_xy,e.d.site_xpos[:,e.sid,:2])
            assert torch.allclose(e.d.mocap_pos[:,e.goal_mocap,:2],e.goal[:,:2])
            assert torch.allclose(e.d.mocap_pos[:,e.gate_mocap,0],torch.full((n,),GATE_X if stage>=4 else 5.,device=e.device,dtype=e.dtype))
            xy,yaw,_=e.state();v=block_vertices(xy,yaw)
            assert (v[:,:,0]>.08).all() and (v[:,:,0]<GATE_X-.006).all() and (v[:,:,1].abs()<.14).all()
            if stage>=4:
                gv=block_vertices(e.goal[:,:2],e.goal[:,2]);assert (gv[:,:,0]>GATE_X+.006).all()
            # Reconstruct every sampled reset in standard MuJoCo for overlap inspection.
            for i in range(n):
                d=e.snapshot(i)
                max_overlap=max(max_overlap,max([-float(c.dist) for c in d.contact]+[0.]))
            if batch==0:
                for point in [xy[0].cpu().numpy(),e.goal[0,:2].cpu().numpy()]:
                    _,err,_=solve_ik(e.model,point);ik_error=max(ik_error,err);assert err<.001
    assert max_overlap<1e-6,(backend,max_overlap)
    # Rule history must survive a state round trip, and subset reset must preserve neighbors.
    e.rules.hold[:]=123;e.rules.mask[:]=7
    saved=e.state_dict();e.reset();e.load_state_dict(saved)
    assert (e.rules.hold==123).all() and (e.rules.mask==7).all()
    if n>1:
        before=e.state_dict();ids=torch.arange(0,n,2,device=e.device);keep=torch.arange(1,n,2,device=e.device)
        e.reset(ids)
        after=e.state_dict()
        for group in ['task','sim','rules']:
            for key in before[group]:assert torch.equal(before[group][key][keep],after[group][key][keep]),(group,key)
        assert not e.rules.hold[ids].any() and not e.rules.mask[ids].any()
    settling_episodes=0
    for _ in range(20 if n==1 else 1):
        e.reset();settling_episodes+=n
        for _ in range(20):
            obs,_,done,_=e.step(torch.zeros((n,2),device=e.device))
            assert torch.isfinite(obs['actor']).all() and not done.any(),e.last_terminal
    return dict(backend=backend,resets=count,max_initial_overlap_m=max_overlap,sampled_center_ik_error_m=ik_error,subset_reset_pass=True if n>1 else None,state_roundtrip_pass=True,one_second_settling_episodes=settling_episodes)

def replay(dt,stem):
    e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5,timestep=dt))
    z=np.load(ARTIFACTS/(stem+'.npz'));d=e.sim.mj_data
    if 'end_effector' not in z or str(z['end_effector'])!='closed_gripper':
        raise RuntimeError('Retired attachment demonstration cannot validate the gripper-only environment.')
    d.qpos[6:13]=[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]
    d.mocap_pos[:]=z['mocap_pos'];d.mocap_quat[:]=z['mocap_quat'];e.goal[0]=torch.tensor([GOAL_X,0.,math.pi/2],dtype=e.dtype)
    e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.tensor([0]))
    for a in np.vstack([z['actions'],np.zeros((40,2))]):
        _,_,done,_=e.step(torch.tensor([a.tolist()],dtype=e.dtype))
        if done.any():return dict(timestep=dt,input=stem,**e.last_terminal[0])
    return dict(timestep=dt,success=False,reason='no terminal event')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resets-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    out={'scope':'Geometric reset randomization; no physics randomization or policy success-rate claim.'}
    out['resets']=[audit('native',1,100),audit('warp',128,8)]
    print(json.dumps(out),flush=True)
    out['demonstration_replays']=[] if args.resets_only else [replay(.002,'feedback_path_2ms'),replay(.001,'feedback_path_1ms')]
    out['pass']=bool(out['resets']) and all(r.get('success') and not r.get('failed') for r in out['demonstration_replays'])
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py',ROOT/'scripts/validate_layout.py']}
    out['full_path_tested']=not args.resets_only
    name='gripper_reset_validation.json' if args.resets_only else 'layout_validation.json'
    (ARTIFACTS/name).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
    if not out['pass']:raise SystemExit(2)
if __name__=='__main__':main()
