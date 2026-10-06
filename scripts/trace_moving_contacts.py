"""Millisecond GPU contact/solver trace; continues after failure for diagnosis."""
import argparse,collections,hashlib,json,math
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT,GOAL_X
from replay_sensitivity import clean

@wp.kernel
def state_record(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),a:wp.array2d(dtype=float),f:wp.array2d(dtype=float),ctrl:wp.array2d(dtype=float),flags:wp.array(dtype=int),iterations:wp.array(dtype=int),nefc:wp.array(dtype=int),hold:wp.array(dtype=int),out:wp.array3d(dtype=float),t:int):
    w=wp.tid()
    for j in range(13):out[t,w,j]=q[w,j]
    for j in range(12):
        out[t,w,13+j]=v[w,j];out[t,w,25+j]=a[w,j];out[t,w,37+j]=f[w,j]
    for j in range(6):out[t,w,49+j]=ctrl[w,j]
    out[t,w,55]=float(flags[w]);out[t,w,56]=float(iterations[w]);out[t,w,57]=float(nefc[w]);out[t,w,58]=float(hold[w])

@wp.kernel
def contact_record(n:wp.array(dtype=int),world:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),dist:wp.array(dtype=float),address:wp.array2d(dtype=int),force:wp.array2d(dtype=float),out:wp.array3d(dtype=float),t:int):
    i=wp.tid();out[t,i,0]=-1.
    if i<n[0]:
        w=world[i];g=geom[i];adr=address[i,0]
        out[t,i,0]=float(w);out[t,i,1]=float(g[0]);out[t,i,2]=float(g[1]);out[t,i,3]=dist[i]
        out[t,i,4]=0.
        if w>=0 and adr>=0 and adr<force.shape[1]:out[t,i,4]=force[w,adr]

def main():
    p=argparse.ArgumentParser();p.add_argument('--plane',action='store_true');p.add_argument('--worlds',type=int,default=3);p.add_argument('--tag',default='');p.add_argument('--reset-failed',action='store_true');args=p.parse_args()
    if args.plane:
        from table_candidate import enable
        enable()
    torch.set_num_threads(1);wp.init();stem='moving_trace_'+('plane' if args.plane else 'box')+('_resets' if args.reset_failed else '')+args.tag
    with wp.ScopedDevice('cuda:0'),torch.cuda.stream(wp.stream_to_torch('cuda:0')):
        e=PushTurnParkEnv(EnvConfig(num_envs=args.worlds,stage=5,training=False));z=np.load(ARTIFACTS/'gripper_complete_path.npz')
        e.d.qpos[:,6:]=torch.tensor([.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)],device=e.device)
        for name in ['mocap_pos','mocap_quat']:getattr(e.d,name)[:]=torch.tensor(z[name],device=e.device,dtype=e.dtype)
        e.goal[:]=torch.tensor([GOAL_X,0,math.pi/2],device=e.device);e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.arange(e.num_envs,device=e.device))
        initial={k:dict(spread=float((getattr(e.d,k)-getattr(e.d,k)[0]).abs().max()),max_abs=float(getattr(e.d,k).abs().max())) for k in ['qpos','qvel','qacc_warmstart','ctrl','site_xpos']}
        d=e.sim.wp_data;out=wp.zeros((50,e.num_envs,59),dtype=float,device=e.device);con=wp.zeros((50,d.naconmax,5),dtype=float,device=e.device)
        with wp.ScopedCapture() as capture:
            for j in range(50):
                mjw.step(e.sim.wp_model,d);e.rules.launch()
                wp.launch(state_record,dim=e.num_envs,inputs=[d.qpos,d.qvel,d.qacc,d.qfrc_constraint,d.ctrl,e.rules.wp_flags,d.solver_niter,d.nefc,e.rules.wp_hold,out,j],device=e.device)
                wp.launch(contact_record,dim=d.naconmax,inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,d.contact.efc_address,d.efc.force,con,j],device=e.device)
            mjw.forward(e.sim.wp_model,d);e.rules.launch(advance=False)
        states=[];events=[];seen=set();ring=collections.deque(maxlen=10);first_nonfinite=None;first_success=[None]*e.num_envs;reset_events=[]
        def describe_contacts(c,w):
            chosen=c[c[:,0]==w];chosen=chosen[np.argsort(chosen[:,3])]
            return [dict(pair=[e.model.geom(int(r[1])).name,e.model.geom(int(r[2])).name],distance_m=float(r[3]),normal_constraint_force_n=float(r[4])) for r in chosen[:20]]
        for frame in range(600):
            action=z['actions'][frame] if frame<len(z['actions']) else np.zeros(2)
            e._control(torch.tensor(action,device=e.device,dtype=e.dtype)[None].repeat(e.num_envs,1));wp.capture_launch(capture.graph);wp.synchronize()
            s=out.numpy();c=con.numpy();states.append(s);ring.append(c)
            for w in range(e.num_envs):
                flags=s[:,w,55].astype(int)
                if w not in seen and np.any(flags):
                    j=int(np.flatnonzero(flags)[0]);seen.add(w)
                    event=dict(world=w,seconds=frame*.05+(j+1)*.001,flags=int(flags[j]),qpos=s[j,w,:13].tolist(),qvel=s[j,w,13:25].tolist(),qacc=s[j,w,25:37].tolist(),solver_iterations=float(s[j,w,56]),contacts=describe_contacts(c[j],w))
                    events.append(event);print(json.dumps(clean(event),allow_nan=False),flush=True)
                if first_success[w] is None and np.any((s[:,w,58]>=1000)&(flags==0)):
                    j=int(np.flatnonzero((s[:,w,58]>=1000)&(flags==0))[0]);first_success[w]=frame*.05+(j+1)*.001
            bad=np.argwhere(~np.isfinite(s[:,:,:49]).all(axis=-1))
            if len(bad):
                j,w=map(int,bad[0]);first_nonfinite=dict(world=w,seconds=frame*.05+(j+1)*.001,contacts=describe_contacts(c[max(0,j-1)],w),previous_state=s[max(0,j-1),w].tolist())
                print('NONFINITE '+json.dumps(clean(first_nonfinite),allow_nan=False),flush=True);break
            if args.reset_failed:
                ids=(e.rules.flags!=0).nonzero().flatten()
                if len(ids):
                    remaining=torch.ones(e.num_envs,device=e.device,dtype=torch.bool);remaining[ids]=False
                    before={k:getattr(e.d,k)[remaining].clone() for k in ['qpos','qvel','qacc_warmstart','ctrl']}
                    e.reset(ids);wp.synchronize()
                    delta={k:float((getattr(e.d,k)[remaining]-v).abs().max()) if v.numel() else 0 for k,v in before.items()}
                    reset_events.append(dict(seconds=(frame+1)*.05,ids=ids.cpu().tolist(),unreset_world_max_abs_change=delta))
        data=np.concatenate(states);np.savez_compressed(ARTIFACTS/f'{stem}_detail.npz',states=data,last_contact_chunks=np.array(ring),dt=.001)
        for w in range(e.num_envs):
            q=data[49::50,w,:13];q=q[np.isfinite(q).all(axis=1)]
            np.savez(ARTIFACTS/f'{stem}_{w}.npz',qpos=q,dt=.05,mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'],end_effector='closed_gripper')
        report=dict(plane_candidate=args.plane,worlds=e.num_envs,reset_failed_worlds=args.reset_failed,reset_events=reset_events,initial=initial,first_failures=events,first_nonfinite=first_nonfinite,first_success_seconds=first_success,simulated_seconds=len(data)*.001,final_flags=e.rules.flags.cpu().tolist(),scope='Every physics step; diagnostic continuation after failure or optional reset. Success applies only before the first safety failure/reset.',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/trace_moving_contacts.py',ROOT/'scripts/table_candidate.py',ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py']})
        (ARTIFACTS/f'{stem}.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False));print(json.dumps(clean(report),allow_nan=False),flush=True)

if __name__=='__main__':main()
