"""Compare identical worlds at controller/physics boundaries, without resets."""
import argparse,json,math,hashlib
import numpy as np
import torch
import warp as wp
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hold','drive'],default='hold');p.add_argument('--no-warmstart',action='store_true');p.add_argument('--box-jaws',action='store_true');a=p.parse_args()
    if a.box_jaws:
        import so101_m1.env as module
        original=module.load_model
        def diagnostic_model():
            m=original()
            for i in range(m.ngeom):
                if m.geom(i).name.startswith(('fixed_jaw_mesh','moving_jaw_mesh')):m.geom_contype[i]=0;m.geom_conaffinity[i]=0
            return m
        module.load_model=diagnostic_model
    torch.set_num_threads(1);wp.init()
    with wp.ScopedDevice('cuda:0'),torch.cuda.stream(wp.stream_to_torch('cuda:0')):
        e=PushTurnParkEnv(EnvConfig(num_envs=3,stage=5,training=False))
        z=np.load(ARTIFACTS/'gripper_complete_path_audit.npz');actions=np.load(ARTIFACTS/'gripper_complete_path.npz')['actions']
        q=z['qpos'][-1] if a.mode=='hold' else np.r_[e.q_initial,[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]]
        e.d.qpos[:]=torch.tensor(q,device=e.device,dtype=e.dtype);e.d.qvel[:]=0;e.d.qacc_warmstart[:]=0;e.d.ctrl[:]=e.d.qpos[:,:6]
        for name in ['mocap_pos','mocap_quat']:getattr(e.d,name)[:]=torch.tensor(z[name],device=e.device,dtype=e.dtype)
        e.sim.forward();e.d.qacc_warmstart[:]=0;e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.arange(3,device=e.device))
        initial={k:float((getattr(e.d,k)-getattr(e.d,k)[0]).abs().max()) for k in ['qpos','qvel','qacc_warmstart','ctrl','site_xpos']}
        frames=[];logs=[];first_difference=None
        for f in range(600 if a.mode=='drive' else 200):
            action=actions[f] if a.mode=='drive' and f<len(actions) else np.zeros(2)
            e._control(torch.tensor(action,device=e.device,dtype=e.dtype)[None].repeat(3,1))
            ctrl=e.d.ctrl.cpu().numpy().copy();jac=e.jp.cpu().numpy().copy()
            if a.no_warmstart:e.d.qacc_warmstart[:]=0
            wp.capture_launch(e.control_graph);wp.synchronize()
            q=e.d.qpos.cpu().numpy().copy();v=e.d.qvel.cpu().numpy().copy()
            spread=float(np.max(np.abs(q-q[0])))
            if spread>1e-7 and first_difference is None:first_difference=(f+1)*.05
            logs.append(dict(seconds=(f+1)*.05,qpos_spread=spread,ctrl_spread=float(np.max(np.abs(ctrl-ctrl[0]))),jac_spread=float(np.max(np.abs(jac-jac[0]))),linear=np.linalg.norm(v[:,6:9],axis=-1).tolist(),angular=np.linalg.norm(v[:,9:12],axis=-1).tolist(),flags=e.rules.flags.cpu().tolist()))
            frames.append(q)
            if not np.isfinite(q).all() or not np.isfinite(v).all():break
        stem=f'isolation_controller_{a.mode}_cold{a.no_warmstart}'+('_boxjaws' if a.box_jaws else '')
        out=dict(mode=a.mode,box_jaws_diagnostic_only=a.box_jaws,clear_warmstart_every_control_step=a.no_warmstart,initial_spread=initial,first_qpos_difference_over_1e_7=first_difference,finite=bool(np.isfinite(q).all() and np.isfinite(v).all()),logs=logs,source_sha256=hashlib.sha256((ROOT/'scripts/isolate_controller.py').read_bytes()).hexdigest())
        from replay_sensitivity import clean
        (ARTIFACTS/f'{stem}.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False))
        for i in range(3):np.savez(ARTIFACTS/f'{stem}_{i}.npz',qpos=np.array(frames)[:,i],dt=.05,mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'],end_effector='closed_gripper')
        print(json.dumps(clean({k:v for k,v in out.items() if k!='logs'})));print(json.dumps(clean(logs[-1])))

if __name__=='__main__':main()
