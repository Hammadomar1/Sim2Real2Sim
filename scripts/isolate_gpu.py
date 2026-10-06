"""Short controlled GPU settling tests; no training or production changes."""
import argparse, contextlib, hashlib, json, math
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from so101_m1.env import PushTurnParkEnv, EnvConfig
from so101_m1.scene import ARTIFACTS, ROOT

@wp.kernel
def record(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),out:wp.array3d(dtype=float),frame:int):
    w,j=wp.tid()
    out[frame,w,j]=q[w,j]
    if j<12:out[frame,w,13+j]=v[w,j]

def main():
    p=argparse.ArgumentParser();p.add_argument('--shared-stream',action='store_true');p.add_argument('--replay',action='store_true');p.add_argument('--plane-table',action='store_true');a=p.parse_args()
    if a.plane_table:
        import mujoco
        import so101_m1.env as module
        from so101_m1.scene import build_scene
        def diagnostic_model():
            spec=mujoco.MjSpec.from_file(str(build_scene()))
            table=spec.geom('table');table.type=mujoco.mjtGeom.mjGEOM_PLANE;table.pos[2]=0
            return spec.compile()
        module.load_model=diagnostic_model
    torch.set_num_threads(1);wp.init()
    with wp.ScopedDevice('cuda:0'):
        stream_context=torch.cuda.stream(wp.stream_to_torch(wp.get_stream())) if a.shared_stream else contextlib.nullcontext()
        with stream_context:
            if a.replay:
                from replay_sensitivity import run
                run('warp',[dict(name=f'nominal_{i}') for i in range(3)],'isolation_replay_'+('shared' if a.shared_stream else 'default'))
                return
            results=[]
            for case in ['start_block','park_block_arm_away','park_block_gripper_present','gpu_terminal_gripper_present','gpu_terminal_arm_away']:
                e=PushTurnParkEnv(EnvConfig(num_envs=3,backend='warp',stage=5,training=False))
                source=np.load(ARTIFACTS/'gripper_complete_path_audit.npz')
                q=np.array(source['qpos'][-1] if case=='park_block_gripper_present' else np.r_[e.q_initial,[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]])
                if case=='park_block_arm_away':q[6:]=source['qpos'][-1,6:]
                if case.startswith('gpu_terminal'):q=np.load(ARTIFACTS/'full_replay_warp_single_nominal.npz')['qpos'][-1].copy()
                if case=='gpu_terminal_arm_away':q[:6]=e.q_initial
                e.d.qpos[:]=torch.tensor(q,device=e.device,dtype=e.dtype)
                e.d.qvel[:]=0;e.d.qacc_warmstart[:]=0;e.d.ctrl[:]=e.d.qpos[:,:6]
                e.d.mocap_pos[:]=torch.tensor(source['mocap_pos'],device=e.device,dtype=e.dtype)
                e.d.mocap_quat[:]=torch.tensor(source['mocap_quat'],device=e.device,dtype=e.dtype)
                wp.synchronize();torch.cuda.synchronize();e.sim.forward();wp.synchronize()
                initial_spread=float((e.d.qpos-e.d.qpos[0]).abs().max())
                # Freeze actuator targets: no IK, action filtering, reward or reset.
                out=wp.zeros((10000,3,25),dtype=float,device=e.device)
                with wp.ScopedCapture() as capture:
                    for step in range(100):
                        mjw.step(e.sim.wp_model,e.sim.wp_data)
                        wp.launch(record,dim=(3,13),inputs=[e.sim.wp_data.qpos,e.sim.wp_data.qvel,out,step],device=e.device)
                chunks=[]
                for _ in range(100):
                    wp.capture_launch(capture.graph);wp.synchronize()
                    chunks.append(out.numpy()[:100].copy())
                values=np.concatenate(chunks);np.savez_compressed(ARTIFACTS/(f'isolation_{case}_{a.shared_stream}'+('_plane' if a.plane_table else '')+'.npz'),states=values,dt=.001)
                tail=values[-5000:];lin=np.linalg.norm(tail[:,:,19:22],axis=-1);ang=np.linalg.norm(tail[:,:,22:25],axis=-1)
                good=(lin<=.005)&(ang<=math.radians(5));holds=np.zeros(3);best=np.zeros(3)
                for row in good:holds=np.where(row,holds+1,0);best=np.maximum(best,holds)
                result=dict(case=case,plane_table_diagnostic_only=a.plane_table,shared_stream=a.shared_stream,initial_world_qpos_spread=initial_spread,finite=bool(np.isfinite(values).all()),last_5s_max_linear=lin.max(0).tolist(),last_5s_max_angular=ang.max(0).tolist(),longest_velocity_hold_seconds=(best*.001).tolist(),max_world_qpos_spread=float(np.max(np.abs(values[:,:,:13]-values[:,0:1,:13]))))
                results.append(result);print(json.dumps(result),flush=True)
            report=dict(results=results,scope='10 s frozen actuator targets; every 1 ms sampled. Hold checks speeds only, not full task success.',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/isolate_gpu.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/scene.py']})
            (ARTIFACTS/('isolation_settling_'+str(a.shared_stream)+('_plane' if a.plane_table else '')+'.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
