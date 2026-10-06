"""Native reference for the GPU-terminal-pose settling isolation."""
import json,math
import numpy as np
import mujoco
from so101_m1.scene import load_model,solve_ik,RESET_XY,ARTIFACTS

results=[]
z=np.load(ARTIFACTS/'full_replay_warp_single_nominal.npz')
for away in [False,True]:
    m=load_model();d=mujoco.MjData(m);d.qpos[:]=z['qpos'][-1]
    if away:d.qpos[:6]=solve_ik(m,RESET_XY)[0]
    d.ctrl[:]=d.qpos[:6];d.mocap_pos[:]=z['mocap_pos'];d.mocap_quat[:]=z['mocap_quat'];mujoco.mj_forward(m,d)
    speeds=[]
    for i in range(10000):
        mujoco.mj_step(m,d)
        if i>=5000:speeds.append([np.linalg.norm(d.qvel[6:9]),np.linalg.norm(d.qvel[9:12])])
    speeds=np.array(speeds);hold=best=0
    for row in speeds:
        hold=hold+1 if row[0]<=.005 and row[1]<=math.radians(5) else 0
        best=max(best,hold)
    results.append(dict(arm_away=away,finite=bool(np.isfinite(d.qpos).all()),last_5s_max_linear=float(speeds[:,0].max()),last_5s_max_angular=float(speeds[:,1].max()),longest_velocity_hold_seconds=best*.001))
(ARTIFACTS/'isolation_native_settling.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
