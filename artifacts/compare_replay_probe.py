import sys,numpy as np,mujoco
sys.path.insert(0,'src')
from so101_m1.scene import load_model
m=load_model();d=mujoco.MjData(m)
a=np.load('artifacts/full_replay_native_nominal_nominal.npz')['qpos'];b=np.load('artifacts/full_replay_warp_single_nominal.npz')['qpos']
for n in [0,19,39,59,69,79,88,89]:
 if n<min(len(a),len(b)):print('time',(n+1)*.05,'q max',abs(a[n,:6]-b[n,:6]).max(),'block mm',1000*np.linalg.norm(a[n,6:8]-b[n,6:8]))
d.qpos[:]=b[-1];mujoco.mj_forward(m,d)
print('FINAL CONTACTS',[(m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist)) for c in d.contact if c.dist<-.00005])
for i in range(3):
 q=np.load(f'artifacts/full_replay_warp_replicas_3_nominal_{i}.npz')['qpos']
 print('world',i,'firstjointdiff',abs(q[0,:6]-b[0,:6]).max(),'at3s',abs(q[59,:6]-b[59,:6]).max())
