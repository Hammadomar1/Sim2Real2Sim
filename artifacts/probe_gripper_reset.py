import sys
sys.path.insert(0,'src')
from so101_m1.scene import *
import mujoco,math
m=load_model(); rng=np.random.default_rng(7)
for xy in [[.12,-.055],[.12,-.06],[.125,-.06],[.13,-.065],[.14,-.07],[.13,-.08],[.14,-.09]]:
 q,err,_=solve_ik(m,xy);d=mujoco.MjData(m);d.qpos[:6]=q;worst=0;pair=None
 for i in range(1000):
  yaw=(math.pi/2 if i%2 else 0)+rng.uniform(-.3,.3)
  d.qpos[6:13]=[rng.uniform(.142,.158),rng.uniform(-.02,.02),.0092,math.cos(yaw/2),0,0,math.sin(yaw/2)];mujoco.mj_forward(m,d)
  for c in d.contact:
   if c.dist< -worst:worst=-c.dist;pair=(m.geom(c.geom1).name,m.geom(c.geom2).name)
 print(xy,err,worst,pair,q,flush=True)
