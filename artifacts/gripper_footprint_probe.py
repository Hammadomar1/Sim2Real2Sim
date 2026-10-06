import sys
sys.path.insert(0,'src')
from so101_m1.scene import *
import mujoco
m=load_model(); d=mujoco.MjData(m); d.qpos[:6]=solve_ik(m,RESET_XY)[0]; d.qpos[6:13]=[.15,0,.0092,1,0,0,0];mujoco.mj_forward(m,d)
for i in range(m.ngeom):
 if m.geom(i).name.startswith(('fixed_jaw_mesh','moving_jaw_mesh')):
  mid=m.geom_dataid[i];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
  low=v[v[:,2]<.020]
  if len(low): print(m.geom(i).name,len(low),(low[:,:2]-d.site_xpos[m.site('tool_tip').id,:2]).min(0),(low[:,:2]-d.site_xpos[m.site('tool_tip').id,:2]).max(0))
