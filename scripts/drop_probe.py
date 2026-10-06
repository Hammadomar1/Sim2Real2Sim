from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
import argparse
parser=argparse.ArgumentParser(); parser.add_argument('--solver',default='Newton');parser.add_argument('--cone',default='elliptic');parser.add_argument('--impratio',type=float,default=10);parser.add_argument('--iterations',type=int,default=100);parser.add_argument('--ls-iterations',type=int,default=50);args=parser.parse_args()
from so101_m1.scene import build_scene,solve_ik
m=mujoco.MjModel.from_xml_path(str(build_scene(Path(__file__).resolve().parents[1]/'artifacts/probe_native.xml')))
d=mujoco.MjData(m)
m.opt.iterations=args.iterations;m.opt.ls_iterations=args.ls_iterations
m.opt.solver=getattr(mujoco.mjtSolver,'mjSOL_'+args.solver.upper())
m.opt.cone=getattr(mujoco.mjtCone,'mjCONE_'+args.cone.upper());m.opt.impratio=args.impratio
q,_,_=solve_ik(m,[.12,-.055]);d.qpos[:6]=q;d.ctrl[:]=q
d.qpos[6:13]=[.15,0,.0092,np.sqrt(.5),0,0,np.sqrt(.5)]
for name in ['table','block_0','block_1','pusher','gate_north','gate_south']:
    m.geom(name).solref[:]=[.008,1]
    m.geom(name).solimp[:]=[.99,.999,.001,.5,2]
mujoco.mj_forward(m,d)
for i in range(500):
    mujoco.mj_step(m,d)
    if i in [25,100,499]: print(i, d.qpos[6:9],d.qvel[6:])
