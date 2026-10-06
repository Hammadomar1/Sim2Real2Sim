"""Search a tighter final parking push with the actual gripper contacts."""
import math,json
import numpy as np
import torch
from table_candidate import enable
enable()
from finish_parking import ParkingPlanner
from so101_m1.scene import ARTIFACTS

class ConsistentPlanner(ParkingPlanner):
    def __init__(self):
        super().__init__()
        self.d.qpos[9:13]=[math.cos(math.pi/4),0,0,math.sin(math.pi/4)]
        self.e.sim.forward()
        self.e.goal[0]=torch.tensor([.27,0,math.pi/2],dtype=self.e.dtype)
        self.e.rules.reset(torch.tensor([0]))
    def step(self,a,record=False):
        # Match production stepping. An extra mj_forward inside every substep
        # can change solver state and therefore is not a passive inspection.
        self.e._control(torch.tensor([np.asarray(a).tolist()],dtype=self.e.dtype))
        for _ in range(self.e.decimation):
            self.e.sim.step();self.e.rules.native()
            self.pen=max(self.pen,max([-float(c.dist) for c in self.d.contact]+[0.]))
        self.e.sim.forward();self.e.rules.native(advance=False)
        self.bad |= bool(self.e.rules.flags[0])
        self.commands.append(np.asarray(a).copy())
        if record:
            self.frames.append(self.d.qpos.copy());self.actions.append(np.asarray(a).copy())
            xy,yaw=self.pose();self.trace.append(dict(time=self.d.time,xy=xy.tolist(),yaw=yaw,bad=self.bad,pen=self.pen))

def main():
    p=ConsistentPlanner();p.stem='gripper_precision_path';z=np.load(ARTIFACTS/'gripper_complete_path.npz')
    for a in z['actions'][:492]:
        p.step(a,True)
        if p.bad:raise RuntimeError(f'Prefix failed at {p.d.time}: flags {int(p.e.rules.flags[0])}, pose {p.pose()}')
    initial=p.save();best=None
    for edge in [3,4,5,2,0,1]:
        for f in [.5,.35,.65,.2,.8]:
            p.restore(initial);p.commands=[]
            point,normal=p.geometry(edge,f);route=p.route(point-p.support(normal)+normal*.010)
            if route is None or not all(p.move(t,maxsteps=100) for t in route):continue
            approach=p.save();approach_commands=p.commands.copy()
            for theta in [-.9,-1.1,-.7,-.5,-.3,0.,.3,.6,.9]:
                p.restore(approach);p.commands=approach_commands.copy()
                for n in range(65):
                    p.push(edge,f,theta)
                    if p.bad or p.pen>.001 or p.d.time>28.8:break
                    pe,ae=p.error()
                    if pe<.010 and ae<math.radians(12) and n%2==0:
                        candidate=p.save();commands=p.commands.copy()
                        for _ in range(25):p.step([0,0])
                        pe,ae=p.error();score=pe+.02*ae
                        if not p.bad and p.pen<.001 and (best is None or score<best[0]):
                            best=(score,p.commands.copy(),dict(edge=edge,fraction=f,theta=theta,steps=n+1,pos=pe,angle=ae,time=p.d.time))
                            print('BEST',best[2],flush=True)
                        if not p.bad and p.pen<.001 and pe<.002 and ae<math.radians(5) and p.d.time<30:
                            final=p.commands.copy();p.restore(initial)
                            for a in final:p.step(a,True)
                            p.decisions.append(best[2]);p.write('precision_candidate');return
                        p.restore(candidate);p.commands=commands
            print('SEARCH',edge,f,'best',None if best is None else best[2],flush=True)
    if best:
        p.restore(initial)
        for a in best[1]:p.step(a,True)
        p.decisions.append(best[2]);p.write('best_precision_candidate')
    else:raise RuntimeError('No precision candidate')

if __name__=='__main__':main()
