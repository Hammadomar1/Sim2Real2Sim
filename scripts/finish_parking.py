"""Search the final push and stationary hold, from a continuous rotated prefix."""
import math
import numpy as np
from feedback_path import FeedbackPlanner,angle
from so101_m1.scene import ARTIFACTS

class ParkingPlanner(FeedbackPlanner):
    def __init__(self):
        super().__init__(.001);self.commands=[];self.stem='gripper_complete_path';self.phase=2
    def step(self,a,record=False):
        super().step(a,record);self.commands.append(np.asarray(a).copy())
    def error(self):
        xy,yaw=self.pose();return np.linalg.norm(xy-[.27,0]),abs(angle(yaw-math.pi/2))

def main():
    p=ParkingPlanner();z=np.load(ARTIFACTS/'gripper_parking_prefix.npz')
    for a in z['actions'][:492]:p.step(a,True)
    assert not p.bad
    initial=p.save();print('PREFIX',p.d.time,p.pose(),flush=True)
    best=1.
    for edge in [3,4,5,2,0,1]:
        for f in [.2,.35,.5,.65,.8,.9]:
            p.restore(initial);p.commands=[]
            point,normal=p.geometry(edge,f);route=p.route(point-p.support(normal)+normal*.010)
            if route is None or not all(p.move(t,maxsteps=100) for t in route):continue
            approach=p.save();approach_commands=p.commands.copy()
            for theta in [-.9,-.65,-.4,-.15,.1,.35,.65,.9]:
                p.restore(approach);p.commands=approach_commands.copy()
                for n in range(60):
                    p.push(edge,f,theta)
                    if p.bad or p.pen>.001:break
                    pos,ang=p.error();best=min(best,pos+.04*ang)
                    if pos<.012 and ang<math.radians(12):
                        candidate=p.save();commands=p.commands.copy()
                        for _ in range(35):p.step([0,0])
                        pe,ae=p.error()
                        if not p.bad and p.pen<.001 and pe<.0095 and ae<math.radians(9):
                            final_commands=p.commands.copy();p.restore(initial)
                            for a in final_commands:p.step(a,True)
                            p.decisions.append(dict(edge=edge,fraction=f,theta=theta,push_steps=n+1,settle_steps=35))
                            p.write('parking_candidate');print('PARKED',p.d.time,p.pose(),p.error(),flush=True);return
                        p.restore(candidate);p.commands=commands
            print('SEARCH',edge,f,'best',best,flush=True)
    print('NO SAFE PARKING CANDIDATE',best,flush=True);raise SystemExit(2)
if __name__=='__main__':main()
