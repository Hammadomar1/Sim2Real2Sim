"""Search gripper withdrawal and rear-face pushes from a continuous valid prefix."""
import argparse,json,math
import numpy as np
from feedback_path import FeedbackPlanner,angle
from so101_m1.scene import ARTIFACTS,BLOCK_VERTICES

class GatePlanner(FeedbackPlanner):
    def __init__(self):
        super().__init__(.001);self.commands=[];self.stem='gripper_gate_reposition'
    def step(self,a,record=False):
        super().step(a,record);self.commands.append(np.asarray(a).copy())
    def cost(self,phase):
        xy,yaw=self.pose()
        return abs(xy[0]-.259)+4*abs(xy[1])+.08*abs(angle(yaw))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',type=int,default=140);parser.add_argument('--continue-push',action='store_true');args=parser.parse_args()
    if args.continue_push:raise SystemExit(0 if continue_push() else 2)
    p=GatePlanner();z=np.load(ARTIFACTS/'gripper_gate_prefix_source.npz')
    for a in z['actions'][:args.prefix]:p.step(a,True)
    assert not p.bad
    state=p.save();best=None
    # First move back from the barrier; this may push the block backward
    # through its inner face. Then withdraw to the open upper side.
    for x in [.19,.18,.17,.16]:
        for y in [-.005,.005,.015,.025,.035]:
            p.restore(state);p.commands=[]
            if not p.move([x,y],maxsteps=65) or p.pen>.001:continue
            if not p.move([x,.040],maxsteps=65) or p.pen>.001:continue
            escape=p.save();escape_commands=p.commands.copy()
            for f in [.35,.5,.643,.8]:
                p.restore(escape);p.commands=escape_commands.copy()
                point,normal=p.geometry(5,f);route=p.route(point-p.support(normal)+normal*.010)
                if route is None or not all(p.move(t,maxsteps=80) for t in route):continue
                approach=p.save();approach_commands=p.commands.copy()
                for theta in [-.3,0,.3]:
                    p.restore(approach);p.commands=approach_commands.copy()
                    for n in range(80):
                        p.push(5,f,theta)
                        if p.bad or p.pen>.001:break
                        score=p.cost(1)
                        if best is None or score<best[0]:
                            best=(score,p.commands.copy(),dict(escape=[x,y],fraction=f,theta=theta,steps=n+1,pose=[*p.pose()[0],p.pose()[1]]))
            print('SEARCH',x,y,'best',None if best is None else best[0:1]+(best[2],),flush=True)
    p.restore(state)
    if best is None:p.write('no_valid_reposition');raise SystemExit(2)
    for a in best[1]:p.step(a,True)
    p.decisions.append(best[2]);p.phase=1;p.write('reposition_candidate')
    print('RESULT',best[2], 'bad',p.bad,'seconds',p.d.time,flush=True)
    if p.bad:raise SystemExit(2)

def continue_push():
    p=GatePlanner();z=np.load(ARTIFACTS/'gripper_gate_escape.npz')
    for a in z['actions']:p.step(a,True)
    assert not p.bad
    for iteration in range(8):
        state=p.save();initial=p.cost(1);best=None
        for f in [.3,.4,.5,.6,.7,.8]:
            for theta in [-.45,-.25,0.,.25,.45]:
                p.restore(state);p.commands=[]
                for n in range(80):
                    p.push(5,f,theta)
                    if p.bad or p.pen>.001:break
                    # A push is only accepted as passage if it can withdraw
                    # and settle safely; a moving endpoint can hide a collision.
                    if p.pose()[0][0]>.249:
                        endpoint=p.save();prefix=p.commands.copy()
                        for brake in [[-1,0],[-.5,.5],[0,1],[0,0]]:
                            p.restore(endpoint);p.commands=prefix.copy()
                            for _ in range(8):p.step(brake)
                            for _ in range(30):p.step([0,0])
                            xy,yaw=p.pose();rot=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
                            clear=(BLOCK_VERTICES@rot+xy)[:,0].min()>p.gate+.0065
                            if clear and not p.bad and p.pen<.001:
                                commands=p.commands.copy();p.restore(state)
                                for a in commands:p.step(a,True)
                                p.phase=2;p.decisions.append(dict(fraction=f,theta=theta,steps=n+1,withdrawal=brake));p.write('gate_passage_candidate')
                                print('PASSAGE',p.pose(),p.d.time,flush=True);return not p.bad
                        p.restore(endpoint);p.commands=prefix
                    score=p.cost(1)
                    if best is None or score<best[0]:best=(score,p.commands.copy(),dict(fraction=f,theta=theta,steps=n+1))
        p.restore(state)
        if best is None or best[0]>=initial-.0002:
            p.write('rear_push_stalled');print('STALLED',p.pose(),flush=True);return False
        for a in best[1]:p.step(a,True)
        p.decisions.append(best[2]);p.phase=1;p.write('rear_pushing')
        xy,yaw=p.pose();rot=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
        full=(BLOCK_VERTICES@rot+xy)[:,0].min()>p.gate+.008
        print('PUSH',iteration,p.pose(),'full',full,flush=True)
    p.write('search_budget_exhausted');return False
if __name__=='__main__':main()
