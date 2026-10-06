"""Model-based diagnostic push planner, separate from RL. Searches real contact rollouts."""
import argparse,heapq,json,math,time
import numpy as np
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,BLOCK_VERTICES,BLOCK_BOXES

def angle(x):return math.atan2(math.sin(x),math.cos(x))

class Planner:
    def __init__(self,gate=.215,resume=None):
        torch.set_num_threads(1)
        self.e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=5,training=False))
        self.d=self.e.sim.mj_data;self.frames=[];self.actions=[];self.trace=[]
        self.d.qpos[6:13]=[.15,0,.0092,math.sqrt(.5),0,0,math.sqrt(.5)]
        self.gate=gate
        self.d.mocap_pos[self.e.gate_mocap,0]=gate
        self.d.mocap_pos[self.e.goal_mocap,:]=[gate+.055,0,.001]
        self.d.mocap_quat[self.e.goal_mocap]=[math.sqrt(.5),0,0,math.sqrt(.5)]
        self.e.sim.forward();self.e.target_xy[:]=self.e.d.site_xpos[:,self.e.sid,:2]
        self.bad=False;self.pen=0
        if resume:
            # Reconstruct the actual dynamic/controller state from one reset.
            # Never splice a saved terminal pose into the executed trajectory.
            z=np.load(ARTIFACTS/(resume+'.npz'))
            for action in z['actions']:self.step(action,True)
            print('CONTINUOUS PREFIX',self.d.time,self.pose(),flush=True)
    def pose(self):return self.d.qpos[6:8].copy(),float(self.e.state()[1][0])
    def save(self):return self.e.state_dict(),self.d.time
    def restore(self,state):self.e.load_state_dict(state[0]);self.d.time=state[1];self.bad=False;self.pen=0
    def step(self,a,record=False):
        self.e._control(torch.tensor([np.asarray(a).tolist()],dtype=torch.float64))
        for _ in range(self.e.decimation):
            self.e.sim.step()
            # Inspect post-integration contacts, matching the independent
            # path auditor; pre-step distances can miss a brief jaw/gate touch.
            self.e.sim.forward()
            for c in self.d.contact:
                if c.dist<-.00005:
                    names={self.e.model.geom(c.geom1).name,self.e.model.geom(c.geom2).name}
                    self.pen=max(self.pen,-float(c.dist))
                    allowed=bool(self.e.rules.allowed[c.geom1,c.geom2])
                    if not allowed:self.bad=True
        self.e.sim.forward()
        if self.d.xmat[self.e.block].reshape(3,3)[2,2]<.966 or abs(self.d.qpos[8]-.009)>.004:self.bad=True
        if record:
            self.frames.append(self.d.qpos.copy());self.actions.append(np.asarray(a).copy())
            xy,yaw=self.pose();self.trace.append(dict(time=self.d.time,xy=xy.tolist(),yaw=yaw,bad=self.bad,pen=self.pen))
    def move(self,p,record=False,maxsteps=200):
        for _ in range(maxsteps):
            delta=np.array(p)-self.d.site_xpos[self.e.sid,:2]
            if np.linalg.norm(delta)<.0008:return True
            self.step(np.clip(delta/.007,-1,1),record)
            if self.bad:return False
        return False
    def free(self,p,xy,yaw,padding=.009):
        if not (.111<=p[0]<=.294 and abs(p[1])<=.125):return False
        if abs(p[0]-self.gate)<.006+padding and abs(p[1])>.024-padding:return False
        c,s=math.cos(yaw),math.sin(yaw);local=np.array([[c,s],[-s,c]])@(np.array(p)-xy)
        for center,size in BLOCK_BOXES:
            delta=np.abs(local-np.array(center[:2]))-np.array(size[:2])
            dist=np.linalg.norm(np.maximum(delta,0))+min(max(delta),0)
            if dist<padding:return False
        return True
    def route(self,target):
        xy,yaw=self.pose();start=self.d.site_xpos[self.e.sid,:2].copy()
        # Visibility graph around a 2.5 mm grid; start can be at existing contact.
        scale=.0025
        def key(p):return tuple(np.round(np.array(p)/scale).astype(int))
        a,b=key(start),key(target)
        frontier=[(0.,a)];cost={a:0.};prev={}
        def free(p):return self.free(p,xy,yaw) or (np.linalg.norm(np.array(p)-start)<.01 and self.free(p,xy,yaw,padding=.0048))
        while frontier:
            _,u=heapq.heappop(frontier)
            if u==b:break
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                v=(u[0]+dx,u[1]+dy);p=np.array(v)*scale
                if not free(p):continue
                # Do not cut diagonally through the object or gate corners.
                if not free((np.array(u)+np.array(v))*scale/2) and u!=a:continue
                nc=cost[u]+math.hypot(dx,dy)
                if nc<cost.get(v,1e10):cost[v]=nc;prev[v]=u;heapq.heappush(frontier,(nc+math.dist(v,b),v))
        if b not in cost:return None
        path=[b]
        while path[-1]!=a:path.append(prev[path[-1]])
        pts=[start]+[np.array(p)*scale for p in path[::-1][1:]]+[np.array(target)]
        # Greedy line-of-sight simplification, preserving 1 mm clearance samples.
        out=[];i=0
        while i<len(pts)-1:
            j=len(pts)-1
            while j>i+1:
                n=max(2,int(np.linalg.norm(pts[j]-pts[i])/.001)+1)
                if all(free(p) for p in np.linspace(pts[i],pts[j],n)[1:]):break
                j-=1
            out.append(pts[j]);i=j
        return out
    def candidates(self):
        xy,yaw=self.pose();c,s=math.cos(yaw),math.sin(yaw);rot=np.array([[c,-s],[s,c]])
        result=[]
        for i,(a,b) in enumerate(zip(BLOCK_VERTICES,np.roll(BLOCK_VERTICES,-1,axis=0))):
            tangent=(b-a)/np.linalg.norm(b-a);outward=np.array([tangent[1],-tangent[0]])
            for f in [.2,.5,.8]:
                point=xy+rot@(a+f*(b-a));normal=rot@outward
                start=point+normal*.013
                route=self.route(start)
                if route is None:continue
                for distance in [.016,.029]:
                    for theta in [-.65,0,.65]:
                        direction=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])@(-normal)
                        result.append((route,start+direction*distance,(i,f,distance,theta)))
        return result
    def cost(self,phase):
        xy,yaw=self.pose()
        target=np.array([.155,0.]) if phase==0 else np.array([self.gate+.044,0.]) if phase==1 else np.array([self.gate+.055,0.])
        orientation=0 if phase<2 else math.pi/2
        return np.linalg.norm(xy-target)+.04*abs(angle(yaw-orientation))
    def run(self,iterations):
        phase=2 if self.pose()[0][0]>self.gate+.03 else 0
        for _ in range(10):self.step([0,0],True)
        for iteration in range(iterations):
            xy,yaw=self.pose()
            if phase==0 and abs(yaw)<.20 and abs(xy[1])<.004:phase=1
            verts=BLOCK_VERTICES@np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])+xy
            if phase==1 and min(verts[:,0])>self.gate+.008:phase=2
            if phase==2 and self.cost(phase)<.008:break
            state=self.save();initial=self.cost(phase);best=None
            candidates=self.candidates()
            for route,end,description in candidates:
                self.restore(state)
                valid=True
                for p in route+[end]:
                    if not self.move(p,maxsteps=110):valid=False;break
                for _ in range(3):self.step([0,0])
                score=self.cost(phase)+.00015*(self.d.time-state[1])
                if valid and not self.bad and self.pen<.001 and (best is None or score<best[0]):best=(score,route,end,description)
            self.restore(state)
            if best is None or best[0]>=initial-.0002:
                print('STALLED',iteration,phase,initial,'best',best[0] if best else None,'pose',self.pose(),flush=True);break
            for p in best[1]+[best[2]]:self.move(p,True)
            for _ in range(3):self.step([0,0],True)
            print('STEP',iteration,'phase',phase,'primitive',best[3],'score',best[0],'time',self.d.time,'pose',self.pose(),flush=True)
            np.savez(ARTIFACTS/'planned_path.npz',qpos=self.frames,mocap_pos=self.d.mocap_pos.copy(),mocap_quat=self.d.mocap_quat.copy(),dt=.05,actions=self.actions)
            (ARTIFACTS/'planned_path.json').write_text(json.dumps(self.trace,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=15);parser.add_argument('--gate',type=float,default=.215);parser.add_argument('--resume');args=parser.parse_args()
    Planner(args.gate,args.resume).run(args.iterations)
