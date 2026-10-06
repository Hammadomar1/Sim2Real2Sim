"""Diagnostic receding-horizon pushes with object-pose feedback at 20 Hz. No RL."""
import argparse,json,math,time
import numpy as np
import torch
from plan_path import Planner,angle
from so101_m1.scene import ARTIFACTS,BLOCK_VERTICES,PHYSICS_TIMESTEP


class FeedbackPlanner(Planner):
    def __init__(self,dt):
        super().__init__()
        self.e.cfg.timestep=dt;self.e.model.opt.timestep=dt;self.e.decimation=round(.05/dt)
        self.e.sim.forward()
        self.decisions=[];self.phase=0
        self.stem='gripper_feedback_2ms' if dt==.002 else 'gripper_feedback_1ms'

    def support(self,normal):
        """Contact offset of the actual jaw meshes relative to the control site."""
        points=[];m=self.e.model
        for i in range(m.ngeom):
            if m.geom(i).name.startswith(('fixed_jaw_mesh','moving_jaw_mesh')):
                mid=m.geom_dataid[i]
                local=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                world=local@self.d.geom_xmat[i].reshape(3,3).T+self.d.geom_xpos[i]
                points.extend(world[world[:,2]<.019,:2])
        footprint=np.array(points)-self.d.site_xpos[self.e.sid,:2]
        projection=footprint@normal
        return footprint[projection<projection.min()+.0003].mean(0)

    def move(self,p,record=False,maxsteps=120):
        for _ in range(maxsteps):
            delta=np.array(p)-self.d.site_xpos[self.e.sid,:2]
            if np.linalg.norm(delta)<.0008:return True
            self.step(np.clip(delta/.004,-1,1),record)
            if self.bad:return False
        return False

    def geometry(self,edge,f):
        a,b=BLOCK_VERTICES[edge],BLOCK_VERTICES[(edge+1)%6]
        tangent=(b-a)/np.linalg.norm(b-a)
        normal=np.array([tangent[1],-tangent[0]])
        xy,yaw=self.pose();c,s=math.cos(yaw),math.sin(yaw);rot=np.array([[c,-s],[s,c]])
        return xy+rot@(a+f*(b-a)),rot@normal

    def push(self,edge,f,theta,record=False):
        # Recompute contact point and direction from the CURRENT object pose.
        point,normal=self.geometry(edge,f)
        tangent=np.array([-normal[1],normal[0]])
        direction=-normal*math.cos(theta)+tangent*math.sin(theta)
        tip=self.d.site_xpos[self.e.sid,:2]
        lateral=float((point-self.support(normal)+normal*.0005-tip)@tangent)
        velocity=.03*direction+6*lateral*tangent
        velocity*=min(1.,.04/max(np.linalg.norm(velocity),1e-9))
        self.step(velocity/.04,record)

    def generate(self):
        for edge in range(6):
            fractions=[.15,.5,.85]
            if self.phase==1 and edge==5:fractions=[.5,.643,.8]
            for f in fractions:
                point,normal=self.geometry(edge,f)
                route=self.route(point-self.support(normal)+normal*.010)
                if route is not None:
                    for theta in [-.65,0,.65]:yield route,edge,f,theta

    def update_phase(self):
        xy,yaw=self.pose()
        if self.phase==0 and abs(yaw)<.2 and abs(xy[1])<.010:self.phase=1
        rot=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
        if self.phase==1 and (BLOCK_VERTICES@rot+xy)[:,0].min()>self.gate+.008:self.phase=2

    def cost(self,phase):
        if phase!=0:return super().cost(phase)
        xy,yaw=self.pose()
        # Alignment may translate the block toward the gate. Penalizing all
        # forward motion made physically useful turn-and-push actions stall.
        return abs(xy[1])+.06*abs(angle(yaw))+.2*max(0.,xy[0]-.19)

    def finished(self):
        xy,yaw=self.pose()
        return self.phase==2 and np.linalg.norm(xy-[.27,0])<.006 and abs(angle(yaw-math.pi/2))<math.radians(6)

    def retreat(self):
        """Search a short non-gripping withdrawal before changing contact side."""
        state=self.save();xy,yaw=self.pose();tip=self.d.site_xpos[self.e.sid,:2].copy();best=None
        for theta in np.linspace(-math.pi,math.pi,16,endpoint=False):
            self.restore(state);target=tip+.020*np.array([math.cos(theta),math.sin(theta)])
            if not self.move(target,maxsteps=40) or self.bad or self.pen>.001:continue
            after,ang=self.pose();dist=np.linalg.norm(after-xy)+.03*abs(angle(ang-yaw))
            separation=np.linalg.norm(self.d.site_xpos[self.e.sid,:2]-after)
            score=dist-.1*separation
            if best is None or score<best[0]:best=(score,target)
        self.restore(state)
        if best is None:return False
        self.move(best[1],True,maxsteps=40)
        self.decisions.append(dict(reposition=True,seconds=float(self.d.time),target=best[1].tolist()))
        return not self.bad

    def write(self,status):
        np.savez(ARTIFACTS/(self.stem+'.npz'),qpos=self.frames,actions=self.actions,mocap_pos=self.d.mocap_pos.copy(),mocap_quat=self.d.mocap_quat.copy(),dt=.05,end_effector='closed_gripper')
        (ARTIFACTS/(self.stem+'.json')).write_text(json.dumps(dict(status=status,timestep=self.e.cfg.timestep,seconds=float(self.d.time),pose=[*self.pose()[0],self.pose()[1]],phase=self.phase,decisions=self.decisions,trace=self.trace),indent=2))

    def run_feedback(self,iterations):
        for _ in range(4):self.step([0,0],True)
        for iteration in range(iterations):
            self.update_phase()
            if self.finished():break
            state=self.save();initial=self.cost(self.phase);best=None
            candidates=list(self.generate())
            for route,edge,f,theta in candidates:
                self.restore(state)
                if not all(self.move(p,maxsteps=100) for p in route):continue
                for n in range(1,61):
                    self.push(edge,f,theta)
                    if self.bad or self.pen>.001:break
                    xy,_=self.pose()
                    if xy[0]>.3 or xy[0]<.12 or abs(xy[1])>.10:break
                    progress=initial-self.cost(self.phase)
                    duration=self.d.time-state[1]
                    score=progress/(duration+.4)
                    if progress>.0005 and (best is None or score>best[0]):
                        best=(score,route,edge,f,theta,n,progress,duration)
            self.restore(state)
            if best is None:
                if self.retreat():
                    self.write('repositioning');print('REPOSITION',self.d.time,self.pose(),flush=True);continue
                self.write('stalled');print('STALLED',self.phase,self.d.time,self.pose(),flush=True);return False
            _,route,edge,f,theta,n,progress,duration=best
            for p in route:
                if not self.move(p,True):self.write('execution_failed');return False
            for _ in range(n):
                self.push(edge,f,theta,True)
                if self.bad or self.pen>.001:self.write('invalid');return False
            # No fixed dwell between pushes: the next decision uses measured pose.
            self.decisions.append(dict(phase=self.phase,edge=edge,fraction=f,theta=theta,push_steps=n,seconds=float(self.d.time),planned_progress=progress,planned_duration=duration,route=[p.tolist() for p in route]))
            self.write('running')
            print('FEEDBACK',iteration,'phase',self.phase,'time',round(self.d.time,2),'pose',self.pose(),'contact',(edge,f,theta,n),flush=True)
            if self.bad: self.write('invalid');return False
        for _ in range(30):self.step([0,0],True)
        self.write('finished' if self.finished() else 'incomplete')
        print('FINAL',self.finished(),self.d.time,self.pose(),flush=True)
        return self.finished()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--timestep',type=float,default=PHYSICS_TIMESTEP);parser.add_argument('--iterations',type=int,default=18);args=parser.parse_args()
    if not FeedbackPlanner(args.timestep).run_feedback(args.iterations):raise SystemExit(2)
