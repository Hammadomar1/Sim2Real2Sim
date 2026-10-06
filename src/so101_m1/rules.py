"""Physics-step safety, passage and hold monitor; independent of reward shaping."""
import math
import numpy as np
import torch
import warp as wp
from .scene import BLOCK_CORNERS, BLOCK_BOXES, GATE_X, GATE_HALF

NONFINITE=1
TIPPED=2
OUTSIDE=4
SHORTCUT=8
COLLISION=16
PENETRATION=32
JOINT_LIMIT=64
OVERFLOW=128
FAILURE_NAMES={1:'nonfinite',2:'tipping_or_lifting',4:'outside_workspace',8:'invalid_gate_crossing',16:'invalid_robot_contact',32:'excessive_penetration',64:'joint_limit',128:'contact_capacity_overflow'}
CONTACT_TOLERANCE=.00005
PENETRATION_LIMIT=.001


@wp.kernel
def geometry(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),site:wp.array2d(dtype=wp.vec3),
             stage:wp.array(dtype=wp.int64),goal:wp.array2d(dtype=float),corners:wp.array(dtype=wp.vec3),
             lower:wp.array(dtype=float),upper:wp.array(dtype=float),previous:wp.array2d(dtype=wp.vec3),
             mask:wp.array(dtype=int),passed:wp.array(dtype=int),hold:wp.array(dtype=int),bad:wp.array(dtype=int),touch:wp.array(dtype=int),
             tip_id:int,gate:float,half:float):
    w=wp.tid(); flags=int(0)
    for j in range(q.shape[1]):
        if not wp.isfinite(q[w,j]):flags=flags|1
    for j in range(v.shape[1]):
        if not wp.isfinite(v[w,j]):flags=flags|1
    if flags!=0:
        wp.atomic_or(bad,w,flags);hold[w]=0;return
    pos=wp.vec3(q[w,6],q[w,7],q[w,8]);quat=wp.quat(q[w,10],q[w,11],q[w,12],q[w,9])
    up=wp.quat_rotate(quat,wp.vec3(0.,0.,1.))
    upright=up[2]>0.966 and wp.abs(pos[2]-0.009)<0.006
    if not upright:flags=flags|2
    lo=wp.vec3(100.,100.,100.);hi=wp.vec3(-100.,-100.,-100.)
    crossed=mask[w]
    for k in range(16):
        p=pos+wp.quat_rotate(quat,corners[k]);a=previous[w,k]
        lo=wp.min(lo,p);hi=wp.max(hi,p)
        if stage[w]>=4:
            if wp.length(p-a)>0.01:flags=flags|8
            if a[0]<=gate and p[0]>gate:
                y=a[1]+(p[1]-a[1])*(gate-a[0])/(p[0]-a[0])
                if wp.abs(y)>half+0.001:flags=flags|8
                crossed=crossed|(1<<k)
        previous[w,k]=p
    mask[w]=crossed
    full=lo[0]>gate+0.006
    if stage[w]>=4:
        if full and crossed!=65535:flags=flags|8
        if full and crossed==65535 and upright:passed[w]=1
    if lo[0]<0.080 or hi[0]>0.320 or lo[1]<-0.14 or hi[1]>0.14:flags=flags|4
    for j in range(6):
        if q[w,j]<lower[j]-0.001 or q[w,j]>upper[j]+0.001:flags=flags|64
    wp.atomic_or(bad,w,flags)
    yaw=wp.atan2(2.*(q[w,9]*q[w,12]+q[w,10]*q[w,11]),1.-2.*(q[w,11]*q[w,11]+q[w,12]*q[w,12]))
    err=yaw-goal[w,2];ang=wp.abs(wp.atan2(wp.sin(err),wp.cos(err)))
    distance=wp.length(wp.vec2(pos[0]-goal[w,0],pos[1]-goal[w,1]))
    linear=wp.length(wp.vec3(v[w,6],v[w,7],v[w,8]));angular=wp.length(wp.vec3(v[w,9],v[w,10],v[w,11]))
    good=distance<=0.010 and ang<=0.1745329252 and linear<=0.005 and angular<=0.0872664626 and upright
    if stage[w]>=4:good=good and passed[w]==1 and full
    if stage[w]==2:good=distance<=0.010 and linear<=0.005 and upright
    if stage[w]==1:good=touch[w]!=0 and upright
    if good and bad[w]==0:hold[w]=hold[w]+1
    else:hold[w]=0


@wp.kernel
def contacts(n:wp.array(dtype=int),world:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),dist:wp.array(dtype=float),
             allowed:wp.array2d(dtype=int),overflow:wp.array(dtype=int),bad:wp.array(dtype=int),touch:wp.array(dtype=int)):
    i=wp.tid()
    if i<bad.shape[0] and overflow[i]!=0:wp.atomic_or(bad,i,128)
    if i<n[0] and i<world.shape[0]:
        w=world[i];g=geom[i];depth=dist[i]
        if w>=0 and w<bad.shape[0]:
            if g[0]>=0 and g[1]>=0 and depth<=0.0:
                if allowed[g[0],g[1]]==2:wp.atomic_or(touch,w,1)
            if depth< -0.001:wp.atomic_or(bad,w,32)
            if depth< -0.00005:
                if g[0]<0 or g[1]<0:wp.atomic_or(bad,w,16)
                elif allowed[g[0],g[1]]==0:wp.atomic_or(bad,w,16)


class PhysicsRules:
    def __init__(self,env):
        self.e=env;self.gpu=env.cfg.backend=='warp'
        self.allowed=np.zeros((env.model.ngeom,env.model.ngeom),np.int32)
        jaw_names=[env.model.geom(i).name for i in range(env.model.ngeom) if env.model.geom(i).name.startswith(('fixed_jaw_','moving_jaw_'))]
        for block in ['block_0','block_1']:
            for other in ['table','gate_north','gate_south']+jaw_names:
                a,b=env.model.geom(block).id,env.model.geom(other).id
                self.allowed[a,b]=self.allowed[b,a]=2 if other in jaw_names else 1
        n=env.num_envs
        self.flags=torch.zeros(n,device=env.device,dtype=torch.int32)
        self.mask=torch.zeros_like(self.flags);self.passed=torch.zeros_like(self.flags);self.hold=torch.zeros_like(self.flags)
        self.touch=torch.zeros_like(self.flags)
        self.previous=torch.zeros((n,16,3),device=env.device,dtype=env.dtype)
        if self.gpu:
            self.wp_flags=wp.from_torch(self.flags);self.wp_mask=wp.from_torch(self.mask);self.wp_passed=wp.from_torch(self.passed);self.wp_hold=wp.from_torch(self.hold)
            self.wp_touch=wp.from_torch(self.touch)
            self.wp_previous=wp.from_torch(self.previous,dtype=wp.vec3)
            self.wp_corners=wp.array(BLOCK_CORNERS,dtype=wp.vec3,device=env.device)
            self.wp_lower=wp.array(env.model.jnt_range[:6,0],dtype=float,device=env.device)
            self.wp_upper=wp.array(env.model.jnt_range[:6,1],dtype=float,device=env.device)
            self.wp_allowed=wp.array(self.allowed,dtype=int,device=env.device)
            self.wp_stage=wp.from_torch(env.stage);self.wp_goal=wp.from_torch(env.goal)

    def reset(self,ids):
        for t in [self.flags,self.mask,self.passed,self.hold,self.touch]:t[ids]=0
        r=self.e.d.xmat[ids,self.e.block].reshape(-1,3,3)
        vertices=torch.as_tensor(BLOCK_CORNERS,device=self.e.device,dtype=self.e.dtype)
        self.previous[ids]=vertices[None]@r.transpose(1,2)+self.e.d.qpos[ids,None,6:9]

    def launch(self,advance=True):
        e=self.e;d=e.sim.wp_data
        self.wp_touch.zero_()
        wp.launch(contacts,dim=max(d.naconmax,e.num_envs),inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,self.wp_allowed,d.overflow,self.wp_flags,self.wp_touch],device=e.device)
        if advance:
            wp.launch(geometry,dim=e.num_envs,inputs=[d.qpos,d.qvel,d.site_xpos,self.wp_stage,self.wp_goal,self.wp_corners,self.wp_lower,self.wp_upper,self.wp_previous,self.wp_mask,self.wp_passed,self.wp_hold,self.wp_flags,self.wp_touch,e.sid,GATE_X,GATE_HALF],device=e.device)

    def inspect_contact(self,a,b,distance):
        if a>=0 and b>=0 and distance<=0 and self.allowed[a,b]==2:self.touch[0]=1
        if distance< -PENETRATION_LIMIT:self.flags[0]|=PENETRATION
        if distance< -CONTACT_TOLERANCE and (a<0 or b<0 or not self.allowed[a,b]):self.flags[0]|=COLLISION

    def native(self,advance=True):
        e=self.e;d=e.sim.mj_data
        self.touch[0]=0
        for c in d.contact:self.inspect_contact(c.geom1,c.geom2,float(c.dist))
        if sum(d.warning.number):self.flags[0]|=NONFINITE
        if not advance:return
        q,v=d.qpos,d.qvel
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            self.flags[0]|=NONFINITE;self.hold[0]=0;return
        # mj_step updates qpos after its kinematics; use current quaternion here.
        import mujoco
        r=np.empty(9);mujoco.mju_quat2Mat(r,q[9:13]);r=r.reshape(3,3)
        vertices=BLOCK_CORNERS@r.T+q[6:9];lo,hi=vertices.min(0),vertices.max(0)
        upright=r[2,2]>.966 and abs(q[8]-.009)<.006
        if not upright:self.flags[0]|=TIPPED
        stage=int(e.stage[0]);previous=self.previous[0].numpy()
        if stage>=4:
            if np.linalg.norm(vertices-previous,axis=1).max()>.01:self.flags[0]|=SHORTCUT
            for k,(a,b) in enumerate(zip(previous,vertices)):
                if a[0]<=GATE_X<b[0]:
                    y=a[1]+(b[1]-a[1])*(GATE_X-a[0])/(b[0]-a[0])
                    if abs(y)>GATE_HALF+.001:self.flags[0]|=SHORTCUT
                    self.mask[0]|=1<<k
            if lo[0]>GATE_X+.006:
                if int(self.mask[0])!=65535:self.flags[0]|=SHORTCUT
                elif upright:self.passed[0]=1
        self.previous[0]=torch.from_numpy(vertices)
        if lo[0]<.08 or hi[0]>.32 or lo[1]<-.14 or hi[1]>.14:self.flags[0]|=OUTSIDE
        if np.any(q[:6]<e.model.jnt_range[:6,0]-.001) or np.any(q[:6]>e.model.jnt_range[:6,1]+.001):self.flags[0]|=JOINT_LIMIT
        yaw=math.atan2(2*(q[9]*q[12]+q[10]*q[11]),1-2*(q[11]**2+q[12]**2));goal=e.goal[0].numpy()
        poserr=np.linalg.norm(q[6:8]-goal[:2]);yawerr=abs(math.atan2(math.sin(yaw-goal[2]),math.cos(yaw-goal[2])))
        linear=np.linalg.norm(v[6:9]);angular=np.linalg.norm(v[9:12])
        good=poserr<=.010 and yawerr<=math.radians(10) and linear<=.005 and angular<=math.radians(5) and upright
        if stage>=4:good=good and bool(self.passed[0]) and lo[0]>GATE_X+.006
        if stage==2:good=poserr<=.010 and linear<=.005 and upright
        if stage==1:good=bool(self.touch[0]) and upright
        self.hold[0]=self.hold[0]+1 if good and not self.flags[0] else 0

    def state_dict(self):return {k:getattr(self,k).clone() for k in ['flags','mask','passed','hold','previous','touch']}
    def load_state_dict(self,state):
        for k,v in state.items():getattr(self,k)[:]=v.to(self.e.device)
