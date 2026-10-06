"""Batched state-based task, using mjlab's GPU simulation and RSL-RL contract.

No planner or demonstration action is used by step(). The only task-space
controller is a velocity-limited DLS inverse-kinematics mapping.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
import mujoco
import warp as wp
import mujoco_warp as mjw
from tensordict import TensorDict
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
from .scene import load_model, solve_ik, TOOL_Z, GATE_X, GOAL_X, GATE_HALF, BLOCK_VERTICES, SOLVER_ITERATIONS, SOLVER_LS_ITERATIONS
from .rules import PhysicsRules, FAILURE_NAMES
from .scene import GRIPPER_CLOSED, RESET_XY, PHYSICS_TIMESTEP

@dataclass
class EnvConfig:
    num_envs: int = 512
    seed: int = 0
    stage: int = 1
    training: bool = True
    randomized: bool = False
    smoothness: bool = True
    backend: str = "warp"
    timestep: float = PHYSICS_TIMESTEP
    control_dt: float = .05
    episode_seconds: float = 30.
    speed_limit: float = .04
    acceleration_limit: float = .2

def wrap_angle(x):
    return torch.atan2(torch.sin(x),torch.cos(x))

def block_vertices(xy, yaw):
    v=torch.as_tensor(BLOCK_VERTICES,dtype=xy.dtype,device=xy.device)
    c,s=torch.cos(yaw)[:,None],torch.sin(yaw)[:,None]
    return torch.stack((c*v[:,0]-s*v[:,1],s*v[:,0]+c*v[:,1]),-1)+xy[:,None,:]

def pose_success(pos_error, yaw_error, linear_speed, angular_speed, upright):
    return (pos_error<=.010)&(yaw_error<=math.radians(10))&(linear_speed<=.005)&(angular_speed<=math.radians(5))&upright

class NativeSimulation:
    """Single-world standard MuJoCo, sharing task/controller code for audit."""
    def __init__(self, model):
        from types import SimpleNamespace
        self.mj_model=model
        self.mj_data=mujoco.MjData(model)
        self.data=SimpleNamespace()
        for name in ["qpos","qvel","qacc_warmstart","ctrl","mocap_pos","mocap_quat","site_xpos","xmat","xpos","xquat","time"]:
            val=getattr(self.mj_data,name)
            if name=="time": val=np.array([0.])
            setattr(self.data,name,torch.from_numpy(np.asarray(val)).unsqueeze(0))
    def reset(self, ids=None):
        mujoco.mj_resetData(self.mj_model,self.mj_data)
    def forward(self): mujoco.mj_forward(self.mj_model,self.mj_data)
    def step(self): mujoco.mj_step(self.mj_model,self.mj_data)

class PushTurnParkEnv:
    def __init__(self,cfg: EnvConfig):
        self.cfg=cfg
        self.num_envs=cfg.num_envs
        self.num_actions=2
        self.device="cuda:0" if cfg.backend=="warp" else "cpu"
        self.dtype=torch.float32 if cfg.backend=="warp" else torch.float64
        self.rng=torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.model=load_model()
        self.model.opt.timestep=cfg.timestep
        self.decimation=round(cfg.control_dt/cfg.timestep)
        assert abs(self.decimation*cfg.timestep-cfg.control_dt)<1e-8
        self.max_episode_length=round(cfg.episode_seconds/cfg.control_dt)
        self.sid=self.model.site("tool_tip").id
        self.gripper=self.model.body("gripper").id
        self.block=self.model.body("block").id
        self.pusher=self.model.geom("fixed_jaw_box3").id
        self.gate_mocap=self.model.body_mocapid[self.model.body("gate").id]
        self.goal_mocap=self.model.body_mocapid[self.model.body("goal").id]
        self.q_initial,err,axis=solve_ik(self.model,RESET_XY)
        assert err<.001 and axis[2]>.99,(err,axis)
        self.control_graph=None
        if cfg.backend=="warp":
            wp.init()
            self.sim=Simulation(cfg.num_envs,SimulationCfg(nconmax=96,njmax=256,mujoco=MujocoCfg(timestep=cfg.timestep,integrator="implicitfast",cone="elliptic",impratio=10,iterations=SOLVER_ITERATIONS,ls_iterations=SOLVER_LS_ITERATIONS)),model=self.model,device=self.device)
            self.sim.expand_model_fields(("geom_friction","body_mass","body_inertia","actuator_gainprm","actuator_biasprm"))
            self.default_fields={k:getattr(self.sim.model,k).clone() for k in ("geom_friction","body_mass","body_inertia","actuator_gainprm","actuator_biasprm")}
            with wp.ScopedDevice(self.device):
                self.jp_wp=wp.zeros((cfg.num_envs,3,self.model.nv),dtype=float)
                self.jr_wp=wp.zeros((cfg.num_envs,3,self.model.nv),dtype=float)
                self.point_wp=wp.zeros(cfg.num_envs,dtype=wp.vec3)
                self.body_wp=wp.full(cfg.num_envs,self.gripper,dtype=wp.int32)
                self.jp=wp.to_torch(self.jp_wp)
                self.jr=wp.to_torch(self.jr_wp)
                self.point=wp.to_torch(self.point_wp)
                # Instantiate child graph executables before nesting them.
                self.sim.step()
                self.sim.forward()
                wp.synchronize()
        else:
            assert cfg.num_envs==1,"Native audit backend uses one world"
            self.sim=NativeSimulation(self.model)
        self.d=self.sim.data
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.stage=torch.full_like(self.episode_length_buf,cfg.stage)
        self.phase=torch.zeros_like(self.stage)
        self.hold=torch.zeros_like(self.stage)
        self.entered=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.passed=self.entered.clone()
        self.goal=self.zeros(3)
        self.previous_action=self.zeros(2)
        self.velocity=self.zeros(2)
        self.previous_velocity=self.zeros(2)
        self.target_xy=self.zeros(2)
        self.delayed_action=self.zeros(2)
        self.delay_mask=self.entered.clone()
        self.command_scale=torch.ones((self.num_envs,1),device=self.device,dtype=self.dtype)
        self.episode_return=self.zeros()
        self.jerk_integral=self.zeros()
        self.prev_qvel=self.zeros(5)
        self.prev_acc=self.zeros(5)
        self.last_terminal=[]
        self.last_numerical_failure=None
        self.lower=torch.tensor(self.model.jnt_range[:5,0]+.02,device=self.device,dtype=self.dtype)
        self.upper=torch.tensor(self.model.jnt_range[:5,1]-.02,device=self.device,dtype=self.dtype)
        self.rules=PhysicsRules(self)
        self.reset()
        if cfg.backend=="warp":
            with wp.ScopedDevice(self.device):
                # Compile the monitors before capture, then reset their state.
                self.rules.launch();wp.synchronize()
                self.rules.reset(torch.arange(self.num_envs,device=self.device))
                with wp.ScopedCapture() as capture:
                    for _ in range(self.decimation):
                        mjw.step(self.sim.wp_model,self.sim.wp_data)
                        self.rules.launch()
                    mjw.forward(self.sim.wp_model,self.sim.wp_data)
                    self.rules.launch(advance=False)
                self.control_graph=capture.graph

    def zeros(self,*shape): return torch.zeros((self.num_envs,*shape),device=self.device,dtype=self.dtype)
    def rand(self,n,*shape): return torch.rand((n,*shape),device=self.device,dtype=self.dtype,generator=self.rng)

    def reset(self,ids=None,*,preserve_terminal=False):
        if not preserve_terminal:self.last_terminal=[]
        if ids is None: ids=torch.arange(self.num_envs,device=self.device)
        n=len(ids)
        if not n: return
        self.sim.reset(ids)
        stage=torch.full((n,),self.cfg.stage,device=self.device,dtype=torch.long)
        if self.cfg.training and self.cfg.stage>1:
            easier=self.rand(n)<.2
            stage=torch.where(easier,1+(self.rand(n)*(self.cfg.stage-1)).long(),stage)
        self.stage[ids]=stage
        self.d.qpos[ids,:6]=torch.tensor(self.q_initial,device=self.device,dtype=self.dtype)
        self.d.ctrl[ids,:6]=self.d.qpos[ids,:6]
        self.d.qvel[ids]=0
        self.d.qacc_warmstart[ids]=0
        xy=torch.stack((.150+(self.rand(n)-.5)*.016,(self.rand(n)-.5)*.04),-1)
        yaw=(self.rand(n)-.5)*.6
        yaw=torch.where(stage==5,math.pi/2+(self.rand(n)-.5)*.4,yaw)
        self.d.qpos[ids,6:8]=xy
        self.d.qpos[ids,8]=.0092
        self.d.qpos[ids,9:13]=0
        self.d.qpos[ids,9]=torch.cos(yaw/2)
        self.d.qpos[ids,12]=torch.sin(yaw/2)
        self.goal[ids,:2]=xy+torch.stack((.030+self.rand(n)*.010,(self.rand(n)-.5)*.025),-1)
        self.goal[ids,2]=torch.where(stage<=2,yaw,(self.rand(n)-.5)*2.4)
        later=stage>=4
        self.goal[ids,0]=torch.where(later,GOAL_X+(self.rand(n)-.5)*.01,self.goal[ids,0])
        self.goal[ids,1]=torch.where(later,(self.rand(n)-.5)*.02,self.goal[ids,1])
        self.goal[ids,2]=torch.where(stage==4,0.,self.goal[ids,2])
        self.goal[ids,2]=torch.where(stage==5,math.pi/2+(self.rand(n)-.5)*.3,self.goal[ids,2])
        self.d.mocap_pos[ids,self.gate_mocap,:]=0
        self.d.mocap_pos[ids,self.gate_mocap,0]=torch.where(later,GATE_X,5.).to(self.dtype)
        self.d.mocap_pos[ids,self.goal_mocap,:2]=self.goal[ids,:2]
        self.d.mocap_pos[ids,self.goal_mocap,2]=.001
        self.d.mocap_quat[ids,self.goal_mocap,:]=0
        self.d.mocap_quat[ids,self.goal_mocap,0]=torch.cos(self.goal[ids,2]/2)
        self.d.mocap_quat[ids,self.goal_mocap,3]=torch.sin(self.goal[ids,2]/2)
        for name in ["phase","hold","entered","passed","velocity","previous_velocity","previous_action","delayed_action","episode_return","jerk_integral","prev_qvel","prev_acc","episode_length_buf"]:
            getattr(self,name)[ids]=0
        self.delay_mask[ids]=False
        self.command_scale[ids]=1.
        if self.cfg.randomized:
            self.command_scale[ids]=.85+.3*self.rand(n,1)
            self.delay_mask[ids]=self.rand(n)<.5
            if self.cfg.backend=="warp":
                for key,val in self.default_fields.items(): getattr(self.sim.model,key)[ids]=val[ids]
                self.sim.model.geom_friction[ids,:,0]*=(.7+.6*self.rand(n,1))
                scale=.8+.4*self.rand(n)
                self.sim.model.body_mass[ids,self.block]*=scale
                self.sim.model.body_inertia[ids,self.block,:]*=scale[:,None]
                # Joint-space mass constants must follow body mass changes.
                from mjlab.managers.event_manager import RecomputeLevel
                self.sim.recompute_constants(RecomputeLevel.set_const)
        self.sim.forward()
        self.target_xy[ids]=self.d.site_xpos[ids,self.sid,:2]
        self.rules.reset(ids)
        return self.get_observations()

    def state(self):
        q=self.d.qpos[:,9:13]
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        return self.d.qpos[:,6:8],yaw,self.d.site_xpos[:,self.sid,:]

    def contact_distance(self,xy,yaw,tip):
        from .scene import BLOCK_BOXES
        rel=tip[:,:2]-xy; c,s=torch.cos(yaw),torch.sin(yaw)
        local=torch.stack((c*rel[:,0]+s*rel[:,1],-s*rel[:,0]+c*rel[:,1]),-1)
        ds=[]
        for pos,size in BLOCK_BOXES:
            delta=(local-torch.tensor(pos[:2],device=self.device,dtype=self.dtype)).abs()-torch.tensor(size[:2],device=self.device,dtype=self.dtype)
            ds.append(torch.linalg.vector_norm(delta.clamp(min=0),dim=-1)+delta.max(-1).values.clamp(max=0))
        return torch.stack(ds,-1).min(-1).values-.0011

    def get_observations(self):
        xy,yaw,tip=self.state()
        subgoal=self.goal.clone()
        before=(self.stage>=4)&~self.passed
        subgoal[before,0]=GATE_X+.043
        subgoal[before,1]=0
        subgoal[before,2]=0
        parts=[self.d.qpos[:,:5],self.d.qvel[:,:5]*.1,tip*5,xy*5,
               torch.stack((torch.sin(yaw),torch.cos(yaw)),-1),self.d.qvel[:,6:12]*.1,
               (subgoal[:,:2]-xy)*10,torch.stack((torch.sin(subgoal[:,2]-yaw),torch.cos(subgoal[:,2]-yaw)),-1),
               (self.goal[:,:2]-xy)*10,torch.stack((torch.sin(self.goal[:,2]-yaw),torch.cos(self.goal[:,2]-yaw)),-1),
               (xy-tip[:,:2])*10,(self.target_xy-tip[:,:2])*10, self.previous_action,self.velocity/.04,
               torch.nn.functional.one_hot(self.stage-1,5).to(self.dtype),self.passed[:,None].to(self.dtype),
               torch.full((self.num_envs,1),GATE_X,device=self.device,dtype=self.dtype)*5]
        obs=torch.cat(parts,-1).float()
        if self.cfg.randomized:
            # Small bounded state noise, applied consistently in train and test.
            obs=obs+(self.rand(self.num_envs,obs.shape[1]).float()-.5)*.01
        return TensorDict({"actor":obs,"critic":obs.clone()},batch_size=[self.num_envs],device=self.device)

    def _control(self,actions):
        actions=actions.to(self.dtype).clamp(-1,1)
        desired=actions/torch.linalg.vector_norm(actions,dim=-1,keepdim=True).clamp(min=1)*self.cfg.speed_limit
        use=torch.where(self.delay_mask[:,None],self.delayed_action,desired)
        self.delayed_action[:]=desired
        dv=use-self.velocity
        dv=dv/torch.linalg.vector_norm(dv,dim=-1,keepdim=True).clamp(min=1e-9)*torch.linalg.vector_norm(dv,dim=-1,keepdim=True).clamp(max=self.cfg.acceleration_limit*self.cfg.control_dt)
        self.previous_velocity[:]=self.velocity
        self.velocity+=dv
        tip=self.d.site_xpos[:,self.sid]
        self.target_xy+=self.velocity*self.cfg.control_dt*self.command_scale
        # Bounded tracking error prevents accumulating commands into a barrier.
        self.target_xy[:]=torch.minimum(torch.maximum(self.target_xy,tip[:,:2]-.010),tip[:,:2]+.010)
        self.target_xy[:,0].clamp_(.110,.295); self.target_xy[:,1].clamp_(-.13,.13)
        desired_pos=torch.cat((self.target_xy,torch.full((self.num_envs,1),TOOL_Z,device=self.device,dtype=self.dtype)),-1)
        if self.cfg.backend=="warp":
            self.point[:]=tip
            with wp.ScopedDevice(self.device): mjw.jac(self.sim.wp_model,self.sim.wp_data,self.jp_wp,self.jr_wp,self.point_wp,self.body_wp)
            jp,jr=self.jp[:,:,:5],self.jr[:,:,:5]
        else:
            jp0=np.zeros((3,self.model.nv)); jr0=jp0.copy()
            mujoco.mj_jacSite(self.model,self.sim.mj_data,jp0,jr0,self.sid)
            jp=torch.from_numpy(jp0[None,:,:5]); jr=torch.from_numpy(jr0[None,:,:5])
        axis=self.d.xmat[:,self.gripper].reshape(-1,3,3)[:,:,2]
        up=torch.zeros_like(axis); up[:,2]=1
        rot=torch.linalg.cross(axis,up,dim=-1)
        jac=torch.cat((jp,.10*jr[:,:2,:]),1)
        error=torch.cat((desired_pos-tip,.10*rot[:,:2]),-1)
        jt=jac.transpose(1,2)
        lhs=jt@jac+.002**2*torch.eye(5,device=self.device,dtype=self.dtype)
        dq=torch.linalg.solve(lhs,(jt@error.unsqueeze(-1))).squeeze(-1).clamp(-.05,.05)
        self.d.ctrl[:,:5]=torch.minimum(torch.maximum(self.d.qpos[:,:5]+dq,self.lower),self.upper)
        self.d.ctrl[:,5]=GRIPPER_CLOSED
        return actions

    def step(self,actions):
        old_xy,old_yaw,old_tip=self.state()
        old_xy=old_xy.clone(); old_yaw=old_yaw.clone(); old_tip=old_tip.clone()
        old_dist=self.contact_distance(old_xy,old_yaw,old_tip)
        old_pass=self.passed.clone()
        act=self._control(actions)
        if self.control_graph is not None:
            with wp.ScopedDevice(self.device): wp.capture_launch(self.control_graph)
        else:
            for _ in range(self.decimation):
                self.sim.step()
                self.rules.native()
            self.sim.forward()
            self.rules.native(advance=False)
        self.episode_length_buf+=1
        xy,yaw,tip=self.state()
        vertices=block_vertices(xy,yaw)
        lo,hi=vertices.min(1).values,vertices.max(1).values
        upright=self.d.xmat[:,self.block].reshape(-1,3,3)[:,2,2]>.966
        level=(self.d.qpos[:,8]-.009).abs()<.006
        self.entered[:]=self.rules.mask!=0
        self.passed[:]=self.rules.passed!=0
        finite=torch.isfinite(self.d.qpos).all(-1)&torch.isfinite(self.d.qvel).all(-1)
        outside=(lo[:,0]<.080)|(hi[:,0]>.320)|(lo[:,1]<-.14)|(hi[:,1]>.14)
        failed=(self.rules.flags!=0)|~finite
        poserr=torch.linalg.vector_norm(xy-self.goal[:,:2],dim=-1)
        angerr=wrap_angle(yaw-self.goal[:,2]).abs()
        linear=torch.linalg.vector_norm(self.d.qvel[:,6:9],dim=-1)
        angular=torch.linalg.vector_norm(self.d.qvel[:,9:12],dim=-1)
        near=self.contact_distance(xy,yaw,tip)
        self.hold[:]=self.rules.hold//self.decimation
        required=torch.where(self.stage==1,round(.15/self.cfg.timestep),round(1/self.cfg.timestep))
        success=(self.rules.hold>=required)&~failed
        timeout=self.episode_length_buf>=self.max_episode_length
        done=success|failed|timeout
        numerical_failure=((self.rules.flags&1)!=0)|~finite
        if numerical_failure.any():self.last_numerical_failure=self.state_dict()
        sub=self.goal.clone(); before=(self.stage>=4)&~old_pass
        sub[before,0]=GATE_X+.043; sub[before,1]=0; sub[before,2]=0
        olderr=torch.linalg.vector_norm(old_xy-sub[:,:2],dim=-1)+.025*wrap_angle(old_yaw-sub[:,2]).abs()
        newerr=torch.linalg.vector_norm(xy-sub[:,:2],dim=-1)+.025*wrap_angle(yaw-sub[:,2]).abs()
        progress=(olderr-newerr)*100
        reaching=(old_dist-near)*30
        reward=torch.where(self.stage==1,reaching,progress+.15*reaching)
        reward+=.02*torch.exp(-near.clamp(min=0)/.02)
        reward+=2*(self.passed&~old_pass).to(self.dtype)+10*success.to(self.dtype)-5*failed.to(self.dtype)-.002
        if self.cfg.smoothness:
            reward-=.01*(act-self.previous_action).square().sum(-1)+.001*self.d.qvel[:,:5].square().sum(-1)
        acc=(self.d.qvel[:,:5]-self.prev_qvel)/self.cfg.control_dt
        jerk=(acc-self.prev_acc)/self.cfg.control_dt
        self.jerk_integral+=jerk.square().mean(-1)*self.cfg.control_dt
        self.prev_qvel[:]=self.d.qvel[:,:5]; self.prev_acc[:]=acc
        self.previous_action[:]=act
        self.episode_return+=reward
        self.last_terminal=[]
        ids=done.nonzero().flatten()
        if len(ids):
            values=torch.stack((ids,self.stage[ids],success[ids],failed[ids],timeout[ids],poserr[ids],angerr[ids],self.episode_length_buf[ids]*self.cfg.control_dt,self.episode_return[ids],self.jerk_integral[ids],self.passed[ids],self.rules.flags[ids]),-1).detach().cpu().tolist()
            keys=["env","stage","success","failed","timeout","position_error_m","orientation_error_rad","seconds","return","joint_jerk_integral","passed_gate","failure_flags"]
            self.last_terminal=[dict(zip(keys,row)) for row in values]
            for row in self.last_terminal:
                row['failure_flags']=int(row['failure_flags'])
                row['failure_categories']=[name for bit,name in FAILURE_NAMES.items() if row['failure_flags']&bit]
            self.reset(ids,preserve_terminal=True)
        extras={"time_outs":timeout&~success&~failed,"numerical_failures":numerical_failure,"log":{}}
        return self.get_observations(),torch.nan_to_num(reward).float(),done,extras

    def snapshot(self,index=0):
        data=mujoco.MjData(self.model)
        for name in ["qpos","qvel","ctrl","mocap_pos","mocap_quat"]:
            getattr(data,name)[:]=getattr(self.d,name)[index].detach().cpu().numpy()
        mujoco.mj_forward(self.model,data)
        return data

    def state_dict(self):
        names=["stage","phase","hold","entered","passed","goal","previous_action","velocity","previous_velocity","target_xy","delayed_action","delay_mask","command_scale","episode_return","jerk_integral","prev_qvel","prev_acc","episode_length_buf"]
        return {"task":{k:getattr(self,k).clone() for k in names},"rules":self.rules.state_dict(),"sim":{k:getattr(self.d,k).clone() for k in ["qpos","qvel","qacc_warmstart","ctrl","mocap_pos","mocap_quat"]},"rng":self.rng.get_state(),"stage":self.cfg.stage}

    def load_state_dict(self,state):
        self.cfg.stage=state["stage"]
        for k,v in state["task"].items(): getattr(self,k)[:]=v.to(self.device)
        for k,v in state["sim"].items(): getattr(self.d,k)[:]=v.to(self.device)
        self.rng.set_state(state["rng"].cpu())
        self.sim.forward()
        if 'rules' in state:self.rules.load_state_dict(state['rules'])
        else:self.rules.reset(torch.arange(self.num_envs,device=self.device))
