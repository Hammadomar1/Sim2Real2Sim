"""Opt-in stage-2 reward for reaching with real jaw tips and sustained useful pushes."""
import torch
from compare_stage2_lr import ContactEnv
from stage2_directional import approach_error
from so101_m1.scene import BLOCK_BOXES

class EfficientPushEnv(ContactEnv):
    def __init__(self,cfg,level=0):
        super().__init__(cfg,level)
        self.separation_steps=torch.full((self.num_envs,),5,device=self.device,dtype=torch.long)
        self.had_contact=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
    def jaw_gap(self):
        # Sphere/box signed distances use the six original Menagerie jaw-tip spheres.
        # Other jaw geometry is accounted for when the actual collision flag is set.
        if not hasattr(self,'jaw_ids'):
            ids=[i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith(('fixed_jaw_sph_tip','moving_jaw_sph_tip'))]
            self.jaw_ids=torch.tensor(self.model.geom_bodyid[ids],device=self.device,dtype=torch.long)
            self.jaw_local=torch.tensor(self.model.geom_pos[ids],device=self.device,dtype=self.dtype)
            self.jaw_radius=torch.tensor(self.model.geom_size[ids,0],device=self.device,dtype=self.dtype)
        rot=self.d.xmat[:,self.jaw_ids].reshape(self.num_envs,-1,3,3)
        centers=self.d.xpos[:,self.jaw_ids]+(rot@self.jaw_local[None,:,:,None]).squeeze(-1)
        block_rot=self.d.xmat[:,self.block].reshape(-1,3,3)
        local=(centers-self.d.xpos[:,self.block,None,:])@block_rot
        gaps=[]
        for pos,size in BLOCK_BOXES:
            delta=(local-torch.tensor(pos,device=self.device,dtype=self.dtype)).abs()-torch.tensor(size,device=self.device,dtype=self.dtype)
            gap=delta.clamp(min=0).norm(dim=-1)+delta.max(-1).values.clamp(max=0)-self.jaw_radius
            gaps.append(gap.min(-1).values)
        gap=torch.stack(gaps,-1).min(-1).values.clamp(min=0)
        return torch.where(self.rules.touch!=0,torch.zeros_like(gap),gap)
    def reset(self,ids=None,*,preserve_terminal=False):
        if preserve_terminal:self.terminal_gap=self.jaw_gap().clone()
        result=super().reset(ids,preserve_terminal=preserve_terminal)
        if hasattr(self,'had_contact'):
            if ids is None:ids=torch.arange(self.num_envs,device=self.device)
            self.had_contact[ids]=False;self.separation_steps[ids]=5
        return result
    def step(self,actions):
        stage=self.stage.clone();goal=self.goal.clone();xy,yaw,tip=(v.clone() for v in self.state())
        old_gap=self.jaw_gap();old_approach=approach_error(xy,yaw,tip,goal)
        old_action=self.previous_action.clone();had=self.had_contact.clone();separation=self.separation_steps.clone()
        self.terminal_gap=None
        obs,reward,done,extras=super().step(actions)
        endxy,endyaw,endtip=self.terminal_geometry if self.terminal_geometry is not None else self.state()
        new_gap=self.terminal_gap if self.terminal_gap is not None else self.jaw_gap()
        touch=self.end_touch!=0
        restart=had&touch&(separation>=5)
        progress=(xy-goal[:,:2]).norm(dim=-1)-(endxy-goal[:,:2]).norm(dim=-1)
        extra_progress=300*progress
        replace_approach=30*(old_gap-new_gap)-30*(old_approach-approach_error(endxy,endyaw,endtip,goal))
        smooth=.04*(actions.clamp(-1,1)-old_action).square().sum(-1)
        failed=torch.zeros_like(progress)
        for row in self.last_terminal:
            if row['failed']:failed[int(row['env'])]=1
        # Total failure cost is 20, exceeding a full 30-second idle time cost of 12.
        delta=torch.where(stage==2,extra_progress+replace_approach-.01-.10*restart.float()-smooth-15*failed,0.)
        reward+=delta.float();self.episode_return[~done]+=delta[~done]
        self.had_contact[~done]=(had|touch)[~done]
        self.separation_steps[~done]=torch.where(touch,0,separation+1)[~done]
        for row in self.last_terminal:row['return']+=float(delta[int(row['env'])])
        return obs,reward,done,extras
    def state_dict(self):
        s=super().state_dict()
        if hasattr(self,'had_contact'):s['efficient_push_state']={'had_contact':self.had_contact.clone(),'separation_steps':self.separation_steps.clone()}
        return s
    def load_state_dict(self,s):
        super().load_state_dict(s)
        if 'efficient_push_state' in s:
            for key,value in s['efficient_push_state'].items():getattr(self,key)[:]=value.to(self.device)
