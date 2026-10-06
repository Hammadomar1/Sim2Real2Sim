"""Versioned stage-2-only reward experiment; production physics remains unchanged."""
import torch
from so101_m1.env import PushTurnParkEnv,wrap_angle

def reward_terms(old_xy,xy,old_yaw,yaw,old_near,near,goal):
    position=(torch.linalg.vector_norm(old_xy-goal[:,:2],dim=-1)-torch.linalg.vector_norm(xy-goal[:,:2],dim=-1))
    orientation=.025*(wrap_angle(old_yaw-goal[:,2]).abs()-wrap_angle(yaw-goal[:,2]).abs())
    approach=4.5*(old_near-near)
    proximity=.02*torch.exp(-near.clamp(min=0)/.02)
    old=100*(position+orientation)+approach+proximity-.002
    # No annuity for idle contact, no orientation requirement in a position-only stage.
    new=300*position+approach-.01
    return dict(position=100*position,orientation=100*orientation,approach=approach,proximity=proximity,old=old,new=new,delta=new-old)

class Stage2RewardEnv(PushTurnParkEnv):
    def __init__(self,cfg,variant='corrected'):
        self.variant=variant;self.terminal_geometry=None;self.reward_totals={}
        super().__init__(cfg)

    def reset(self,ids=None,*,preserve_terminal=False):
        if preserve_terminal:
            self.terminal_geometry=tuple(t.clone() for t in self.state())
        return super().reset(ids,preserve_terminal=preserve_terminal)

    def step(self,actions):
        old_xy,old_yaw,old_tip=(t.clone() for t in self.state());stage=self.stage.clone();goal=self.goal.clone()
        old_near=self.contact_distance(old_xy,old_yaw,old_tip)
        self.terminal_geometry=None
        obs,reward,done,extras=super().step(actions)
        xy,yaw,tip=self.terminal_geometry if self.terminal_geometry is not None else self.state()
        terms=reward_terms(old_xy,xy,old_yaw,yaw,old_near,self.contact_distance(xy,yaw,tip),goal)
        mask=stage==2;delta=torch.where(mask,terms['delta'],0.)
        self.last_reward_terms=terms
        if self.variant=='corrected':
            reward=reward+delta.float();self.episode_return[~done]+=delta[~done]
            for row in self.last_terminal:row['return']+=float(delta[int(row['env'])])
        return obs,reward,done,extras
