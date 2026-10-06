"""Stage-2 shaping for approaching the side opposite the goal; no scripted actions."""
import torch
from so101_m1.env import block_vertices
from stage2_rewards import Stage2RewardEnv

def approach_error(xy,yaw,tip,goal):
    direction=goal[:,:2]-xy
    direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp(min=1e-6)
    relative=block_vertices(xy,yaw)-xy[:,None,:]
    # Supporting edge opposite the goal, plus 5 mm clearance for approaching.
    radius=-(relative*direction[:,None,:]).sum(-1).min(-1).values
    target=xy-direction*(radius+.005)[:,None]
    target=torch.stack((target[:,0].clamp(.110,.295),target[:,1].clamp(-.13,.13)),-1)
    return torch.linalg.vector_norm(tip[:,:2]-target,dim=-1)

class DirectionalRewardEnv(Stage2RewardEnv):
    def __init__(self,cfg,variant='directional'):
        super().__init__(cfg,variant='baseline')

    def step(self,actions):
        old_xy,old_yaw,old_tip=(t.clone() for t in self.state());goal=self.goal.clone();stage=self.stage.clone()
        before=approach_error(old_xy,old_yaw,old_tip,goal)
        obs,reward,done,extras=super().step(actions)
        xy,yaw,tip=self.terminal_geometry if self.terminal_geometry is not None else self.state()
        approach=30*(before-approach_error(xy,yaw,tip,goal))
        new=3*self.last_reward_terms['position']+approach-.01
        delta=torch.where(stage==2,new-self.last_reward_terms['old'],0.)
        reward+=delta.float();self.episode_return[~done]+=delta[~done]
        for row in self.last_terminal:row['return']+=float(delta[int(row['env'])])
        return obs,reward,done,extras
