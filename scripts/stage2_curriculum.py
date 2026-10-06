"""Distance-only curriculum layered on the validated stage-2 pushing environment."""
from dataclasses import dataclass
import torch
from stage2_directional import DirectionalRewardEnv

DISTANCES=((.015,.020),(.020,.025),(.025,.030),(.030,.040))
LABELS=('15-20 mm','20-25 mm','25-30 mm','original 30-40 mm X offset')

@dataclass
class Promotion:
    level:int=0
    streak:int=0
    complete:bool=False

    def observe(self,successes,episodes):
        if episodes<64:raise ValueError('At least 64 validation episodes are required')
        self.streak=self.streak+1 if successes/episodes>=.90 else 0
        if self.streak<3:return False
        if self.level==len(DISTANCES)-1:self.complete=True;return False
        self.level+=1;self.streak=0;return True

class DistanceCurriculumEnv(DirectionalRewardEnv):
    def __init__(self,cfg,level=0):
        if not 0<=level<len(DISTANCES):raise ValueError('Invalid distance level')
        self.level=level
        super().__init__(cfg)

    def reset(self,ids=None,*,preserve_terminal=False):
        super().reset(ids,preserve_terminal=preserve_terminal)
        if not hasattr(self,'distance_levels'):
            self.distance_levels=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
            self.initial_goal_distance=self.zeros()
        if ids is None:ids=torch.arange(self.num_envs,device=self.device)
        self.distance_levels[ids]=-1
        chosen=ids[self.stage[ids]==2];n=len(chosen)
        if n:
            levels=torch.full((n,),self.level,device=self.device,dtype=torch.long)
            if self.cfg.training and self.level>0:
                levels=torch.where(self.rand(n)<.2,(self.rand(n)*self.level).long(),levels)
            self.distance_levels[chosen]=levels
            original=self.goal[chosen,:2]-self.d.qpos[chosen,6:8]
            direction=original/torch.linalg.vector_norm(original,dim=-1,keepdim=True)
            bounds=torch.tensor(DISTANCES,device=self.device,dtype=self.dtype)[levels]
            distance=bounds[:,0]+self.rand(n)*(bounds[:,1]-bounds[:,0])
            target=self.d.qpos[chosen,6:8]+direction*distance[:,None]
            # Last level exactly preserves the original production reset distribution.
            self.goal[chosen,:2]=torch.where((levels==3)[:,None],self.goal[chosen,:2],target)
            self.d.mocap_pos[chosen,self.goal_mocap,:2]=self.goal[chosen,:2]
        self.initial_goal_distance[ids]=torch.linalg.vector_norm(self.goal[ids,:2]-self.d.qpos[ids,6:8],dim=-1)
        self.sim.forward()
        return self.get_observations()

    def step(self,actions):
        levels=self.distance_levels.clone();initial=self.initial_goal_distance.clone()
        result=super().step(actions)
        for row in self.last_terminal:
            i=int(row['env']);row['distance_level']=int(levels[i]);row['initial_goal_distance_m']=float(initial[i])
        return result

    def state_dict(self):
        state=super().state_dict()
        state['distance_curriculum']=dict(level=self.level,levels=self.distance_levels.clone(),initial_goal_distance=self.initial_goal_distance.clone())
        return state

    def load_state_dict(self,state):
        super().load_state_dict(state)
        extra=state['distance_curriculum'];self.level=extra['level']
        self.distance_levels[:]=extra['levels'].to(self.device);self.initial_goal_distance[:]=extra['initial_goal_distance'].to(self.device)
