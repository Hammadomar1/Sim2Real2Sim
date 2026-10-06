"""Opt-in recovery-reset curriculum; original physics and success rules."""
import torch
from push_filter import FilteredEfficientPushEnv

class RecoveryEnv(FilteredEfficientPushEnv):
    def __init__(self,cfg,level=0,alpha=.5,bank_path='artifacts/push_recovery/stalled_bank.pt',recovery_fraction=.4):
        self.recovery_bank=None;self.recovery_fraction=recovery_fraction
        super().__init__(cfg,level,alpha)
        bank=torch.load(bank_path,map_location=self.device,weights_only=False)
        self.recovery_bank=bank['state']
        valid=[i for i in range(64) if i not in bank['ended'] and int(bank['state']['rules']['flags'][i])==0]
        self.bank_indices=torch.tensor(valid,device=self.device,dtype=torch.long)
        self.from_recovery=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.reset()

    def reset(self,ids=None,*,preserve_terminal=False):
        super().reset(ids,preserve_terminal=preserve_terminal)
        if self.recovery_bank is None:return self.get_observations()
        if ids is None:ids=torch.arange(self.num_envs,device=self.device)
        self.from_recovery[ids]=False
        chosen=ids[(self.stage[ids]==2)&(self.rand(len(ids))<self.recovery_fraction)] if self.cfg.training else ids[:0]
        if len(chosen):
            source=self.bank_indices[(self.rand(len(chosen))*len(self.bank_indices)).long()]
            self.install_recovery(chosen,source)
        return self.get_observations()

    def install_recovery(self,ids,source):
        state=self.recovery_bank
        for group in ['sim','task','rules']:
            for key,value in state[group].items():
                target=getattr(self.d if group=='sim' else self.rules if group=='rules' else self,key)
                target[ids]=value[source]
        # A reset starts a new 30-second learning episode, not a completed prefix.
        for key in ['episode_length_buf','episode_return','jerk_integral','hold']:
            getattr(self,key)[ids]=0
        self.rules.hold[ids]=0
        self.had_contact[ids]=True;self.separation_steps[ids]=5
        self.distance_levels[ids]=0
        self.initial_goal_distance[ids]=(self.goal[ids,:2]-self.d.qpos[ids,6:8]).norm(dim=-1)
        self.from_recovery[ids]=True
        self.sim.forward()

    def step(self,actions):
        stage=self.stage.clone();had=self.had_contact.clone();gap=self.separation_steps.clone();kind=self.from_recovery.clone()
        obs,reward,done,extras=super().step(actions)
        # Remove the blanket recontact cost; useful changes of contact side must
        # not be charged merely for breaking contact. No contact annuity added.
        refund=.1*((stage==2)&had&(gap>=5)&(self.end_touch!=0)).float()
        reward+=refund;self.episode_return[~done]+=refund[~done]
        for row in self.last_terminal:
            i=int(row['env']);row['return']+=float(refund[i]);row['recovery_reset']=bool(kind[i])
        return obs,reward,done,extras

    def state_dict(self):
        state=super().state_dict()
        if hasattr(self,'from_recovery'):state['recovery_reset_flags']=self.from_recovery.clone()
        return state

    def load_state_dict(self,state):
        super().load_state_dict(state)
        if 'recovery_reset_flags' in state:self.from_recovery[:]=state['recovery_reset_flags'].to(self.device)
