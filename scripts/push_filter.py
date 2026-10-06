"""Gentle command filter for the gripper-efficiency policy's playback/controller."""
from efficient_push import EfficientPushEnv

class FilteredEfficientPushEnv(EfficientPushEnv):
    def __init__(self,cfg,level=0,alpha=.5):
        if not 0<alpha<=1:raise ValueError('Filter alpha must be in (0, 1]')
        self.filter_alpha=alpha
        super().__init__(cfg,level)
    def step(self,actions):
        command=self.previous_action+self.filter_alpha*(actions.clamp(-1,1)-self.previous_action)
        return super().step(command)
