import sys
sys.path.insert(0,'scripts')
from feedback_path import FeedbackPlanner
from so101_m1.scene import ARTIFACTS
import numpy as np
p=FeedbackPlanner(.001)
z=np.load(ARTIFACTS/'gripper_feedback_1ms.npz')
for a in z['actions'][:153]:p.step(a,True)
print('pose',p.pose(),'tip',p.d.site_xpos[p.e.sid], 'bad',p.bad,'pen',p.pen)
print('q',p.d.qpos[:6])
