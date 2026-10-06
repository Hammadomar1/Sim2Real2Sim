import numpy as np
from feedback_path import FeedbackPlanner
from so101_m1.scene import ARTIFACTS
p=FeedbackPlanner(.001)
for a in np.load(ARTIFACTS/'gripper_park_stall.npz')['actions']:p.step(a)
p.phase=2;state=p.save();initial=p.cost(2)
print('START',p.pose(),'tip',p.d.site_xpos[p.e.sid],flush=True)
for edge in [0,1,2,3,4,5]:
 for f in [.15,.5,.643,.85]:
  p.restore(state);point,normal=p.geometry(edge,f);target=point-p.support(normal)+normal*.010;route=p.route(target)
  if route is None:print(edge,f,'NO ROUTE',target,flush=True);continue
  if not all(p.move(t,maxsteps=100) for t in route):print(edge,f,'MOVE FAIL',p.bad,'tip',p.d.site_xpos[p.e.sid],flush=True);continue
  best=-1
  for _ in range(30):
   p.push(edge,f,0)
   if p.bad:break
   best=max(best,initial-p.cost(2))
  print(edge,f,'PROGRESS',best,'BAD',p.bad,'pose',p.pose(),flush=True)
