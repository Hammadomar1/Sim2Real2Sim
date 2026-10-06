import numpy as np,json
from feedback_path import FeedbackPlanner
from so101_m1.scene import ARTIFACTS
p=FeedbackPlanner(.002)
z=np.load(ARTIFACTS/'gripper_feedback_2ms.npz')
for a in z['actions']:p.step(a)
p.update_phase();state=p.save();initial=p.cost(p.phase);counts={'candidates':0,'route_failure':0,'push_invalid':0,'valid':0};best=0.
for route,edge,f,theta in list(p.generate()):
    counts['candidates']+=1;p.restore(state)
    if not all(p.move(point,maxsteps=100) for point in route):counts['route_failure']+=1;continue
    candidate_best=-1.
    for _ in range(60):
        p.push(edge,f,theta)
        if p.bad or p.pen>.001:counts['push_invalid']+=1;break
        candidate_best=max(candidate_best,initial-p.cost(p.phase));best=max(best,initial-p.cost(p.phase));counts['valid']+=1
    print(edge,f,theta,'progress',candidate_best,'pose',p.pose(),flush=True)
print(json.dumps(dict(counts=counts,best_progress=best)),flush=True)

