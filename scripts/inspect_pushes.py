from plan_path import Planner
p=Planner(resume='planned_path');state=p.save()
for route,end,desc in p.candidates():
    if desc != (3,.8,.025,.65):continue
    p.restore(state);valid=True
    for point in route+[end]:
        if not p.move(point,maxsteps=110):valid=False;break
        print('AT',point,'actual',p.d.site_xpos[p.e.sid],'object',p.pose(),flush=True)
    print(desc,'valid',valid,'bad',p.bad,'pen',p.pen,'cost',p.cost(2),'pose',p.pose(),'route',[x.tolist() for x in route],flush=True)
