import json
z=json.load(open('artifacts/gripper_feedback_1ms.json'))
for t in z['trace']:
 if round(t['time']*20) in [120,125,130,135,140,145,150,153]: print(t)
