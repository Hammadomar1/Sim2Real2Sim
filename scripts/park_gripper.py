"""Continue verified gate actions into a diagnostic gripper parking search."""
import argparse
import numpy as np
from feedback_path import FeedbackPlanner
from so101_m1.scene import ARTIFACTS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=20);parser.add_argument('--resume');args=parser.parse_args()
    p=FeedbackPlanner(.001);p.stem='gripper_complete_path'
    z=np.load(ARTIFACTS/((args.resume or 'gripper_gate_reposition')+'.npz'))
    for a in z['actions']:p.step(a,True)
    if p.bad:raise RuntimeError('Invalid continuous prefix')
    p.phase=2
    if not p.run_feedback(args.iterations):raise SystemExit(2)
if __name__=='__main__':main()
