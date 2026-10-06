"""Adversarial synthetic snapshots test rules, not physical task feasibility."""
import math
import numpy as np
import pytest
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import GATE_X,GOAL_X
from so101_m1.rules import *

@pytest.fixture(scope='module')
def env():
    torch.set_num_threads(1)
    return PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5))

def place(e,x=.15,y=0.,yaw=0.,z=.0092,roll=0.,reset=False):
    d=e.sim.mj_data
    d.qpos[6:13]=[x,y,z,math.cos(yaw/2)*math.cos(roll/2),math.cos(yaw/2)*math.sin(roll/2),math.sin(yaw/2)*math.sin(roll/2),math.sin(yaw/2)*math.cos(roll/2)]
    d.qvel[:]=0;e.sim.forward()
    if reset:e.rules.reset(torch.tensor([0]))

def parked(e):
    e.reset();e.stage[:]=5;e.goal[0]=torch.tensor([GOAL_X,0.,math.pi/2])
    place(e,x=GOAL_X,yaw=math.pi/2,reset=True)
    e.rules.mask[:]=65535;e.rules.passed[:]=1

def tick(e,n=1):
    for _ in range(n):e.rules.native()

def test_layout(env):
    env.reset()
    assert GATE_X==.215 and GOAL_X==.270
    assert float(env.d.mocap_pos[0,env.gate_mocap,0])==pytest.approx(GATE_X,abs=1e-8)

def test_real_jaws_replace_attachment(env):
    names=[env.model.geom(i).name for i in range(env.model.ngeom)]
    assert 'pusher' not in names
    for name in ['fixed_jaw_box3','fixed_jaw_mesh_0','moving_jaw_mesh_1']:
        assert env.rules.allowed[env.model.geom('block_0').id,env.model.geom(name).id]
        assert not env.rules.allowed[env.model.geom('table').id,env.model.geom(name).id]
    env.reset();env._control(torch.zeros((1,2)))
    assert float(env.d.ctrl[0,5])==pytest.approx(-.174)
    assert .265<=float(env.goal[0,0])<=.275

def test_entire_block_must_cross(env):
    env.reset();place(env,reset=True)
    for x in np.arange(.150,.236,.001):place(env,x=x);tick(env)
    assert not env.rules.passed[0] # rear of the 60 mm block is still before far gate face
    for x in np.arange(.236,.261,.001):place(env,x=x);tick(env)
    assert env.rules.passed[0] and env.rules.mask[0]==65535
    assert not env.rules.flags[0]

def test_teleport_cannot_count_as_crossing(env):
    env.reset();place(env,reset=True);place(env,x=.27);tick(env)
    assert int(env.rules.flags[0])&SHORTCUT

def test_enter_then_go_around_gate_is_rejected(env):
    env.reset();place(env,reset=True)
    for x in np.arange(.15,.216,.001):place(env,x=x);tick(env)
    assert env.rules.mask[0]!=0
    for y in np.arange(0,.20,.001):place(env,x=.215,y=y);tick(env)
    for x in np.arange(.215,.271,.001):place(env,x=x,y=.20);tick(env)
    assert int(env.rules.flags[0])&(SHORTCUT|OUTSIDE)

@pytest.mark.parametrize('kind',['tip','lift','nan','joint'])
def test_failures_latch_until_reset(env,kind):
    parked(env)
    if kind=='tip':place(env,x=GOAL_X,roll=math.radians(20))
    if kind=='lift':place(env,x=GOAL_X,z=.03)
    if kind=='nan':env.sim.mj_data.qvel[6]=float('nan')
    if kind=='joint':env.sim.mj_data.qpos[0]=env.model.jnt_range[0,1]+.01
    tick(env);assert env.rules.flags[0]!=0
    env.sim.mj_data.qpos[:6]=env.q_initial;place(env,x=GOAL_X,yaw=math.pi/2);tick(env,5)
    assert env.rules.flags[0]!=0 and env.rules.hold[0]==0
    env.reset();assert env.rules.flags[0]==0 and env.rules.mask[0]==0 and env.rules.hold[0]==0

def test_one_continuous_second_required(env):
    parked(env);tick(env,499);assert env.rules.hold[0]==499
    tick(env);assert env.rules.hold[0]==500
    env.sim.mj_data.qvel[6]=.0051;tick(env);assert env.rules.hold[0]==0
    env.sim.mj_data.qvel[:]=0;tick(env,10);assert env.rules.hold[0]==10

@pytest.mark.parametrize('position,angle_deg,linear,angular,valid',[(.0099,9.9,.0049,4.9,True),(.0101,0,0,0,False),(0,10.1,0,0,False),(0,0,.0051,0,False),(0,0,0,5.1,False)])
def test_pose_and_speed_thresholds(env,position,angle_deg,linear,angular,valid):
    parked(env);place(env,x=GOAL_X+position,yaw=math.pi/2+math.radians(angle_deg),reset=True)
    env.rules.mask[:]=65535;env.rules.passed[:]=1
    env.sim.mj_data.qvel[6]=linear;env.sim.mj_data.qvel[11]=math.radians(angular)
    tick(env);assert bool(env.rules.hold[0])==valid

def test_yaw_wraparound(env):
    parked(env);env.goal[0,2]=math.pi-.01;place(env,x=GOAL_X,yaw=-math.pi+.01,reset=True)
    env.rules.mask[:]=65535;env.rules.passed[:]=1;tick(env);assert env.rules.hold[0]==1

def test_historical_pass_does_not_allow_partial_parking(env):
    parked(env);env.goal[0]=torch.tensor([GATE_X+.02,0.,0.])
    place(env,x=GATE_X+.02,reset=True);env.rules.mask[:]=65535;env.rules.passed[:]=1
    tick(env,500);assert env.rules.hold[0]==0

@pytest.mark.parametrize('a,b,depth,flag',[('fixed_jaw_box3','gate_north',-.000049,0),('fixed_jaw_box3','gate_north',-.000051,COLLISION),('fixed_jaw_box3','table',-.0001,COLLISION),('block_0','fixed_jaw_box3',-.0008,0),('block_1','gate_south',-.0008,0),('block_0','table',-.00101,PENETRATION),('robot_geom_0','block_0',-.0001,COLLISION)])
def test_collision_rules(env,a,b,depth,flag):
    env.reset();env.rules.inspect_contact(env.model.geom(a).id,env.model.geom(b).id,depth)
    assert int(env.rules.flags[0])==flag

def test_actual_step_failure_and_auto_reset(env):
    env.reset();env.rules.flags[0]=COLLISION
    _,_,done,extras=env.step(torch.zeros((1,2)))
    assert done[0] and not extras['time_outs'][0]
    assert env.last_terminal[0]['failure_categories']==['invalid_robot_contact']
    assert env.rules.flags[0]==0 and env.rules.mask[0]==0
    env.reset();assert env.last_terminal==[]

def test_timeout_is_separate_from_failure(env):
    env.reset();env.episode_length_buf[:]=env.max_episode_length-1
    _,_,done,extras=env.step(torch.zeros((1,2)))
    assert done[0] and extras['time_outs'][0]
    assert env.last_terminal[0]['timeout'] and not env.last_terminal[0]['failed']

