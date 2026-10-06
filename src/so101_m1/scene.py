"""Deterministic MJCF construction; all dynamics belong to the actual arm."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "menagerie" / "robotstudio_so101"
ARTIFACTS = ROOT / "artifacts"
MENAGERIE_REVISION = "4d038b3feae26ec82b46a4d586379114012a8ac7"
TOOL_Z = 0.010
PHYSICS_TIMESTEP = 0.001
GRIPPER_CLOSED = -0.174
RESET_XY = [0.120, -0.060]
GATE_X = 0.215
GOAL_X = 0.270
GATE_HALF = 0.024
# Validated together: stiff contacts require adequate Newton line-search work.
SOLVER_ITERATIONS = 100
SOLVER_LS_ITERATIONS = 50
CONTACT_SOLREF = "0.008 1"
CONTACT_SOLIMP = "0.99 0.999 0.001 0.5 2"
# Two disjoint boxes, geometric reference at the bounding rectangle center.
BLOCK_BOXES = [((-0.0225, 0., 0.), (0.0075, 0.0175, 0.009)),
               ((0.0075, -0.010, 0.), (0.0225, 0.0075, 0.009))]
BLOCK_VERTICES = np.array([[-.030,-.0175],[.030,-.0175],[.030,-.0025],[-.015,-.0025],[-.015,.0175],[-.030,.0175]])
BLOCK_CORNERS = np.vstack([np.array(pos)+np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*size for pos,size in BLOCK_BOXES])
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def _s(values):
    return " ".join(str(float(x)) for x in values)

def build_scene(path=None):
    root = ET.parse(ASSETS / "so101.xml").getroot()
    root.set("model", "SO101 Push Turn Park")
    root.find("compiler").set("meshdir", str(ASSETS / "assets"))
    opt = root.find("option")
    opt.set("timestep", str(PHYSICS_TIMESTEP))
    opt.set("iterations", str(SOLVER_ITERATIONS))
    opt.set("ls_iterations", str(SOLVER_LS_ITERATIONS))
    ET.SubElement(opt, "flag", multiccd="enable")
    root.find("visual").append(ET.Element("global", offwidth="1280", offheight="720"))
    for i, geom in enumerate(root.findall(".//worldbody//geom")):
        if "name" not in geom.attrib:
            geom.set("name", f"robot_geom_{i}")
    for body_name,prefix in [('gripper','fixed_jaw'),('moving_jaw_so101_v1','moving_jaw')]:
        body=root.find(f".//body[@name='{body_name}']")
        for i,g in enumerate(body.findall("geom[@class='collision_gripper_mesh']")):
            g.set('name',f'{prefix}_mesh_{i}')
    world = root.find("worldbody")
    ET.SubElement(world,"light",pos="0.1 -0.3 0.8",dir="0 0 -1",diffuse="0.8 0.8 0.8")
    ET.SubElement(world,"light",pos="0.4 0.3 0.6",dir="0 0 -1",diffuse="0.5 0.5 0.5")
    # Plane contact avoids the reproduced GPU box/box resting jitter. The
    # finite task workspace is strictly inside the visible slab and is checked
    # every physics step. Falling off the physical slab is not modeled.
    ET.SubElement(world,"geom",name="table",type="plane",size="0.35 0.30 0.025",pos="0.14 0 0",rgba="0 0 0 0",friction="0.45 0.005 0.0001")
    gate=ET.SubElement(world,"body",name="gate",mocap="true",pos=_s([GATE_X,0,0]))
    for sign, label in [(-1,"south"),(1,"north")]:
        ET.SubElement(gate,"geom",name=f"gate_{label}",type="box",size=_s([.006,(.18-GATE_HALF)/2,.020]),pos=_s([0,sign*(.18+GATE_HALF)/2,.020]),rgba="0.90 0.43 0.13 1",friction="0.3 0.001 0.0001")
    gripper = root.find(".//body[@name='gripper']")
    # A massless, invisible control reference on the original fixed fingertip.
    # All physical contact uses the original Menagerie jaw collision geometry.
    ET.SubElement(gripper,"site",name="tool_tip",pos="-0.009 0 -0.103",size="0.001",rgba="0 0 0 0")
    block = ET.SubElement(world,"body",name="block",pos="0.155 0 0.009")
    ET.SubElement(block,"freejoint",name="block_free")
    for i,(pos,size) in enumerate(BLOCK_BOXES):
        ET.SubElement(block,"geom",name=f"block_{i}",type="box",pos=_s(pos),size=_s(size),mass=str([.025,.025][i]),rgba="0.1 0.65 0.9 1",friction="0.45 0.005 0.0001",condim="3",solref="0.01 1")
    # Specify both sides to avoid averaging a task contact with softer defaults.
    jaw_names=[g.get('name') for g in world.findall('.//geom') if g.get('name','').startswith(('fixed_jaw_','moving_jaw_'))]
    for name in ["table", "block_0", "block_1", "gate_north", "gate_south"]+jaw_names:
        geom = world.find(f".//geom[@name='{name}']")
        geom.set("solref", CONTACT_SOLREF)
        geom.set("solimp", CONTACT_SOLIMP)
        if name in jaw_names:
            # Point contacts carry normal and sliding forces. Multiple contact
            # points on the original jaw surfaces still generate object torque.
            geom.set('condim','3')
    target=ET.SubElement(world,"body",name="goal",mocap="true",pos=_s([GOAL_X,0,.001]))
    # The goal is a visual outline only: no artificial constraint or attraction.
    for i,(a,b) in enumerate(zip(BLOCK_VERTICES,np.roll(BLOCK_VERTICES,-1,axis=0))):
        ET.SubElement(target,"geom",name=f"goal_edge_{i}",type="capsule",fromto=_s([*a,0,*b,0]),size="0.0007",rgba="0.3 1 0.45 0.8",contype="0",conaffinity="0",mass="0")
    visible=dict(world.find("geom[@name='table']").attrib)
    visible.update(name='table_visual',type='box',pos='0.14 0 -0.025',rgba='0.12 0.16 0.22 1',contype='0',conaffinity='0',mass='0')
    ET.SubElement(world,'geom',**visible)
    path = Path(path or ARTIFACTS / "scene.xml")
    path.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8",xml_declaration=True)
    return path

def load_model():
    import mujoco
    return mujoco.MjModel.from_xml_path(str(build_scene()))

def solve_ik(model, xy, initial=None):
    """CPU reference IK used to verify/reset; runtime has a batched equivalent."""
    import mujoco
    from scipy.optimize import least_squares
    data = mujoco.MjData(model)
    sid=model.site("tool_tip").id
    bid=model.body("gripper").id
    target=np.array([*xy,TOOL_Z])
    def residual(q):
        data.qpos[:5]=q
        data.qpos[5]=GRIPPER_CLOSED
        mujoco.mj_forward(model,data)
        axis=data.xmat[bid].reshape(3,3)[:,2]
        return np.r_[data.site_xpos[sid]-target, .10*(axis-np.array([0.,0.,1.]))]
    low=model.jnt_range[:5,0]+.01
    high=model.jnt_range[:5,1]-.01
    seeds=[initial] if initial is not None else [np.array([0,.4,-.6,-.8,0]),np.zeros(5),np.array([0,-.8,.8,.8,0])]
    best=None
    for seed in seeds:
        sol=least_squares(residual,np.clip(seed,low,high),bounds=(low,high),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        err=np.linalg.norm(residual(sol.x))
        if best is None or err<best[0]: best=(err,sol.x)
    residual(best[1])
    return np.r_[best[1],GRIPPER_CLOSED], float(np.linalg.norm(data.site_xpos[sid]-target)), data.xmat[bid].reshape(3,3)[:,2].copy()

def inspect_workspace():
    import mujoco
    model=load_model()
    records=[]
    for x in [.10,.14,.18,.22,.26,.30,.34]:
        for y in [-.10,-.05,0,.05,.10]:
            q,err,axis=solve_ik(model,[x,y])
            data=mujoco.MjData(model); data.qpos[:6]=q; data.qpos[6:13]=[.5,.5,.009,1,0,0,0]
            mujoco.mj_forward(model,data)
            collisions=[]
            for c in data.contact:
                if c.dist < -.001:
                    collisions.append([model.geom(c.geom1).name,model.geom(c.geom2).name,float(c.dist)])
            records.append(dict(x=x,y=y,error_m=err,axis=axis.tolist(),q=q.tolist(),penetrations=collisions))
    out=ARTIFACTS / "workspace.json"
    out.write_text(json.dumps(records,indent=2))
    return records
