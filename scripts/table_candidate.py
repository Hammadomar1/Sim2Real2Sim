"""Opt-in plane contact within the existing finite task workspace.

The visual slab retains its dimensions. The plane is infinite: episodes must
terminate at the existing, stricter workspace boundary; this is not a model of
falling off a physical table. Original gripper collisions remain unchanged.
"""
import xml.etree.ElementTree as ET
import mujoco
from so101_m1.scene import build_scene,ARTIFACTS

def load_candidate():
    path=build_scene(ARTIFACTS/'table_candidate_source.xml')
    tree=ET.parse(path);world=tree.getroot().find('worldbody');table=world.find("geom[@name='table']")
    if table.get('type')=='plane':
        target=ARTIFACTS/'table_candidate.xml';tree.write(target,encoding='utf-8',xml_declaration=True)
        return mujoco.MjModel.from_xml_path(str(target))
    attrs=dict(table.attrib);attrs.update(name='table_visual',contype='0',conaffinity='0',mass='0')
    ET.SubElement(world,'geom',**attrs)
    table.set('type','plane');table.set('pos','0.14 0 0');table.set('rgba','0 0 0 0')
    path=ARTIFACTS/'table_candidate.xml';tree.write(path,encoding='utf-8',xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(path))

def enable():
    import so101_m1.env as module
    module.load_model=load_candidate
