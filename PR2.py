import pybullet as p
import time
import pybullet_data
from pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM, PR2_URDF, DRAKE_PR2_URDF, \
    SIDE_HOLDING_LEFT_ARM, PR2_GROUPS, open_arm, get_disabled_collisions, REST_LEFT_ARM, rightarm_from_leftarm
from pybullet_tools.utils import set_base_values, joint_from_name, quat_from_euler, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, wait_if_gui, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_if_gui, load_pybullet, set_quat, Euler, PI, RED, add_line, \
    wait_for_duration, LockRenderer, base_aligned_z, Point, set_point, get_aabb, stable_z_on_aabb, AABB
from pybullet_tools.ikfast.pr2.ik import get_tool_pose, get_ik_generator
import numpy as np

# connect to GUI simulator 
connect(use_gui=True)

# load objects into the environment 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10) # set gravity
planeId = p.loadURDF("plane.urdf") # add ground
startPos = [0,0,0]
startOrientation = [0,0,0,1]#p.getQuaternionFromEuler([0,0,0])
pr2_urdf = "/home/kk/pybullet-planning/models/drake/pr2_description/urdf/pr2_simplified.urdf"
with HideOutput():
    pr2 = load_model(pr2_urdf, fixed_base=True)
#pr2 = p.loadURDF("/home/kk/pybullet-planning/models/drake/pr2_description/urdf/pr2_simplified.urdf",startPos, startOrientation,useFixedBase = 1) # add robot
table_id = p.loadURDF("table/table.urdf", basePosition=[1.05, -0.2, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071]) # add table
cube3_id = p.loadURDF("cube.urdf", basePosition=[0.6, -0.1, 0.65], globalScaling=0.04) # add target cube 3
cube4_id = p.loadURDF("cube.urdf", basePosition=[0.6, -0.3, 0.65], globalScaling=0.04) # add target cube 4
cube6_id = p.loadURDF("/home/kk/pybullet-planning/models/drake/objects/simple_cylinder.urdf", basePosition=[0.6, 0.1, 0.7], globalScaling=1) # add target cube 6
cup1_id = p.loadURDF("/home/kk/pybullet-planning/models/cup_red.urdf", basePosition=[0.8, -0.1, 0.7], globalScaling=1.0, baseOrientation=[0, 0, 0.7071, 0.7071]) # add cup 1. We want to place the cube into it
cup2_id = p.loadURDF("/home/kk/pybullet-planning/models/cup_blue.urdf", basePosition=[0.8, 0.1, 0.7], globalScaling=1.0, baseOrientation=[0, 0, 0.7071, 0.7071]) # add cup 2. We want to place the cube into it

dump_body(pr2) #print the information of the robot including joints and links

left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
torso_joints = joints_from_names(pr2, PR2_GROUPS['torso'])
torso_left = torso_joints + left_joints # get the number of joints for the left arm
torso_right = torso_joints + right_joints # get the number of joints for the right arm
left_gripper_joints = [74,76,75,77]
right_gripper_joints = [53,55,54,56]

# example to move left arm
def move_control(robo_id,arm,target_pose,pre_pose=[0, 0, 0, 0, 0, 0, 0, 0]):
    base = [15,15,15,15,15,15,15,15]
    generator = get_ik_generator(robo_id, arm, target_pose, torso_limits=False)
    for j in range(500):
        solutions = next(generator)
        if solutions != []:
            if np.sum(np.square(np.array(base)-np.array(pre_pose)))>np.sum(np.square(np.array(solutions[0])-np.array(pre_pose))):
                base = solutions[0]
    for j in range(len(base)):
        base[j] =  1/1000*base[j]-1/1000*q[j]
    return base

def close_gripper(gripper_joints):
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[0], controlMode=p.POSITION_CONTROL,targetPosition = 0,force=10)
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[1], controlMode=p.POSITION_CONTROL,targetPosition = 0,force=10)
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[2], controlMode=p.POSITION_CONTROL,targetPosition = 0,force=10)
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[3], controlMode=p.POSITION_CONTROL,targetPosition = 0,force=10)  
    '''
    for i in range (500):
        p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[0], controlMode=p.POSITION_CONTROL,targetPosition = 0.5-0.5*i/499,force=10)
        p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[1], controlMode=p.POSITION_CONTROL,targetPosition = 0.5-0.5*i/499,force=10)
        p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[2], controlMode=p.POSITION_CONTROL,targetPosition = -1+1.0*i/499,force=10)
        p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[3], controlMode=p.POSITION_CONTROL,targetPosition = -1+1.0*i/499,force=10)
        p.stepSimulation()
        time.sleep(1./240.)'''

def open_gripper(gripper_joints):
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[0], controlMode=p.POSITION_CONTROL,targetPosition = 0.5)
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[1], controlMode=p.POSITION_CONTROL,targetPosition = 0.5)
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[2], controlMode=p.POSITION_CONTROL,targetPosition = -1)
    p.setJointMotorControl2(bodyIndex=pr2, jointIndex=gripper_joints[3], controlMode=p.POSITION_CONTROL,targetPosition = -1)

def move_arm(q,base):
    for i in range (1000):
        for j in range(len(q)):
            q[j] +=  base[j]
        p.setJointMotorControlArray(bodyIndex=pr2, jointIndices=list(torso_left), controlMode=p.POSITION_CONTROL, targetPositions = q)
        p.stepSimulation()
        time.sleep(1./240.)


arm = 'left'
x = 0.6
y = 0.2
z = 0.9
target_pose = ((x, y, z), (1.414, -1.414, 0, 0))   
q = [0, 0, 0, 0, 0, 0, 0, 0]
base = move_control(pr2, arm, target_pose,q)
open_gripper(left_gripper_joints)
open_gripper(right_gripper_joints)
move_arm(q,base)


x = 0.6
y = 0.3
z = 0.72
target_pose = ((x, y, z), (1.414, -1.414, 0, 0))   
base = move_control(pr2, arm, target_pose,q)
move_arm(q,base)

x = 0.6
y = 0.08
z = 0.72
target_pose = ((x, y, z), (1.414, -1.414, 0, 0))
base = move_control(pr2, arm, target_pose,q)
move_arm(q,base)



x = 0.6
y = 0.08
z = 0.75
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
close_gripper(left_gripper_joints)
close_gripper(right_gripper_joints)
move_arm(q,base)

x = 0.6
y = 0.08
z = 0.9
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
move_arm(q,base)

x = 0.8
y = 0.1
z = 0.9
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
move_arm(q,base)
open_gripper(left_gripper_joints)
open_gripper(right_gripper_joints)

x = 0.75
y = 0.1
z = 0.75
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
move_arm(q,base)

p.disconnect()
