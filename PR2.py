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
from datetime import datetime

# connect to GUI simulator 
connect(use_gui=True)

# load objects into the environment 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10) # set gravity
planeId = p.loadURDF("plane.urdf") # add ground
startPos = [0,0,0]
startOrientation = [0,0,0,1]#p.getQuaternionFromEuler([0,0,0])
pr2 = p.loadURDF("/home/kk/pybullet-planning/models/drake/pr2_description/urdf/pr2_simplified.urdf",startPos, startOrientation,useFixedBase = 1,flags = p.URDF_USE_SELF_COLLISION) 

table_id = p.loadURDF("table/table.urdf", basePosition=[1.05, -0.2, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071]) # add table
#cube3_id = p.loadURDF("/home/kk/pybullet-planning/models/drake/objects/simple_cylinder.urdf", basePosition=[0.6, -0.1, 0.7], globalScaling=1) # add target cube 3
#cube1_id = p.loadURDF("/home/kk/pybullet-planning/models/drake/objects/simple_cylinder.urdf", basePosition=[0.7, 0.05, 0.7], globalScaling=1) # add target cube 3
#cube4_id = p.loadURDF("/home/kk/pybullet-planning/models/drake/objects/simple_cylinder.urdf", basePosition=[0.7, -0.05, 0.7], globalScaling=1) # add target cube 4
cube6_id = p.loadURDF("/home/kk/pybullet-planning/models/drake/objects/simple_cylinder.urdf", basePosition=[0.6, 0.1, 0.7], globalScaling=1) # add target cube 6
cup1_id = p.loadURDF("/home/kk/pybullet-planning/models/cup_red.urdf", basePosition=[0.8, -0.1, 0.7], globalScaling=1.0, baseOrientation=[0, 0, 0.7071, 0.7071]) # add cup 1. We want to place the cube into it
cup2_id = p.loadURDF("/home/kk/pybullet-planning/models/cup_blue.urdf", basePosition=[0.8, 0.1, 0.7], globalScaling=1.0, baseOrientation=[0, 0, 0.7071, 0.7071]) # add cup 2. We want to place the cube into it
picked = 0
print(planeId,pr2,table_id,cube6_id)
lit_mass = [50,52,59,60,71,73,80,81]
for i in lit_mass:
    p.changeDynamics(bodyUniqueId = pr2,linkIndex=i,mass=0.0)
p.changeDynamics(bodyUniqueId = pr2,linkIndex=-1,mass=1000.0)
dump_body(pr2) #print the information of the robot including joints and links


left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
torso_joints = joints_from_names(pr2, PR2_GROUPS['torso'])
torso_left = torso_joints + left_joints # get the number of joints for the left arm
torso_right = torso_joints + right_joints # get the number of joints for the right arm
left_gripper_joints = [74,76,75,77]
right_gripper_joints = [53,55,54,56]

for i in range(len(right_joints)):
    for j in range(len(right_joints)):
        p.setCollisionFilterPair(pr2,pr2,left_joints[i],right_joints[j],enableCollision=1)
# example to move left arm

def move_control(robo_id,arm,target_pose,pre_pose=[0, 0, 0, 0, 0, 0, 0, 0]):
    base = [15,15,15,15,15,15,15,15]
    generator = get_ik_generator(robo_id, arm, target_pose, torso_limits=False)
    for j in range(500):
        solutions = next(generator)
        if solutions != []:
            if np.sum(np.square(np.array(base)-np.array(pre_pose)))>np.sum(np.square(np.array(solutions[0])-np.array(pre_pose))):
                if -0.715 <= solutions[0][1] <= 2.285 and -0.524 <= solutions[0][2] <= 1.396 and \
                -0.800 <= solutions[0][3] <= 3.900 and -2.321 <= solutions[0][4] <= 0.000 and \
                -3.142 <= solutions[0][5] <= 3.142 and -2.094 <= solutions[0][6] <= 0.000 and \
                -3.142 <= solutions[0][7] <= 3.142 and 0 <= solutions[0][0] <= 0.31:
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

def move_arm(q,base,picked):
    for i in range (1000):
        for j in range(len(q)):
            q[j] +=  base[j]
        p.setJointMotorControlArray(bodyIndex=pr2, jointIndices=list(torso_left), controlMode=p.POSITION_CONTROL, targetPositions = q,targetVelocities=[0]*len(q))
        if compute_dist()<0.005 and (picked == 0):
            close_gripper(left_gripper_joints)
            picked = 1
        check_collisions()
        p.stepSimulation()
        time.sleep(1./240.)
    return picked

def compute_dist():
    gripper1 = p.getLinkState(pr2,75)[0]
    #print(gripper1)
    gripper2 = p.getLinkState(pr2,77)[0]
    #print(gripper2)
    gripper_mid = (np.array(gripper1)+np.array(gripper2))/2
    plam_pos = p.getLinkState(pr2,70)[0]
    objects = p.getBasePositionAndOrientation(cube6_id)[0]
    #print(np.dot((gripper_mid-plam_pos)/np.linalg.norm(gripper_mid-plam_pos),(objects-gripper_mid)/np.linalg.norm(objects-gripper_mid)))
    return np.sqrt(np.sum(np.square(np.array(objects[0:2])-gripper_mid[0:2])))

def check_collisions():
    collisions = p.getContactPoints(pr2)
    print("+++++++++++++++")
    for i in range(len(collisions)):
        print(collisions[i][1],collisions[i][2],collisions[i][3],collisions[i][4])
        #return True
    print("+++++++++++++++")
    #return False

pi = 3.1415926/2
p.setJointMotorControlArray(bodyIndex=pr2, jointIndices=list(right_joints), controlMode=p.POSITION_CONTROL, targetPositions=[pi, 0, 0, 0, 0, 0, 0])
for j in range(500):
    p.setJointMotorControlArray(bodyIndex=pr2, jointIndices=list(torso_left), controlMode=p.POSITION_CONTROL, targetPositions=[0, 0, 0, 0, 0, 0, 0, 0])
    p.stepSimulation()
    time.sleep(1./240.)
#print(p.getJointStates(pr2,torso_left))

arm = 'left'
x = 0.6
y = 0.2
z = 0.9
target_pose = ((x, y, z), (1.414, -1.414, 0, 0))   
q = [0, 0, 0, 0, 0, 0, 0, 0]
base = move_control(pr2, arm, target_pose,q)
open_gripper(left_gripper_joints)
open_gripper(right_gripper_joints)
picked = move_arm(q,base,picked)


x = 0.6
y = 0.3
z = 0.72
target_pose = ((x, y, z), (1.414, -1.414, 0, 0))   
base = move_control(pr2, arm, target_pose,q)
picked = move_arm(q,base,picked)
#print(p.getJointStates(pr2,torso_left))
x = 0.6
y = 0.08
z = 0.72
target_pose = ((x, y, z), (1.414, -1.414, 0, 0))
base = move_control(pr2, arm, target_pose,q)
picked = move_arm(q,base,picked)



x = 0.6
y = 0.08
z = 0.75
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
close_gripper(right_gripper_joints)
picked = move_arm(q,base,picked)

x = 0.6
y = 0.08
z = 0.9
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
picked = move_arm(q,base,picked)

x = 0.8
y = 0.1
z = 0.9
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
picked = move_arm(q,base,picked)
open_gripper(left_gripper_joints)
open_gripper(right_gripper_joints)

x = 0.75
y = 0.1
z = 0.75
target_pose = ((x, y, z), (-1, 0, 0, 0))
base = move_control(pr2, arm, target_pose,q)
picked = move_arm(q,base,picked)

p.disconnect()
