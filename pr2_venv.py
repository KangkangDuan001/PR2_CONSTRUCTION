# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet

import random
import time
import numpy as np
import sys
from gym import spaces
import gym

import os
import math 
import pybullet
import pybullet_data
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict

ROBOT_URDF_PATH = "/home/kk/pybullet-planning/models/drake/pr2_description/urdf/pr2_simplified.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)


class pr2GymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 # useIK=True,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=100,
                 # numControlledJoints=3, # XYZ, we use IK here!
                 simulatedGripper=False,
                 randObjPos=False,
                 task=0, # here target number
                 learning_param=0):

        self.renders = renders
        self.actionRepeat = actionRepeat

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-10)
        pybullet.setRealTimeSimulation(False)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME,1)
        pybullet.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        
        # setup robot arm:

        self.plane = p.loadURDF(PLANE_URDF_PATH) # add ground
        startPos = [0,0,0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.pr2 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [1.05, -0.2, 0.0], [0, 0, 0.7071, 0.7071],useFixedBase = 1)
        
        self.end_effector_index_l = 
        self.end_effector_index_r = 
        self.joints_list_r = [40, 41, 42, 44, 45, 47, 48]
        self.joints_list_l = [61, 62, 63, 65, 66, 68, 69]
        self.control_joints_l = ["l_shoulder_pan_joint", "l_shoulder_lift_joint", "l_upper_arm_roll_joint", "l_elbow_flex_joint", "l_forearm_roll_joint", "l_wrist_flex_joint","l_wrist_roll_joint"]
        self.control_joints_r = ["r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint","r_wrist_roll_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints_l = AttrDict()
        self.joints_r = AttrDict()
        get_joints_lists(joints_list_r,"right"):
        get_joints_lists(joints_list_l,"left"):

        # object:
        self.initial_cube1_pos = [0.6, -0.1, 0.65] # initial object pos
        self.initial_cube2_pos = [0.6, -0.3, 0.65] # initial object pos
        self.cube1_id = p.loadURDF(CUBE_URDF_PATH, basePosition=initial_cube1_pos, globalScaling=0.05)
        self.cube2_id = p.loadURDF(CUBE_URDF_PATH, basePosition=initial_cube2_pos, globalScaling=0.05)

        self.name = 'pr2GymEnv'
        
        self.simulatedGripper = simulatedGripper
        self.action_dim = 4
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)

        self.task = task
        self.learning_param = learning_param
     
        self._action_bound = 1.0 # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
        high = np.array([10]*self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')

    def get_joints_lists(self,joint_lists,arm):
        
        for i in joint_lists:
            info = pybullet.getJointInfo(self.pr2, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True 
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.pr2, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
        if arm == "left":
            self.joints_l[info.name] = info
        if arm == "rignt":
            self.joints_r[info.name] = info

    def set_joint_angles(self, joint_angles,arm):
        poses = []
        indexes = []
        forces = []
        if arm == "left":
            control_joints = self.control_joints_l
        if arm == "rignt":
            control_joints = self.control_joints_r

        for i, name in enumerate(control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.pr2, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )

    def get_joint_angles(self,arm):
        if arm == "left":
            j = pybullet.getJointStates(self.pr2, self.joints_list_l)
        if arm == "rignt":
            j = pybullet.getJointStates(self.pr2, self.joints_list_r)
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False

       
        
    def get_current_pose(self,arm):
        if arm == "left":
            linkstate = pybullet.getLinkState(self.pr2, self.end_effector_index_l)
        if arm == "right":
            linkstate = pybullet.getLinkState(self.pr2, self.end_effector_index_r)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)


    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.pr2_or = [0.0, 1/2*math.pi, 0.0]


        pybullet.resetBasePositionAndOrientation(self.cube1_id, self.initial_obj_pos, [0.,0.,0.,1.0]) # reset object pos
        pybullet.resetBasePositionAndOrientation(self.cube2_id, self.initial_obj_pos, [0.,0.,0.,1.0]) # reset object pos

        # reset robot simulation and position:
        joint_angles = (0, 0, 0, 0, 0, 0) 
        self.set_joint_angles(joint_angles,"left")
        self.set_joint_angles(joint_angles,"right")

        # step simualator:
        for i in range(100):
            pybullet.stepSimulation()

        # get obs and return:
        self.getExtendedObservation()
        return self.observation
    
    
    def step(self, action):
        action = np.array(action)
        arm_action = action[0:self.action_dim-1].astype(float) # dX, dY, dZ - range: [-1,1]
        gripper_action = action[self.action_dim-1].astype(float) # gripper - range: [-1=closed,1=open]

        # get current position:
        cur_p = self.get_current_pose()
        # add delta position:
        new_p = np.array(cur_p[0]) + arm_action
        # actuate: 
        joint_angles = self.calculate_ik(new_p, self.ur5_or) # XYZ and angles set to zero
        self.set_joint_angles(joint_angles)
        
        # step simualator:
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)
        
        self.getExtendedObservation()
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, None)
        done = self.my_task_done()

        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True

        self.stepCounter += 1

        return self.observation, reward, done, info


    # observations are: arm (tip/tool) position, arm acceleration, ...
    def getExtendedObservation(self):

        tool_pos_l = self.get_current_pose("left")[0]
        tool_rot_l = self.get_current_pose("left")[1]
        tool_pos_r = self.get_current_pose("right")[0]
        tool_rot_r = self.get_current_pose("right")[1]
        self.cube1_pos,_ = pybullet.getBasePositionAndOrientation(self.cube1_id)
        self.cube2_pos,_ = pybullet.getBasePositionAndOrientation(self.cube2_id)
        cube_1_pos = self.cube1_pos
        cube_2_pos = self.cube2_pos       
        goal_pos = self.goal_pos

        self.observation = np.array(np.concatenate((tool_pos_l,tool_rot_l,tool_pos_r,tool_rot_r, cube_1_pos, cube_2_pos, )))
        self.achieved_goal = np.array(np.concatenate((objects_pos, tool_pos)))
        self.desired_goal = np.array(goal_pos)


    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c


    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)
 
        grip_pos = achieved_goal[-3:]
            
        self.target_dist = goal_distance(grip_pos, desired_goal)
        # print(grip_pos, desired_goal, self.target_dist)

        # check approach velocity:
        # tv = self.tool.getVelocity()
        # approach_velocity = np.sum(tv)

        # print(approach_velocity)
        # input()

        reward += -self.target_dist * 10

        # task 0: reach object:
        if self.target_dist < self.learning_param:# and approach_velocity < 0.05:
            self.terminated = True
            # print('Successful!')

        # penalize if it tries to go lower than desk / platform collision:
        # if grip_trans[1] < self.desired_goal[1]-0.08: # lower than position of object!
            # reward[i] += -1
            # print('Penalty: lower than desk!')

        # check collisions:
        if self.check_collisions(): 
            reward += -1
            # print('Collision!')

        # print(target_dist, reward)
        # input()

        return reward