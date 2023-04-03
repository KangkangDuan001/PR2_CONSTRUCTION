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

ROBOT_URDF_PATH = "pr2_simplified.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table.urdf")
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "pr2_simplified.urdf")
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
                 actionRepeat=1,
                 renders=False,
                 maxSteps=2000,
                 simulatedGripper=False,
                 randObjPos=False,
                 task=0, 
                 learning_param=0):

        self.renders = renders

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        self.actionRepeat = actionRepeat

        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-10)
        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        
        # setup robot arm:

        self.plane = pybullet.loadURDF(PLANE_URDF_PATH) # add ground
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.pr2 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [1.05, -0.2, 0.0], [0, 0, 0.7071, 0.7071],useFixedBase = 1)
        
        self.end_effector_index_l = [75,77]
        self.end_effector_index_r = [54,56]
        self.joints_list_r = [40, 41, 42, 44, 45, 47, 48]
        self.joints_list_l = [61, 62, 63, 65, 66, 68, 69]
        self.control_joints_l = ["l_shoulder_pan_joint", "l_shoulder_lift_joint", "l_upper_arm_roll_joint", "l_elbow_flex_joint", "l_forearm_roll_joint", "l_wrist_flex_joint","l_wrist_roll_joint"]
        self.control_joints_r = ["r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint","r_wrist_roll_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints_l = AttrDict()
        self.joints_r = AttrDict()
        self.get_joints_lists(self.joints_list_r,"right")
        self.get_joints_lists(self.joints_list_l,"left")

        # object:
        self.initial_cube1_pos = [0.6, 0.1, 0.65] # initial object pos
        self.initial_cube2_pos = [0.6, -0.3, 0.65] # initial object pos
        self.cube1_id = pybullet.loadURDF(CUBE_URDF_PATH, basePosition=self.initial_cube1_pos, globalScaling=0.8)
        self.cube2_id = pybullet.loadURDF(CUBE_URDF_PATH, basePosition=self.initial_cube2_pos, globalScaling=0.8)

        self.name = 'pr2GymEnv'
        self.place_pos = [0.9,0.0,0.9]
        self.is_pick_l = False
        self.is_pick_r = False
        self.left_gripper_joints = [74,76,75,77]
        self.right_gripper_joints = [53,55,54,56]
        
        self.simulatedGripper = simulatedGripper
        self.action_dim = 14
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated_r = False
        self.terminated_l = False
        self.randObjPos = randObjPos

        self.task = task
        self.learning_param = learning_param
     
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float64)
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64)

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
            for i, name in enumerate(control_joints):
                joint = self.joints_l[name]
                poses.append(joint_angles[i])
                indexes.append(joint.id)
                forces.append(joint.maxForce)
        if arm == "rignt":
            control_joints = self.control_joints_r
            for i, name in enumerate(control_joints):
                joint = self.joints_r[name]
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
            return True
        return False

       
        
    def get_current_pose(self,arm):
        if arm == "left":
            linkstate0 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[0],computeForwardKinematics=True)
            linkstate1 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[1],computeForwardKinematics=True)
            linkstate2 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[2],computeForwardKinematics=True)
        if arm == "right":
            linkstate0 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[0],computeForwardKinematics=True)
            linkstate1 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[1],computeForwardKinematics=True)
            linkstate2 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[2],computeForwardKinematics=True)
        position, orientation = (np.array(linkstate0[0])+np.array(linkstate1[0]))/2, linkstate2[1]
        return (position, orientation)

    def close_gripper(self, gripper_joints):
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[0], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[1], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[2], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[3], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10) 

    def open_gripper(self, gripper_joints):
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[0], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[1], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[2], controlMode=pybullet.POSITION_CONTROL,targetPosition = -1)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[3], controlMode=pybullet.POSITION_CONTROL,targetPosition = -1)

    def reset(self):
        self.stepCounter = 0
        self.terminated_r = False
        self.terminated_l = False
        self.is_pick_l = False
        self.is_pick_r = False
        self.pick_to_place_l = goal_distance(self.cube1_pos, self.place_pos)
        self.pick_to_place_r = goal_distance(self.cube2_pos, self.place_pos)
        self.open_gripper(self.left_gripper_joints)
        self.open_gripper(self.right_gripper_joints)

        pybullet.resetBasePositionAndOrientation(self.cube1_id, self.initial_cube1_pos, [0.,0.,0.,1.0]) # reset object pos
        pybullet.resetBasePositionAndOrientation(self.cube2_id, self.initial_cube2_pos, [0.,0.,0.,1.0]) # reset object pos
        pybullet.resetBasePositionAndOrientation(self.pr2, [0, 0, 0], [0.,0.,0.,1.0]) # reset object pos

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
        action_arm_l = np.array(action[0:7]).astype(float)
        action_arm_r = np.array(action[7:14]).astype(float)

        # get current position:
        cur_p_l = pybullet.getJointStates(self.pr2,self.joints_list_l)[0]
        cur_p_r = pybullet.getJointStates(self.pr2,self.joints_list_r)[0]
        
        # add delta position:
        new_p_l = np.array(cur_p_l[0]) + action_arm_l
        new_p_r = np.array(cur_p_r[0]) + action_arm_r
        
        # actuate: 
        self.set_joint_angles(new_p_l,"left")
        self.set_joint_angles(new_p_r,"right")
        
        # step simualator:
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)
        
        self.getExtendedObservation()
        reward_left, reward_right = self.compute_reward(self.gripper_pos)
        done_left, done_right = self.my_task_done()

        info = {'is_success': False}
        if self.terminated_l and self.terminated_r:
            info['is_success'] = True

        self.stepCounter += 1
        return self.observation, reward_left, reward_right, done_left, done_right, info


    # observations are: arm (tip/tool) position, arm acceleration, ...
    def getExtendedObservation(self):

        tool_pos_l = self.get_current_pose("left")[0]
        tool_rot_l = self.get_current_pose("left")[1]
        tool_pos_r = self.get_current_pose("right")[0]
        tool_rot_r = self.get_current_pose("right")[1]
        self.gripper_pos = (tool_pos_l,tool_pos_r)
        self.cube1_pos,_ = pybullet.getBasePositionAndOrientation(self.cube1_id)
        self.cube2_pos,_ = pybullet.getBasePositionAndOrientation(self.cube2_id)
        cube_1_pos = self.cube1_pos
        cube_2_pos = self.cube2_pos
        joints_state_l = pybullet.getJointStates(self.pr2,self.joints_list_l)
        joints_state_r = pybullet.getJointStates(self.pr2,self.joints_list_r)
        joints_pos_l = joints_state_l[0]
        joints_pos_r = joints_state_r[0]
        joints_vel_l = joints_state_l[1]
        joints_vel_r = joints_state_r[1]

        self.observation = np.array(np.concatenate((tool_pos_l,tool_rot_l,tool_pos_r,tool_rot_r, cube_1_pos, cube_2_pos,joints_pos_l,joints_pos_r,joints_vel_l,joints_vel_r )))

    def my_task_done(self):
        done_left = (self.terminated_l == True or self.stepCounter > self.maxSteps)
        done_right = (self.terminated_r == True or self.stepCounter > self.maxSteps)
        return done_left, done_right


    def compute_reward(self, grip_pos):
        rewards = np.zeros(2)

        self.pick_dist_l = goal_distance2d(grip_pos[0], np.array(self.cube1_pos))
        self.place_dist_l = goal_distance(grip_pos[0], np.array(self.place_pos))
        self.pick_dist_r = goal_distance2d(grip_pos[1], np.array(self.cube2_pos))
        self.place_dist_r = goal_distance(grip_pos[1], np.array(self.place_pos))
        hight_dis_l = abs(grip_pos[0][2] - self.cube1_pos[2])
        hight_dis_r = abs(grip_pos[1][2] - self.cube2_pos[2])

        reward += -self.target_dist * 10

        # task 0: reach object:
        if self.pick_dist_l < 0.005 and self.is_pick_l == False and hight_dis_l < 0.02:
            self.close_gripper(self.left_gripper_joints)
            self.is_pick_l = True
        
        if self.pick_dist_r < 0.005 and self.is_pick_r == False and hight_dis_r < 0.02:
            self.is_pick_r = True
            self.close_gripper(self.right_gripper_joints)

        if self.is_pick_l and self.place_dist_l < 0.05:
            self.terminated_l = True

        if self.is_pick_r and self.place_dist_r < 0.05:
            self.terminated_r = True
        
        if self.is_pick_l:
            rewards[0] += -self.place_dist_l * 1.0
        else:
            rewards[0] += -self.pick_dist_l * 2.0 - self.pick_to_place_l
        
        if self.is_pick_r:
            rewards[1] += -self.place_dist_r * 1.0
        else:
            rewards[1] += -self.pick_dist_r * 2.0 - self.pick_to_place_r

        # check collisions:
        #if self.check_collisions(): 
            #reward += -1

        return rewards[0], rewards[1]