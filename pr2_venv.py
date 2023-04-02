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
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.pr2 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [1.05, -0.2, 0.0], [0, 0, 0.7071, 0.7071],useFixedBase = 1)
        
        self.end_effector_index_l = []
        self.end_effector_index_r = []
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
        self.cube1_id = pybullet.loadURDF(CUBE_URDF_PATH, basePosition=self.initial_cube1_pos, globalScaling=0.05)
        self.cube2_id = pybullet.loadURDF(CUBE_URDF_PATH, basePosition=self.initial_cube2_pos, globalScaling=0.05)

        self.name = 'pr2GymEnv'
        self.place_pos = [0.9,0.0,0.9]
        self.is_pick_l = False
        self.is_pick_r = False
        
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
            linkstate0 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[0],computeForwardKinematics=True)
            linkstate1 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[1],computeForwardKinematics=True)
            linkstate2 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[2],computeForwardKinematics=True)
        if arm == "right":
            linkstate0 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[0],computeForwardKinematics=True)
            linkstate1 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[1],computeForwardKinematics=True)
            linkstate2 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[2],computeForwardKinematics=True)
        position, orientation = list((np.array(linkstate0[0])+np.array(linkstate1[0]))/2), linkstate2[1]
        return (position, orientation)


    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.is_pick_l = False
        self.is_pick_r = False
        self.pick_to_place_l = goal_distance(self.cube1_pos, self.place_pos)
        self.pick_to_place_r = goal_distance(self.cube2_pos, self.place_pos)

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
    
    
    def step(self, action_l,action_r):
        action_arm_l = np.array(action_l).astype(float)
        action_arm_r = np.array(action_r).astype(float)

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
        reward_left, reward_right = self.compute_reward(self.gripper_pos, self.desired_goal, None)
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
        #self.achieved_goal = np.array(np.concatenate((objects_pos, tool_pos)))
        #self.desired_goal = np.array(goal_pos)


    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        done_left = (self.terminated_l == True or self.stepCounter > self.maxSteps)
        done_right = (self.terminated_r == True or self.stepCounter > self.maxSteps)
        return done_left, done_right


    def compute_reward(self, grip_pos, desired_goal, info):
        rewards = np.zeros(2)

        self.pick_dist_l = goal_distance(grip_pos[0], self.cube1_pos)
        self.place_dist_l = goal_distance(grip_pos[0], self.place_pos)
        self.pick_dist_r = goal_distance(grip_pos[1], self.cube2_pos)
        self.place_dist_r = goal_distance(grip_pos[1], self.place_pos)

        reward += -self.target_dist * 10

        # task 0: reach object:
        if self.pick_dist_l < 0.02:
            self.is_pick_l = True
        if self.pick_dist_r < 0.02:
            self.is_pick_r = True
        if self.is_pick_l and self.place_dist_l < 0.05:
            self.terminated_l = True
        
        if self.is_pick_l:
            rewards[0] += -self.place_dist_l * 1.0
        else:
            rewards[0] += -self.pick_dist_l * 2.0 - self.pick_to_place_l
        
        if self.is_pick_r:
            rewards[1] += -self.place_dist_r * 1.0
        else:
            rewards[1] += -self.pick_dist_r * 2.0 - self.pick_to_place_r

        # check collisions:
        if self.check_collisions(): 
            reward += -1

        return rewards[0], rewards[1]