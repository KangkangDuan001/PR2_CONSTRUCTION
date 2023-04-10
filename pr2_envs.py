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
from gym.utils import seeding

ROBOT_URDF_PATH = "/home/kk/pybullet-planning/models/drake/pr2_description/urdf/pr2_simplified.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
PLAT_URDF_PATH = "/home/kk/pybullet-planning/models/cup_targ.urdf"
CUBE_URDF_PATH = "/home/kk/pybullet-planning/models/drake/objects/simple_cylinder.urdf"
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
                 actionRepeat=50,
                 renders=False,
                 maxSteps=20,
                 num_envs = 1):

        self.renders = renders
        self.num_envs = num_envs

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        
        self.actionRepeat = actionRepeat

        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-9.8)
        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=90, cameraPitch=-60, cameraTargetPosition=[0,0,0])
        
        # setup robot arm:

        self.plane = pybullet.loadURDF(PLANE_URDF_PATH) # add ground
        flags = pybullet.URDF_USE_SELF_COLLISION
        
        self.initial_cube1_pos = [0.75, 0.0, 0.7]
        self.initial_cube2_pos = [0.75, 0.0, 0.7]
        self.initial_robo_pos = [0.0, 0.0, 0.0]
        self.initial_table_pos = [1.05, 0, 0.0]
        self.initial_plat_pos = [0.75, 0.0, 0.66]
        self.robos = []
        self.tables = []
        self.cubes1 = []
        self.cubes2 = []
        self.plats = []
        self.origins = []
        self.target_pos = []

        for i in range(self.num_envs):
            self.robos.append(pybullet.loadURDF(ROBOT_URDF_PATH, self.list_add(self.initial_robo_pos,[self.num_envs/2+i, 0, 0]), [0, 0, 0, 1], flags=flags)) 
            self.tables.append(pybullet.loadURDF(TABLE_URDF_PATH, self.list_add(self.initial_table_pos,[self.num_envs/2+i, 0, 0]), [0, 0, 0.7071, 0.7071],useFixedBase=True))
            self.cubes1.append(pybullet.loadURDF(CUBE_URDF_PATH, self.list_add(self.initial_cube1_pos,[self.num_envs/2+i, 0, 0]), globalScaling=0.8))
            self.cubes2.append(pybullet.loadURDF(CUBE_URDF_PATH, self.list_add(self.initial_cube2_pos,[self.num_envs/2+i, 0, 0]), globalScaling=0.8))
            self.plats.append(pybullet.loadURDF(PLAT_URDF_PATH, self.list_add(self.initial_plat_pos,[self.num_envs/2+i, 0, 0]), globalScaling=1,useFixedBase = 1))
            self.origins.append(self.list_add([0,0,0],[self.num_envs/2+i, 0, 0]))
            self.target_pos.append(self.list_add([0.8,0,1.0],[self.num_envs/2+i, 0, 0]))

        self.left_gripper_joints = [74,76,75,77]
        self.right_gripper_joints = [53,55,54,56]
        self.end_effector_index_l = [75,77,70]
        self.end_effector_index_r = [54,56,49]
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

        self.bad_tran_l = np.zeros(self.num_envs)
        self.bad_tran_r = np.zeros(self.num_envs)
        self.is_pick_l = np.zeros(self.num_envs)
        self.is_pick_r = np.zeros(self.num_envs)
        self.cube1_pos = []
        self.cube2_pos = []
        self.stepCounter_l = np.zeros(self.num_envs)
        self.stepCounter_r = np.zeros(self.num_envs)
        self.terminated_r = np.zeros(self.num_envs)
        self.terminated_l = np.zeros(self.num_envs)
        self.l_collision = np.zeros(self.num_envs)
        self.r_collision = np.zeros(self.num_envs)
        self.gripper_pos = [0 for i in range(self.num_envs)]
        self.gripper_rot = [0 for i in range(self.num_envs)]
        self.pick_dist_l = np.zeros(self.num_envs)
        self.pick_dist_r = np.zeros(self.num_envs)
        self.place_dist_l = np.zeros(self.num_envs)
        self.place_dist_r = np.zeros(self.num_envs)

        for i in range(self.num_envs):
            self.set_mass(self.robos[i])
            self.set_collisions_pairs(self.robos[i],self.cubes1[i],self.cubes2[i])
            self.cube1_pos.append(pybullet.getBasePositionAndOrientation(self.cubes1[i])[0])
            self.cube2_pos.append(pybullet.getBasePositionAndOrientation(self.cubes2[i])[0])

        self.name = 'pr2GymEnv'
        self.maxSteps = maxSteps
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float64)
        self.reset()
    
    def set_collisions_pairs(self,id,cubes1,cubes2):
        for i in range(7):
            for j in range(7):
                pybullet.setCollisionFilterPair(id,id,self.joints_list_r[i],self.joints_list_l[j],enableCollision=1)
        for i in range(44,82):
            pybullet.setCollisionFilterPair(id,cubes1,i,-1,enableCollision=1)
            pybullet.setCollisionFilterPair(id,cubes2,i,-1,enableCollision=1)

    def set_mass(self,id):
        lit_mass = [50,52,59,60,71,73,80,81]
        for i in lit_mass:
            pybullet.changeDynamics(bodyUniqueId = id,linkIndex=i,mass=0.0)
        for i in [-1,0,1,2,3]:
            pybullet.changeDynamics(bodyUniqueId = id,linkIndex=i,mass=99999.0)

    def get_joints_lists(self,joint_lists,arm):
        if arm == "left":
            for i in joint_lists:
                info = pybullet.getJointInfo(self.robos[0], i)
                jointID = info[0]
                jointName = info[1].decode("utf-8")
                jointType = self.joint_type_list[info[2]]
                jointLowerLimit = info[8]
                jointUpperLimit = info[9]
                jointMaxForce = info[10]
                jointMaxVelocity = info[11]
                controllable = True 
                info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
                self.joints_l[info.name] = info
        if arm == "right":
            for i in joint_lists:
                info = pybullet.getJointInfo(self.robos[0], i)
                jointID = info[0]
                jointName = info[1].decode("utf-8")
                jointType = self.joint_type_list[info[2]]
                jointLowerLimit = info[8]
                jointUpperLimit = info[9]
                jointMaxForce = info[10]
                jointMaxVelocity = info[11]
                controllable = True 
                info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
                self.joints_r[info.name] = info

    def set_joint_angles(self, joint_angles,arm,id):
        poses = []
        indexes = []
        forces = []
        if arm == "left":
            control_joints = self.control_joints_l
            for i, name in enumerate(control_joints):
                joint = self.joints_l[name]
                poses.append(joint_angles[i])
                indexes.append(joint.id)
                forces.append(joint.maxForce*2)
        if arm == "right":
            control_joints = self.control_joints_r
            for i, name in enumerate(control_joints):
                joint = self.joints_r[name]
                poses.append(joint_angles[i])
                indexes.append(joint.id)
                forces.append(joint.maxForce*2)
        pybullet.setJointMotorControlArray(
            id, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.02]*len(poses),
            forces = forces
        )

    def get_joint_angles(self,arm,id):
        if arm == "left":
            j = pybullet.getJointStates(id, self.joints_list_l)
        if arm == "right":
            j = pybullet.getJointStates(id, self.joints_list_r)
        joints = [i[0] for i in j]
        return joints
         
    def list_add(self,list1,list2):
        res = []
        if len(list1) == len(list2):
            for i in range(len(list1)):
                res.append(list1[i] + list2[i])
            return res
        else:
            return []
    def list_min(self,list1,list2):
        res = []
        if len(list1) == len(list2):
            for i in range(len(list1)):
                res.append(list1[i] - list2[i])
            return res
        else:
            return []
    def get_current_pose(self,arm,id):
        if arm == "left":
            linkstate0 = pybullet.getLinkState(id, self.end_effector_index_l[0])
            linkstate1 = pybullet.getLinkState(id, self.end_effector_index_l[1])
            linkstate2 = pybullet.getLinkState(id, self.end_effector_index_l[2])
        if arm == "right":
            linkstate0 = pybullet.getLinkState(id, self.end_effector_index_r[0])
            linkstate1 = pybullet.getLinkState(id, self.end_effector_index_r[1])
            linkstate2 = pybullet.getLinkState(id, self.end_effector_index_r[2])
        position, orientation = list((np.array(linkstate0[0])+np.array(linkstate1[0]))/2), linkstate2[1]
        return (position, orientation)

    def close_gripper(self, gripper_joints,arm,gripper_pos,gripper_rot,id,cube1,cube2):
        if arm == "left":
            pybullet.resetBasePositionAndOrientation(cube1,gripper_pos,gripper_rot)
            #print(self.gripper_pos)
        if arm == "right":
            pybullet.resetBasePositionAndOrientation(cube2,gripper_pos,gripper_rot)
            #print(self.gripper_pos)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[0], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[1], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[2], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[3], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0,force=10) 

    def open_gripper(self, gripper_joints,id):
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[0], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[1], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[2], controlMode=pybullet.POSITION_CONTROL,targetPosition = -1)
        pybullet.setJointMotorControl2(bodyIndex=id, jointIndex=gripper_joints[3], controlMode=pybullet.POSITION_CONTROL,targetPosition = -1)
    
    def reset_body(self, body, fixed=False):
        for joint in self.get_joints(body):
            if joint == 17:
                pybullet.resetJointState(bodyUniqueId=body, jointIndex=17, targetValue=0.2)
                pybullet.setJointMotorControl2(bodyIndex=body, jointIndex=17, controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.2)
            else:
                if self.is_movable(body, joint):
                    pybullet.resetJointState(bodyUniqueId=body, jointIndex=joint, targetValue = 0)

    def reset_body_base(self, body, fixed=False):
        for joint in self.get_joints(body):
            if joint == 17:
                pybullet.resetJointState(bodyUniqueId=body, jointIndex=17, targetValue=0.2)
                pybullet.setJointMotorControl2(bodyIndex=body, jointIndex=17, controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.2)
            else:
                if self.is_movable(body, joint):
                    if joint in self.joints_list_l+self.joints_list_r+self.left_gripper_joints+self.right_gripper_joints:
                        pass
                    else:
                        pybullet.resetJointState(bodyUniqueId=body, jointIndex=joint, targetValue = 0)

    def reset_body_l(self, body):
        for joint in self.joints_list_l:
            pybullet.resetJointState(bodyUniqueId=body, jointIndex=joint, targetValue = 0)

    def reset_body_r(self, body):
        for joint in self.joints_list_r:
            pybullet.resetJointState(bodyUniqueId=body, jointIndex=joint, targetValue = 0)

    def is_movable(self,body, joint):
        return not self.is_fixed(body, joint)

    def is_fixed(self,body, joint):
        return self.get_joint_type(body, joint) == pybullet.JOINT_FIXED

    def get_joint_type(self,body, joint):
        return self.get_joint_info(body, joint).jointType

    def get_joint_info(self,body, joint):
        JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])
        return JointInfo(*pybullet.getJointInfo(body, joint))

    def get_num_joints(self,body):
        return pybullet.getNumJoints(body)

    def get_joints(self,body):
        return list(range(self.get_num_joints(body)))

    def check_collisions(self,id):
        collisions = pybullet.getContactPoints(id,id)
        return len(collisions)

    def check_collisions_r(self,id,cube1,cube2):
        collisions_cube1 = 0
        collisions_cube2 = 0
        for i in range(44,61):
            collisions_cube1 += len(pybullet.getClosestPoints(id,cube1,distance = 0.01, linkIndexA = i))
            collisions_cube2 += len(pybullet.getClosestPoints(id,cube2,distance = 0.01, linkIndexA = i))
        return collisions_cube1 + collisions_cube2

    def check_collisions_l(self,id,cube1,cube2):
        collisions_cube1 = 0
        collisions_cube2 = 0
        for i in range(65,82):
            collisions_cube1 += len(pybullet.getClosestPoints(id,cube1,distance = 0.01, linkIndexA = i))
            collisions_cube2 += len(pybullet.getClosestPoints(id,cube2,distance = 0.01, linkIndexA = i))
        return collisions_cube1 + collisions_cube2

    def reset(self,seed=None):
        self.bad_tran_l = np.zeros(self.num_envs)
        self.bad_tran_r = np.zeros(self.num_envs)
        self.is_pick_l = np.zeros(self.num_envs)
        self.is_pick_r = np.zeros(self.num_envs)
        self.stepCounter_l = np.zeros(self.num_envs)
        self.stepCounter_r = np.zeros(self.num_envs)
        self.terminated_r = np.zeros(self.num_envs)
        self.terminated_l = np.zeros(self.num_envs)
        self.l_collision = np.zeros(self.num_envs)
        self.r_collision = np.zeros(self.num_envs)
        self.pick_to_place_l = np.zeros(self.num_envs)
        self.pick_to_place_r = np.zeros(self.num_envs)
        observation = []
        for i in range(self.num_envs):
            self.open_gripper(self.left_gripper_joints,self.robos[i])
            self.open_gripper(self.right_gripper_joints,self.robos[i])
            new_pos_l = self.list_add(self.list_add(self.initial_cube1_pos,[self.num_envs/2+i, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
            new_pos_r = self.list_add(self.list_add(self.initial_cube2_pos,[self.num_envs/2+i, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
            while goal_distance(np.array(new_pos_l), np.array(new_pos_r)) < 0.1:
                new_pos_r = self.list_add(self.list_add(self.initial_cube2_pos,[self.num_envs/2+i, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])

            pybullet.resetBasePositionAndOrientation(self.cubes1[i], new_pos_l, [0.,0.,0.,1.0]) # reset object pos
            pybullet.resetBasePositionAndOrientation(self.cubes2[i], new_pos_r, [0.,0.,0.,1.0]) # reset object pos

            self.pick_to_place_l[i] = goal_distance(np.array(new_pos_l), np.array(self.target_pos[i]))
            self.pick_to_place_r[i] = goal_distance(np.array(new_pos_r), np.array(self.target_pos[i]))
            self.reset_body(self.robos[i])
            observation.append(self.getExtendedObservation(i))
        return np.array(observation)
    
    def reset_l(self,num):
        self.l_collision = np.zeros(self.num_envs)
        self.stepCounter_l = np.zeros(self.num_envs)
        self.terminated_l = np.zeros(self.num_envs)
        self.bad_tran_l = np.zeros(self.num_envs)
        self.is_pick_l = np.zeros(self.num_envs)
        self.open_gripper(self.left_gripper_joints,self.robos[num])
        new_pos_l = self.list_add(self.list_add(self.initial_cube1_pos,[self.num_envs/2+num, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        while goal_distance(np.array(new_pos_l), np.array(pybullet.getBasePositionAndOrientation(self.cubes2[num])[0])) < 0.1:
            new_pos_l = self.list_add(self.list_add(self.initial_cube1_pos,[self.num_envs/2+num, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        pybullet.resetBasePositionAndOrientation(self.cubes1[num], new_pos_l, [0.,0.,0.,1.0]) # reset object pos
        self.pick_to_place_l[num] = goal_distance(np.array(new_pos_l), np.array(self.target_pos[num]))
        self.reset_body_l(self.robos[num])

    def reset_r(self,i):
        self.r_collision = np.zeros(self.num_envs)
        self.stepCounter_r = np.zeros(self.num_envs)
        self.terminated_r = np.zeros(self.num_envs)
        self.bad_tran_r = np.zeros(self.num_envs)
        self.is_pick_r = np.zeros(self.num_envs)
        self.open_gripper(self.right_gripper_joints,self.robos[i])
        new_pos_r = self.list_add(self.list_add(self.initial_cube2_pos,[self.num_envs/2+i, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        while goal_distance(np.array(new_pos_r), np.array(pybullet.getBasePositionAndOrientation(self.cubes1[i])[0])) < 0.1:
            new_pos_r = self.list_add(self.list_add(self.initial_cube2_pos,[self.num_envs/2+i, 0, 0]), [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        pybullet.resetBasePositionAndOrientation(self.cubes2[i], new_pos_r, [0.,0.,0.,1.0]) # reset object pos
        self.pick_to_place_r[i] = goal_distance(np.array(new_pos_r), np.array(self.target_pos[i]))
        self.reset_body_r(self.robos[i])

    def step(self, action,num):
        action_arm_l = np.zeros(7)
        action_arm_r = np.zeros(7)
        action_arm_l[0] = np.clip(np.array(action[0]).astype(float), -3.142, 3.142)
        action_arm_l[1] = np.clip(np.array(action[1]).astype(float), -3.142, 3.142)
        action_arm_l[2] = np.clip(np.array(action[2]).astype(float), -3.142, 3.142)
        action_arm_l[3] = np.clip(np.array(action[3]).astype(float), -3.142, 3.142)
        action_arm_l[4] = np.clip(np.array(action[4]).astype(float), -3.142, 3.142)
        action_arm_l[5] = np.clip(np.array(action[5]).astype(float), -3.142, 3.142)
        action_arm_l[6] = np.clip(np.array(action[6]).astype(float), -3.142, 3.142)

        action_arm_r[0] = np.clip(np.array(action[7]).astype(float), -3.142, 3.142)
        action_arm_r[1] = np.clip(np.array(action[8]).astype(float), -3.142, 3.142)
        action_arm_r[2] = np.clip(np.array(action[9]).astype(float), -3.142, 3.142)
        action_arm_r[3] = np.clip(np.array(action[10]).astype(float), -3.142, 3.142)
        action_arm_r[4] = np.clip(np.array(action[11]).astype(float), -3.142, 3.142)
        action_arm_r[5] = np.clip(np.array(action[12]).astype(float), -3.142, 3.142)
        action_arm_r[6] = np.clip(np.array(action[13]).astype(float), -3.142, 3.142) 

        # actuate: 
        self.set_joint_angles(action_arm_l,"left",self.robos[num])
        self.set_joint_angles(action_arm_r,"right",self.robos[num])

        observation = self.getExtendedObservation(num)

        distance = goal_distance(np.array(pybullet.getBasePositionAndOrientation(self.robos[num])[0]), np.array(self.list_add(self.initial_robo_pos,[self.num_envs/2+num, 0, 0])))
        error = goal_distance(np.array(pybullet.getBasePositionAndOrientation(self.robos[num])[1]), np.array([0,0,0,1]))
        
        if distance > 0.01 or error > 0.01:
            pybullet.resetBasePositionAndOrientation(self.robos[num],self.list_add(self.initial_robo_pos,[self.num_envs/2+num, 0, 0]),[0,0,0,1])
            print("error")

        reward_left, reward_right = self.compute_reward(self.gripper_pos[num],self.gripper_rot[num],self.robos[num])
        done_left, done_right = self.my_task_done(num)

        info = {'is_success': False,'episode_l': reward_left, 'episode_r': reward_right}
        if self.terminated_l[num] == 1 and self.terminated_r[num] == 1:
            info['is_success'] = True
        if self.bad_tran_l[num] == 1:
            info['bad_transition_l'] = True
        if self.bad_tran_r[num] == 1:
            info['bad_transition_r'] = True

        self.stepCounter_l[num] += 1
        self.stepCounter_r[num] += 1
        return observation, reward_left, reward_right, done_left, done_right, info
    
    def setp_envs(self,actions):
        observations = []
        rewards_left = []
        rewards_right = []
        dones_left = []
        dones_right = []
        infos = []
        for i in range(self.num_envs):
            observation, reward_left, reward_right, done_left, done_right, info = self.step(actions[i,:],i)
            observations.append(observation)
            rewards_left.append(reward_left)
            rewards_right.append(reward_right)
            dones_left.append(done_left)
            dones_right.append(done_right)
            infos.append(info)
        for i in range(50):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)
        return np.array(observations), np.array(rewards_left).reshape(-1,1), np.array(rewards_right).reshape(-1,1), np.array(dones_left).reshape(-1,1), np.array(dones_right).reshape(-1,1), infos
    # observations are: arm (tip/tool) position, arm acceleration, ...
    
    def getExtendedObservation(self,num):

        tool_pos_l = self.get_current_pose("left",self.robos[num])[0]
        tool_rot_l = self.get_current_pose("left",self.robos[num])[1]
        tool_pos_r = self.get_current_pose("right",self.robos[num])[0]
        tool_rot_r = self.get_current_pose("right",self.robos[num])[1]
        self.gripper_pos[num] = (tool_pos_l,tool_pos_r)
        self.gripper_rot[num] = (tool_rot_l,tool_rot_r)
        tool_pos_l = self.list_min(tool_pos_l,self.origins[num])
        tool_pos_r = self.list_min(tool_pos_r,self.origins[num])

        self.cube1_pos[num] = pybullet.getBasePositionAndOrientation(self.cubes1[num])[0]
        self.cube2_pos[num] = pybullet.getBasePositionAndOrientation(self.cubes2[num])[0]
        cube_1_pos = self.list_min(self.cube1_pos[num],self.origins[num])
        cube_2_pos = self.list_min(self.cube2_pos[num],self.origins[num])
        joints_state_l = pybullet.getJointStates(self.robos[num],self.joints_list_l)
        joints_state_r = pybullet.getJointStates(self.robos[num],self.joints_list_r)
        bot_base_pos,bot_base_rot = pybullet.getBasePositionAndOrientation(self.pr2)
        bot_base_pos = self.list_min(bot_base_pos,self.origins[num])
        observation = np.array(np.concatenate((bot_base_pos,bot_base_rot,tool_pos_l,tool_rot_l,tool_pos_r,tool_rot_r,cube_1_pos, cube_2_pos,\
            [joints_state_l[0][0]/3.142,joints_state_l[1][0]/3.142,\
            joints_state_l[2][0]/3.142,joints_state_l[3][0]/3.142,\
            joints_state_l[4][0]/3.142,joints_state_l[5][0]/3.142,\
            joints_state_l[6][0]/3.142,joints_state_r[0][0]/3.142,\
            joints_state_r[1][0]/3.142,joints_state_r[2][0]/3.142,\
            joints_state_r[3][0]/3.142,joints_state_r[4][0]/3.142,\
            joints_state_r[5][0]/3.142,joints_state_r[6][0]/3.142]
            )))
        return observation

    def my_task_done(self,num):
        done_left = (self.terminated_l[num] == True or self.stepCounter_l[num] > self.maxSteps)
        done_right = (self.terminated_r[num] == True or self.stepCounter_r[num] > self.maxSteps)
        return done_left, done_right


    def compute_reward(self, grip_pos,gripper_rot,num):
        rewards = np.zeros(2)
        self.pick_dist_l[num] = goal_distance2d(np.array(grip_pos[0]), np.array(self.cube1_pos[num]))
        self.place_dist_l[num] = goal_distance(np.array(grip_pos[0]), np.array(self.target_pos[num]))
        self.pick_dist_r[num] = goal_distance2d(np.array(grip_pos[1]), np.array(self.cube2_pos[num]))
        self.place_dist_r[num] = goal_distance(np.array(grip_pos[1]), np.array(self.target_pos[num]))
        hight_dis_l = abs(grip_pos[0][2] - self.cube1_pos[num][2])
        hight_dis_r = abs(grip_pos[1][2] - self.cube2_pos[num][2])

        plam_pos_l = pybullet.getLinkState(self.robos[num],70)[0]
        plam_pos_r = pybullet.getLinkState(self.robos[num],49)[0]

        orien_loss_l = np.dot((np.array(grip_pos[0])-np.array(plam_pos_l))/np.linalg.norm(np.array(grip_pos[0])-
            np.array(plam_pos_l)),(np.array(self.cube1_pos[num])-np.array(grip_pos[0]))/np.linalg.norm(np.array(self.cube1_pos[num])-np.array(grip_pos[0])))
        orien_loss_r = np.dot((np.array(grip_pos[1])-np.array(plam_pos_r))/np.linalg.norm(np.array(grip_pos[1])-
            np.array(plam_pos_r)),(np.array(self.cube2_pos[num])-np.array(grip_pos[1]))/np.linalg.norm(np.array(self.cube2_pos[num])-np.array(grip_pos[1])))

        self_collision = self.check_collisions(self.robos[num])
        self.l_collision[num] = self.check_collisions_l(self.robos[num],self.cubes1[num],self.cubes2[num])
        self.r_collision[num] = self.check_collisions_r(self.robos[num],self.cubes1[num],self.cubes2[num])

        # task 0: reach object:
        if self.pick_dist_l[num] < 0.07 and self.is_pick_l[num] == 0 and orien_loss_l >= 0.0 and hight_dis_l < 0.07: # 
            self.close_gripper(self.left_gripper_joints,"left",grip_pos[0],gripper_rot[0],self.robos[num],self.cubes1[num],self.cubes2[num])
            self.is_pick_l[num] = 1
        
        if self.pick_dist_r[num] < 0.07 and self.is_pick_r[num] == 0 and orien_loss_r >= 0.0 and hight_dis_l < 0.07: #
            self.is_pick_r[num] = 1
            self.close_gripper(self.right_gripper_joints,"right",grip_pos[1],gripper_rot[1],self.robos[num],self.cubes1[num],self.cubes2[num])

        if self.is_pick_l[num] == 1 and self.place_dist_l[num] < 0.05:
            self.terminated_l[num] = 1

        if self.is_pick_r[num] == 1 and self.place_dist_r[num] < 0.05:
            self.terminated_r[num] = 1
        

        if self.cube1_pos[num][2] < 0.68:

            if self.l_collision[num]>0 and self.is_pick_l[num] == 0:
                rewards[0] += -1.0
                self.bad_tran_l[num] = 1
                self.terminated_l[num] = 1
            elif self.r_collision[num]>0:
                rewards[1] += -1.0
                self.bad_tran_r[num] = 1
                self.terminated_l[num] = 1
            else:
                self.terminated_l[num] = 1

        if self.cube2_pos[num][2] < 0.68:

            if self.l_collision[num]>0:
                rewards[0] += -1.0
                self.terminated_r[num] = 1
                self.bad_tran_l[num] = 1
            elif self.r_collision[num]>0 and self.is_pick_r[num] == 0:
                rewards[1] += -1.0
                self.bad_tran_r[num] = 1
                self.terminated_r[num] = 1
            else:
                self.terminated_r[num] = 1

        if self.is_pick_l[num] == 1:
            rewards[0] += -self.place_dist_l[num] * 0.1 + 0.5
        else:
            rewards[0] += -self.pick_dist_l[num] * 0.5 - hight_dis_l * 0.5 - self.pick_to_place_l[num] * 0.1 + orien_loss_l * 0.5
        
        if self.is_pick_r[num] == 1:
            rewards[1] += -self.place_dist_r[num] * 0.1 + 0.5
        else:
            rewards[1] += -self.pick_dist_r[num] * 0.5 - hight_dis_r * 0.5 - self.pick_to_place_r[num] * 0.1 + orien_loss_r * 0.5

        if self_collision > 0:
            rewards[0] += -1
            rewards[1] += -1
            self.bad_tran_r[num] = 1
            self.bad_tran_l[num] = 1

        return rewards[0], rewards[1]