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
CUBE_URDF_PATH2 = "/home/kk/pybullet-planning/models/drake/objects/simple_cylinder_2.urdf"
PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
SPHERE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "sphere2red.urdf")
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
                 maxSteps=20,
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
        pybullet.setGravity(0,0,-9.8)
        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=90, cameraPitch=-60, cameraTargetPosition=[0,0,0])
        
        # setup robot arm:

        self.plane = pybullet.loadURDF(PLANE_URDF_PATH) # add ground
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.pr2 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [1.05, 0, 0.0], [0, 0, 0.7071, 0.7071],useFixedBase=True)
        lit_mass = [50,52,59,60,71,73,80,81]
        for i in lit_mass:
            pybullet.changeDynamics(bodyUniqueId = self.pr2,linkIndex=i,mass=0.0)
        for i in [-1,0,1,2,3]:
            pybullet.changeDynamics(bodyUniqueId = self.pr2,linkIndex=i,mass=99999.0)
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
        self.bad_tran_l = False
        self.bad_tran_r = False
        for i in range(7):
            for j in range(7):
                pybullet.setCollisionFilterPair(self.pr2,self.pr2,self.joints_list_r[i],self.joints_list_l[j],enableCollision=1)

        print(self.joints_l)
        print(self.joints_r)

        # object:
        self.initial_cube1_pos = [0.75, 0.0, 0.7] # initial object pos
        self.initial_cube2_pos = [0.75, 0.0, 0.7] # initial object pos
        self.place_pos = [0.85,0.0,1.1]
        self.cube1_id = pybullet.loadURDF(CUBE_URDF_PATH, basePosition=self.initial_cube1_pos, globalScaling=0.8)
        self.cube2_id = pybullet.loadURDF(CUBE_URDF_PATH2, basePosition=self.initial_cube2_pos, globalScaling=0.8)
        self.sphere_id = pybullet.loadURDF(SPHERE_URDF_PATH, basePosition=self.place_pos, globalScaling=0.01,useFixedBase = 1)
        self.plat_id = pybullet.loadURDF(PLAT_URDF_PATH, basePosition=[0.75, 0.0, 0.63], globalScaling=1,useFixedBase = 1)
        self.cube1_pos,_ = pybullet.getBasePositionAndOrientation(self.cube1_id)
        self.cube2_pos,_ = pybullet.getBasePositionAndOrientation(self.cube2_id)
        for i in range(44,82):
            pybullet.setCollisionFilterPair(self.pr2,self.cube1_id,i,-1,enableCollision=1)
            pybullet.setCollisionFilterPair(self.pr2,self.cube2_id,i,-1,enableCollision=1)
        self.name = 'pr2GymEnv'

        self.is_pick_l = False
        self.is_pick_r = False
        self.left_gripper_joints = [74,76,75,77]
        self.right_gripper_joints = [53,55,54,56]
        
        self.simulatedGripper = simulatedGripper
        self.action_dim = 7
        self.stepCounter_l = 0
        self.stepCounter_r = 0
        self.maxSteps = maxSteps
        self.terminated_r = False
        self.terminated_l = False
        self.randObjPos = randObjPos
        self.l_collision = 0
        self.r_collision = 0
        self.task = task
        self.learning_param = learning_param
     
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(71,), dtype=np.float64)

    def get_joints_lists(self,joint_lists,arm):
        if arm == "left":
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
                self.joints_l[info.name] = info
        if arm == "right":
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
                forces.append(joint.maxForce*2)
        if arm == "right":
            control_joints = self.control_joints_r
            for i, name in enumerate(control_joints):
                joint = self.joints_r[name]
                poses.append(joint_angles[i])
                indexes.append(joint.id)
                forces.append(joint.maxForce*2)
        pybullet.setJointMotorControlArray(
            self.pr2, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.02]*len(poses),
            forces = forces
        )

    def get_joint_angles(self,arm):
        if arm == "left":
            j = pybullet.getJointStates(self.pr2, self.joints_list_l)
        if arm == "right":
            j = pybullet.getJointStates(self.pr2, self.joints_list_r)
        joints = [i[0] for i in j]
        return joints
         
    def list_add(self,list1,list2):
        res = []
        for i in range(len(list1)):
            res.append(list1[i] + list2[i])
        return res

    def get_current_pose(self,arm):
        if arm == "left":
            linkstate0 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[0])
            linkstate1 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[1])
            linkstate2 = pybullet.getLinkState(self.pr2, self.end_effector_index_l[2])
        if arm == "right":
            linkstate0 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[0])
            linkstate1 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[1])
            linkstate2 = pybullet.getLinkState(self.pr2, self.end_effector_index_r[2])
        position, orientation = list((np.array(linkstate0[0])+np.array(linkstate1[0]))/2), linkstate2[1]
        return (position, orientation)

    def close_gripper(self, gripper_joints,arm,gripper_pos,gripper_rot):
        if arm == "left":
            pybullet.resetBasePositionAndOrientation(self.cube1_id,gripper_pos,gripper_rot)
            self.constraint_id_l = pybullet.createConstraint(self.pr2,70,self.cube1_id, -1,pybullet.JOINT_FIXED,[1, 0, 0],[0,0,0],
                [-0.1, 0, 0],pybullet.getQuaternionFromEuler([0, 0, 0]),pybullet.getQuaternionFromEuler([0, 0, 0]))
        if arm == "right":
            pybullet.resetBasePositionAndOrientation(self.cube2_id,gripper_pos,gripper_rot)
            #self.constraint_id_r = pybullet.createConstraint(self.pr2,49,self.cube2_id, -1,pybullet.JOINT_FIXED,[0, 0, 0],
                #gripper_pos,gripper_rot,pybullet.getQuaternionFromEuler([0, 0, 0]))
            #print(self.gripper_pos)
            self.constraint_id_r = pybullet.createConstraint(self.pr2,49,self.cube2_id, -1,pybullet.JOINT_FIXED,[1, 0, 0],[0,0,0],
                [-0.1, 0, 0],pybullet.getQuaternionFromEuler([0, 0, 0]),pybullet.getQuaternionFromEuler([0, 0, 0]))
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[0], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5,force=100)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[1], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5,force=100)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[2], controlMode=pybullet.POSITION_CONTROL,targetPosition = -0.5,force=100)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[3], controlMode=pybullet.POSITION_CONTROL,targetPosition = -0.5,force=100) 

    def open_gripper(self, gripper_joints):
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[0], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[1], controlMode=pybullet.POSITION_CONTROL,targetPosition = 0.5)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[2], controlMode=pybullet.POSITION_CONTROL,targetPosition = -1)
        pybullet.setJointMotorControl2(bodyIndex=self.pr2, jointIndex=gripper_joints[3], controlMode=pybullet.POSITION_CONTROL,targetPosition = -1)
    
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

    def check_collisions(self):
        collisions = pybullet.getContactPoints(self.pr2,self.pr2)
        return len(collisions)

    def check_collisions_r(self):
        collisions_cube1 = 0
        collisions_cube2 = 0
        for i in range(44,61):
            collisions_cube1 += len(pybullet.getClosestPoints(self.pr2,self.cube1_id,distance = 0.01, linkIndexA = i))
            collisions_cube2 += len(pybullet.getClosestPoints(self.pr2,self.cube2_id,distance = 0.01, linkIndexA = i))
        return [collisions_cube1,collisions_cube2]

    def check_collisions_l(self):
        collisions_cube1 = 0
        collisions_cube2 = 0
        for i in range(65,82):
            collisions_cube1 += len(pybullet.getClosestPoints(self.pr2,self.cube1_id,distance = 0.01, linkIndexA = i))
            collisions_cube2 += len(pybullet.getClosestPoints(self.pr2,self.cube2_id,distance = 0.01, linkIndexA = i))
        return [collisions_cube1,collisions_cube2]

    def reset(self,seed=None):
        if self.is_pick_l:
            pybullet.removeConstraint(self.constraint_id_l)
        if self.is_pick_r:
            pybullet.removeConstraint(self.constraint_id_r)
        self.l_collision_l = 0
        self.l_collision_r = 0
        self.r_collision_l = 0
        self.r_collision_r = 0
        self.stepCounter_l = 0
        self.stepCounter_r = 0
        self.terminated_r = False
        self.terminated_l = False
        self.bad_tran_l = False
        self.bad_tran_r = False
        self.is_pick_l = False
        self.is_pick_r = False
        self.open_gripper(self.left_gripper_joints)
        self.open_gripper(self.right_gripper_joints)


        new_pos_l = self.list_add(self.initial_cube1_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        new_pos_r = self.list_add(self.initial_cube2_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])

        while goal_distance(np.array(new_pos_l), np.array(new_pos_r)) < 0.1:
            new_pos_r = self.list_add(self.initial_cube2_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])

        pybullet.resetBasePositionAndOrientation(self.cube1_id, new_pos_l, [0.,0.,0.,1.0]) # reset object pos
        pybullet.resetBasePositionAndOrientation(self.cube2_id, new_pos_r, [0.,0.,0.,1.0]) # reset object pos

        self.pick_to_place_l = goal_distance(np.array(self.initial_cube1_pos), np.array(self.place_pos))
        self.pick_to_place_r = goal_distance(np.array(self.initial_cube2_pos), np.array(self.place_pos))
        self.reset_body(self.pr2)

        # get obs and return:
        self.getExtendedObservation()
        return self.observation
    
    def reset_l(self,seed=None):
        if self.is_pick_l:
            pybullet.removeConstraint(self.constraint_id_l)

        self.l_collision_l = 0
        self.l_collision_r = 0
        self.stepCounter_l = 0
        self.terminated_l = False
        self.bad_tran_l = False
        self.is_pick_l = False
        self.open_gripper(self.left_gripper_joints)
        new_pos_l = self.list_add(self.initial_cube1_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        while goal_distance(np.array(new_pos_l), np.array(self.cube2_pos)) < 0.1:
            new_pos_l = self.list_add(self.initial_cube1_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        pybullet.resetBasePositionAndOrientation(self.cube1_id, new_pos_l, [0.,0.,0.,1.0]) # reset object pos

        self.pick_to_place_l = goal_distance(np.array(self.initial_cube1_pos), np.array(self.place_pos))
        self.reset_body_l(self.pr2)
        self.getExtendedObservation()
        return self.observation

    def reset_r(self,seed=None):
        if self.is_pick_r:
            pybullet.removeConstraint(self.constraint_id_r)
        self.r_collision_l = 0
        self.r_collision_r = 0
        self.stepCounter_r = 0
        self.terminated_r = False
        self.bad_tran_r = False
        self.is_pick_r = False
        self.open_gripper(self.right_gripper_joints)
        new_pos_r = self.list_add(self.initial_cube2_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        while goal_distance(np.array(self.cube1_pos), np.array(new_pos_r)) < 0.1:
            new_pos_r = self.list_add(self.initial_cube2_pos, [np.random.default_rng().uniform(-0.12,0.12),np.random.default_rng().uniform(-0.2,0.2), 0])
        pybullet.resetBasePositionAndOrientation(self.cube2_id, new_pos_r, [0.,0.,0.,1.0]) # reset object pos
        self.pick_to_place_r = goal_distance(np.array(self.initial_cube2_pos), np.array(self.place_pos))
        self.reset_body_r(self.pr2)
        self.getExtendedObservation()
        return self.observation

    def set_cube_pose(self,action_pos):
        new_pos_l = self.list_add(self.initial_cube1_pos, [action_pos[0],action_pos[1], 0])
        new_pos_r = self.list_add(self.initial_cube2_pos, [action_pos[2],action_pos[3], 0])
        if goal_distance(np.array(new_pos_l), np.array(new_pos_r)) > 0.1:
            if not self.is_pick_l:
                pybullet.resetBasePositionAndOrientation(self.cube1_id, new_pos_l, [0.,0.,0.,1.0]) # reset object pos
            if not self.is_pick_r:
                pybullet.resetBasePositionAndOrientation(self.cube2_id, new_pos_r, [0.,0.,0.,1.0]) # reset object pos






    def step(self, action,pos_action):
        action_arm_l = np.zeros(7)
        action_arm_r = np.zeros(7)
        pos_action_ = np.zeros(4)
        #action_arm_l[0] = np.clip(np.array(action[0]).astype(float), -0.715, 2.285)
        #action_arm_l[1] = np.clip(np.array(action[1]).astype(float), -0.524, 1.396)
        #action_arm_l[2] = np.clip(np.array(action[2]).astype(float), -0.800, 3.900)
        #action_arm_l[3] = np.clip(np.array(action[3]).astype(float), -2.321, 0.000)
        #action_arm_l[4] = np.clip(np.array(action[4]).astype(float), -3.142, 3.142)
        #action_arm_l[5] = np.clip(np.array(action[5]).astype(float), -2.094, 0.000)
        #action_arm_l[6] = np.clip(np.array(action[6]).astype(float), -3.142, 3.142)

        #action_arm_r[0] = np.clip(np.array(action[7]).astype(float), -2.285, 0.715)
        #action_arm_r[1] = np.clip(np.array(action[8]).astype(float), -0.524, 1.396)
        #action_arm_r[2] = np.clip(np.array(action[9]).astype(float), -3.900, 0.800)
        #action_arm_r[3] = np.clip(np.array(action[10]).astype(float), -2.321, 0.00)
        #action_arm_r[4] = np.clip(np.array(action[11]).astype(float), -3.142, 3.142)
        #action_arm_r[5] = np.clip(np.array(action[12]).astype(float), -2.094, 0.000)
        #action_arm_r[6] = np.clip(np.array(action[13]).astype(float), -3.142, 3.142)
        action_arm_l[0] = np.clip(np.array(action[0]).astype(float)*3.142, -3.142, 3.142)
        action_arm_l[1] = np.clip(np.array(action[1]).astype(float)*3.142, -3.142, 3.142)
        action_arm_l[2] = np.clip(np.array(action[2]).astype(float)*3.142, -3.142, 3.142)
        action_arm_l[3] = np.clip(np.array(action[3]).astype(float)*3.142, -3.142, 3.142)
        action_arm_l[4] = np.clip(np.array(action[4]).astype(float)*3.142, -3.142, 3.142)
        action_arm_l[5] = np.clip(np.array(action[5]).astype(float)*3.142, -3.142, 3.142)
        action_arm_l[6] = np.clip(np.array(action[6]).astype(float)*3.142, -3.142, 3.142)

        action_arm_r[0] = np.clip(np.array(action[7]).astype(float)*3.142, -3.142, 3.142)
        action_arm_r[1] = np.clip(np.array(action[8]).astype(float)*3.142, -3.142, 3.142)
        action_arm_r[2] = np.clip(np.array(action[9]).astype(float)*3.142, -3.142, 3.142)
        action_arm_r[3] = np.clip(np.array(action[10]).astype(float)*3.142, -3.142, 3.142)
        action_arm_r[4] = np.clip(np.array(action[11]).astype(float)*3.142, -3.142, 3.142)
        action_arm_r[5] = np.clip(np.array(action[12]).astype(float)*3.142, -3.142, 3.142)
        action_arm_r[6] = np.clip(np.array(action[13]).astype(float)*3.142, -3.142, 3.142)
        #print(pos_action)
        pos_action_[0] = np.clip(np.array(pos_action[0,0]).astype(float)/8, -0.12, 0.12)
        pos_action_[1] = np.clip(np.array(pos_action[0,1]).astype(float)/5, -0.2, 0.2)
        pos_action_[2] = np.clip(np.array(pos_action[0,2]).astype(float)/8, -0.12, 0.12)
        pos_action_[3] = np.clip(np.array(pos_action[0,3]).astype(float)/5, -0.2, 0.2)
        #print(pos_action_)

        # actuate: 
        self.set_joint_angles(action_arm_l,"left")
        self.set_joint_angles(action_arm_r,"right")
        self.set_cube_pose(pos_action_)

        # step simualator:
        for i in range(25):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)
        
        self.getExtendedObservation()

        distance = goal_distance(np.array(pybullet.getBasePositionAndOrientation(self.pr2)[0]), np.array([0,0,0]))
        error = goal_distance(np.array(pybullet.getBasePositionAndOrientation(self.pr2)[1]), np.array([0,0,0,1]))
        
        if distance > 0.01 or error > 0.01:
            pybullet.resetBasePositionAndOrientation(self.pr2,[0,0,0],[0,0,0,1])
            print("error")

        reward_left, reward_right = self.compute_reward(self.gripper_pos,self.gripper_rot)
        done_left, done_right = self.my_task_done()

        info = {'is_success': False, 'episode_l': reward_left, 'episode_r': reward_right}
        if self.terminated_l and self.terminated_r:
            info['is_success'] = True
        if self.bad_tran_l:
            info['bad_transition_l'] = True
        if self.bad_tran_r:
            info['bad_transition_r'] = True

        self.stepCounter_l += 1
        self.stepCounter_r += 1
        return self.observation, reward_left, reward_right, done_left, done_right, info


    # observations are: arm (tip/tool) position, arm acceleration, ...
    def getExtendedObservation(self):

        tool_pos_l = self.get_current_pose("left")[0]
        tool_rot_l = self.get_current_pose("left")[1]
        tool_pos_r = self.get_current_pose("right")[0]
        tool_rot_r = self.get_current_pose("right")[1]
        self.gripper_pos = (tool_pos_l,tool_pos_r)
        self.gripper_rot = (tool_rot_l,tool_rot_r)
        self.cube1_pos = pybullet.getBasePositionAndOrientation(self.cube1_id)[0]
        self.cube2_pos = pybullet.getBasePositionAndOrientation(self.cube2_id)[0]
        cube_1_pos = self.cube1_pos
        cube_2_pos = self.cube2_pos
        plam_pos_l = pybullet.getLinkState(self.pr2,70)[0]
        plam_pos_r = pybullet.getLinkState(self.pr2,49)[0]
        tip_pos_l_l = pybullet.getLinkState(self.pr2,75)[0]
        tip_pos_l_r = pybullet.getLinkState(self.pr2,77)[0]
        tip_pos_r_r = pybullet.getLinkState(self.pr2,56)[0]
        tip_pos_r_l = pybullet.getLinkState(self.pr2,54)[0]
        gripper_pos_l_r = pybullet.getLinkState(self.pr2,76)[0]
        gripper_pos_l_l = pybullet.getLinkState(self.pr2,74)[0]
        gripper_pos_r_r = pybullet.getLinkState(self.pr2,55)[0]
        gripper_pos_r_l = pybullet.getLinkState(self.pr2,53)[0] 
        joints_state_l = pybullet.getJointStates(self.pr2,self.joints_list_l)
        joints_state_r = pybullet.getJointStates(self.pr2,self.joints_list_r)
        bot_base_pos,bot_base_rot = pybullet.getBasePositionAndOrientation(self.pr2)
        self.observation = np.array(np.concatenate((bot_base_pos,bot_base_rot,tool_pos_l,tool_rot_l,tool_pos_r,tool_rot_r,cube_1_pos, cube_2_pos,\
            plam_pos_l,plam_pos_r,tip_pos_l_l,tip_pos_l_r,tip_pos_r_r,tip_pos_r_l,gripper_pos_l_r,gripper_pos_l_l,gripper_pos_r_r,gripper_pos_r_l,\
            [joints_state_l[0][0]/3.142,joints_state_l[1][0]/3.142,\
            joints_state_l[2][0]/3.142,joints_state_l[3][0]/3.142,\
            joints_state_l[4][0]/3.142,joints_state_l[5][0]/3.142,\
            joints_state_l[6][0]/3.142,joints_state_r[0][0]/3.142,\
            joints_state_r[1][0]/3.142,joints_state_r[2][0]/3.142,\
            joints_state_r[3][0]/3.142,joints_state_r[4][0]/3.142,\
            joints_state_r[5][0]/3.142,joints_state_r[6][0]/3.142]
            )))

    def my_task_done(self):
        done_left = (self.terminated_l == True or self.stepCounter_l > self.maxSteps)
        done_right = (self.terminated_r == True or self.stepCounter_r > self.maxSteps)
        #if self.stepCounter_r > self.maxSteps:
            #print("max r")
        #if self.stepCounter_l > self.maxSteps:
            #print("max l")
        return done_left, done_right


    def compute_reward(self, grip_pos,gripper_rot):
        rewards = np.zeros(2)

        self.pick_dist_l = goal_distance2d(np.array(grip_pos[0]), np.array(self.cube1_pos))
        self.place_dist_l = goal_distance(np.array(grip_pos[0]), np.array(self.place_pos))
        self.pick_dist_r = goal_distance2d(np.array(grip_pos[1]), np.array(self.cube2_pos))
        self.place_dist_r = goal_distance(np.array(grip_pos[1]), np.array(self.place_pos))
        tip_l_l = pybullet.getLinkState(self.pr2, self.end_effector_index_l[0])
        tip_l_r = pybullet.getLinkState(self.pr2, self.end_effector_index_l[1])

        tip_r_l = pybullet.getLinkState(self.pr2, self.end_effector_index_r[0])
        tip_r_r = pybullet.getLinkState(self.pr2, self.end_effector_index_r[1])

        hight_dis_l_l = abs(tip_l_l[0][2] - self.cube1_pos[2])
        hight_dis_l_r = abs(tip_l_r[0][2] - self.cube1_pos[2])
        hight_dis_r_l = abs(tip_r_l[0][2] - self.cube2_pos[2])
        hight_dis_r_r = abs(tip_r_r[0][2] - self.cube2_pos[2])

        plam_pos_l = pybullet.getLinkState(self.pr2,70)[0]
        plam_pos_r = pybullet.getLinkState(self.pr2,49)[0]

        orien_loss_l = np.dot((np.array(grip_pos[0])-np.array(plam_pos_l))/np.linalg.norm(np.array(grip_pos[0])-
            np.array(plam_pos_l)),(np.array(self.cube1_pos)-np.array(grip_pos[0]))/np.linalg.norm(np.array(self.cube1_pos)-np.array(grip_pos[0])))
        orien_loss_r = np.dot((np.array(grip_pos[1])-np.array(plam_pos_r))/np.linalg.norm(np.array(grip_pos[1])-
            np.array(plam_pos_r)),(np.array(self.cube2_pos)-np.array(grip_pos[1]))/np.linalg.norm(np.array(self.cube2_pos)-np.array(grip_pos[1])))

        self_collision = self.check_collisions()
        self.l_collision_l += self.check_collisions_l()[0]
        self.l_collision_r = self.check_collisions_l()[1]
        if self.l_collision_r>0:
            rewards[0] += -1.0
        self.r_collision_l = self.check_collisions_r()[0]
        if self.r_collision_l>0:
            rewards[1] += -1.0
        self.r_collision_r += self.check_collisions_r()[1]

        # task 0: reach object:
        if self.pick_dist_l < 0.1 and self.is_pick_l == False and orien_loss_l >= 0.0 and hight_dis_l_l < 0.075 and hight_dis_l_r < 0.075: # 
            self.close_gripper(self.left_gripper_joints,"left",grip_pos[0],gripper_rot[0])
            self.is_pick_l = True
            rewards[0] += 1.0
        
        if self.pick_dist_r < 0.1 and self.is_pick_r == False and orien_loss_r >= 0.0 and hight_dis_r_l < 0.075 and hight_dis_r_r < 0.075: #
            self.is_pick_r = True
            rewards[1] += 1.0
            self.close_gripper(self.right_gripper_joints,"right",grip_pos[1],gripper_rot[1])

        if self.is_pick_l and self.place_dist_l < 0.1:
            rewards[0] += 1.0
            self.terminated_l = True
            #print("success l")

        if self.is_pick_r and self.place_dist_r < 0.1:
            rewards[1] += 1.0
            self.terminated_r = True
            #print("success r")
        
        #if self.cube1_pos[2] < 0.68 or self.cube2_pos[2] < 0.68:
            #self.terminated_r = True
            #self.terminated_l = True
        #if l_collision>0 or r_collision>0:
            #print("l_collision",l_collision,"r_collision",r_collision)

        if self.cube1_pos[2] < 0.65:
            if self.is_pick_l == True:
                pass
            else:
                if self.l_collision_l>0:
                    rewards[0] += -1.0
                self.terminated_l = True

        if self.cube2_pos[2] < 0.65:
            if self.is_pick_r == True:
                pass
            else:
                if self.r_collision_r>0:
                    rewards[1] += -1.0
                self.terminated_r = True

        if self.is_pick_l:
            rewards[0] += -self.place_dist_l * 0.2 + 0.1
            #print("left",-self.place_dist_l * 0.2 + 0.1, -self.pick_dist_l * 0.2 - (hight_dis_l_l + hight_dis_l_r) * 0.2 - self.pick_to_place_l * 0.2 + orien_loss_l * 0.1)
        else:
            #print("not pick left", self.stepCounter_l)
            rewards[0] += -self.pick_dist_l * 0.2 - (hight_dis_l_l + hight_dis_l_r) * 0.2 - self.pick_to_place_l * 0.2 + orien_loss_l * 0.1
        
        if self.is_pick_r:
            rewards[1] += -self.place_dist_r * 0.2 + 0.1
            #print("right",-self.place_dist_r * 0.2 + 0.1,-self.pick_dist_r * 0.2, - (hight_dis_r_l + hight_dis_r_r) * 0.2, - self.pick_to_place_r * 0.2, + orien_loss_r * 0.1)
        else:
            #print("not pick right",self.stepCounter_r)
            rewards[1] += -self.pick_dist_r * 0.2 - (hight_dis_r_l + hight_dis_r_r) * 0.2 - self.pick_to_place_r * 0.2 + orien_loss_r * 0.1

        if self_collision > 0:
            rewards[0] += -1
            rewards[1] += -1
            self.bad_tran_r = True
            self.bad_tran_l = True

        return rewards[0], rewards[1]