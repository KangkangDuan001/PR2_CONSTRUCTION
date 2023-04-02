# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet
# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py

import os
import argparse
from datetime import datetime
import numpy as np
from itertools import count
from collections import namedtuple, deque
import pickle
import torch
import gym
import random
from ppo import PPO, Memory, ActorCritic
from gym_env import pr2GymEnv

title = 'PyBullet pr2 robot'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    # arg('--env_name', type=str, default='ur5GymEnv', help='environment name')
    arg('--render', action='store_true', default=False, help='render the environment')
    arg('--randObjPos', action='store_true', default=False, help='fixed object position to pick up')
    arg('--mel', type=int, default=100, help='max episode length')
    arg('--repeat', type=int, default=1, help='repeat action')
    arg('--simgrip', action='store_true', default=False, help='simulated gripper')
    arg('--task', type=int, default=0, help='task to learn: 0 move, 1 pick-up, 2 drop')
    arg('--lp', type=float, default=0.1, help='learning parameter for task')
    # train:
    arg('--seed', type=int, default=987, help='random seed')
    arg('--emb_size',   type=int, default=128, help='embedding size')
    arg('--solved_reward', type=int, default=0, help='stop training if avg_reward > solved_reward')
    arg('--log_interval', type=int, default=100, help='interval for log')
    arg('--save_interval', type=int, default=100, help='interval for saving model')
    arg('--max_episodes', type=int, default=2500, help='max training episodes')
    arg('--update_timestep', type=int, default=1000, help='update policy every n timesteps')
    arg('--action_std', type=float, default=1.0, help='constant std for action distribution (Multivariate Normal)')
    arg('--K_epochs', type=int, default=100, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.01, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--save_dir', type=str, default='saved_rl_models/', help='path to save the models')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='Use cuda to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

def write_file(filepath, data, mode):
    f = open(filepath, mode)
    f.write(data)
    f.close()

args.filename_tl = 'training_log.txt' # log file

args.device = torch.device('cuda:'+str(args.device_num) if args.cuda else 'cpu')
print('Using device:', 'cuda' if args.cuda else 'cpu', ', device number:', args.device_num, ', GPUs in system:', torch.cuda.device_count())


def main():
    args.env_name = title
    print(CP_G + 'Environment name:', args.env_name, ''+CP_C)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    env = pr2GymEnv(renders=args.render, maxSteps=args.mel, 
            actionRepeat=args.repeat, task=args.task, randObjPos=args.randObjPos,
            simulatedGripper=args.simgrip, learning_param=args.lp)
    
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory_left = Memory()
    memory_right = Memory()
    ppo_left = PPO(args, env,"left")
    ppo_right = PPO(args, env,"right")

    # logging variables
    running_reward_left = 0
    running_reward_right = 0
    avg_length = 0
    time_step = 0

    # training loop:
    print('Starting training with learning_param:', args.lp)
    for i_episode in range(1, args.max_episodes+1):
        state = env.reset()
        for t in range(args.mel):
            time_step += 1

            action_left = ppo_left.select_action(state, memory_left)
            action_right = ppo_right.select_action(state, memory_right)
            state, reward_left,reward_right, done_left,done_right, _ = env.step(action_left,action_right)
            
            # Saving reward and is_terminals:
            memory_left.rewards.append(reward_left)
            memory_right.rewards.append(reward_right)
            memory_left.is_terminals.append(done_left)
            memory_right.is_terminals.append(done_right)
            
            # learning:
            if time_step % args.update_timestep == 0:
                ppo_left.update(memory_left)
                ppo_right.update(memory_right)
                memory_left.clear_memory()
                memory_right.clear_memory()
                time_step = 0
            running_reward_left += reward_left
            running_reward_right += reward_right

            if done_left and done_right:
                break
        
        avg_length += t
        
        
        # save every few episodes
        if i_episode % args.save_interval == 0:
            torch.save(ppo_left.policy.state_dict(), args.save_dir+'/model_epoch_'+str(int(i_episode/args.save_interval))+'_l.pth')
            torch.save(ppo_right.policy.state_dict(), args.save_dir+'/model_epoch_'+str(int(i_episode/args.save_interval))+'_r.pth')
            
        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length/args.log_interval)
            running_reward_left = int((running_reward_left/args.log_interval))
            running_reward_right = int((running_reward_right/args.log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward left: {} \t Avg reward right: {} '.format(i_episode, avg_length, running_reward_left, running_reward_right))
            running_reward_right = 0
            running_reward_left = 0
            avg_length = 0


if __name__ == '__main__':
    main()