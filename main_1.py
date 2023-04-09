import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from pr2_env_1 import pr2GymEnv

def main():
    args = get_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         #args.gamma, args.log_dir, device, False)

    envs = pr2GymEnv(renders=True, maxSteps=20, 
            actionRepeat=1, task=0, randObjPos=True,
            simulatedGripper=False, learning_param=0.1)

    actor_critic_l = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_l.to(device)
    #print(len(envs.observation_space.shape),envs.observation_space.shape)
    actor_critic_r = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_r.to(device)

    if args.algo == 'ppo':
        agent_l = algo.PPO(
            actor_critic_l,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        agent_r = algo.PPO(
            actor_critic_r,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)


    rollouts_l = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic_l.recurrent_hidden_state_size)
    rollouts_r = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic_r.recurrent_hidden_state_size)
    print("*",actor_critic_r.recurrent_hidden_state_size)
    obs = torch.from_numpy(envs.reset())
    rollouts_r.obs[0].copy_(obs)
    rollouts_r.to(device)
    rollouts_l.obs[0].copy_(obs)
    rollouts_l.to(device)

    episode_rewards_l = 0
    episode_rewards_r = 0

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    num_episode_l = 0
    num_episode_r = 0
    total_episodes_rewards_l = 0
    total_episodes_rewards_r = 0
    for j in range(num_updates):
        next_init_step_l = 0
        next_init_step_r = 0
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent_l.optimizer, j, num_updates,args.lr)
            utils.update_linear_schedule(
                agent_r.optimizer, j, num_updates,args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value_l, action_l, action_log_prob_l, recurrent_hidden_states_l = actor_critic_l.act(
                    rollouts_l.obs[step], rollouts_l.recurrent_hidden_states[step],
                    rollouts_l.masks[step])
            with torch.no_grad():
                value_r, action_r, action_log_prob_r, recurrent_hidden_states_r = actor_critic_r.act(
                    rollouts_r.obs[step], rollouts_r.recurrent_hidden_states[step],
                    rollouts_r.masks[step])

            # Obser reward and next obs
            # print(np.concatenate((action_l,action_r), axis=1).flatten())

            obs, reward_left,reward_right, done_left,done_right, info = envs.step(np.concatenate((action_l,action_r), axis=1).flatten())
            obs = torch.from_numpy(obs)

            episode_rewards_l += reward_left
            episode_rewards_r += reward_right


            # If done then clean the history of observations.
            masks_l = torch.FloatTensor(
                [[0.0] if done_left else [1.0]])
            bad_masks_l = torch.FloatTensor(
                [[0.0] if 'bad_transition_l' in info.keys() else [1.0]])
            #if 'bad_transition_l' in info.keys():
                #print(info)
            rollouts_l.insert(obs, recurrent_hidden_states_l, action_l,
                            action_log_prob_l, value_l, reward_left, masks_l, bad_masks_l)

            masks_r = torch.FloatTensor(
                [[0.0] if done_right else [1.0]])
            bad_masks_r = torch.FloatTensor(
                [[0.0] if 'bad_transition_r' in info.keys() else [1.0]])
            #if 'bad_transition_r' in info.keys():
            #print("recurrent_hidden_states_r",recurrent_hidden_states_r)
            #print("action_log_prob_r",action_log_prob_r)
            rollouts_r.insert(obs, recurrent_hidden_states_r, action_r,
                            action_log_prob_r, value_r, reward_right, masks_r, bad_masks_r)

            if done_left:
                envs.reset_l()
                with torch.no_grad():
                    next_value_l = actor_critic_l.get_value(
                        rollouts_l.obs[step + 1], rollouts_l.recurrent_hidden_states[step + 1],
                        rollouts_l.masks[step + 1]).detach()


                rollouts_l.compute_returns(next_value_l, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits,next_init_step_l,step + 1)

                next_init_step_l = step + 1
                num_episode_l += 1
                total_episodes_rewards_l += episode_rewards_l

                episode_rewards_l = 0


            if done_right:
                envs.reset_r()
                with torch.no_grad():
                    next_value_r = actor_critic_r.get_value(
                        rollouts_r.obs[step + 1], rollouts_r.recurrent_hidden_states[step + 1],
                        rollouts_r.masks[step + 1]).detach()

                rollouts_r.compute_returns(next_value_r, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits,next_init_step_r,step + 1)

                next_init_step_r = step + 1
                num_episode_r += 1
                total_episodes_rewards_r += episode_rewards_r
                episode_rewards_r = 0
                #print("rollouts",rollouts_l.rewards,rollouts_l.rewards.size(),rollouts_r.rewards,rollouts_r.rewards.size())

        with torch.no_grad():
            next_value_l = actor_critic_l.get_value(
                        rollouts_l.obs[-1], rollouts_l.recurrent_hidden_states[-1],
                        rollouts_l.masks[-1]).detach()
            next_value_r = actor_critic_r.get_value(
                rollouts_r.obs[-1], rollouts_r.recurrent_hidden_states[-1],
                rollouts_r.masks[-1]).detach()

        rollouts_l.compute_returns(next_value_l, args.use_gae, args.gamma,
                            args.gae_lambda, args.use_proper_time_limits,next_init_step_l,args.num_steps)
        rollouts_r.compute_returns(next_value_r, args.use_gae, args.gamma,
                            args.gae_lambda, args.use_proper_time_limits,next_init_step_r,args.num_steps)
        #print("rollouts",rollouts_l.rewards,rollouts_l.rewards.size(),rollouts_r.rewards,rollouts_r.rewards.size())
        
        value_loss_r, action_loss_r, dist_entropy_r = agent_r.update(rollouts_r)
        value_loss_l, action_loss_l, dist_entropy_l = agent_l.update(rollouts_l)

        rollouts_r.after_update()
        rollouts_l.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic_l,
            ], os.path.join(save_path, args.env_name + "l_1.pt"))

            torch.save([
                actor_critic_r,
            ], os.path.join(save_path, args.env_name + "r_1.pt"))

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n training episodes: left reward {:.5f}, right reward {:.5f}, total reward {:.5f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        total_episodes_rewards_l/num_episode_l,
                        total_episodes_rewards_r/num_episode_r,
                        total_episodes_rewards_r/num_episode_r+total_episodes_rewards_l/num_episode_l))
            total_episodes_rewards_l = 0
            total_episodes_rewards_r = 0
            num_episode_r = 0
            num_episode_l = 0


if __name__ == "__main__":
    main()
