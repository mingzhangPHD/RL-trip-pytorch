"""The main framwork for this work.
See README for usage.
"""
import gym
import argparse
import numpy as np
import os
import shutil

# import pandas as pd
import torch
# import time
import utli

from tools.learning_rate import LearningRate

# import config

from tensorboardX import SummaryWriter
# writer = SummaryWriter()

from collections import deque
from tools.multiprocessing_env import SubprocVecEnv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set key parameters
def argsparser():
    parser = argparse.ArgumentParser("PPO")
    parser.add_argument('--env_id', help='environment ID', default='Pendulum-v0')
    parser.add_argument('--algo_id', help='algorithm ID', default='PPO')
    parser.add_argument('--num_envs', help='num envs', type=int, default=16)
    parser.add_argument('--buffer_size', help='buffer size', type=int, default=160)
    parser.add_argument('--batch_size', help='batch size', type=int, default=5)
    parser.add_argument('--ppo_epoch', help='ppo epoch num', type=int, default=4)
    parser.add_argument('--num_steps', help='num steps', type=int, default=1000000)
    parser.add_argument('--learning_rate', help='learning rate', type=int, default=3e-4)
    parser.add_argument('--evaluate_every', help='evaluate every', type=int, default=256)

    return parser.parse_args()

class RUN(object):
    def __init__(self, cl_args):
        self.cl_args = cl_args

    # the main function for this work
    def train(self):
        from algos.ppo4multienvs import PPO, ReplayBuffer
        from nets.network import ActorCritic_Norm as ActorCritic
        cl_args = self.cl_args

        log_save_name = cl_args.algo_id + '_' + cl_args.env_id + '_buffer_{}_batch_{}_hidden_{}_lr_{}_maxsteps_{}'.format(
            cl_args.buffer_size, cl_args.batch_size, cl_args.hidden_size, cl_args.learning_rate,
            cl_args.max_steps_per_episodes)
        log_save_path = os.path.join("./runs", log_save_name)
        if os.path.exists(log_save_path):
            shutil.rmtree(log_save_path)
        utli.writer = SummaryWriter(log_save_path)

        model_dir = utli.Save_model_dir(cl_args.algo_id, cl_args.env_id)

        # Create the environment to train on.
        num_envs = 8

        def make_env():
            def _thunk():
                env = gym.make(cl_args.env_id)
                return env

            return _thunk

        envs = [make_env() for i in range(num_envs)]
        envs = SubprocVecEnv(envs)

        env = gym.make(cl_args.env_id)
        env.seed(0)
        buffer_size = cl_args.buffer_size
        batch_size = cl_args.batch_size

        # Train for 1 million timesteps.
        num_steps = cl_args.num_steps

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        lr = LearningRate.get_instance()
        lr.lr = 10 ** (-3)
        lr.decay_factor = 0.5

        lr = cl_args.learning_rate

        evaluate_every = cl_args.evaluate_every

        # The buffer
        replay_buffer = ReplayBuffer(num_total_sizes=buffer_size, obs_dims=state_dim, act_dims=action_dim,
                                     batch_size=batch_size)

        # network
        model = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_size=128).to(device)

        # policy
        policy = PPO(model=model, replay_buf=replay_buffer, lr=lr, device=device)

        time_step = 0
        # Evaluate the initial network
        evaluations = []

        # begin optimize
        cur_state = envs.reset()
        reward_window = deque(maxlen=50)

        while time_step < num_steps:
            replay_buffer.clear()
            train_r = 0
            for _ in range(buffer_size // batch_size):
                state = torch.FloatTensor(cur_state).unsqueeze(0)
                dist, value = model(state.to(device))
                action = dist.sample()
                log_prob = dist.log_prob(action)

                action = action.cpu().detach().numpy()[0]
                log_prob = log_prob.cpu().detach().numpy()[0]
                value = value.cpu().detach().numpy()[0]

                next_state, reward, done, _ = envs.step(action)

                train_r += reward.sum()

                reward = np.expand_dims(reward, axis=1)
                done = np.expand_dims(done, axis=1)
                replay_buffer.add(cur_obs=cur_state, cur_action=action, reward=reward, done=done, old_log_prob=log_prob,
                                  value=value)
                cur_state = next_state

                time_step += 1
                if time_step % evaluate_every == 0:
                    evaluation, mean_reward, mean_step = self.evaluate_policy(env=env, model=model, time_step=time_step,
                                                                         evaluation_trajectories=6)
                    evaluations.append(evaluation)
                    reward_window.append(mean_reward)
                    print(np.mean(reward_window))

                    utli.recordEvaluateResults(results=(mean_reward,
                                                        mean_step,
                                                        np.mean(reward_window)),
                                               time_step=time_step)

            # compute returns
            returns = policy.compute_gae(next_state=next_state)

            returns = replay_buffer.cat(returns)

            # training PPO policy
            value_losses, ppo_losses, entropys, losses = policy.train(returns=returns)

            utli.recordTrainResults(results=(train_r,
                                             np.mean(np.array(value_losses)),
                                             np.mean(np.array(ppo_losses)),
                                             np.mean(np.array(entropys)),
                                             np.mean(np.array(losses))),
                                    time_step=time_step)

        # last evalution
        last_evaluation, mean_reward, mean_step = self.evaluate_policy(env=env, model=model, time_step=time_step,
                                                                  evaluation_trajectories=6)
        evaluations.append(last_evaluation)
        reward_window.append(mean_reward)
        print(np.mean(reward_window))

        utli.recordEvaluateResults(results=(mean_reward,
                                            mean_step,
                                            np.mean(reward_window)),
                                   time_step=time_step)

        # store results
        utli.store_results(evaluations, (time_step + 1), cl_args)

        # Runs policy for X episodes and returns average reward

    def evaluate_policy(self, env, model, time_step, evaluation_trajectories=6):
        """

        Args:
            env: The environment being trained on.
            policy:	The policy being evaluated
            time_step (int): The number of time steps the policy has been trained for.
            evaluation_trajectories (int): The number of trajectories on which to evaluate.

        Returns:
            (list)	- The time_step, followed by all the rewards.
        """
        rewards = []
        steps = []
        for _ in range(evaluation_trajectories):
            r = 0.
            i = 0
            obs = env.reset()
            done = False
            while not done:
                state = torch.FloatTensor(obs).unsqueeze(0)
                dist, value = model(state.to(device))
                action = dist.sample()
                action = action.cpu().detach().numpy()[0]
                obs, reward, done, _ = env.step(action)
                r += reward
                i += 1
            rewards.append(r)
            steps.append(i)
        print("Average reward at timestep {}: {}".format(time_step, np.mean(rewards)))
        mean_reward = np.mean(rewards)
        mean_step = np.mean(steps)
        rewards.append(time_step)
        return rewards, mean_reward, mean_step

if __name__ == '__main__':
    args = argsparser()
    run = RUN(cl_args=args)
    run.train()
