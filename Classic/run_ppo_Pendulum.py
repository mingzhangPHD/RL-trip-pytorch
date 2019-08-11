"""The main framwork for this work.
See README for usage.
"""
import gym
import argparse
import numpy as np

import torch
import utli
import os
import time

from tools.learning_rate import LearningRate

from collections import deque
from tensorboardX import SummaryWriter
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set key parameters
def argsparser():
    parser = argparse.ArgumentParser("PPO")
    parser.add_argument('--env_id', help='environment ID', default='Pendulum-v0')
    parser.add_argument('--algo_id', help='algorithm ID', default='PPO')
    parser.add_argument('--buffer_size', help='buffer size', type=int, default=256)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--ppo_epoch', help='ppo epoch num', type=int, default=4)
    parser.add_argument('--hidden_size', help='hidden size', type=int, default=128)
    parser.add_argument('--num_steps', help='num steps', type=int, default=1000000)
    parser.add_argument('--max_steps_per_episodes', help='max steps per episodes', type=int, default=200)
    parser.add_argument('--learning_rate', help='learning rate', type=int, default=3e-4)
    parser.add_argument('--evaluate_every', help='evaluate every', type=int, default=2000)
    parser.add_argument('--stop_condition', help='stop_condition', type=int, default=-180)
    parser.add_argument('--num_model', help='num_model', type=int, default=10)
    parser.add_argument('--use_device', help='stop_condition', type=bool, default=True)

    return parser.parse_args()

class RUN(object):
    def __init__(self, cl_args):
        self.cl_args = cl_args

    # the main function for this work
    def train(self):
        from algos.ppo4Normal import PPO, ReplayBuffer
        from nets.network import ActorCritic_Norm_wyw as ActorCritic
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
        env = gym.make(cl_args.env_id)
        env_evaluate = gym.make(cl_args.env_id)
        # env.seed(0)
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

        ppo_epoch = cl_args.ppo_epoch

        evaluate_every = cl_args.evaluate_every

        max_steps_per_episodes = cl_args.max_steps_per_episodes

        stop_condition = cl_args.stop_condition

        use_device = cl_args.use_device

        # The buffer
        replay_buffer = ReplayBuffer(num_total_sizes=buffer_size, obs_dims=state_dim, act_dims=1, batch_size=batch_size)

        # network
        if use_device:
            model = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_size=cl_args.hidden_size).to(device)
        else:
            model = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_size=cl_args.hidden_size)

        # policy
        policy = PPO(model=model, replay_buf=replay_buffer, lr=lr, device=device, use_device=use_device,
                     ppo_epoch=ppo_epoch, weight_epsilon=0.0)

        time_step = 0
        # Evaluate the initial network
        evaluations = []

        # begin optimize
        # cur_state = env.reset()
        reward_window4Train = deque(maxlen=100)
        reward_window4Evaluate = deque(maxlen=100)
        episode_t = 0
        count = 0
        S_time = time.time()
        while time_step < num_steps:
            episode_t += 1
            cur_state = env.reset()
            path_length, path_rewards = 0, 0.

            while True:
                path_length += 1

                state = torch.FloatTensor(cur_state).unsqueeze(0)
                # state = torch.FloatTensor(cur_state[None])
                with torch.no_grad():
                    if use_device:
                        action, old_log_prob, value = model.select_action(state.to(device))
                    else:
                        action, old_log_prob, value = model.select_action(state)

                action = np.expand_dims(action, axis=1)
                value = np.expand_dims(value, axis=1)

                next_state, reward, done, _ = env.step(action)
                reward = np.expand_dims(reward, axis=1)
                done = np.expand_dims(done, axis=1)

                replay_buffer.add(cur_obs=cur_state, cur_action=action, reward=reward, done=done,
                                  old_log_prob=old_log_prob.cpu(),
                                  value=value)

                cur_state = next_state
                path_rewards += reward

                if replay_buffer.enough_data:
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                    with torch.no_grad():
                        if use_device:
                            _, _, next_value = model.select_action(next_state.to(device))
                        else:
                            _, _, next_value = model.select_action(next_state)

                    # compute returns
                    returns = replay_buffer.compute_gae(next_value=next_value)

                    # training PPO policya
                    value_losses, ppo_losses, entropys, losses = policy.train(returns=returns)

                    utli.recordLossResults(results=(np.mean(np.array(value_losses)),
                                                    np.mean(np.array(ppo_losses)),
                                                    np.mean(np.array(entropys)),
                                                    np.mean(np.array(losses))),
                                           time_step=time_step)

                    replay_buffer.clear()

                if time_step % evaluate_every == 0:
                    evaluation, mean_reward, mean_step = self.evaluate_policy(env=env_evaluate, model=model,
                                                                         time_step=time_step, use_device=use_device,
                                                                         max_step=max_steps_per_episodes,
                                                                         evaluation_trajectories=6)
                    evaluations.append(evaluation)
                    reward_window4Evaluate.append(mean_reward)

                    utli.recordEvaluateResults(results=(mean_reward,
                                                        mean_step,
                                                        np.mean(reward_window4Evaluate)),
                                               time_step=time_step)
                time_step += 1
                if done or max_steps_per_episodes == path_length:
                    break

            reward_window4Train.append(path_rewards)
            utli.recordTrainResults(results=(path_rewards,
                                             path_length,
                                             np.mean(reward_window4Train)),
                                    time_step=time_step)
            print("Episode: %d,      Time steps: %d,        Path length: %d       Reward: %f" % (
            episode_t, time_step, path_length, path_rewards))

            count = utli.Save_trained_model(count=count, num=cl_args.num_model, model=model, model_dir=model_dir,
                                            stop_condition=stop_condition,
                                            reward_window4Train=reward_window4Train,
                                            reward_window4Evaluate=reward_window4Evaluate)

        # last evalution
        evaluation, mean_reward, mean_step = self.evaluate_policy(env=env_evaluate, model=model, time_step=time_step,
                                                             use_device=use_device,
                                                             max_step=max_steps_per_episodes, evaluation_trajectories=6)
        evaluations.append(evaluation)
        reward_window4Evaluate.append(mean_reward)

        utli.recordEvaluateResults(results=(mean_reward,
                                            mean_step,
                                            np.mean(reward_window4Evaluate)),
                                   time_step=time_step)
        E_time = time.time()
        # store results
        utli.store_results(evaluations, (time_step + 1), cl_args, S_time=S_time, E_time=E_time)

        # Runs policy for X episodes and returns average reward

    def evaluate_policy(self, env, model, time_step, max_step, use_device, evaluation_trajectories=6):
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
            step = 0
            obs = env.reset()
            done = False
            while not done:
                # env.render()

                state = torch.FloatTensor(obs).unsqueeze(0)
                if use_device:
                    with torch.no_grad():
                        action, _, _ = model.select_action(state.to(device))

                action = np.expand_dims(action, axis=1)
                obs, reward, done, _ = env.step(action)
                r += reward
                step += 1
                if step == max_step:
                    break
            rewards.append(r)
            steps.append(step)
        print('----------------------------------------------------------')
        print("Average reward at timestep {}: {}".format(time_step, np.mean(rewards)))
        print('----------------------------------------------------------')
        mean_reward = np.mean(rewards)
        mean_step = np.mean(steps)
        rewards.append(time_step)
        return rewards, mean_reward, mean_step



if __name__ == '__main__':
    args = argsparser()
    run = RUN(cl_args=args)
    run.train()
