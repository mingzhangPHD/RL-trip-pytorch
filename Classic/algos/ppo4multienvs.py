from collections import deque

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class ReplayBuffer(object):
    def __init__(self, num_total_sizes, act_dims, obs_dims, batch_size=32):
        self.num_total_sizes = num_total_sizes
        self.batch_size = batch_size
        self.states, self.actions, self.rewards = [], [], []
        self.old_log_probs, self.values, self.dones = [], [], []

    def add(self, cur_obs, cur_action, reward, done, old_log_prob, value):
        """
        cur_obs:                   numpy.array               (obs_dims, envs_sim )
        cur_action:                numpy.array                (act_dims, envs_sim )
        reward:                   numpy.array                 (1,   envs_sim     )
        done:                     numpy.array                 (1,   envs_sim     )
        old_log_prob:             numpy.array                 (1,   envs_sim     )
        value:                    numpy.array                 (1,   envs_sim     )
        """
        self.states.append(cur_obs)
        self.actions.append(cur_action)
        self.rewards.append(reward)
        self.old_log_probs.append(old_log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.old_log_probs, self.values, self.dones = [], [], []

    def cat(self, returns):
        self.states = np.concatenate(self.states)
        self.actions = np.concatenate(self.actions)
        self.rewards = np.concatenate(self.rewards)
        self.old_log_probs = np.concatenate(self.old_log_probs)
        self.values = np.concatenate(self.values)
        self.dones = np.concatenate(self.dones)
        returns = np.concatenate(returns)

        return returns

    @property
    def enough_data(self):
        return self.cur_index == self.num_total_sizes



class PPO(object):
    def __init__(self, model, replay_buf, device, lr=3e-4,
                 clip_epsilon=0.2, gamma=0.99, ppo_epoch=4, weight_epsilon=0.001):
        self.model = model
        self.replay_buf = replay_buf
        self.device = device
        # self.rewards_learning_prcoess = []

        self.clip_epsilon, self.gamma, self.ppo_epoch, self.weight_epsilon = clip_epsilon, gamma, ppo_epoch, weight_epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def train(self, returns):

        minibatch = max(int(self.replay_buf.num_total_sizes / self.replay_buf.batch_size), 1)

        states, actions, old_log_probs, values = \
            self.replay_buf.states, self.replay_buf.actions, self.replay_buf.old_log_probs, self.replay_buf.values

        device = self.device
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        values = torch.FloatTensor(values).to(device)
        returns = torch.FloatTensor(returns).to(device)

        value_losses = []
        ppo_losses = []
        entropys = []
        losses = []

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.replay_buf.num_total_sizes)), minibatch, True):

                new_dists, new_values = self.model(states[index])
                entropy = new_dists.entropy().mean()
                new_log_probs = new_dists.log_prob(actions[index])

                ratios = torch.exp(new_log_probs - old_log_probs[index])
                advantages = returns[index] - values[index]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1. - self.clip_epsilon, 1. + self.clip_epsilon) * advantages
                ppo_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[index] - new_values).pow(2).mean()
                loss = 0.5 * value_loss + ppo_loss - 0.00 * entropy

                value_losses.append(value_loss.item())
                ppo_losses.append(ppo_loss.item())
                entropys.append(entropy.item())
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return value_losses, ppo_losses, entropys, losses

    def compute_gae(self, next_state, lammbda=0.95, use_gae=True):
        # _, _, value = self.actor_critic.select_action(torch.FloatTensor(next_obs[None]))
        rewards, values, dones = \
            self.replay_buf.rewards, self.replay_buf.values, self.replay_buf.dones
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value = self.model(next_state.to(self.device))
        next_value = next_value.cpu().detach().numpy()[0]
        returns = []
        if use_gae:
            values = values + [next_value]
            gae = 0
            for step in reversed(range(len(rewards))):
                mask = 1. - dones[step]
                delta = rewards[step] + self.gamma * values[step + 1] * mask - values[step]
                gae = delta + self.gamma * lammbda * mask * gae
                returns.insert(0, gae + values[step])

        else:
            _return = next_value
            for step in reversed(range(rewards.shape[0])):
                mask = 1. - dones[step]
                _return = rewards[step] + self.gamma * _return
                returns[step] = _return

        return returns

