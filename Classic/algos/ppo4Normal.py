
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class ReplayBuffer(object):
    def __init__(self, num_total_sizes, act_dims, obs_dims, batch_size=32):
        self.num_total_sizes = num_total_sizes
        self.batch_size = batch_size
        self.states, self.actions, self.rewards = np.zeros([num_total_sizes, obs_dims]), \
                                                        np.zeros([num_total_sizes, act_dims]), np.zeros([num_total_sizes, 1])
        self.old_log_probs, self.values, self.dones = np.zeros([num_total_sizes, 1]), np.zeros([num_total_sizes, 1]), \
                                                      np.zeros([num_total_sizes, 1])
        self.cur_index = 0

    def add(self, cur_obs, cur_action, reward, done, old_log_prob, value):
        """
        cur_obs:                   numpy.array                (obs_dims, )
        cur_action:                numpy.array                (act_dims, )
        reward:                   numpy.array                 (1,        )
        done:                     numpy.array                 (1,        )
        old_log_prob:             numpy.array                 (1,        )
        value:                    numpy.array                 (1,        )
        """
        self.states[self.cur_index] = cur_obs
        self.actions[self.cur_index] = cur_action
        self.rewards[self.cur_index] = reward
        self.old_log_probs[self.cur_index] = old_log_prob
        self.dones[self.cur_index] = done
        self.values[self.cur_index] = value
        self.cur_index += 1

    def clear(self):
        self.cur_index = 0

    def compute_gae(self, next_value, gamma=0.99, lammbda=0.95):

        rewards, values, dones = \
            self.rewards, self.values, self.dones
        value = next_value
        gae = 0
        returns = np.zeros_like(values)
        for step in reversed(range(rewards.shape[0])):
            td_delta = rewards[step] + gamma * (1. - dones[step]) * value - values[step]
            value = values[step]
            gae = gamma * lammbda * (1. - dones[step]) * gae + td_delta
            returns[step] = gae + values[step]
        return returns

    @property
    def enough_data(self):
        return self.cur_index == self.num_total_sizes

class PPO(object):
    def __init__(self, model, replay_buf, device, use_device, lr=3e-4,
                 clip_epsilon=0.2, gamma=0.99, ppo_epoch=4, weight_epsilon=0.001):
        self.model = model
        self.replay_buf = replay_buf
        self.device = device
        self.use_device = use_device
        # self.rewards_learning_prcoess = []

        self.clip_epsilon, self.gamma, self.ppo_epoch, self.weight_epsilon = clip_epsilon, gamma, ppo_epoch, weight_epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def train(self, returns):

        minibatch = max(int(self.replay_buf.num_total_sizes / self.replay_buf.batch_size), 1)

        states, actions, old_log_probs, values = \
            self.replay_buf.states, self.replay_buf.actions, self.replay_buf.old_log_probs, self.replay_buf.values

        if self.use_device:
            device = self.device
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(device)
            values = torch.FloatTensor(values).to(device)
            returns = torch.FloatTensor(returns).to(device)
        else:
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            values = torch.FloatTensor(values)
            returns = torch.FloatTensor(returns)

        value_losses = []
        ppo_losses = []
        entropys = []
        losses = []

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.replay_buf.num_total_sizes)), minibatch, True):

                # new_dists, new_values = self.model(states[index])
                # entropy = new_dists.entropy().mean()
                # new_log_probs = new_dists.log_prob(actions[index])

                new_dists, policy_entropys, new_values = self.model.compute_action(states[index])
                entropy = policy_entropys.mean()
                new_log_probs1 = new_dists.log_prob(actions[index])

                new_log_probs = new_log_probs1.sum(1, keepdim=True)
                ttt = old_log_probs[index]
                ratios = torch.exp(new_log_probs - old_log_probs[index])
                advantages = returns[index] - values[index]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1. - self.clip_epsilon, 1. + self.clip_epsilon) * advantages
                ppo_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[index] - new_values).pow(2).mean()
                loss = 0.5 * value_loss + ppo_loss - self.weight_epsilon * entropy

                value_losses.append(value_loss.item())
                ppo_losses.append(ppo_loss.item())
                entropys.append(entropy.item())
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return value_losses, ppo_losses, entropys, losses

