import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
MAX_LOG_STD = 2
MIN_LOG_STD = -5


class ActorCritic(nn.Module):
    def __init__(self, policy_obs_dims, policy_hidden_sizes, action_dims,
                 value_obs_dims, value_hidden_sizes, policy_hidden_activation=torch.tanh,
                 value_hidden_activation=torch.tanh):
        super(ActorCritic, self).__init__()
        self.policy_hidden_activation = policy_hidden_activation
        self.value_hidden_activation = value_hidden_activation
        self.policy_fcs, self.value_fcs = [], []

        policy_in_size, value_in_size = policy_obs_dims, value_obs_dims
        for i, policy_next_dims in enumerate(policy_hidden_sizes):
            policy_fc = nn.Linear(policy_in_size, policy_next_dims)
            policy_in_size = policy_next_dims
            self.__setattr__("policy_fc{}".format(i), policy_fc)
            self.policy_fcs.append(policy_fc)
        self.policy_mean = nn.Linear(policy_in_size, action_dims)
        self.policy_log_std = nn.Linear(policy_in_size, action_dims)
        # self.log_std = nn.Parameter(torch.ones(1, action_dims) * 0.0)

        for i, value_next_dims in enumerate(value_hidden_sizes):
            value_fc = nn.Linear(value_in_size, value_next_dims)
            value_in_size = value_next_dims
            self.__setattr__("value_fc{}".format(i), value_fc)
            self.value_fcs.append(value_fc)
        self.value_last = nn.Linear(value_in_size, 1)

    def forward(self, cur_obs_tensor):
        # compute the mean and log_std for policy.
        policy_h = cur_obs_tensor
        for i, policy_fc in enumerate(self.policy_fcs):
            policy_h = self.policy_hidden_activation(policy_fc(policy_h))
        mean = self.policy_mean(policy_h)
        log_std = self.policy_log_std(policy_h).clamp(MIN_LOG_STD, MAX_LOG_STD)
        # log_std = self.log_std

        value_h = cur_obs_tensor
        for i, value_fc in enumerate(self.value_fcs):
            value_h = self.value_hidden_activation(value_fc(value_h))
        value_obs_t = self.value_last(value_h)

        # return mean, log_std.exp().expand_as(mean), value_obs_t
        return mean, log_std.exp(), value_obs_t

    def select_action(self, cur_obs_tensor, max_action=1.0):
        m, std, v = self.forward(cur_obs_tensor)
        dist = Normal(m, std)
        action = dist.sample()             # .clamp(-max_action, max_action)
        log_prob = dist.log_prob(action).sum(1)   #
        return action[0].numpy(), log_prob.detach(), v.squeeze().detach().item()

    def compute_action(self, cur_obs_tensor):
        """
        cur_obs_tensor    --->    [batch_size, obs_dims]
        """
        m, std, v = self.forward(cur_obs_tensor)
        dist = Normal(m, std)

        entropy = dist.entropy().sum(1, keepdim=True)
        return dist, entropy, v


if __name__ == '__main__':
    actor_critic = ActorCritic(4, [5, 5], 2, 4, [5, 5])
    cur_obs = torch.FloatTensor([[1.1, 1.2, 1.3, 1.4], [1.1, 1.2, 1.3, 1.4]])

    # act, fixed_log_prob, value = actor_critic.select_action(cur_obs)
    # print(fixed_log_prob)
    # act, fixed_log_prob, value = actor_critic.select_action1(cur_obs)
    # print(fixed_log_prob)
    # act, fixed_log_prob, value = actor_critic.select_action(cur_obs)
    # print(fixed_log_prob)











