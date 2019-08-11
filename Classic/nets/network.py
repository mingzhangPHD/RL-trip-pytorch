import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical

import numpy as np

MIN_LOG_STD = -2
MAX_LOG_STD = 5
min_log_std = -5
max_log_std = 2


class ActorCritic_Norm_wyw(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic_Norm_wyw, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_dim)
        self.sigma = nn.Linear(hidden_size, action_dim)

        self.l3 = nn.Linear(state_dim, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        a = torch.tanh(self.l1(x))
        a = torch.tanh(self.l2(a))

        mu = self.mu(a)
        log_std = self.sigma(a).clamp(min_log_std, max_log_std)

        value = torch.tanh(self.l3(x))
        value = torch.tanh(self.l4(value))
        value = self.l5(value)

        return mu, log_std.exp(), value

    def select_action(self, cur_obs_tensor):
        m, std, v = self.forward(cur_obs_tensor)
        dist = Normal(m, std)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        # log_prob = log_prob1.sum(1)
        return action.item(), log_prob.detach(), v.squeeze().detach().item()

    def compute_action(self, cur_obs_tensor):

        m, std, v = self.forward(cur_obs_tensor)
        dist = Normal(m, std)

        entropy1 = dist.entropy()
        entropy = dist.entropy().sum(1, keepdim=True)
        return dist, entropy, v

class ActorCritic_Cate_wyw(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(ActorCritic_Cate_wyw, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        x1 = self.fc3(x)
        action = F.softmax(x1, dim=1)
        value = self.fc4(x)

        return action, value

    def select_action(self, cur_obs_tensor):
        action_probs, value = self.forward(cur_obs_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.squeeze().detach().item()

    def compute_action(self, cur_obs_tensor):
        action_probs, value = self.forward(cur_obs_tensor)
        dist = Categorical(action_probs)

        entropy1 = dist.entropy()
        entropy = entropy1.unsqueeze(0)

        return dist, entropy, value


class ActorCritic_Cate_zm01(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(ActorCritic_Cate_zm01, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.pi = nn.Linear(hidden_size, action_dim)

        self.fc4 = nn.Linear(state_dim, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)


    def forward(self, x):
        a = torch.tanh(self.fc1(x))
        a = torch.tanh(self.fc2(a))

        pi = self.pi(a)
        action = F.softmax(pi, dim=1)

        v = torch.tanh(self.fc4(x))
        v = torch.tanh(self.fc5(v))
        value = self.v(v)

        return action, value

    def select_action(self, cur_obs_tensor):
        action_probs, value = self.forward(cur_obs_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.squeeze().detach().item()

    def compute_action(self, cur_obs_tensor):
        action_probs, value = self.forward(cur_obs_tensor)
        dist = Categorical(action_probs)

        entropy1 = dist.entropy()
        entropy = entropy1.unsqueeze(0)

        return dist, entropy, value

class ActorCritic_Norm(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(ActorCritic_Norm, self).__init__()

        # Actor architecture
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

        # mu
        self.mu = nn.Linear(hidden_size, action_dim)
        # sigma
        self.sigma = nn.Linear(hidden_size, action_dim)

        # Critic architecture
        self.l5 = nn.Linear(state_dim, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        # self.apply(init_weights)


    def forward(self, x):

        a = torch.tanh(self.l1(x))
        a = torch.tanh(self.l2(a))
        mu = self.mu(a)
        log_std = self.sigma(a).clamp(MIN_LOG_STD, MAX_LOG_STD)

        v = torch.tanh(self.l5(x))
        v = torch.tanh(self.l6(v))
        value = self.v(v)

        dist = Normal(mu, log_std.exp())

        return dist, value

class ActorCritic_Cate(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(ActorCritic_Cate, self).__init__()

        # Actor architecture
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

        # pi
        self.pi = nn.Linear(hidden_size, action_dim)

        # Critic architecture
        self.l5 = nn.Linear(state_dim, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        # self.apply(init_weights)


    def forward(self, x):

        a = torch.tanh(self.l1(x))
        a = torch.tanh(self.l2(a))
        pi = F.softmax(self.pi(a), dim=1)
        # pi = self.pi(a)

        dist = Categorical(logits=pi)
        # dist = Categorical(probs=pi)

        v = torch.tanh(self.l5(x))
        v = torch.tanh(self.l6(v))
        value = self.v(v)



        return dist, value





