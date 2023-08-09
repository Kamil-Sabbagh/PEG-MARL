from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.distributions.categorical import Categorical
from einops import reduce
import numpy as np


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_params_2(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.orthogonal_(m.weight, nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 1e-6)


def to_tensor(obs, device, dtype=torch.float32):
    if len(obs.shape) == 1:
        obs = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
    elif len(obs.shape) == 2:
        obs = torch.tensor(obs, dtype=dtype, device=device)
    return obs


class ActorModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # layer_norm(self.layer1, nn.init.calculate_gain('relu'))
        # layer_norm(self.layer2, 0.1)

        # Initialize parameters correctly
        self.apply(init_params_2)

    def forward(self, obs):
        x = self.actor(to_tensor(obs, self.device))
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        return dist

    def get_action(self, dist):
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        return action, log_prob_action


class ACModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params_2)

    def forward(self, obs, mask):
        obs = to_tensor(obs, self.device)
        if mask is not None:
            mask = to_tensor(mask, self.device, torch.bool)
        x = self.actor(obs)
        # dist = Categorical(logits=F.log_softmax(x, dim=1))
        dist = CategoricalMasked(logits=F.log_softmax(x, dim=1), mask=mask)

        y = self.critic(obs)
        value = y.squeeze(1)

        return dist, value

    def get_action(self, dist):
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        return action, log_prob_action


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state_action):
        reward = self.model(to_tensor(state_action, self.device))
        return reward


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype
            )
            logits = torch.where(self.mask, logits, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)



class SharedForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SharedForwardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        return next_state_pred
