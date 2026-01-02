"""
Base Algorithm Classes
======================
Common utilities for all MARL algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class TrainConfig:
    """Training configuration."""
    env_name: str = "simple_spread"
    n_agents: int = 3
    max_steps: int = 50
    n_steps: int = 250000
    rollout_length: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    log_interval: int = 1000
    eval_interval: int = 5000
    eval_episodes: int = 10
    save_interval: int = 10000
    seed: int = 42
    device: str = "cpu"
    results_dir: str = "./resultats"


class ReplayBuffer:
    """Replay buffer for on-policy algorithms."""
    
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int,
                 action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        self.clear()
    
    def clear(self):
        self.obs = []
        self.state = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.avail_actions = []
    
    def add(self, obs, state, actions, rewards, dones, log_probs, values, avail_actions):
        self.obs.append(obs)
        self.state.append(state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.avail_actions.append(avail_actions)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        batch = {
            'obs': torch.FloatTensor(np.array(self.obs)).to(self.device),
            'state': torch.FloatTensor(np.array(self.state)).to(self.device),
            'actions': torch.LongTensor(np.array(self.actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(self.device),
            'dones': torch.FloatTensor(np.array(self.dones)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(self.device),
            'values': torch.FloatTensor(np.array(self.values)).to(self.device),
            'avail_actions': torch.FloatTensor(np.array(self.avail_actions)).to(self.device),
        }
        return batch
    
    def __len__(self):
        return len(self.obs)


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return torch.FloatTensor(advantages)


class MLP(nn.Module):
    """Simple MLP network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    """Actor network for policy."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, avail_actions=None):
        logits = self.net(obs)
        if avail_actions is not None:
            logits = logits.masked_fill(avail_actions == 0, -1e10)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class BaseAlgorithm:
    """Base class for all algorithms."""
    
    name = "Base"
    
    def __init__(self, config: TrainConfig, env_info: Dict):
        self.config = config
        self.n_agents = env_info['n_agents']
        self.obs_dim = env_info['obs_dim']
        self.state_dim = env_info['state_dim']
        self.action_dim = env_info['action_dim']
        self.device = config.device
        
        self.global_step = 0
        self.episode_count = 0
    
    def get_actions(self, obs, avail_actions=None, deterministic=False):
        raise NotImplementedError
    
    def get_value(self, state):
        raise NotImplementedError
    
    def update(self, batch):
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError
