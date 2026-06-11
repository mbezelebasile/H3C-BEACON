"""
QMIX: Monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning
Reference: Rashid et al., 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque
import random
from .base_trainer import BaseTrainer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class QMIXMixer(nn.Module):
    """Mixing network with monotonicity constraint via abs(weights)."""
    
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = q_values.shape[0]
        
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.squeeze(-1).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition: Dict):
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class QMIXTrainer(BaseTrainer):
    """QMIX with monotonic value decomposition."""
    
    def __init__(self, env, n_agents: int, obs_dim: int, act_dim: int,
                 config: Dict[str, Any], device: str = "cpu"):
        super().__init__(env, n_agents, obs_dim, act_dim, config, device)
        
        self.lr = config.get('lr', 5e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 500000)
        self.batch_size = config.get('batch_size', 32)
        self.target_update = config.get('target_update', 200)
        
        self.state_dim = obs_dim * n_agents
        
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, act_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_q_networks = nn.ModuleList([
            QNetwork(obs_dim, act_dim).to(self.device)
            for _ in range(n_agents)
        ])
        
        self.mixer = QMIXMixer(n_agents, self.state_dim).to(self.device)
        self.target_mixer = QMIXMixer(n_agents, self.state_dim).to(self.device)
        
        self._update_targets(tau=1.0)
        
        params = list(self.mixer.parameters())
        for q_net in self.q_networks:
            params.extend(q_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        
        self.buffer = ReplayBuffer()
        self.update_counter = 0
    
    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.total_steps / self.epsilon_decay)
    
    def select_actions(self, observations: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        actions = []
        
        for i, (obs, q_net) in enumerate(zip(observations, self.q_networks)):
            if explore and random.random() < self.epsilon:
                action = random.randint(0, self.act_dim - 1)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = q_net(obs_tensor)
                action = q_values.argmax(dim=-1).item()
            
            actions.append(action)
        
        return actions
    
    def collect_episode(self, max_steps: int = 25) -> Tuple[List[Dict], float]:
        observations = self.env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            actions = self.select_actions(observations, explore=True)
            next_observations, rewards, dones, _ = self.env.step(actions)
            
            self.buffer.push({
                'obs': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.mean(rewards) if isinstance(rewards, (list, np.ndarray)) else rewards,
                'next_obs': np.array(next_observations),
                'done': all(dones) if isinstance(dones, (list, np.ndarray)) else dones
            })
            
            if isinstance(rewards, (list, np.ndarray)):
                episode_reward += np.mean(rewards)
            else:
                episode_reward += rewards
            
            observations = next_observations
            self.total_steps += self.n_agents
            
            if all(dones) if isinstance(dones, (list, np.ndarray)) else dones:
                break
        
        self.episodes += 1
        return [], episode_reward
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {'loss': 0.0, 'epsilon': self.epsilon}
        
        transitions = self.buffer.sample(self.batch_size)
        
        obs = torch.FloatTensor(np.array([t['obs'] for t in transitions])).to(self.device)
        actions = torch.LongTensor(np.array([t['actions'] for t in transitions])).to(self.device)
        rewards = torch.FloatTensor(np.array([t['rewards'] for t in transitions])).to(self.device)
        next_obs = torch.FloatTensor(np.array([t['next_obs'] for t in transitions])).to(self.device)
        dones = torch.FloatTensor(np.array([t['done'] for t in transitions])).to(self.device)
        
        batch_size = obs.shape[0]
        
        q_values = []
        for i, q_net in enumerate(self.q_networks):
            q = q_net(obs[:, i])
            q_taken = q.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            q_values.append(q_taken)
        q_values = torch.stack(q_values, dim=1)
        
        state = obs.view(batch_size, -1)
        q_tot = self.mixer(q_values, state)
        
        with torch.no_grad():
            target_q_values = []
            for i, target_q_net in enumerate(self.target_q_networks):
                q = target_q_net(next_obs[:, i])
                q_max = q.max(dim=-1)[0]
                target_q_values.append(q_max)
            target_q_values = torch.stack(target_q_values, dim=1)
            
            next_state = next_obs.view(batch_size, -1)
            target_q_tot = self.target_mixer(target_q_values, next_state)
            
            targets = rewards + self.gamma * (1 - dones) * target_q_tot
        
        loss = F.mse_loss(q_tot, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
        for q_net in self.q_networks:
            nn.utils.clip_grad_norm_(q_net.parameters(), 10)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self._update_targets(tau=0.01)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def _update_targets(self, tau: float = 0.01):
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            for param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path: str):
        torch.save({
            'q_networks': [q.state_dict() for q in self.q_networks],
            'mixer': self.mixer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        for i, q_net in enumerate(self.q_networks):
            q_net.load_state_dict(checkpoint['q_networks'][i])
        self.mixer.load_state_dict(checkpoint['mixer'])