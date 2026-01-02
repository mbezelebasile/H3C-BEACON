"""
FACMAC: Factored Multi-Agent Centralised Policy Gradients
Reference: Peng et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from .base_trainer import BaseTrainer


class FACMACActor(nn.Module):
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
        return F.softmax(self.network(obs), dim=-1)


class FACMACCritic(nn.Module):
    """Factored centralized critic."""
    
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int, hidden_dim: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        
        self.agent_q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        state_dim = obs_dim * n_agents
        self.mixer = nn.Sequential(
            nn.Linear(n_agents + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        
        if actions.dim() == 2:
            actions = F.one_hot(actions, self.act_dim).float()
        
        q_values = []
        for i in range(self.n_agents):
            agent_input = torch.cat([obs[:, i], actions[:, i]], dim=-1)
            q = self.agent_q(agent_input)
            q_values.append(q)
        q_values = torch.cat(q_values, dim=-1)
        
        state = obs.view(batch_size, -1)
        mixer_input = torch.cat([q_values, state], dim=-1)
        q_tot = self.mixer(mixer_input)
        
        return q_tot.squeeze(-1)


class FACMACTrainer(BaseTrainer):
    """FACMAC with factored critics."""
    
    def __init__(self, env, n_agents: int, obs_dim: int, act_dim: int,
                 config: Dict[str, Any], device: str = "cpu"):
        super().__init__(env, n_agents, obs_dim, act_dim, config, device)
        
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        self.actors = nn.ModuleList([
            FACMACActor(obs_dim, act_dim).to(self.device)
            for _ in range(n_agents)
        ])
        
        self.critic = FACMACCritic(obs_dim, act_dim, n_agents).to(self.device)
        self.target_critic = FACMACCritic(obs_dim, act_dim, n_agents).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        actor_params = []
        for actor in self.actors:
            actor_params.extend(actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    
    def select_actions(self, observations: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        actions = []
        
        for i, (obs, actor) in enumerate(zip(observations, self.actors)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = actor(obs_tensor).squeeze(0)
            
            if explore:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            else:
                action = probs.argmax()
            
            actions.append(action.item())
        
        return actions
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        T = obs.shape[0]
        mean_rewards = rewards.mean(dim=-1)
        mean_dones = dones.mean(dim=-1)
        
        actions_onehot = F.one_hot(actions, self.act_dim).float()
        
        # Critic update
        with torch.no_grad():
            next_actions = []
            for i, actor in enumerate(self.actors):
                probs = actor(next_obs[:, i])
                next_actions.append(probs)
            next_actions = torch.stack(next_actions, dim=1)
            
            next_q = self.target_critic(next_obs, next_actions)
            targets = mean_rewards + self.gamma * (1 - mean_dones) * next_q
        
        current_q = self.critic(obs, actions_onehot)
        critic_loss = F.mse_loss(current_q, targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Actor update
        current_actions = []
        entropies = []
        
        for i, actor in enumerate(self.actors):
            probs = actor(obs[:, i])
            current_actions.append(probs)
            
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropies.append(entropy)
        
        current_actions = torch.stack(current_actions, dim=1)
        
        q_values = self.critic(obs, current_actions)
        actor_loss = -q_values.mean()
        
        entropy = torch.stack(entropies, dim=1).mean()
        actor_loss -= self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()], self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'loss': actor_loss.item() + critic_loss.item(), 'entropy': entropy.item()}
    
    def save(self, path: str):
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])