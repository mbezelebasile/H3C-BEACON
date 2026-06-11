"""
COMA: Counterfactual Multi-Agent Policy Gradients
Reference: Foerster et al., 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from .base_trainer import BaseTrainer


class COMActor(nn.Module):
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


class COMACritic(nn.Module):
    """Centralized critic with counterfactual baseline."""
    
    def __init__(self, state_dim: int, n_agents: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        
        input_dim = state_dim + n_agents + n_agents * act_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, state: torch.Tensor, agent_id: int, actions_onehot: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        
        agent_onehot = torch.zeros(batch_size, self.n_agents).to(state.device)
        agent_onehot[:, agent_id] = 1
        
        actions_flat = actions_onehot.view(batch_size, -1)
        inputs = torch.cat([state, agent_onehot, actions_flat], dim=-1)
        
        return self.network(inputs)


class COMATrainer(BaseTrainer):
    """COMA with counterfactual baseline."""
    
    def __init__(self, env, n_agents: int, obs_dim: int, act_dim: int,
                 config: Dict[str, Any], device: str = "cpu"):
        super().__init__(env, n_agents, obs_dim, act_dim, config, device)
        
        self.lr_actor = config.get('lr_actor', 1e-4)
        self.lr_critic = config.get('lr_critic', 1e-3)
        self.gamma = config.get('gamma', 0.99)
        self.td_lambda = config.get('td_lambda', 0.8)
        self.max_grad_norm = config.get('max_grad_norm', 10)
        
        self.state_dim = obs_dim * n_agents
        self.actors = nn.ModuleList([
            COMActor(obs_dim, act_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.critic = COMACritic(self.state_dim, n_agents, act_dim).to(self.device)
        
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
        states = obs.view(T, -1)
        actions_onehot = F.one_hot(actions, self.act_dim).float()
        
        mean_rewards = rewards.mean(dim=-1)
        mean_dones = dones.mean(dim=-1)
        
        # Critic update
        critic_loss = 0
        for agent_id in range(self.n_agents):
            q_values = self.critic(states, agent_id, actions_onehot)
            q_taken = q_values.gather(1, actions[:, agent_id].unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_states = next_obs.view(T, -1)
                next_q = self.critic(next_states, agent_id, actions_onehot)
                next_v = (next_q * F.softmax(next_q, dim=-1)).sum(dim=-1)
                targets = mean_rewards + self.gamma * (1 - mean_dones) * next_v
            
            critic_loss += F.mse_loss(q_taken, targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Actor update with counterfactual baseline
        actor_loss = 0
        total_entropy = 0
        
        for agent_id, actor in enumerate(self.actors):
            agent_obs = obs[:, agent_id]
            probs = actor(agent_obs)
            
            with torch.no_grad():
                q_values = self.critic(states, agent_id, actions_onehot)
                baseline = (probs * q_values).sum(dim=-1)
                q_taken = q_values.gather(1, actions[:, agent_id].unsqueeze(1)).squeeze(1)
                advantage = q_taken - baseline
                # CRITICAL: Normalize advantages for stable training
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            log_probs = torch.log(probs + 1e-8)
            log_prob_taken = log_probs.gather(1, actions[:, agent_id].unsqueeze(1)).squeeze(1)
            
            actor_loss -= (log_prob_taken * advantage).mean()
            
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            total_entropy += entropy.item()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()], self.max_grad_norm)
        self.actor_optimizer.step()
        
        return {'loss': actor_loss.item() + critic_loss.item(), 'entropy': total_entropy / self.n_agents}
    
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