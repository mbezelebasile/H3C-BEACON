"""
MAPPO: Multi-Agent Proximal Policy Optimization
Reference: Yu et al., 2022 - "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from .base_trainer import BaseTrainer


class Actor(nn.Module):
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
    
    def get_action(self, obs: torch.Tensor, explore: bool = True):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        
        if explore:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MAPPOTrainer(BaseTrainer):
    """MAPPO with centralized critic and decentralized actors."""
    
    def __init__(self, env, n_agents: int, obs_dim: int, act_dim: int, 
                 config: Dict[str, Any], device: str = "cpu"):
        super().__init__(env, n_agents, obs_dim, act_dim, config, device)
        
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        
        state_dim = obs_dim * n_agents
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    
    def select_actions(self, observations: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        # Handle potentially heterogeneous observations by stacking safely
        try:
            obs_array = np.stack([np.array(o, dtype=np.float32) for o in observations])
        except ValueError:
            # Fallback: pad to max length
            max_len = max(len(o) for o in observations)
            obs_array = np.zeros((len(observations), max_len), dtype=np.float32)
            for i, o in enumerate(observations):
                obs_array[i, :len(o)] = o
        
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        
        with torch.no_grad():
            actions, _, _ = self.actor.get_action(obs_tensor, explore)
        
        return actions.cpu().numpy().tolist()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        T = obs.shape[0]
        
        with torch.no_grad():
            states = obs.view(T, -1)
            next_states = next_obs.view(T, -1)
            
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            
            mean_rewards = rewards.mean(dim=-1)
            mean_dones = dones.mean(dim=-1)
            
            advantages = torch.zeros(T).to(self.device)
            gae = 0
            for t in reversed(range(T)):
                delta = mean_rewards[t] + self.gamma * next_values[t] * (1 - mean_dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - mean_dones[t]) * gae
                advantages[t] = gae
            
            # CRITICAL: Normalize advantages for stable training
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            returns = advantages + values
        
        with torch.no_grad():
            old_logits = self.actor(obs.view(-1, self.obs_dim))
            old_probs = F.softmax(old_logits, dim=-1)
            old_dist = torch.distributions.Categorical(old_probs)
            old_log_probs = old_dist.log_prob(actions.view(-1)).view(T, self.n_agents)
        
        total_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            logits = self.actor(obs.view(-1, self.obs_dim))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions.view(-1)).view(T, self.n_agents)
            entropy = dist.entropy().view(T, self.n_agents).mean()
            
            agent_advantages = advantages.unsqueeze(-1).expand(-1, self.n_agents)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * agent_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            states = obs.view(T, -1)
            value_pred = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(value_pred, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_loss += actor_loss.item()
            total_entropy += entropy.item()
        
        return {'loss': total_loss / self.ppo_epochs, 'entropy': total_entropy / self.ppo_epochs}
    
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])