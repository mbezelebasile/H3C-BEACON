"""
IPPO: Independent Proximal Policy Optimization
Reference: de Witt et al., 2020 - "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from .base_trainer import BaseTrainer


class IndependentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value
    
    def get_action(self, obs: torch.Tensor, explore: bool = True):
        logits, value = self.forward(obs)
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
        
        return action, log_prob, entropy, value.squeeze(-1)


class IPPOTrainer(BaseTrainer):
    """Independent PPO - each agent learns independently."""
    
    def __init__(self, env, n_agents: int, obs_dim: int, act_dim: int,
                 config: Dict[str, Any], device: str = "cpu"):
        super().__init__(env, n_agents, obs_dim, act_dim, config, device)
        
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        
        self.agents = nn.ModuleList([
            IndependentActorCritic(obs_dim, act_dim).to(self.device)
            for _ in range(n_agents)
        ])
        
        self.optimizers = [
            torch.optim.Adam(agent.parameters(), lr=self.lr)
            for agent in self.agents
        ]
    
    def select_actions(self, observations: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        actions = []
        
        for i, (obs, agent) in enumerate(zip(observations, self.agents)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action(obs_tensor, explore)
            
            actions.append(action.item())
        
        return actions
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        T = obs.shape[0]
        total_loss = 0
        total_entropy = 0
        
        for agent_idx, (agent, optimizer) in enumerate(zip(self.agents, self.optimizers)):
            agent_obs = obs[:, agent_idx]
            agent_actions = actions[:, agent_idx]
            agent_rewards = rewards[:, agent_idx]
            agent_next_obs = next_obs[:, agent_idx]
            agent_dones = dones[:, agent_idx]
            
            with torch.no_grad():
                _, values = agent(agent_obs)
                _, next_values = agent(agent_next_obs)
                values = values.squeeze(-1)
                next_values = next_values.squeeze(-1)
                
                advantages = torch.zeros(T).to(self.device)
                gae = 0
                for t in reversed(range(T)):
                    delta = agent_rewards[t] + self.gamma * next_values[t] * (1 - agent_dones[t]) - values[t]
                    gae = delta + self.gamma * self.gae_lambda * (1 - agent_dones[t]) * gae
                    advantages[t] = gae
                
                # CRITICAL: Normalize advantages for stable training
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                returns = advantages + values
            
            with torch.no_grad():
                old_logits, _ = agent(agent_obs)
                old_probs = F.softmax(old_logits, dim=-1)
                old_dist = torch.distributions.Categorical(old_probs)
                old_log_probs = old_dist.log_prob(agent_actions)
            
            for _ in range(self.ppo_epochs):
                logits, values_pred = agent(agent_obs)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(agent_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(values_pred.squeeze(-1), returns)
                
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                total_entropy += entropy.item()
        
        n_updates = self.n_agents * self.ppo_epochs
        return {'loss': total_loss / n_updates, 'entropy': total_entropy / n_updates}
    
    def save(self, path: str):
        torch.save({f'agent_{i}': agent.state_dict() for i, agent in enumerate(self.agents)}, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(checkpoint[f'agent_{i}'])