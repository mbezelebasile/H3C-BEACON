"""
MAPPO Baseline - FINAL CORRECTED VERSION
=========================================

This implementation fixes ALL identified issues:
1. Episode tracking with max_steps fallback
2. Proper gradient flow 
3. Correct reward accumulation
4. Value normalization
5. Compatible with train.py interface

Author: H3C-BEACON Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Dict, List, Any, Tuple


# ============================================================
# NETWORKS
# ============================================================

class ActorNetwork(nn.Module):
    """Policy network."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
        
        # Orthogonal init
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, obs, explore=True):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if explore:
            action = dist.sample()
        else:
            action = probs.argmax(dim=-1)
        
        return action, dist.log_prob(action), dist.entropy()
    
    def evaluate(self, obs, actions):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy()


class CriticNetwork(nn.Module):
    """Centralized value function."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# ============================================================
# MAPPO TRAINER
# ============================================================

class MAPPOTrainer:
    """
    MAPPO with all fixes for compatibility with train.py
    """
    
    def __init__(self, env, n_agents: int, obs_dim: int, act_dim: int,
                 config: Dict[str, Any] = None, device: str = "cpu"):
        
        self.env = env
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        
        # Config with defaults
        config = config or {}
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.lr = config.get('lr', 3e-4)
        self.lr_actor = config.get('lr_actor', self.lr)
        self.lr_critic = config.get('lr_critic', self.lr)
        
        # CRITICAL: Episode length for environments that don't signal done
        self.max_episode_steps = config.get('max_episode_steps', 25)
        
        # Networks
        state_dim = obs_dim * n_agents
        hidden_dim = config.get('hidden_dim', 64)
        
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic, eps=1e-5
        )
        
        # Running stats for value normalization
        self.value_mean = 0.0
        self.value_std = 1.0
        self.value_count = 1e-4
    
    def _update_value_stats(self, returns):
        """Update running mean/std for value normalization."""
        batch_mean = np.mean(returns)
        batch_var = np.var(returns)
        batch_count = len(returns)
        
        delta = batch_mean - self.value_mean
        tot_count = self.value_count + batch_count
        
        self.value_mean = self.value_mean + delta * batch_count / tot_count
        m_a = self.value_std**2 * self.value_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.value_count * batch_count / tot_count
        self.value_std = np.sqrt(M2 / tot_count + 1e-8)
        self.value_count = tot_count
    
    def _process_obs(self, observations) -> np.ndarray:
        """Convert observations to numpy array."""
        if isinstance(observations, np.ndarray):
            if observations.ndim == 1:
                return observations.reshape(1, -1).astype(np.float32)
            return observations.astype(np.float32)
        
        try:
            result = np.stack([np.asarray(o, dtype=np.float32).flatten() for o in observations])
            return result
        except Exception as e:
            # Handle edge cases
            max_len = max(
                len(o) if hasattr(o, '__len__') else 1 
                for o in observations
            )
            result = np.zeros((len(observations), max_len), dtype=np.float32)
            for i, o in enumerate(observations):
                if hasattr(o, '__len__'):
                    arr = np.asarray(o, dtype=np.float32).flatten()
                    result[i, :len(arr)] = arr
                else:
                    result[i, 0] = float(o)
            return result
    
    def select_actions(self, observations, explore: bool = True) -> List[int]:
        """Select actions for all agents."""
        obs_array = self._process_obs(observations)
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        
        with torch.no_grad():
            actions, _, _ = self.actor.get_action(obs_tensor, explore)
        
        return actions.cpu().numpy().tolist()
    
    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        
        # Append next_value to values for easier indexing
        all_values = np.append(values, next_value)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * all_values[t + 1] * (1 - dones[t]) - all_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, rollout: Dict) -> Dict[str, float]:
        """Perform PPO update."""
        # Extract data
        obs_list = rollout['obs']  # List of (n_agents, obs_dim) arrays
        actions_list = rollout['actions']  # List of (n_agents,) arrays
        rewards_list = rollout['rewards']  # List of (n_agents,) arrays
        dones_list = rollout['dones']  # List of (n_agents,) arrays
        values_list = rollout['values']  # List of floats
        log_probs_list = rollout['log_probs']  # List of (n_agents,) arrays
        next_value = rollout['next_value']
        
        T = len(obs_list)
        
        # Stack into arrays
        obs = np.array(obs_list)  # (T, n_agents, obs_dim)
        actions = np.array(actions_list)  # (T, n_agents)
        rewards = np.array(rewards_list)  # (T, n_agents)
        dones = np.array(dones_list)  # (T, n_agents)
        values = np.array(values_list)  # (T,)
        old_log_probs = np.array(log_probs_list)  # (T, n_agents)
        
        # Mean rewards across agents (cooperative)
        mean_rewards = rewards.mean(axis=-1)  # (T,)
        mean_dones = dones.mean(axis=-1)  # (T,)
        
        # Compute GAE
        advantages, returns = self.compute_gae(mean_rewards, values, next_value, mean_dones)
        
        # Update value stats
        self._update_value_stats(returns)
        
        # Normalize returns
        returns_norm = (returns - self.value_mean) / (self.value_std + 1e-8)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns_norm).to(self.device)
        
        states_t = obs_t.view(T, -1)  # (T, n_agents * obs_dim)
        
        # PPO epochs
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.ppo_epochs):
            # Normalize advantages per epoch
            adv_norm = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            
            # Compute actor loss for each agent
            actor_loss = torch.tensor(0.0, device=self.device)
            entropy_sum = torch.tensor(0.0, device=self.device)
            
            for i in range(self.n_agents):
                agent_obs = obs_t[:, i, :]  # (T, obs_dim)
                agent_actions = actions_t[:, i]  # (T,)
                agent_old_lp = old_lp_t[:, i]  # (T,)
                
                new_lp, entropy = self.actor.evaluate(agent_obs, agent_actions)
                
                ratio = torch.exp(new_lp - agent_old_lp)
                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_norm
                
                actor_loss = actor_loss - torch.min(surr1, surr2).mean()
                entropy_sum = entropy_sum + entropy.mean()
            
            actor_loss = actor_loss / self.n_agents
            entropy_loss = -self.entropy_coef * entropy_sum / self.n_agents
            
            # Update actor
            self.actor_optimizer.zero_grad()
            (actor_loss + entropy_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic loss
            value_pred = self.critic(states_t).squeeze(-1)
            critic_loss = self.value_coef * F.mse_loss(value_pred, returns_t)
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += (entropy_sum / self.n_agents).item()
        
        return {
            'loss': total_actor_loss / self.ppo_epochs,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs,
        }
    
    def train(self, total_steps: int = 1_000_000, eval_interval: int = 160_000,
              log_interval: int = 32000, save_path: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main training loop - Compatible with train.py
        """
        print(f"Training MAPPO on {self.env.__class__.__name__}")
        print(f"Agents: {self.n_agents} | Obs: {self.obs_dim} | Act: {self.act_dim}")
        print(f"Total steps: {total_steps} | Max ep steps: {self.max_episode_steps}")
        print("=" * 60)
        
        # Storage
        rollout = {
            'obs': [], 'actions': [], 'rewards': [], 
            'dones': [], 'values': [], 'log_probs': []
        }
        rollout_length = 512
        
        eval_history = []
        all_train_rewards = []
        best_reward = float('-inf')
        
        start_time = time.time()
        
        # Initial evaluation
        print("\nRunning initial evaluation...")
        init_r, init_std, init_win = self._evaluate(10)
        print(f"Initial: {init_r:.2f} ± {init_std:.2f}\n")
        eval_history.append({'step': 0, 'mean': init_r, 'std': init_std, 'win_rate': init_win})
        best_reward = init_r
        
        # Training state
        obs = self.env.reset()
        episode_reward = 0.0
        episode_step = 0
        episode_rewards = []
        
        for step in range(1, total_steps + 1):
            # Process observation
            obs_array = self._process_obs(obs)
            obs_tensor = torch.FloatTensor(obs_array).to(self.device)
            state_tensor = obs_tensor.view(1, -1)
            
            # Get actions and values
            with torch.no_grad():
                actions, log_probs, _ = self.actor.get_action(obs_tensor, explore=True)
                value = self.critic(state_tensor)
            
            actions_list = actions.cpu().numpy().tolist()
            
            # Environment step
            next_obs, rewards, dones, info = self.env.step(actions_list)
            
            # Process rewards
            if isinstance(rewards, (list, np.ndarray)):
                rewards_array = np.array(rewards, dtype=np.float32)
            else:
                rewards_array = np.array([rewards] * self.n_agents, dtype=np.float32)
            
            # Process dones
            if isinstance(dones, (list, np.ndarray)):
                dones_array = np.array(dones, dtype=np.float32)
            else:
                dones_array = np.array([float(dones)] * self.n_agents, dtype=np.float32)
            
            # Store transition
            rollout['obs'].append(obs_array)
            rollout['actions'].append(np.array(actions_list))
            rollout['rewards'].append(rewards_array)
            rollout['dones'].append(dones_array)
            rollout['values'].append(value.item())
            rollout['log_probs'].append(log_probs.cpu().numpy())
            
            # Accumulate episode reward
            episode_reward += float(np.mean(rewards_array))
            episode_step += 1
            
            # Check episode end
            env_done = any(dones) if isinstance(dones, (list, np.ndarray)) else dones
            time_limit = episode_step >= self.max_episode_steps
            episode_done = env_done or time_limit
            
            if episode_done:
                episode_rewards.append(episode_reward)
                all_train_rewards.append(episode_reward)
                episode_reward = 0.0
                episode_step = 0
                obs = self.env.reset()
            else:
                obs = next_obs
            
            # Update policy
            if len(rollout['obs']) >= rollout_length:
                # Get next value for GAE
                next_obs_array = self._process_obs(obs)
                next_state = torch.FloatTensor(next_obs_array).view(1, -1).to(self.device)
                with torch.no_grad():
                    next_value = self.critic(next_state).item()
                
                rollout['next_value'] = next_value
                
                # Update
                update_info = self.update(rollout)
                
                # Clear rollout
                rollout = {
                    'obs': [], 'actions': [], 'rewards': [], 
                    'dones': [], 'values': [], 'log_probs': []
                }
                
                # Log
                if step % log_interval == 0:
                    elapsed = (time.time() - start_time) / 60
                    recent_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    episode_rewards = []  # Reset for next interval
                    print(f"Step {step:7d} | Train: {recent_r:8.2f} | "
                          f"Loss: {update_info['loss']:.4f} | H: {update_info['entropy']:.2f} | "
                          f"Time: {elapsed:.1f}min")
            
            # Evaluation
            if step % eval_interval == 0:
                eval_r, eval_std, win_rate = self._evaluate(10)
                eval_history.append({
                    'step': step, 'mean': eval_r, 'std': eval_std, 'win_rate': win_rate
                })
                
                is_best = eval_r > best_reward
                if is_best:
                    best_reward = eval_r
                    if save_path:
                        self._save_best(save_path)
                
                status = f"★ NEW BEST: {eval_r:.2f}" if is_best else f"Best: {best_reward:.2f}"
                print(f"[EVAL] Step {step} | Reward: {eval_r:.2f} ± {eval_std:.2f} | "
                      f"{status} | Win: {win_rate*100:.1f}%")
        
        # Final evaluation
        final_r, final_std, final_win = self._evaluate(10)
        
        # Compute metrics
        eval_values = [e['mean'] for e in eval_history]
        auc = np.trapz(eval_values) / len(eval_values) if len(eval_values) > 1 else eval_values[0]
        variance = np.var(eval_values) if len(eval_values) > 1 else 0
        mean_eval = np.mean(eval_values) if eval_values else 0
        cv = np.std(eval_values) / (abs(mean_eval) + 1e-8) if len(eval_values) > 1 else 0
        stability = 1 / (1 + cv)
        
        # Steps to 90%
        target = best_reward * 0.9 if best_reward >= 0 else best_reward * 1.1
        steps_to_90 = None
        for e in eval_history:
            if e['mean'] >= target:
                steps_to_90 = e['step']
                break
        
        results = {
            'algorithm': 'MAPPO',
            'best': best_reward,
            'final_mean': final_r,
            'final_std': final_std,
            'final_win_rate': final_win * 100,
            'auc': auc,
            'variance': variance,
            'cv': cv,
            'stability': stability,
            'steps_to_90': steps_to_90,
            'convergence_speed': 0,
            'total_steps': total_steps,
            'eval_history': eval_history,
        }
        
        # Summary
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Final: {final_r:.2f} ± {final_std:.2f}")
        print(f"Best:  {best_reward:.2f}")
        print(f"Win Rate: {final_win*100:.1f}%")
        print(f"\nStability Metrics:")
        print(f"   Variance: {variance:.4f}")
        print(f"   CV: {cv:.4f}")
        print(f"   Stability Score: {stability:.4f}")
        print(f"\nConvergence Metrics:")
        print(f"   AUC: {auc:.2f}")
        print(f"   Steps to 90%: {steps_to_90 if steps_to_90 else 'N/A'}")
        print("=" * 60 + "\n")
        
        return results
    
    def _evaluate(self, n_episodes: int = 10) -> Tuple[float, float, float]:
        """Evaluate policy."""
        rewards = []
        wins = 0
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            ep_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < self.max_episode_steps:
                actions = self.select_actions(obs, explore=False)
                obs, r, dones, info = self.env.step(actions)
                
                if isinstance(r, (list, np.ndarray)):
                    ep_reward += float(np.mean(r))
                else:
                    ep_reward += float(r)
                
                steps += 1
                
                if isinstance(dones, (list, np.ndarray)):
                    done = any(dones)
                else:
                    done = bool(dones)
            
            rewards.append(ep_reward)
            
            if isinstance(info, dict) and info.get('win', False):
                wins += 1
            elif ep_reward > 0:
                wins += 1
        
        return float(np.mean(rewards)), float(np.std(rewards)), wins / n_episodes
    
    def _save_best(self, save_path: str):
        """Save best model."""
        if os.path.isdir(save_path):
            path = os.path.join(save_path, "best_model.pt")
        else:
            path = save_path
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])