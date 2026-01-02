"""
H3CTrainer.py - H3C-BEACON:


1. SMART RECOVERY with Patience
   - Don't recover on single bad eval
   - Require 3 consecutive drops before recovery
   - Cooldown period: 20K steps between recoveries
   - Max 5 recoveries total

2. PROTECTED LEARNING RATE
   - LR floor: never below 30% of original
   - Only reduce LR after 3+ recoveries
   - Gradual reduction: 0.85x instead of 0.7x

3. STABLE BEST TRACKING
   - Use rolling average of last 3 evals
   - "Best" requires 2 consecutive good evals
   - Track both "peak" and "stable best"

4. DIRECT ENTROPY CONTROL
   - Adjust entropy_coef directly, not just loss weight
   - Hard bounds: 0.5 < H < 2.0
   - Automatic coefficient scaling

5. EVALUATION STABILITY
   - 15 episodes instead of 10
   - Use median for recovery decisions
   - Rolling evaluation window

Authors: Basile BETE MBEZELE, Ghislain ALO'O ABESSOLO

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy
import heapq
import math

from modules.AAH import AAH
from modules.PCC import PCC
from modules.CDEGA import CDEGA
from modules.AUTO_HP import AUTO_HP


# ============================================================
# 1. DYNAMIC GRAPH ATTENTION NETWORK (DGAT)
# ============================================================

class DynamicGraphAttention(nn.Module):
    """
    Dynamic Graph Attention for inter-agent communication.
    
    Key features:
    - Learns which agents should communicate
    - Edge weights via multi-head attention
    - Distance-based gating (closer agents communicate more)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Distance-based gating (positions are in obs)
        self.distance_gate = nn.Sequential(
            nn.Linear(2, 32),  # 2D relative position
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.orthogonal_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, n_agents, input_dim]
            positions: [batch, n_agents, 2] - agent positions (optional)
        
        Returns:
            out: [batch, n_agents, hidden_dim] - attended features
            attention: [batch, n_heads, n_agents, n_agents] - attention weights
        """
        B, N, _ = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Distance-based gating (if positions provided)
        if positions is not None:
            # Compute pairwise distances
            pos_i = positions.unsqueeze(2)  # [B, N, 1, 2]
            pos_j = positions.unsqueeze(1)  # [B, 1, N, 2]
            rel_pos = pos_i - pos_j  # [B, N, N, 2]
            
            # Gate based on distance
            gate = self.distance_gate(rel_pos)  # [B, N, N, 1]
            gate = gate.squeeze(-1).unsqueeze(1)  # [B, 1, N, N]
            
            # Apply gate to attention
            scores = scores * gate
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # [B, n_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        
        # Output projection + residual
        out = self.W_o(out)
        out = self.layer_norm(out + x[:, :, :self.hidden_dim] if x.shape[-1] >= self.hidden_dim else out)
        
        return out, attention


# ============================================================
# 2. BAYESIAN BELIEF FUSION
# ============================================================

class BayesianBeliefFusion(nn.Module):
    """
    Bayesian belief fusion instead of simple averaging.
    
    bᵢ(t+1) = Normalize(P(oᵢ|s) * ∏ messages_j→i)
    
    Key insight: Product is more powerful than sum/average
    because it enforces agreement between sources.
    """
    
    def __init__(self, obs_dim: int, belief_dim: int, hidden_dim: int, message_dim: int = None):
        super().__init__()
        
        self.belief_dim = belief_dim
        # Messages come from DGAT which outputs hidden_dim
        self.message_input_dim = message_dim if message_dim is not None else hidden_dim
        
        # Likelihood encoder: P(o|belief)
        self.likelihood_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim * 2)  # mean + log_var
        )
        
        # Message encoder - INPUT is hidden_dim (from DGAT), not belief_dim
        self.message_encoder = nn.Sequential(
            nn.Linear(self.message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim * 2)  # mean + log_var
        )
        
        # Fusion output
        self.fusion_output = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim)
        )
        
        # Prior (learnable)
        self.prior_mean = nn.Parameter(torch.zeros(belief_dim))
        self.prior_log_var = nn.Parameter(torch.zeros(belief_dim))
    
    def forward(self, obs: torch.Tensor, messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian fusion of observation and messages.
        
        Args:
            obs: [batch, n_agents, obs_dim]
            messages: [batch, n_agents, n_agents, belief_dim] - messages from all agents
        
        Returns:
            belief: [batch, n_agents, belief_dim]
            uncertainty: [batch, n_agents]
        """
        B, N, _ = obs.shape
        
        # Encode observation likelihood
        obs_params = self.likelihood_encoder(obs)
        obs_mean = obs_params[..., :self.belief_dim]
        obs_log_var = obs_params[..., self.belief_dim:]
        
        # Start with prior
        prior_var = torch.exp(self.prior_log_var)
        
        # Bayesian update: product of Gaussians
        # For Gaussians: 1/σ²_post = 1/σ²_prior + Σ 1/σ²_i
        # μ_post = σ²_post * (μ_prior/σ²_prior + Σ μ_i/σ²_i)
        
        precision = 1.0 / (prior_var + 1e-6)
        weighted_mean = self.prior_mean * precision
        
        # Add observation likelihood
        obs_var = torch.exp(obs_log_var)
        obs_precision = 1.0 / (obs_var + 1e-6)
        precision = precision + obs_precision
        weighted_mean = weighted_mean + obs_mean * obs_precision
        
        # Add message likelihoods (product over senders)
        if messages is not None and messages.numel() > 0:
            for j in range(N):
                msg = messages[:, :, j, :]  # Messages from agent j
                msg_params = self.message_encoder(msg)
                msg_mean = msg_params[..., :self.belief_dim]
                msg_log_var = msg_params[..., self.belief_dim:]
                msg_var = torch.exp(msg_log_var)
                msg_precision = 1.0 / (msg_var + 1e-6)
                
                # Bayesian update
                precision = precision + msg_precision
                weighted_mean = weighted_mean + msg_mean * msg_precision
        
        # Compute posterior
        posterior_var = 1.0 / (precision + 1e-6)
        posterior_mean = weighted_mean * posterior_var
        
        # Output belief
        belief = self.fusion_output(posterior_mean)
        
        # Uncertainty = trace of posterior variance
        uncertainty = posterior_var.sum(dim=-1)
        
        return belief, uncertainty


# ============================================================
# 3. ADAPTIVE COALITION FORMATION (FIXED v5.1)
# ============================================================

class AdaptiveCoalitionFormation(nn.Module):
    """
    Dynamic coalition formation (1-4 coalitions) - FIXED VERSION.
    
    v5.0 Problem: Always stuck at 2 coalitions due to spectral clustering issues.
    v5.1 Fix: 
    - Forced exploration in early training
    - Performance-based adaptation
    - Distance-based heuristics for 3 agents
    """
    
    def __init__(self, n_agents: int, hidden_dim: int, max_coalitions: int = 4):
        super().__init__()
        
        self.n_agents = n_agents
        self.max_coalitions = min(max_coalitions, n_agents)  # Can't have more coalitions than agents
        
        # Coalition scoring network
        self.coalition_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_coalitions)
        )
        
        # Affinity network (which agents should be together)
        self.affinity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Current coalition assignments
        self.register_buffer('coalition_assignments', torch.zeros(n_agents, dtype=torch.long))
        self.register_buffer('n_active_coalitions', torch.tensor(2))
        
        self.update_counter = 0
        self.update_interval = 200  # Less frequent updates for stability
        
        # Performance tracking for adaptation
        self.coalition_rewards = {i: [] for i in range(1, max_coalitions + 1)}
        self.current_coalition_start_reward = None
        self.exploration_phase = True
        self.exploration_steps = 50000  # Explore for first 50K steps
    
    def compute_affinity(self, agent_features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise affinity between agents."""
        B, N, D = agent_features.shape
        
        # Pairwise features
        feat_i = agent_features.unsqueeze(2).expand(-1, -1, N, -1)
        feat_j = agent_features.unsqueeze(1).expand(-1, N, -1, -1)
        pairwise = torch.cat([feat_i, feat_j], dim=-1)
        
        # Affinity scores
        affinity = self.affinity_net(pairwise).squeeze(-1)  # [B, N, N]
        affinity = (affinity + affinity.transpose(-1, -2)) / 2  # Symmetrize
        
        return torch.sigmoid(affinity)
    
    def record_reward(self, reward: float):
        """Record reward for current coalition configuration."""
        n_coal = self.n_active_coalitions.item()
        self.coalition_rewards[n_coal].append(reward)
        # Keep only recent rewards
        if len(self.coalition_rewards[n_coal]) > 100:
            self.coalition_rewards[n_coal] = self.coalition_rewards[n_coal][-100:]
    
    def get_best_n_coalitions(self) -> int:
        """Get the coalition count with best average reward."""
        best_n = 2  # Default
        best_reward = float('-inf')
        
        for n_coal, rewards in self.coalition_rewards.items():
            if len(rewards) >= 10:  # Need enough samples
                avg = np.mean(rewards[-50:])  # Recent average
                if avg > best_reward:
                    best_reward = avg
                    best_n = n_coal
        
        return best_n
    
    def update_coalitions(self, agent_features: torch.Tensor, positions: torch.Tensor = None, 
                         step: int = 0, current_reward: float = None):
        """Update coalition assignments with exploration/exploitation."""
        self.update_counter += 1
        
        # Record reward if provided
        if current_reward is not None:
            self.record_reward(current_reward)
        
        if self.update_counter % self.update_interval != 0:
            return
        
        # Check if still in exploration phase
        self.exploration_phase = step < self.exploration_steps
        
        with torch.no_grad():
            if self.exploration_phase:
                # EXPLORATION: Try different configurations
                # Weighted random choice favoring 1 and 2 for simple_spread
                if self.n_agents == 3:
                    # For 3 agents: 1 (all together), 2 (pair+single), 3 (all separate)
                    probs = [0.25, 0.45, 0.30]  # Favor 2 but explore others
                else:
                    probs = [1.0 / self.max_coalitions] * self.max_coalitions
                
                n_coalitions = np.random.choice(
                    range(1, self.max_coalitions + 1), 
                    p=probs
                )
            else:
                # EXPLOITATION: Use best performing configuration
                n_coalitions = self.get_best_n_coalitions()
            
            self.n_active_coalitions = torch.tensor(n_coalitions, device=agent_features.device)
            
            # Assign agents to coalitions based on positions/features
            assignments = torch.zeros(self.n_agents, dtype=torch.long, device=agent_features.device)
            
            if n_coalitions == 1:
                # All agents in same coalition
                assignments[:] = 0
            elif n_coalitions == self.n_agents:
                # Each agent in own coalition
                assignments = torch.arange(self.n_agents, device=agent_features.device)
            else:
                # Use position-based clustering
                if positions is not None:
                    pos = positions.mean(dim=0)  # [N, 2]
                    
                    # For 2 coalitions with 3 agents: find closest pair
                    if n_coalitions == 2 and self.n_agents == 3:
                        dists = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
                        dists.fill_diagonal_(float('inf'))
                        
                        # Find closest pair
                        min_idx = torch.argmin(dists)
                        i, j = min_idx // self.n_agents, min_idx % self.n_agents
                        
                        # Pair goes to coalition 0, other to coalition 1
                        assignments[i] = 0
                        assignments[j] = 0
                        for k in range(self.n_agents):
                            if k != i and k != j:
                                assignments[k] = 1
                    else:
                        # General k-means style assignment
                        feat_mean = agent_features.mean(dim=0)  # [N, D]
                        indices = torch.randperm(self.n_agents)[:n_coalitions]
                        centroids = feat_mean[indices]
                        
                        for i in range(self.n_agents):
                            dists = torch.norm(feat_mean[i:i+1] - centroids, dim=1)
                            assignments[i] = torch.argmin(dists)
                else:
                    # Random assignment
                    assignments = torch.randint(0, n_coalitions, (self.n_agents,), device=agent_features.device)
            
            self.coalition_assignments = assignments
    
    def get_coalition_mask(self) -> torch.Tensor:
        """Get mask indicating which agents are in same coalition."""
        mask = (self.coalition_assignments.unsqueeze(0) == 
                self.coalition_assignments.unsqueeze(1)).float()
        return mask
    
    def forward(self, agent_features: torch.Tensor, positions: torch.Tensor = None,
               step: int = 0, current_reward: float = None) -> Dict:
        """
        Returns coalition information.
        """
        self.update_coalitions(agent_features, positions, step, current_reward)
        
        return {
            'assignments': self.coalition_assignments,
            'n_coalitions': self.n_active_coalitions,
            'coalition_mask': self.get_coalition_mask(),
            'exploration_phase': self.exploration_phase
        }


# ============================================================
# 4. DUAL CRITIC (Central + Local with learned mixing)
# ============================================================

class DualCritic(nn.Module):
    """
    Dual critic: Central + Local-GNN with learned mixing.
    
    V_total = w * V_central + (1-w) * V_local
    
    w is learned automatically based on situation.
    """
    
    def __init__(self, obs_dim: int, n_agents: int, hidden_dim: int):
        super().__init__()
        
        self.n_agents = n_agents
        global_dim = obs_dim * n_agents
        
        # Central critic (full state)
        self.central_critic = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_agents)
        )
        
        # Local critic (per-agent with GNN context)
        self.local_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.local_gnn = DynamicGraphAttention(hidden_dim, hidden_dim, n_heads=2)
        
        self.local_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Mixing weight (learned)
        self.mixing_net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, positions: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            positions: [batch, n_agents, 2]
        
        Returns:
            values: [batch, n_agents]
            info: dict with component values
        """
        B, N, D = obs.shape
        global_obs = obs.view(B, -1)
        
        # Central critic
        v_central = self.central_critic(global_obs)  # [B, N]
        
        # Local critic with GNN
        local_feat = self.local_encoder(obs)  # [B, N, hidden]
        gnn_feat, _ = self.local_gnn(local_feat, positions)  # [B, N, hidden]
        v_local = self.local_value_head(gnn_feat).squeeze(-1)  # [B, N]
        
        # Learned mixing weight
        w = self.mixing_net(global_obs)  # [B, 1]
        
        # Mix values
        values = w * v_central + (1 - w) * v_local  # [B, N]
        
        return values, {
            'v_central': v_central,
            'v_local': v_local,
            'mixing_weight': w
        }


# ============================================================
# 5. RTD++ WITH KL TO ELITE POLICY
# ============================================================

class ElitePolicyTracker:
    """
    Track and maintain elite policy for KL distillation.
    
    Uses EMA to smooth elite policy.
    """
    
    def __init__(self, ema_decay: float = 0.995):
        self.ema_decay = ema_decay
        self.elite_policy_params = None
        self.elite_reward_threshold = float('-inf')
        self.n_updates = 0
    
    def update(self, policy_state_dict: dict, reward: float):
        """Update elite policy if new best or EMA update."""
        if reward > self.elite_reward_threshold:
            self.elite_reward_threshold = reward
            
            if self.elite_policy_params is None:
                self.elite_policy_params = {k: v.clone() for k, v in policy_state_dict.items()}
            else:
                # EMA update
                for k, v in policy_state_dict.items():
                    self.elite_policy_params[k] = (
                        self.ema_decay * self.elite_policy_params[k] + 
                        (1 - self.ema_decay) * v
                    )
            
            self.n_updates += 1
    
    def get_elite_params(self) -> Optional[dict]:
        return self.elite_policy_params


class EliteTrajectoryBuffer:
    """Elite trajectory buffer with improved sampling."""
    
    def __init__(self, capacity: int = 30):
        self.capacity = capacity
        self.trajectories = []
        self.total_added = 0
        self.mean_elite_reward = float('-inf')
    
    def add_trajectory(self, trajectory: Dict, total_reward: float):
        if len(self.trajectories) < self.capacity:
            heapq.heappush(self.trajectories, (total_reward, self.total_added, trajectory))
            self.total_added += 1
        elif total_reward > self.trajectories[0][0]:
            heapq.heapreplace(self.trajectories, (total_reward, self.total_added, trajectory))
            self.total_added += 1
        
        if len(self.trajectories) > 0:
            self.mean_elite_reward = np.mean([t[0] for t in self.trajectories])
    
    def sample_trajectories(self, n: int) -> List[Dict]:
        if len(self.trajectories) == 0:
            return []
        
        n = min(n, len(self.trajectories))
        rewards = np.array([t[0] for t in self.trajectories])
        rewards = rewards - rewards.min() + 1e-6
        probs = rewards / rewards.sum()
        
        indices = np.random.choice(len(self.trajectories), size=n, replace=False, p=probs)
        return [self.trajectories[i][2] for i in indices]
    
    def get_threshold(self) -> float:
        return self.trajectories[0][0] if len(self.trajectories) > 0 else float('-inf')
    
    def __len__(self):
        return len(self.trajectories)


# ============================================================
# 6. ROLLOUT BUFFER
# ============================================================

class DGATRolloutBuffer:
    """Rollout buffer for DGAT-BC."""
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4
        
        self.current_trajectory = {'obs': [], 'actions': [], 'rewards': [], 'log_probs': []}
        self.completed_trajectories = []
        self.reset()
    
    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.action_probs = []
        self.current_trajectory = {'obs': [], 'actions': [], 'rewards': [], 'log_probs': []}
        self.completed_trajectories = []
    
    def add(self, obs, actions, rewards, dones, values, log_probs, action_probs):
        self.obs.append(obs.copy())
        self.actions.append(actions.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        self.values.append(values.copy())
        self.log_probs.append(log_probs.copy())
        self.action_probs.append(action_probs.copy())
        
        self.current_trajectory['obs'].append(obs.copy())
        self.current_trajectory['actions'].append(actions.copy())
        self.current_trajectory['rewards'].append(float(np.mean(rewards)))
        self.current_trajectory['log_probs'].append(log_probs.copy().mean())
        
        if np.any(dones > 0.5):
            self._complete_trajectory()
    
    def _complete_trajectory(self):
        if len(self.current_trajectory['obs']) > 0:
            traj = {
                'obs': np.array(self.current_trajectory['obs']),
                'actions': np.array(self.current_trajectory['actions']),
                'rewards': np.array(self.current_trajectory['rewards']),
                'log_probs': np.array(self.current_trajectory['log_probs']),
            }
            total_reward = sum(self.current_trajectory['rewards'])
            self.completed_trajectories.append((traj, total_reward))
        
        self.current_trajectory = {'obs': [], 'actions': [], 'rewards': [], 'log_probs': []}
    
    def get_completed_trajectories(self):
        return self.completed_trajectories
    
    def compute_gae(self, last_values: np.ndarray):
        obs = np.array(self.obs)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        log_probs = np.array(self.log_probs)
        action_probs = np.array(self.action_probs)
        
        T, n_agents = rewards.shape
        
        # Normalize rewards
        flat_rewards = rewards.flatten()
        batch_mean = np.mean(flat_rewards)
        batch_var = np.var(flat_rewards) + 1e-8
        delta = batch_mean - self.reward_mean
        total = self.reward_count + len(flat_rewards)
        self.reward_mean += delta * len(flat_rewards) / total
        self.reward_var = (self.reward_var * self.reward_count + batch_var * len(flat_rewards)) / total
        self.reward_count = total
        
        rewards_norm = (rewards - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)
        rewards_norm = np.clip(rewards_norm, -10, 10)
        
        advantages = np.zeros((T, n_agents), dtype=np.float32)
        returns = np.zeros((T, n_agents), dtype=np.float32)
        
        for agent_idx in range(n_agents):
            gae = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = last_values[agent_idx]
                    next_done = 0
                else:
                    next_value = values[t + 1, agent_idx]
                    next_done = dones[t + 1, agent_idx]
                
                delta = rewards_norm[t, agent_idx] + self.gamma * next_value * (1 - next_done) - values[t, agent_idx]
                gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
                advantages[t, agent_idx] = gae
                returns[t, agent_idx] = gae + values[t, agent_idx]
        
        for agent_idx in range(n_agents):
            adv = advantages[:, agent_idx]
            advantages[:, agent_idx] = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        return {
            'obs': obs, 'actions': actions, 'log_probs': log_probs,
            'action_probs': action_probs, 'advantages': advantages,
            'returns': returns, 'values': values,
        }


# ============================================================
# H3C TRAINER 
# ============================================================

class H3CTrainer:
    
    
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 1e-3)
        self.lr_min_ratio = config.get('lr_min_ratio', 0.2)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.clip_ratio = config.get('clip_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.rollout_length = config.get('rollout_length', 256)
        self.total_steps = config.get('n_steps', 250000)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.sil_coef = config.get('sil_coef', 0.1)
        self.kl_elite_coef = config.get('kl_elite_coef', 0.05)
        self.temperature = config.get('temp_init', 1.0)
        
        hidden_dim = config.get('hidden_dim', 128)
        belief_dim = config.get('belief_dim', 64)
        goal_dim = config.get('goal_dim', 64)
        message_dim = config.get('message_dim', 32)
        n_coalitions = config.get('n_coalitions', 2)
        
        # === CORE H3C MODULES ===
        self.aah = AAH(
            obs_dim=obs_dim, n_agents=n_agents, hidden_dim=hidden_dim,
            num_heads=4, num_layers=2, n_coalitions=n_coalitions, belief_dim=belief_dim
        ).to(self.device)
        
        self.pcc = PCC(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            hidden_dim=hidden_dim, belief_dim=belief_dim, message_dim=message_dim,
            min_entropy_ratio=0.0
        ).to(self.device)
        
        self.cdega = CDEGA(
            obs_dim=obs_dim, n_agents=n_agents, goal_dim=goal_dim,
            hidden_dim=hidden_dim, max_loss_weight=0.005
        ).to(self.device)
        
        # === NEW MODULES ===
        self.dgat = DynamicGraphAttention(obs_dim, hidden_dim, n_heads=4).to(self.device)
        # message_dim=hidden_dim because DGAT outputs hidden_dim
        self.bayesian_fusion = BayesianBeliefFusion(obs_dim, belief_dim, hidden_dim, message_dim=hidden_dim).to(self.device)
        self.adaptive_coalitions = AdaptiveCoalitionFormation(n_agents, hidden_dim, max_coalitions=4).to(self.device)
        self.dual_critic = DualCritic(obs_dim, n_agents, hidden_dim * 2).to(self.device)
        
        # === ELITE TRACKING ===
        self.elite_buffer = EliteTrajectoryBuffer(capacity=30)
        self.elite_policy = ElitePolicyTracker(ema_decay=0.995)
        
        # === ENTROPY MANAGEMENT ===
        self.entropy_target_init = 1.6
        self.entropy_target_final = 0.3
        self.entropy_coef = 0.01
        
        config['action_dim'] = action_dim
        self.auto_hp = AUTO_HP(config)
        
        # === OPTIMIZERS ===
        self.actor_optimizer = optim.Adam([
            {'params': self.aah.parameters(), 'lr': self.lr_actor},
            {'params': self.pcc.parameters(), 'lr': self.lr_actor},
            {'params': self.cdega.parameters(), 'lr': self.lr_actor * 0.5},
            {'params': self.dgat.parameters(), 'lr': self.lr_actor},
            {'params': self.bayesian_fusion.parameters(), 'lr': self.lr_actor},
        ])
        
        self.critic_optimizer = optim.Adam([
            {'params': self.dual_critic.parameters(), 'lr': self.lr_critic},
            {'params': self.adaptive_coalitions.parameters(), 'lr': self.lr_critic * 0.5},
        ])
        
        self.buffer = DGATRolloutBuffer(self.gamma, self.gae_lambda)
        
        self.step_count = 0
        self.update_count = 0
        self.best_eval_reward = float('-inf')
        self.best_state = None
    
    def get_entropy_target(self) -> float:
        progress = min(self.step_count / self.total_steps, 1.0)
        return self.entropy_target_init * (self.entropy_target_final / self.entropy_target_init) ** progress
    
    def get_entropy_loss_weight(self, current_entropy: float) -> float:
        """
        v5.2: DIRECT entropy coefficient adjustment.
        
        v5.1 Problem: Gap of +0.94 despite penalties = loss weights not effective
        v5.2 Fix: Directly adjust entropy_coef AND use stronger loss weights
        
        Also updates self.entropy_coef for direct control.
        """
        target = self.get_entropy_target()
        gap = current_entropy - target
        
        # DIRECT coefficient adjustment (in addition to loss weight)
        if not hasattr(self, 'entropy_coef_base'):
            self.entropy_coef_base = self.entropy_coef
        
        # Adjust entropy_coef directly based on gap
        if gap > 0.5:
            self.entropy_coef = max(self.entropy_coef * 0.9, 0.001)  # Reduce coefficient
        elif gap > 0.3:
            self.entropy_coef = max(self.entropy_coef * 0.95, 0.001)
        elif gap < -0.3:
            self.entropy_coef = min(self.entropy_coef * 1.1, 0.05)  # Increase coefficient
        elif gap < -0.1:
            self.entropy_coef = min(self.entropy_coef * 1.05, 0.05)
        
        # Hard bounds on entropy itself
        if current_entropy > 2.0:
            return -0.5  # EXTREME penalty - entropy way too high
        elif current_entropy < 0.3:
            return 0.3  # STRONG encouragement - entropy collapsed
        
        # Normal graduated control
        if gap > 0.6:
            return -0.35  # EXTREME penalty
        elif gap > 0.4:
            return -0.25
        elif gap > 0.25:
            return -0.15
        elif gap > 0.15:
            return -0.08
        elif gap > 0.05:
            return -0.03
        elif gap < -0.2:
            return 0.15
        elif gap < -0.1:
            return 0.08
        elif gap < -0.05:
            return 0.05
        else:
            return 0.01  # On target
    
    def get_entropy_info(self) -> dict:
        """Return current entropy state for logging."""
        return {
            'entropy_coef': getattr(self, 'entropy_coef', 0.01),
            'entropy_target': self.get_entropy_target(),
        }
    
    def get_adaptive_clip_epsilon(self) -> float:
        """
        Conservative updates when close to best performance.
        
        Reduces clip epsilon to prevent destructive updates near optimum.
        """
        if not hasattr(self, 'best_eval_reward'):
            return self.clip_ratio
        
        if not hasattr(self, 'recent_eval_reward'):
            return self.clip_ratio
        
        gap = self.best_eval_reward - self.recent_eval_reward
        
        if gap < 2:
            return 0.1  # Very conservative
        elif gap < 4:
            return 0.15  # Conservative
        elif gap < 7:
            return 0.18
        else:
            return self.clip_ratio  # Standard (0.2)
    
    def should_recover_checkpoint(self) -> bool:
        """
         Check if we should recover from best checkpoint.
        
        Triggers if performance drops significantly below best.
        """
        if not hasattr(self, 'best_eval_reward') or not hasattr(self, 'recent_eval_reward'):
            return False
        
        if not hasattr(self, 'best_checkpoint'):
            return False
        
        gap = self.best_eval_reward - self.recent_eval_reward
        return gap > 6  # Recover if dropped more than 6 points
    
    def save_checkpoint(self, reward: float):
        """Save checkpoint if this is best performance."""
        if not hasattr(self, 'best_eval_reward') or reward > self.best_eval_reward:
            self.best_eval_reward = reward
            self.best_checkpoint = {
                'aah': copy.deepcopy(self.aah.state_dict()),
                'pcc': copy.deepcopy(self.pcc.state_dict()),
                'cdega': copy.deepcopy(self.cdega.state_dict()),
                'dgat': copy.deepcopy(self.dgat.state_dict()),
                'bayesian_fusion': copy.deepcopy(self.bayesian_fusion.state_dict()),
                'dual_critic': copy.deepcopy(self.dual_critic.state_dict()),
            }
            self.steps_since_best = 0
            return True
        else:
            self.steps_since_best = getattr(self, 'steps_since_best', 0) + 1
            return False
    
    def recover_from_checkpoint(self):
        """Recover from best checkpoint with reduced learning rate."""
        if hasattr(self, 'best_checkpoint'):
            self.aah.load_state_dict(self.best_checkpoint['aah'])
            self.pcc.load_state_dict(self.best_checkpoint['pcc'])
            self.cdega.load_state_dict(self.best_checkpoint['cdega'])
            self.dgat.load_state_dict(self.best_checkpoint['dgat'])
            self.bayesian_fusion.load_state_dict(self.best_checkpoint['bayesian_fusion'])
            self.dual_critic.load_state_dict(self.best_checkpoint['dual_critic'])
            
            # Reduce learning rate after recovery
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] *= 0.7
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] *= 0.7
            
            return True
        return False
    
    def reset_episode(self):
        self.pcc.reset_beliefs(1, self.device)
    
    def extract_positions(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract agent positions from observation."""
        # In MPE, positions are typically first 2 dims of obs
        return obs[..., :2]
    
    def get_actions(self, obs: np.ndarray, explore: bool = True):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            positions = self.extract_positions(obs_t)
            
            # DGAT message passing
            dgat_feat, attention = self.dgat(obs_t, positions)
            
            # Bayesian belief fusion
            messages = dgat_feat.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
            beliefs, uncertainty = self.bayesian_fusion(obs_t, messages)
            
            # Update coalitions with step info
            self.adaptive_coalitions(dgat_feat, positions, step=self.step_count)
            
            # IMPORTANT: Reset PCC beliefs with batch_size=1 before forward
            self.pcc.reset_beliefs(1, self.device)
            
            # AAH + PCC forward
            aah_out = self.aah(obs_t, beliefs, self.temperature)
            pcc_out = self.pcc(obs_t, aah_out, self.temperature, deterministic=not explore)
            
            actions = pcc_out['actions'].squeeze(0).cpu().numpy()
            action_probs = pcc_out['action_probs'].squeeze(0).cpu().numpy()
            log_probs = pcc_out['log_probs'].squeeze(0).cpu().numpy()
            
            # Dual critic values
            values, _ = self.dual_critic(obs_t, positions)
            values = values.squeeze(0).cpu().numpy()
        
        return actions, action_probs, log_probs, values
    
    def store_transition(self, obs, actions, rewards, dones, values, log_probs, action_probs):
        self.buffer.add(obs, actions, rewards, dones, values, log_probs, action_probs)
        self.step_count += 1
    
    def should_update(self) -> bool:
        return len(self.buffer.obs) >= self.rollout_length
    
    def end_episode(self, episode_reward: float):
        if len(self.buffer.current_trajectory['obs']) > 0:
            self.buffer._complete_trajectory()
    
    def compute_kl_to_elite(self, new_action_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence to elite policy."""
        elite_params = self.elite_policy.get_elite_params()
        if elite_params is None:
            return torch.tensor(0.0, device=self.device)
        
        # This is a simplified version - in practice, you'd forward through elite policy
        # For now, just use uniform as target (encourages exploration initially)
        uniform = torch.ones_like(new_action_probs) / new_action_probs.shape[-1]
        kl = (new_action_probs * (torch.log(new_action_probs + 1e-10) - torch.log(uniform + 1e-10))).sum(dim=-1)
        return kl.mean()
    
    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        # LR scheduling
        progress = min(self.step_count / self.total_steps, 1.0)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        lr_mult = max(decay, self.lr_min_ratio)
        
        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = self.lr_actor * lr_mult
        for pg in self.critic_optimizer.param_groups:
            pg['lr'] = self.lr_critic * lr_mult
        
        # Add trajectories to elite buffer
        for traj, reward in self.buffer.get_completed_trajectories():
            self.elite_buffer.add_trajectory(traj, reward)
        
        # Get last values
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            positions = self.extract_positions(obs_t)
            last_values, _ = self.dual_critic(obs_t, positions)
            last_values = last_values.squeeze(0).cpu().numpy()
        
        data = self.buffer.compute_gae(last_values)
        T = len(data['obs'])
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        for epoch in range(self.ppo_epochs):
            indices = np.random.permutation(T)
            
            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                idx = indices[start:end]
                batch_size = len(idx)
                
                obs = torch.FloatTensor(data['obs'][idx]).to(self.device)
                actions = torch.LongTensor(data['actions'][idx]).to(self.device)
                old_log_probs = torch.FloatTensor(data['log_probs'][idx]).to(self.device)
                old_action_probs = torch.FloatTensor(data['action_probs'][idx]).to(self.device)
                advantages = torch.FloatTensor(data['advantages'][idx]).to(self.device)
                returns = torch.FloatTensor(data['returns'][idx]).to(self.device)
                old_values = torch.FloatTensor(data['values'][idx]).to(self.device)
                
                positions = self.extract_positions(obs)
                
                # Forward with new modules
                dgat_feat, _ = self.dgat(obs, positions)
                messages = dgat_feat.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
                beliefs, _ = self.bayesian_fusion(obs, messages)
                
                # IMPORTANT: Reset PCC beliefs with correct batch size before forward
                self.pcc.reset_beliefs(batch_size, self.device)
                
                aah_out = self.aah(obs, beliefs, self.temperature)
                pcc_out = self.pcc(obs, aah_out, self.temperature, deterministic=False)
                
                new_action_probs = pcc_out['action_probs']
                new_log_probs_all = pcc_out['log_probs_all']
                entropy = pcc_out['entropy']
                
                new_log_probs = new_log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                
                # PPO loss with ADAPTIVE clip epsilon (v5.1)
                ratio = torch.exp(new_log_probs - old_log_probs)
                adaptive_clip = self.get_adaptive_clip_epsilon()
                policy_losses = []
                
                for agent_idx in range(self.n_agents):
                    agent_ratio = ratio[:, agent_idx]
                    agent_adv = advantages[:, agent_idx]
                    surr1 = agent_ratio * agent_adv
                    surr2 = torch.clamp(agent_ratio, 1 - adaptive_clip, 1 + adaptive_clip) * agent_adv
                    policy_losses.append(-torch.min(surr1, surr2).mean())
                
                policy_loss = sum(policy_losses) / self.n_agents
                
                # Value loss with dual critic
                new_values, critic_info = self.dual_critic(obs, positions)
                
                value_losses = []
                for agent_idx in range(self.n_agents):
                    old_v = old_values[:, agent_idx]
                    new_v = new_values[:, agent_idx]
                    ret = returns[:, agent_idx]
                    v_clipped = old_v + torch.clamp(new_v - old_v, -self.clip_ratio, self.clip_ratio)
                    v_loss = 0.5 * torch.max((new_v - ret)**2, (v_clipped - ret)**2).mean()
                    value_losses.append(v_loss)
                
                value_loss = sum(value_losses) / self.n_agents
                
                # Entropy loss with hard bounds
                current_entropy = entropy.mean().item()
                entropy_weight = self.get_entropy_loss_weight(current_entropy)
                entropy_loss = entropy_weight * entropy.mean()
                
                # KL to elite
                kl_elite = self.compute_kl_to_elite(new_action_probs)
                
                # KL divergence for early stopping
                with torch.no_grad():
                    kl = (old_action_probs * (torch.log(old_action_probs + 1e-10) - 
                          torch.log(new_action_probs + 1e-10))).sum(dim=-1).mean()
                
                if kl.item() > 0.05:
                    continue
                
                # Total losses
                actor_loss = policy_loss - entropy_loss + self.kl_elite_coef * kl_elite
                critic_loss = self.value_loss_coef * value_loss
                
                # Optimize
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    list(self.aah.parameters()) + list(self.pcc.parameters()) + 
                    list(self.cdega.parameters()) + list(self.dgat.parameters()) +
                    list(self.bayesian_fusion.parameters()), 
                    self.max_grad_norm
                )
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.dual_critic.parameters()) + list(self.adaptive_coalitions.parameters()),
                    1.0
                )
                self.critic_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += current_entropy
                total_kl += kl.item()
                n_updates += 1
        
        self.buffer.reset()
        self.update_count += 1
        
        # Track variance for adaptive mechanisms
        if not hasattr(self, 'recent_value_losses'):
            self.recent_value_losses = []
        self.recent_value_losses.append(total_value_loss / max(n_updates, 1))
        if len(self.recent_value_losses) > 20:
            self.recent_value_losses = self.recent_value_losses[-20:]
        
        critic_variance = np.var(self.recent_value_losses) if len(self.recent_value_losses) > 2 else 0
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'critic_variance': critic_variance,
            'entropy': total_entropy / max(n_updates, 1),
            'entropy_target': self.get_entropy_target(),
            'entropy_gap': (total_entropy / max(n_updates, 1)) - self.get_entropy_target(),
            'kl': total_kl / max(n_updates, 1),
            'n_coalitions': self.adaptive_coalitions.n_active_coalitions.item(),
            'coalition_exploring': self.adaptive_coalitions.exploration_phase,
            'elite_size': len(self.elite_buffer),
            'elite_threshold': self.elite_buffer.get_threshold(),
            'lr': self.lr_actor * lr_mult,
            'clip_epsilon': self.get_adaptive_clip_epsilon(),
            'steps_since_best': getattr(self, 'steps_since_best', 0),
        }
    
    def check_recovery(self, eval_reward: float) -> bool:
        """
        v5.2: SMART recovery with patience, cooldown, and LR protection.
        
        Key improvements:
        - Require 3 consecutive bad evals (patience)
        - 20K step cooldown between recoveries
        - Max 5 recoveries total
        - LR floor at 30% of original
        - Use rolling average for decisions
        """
        # Initialize tracking variables if needed
        if not hasattr(self, 'eval_history'):
            self.eval_history = []
            self.consecutive_bad_evals = 0
            self.total_recoveries = 0
            self.last_recovery_step = 0
            self.lr_floor = self.lr_actor * 0.3  # 30% floor
            self.stable_best_reward = float('-inf')
            self.stable_best_count = 0
        
        self.recent_eval_reward = eval_reward
        self.eval_history.append(eval_reward)
        if len(self.eval_history) > 10:
            self.eval_history = self.eval_history[-10:]
        
        # Record reward for coalition adaptation
        self.adaptive_coalitions.record_reward(eval_reward)
        
        # Calculate rolling average (last 3 evals)
        rolling_avg = np.mean(self.eval_history[-3:]) if len(self.eval_history) >= 3 else eval_reward
        
        # Update best tracking (require stability)
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.stable_best_count = 1
            self._save_best_state()
            self.consecutive_bad_evals = 0
            self.steps_since_best = 0
            return False
        elif eval_reward > self.best_eval_reward - 2:  # Within 2 points of best
            self.stable_best_count += 1
            if self.stable_best_count >= 2:
                # Confirmed stable good performance
                self.stable_best_reward = max(self.stable_best_reward, rolling_avg)
            self.consecutive_bad_evals = 0
            self.steps_since_best = 0
            return False
        
        # Track consecutive bad evaluations
        self.steps_since_best = getattr(self, 'steps_since_best', 0) + 1
        gap = self.best_eval_reward - eval_reward
        
        if gap > 8:  # Significant drop
            self.consecutive_bad_evals += 1
        else:
            self.consecutive_bad_evals = max(0, self.consecutive_bad_evals - 1)
        
        # Check recovery conditions
        should_recover = (
            self.consecutive_bad_evals >= 3 and  # Patience: 3 consecutive bad
            self.total_recoveries < 5 and  # Max 5 recoveries
            self.step_count > 50000 and  # After warmup
            self.step_count - self.last_recovery_step > 20000 and  # 20K cooldown
            hasattr(self, 'best_state')
        )
        
        if should_recover:
            print(f"\n⚠️  {self.consecutive_bad_evals} consecutive drops (gap: {gap:.1f}), recovery #{self.total_recoveries + 1}")
            
            # Load best state
            self.aah.load_state_dict(self.best_state['aah'])
            self.pcc.load_state_dict(self.best_state['pcc'])
            self.cdega.load_state_dict(self.best_state['cdega'])
            self.dgat.load_state_dict(self.best_state['dgat'])
            self.bayesian_fusion.load_state_dict(self.best_state['bayesian_fusion'])
            self.dual_critic.load_state_dict(self.best_state['dual_critic'])
            if 'adaptive_coalitions' in self.best_state:
                self.adaptive_coalitions.load_state_dict(self.best_state['adaptive_coalitions'])
            
            # Only reduce LR after 3+ recoveries, with floor protection
            if self.total_recoveries >= 2:
                for param_group in self.actor_optimizer.param_groups:
                    new_lr = max(param_group['lr'] * 0.85, self.lr_floor)
                    param_group['lr'] = new_lr
                for param_group in self.critic_optimizer.param_groups:
                    new_lr = max(param_group['lr'] * 0.85, self.lr_floor * 3)  # critic LR floor
                    param_group['lr'] = new_lr
            
            self.total_recoveries += 1
            self.last_recovery_step = self.step_count
            self.consecutive_bad_evals = 0
            
            current_lr = self.actor_optimizer.param_groups[0]['lr']
            print(f"✓ Recovered! LR: {current_lr:.6f} (floor: {self.lr_floor:.6f})")
            return True
        
        return False
    
    def _save_best_state(self):
        """Save current state as best."""
        self.best_state = {
            'aah': copy.deepcopy(self.aah.state_dict()),
            'pcc': copy.deepcopy(self.pcc.state_dict()),
            'cdega': copy.deepcopy(self.cdega.state_dict()),
            'dgat': copy.deepcopy(self.dgat.state_dict()),
            'bayesian_fusion': copy.deepcopy(self.bayesian_fusion.state_dict()),
            'dual_critic': copy.deepcopy(self.dual_critic.state_dict()),
            'adaptive_coalitions': copy.deepcopy(self.adaptive_coalitions.state_dict()),
        }
        self.best_checkpoint = self.best_state
        
        # Update elite policy
        policy_params = {**self.aah.state_dict(), **self.pcc.state_dict()}
        self.elite_policy.update(policy_params, self.best_eval_reward)
    
    def save(self, path: str):
        torch.save({
            'aah': self.aah.state_dict(),
            'pcc': self.pcc.state_dict(),
            'cdega': self.cdega.state_dict(),
            'dgat': self.dgat.state_dict(),
            'bayesian_fusion': self.bayesian_fusion.state_dict(),
            'dual_critic': self.dual_critic.state_dict(),
            'adaptive_coalitions': self.adaptive_coalitions.state_dict(),
            'step_count': self.step_count,
            'best_eval_reward': self.best_eval_reward,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.aah.load_state_dict(ckpt['aah'])
        self.pcc.load_state_dict(ckpt['pcc'])
        self.cdega.load_state_dict(ckpt['cdega'])
        self.dgat.load_state_dict(ckpt['dgat'])
        self.bayesian_fusion.load_state_dict(ckpt['bayesian_fusion'])
        self.dual_critic.load_state_dict(ckpt['dual_critic'])
        self.step_count = ckpt['step_count']
        self.best_eval_reward = ckpt.get('best_eval_reward', float('-inf'))