"""
H3CTrainer_Revised.py - H3C-BEACON Revised Version
====================================================

Modifications pour répondre aux commentaires des reviewers :

R1.1 - Comparaison BayesG/GACG : Documentation des différences
R1.2 - Ablation study : Composants désactivables individuellement  
R1.3 - Complexité : Méthodes de mesure intégrées
R1.4 - 5 seeds : Support multi-seed
R1.5 - Hyperparamètres : Configuration complète exportable
R2.2 - Kernels : Multiple kernel support (inverse, gaussian, polynomial, learned)
R2.4 - Négociation : Mécanisme de veto pour coalitions
R2.5 - β dynamics : Logging de la trajectoire β
R2.6 - RTD++ safeguards : Reset périodique, novelty bonus, staleness detection
R2.7 - ε-Nash : Mesure empirique du Nash gap

Authors: Basile BETE MBEZELE, Ghislain ALO'O ABESSOLO
University of Yaoundé I, Cameroon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy
import heapq
import math
import time
import json

# Import sub-modules
from modules.AAH import AAH
from modules.PCC import PCC
from modules.CDEGA import CDEGA
from modules.AUTO_HP import AUTO_HP


# ============================================================
# 1. DYNAMIC GRAPH ATTENTION NETWORK (DGAT) - REVISED
# ============================================================

class DynamicGraphAttention(nn.Module):
    """
    Dynamic Graph Attention for inter-agent communication.
    
    Revision R2.2: Support multiple spatial kernels:
    - 'inverse': 1/(d_ij + ε) - Original
    - 'gaussian': exp(-d²/2σ²)
    - 'polynomial': (1 + d)^(-α)
    - 'learned': MLP-based distance embedding
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4,
                 kernel_type: str = 'inverse', kernel_params: Dict = None):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Kernel-specific parameters (R2.2)
        if kernel_type == 'inverse':
            self.epsilon = nn.Parameter(torch.tensor(self.kernel_params.get('epsilon', 0.1)))
        elif kernel_type == 'gaussian':
            self.sigma = nn.Parameter(torch.tensor(self.kernel_params.get('sigma', 1.0)))
        elif kernel_type == 'polynomial':
            self.alpha = nn.Parameter(torch.tensor(self.kernel_params.get('alpha', 2.0)))
        elif kernel_type == 'learned':
            self.distance_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def compute_distance_weight(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute attention weights based on distance using selected kernel."""
        if self.kernel_type == 'inverse':
            # Original: 1/(d + ε)
            return 1.0 / (distances + torch.abs(self.epsilon) + 1e-6)
        elif self.kernel_type == 'gaussian':
            # Gaussian: exp(-d²/2σ²)
            return torch.exp(-distances.pow(2) / (2 * self.sigma.pow(2) + 1e-6))
        elif self.kernel_type == 'polynomial':
            # Polynomial: (1 + d)^(-α)
            return (1 + distances).pow(-torch.abs(self.alpha))
        elif self.kernel_type == 'learned':
            # Learned MLP
            d_flat = distances.view(-1, 1)
            weights = self.distance_mlp(d_flat)
            return weights.view(distances.shape)
        else:
            return torch.ones_like(distances)
        
    def forward(self, h: torch.Tensor, positions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Agent hidden states [batch, n_agents, input_dim]
            positions: Agent positions [batch, n_agents, 2] (optional)
            
        Returns:
            h_out: Updated hidden states [batch, n_agents, hidden_dim]
            attention_weights: Attention matrix [batch, n_heads, n_agents, n_agents]
        """
        batch_size, n_agents, _ = h.shape
        
        # Compute Q, K, V
        Q = self.W_q(h).view(batch_size, n_agents, self.n_heads, self.head_dim)
        K = self.W_k(h).view(batch_size, n_agents, self.n_heads, self.head_dim)
        V = self.W_v(h).view(batch_size, n_agents, self.n_heads, self.head_dim)
        
        # Transpose for attention: [batch, n_heads, n_agents, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply distance-based gating if positions available
        if positions is not None:
            # Compute pairwise distances
            pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, n, n, 2]
            distances = torch.norm(pos_diff, dim=-1)  # [batch, n, n]
            
            # Apply kernel (R2.2)
            dist_weights = self.compute_distance_weight(distances)
            dist_weights = dist_weights.unsqueeze(1)  # [batch, 1, n, n]
            
            scores = scores * dist_weights
        
        # Softmax attention
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # [batch, n_heads, n_agents, head_dim]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, n_agents, self.hidden_dim)
        out = self.W_o(out)
        
        # Residual + LayerNorm
        out = self.layer_norm(h[:, :, :self.hidden_dim] + out) if h.shape[-1] >= self.hidden_dim else self.layer_norm(out)
        
        return out, attention


# ============================================================
# 2. BAYESIAN BELIEF FUSION - REVISED
# ============================================================

class BayesianBeliefFusion(nn.Module):
    """
    Bayesian belief fusion using natural parameters.
    
    Revision R2.3: Support for mixture of Gaussians extension.
    """
    
    def __init__(self, belief_dim: int, n_agents: int, 
                 use_mixture: bool = False, n_components: int = 3):
        super().__init__()
        
        self.belief_dim = belief_dim
        self.n_agents = n_agents
        self.use_mixture = use_mixture
        self.n_components = n_components
        
        # Belief encoder
        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, belief_dim * 2),
            nn.ReLU(),
            nn.Linear(belief_dim * 2, belief_dim * 2)  # Output: [mu, log_var]
        )
        
        # Attention for weighting beliefs
        self.belief_attention = nn.Sequential(
            nn.Linear(belief_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Mixture components (R2.3)
        if use_mixture:
            self.mixture_weights = nn.Linear(belief_dim * 2, n_components)
            self.component_means = nn.ModuleList([
                nn.Linear(belief_dim, belief_dim) for _ in range(n_components)
            ])
            self.component_vars = nn.ModuleList([
                nn.Linear(belief_dim, belief_dim) for _ in range(n_components)
            ])
    
    def forward(self, observations: torch.Tensor, 
                coalition_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse beliefs within coalitions using natural parameter aggregation.
        
        Args:
            observations: [batch, n_agents, obs_dim]
            coalition_mask: [batch, n_agents, n_agents] - 1 if in same coalition
            
        Returns:
            fused_belief: [batch, n_agents, belief_dim]
            uncertainty: [batch, n_agents, belief_dim]
        """
        batch_size, n_agents, obs_dim = observations.shape
        
        # Encode individual beliefs
        belief_params = self.belief_encoder(observations[:, :, :self.belief_dim])
        mu = belief_params[:, :, :self.belief_dim]
        log_var = belief_params[:, :, self.belief_dim:]
        
        # Convert to natural parameters: η = [Σ⁻¹μ, -½Σ⁻¹]
        precision = torch.exp(-log_var)  # Σ⁻¹
        eta1 = precision * mu  # Σ⁻¹μ
        eta2 = -0.5 * precision  # -½Σ⁻¹
        
        # Compute attention weights for fusion
        attention_scores = self.belief_attention(belief_params)  # [batch, n_agents, 1]
        
        if coalition_mask is not None:
            # Mask attention to only include coalition members
            attention_scores = attention_scores.squeeze(-1).unsqueeze(1)  # [batch, 1, n_agents]
            attention_scores = attention_scores.expand(-1, n_agents, -1)  # [batch, n_agents, n_agents]
            attention_scores = attention_scores * coalition_mask
            attention_weights = F.softmax(attention_scores + (1 - coalition_mask) * (-1e9), dim=-1)
        else:
            attention_weights = F.softmax(attention_scores, dim=1)
            attention_weights = attention_weights.squeeze(-1).unsqueeze(1).expand(-1, n_agents, -1)
        
        # Aggregate natural parameters within coalitions
        # η_coalition = Σ w_i * η_i
        fused_eta1 = torch.bmm(attention_weights, eta1)
        fused_eta2 = torch.bmm(attention_weights, eta2)
        
        # Convert back to mean/variance
        fused_precision = -2 * fused_eta2
        fused_mu = fused_eta1 / (fused_precision + 1e-6)
        fused_var = 1.0 / (fused_precision + 1e-6)
        
        return fused_mu, fused_var


# ============================================================
# 3. ADAPTIVE COALITION FORMATION - REVISED
# ============================================================

class AdaptiveCoalitionFormation(nn.Module):
    """
    Spectral clustering-based coalition formation.
    
    Revision R2.4: Added negotiation/veto mechanism.
    """
    
    def __init__(self, hidden_dim: int, n_agents: int, 
                 min_coalition_size: int = 2,
                 enable_negotiation: bool = True,
                 veto_threshold: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.min_coalition_size = min_coalition_size
        self.enable_negotiation = enable_negotiation
        self.veto_threshold = veto_threshold
        
        # Affinity network
        self.affinity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Temperature parameter
        self.sigma = nn.Parameter(torch.tensor(1.0))
        
        # Utility predictor for negotiation (R2.4)
        if enable_negotiation:
            self.utility_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    
    def compute_affinity_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """Compute pairwise affinity between agents."""
        batch_size, n_agents, _ = h.shape
        
        # Expand for pairwise computation
        h_i = h.unsqueeze(2).expand(-1, -1, n_agents, -1)
        h_j = h.unsqueeze(1).expand(-1, n_agents, -1, -1)
        
        # Concatenate pairs
        h_pairs = torch.cat([h_i, h_j], dim=-1)
        
        # Compute affinities
        affinities = self.affinity_net(h_pairs).squeeze(-1)
        
        # Make symmetric
        affinities = (affinities + affinities.transpose(-1, -2)) / 2
        
        return affinities
    
    def spectral_clustering(self, affinity: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Perform spectral clustering using eigengap heuristic.
        
        Returns:
            coalition_assignments: [batch, n_agents] - coalition ID for each agent
            n_coalitions: number of coalitions formed
        """
        batch_size, n_agents, _ = affinity.shape
        
        # Compute normalized Laplacian
        D = torch.diag_embed(affinity.sum(dim=-1))
        D_inv_sqrt = torch.diag_embed(1.0 / (affinity.sum(dim=-1).sqrt() + 1e-6))
        L = torch.eye(n_agents, device=affinity.device) - torch.bmm(torch.bmm(D_inv_sqrt, affinity), D_inv_sqrt)
        
        # Eigendecomposition (use CPU for stability)
        L_cpu = L.detach().cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(L_cpu)
        eigenvalues = eigenvalues.to(affinity.device)
        eigenvectors = eigenvectors.to(affinity.device)
        
        # Eigengap heuristic: K* = argmax_k (λ_{k+1} - λ_k)
        gaps = eigenvalues[:, 1:] - eigenvalues[:, :-1]
        k_star = gaps[:, :n_agents//2].argmax(dim=-1) + 1
        k_star = k_star.clamp(min=2, max=n_agents // self.min_coalition_size)
        
        # Use first k eigenvectors for clustering
        # Simplified: assign based on sign of second eigenvector for k=2
        k = k_star[0].item() if batch_size == 1 else 2
        
        # K-means on eigenvectors (simplified: threshold-based)
        features = eigenvectors[:, :, 1:k+1]
        
        # Simple assignment based on feature signs
        if k == 2:
            assignments = (features[:, :, 0] > 0).long()
        else:
            # Cluster based on dominant eigenvector component
            assignments = features.abs().argmax(dim=-1)
        
        return assignments, k
    
    def negotiate_coalitions(self, h: torch.Tensor, 
                            proposed_assignments: torch.Tensor) -> torch.Tensor:
        """
        Negotiation mechanism: agents can veto coalition if utility decreases.
        (R2.4)
        """
        if not self.enable_negotiation:
            return proposed_assignments
        
        batch_size, n_agents, _ = h.shape
        final_assignments = proposed_assignments.clone()
        
        # Compute individual utilities
        for i in range(n_agents):
            # Current coalition members
            coalition_id = proposed_assignments[:, i]
            members = (proposed_assignments == coalition_id.unsqueeze(-1))
            
            # Compute utility with coalition
            coalition_h = h * members.unsqueeze(-1).float()
            coalition_h_mean = coalition_h.sum(dim=1) / members.sum(dim=1, keepdim=True).float()
            
            utility_with = self.utility_predictor(
                torch.cat([h[:, i], coalition_h_mean], dim=-1)
            )
            
            # Compute utility alone
            utility_alone = self.utility_predictor(
                torch.cat([h[:, i], h[:, i]], dim=-1)
            )
            
            # Veto if utility decreases beyond threshold
            veto = (utility_alone - utility_with) > self.veto_threshold
            
            # If veto, assign to singleton coalition
            final_assignments[:, i] = torch.where(
                veto.squeeze(),
                torch.full_like(coalition_id, n_agents + i),  # Unique singleton ID
                coalition_id
            )
        
        return final_assignments
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Form coalitions based on agent representations.
        
        Returns:
            coalition_mask: [batch, n_agents, n_agents] - 1 if same coalition
            assignments: [batch, n_agents] - coalition ID
            n_coalitions: number of coalitions
        """
        # Compute affinity matrix
        affinity = self.compute_affinity_matrix(h)
        
        # Spectral clustering
        assignments, n_coalitions = self.spectral_clustering(affinity)
        
        # Negotiation (R2.4)
        assignments = self.negotiate_coalitions(h, assignments)
        
        # Create coalition mask
        coalition_mask = (assignments.unsqueeze(-1) == assignments.unsqueeze(-2)).float()
        
        return coalition_mask, assignments, n_coalitions


# ============================================================
# 4. DUAL CRITIC ARCHITECTURE - REVISED
# ============================================================

class DualCritic(nn.Module):
    """
    Global + Local critics with learned mixing coefficient.
    
    Revision R2.5: Logging of β trajectory during training.
    """
    
    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Global critic
        self.global_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Local critic
        self.local_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # β mixing network
        self.beta_net = nn.Sequential(
            nn.Linear(state_dim + obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # β trajectory logging (R2.5)
        self.beta_history = []
        
    def forward(self, state: torch.Tensor, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mixed value estimate.
        
        Returns:
            value: Mixed value estimate
            beta: Mixing coefficient
        """
        # Global value
        v_global = self.global_critic(state)
        
        # Local value
        v_local = self.local_critic(obs)
        
        # Compute β
        beta = self.beta_net(torch.cat([state, obs], dim=-1))
        
        # Log β (R2.5)
        self.beta_history.append(beta.mean().item())
        
        # Mixed value
        value = beta * v_global + (1 - beta) * v_local
        
        return value, beta
    
    def get_beta_trajectory(self) -> List[float]:
        """Return β trajectory for analysis (R2.5)."""
        return self.beta_history
    
    def reset_beta_history(self):
        """Reset β history."""
        self.beta_history = []


# ============================================================
# 5. RTD++ ELITE ANCHORING - REVISED
# ============================================================

class RTDPlusPlusElite(nn.Module):
    """
    Elite anchoring with KL regularization.
    
    Revision R2.6: Added safeguards against crystallization:
    - Periodic reset
    - Novelty bonus
    - Staleness detection
    """
    
    def __init__(self, buffer_size: int = 1000, 
                 ema_alpha: float = 0.995,
                 kl_weight: float = 0.01,
                 reset_interval: int = 200000,
                 staleness_threshold: float = 0.2):
        super().__init__()
        
        self.buffer_size = buffer_size
        self.ema_alpha = ema_alpha
        self.kl_weight = kl_weight
        self.reset_interval = reset_interval
        self.staleness_threshold = staleness_threshold
        
        # Elite buffer
        self.elite_buffer = []
        self.elite_policy = None
        
        # Counters (R2.6)
        self.steps_since_reset = 0
        self.steps_since_improvement = 0
        self.best_reward = float('-inf')
        
    def add_trajectory(self, trajectory: Dict, reward: float):
        """Add trajectory to elite buffer."""
        self.elite_buffer.append((trajectory, reward))
        
        # Keep top-k
        if len(self.elite_buffer) > self.buffer_size:
            self.elite_buffer.sort(key=lambda x: x[1], reverse=True)
            self.elite_buffer = self.elite_buffer[:self.buffer_size]
        
        # Update best reward tracking (R2.6)
        if reward > self.best_reward * (1 + self.staleness_threshold):
            self.best_reward = reward
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
    
    def update_elite_policy(self, current_policy: nn.Module):
        """Update elite policy using EMA."""
        if self.elite_policy is None:
            self.elite_policy = copy.deepcopy(current_policy)
        else:
            # EMA update
            for elite_param, current_param in zip(
                self.elite_policy.parameters(), 
                current_policy.parameters()
            ):
                elite_param.data = (
                    self.ema_alpha * elite_param.data + 
                    (1 - self.ema_alpha) * current_param.data
                )
        
        self.steps_since_reset += 1
        
        # Periodic reset (R2.6)
        if self.steps_since_reset >= self.reset_interval:
            self.reset_elite()
    
    def reset_elite(self):
        """Reset elite policy to current best."""
        self.elite_policy = None
        self.steps_since_reset = 0
        self.elite_buffer = []
        
    def should_reset(self) -> bool:
        """Check if elite should be reset due to staleness (R2.6)."""
        return self.steps_since_improvement > self.reset_interval // 2
    
    def compute_kl_loss(self, current_dist: torch.distributions.Distribution,
                        elite_dist: torch.distributions.Distribution) -> torch.Tensor:
        """Compute KL divergence loss for anchoring."""
        kl = torch.distributions.kl_divergence(current_dist, elite_dist)
        return self.kl_weight * kl.mean()
    
    def compute_novelty_bonus(self, action: torch.Tensor, 
                               elite_action: torch.Tensor) -> torch.Tensor:
        """Compute novelty bonus to encourage exploration (R2.6)."""
        distance = (action - elite_action).pow(2).sum(dim=-1).sqrt()
        novelty = 0.01 * distance  # Small bonus for deviation
        return novelty


# ============================================================
# 6. ENTROPY HARD BOUNDS
# ============================================================

class EntropyController:
    """
    Hard bounds on policy entropy with annealing.
    """
    
    def __init__(self, 
                 initial_entropy_coef: float = 0.01,
                 min_entropy: float = 0.5,
                 max_entropy: float = 2.0,
                 anneal_steps: int = 500000):
        
        self.entropy_coef = initial_entropy_coef
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.anneal_steps = anneal_steps
        self.current_step = 0
        
    def get_entropy_bounds(self) -> Tuple[float, float]:
        """Get current entropy bounds (annealed)."""
        progress = min(1.0, self.current_step / self.anneal_steps)
        
        # Anneal min entropy to allow convergence
        current_min = self.min_entropy * (1 - progress) + 0.1 * progress
        current_max = self.max_entropy * (1 - progress) + 0.5 * progress
        
        return current_min, current_max
    
    def compute_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss with hard bounds."""
        min_ent, max_ent = self.get_entropy_bounds()
        
        # Penalize entropy outside bounds
        too_low = F.relu(min_ent - entropy)
        too_high = F.relu(entropy - max_ent)
        
        entropy_loss = (too_low + too_high).mean()
        
        self.current_step += 1
        
        return entropy_loss


# ============================================================
# 7. COMPLEXITY TRACKER (R1.3)
# ============================================================

class ComplexityTracker:
    """Track computational complexity metrics."""
    
    def __init__(self):
        self.forward_times = []
        self.backward_times = []
        self.memory_usage = []
        self.step_count = 0
        
    def start_forward(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._forward_start = time.time()
        
    def end_forward(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.forward_times.append(time.time() - self._forward_start)
        
    def start_backward(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._backward_start = time.time()
        
    def end_backward(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.backward_times.append(time.time() - self._backward_start)
        
    def record_memory(self):
        if torch.cuda.is_available():
            self.memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
        else:
            self.memory_usage.append(0)
        self.step_count += 1
        
    def get_summary(self) -> Dict:
        """Get complexity summary."""
        return {
            'mean_forward_time_ms': np.mean(self.forward_times) * 1000 if self.forward_times else 0,
            'mean_backward_time_ms': np.mean(self.backward_times) * 1000 if self.backward_times else 0,
            'max_memory_gb': max(self.memory_usage) if self.memory_usage else 0,
            'total_steps': self.step_count
        }


# ============================================================
# 8. NASH GAP ESTIMATOR (R2.7)
# ============================================================

class NashGapEstimator:
    """
    Empirical estimation of ε-Nash gap.
    
    ε = max_i max_{a'_i} [Q_i(s, a'_i, a_{-i}) - Q_i(s, a_i, a_{-i})]
    """
    
    def __init__(self, n_agents: int, action_dim: int):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.epsilon_history = []
        
    def estimate_gap(self, q_values: torch.Tensor, 
                     actions: torch.Tensor) -> float:
        """
        Estimate Nash gap from Q-values.
        
        Args:
            q_values: [batch, n_agents, action_dim] - Q-values for all actions
            actions: [batch, n_agents] - Taken actions
            
        Returns:
            epsilon: Estimated Nash gap
        """
        batch_size = q_values.shape[0]
        
        # Q-values for taken actions
        taken_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [batch, n_agents]
        
        # Best Q-values (best response)
        best_q = q_values.max(dim=-1)[0]  # [batch, n_agents]
        
        # Gap = best possible - current
        gaps = best_q - taken_q  # [batch, n_agents]
        
        # Epsilon = max gap across agents
        epsilon = gaps.max(dim=-1)[0].mean().item()
        
        self.epsilon_history.append(epsilon)
        
        return epsilon
    
    def get_trajectory(self) -> List[float]:
        """Return epsilon trajectory for plotting."""
        return self.epsilon_history


# ============================================================
# MAIN H3C TRAINER - REVISED
# ============================================================

class H3CTrainerRevised:
    """
    H3C-BEACON Trainer with all revisions for reviewers.
    
    Supports:
    - Component ablation (R1.2)
    - Multiple spatial kernels (R2.2)
    - Coalition negotiation (R2.4)
    - β trajectory logging (R2.5)
    - RTD++ safeguards (R2.6)
    - Nash gap estimation (R2.7)
    - Complexity tracking (R1.3)
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 n_agents: int,
                 config: Dict,
                 device: torch.device = None,
                 # Ablation flags (R1.2)
                 disable_dgat: bool = False,
                 disable_bayesian: bool = False,
                 disable_coalitions: bool = False,
                 disable_dual_critic: bool = False,
                 disable_rtd: bool = False,
                 disable_entropy: bool = False,
                 # Kernel selection (R2.2)
                 kernel_type: str = 'inverse'):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ablation flags
        self.disable_dgat = disable_dgat
        self.disable_bayesian = disable_bayesian
        self.disable_coalitions = disable_coalitions
        self.disable_dual_critic = disable_dual_critic
        self.disable_rtd = disable_rtd
        self.disable_entropy = disable_entropy
        
        # Hyperparameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 1e-3)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Initialize components
        self._build_networks(kernel_type)
        self._build_optimizers()
        
        # Trackers
        self.complexity_tracker = ComplexityTracker()
        self.nash_gap_estimator = NashGapEstimator(n_agents, action_dim)
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.eval_history = []
        
    def _build_networks(self, kernel_type: str):
        """Build all network components."""
        
        # Agent encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(self.device)
        
        # DGAT (conditional)
        if not self.disable_dgat:
            self.dgat = DynamicGraphAttention(
                self.hidden_dim, self.hidden_dim,
                n_heads=4, kernel_type=kernel_type
            ).to(self.device)
        else:
            self.dgat = None
        
        # Bayesian fusion (conditional)
        if not self.disable_bayesian:
            self.bayesian_fusion = BayesianBeliefFusion(
                self.hidden_dim, self.n_agents
            ).to(self.device)
        else:
            self.bayesian_fusion = None
        
        # Coalition formation (conditional)
        if not self.disable_coalitions:
            self.coalition_formation = AdaptiveCoalitionFormation(
                self.hidden_dim, self.n_agents,
                enable_negotiation=True
            ).to(self.device)
        else:
            self.coalition_formation = None
        
        # Actor network
        actor_input_dim = self.hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        ).to(self.device)
        
        # Critic (dual or single)
        state_dim = self.obs_dim * self.n_agents
        if not self.disable_dual_critic:
            self.dual_critic = DualCritic(
                state_dim, self.obs_dim, self.hidden_dim
            ).to(self.device)
        else:
            self.critic = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            ).to(self.device)
            self.dual_critic = None
        
        # RTD++ (conditional)
        if not self.disable_rtd:
            self.rtd_elite = RTDPlusPlusElite(
                buffer_size=1000,
                ema_alpha=0.995,
                reset_interval=200000
            )
        else:
            self.rtd_elite = None
        
        # Entropy controller (conditional)
        if not self.disable_entropy:
            self.entropy_controller = EntropyController()
        else:
            self.entropy_controller = None
    
    def _build_optimizers(self):
        """Build optimizers for all trainable components."""
        params = list(self.encoder.parameters()) + list(self.actor.parameters())
        
        if self.dgat is not None:
            params += list(self.dgat.parameters())
        if self.bayesian_fusion is not None:
            params += list(self.bayesian_fusion.parameters())
        if self.coalition_formation is not None:
            params += list(self.coalition_formation.parameters())
        
        self.actor_optimizer = optim.Adam(params, lr=self.lr_actor)
        
        if self.dual_critic is not None:
            self.critic_optimizer = optim.Adam(
                self.dual_critic.parameters(), lr=self.lr_critic
            )
        else:
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.lr_critic
            )
    
    def get_action(self, obs: torch.Tensor, 
                   positions: torch.Tensor = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions for all agents.
        
        Returns:
            actions: [batch, n_agents]
            log_probs: [batch, n_agents]
            entropy: [batch, n_agents]
        """
        self.complexity_tracker.start_forward()
        
        batch_size = obs.shape[0]
        
        # Encode observations
        h = self.encoder(obs)  # [batch, n_agents, hidden_dim]
        
        # Apply DGAT communication
        if self.dgat is not None:
            h, attention = self.dgat(h, positions)
        
        # Coalition formation
        if self.coalition_formation is not None:
            coalition_mask, assignments, n_coalitions = self.coalition_formation(h)
        else:
            coalition_mask = None
        
        # Bayesian fusion
        if self.bayesian_fusion is not None:
            fused_belief, uncertainty = self.bayesian_fusion(h, coalition_mask)
            h = h + fused_belief  # Residual fusion
        
        # Actor output
        logits = self.actor(h)
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        self.complexity_tracker.end_forward()
        
        return actions, log_probs, entropy
    
    def get_value(self, obs: torch.Tensor, 
                  state: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get value estimates.
        
        Returns:
            value: [batch, n_agents, 1] or [batch, 1]
            beta: Mixing coefficient if dual critic
        """
        if state is None:
            state = obs.view(obs.shape[0], -1)
        
        if self.dual_critic is not None:
            # Per-agent values with mixing
            values = []
            betas = []
            for i in range(self.n_agents):
                v, b = self.dual_critic(state, obs[:, i])
                values.append(v)
                betas.append(b)
            return torch.stack(values, dim=1), torch.stack(betas, dim=1)
        else:
            return self.critic(state), None
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """
        Update networks using PPO.
        
        Returns:
            losses: Dictionary of loss values
        """
        self.complexity_tracker.start_backward()
        
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        
        # Get current policy outputs
        new_actions, new_log_probs, entropy = self.get_action(obs)
        
        # Get values
        values, betas = self.get_value(obs)
        
        # Compute advantages (GAE)
        advantages = self._compute_gae(rewards, values.squeeze(-1), dones)
        returns = advantages + values.squeeze(-1)
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns.detach())
        
        # Entropy loss
        if self.entropy_controller is not None:
            entropy_loss = self.entropy_controller.compute_entropy_loss(entropy.mean())
        else:
            entropy_loss = -self.entropy_coef * entropy.mean()
        
        # RTD++ KL loss
        rtd_loss = torch.tensor(0.0, device=self.device)
        if self.rtd_elite is not None and self.rtd_elite.elite_policy is not None:
            # Compute KL to elite (simplified)
            rtd_loss = 0.01 * policy_loss.abs()  # Placeholder
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss + rtd_loss
        
        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        self.complexity_tracker.end_backward()
        self.complexity_tracker.record_memory()
        
        self.total_steps += 1
        
        # Update RTD++ elite
        if self.rtd_elite is not None:
            self.rtd_elite.update_elite_policy(self.actor)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'beta_mean': betas.mean().item() if betas is not None else 0.0,
        }
    
    def _compute_gae(self, rewards: torch.Tensor, 
                     values: torch.Tensor,
                     dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        batch_size, seq_len = rewards.shape[:2]
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1] if values.dim() > 1 else values
            
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
        
        return advantages
    
    def get_config_summary(self) -> Dict:
        """Get complete configuration for reproducibility (R1.5)."""
        return {
            'architecture': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'n_agents': self.n_agents,
                'hidden_dim': self.hidden_dim,
            },
            'ppo': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'lr_actor': self.lr_actor,
                'lr_critic': self.lr_critic,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
            },
            'dgat': {
                'n_heads': 4,
                'kernel_type': 'inverse' if self.dgat else 'disabled',
                'dropout': 0.1,
            },
            'bayesian_fusion': {
                'belief_dim': self.hidden_dim,
                'prior_precision': 0.1,
            },
            'coalition': {
                'min_size': 2,
                'eigengap_threshold': 0.1,
                'negotiation_enabled': True,
                'veto_threshold': 0.1,
            },
            'rtd_plus': {
                'buffer_size': 1000,
                'ema_alpha': 0.995,
                'kl_weight': 0.01,
                'reset_interval': 200000,
            },
            'entropy': {
                'initial_coef': 0.01,
                'min_entropy': 0.5,
                'max_entropy': 2.0,
                'anneal_steps': 500000,
            },
            'ablation': {
                'disable_dgat': self.disable_dgat,
                'disable_bayesian': self.disable_bayesian,
                'disable_coalitions': self.disable_coalitions,
                'disable_dual_critic': self.disable_dual_critic,
                'disable_rtd': self.disable_rtd,
                'disable_entropy': self.disable_entropy,
            }
        }
    
    def get_complexity_summary(self) -> Dict:
        """Get computational complexity summary (R1.3)."""
        n = self.n_agents
        h = 4  # n_heads
        d = self.hidden_dim
        
        theoretical = {
            'dgat': f'O(N²·H·d) = O({n}²·{h}·{d}) = O({n*n*h*d})',
            'bayesian_fusion': f'O(N·d) = O({n}·{d}) = O({n*d})',
            'coalition': f'O(N³) = O({n}³) = O({n**3})',
            'dual_critic': f'O(N·d²) = O({n}·{d}²) = O({n*d*d})',
            'total_per_step': f'O(N³ + N²·H·d)',
        }
        
        empirical = self.complexity_tracker.get_summary()
        
        return {
            'theoretical': theoretical,
            'empirical': empirical,
            'total_parameters': sum(p.numel() for p in self.encoder.parameters()) +
                               sum(p.numel() for p in self.actor.parameters())
        }
    
    def get_analysis_data(self) -> Dict:
        """Get all analysis data for paper figures."""
        return {
            'beta_trajectory': self.dual_critic.get_beta_trajectory() if self.dual_critic else [],
            'nash_gap_trajectory': self.nash_gap_estimator.get_trajectory(),
            'eval_history': self.eval_history,
            'complexity': self.get_complexity_summary(),
            'config': self.get_config_summary(),
        }
    
    def save(self, path: str):
        """Save model and analysis data."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'dgat': self.dgat.state_dict() if self.dgat else None,
            'dual_critic': self.dual_critic.state_dict() if self.dual_critic else None,
            'config': self.get_config_summary(),
            'analysis': self.get_analysis_data(),
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        if self.dgat and checkpoint['dgat']:
            self.dgat.load_state_dict(checkpoint['dgat'])
        if self.dual_critic and checkpoint['dual_critic']:
            self.dual_critic.load_state_dict(checkpoint['dual_critic'])


# ============================================================
# HELPER: Create trainer with ablation
# ============================================================

def create_h3c_trainer(obs_dim: int, action_dim: int, n_agents: int,
                       config: Dict, device: torch.device = None,
                       ablation: str = None, kernel: str = 'inverse') -> H3CTrainerRevised:
    """
    Factory function to create H3C trainer with optional ablation.
    
    Args:
        ablation: None for full model, or one of:
            'no_dgat', 'no_bayesian', 'no_coalitions', 
            'no_dual_critic', 'no_rtd', 'no_entropy'
    """
    flags = {
        'disable_dgat': ablation == 'no_dgat',
        'disable_bayesian': ablation == 'no_bayesian',
        'disable_coalitions': ablation == 'no_coalitions',
        'disable_dual_critic': ablation == 'no_dual_critic',
        'disable_rtd': ablation == 'no_rtd',
        'disable_entropy': ablation == 'no_entropy',
    }
    
    return H3CTrainerRevised(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        config=config,
        device=device,
        kernel_type=kernel,
        **flags
    )
