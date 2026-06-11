"""
H3CTrainer_Fixed.py 
==================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy
import math
import time


# ============================================================
# 1. DYNAMIC GRAPH ATTENTION NETWORK (DGAT)
# ============================================================
# R2.2 — Multiple spatial kernels
# Justification (R2.2) : kernel stationnaire issu de la théorie GP.
# L'inverse-distance approxime un kernel Matérn-½ : k(d) = exp(-λd).
# Le kernel gaussian (RBF) est plus lisse (Matérn-∞).
# Le choix 'inverse' est motivé par la localité physique dans SMAC/MPE.

class DynamicGraphAttention(nn.Module):
    """
    Dynamic Graph Attention Network avec kernel spatial
    ____ Kernels_____  :
      'inverse'    : 1/(d_ij + ε)         — défaut, Matérn-½ approx.
      'gaussian'   : exp(-d²/2σ²)         — plus lisse, Matérn-∞ approx.
      'polynomial' : (1 + d)^(-α)         — décroissance algébrique
      'learned'    : MLP(d) → poids       — appris end-to-end
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4,
                 kernel_type: str = 'inverse', kernel_params: Dict = None):
        super().__init__()
        self.n_heads     = n_heads
        self.head_dim    = hidden_dim // n_heads
        self.hidden_dim  = hidden_dim
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}

        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Paramètres spécifiques au kernel (R2.2)
        if kernel_type == 'inverse':
            self.epsilon = nn.Parameter(
                torch.tensor(self.kernel_params.get('epsilon', 0.1)))
        elif kernel_type == 'gaussian':
            self.sigma = nn.Parameter(
                torch.tensor(self.kernel_params.get('sigma', 1.0)))
        elif kernel_type == 'polynomial':
            self.alpha = nn.Parameter(
                torch.tensor(self.kernel_params.get('alpha', 2.0)))
        elif kernel_type == 'learned':
            self.distance_mlp = nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid())

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def compute_distance_weight(self, distances: torch.Tensor) -> torch.Tensor:
       
        if self.kernel_type == 'inverse':
            return 1.0 / (distances + torch.abs(self.epsilon) + 1e-6)
        elif self.kernel_type == 'gaussian':
            return torch.exp(-distances.pow(2) / (2 * self.sigma.pow(2) + 1e-6))
        elif self.kernel_type == 'polynomial':
            return (1 + distances).pow(-torch.abs(self.alpha))
        elif self.kernel_type == 'learned':
            return self.distance_mlp(distances.view(-1, 1)).view(distances.shape)
        return torch.ones_like(distances)

    def forward(self, h: torch.Tensor,
                positions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h         : [B, N, input_dim]
        positions : [B, N, 2] (optionnel)
        → h_out [B, N, hidden_dim], attention [B, n_heads, N, N]
        """
        B, N, _ = h.shape
        Q = self.W_q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if positions is not None:
            diff  = positions.unsqueeze(2) - positions.unsqueeze(1)
            dist  = torch.norm(diff, dim=-1)
            w     = self.compute_distance_weight(dist).unsqueeze(1)
            scores = scores * w

        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        out = self.W_o(out)
        out = self.layer_norm(
            h[:, :, :self.hidden_dim] + out
            if h.shape[-1] >= self.hidden_dim else out)
        return out, attention


# ============================================================
# 2. BAYESIAN BELIEF FUSION
# ============================================================

class BayesianBeliefFusion(nn.Module):

    def __init__(self, belief_dim: int, n_agents: int,
                 use_mixture: bool = False, n_components: int = 3):
        super().__init__()
        self.belief_dim   = belief_dim
        self.n_agents     = n_agents
        self.use_mixture  = use_mixture
        self.n_components = n_components

        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, belief_dim * 2), nn.ReLU(),
            nn.Linear(belief_dim * 2, belief_dim * 2))
        self.belief_attention = nn.Sequential(
            nn.Linear(belief_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))

        if use_mixture:
            # Extension mixture de gaussiennes (R2.3 future work)
            self.mixture_weights = nn.Linear(belief_dim * 2, n_components)
            self.component_means = nn.ModuleList(
                [nn.Linear(belief_dim, belief_dim) for _ in range(n_components)])
            self.component_vars  = nn.ModuleList(
                [nn.Linear(belief_dim, belief_dim) for _ in range(n_components)])

    def forward(self, observations: torch.Tensor,
                coalition_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = observations.shape
        belief_params = self.belief_encoder(observations[:, :, :self.belief_dim])
        mu      = belief_params[:, :, :self.belief_dim]
        log_var = belief_params[:, :, self.belief_dim:]
        precision = torch.exp(-log_var)
        eta1 = precision * mu
        eta2 = -0.5 * precision

        attn = self.belief_attention(belief_params)   # [B, N, 1]

        if coalition_mask is not None:
            a = attn.squeeze(-1).unsqueeze(1).expand(-1, N, -1)   # [B,N,N]
            a = a * coalition_mask
            attn_w = F.softmax(a + (1 - coalition_mask) * (-1e9), dim=-1)
        else:
            attn_w = F.softmax(attn, dim=1)
            attn_w = attn_w.squeeze(-1).unsqueeze(1).expand(-1, N, -1)

        fused_eta1 = torch.bmm(attn_w, eta1)
        fused_eta2 = torch.bmm(attn_w, eta2)
        fused_prec = -2 * fused_eta2
        fused_mu   = fused_eta1 / (fused_prec + 1e-6)
        fused_var  = 1.0 / (fused_prec + 1e-6)
        return fused_mu, fused_var


# ============================================================
# 3. ADAPTIVE COALITION FORMATION
# ============================================================
# R2.4 — Négociation des coalitions :
#   Mécanisme de veto : agent i quitte la coalition si
#   utility_alone - utility_with > veto_threshold.
#   Coût : O(N²) MLP calls → désactivé pendant update() [FIX-3].

class AdaptiveCoalitionFormation(nn.Module):
   

    def __init__(self, hidden_dim: int, n_agents: int,
                 min_coalition_size: int = 2,
                 enable_negotiation: bool = True,
                 veto_threshold: float = 0.1):
        super().__init__()
        self.hidden_dim         = hidden_dim
        self.n_agents           = n_agents
        self.min_coalition_size = min_coalition_size
        self.enable_negotiation = enable_negotiation
        self.veto_threshold     = veto_threshold

        self.affinity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid())
        self.sigma = nn.Parameter(torch.tensor(1.0))

        if enable_negotiation:
            # R2.4 : prédiction d'utilité pour le veto
            self.utility_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def compute_affinity_matrix(self, h: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        A  = self.affinity_net(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        return (A + A.transpose(-1, -2)) / 2

    def spectral_clustering(self, affinity: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, N, _ = affinity.shape
        d_inv = torch.diag_embed(1.0 / (affinity.sum(dim=-1).sqrt() + 1e-6))
        L = (torch.eye(N, device=affinity.device)
             - torch.bmm(torch.bmm(d_inv, affinity), d_inv))

        eigvals, eigvecs = torch.linalg.eigh(L.detach().cpu())
        eigvals = eigvals.to(affinity.device)
        eigvecs = eigvecs.to(affinity.device)

        gaps  = eigvals[:, 1:] - eigvals[:, :-1]
        k_raw = gaps[:, :N//2].argmax(dim=-1) + 1
        k     = int(k_raw[0].item())
        k     = max(2, min(k, N // max(self.min_coalition_size, 1)))

        feats = eigvecs[:, :, 1:k+1]
        assigns = ((feats[:, :, 0] > 0).long()
                   if k == 2 else feats.abs().argmax(dim=-1))
        return assigns, k

    def negotiate_coalitions(self, h: torch.Tensor,
                              proposed: torch.Tensor) -> torch.Tensor:
        """R2.4 — Veto si utility_alone > utility_with + threshold."""
        if not self.enable_negotiation:
            return proposed
        B, N, _ = h.shape
        final = proposed.clone()
        for i in range(N):
            cid     = proposed[:, i]
            members = (proposed == cid.unsqueeze(-1))
            c_mean  = (h * members.unsqueeze(-1).float()).sum(1) / \
                      members.sum(1, keepdim=True).float().clamp(min=1)
            u_with  = self.utility_predictor(torch.cat([h[:, i], c_mean],   dim=-1))
            u_alone = self.utility_predictor(torch.cat([h[:, i], h[:, i]], dim=-1))
            veto    = (u_alone - u_with).squeeze(-1) > self.veto_threshold
            final[:, i] = torch.where(veto,
                                       torch.full_like(cid, N + i), cid)
        return final

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        A = self.compute_affinity_matrix(h)
        assigns, k = self.spectral_clustering(A)
        assigns = self.negotiate_coalitions(h, assigns)
        mask = (assigns.unsqueeze(-1) == assigns.unsqueeze(-2)).float()
        return mask, assigns, k


# ============================================================
# 4. DUAL CRITIC ARCHITECTURE
# ============================================================


class DualCritic(nn.Module):
    

    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.global_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.local_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.beta_net = nn.Sequential(
            nn.Linear(state_dim + obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())
        # [FIX-4] : deque bornée (pas de fuite mémoire)
        self.beta_history: deque = deque(maxlen=1000)

    def forward(self, state: torch.Tensor,
                obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_g  = self.global_critic(state)
        v_l  = self.local_critic(obs)
        beta = self.beta_net(torch.cat([state, obs], dim=-1))
        self.beta_history.append(float(beta.mean().item()))   # R2.5
        return beta * v_g + (1 - beta) * v_l, beta

    def get_beta_trajectory(self) -> List[float]:
        return list(self.beta_history)

    def reset_beta_history(self):
        self.beta_history.clear()


# ============================================================
# 5. RTD++ ELITE ANCHORING
# ============================================================

class RTDPlusPlusElite(nn.Module):
    

    def __init__(self, buffer_size: int = 1000,
                 ema_alpha: float = 0.995,
                 kl_weight: float = 0.01,
                 reset_interval: int = 200_000,
                 staleness_threshold: float = 0.2):
        super().__init__()
        self.buffer_size         = buffer_size
        self.ema_alpha           = ema_alpha
        self.kl_weight           = kl_weight
        self.reset_interval      = reset_interval
        self.staleness_threshold = staleness_threshold

        self.elite_buffer             = []
        self.elite_policy             = None
        self._elite_params            = None
        self.steps_since_reset        = 0
        self.steps_since_improvement  = 0
        self.best_reward              = float('-inf')
        # R2.6 — fenêtre glissante anti-élitisme
        self._return_window: deque = deque(maxlen=30)

    def add_trajectory(self, trajectory: Dict, reward: float):
        
        self.elite_buffer.append((trajectory, reward))
        if len(self.elite_buffer) > self.buffer_size:
            self.elite_buffer.sort(key=lambda x: x[1], reverse=True)
            self.elite_buffer = self.elite_buffer[:self.buffer_size]
        if reward > self.best_reward * (1 + self.staleness_threshold):
            self.best_reward = reward
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def update_elite_policy(self, current_policy: nn.Module):
        
        if self.elite_policy is None:
            self.elite_policy = copy.deepcopy(current_policy)
        else:
            for ep, cp in zip(self.elite_policy.parameters(),
                               current_policy.parameters()):
                ep.data = self.ema_alpha * ep.data + (1 - self.ema_alpha) * cp.data

        self.steps_since_reset += 1
        if self.steps_since_reset >= self.reset_interval:
            self.reset_elite()

    def update_elite_from_return(self, current_policy: nn.Module, ep_return: float):
        
        self._return_window.append(ep_return)
        if len(self._return_window) < 5:
            return
        rolling = float(np.mean(self._return_window))
        if rolling > self.best_reward:
            self.best_reward = rolling
            curr = torch.cat([p.detach().view(-1)
                              for p in current_policy.parameters()])
            if self._elite_params is None:
                self._elite_params = curr.clone()
            else:
                self._elite_params = (self.ema_alpha * self._elite_params
                                      + (1 - self.ema_alpha) * curr)

    def reset_elite(self):
        self.elite_policy       = None
        self._elite_params      = None
        self.steps_since_reset  = 0
        self.elite_buffer       = []

    def should_reset(self) -> bool:
        return self.steps_since_improvement > self.reset_interval // 2

    def compute_kl_approx(self, current_policy: nn.Module) -> torch.Tensor:
        if self._elite_params is None:
            return torch.tensor(0.0)
        curr = torch.cat([p.view(-1) for p in current_policy.parameters()])
        return self.kl_weight * ((curr - self._elite_params.to(curr.device))**2).mean()

    def compute_novelty_bonus(self, action: torch.Tensor,
                               elite_action: torch.Tensor) -> torch.Tensor:
        return 0.01 * (action - elite_action).pow(2).sum(dim=-1).sqrt()


# ============================================================
# 6. ENTROPY CONTROLLER
# ============================================================

class EntropyController:
    

    def __init__(self, initial_entropy_coef: float = 0.01,
                 min_entropy: float = 0.5, max_entropy: float = 2.0,
                 anneal_steps: int = 500_000):
        self.entropy_coef = initial_entropy_coef
        self.min_entropy  = min_entropy
        self.max_entropy  = max_entropy
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def get_entropy_bounds(self) -> Tuple[float, float]:
        p = min(1.0, self.current_step / self.anneal_steps)
        lo = self.min_entropy * (1 - p) + 0.01 * p
        hi = self.max_entropy * (1 - p) + 0.50 * p
        return lo, hi

    def compute_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        lo, hi = self.get_entropy_bounds()
        loss = (F.relu(lo - entropy) + F.relu(entropy - hi)).mean()
        self.current_step += 1
        return loss


# ============================================================
# 7. COMPLEXITY TRACKER  (R1.3)
# ============================================================

class ComplexityTracker:
   

    def __init__(self):
        self.forward_times  : deque = deque(maxlen=500)
        self.backward_times : deque = deque(maxlen=500)
        self.memory_usage   : deque = deque(maxlen=500)
        self.step_count = 0
        self._t = 0.0

    def start_forward(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self._t = time.time()

    def end_forward(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.forward_times.append(time.time() - self._t)

    def start_backward(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self._t = time.time()

    def end_backward(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.backward_times.append(time.time() - self._t)

    def record_memory(self):
        mem = torch.cuda.memory_allocated() / 1024**3 \
              if torch.cuda.is_available() else 0.0
        self.memory_usage.append(mem)
        self.step_count += 1

    def get_summary(self) -> Dict:
        return {
            'mean_forward_time_ms':  np.mean(self.forward_times)  * 1000
                                     if self.forward_times  else 0.0,
            'mean_backward_time_ms': np.mean(self.backward_times) * 1000
                                     if self.backward_times else 0.0,
            'max_memory_gb':         max(self.memory_usage)
                                     if self.memory_usage   else 0.0,
            'total_steps':           self.step_count,
        }


# ============================================================
# 8. NASH GAP ESTIMATOR  (R2.7)
# ============================================================

class NashGapEstimator:
    

    def __init__(self, n_agents: int, action_dim: int):
        self.n_agents        = n_agents
        self.action_dim      = action_dim
        self.epsilon_history : List[float] = []

    def estimate_gap(self, q_values: torch.Tensor,
                     actions: torch.Tensor) -> float:
        taken_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        best_q  = q_values.max(dim=-1)[0]
        epsilon = (best_q - taken_q).max(dim=-1)[0].mean().item()
        self.epsilon_history.append(epsilon)
        return epsilon

    def get_trajectory(self) -> List[float]:
        return self.epsilon_history


# ============================================================
# MAIN H3C TRAINER — VERSION RÉVISÉE + CORRIGÉE
# ============================================================

class H3CTrainerRevised:
   

    def __init__(self,
                 obs_dim: int, action_dim: int, n_agents: int,
                 config: Dict, device: torch.device = None,
                 disable_dgat: bool = False,
                 disable_bayesian: bool = False,
                 disable_coalitions: bool = False,
                 disable_dual_critic: bool = False,
                 disable_rtd: bool = False,
                 disable_entropy: bool = False,
                 kernel_type: str = 'inverse'):

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.n_agents   = n_agents
        self.config     = config
        self.max_grad_norm    = float(config.get('max_grad_norm', 0.5))
        self.device     = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # R1.2 — flags d'ablation
        self.disable_dgat         = disable_dgat
        self.disable_bayesian     = disable_bayesian
        self.disable_coalitions   = disable_coalitions
        self.disable_dual_critic  = disable_dual_critic
        self.disable_rtd          = disable_rtd
        self.disable_entropy      = disable_entropy

        # Hyperparamètres (R1.5)
        self.hidden_dim   = config.get('hidden_dim',      128)
        self.gamma        = config.get('gamma',           0.99)
        self.gae_lambda   = config.get('gae_lambda',      0.95)
        self.lr_actor     = config.get('lr_actor',        3e-4)
        self.lr_critic    = config.get('lr_critic',       1e-3)
        self.clip_epsilon = config.get('clip_epsilon',    0.2)
        self.entropy_coef = config.get('entropy_coef',   0.01)
        self.value_coef   = config.get('value_loss_coef', 0.5)
        # [FIX-2] n_epochs PPO (défaut 10)
        self.n_epochs     = config.get('n_epochs', 10)

        self._build_networks(kernel_type)
        self._build_optimizers()

        # R1.3
        self.complexity_tracker = ComplexityTracker()
        # R2.7
        self.nash_gap_estimator = NashGapEstimator(n_agents, action_dim)

        self.total_steps   = 0
        self.episode_count = 0
        self.best_reward   = float('-inf')
        self.eval_history  : List[Dict] = []
        self.name          = "H3C"

    # ──────────────────────────────────────────────────────────
    # Construction des réseaux
    # ──────────────────────────────────────────────────────────

    def _build_networks(self, kernel_type: str):
        H = self.hidden_dim
        D = self.obs_dim
        N = self.n_agents

        self.encoder = nn.Sequential(
            nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H)
        ).to(self.device)

        # R2.2 — kernel configurable
        self.dgat = (DynamicGraphAttention(H, H, n_heads=4,
                                           kernel_type=kernel_type)
                     .to(self.device)) if not self.disable_dgat else None

        # R2.3 — fusion bayésienne gaussienne
        self.bayesian_fusion = (BayesianBeliefFusion(H, N)
                                .to(self.device)
                                if not self.disable_bayesian else None)

        # R2.4 — coalition + négociation (enable_negotiation=True en collecte)
        self.coalition_formation = (
            AdaptiveCoalitionFormation(H, N, enable_negotiation=True)
            .to(self.device)) if not self.disable_coalitions else None

        self.actor = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, self.action_dim)
        ).to(self.device)

        state_dim = D * N
        if not self.disable_dual_critic:
            # R2.5 — dual critic avec β logging
            self.dual_critic = DualCritic(state_dim, D, H).to(self.device)
            self.critic       = None
        else:
            self.dual_critic = None
            self.critic = nn.Sequential(
                nn.Linear(state_dim, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1)
            ).to(self.device)

        # R2.6 — RTD++ avec safeguards
        self.rtd_elite = (RTDPlusPlusElite(
            buffer_size=1000, ema_alpha=0.995,
            reset_interval=200_000) if not self.disable_rtd else None)

        self.entropy_controller = (
            EntropyController() if not self.disable_entropy else None)

    def _build_optimizers(self):
        params = (list(self.encoder.parameters()) +
                  list(self.actor.parameters()))
        if self.dgat:                params += list(self.dgat.parameters())
        if self.bayesian_fusion:     params += list(self.bayesian_fusion.parameters())
        if self.coalition_formation: params += list(self.coalition_formation.parameters())

        self.actor_optimizer  = optim.Adam(params, lr=self.lr_actor)
        critic_net = self.dual_critic if self.dual_critic else self.critic
        self.critic_optimizer = optim.Adam(critic_net.parameters(),
                                           lr=self.lr_critic)

    # ──────────────────────────────────────────────────────────
    # get_action  (collecte de données)
    # ──────────────────────────────────────────────────────────

    def get_action(self, obs: torch.Tensor,
                   positions: torch.Tensor = None,
                   deterministic: bool = False,
                   avail_actions_t: torch.Tensor = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        self.complexity_tracker.start_forward()

        h = self.encoder(obs)                            # [B, N, H]

        if self.dgat is not None:
            h, attention = self.dgat(h, positions)

     
        coalition_mask = None
        if self.coalition_formation is not None:
            coalition_mask, assignments, n_coalitions = self.coalition_formation(h)

        # R2.3 — fusion bayésienne
        if self.bayesian_fusion is not None:
            fused_belief, uncertainty = self.bayesian_fusion(h, coalition_mask)
            h = h + fused_belief

        logits = self.actor(h)                           # [B, N, A]

        
        if avail_actions_t is not None:
            logits = logits + (1.0 - avail_actions_t) * (-1e9)

        dist      = torch.distributions.Categorical(logits=logits)
        actions   = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()

        self.complexity_tracker.end_forward()
        return actions, log_probs, entropy

    # ──────────────────────────────────────────────────────────
    # get_value
    # ──────────────────────────────────────────────────────────

    def get_value(self, obs: torch.Tensor,
                  state: torch.Tensor = None
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        → values [B, N, 1], betas [B, N, 1]  (R2.5)
        """
        if state is None:
            state = obs.reshape(obs.shape[0], -1)        # [B, N*D]

        if self.dual_critic is not None:
            vs, bs = [], []
            for i in range(self.n_agents):
                v, b = self.dual_critic(state, obs[:, i])
                vs.append(v); bs.append(b)
            return torch.stack(vs, dim=1), torch.stack(bs, dim=1)
        else:
            v = self.critic(state)                        # [B, 1]
            return v.unsqueeze(1).expand(-1, self.n_agents, -1), None

    # ──────────────────────────────────────────────────────────
    # _compute_gae
    # ──────────────────────────────────────────────────────────

    def _compute_gae(self, rewards: torch.Tensor,
                     values: torch.Tensor,
                     dones: torch.Tensor) -> torch.Tensor:
        """
        GAE robuste SMAC — formats acceptés :
          rewards : [T]     values : [T,N] ou [T]     dones : [T]
        → advantages [T, N]
        """
        if rewards.dim() > 1: rewards = rewards.reshape(-1)
        if dones.dim()   > 1: dones   = dones.reshape(-1)
        values_nd = values
        values_1d = values.detach().mean(dim=-1) \
                    if values.dim() > 1 else values.detach()
        T = rewards.shape[0]

        adv = torch.zeros(T, device=rewards.device, dtype=torch.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv    = float(values_1d[t + 1]) if t < T - 1 else 0.0
            delta = (float(rewards[t]) + self.gamma * nv *
                     (1.0 - float(dones[t])) - float(values_1d[t]))
            gae   = (delta + self.gamma * self.gae_lambda *
                     (1.0 - float(dones[t])) * gae)
            adv[t] = gae

        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = values_nd.shape[-1] if values_nd.dim() > 1 else 1
        return adv.unsqueeze(-1).expand(T, N).contiguous()

    def update(self, batch: Dict) -> Dict[str, float]:
      
        self.complexity_tracker.start_backward()

        obs          = batch['obs'].to(self.device)
        actions      = batch['actions'].to(self.device)
        rewards      = batch['rewards'].to(self.device)
        dones        = batch['dones'].to(self.device)
        old_log_probs= batch['log_probs'].to(self.device)

        if obs.dim()          == 4: obs           = obs.squeeze(0)
        if actions.dim()      == 3: actions        = actions.squeeze(0)
        if rewards.dim()      >  1: rewards        = rewards.reshape(-1)
        if dones.dim()        >  1: dones          = dones.reshape(-1)
        if old_log_probs.dim()== 3: old_log_probs  = old_log_probs.squeeze(0)

        avail = batch.get('avail_actions', None)
        if avail is not None:
            avail = avail.to(self.device)
            if avail.dim() == 4: avail = avail.squeeze(0)

        with torch.no_grad():
            values_init, betas_init = self.get_value(obs)
            values_sq  = values_init.squeeze(-1)                   # [T, N]
            advantages = self._compute_gae(rewards, values_sq, dones)
            returns    = (advantages + values_sq).detach()

        all_losses: List[float] = []

        for _epoch in range(self.n_epochs):

            
            h = self.encoder(obs)
            if self.dgat is not None:
                h, _ = self.dgat(h)
            if self.bayesian_fusion is not None:
                fm, _ = self.bayesian_fusion(h, coalition_mask=None)
                h = h + fm
            logits = self.actor(h)
            if avail is not None:
                logits = logits + (1.0 - avail) * (-1e9)

            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)               # [T, N]
            entropy       = dist.entropy()                       # [T, N]

           
            values_new, betas_new = self.get_value(obs)
            values_new_sq = values_new.squeeze(-1)

            # PPO-CLIP
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio,
                                1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(values_new_sq, returns)
            ent_mean    = entropy.mean()

            if self.entropy_controller is not None:
                entropy_loss = self.entropy_controller.compute_entropy_loss(ent_mean)
            else:
                entropy_loss = -self.entropy_coef * ent_mean

            # RTD++ KL approximation
            rtd_loss = torch.tensor(0.0, device=self.device)
            if self.rtd_elite is not None:
                rtd_loss = self.rtd_elite.compute_kl_approx(self.actor)

            total_loss = (policy_loss
                          + self.value_coef * value_loss
                          + entropy_loss
                          + rtd_loss)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

         
            nn.utils.clip_grad_norm_(
                self.actor_optimizer.param_groups[0]['params'],
                self.max_grad_norm)
            # critic / dual_critic
            nn.utils.clip_grad_norm_(
                self.critic_optimizer.param_groups[0]['params'],
                self.max_grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            all_losses.append(total_loss.item())

        self.complexity_tracker.end_backward()
        self.complexity_tracker.record_memory()
        self.total_steps += 1

        # R2.6 — update elite policy
        if self.rtd_elite is not None:
            self.rtd_elite.update_elite_policy(self.actor)

        return {
            'loss':        float(np.mean(all_losses)),
            'policy_loss': policy_loss.item(),
            'value_loss':  value_loss.item(),
            'entropy':     ent_mean.item(),
            'beta_mean':   float(betas_init.mean().item())
                           if betas_init is not None else 0.0,
        }

    
    def get_config_summary(self) -> Dict:
        """R1.5 — Export complet de la configuration."""
        return {
            'architecture': {
                'obs_dim':    self.obs_dim,
                'action_dim': self.action_dim,
                'n_agents':   self.n_agents,
                'hidden_dim': self.hidden_dim,
            },
            'ppo': {
                'gamma':        self.gamma,
                'gae_lambda':   self.gae_lambda,
                'lr_actor':     self.lr_actor,
                'lr_critic':    self.lr_critic,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'n_epochs':     self.n_epochs,
            },
            'dgat': {
                'n_heads':     4,
                'kernel_type': 'disabled' if self.dgat is None else 'inverse',
            },
            'bayesian_fusion': {
                'belief_dim':      self.hidden_dim,
                'prior_precision': 0.1,
                'gaussian_approx': True,       # R2.3
            },
            'coalition': {
                'min_size':            2,
                'negotiation_enabled': True,   # R2.4
                'veto_threshold':      0.1,
                'disabled_in_update':  True,   # [FIX-3]
            },
            'rtd_plus': {
                'buffer_size':    1000,
                'ema_alpha':      0.995,
                'kl_weight':      0.01,
                'reset_interval': 200_000,
                'window_30ep':    True,        # R2.6 anti-élitisme
            },
            'entropy': {
                'initial_coef': 0.01,
                'min_entropy':  0.5,
                'max_entropy':  2.0,
                'anneal_steps': 500_000,
            },
            'ablation': {
                'disable_dgat':        self.disable_dgat,
                'disable_bayesian':    self.disable_bayesian,
                'disable_coalitions':  self.disable_coalitions,
                'disable_dual_critic': self.disable_dual_critic,
                'disable_rtd':         self.disable_rtd,
                'disable_entropy':     self.disable_entropy,
            },
        }

    def get_complexity_summary(self) -> Dict:
        
        N = self.n_agents; H = 4; d = self.hidden_dim
        return {
            'theoretical': {
                'dgat_dense':     f'O(N²·H·d) = O({N**2*H*d})',
                'dgat_sparse_kNN':f'O(N·k·H·d) pour k voisins',
                'bayesian_fusion':f'O(N·d) = O({N*d})',
                'coalition':      f'O(N³) = O({N**3}) [collecte seulement, FIX-3]',
                'dual_critic':    f'O(N·d²) = O({N*d*d})',
                'total_per_step': f'O(N²·H·d + N·d)',
            },
            'empirical': self.complexity_tracker.get_summary(),
            'total_params': (
                sum(p.numel() for p in self.encoder.parameters()) +
                sum(p.numel() for p in self.actor.parameters())
            ),
        }

    def get_analysis_data(self) -> Dict:
        return {
            'beta_trajectory':     self.dual_critic.get_beta_trajectory()
                                   if self.dual_critic else [],
            'nash_gap_trajectory': self.nash_gap_estimator.get_trajectory(),
            'eval_history':        self.eval_history,
            'complexity':          self.get_complexity_summary(),
            'config':              self.get_config_summary(),
        }

    def save(self, path: str):
        torch.save({
            'encoder':     self.encoder.state_dict(),
            'actor':       self.actor.state_dict(),
            'dgat':        self.dgat.state_dict()        if self.dgat        else None,
            'dual_critic': self.dual_critic.state_dict() if self.dual_critic else None,
            'config':      self.get_config_summary(),
            'analysis':    self.get_analysis_data(),
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ck['encoder'])
        self.actor.load_state_dict(ck['actor'])
        if self.dgat        and ck.get('dgat'):
            self.dgat.load_state_dict(ck['dgat'])
        if self.dual_critic and ck.get('dual_critic'):
            self.dual_critic.load_state_dict(ck['dual_critic'])


# ============================================================
# FACTORY — ablation study (R1.2)
# ============================================================

def create_h3c_trainer(obs_dim: int, action_dim: int, n_agents: int,
                       config: Dict, device: torch.device = None,
                       ablation: str = None,
                       kernel: str = 'inverse') -> H3CTrainerRevised:
   
    flags = {
        'disable_dgat':        ablation == 'no_dgat',
        'disable_bayesian':    ablation == 'no_bayesian',
        'disable_coalitions':  ablation == 'no_coalitions',
        'disable_dual_critic': ablation == 'no_dual_critic',
        'disable_rtd':         ablation == 'no_rtd',
        'disable_entropy':     ablation == 'no_entropy',
    }
    return H3CTrainerRevised(
        obs_dim=obs_dim, action_dim=action_dim,
        n_agents=n_agents, config=config,
        device=device, kernel_type=kernel, **flags)