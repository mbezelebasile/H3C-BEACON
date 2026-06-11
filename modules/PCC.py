

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class StableGRU(nn.Module):
    """GRU with stability improvements."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Orthogonal init for GRU stability
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        out, hidden = self.gru(x, hidden)
        out = self.layer_norm(out)
        return out, hidden


class BeliefEstimator(nn.Module):
    """Stable belief state estimator."""
    
    def __init__(self, obs_dim: int, hidden_dim: int, belief_dim: int):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        
        self.gru = StableGRU(obs_dim, hidden_dim)
        
        self.belief_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Tanh(),
            nn.Linear(hidden_dim, belief_dim),
            nn.Tanh()  # Bounded beliefs
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Bounded [0, 1]
        )
    
    def forward(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        B, N, _ = obs.shape
        
        obs_flat = obs.view(B * N, 1, self.obs_dim)
        
        if hidden is None:
            hidden = torch.zeros(1, B * N, self.hidden_dim, device=obs.device)
        
        gru_out, hidden = self.gru(obs_flat, hidden)
        gru_out = gru_out.squeeze(1)
        
        belief = self.belief_proj(gru_out).view(B, N, self.belief_dim)
        uncertainty = self.uncertainty_head(gru_out).view(B, N, 1)
        
        return belief, hidden, uncertainty
    
    def reset_hidden(self, batch_size: int, n_agents: int, device: torch.device):
        return torch.zeros(1, batch_size * n_agents, self.hidden_dim, device=device)


class PolicyNetwork(nn.Module):
   
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int,
                 min_entropy_ratio: float = 0.0):  # DEFAULT 0 = NO FLOOR
        super().__init__()
        
        self.action_dim = action_dim
        self.max_entropy = np.log(action_dim)
        # NO entropy floor!
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Orthogonal initialization (like MAPPO)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Small init for output layer
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        
        # Bound logits for numerical stability
        logits = torch.clamp(logits, -10, 10)
        
        # Temperature scaling
        temp = max(temperature, 0.01)  # Allow lower temperature
        scaled_logits = logits / temp
        
        # Compute probabilities
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        action_probs = torch.exp(log_probs)
        
        # NO entropy floor mixing! Let it decrease naturally.
        
        return action_probs, log_probs
    
    def get_entropy(self, action_probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        return -(action_probs * log_probs).sum(dim=-1)


class MessageModule(nn.Module):
    """Simplified message passing module."""
    
    def __init__(self, state_dim: int, message_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, message_dim),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(message_dim + state_dim, message_dim),
            nn.LayerNorm(message_dim, eps=1e-6),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor, comm_weights: torch.Tensor) -> torch.Tensor:
        messages = self.encoder(state)
        received = torch.bmm(comm_weights, messages)
        decoded = self.decoder(torch.cat([received, state], dim=-1))
        return decoded


class PCC(nn.Module):
    """
    Probabilistic Coalition Coordination - MAPPO-Inspired
    
    Key change: NO entropy floor!
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 128,
                 belief_dim: int = 64,
                 message_dim: int = 32,
                 min_entropy_ratio: float = 0.0):  # DEFAULT 0!
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.message_dim = message_dim
        
        # Belief estimator
        self.belief_estimator = BeliefEstimator(obs_dim, hidden_dim, belief_dim)
        
        # Policy network (no entropy floor!)
        policy_input_dim = obs_dim + belief_dim + message_dim
        self.policy = PolicyNetwork(
            policy_input_dim, action_dim, hidden_dim, min_entropy_ratio
        )
        
        # Message module
        self.message_module = MessageModule(obs_dim + belief_dim, message_dim)
        
        # Hidden state
        self.hidden_state = None
        
        # Metrics
        self.last_entropy = None
        self.last_uncertainty = None
    
    def reset_beliefs(self, batch_size: int = 1, device: torch.device = None, force_zero: bool = False):
        """Reset belief hidden states.
        
        Args:
            batch_size: Batch size
            device: Device for tensors
            force_zero: If True, completely zero out hidden state (periodic reset)
        """
        if device is None:
            device = next(self.parameters()).device
        
        if force_zero:
            # Complete reset - helps with non-stationarity
            self.hidden_state = torch.zeros(1, batch_size * self.n_agents, 
                                           self.belief_estimator.hidden_dim, device=device)
        else:
            self.hidden_state = self.belief_estimator.reset_hidden(batch_size, self.n_agents, device)
    
    def forward(self,
                obs: torch.Tensor,
                aah_outputs: Dict[str, torch.Tensor],
                temperature: float = 1.0,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        
        if self.hidden_state is None:
            self.reset_beliefs(B, device)
        
        # Belief estimation
        beliefs, self.hidden_state, uncertainty = self.belief_estimator(obs, self.hidden_state)
        
        # Message passing
        state = torch.cat([obs, beliefs], dim=-1)
        comm_weights = aah_outputs.get('comm_weights', 
            torch.ones(B, self.n_agents, self.n_agents, device=device) / self.n_agents)
        messages = self.message_module(state, comm_weights)
        
        # Policy
        policy_input = torch.cat([obs, beliefs, messages], dim=-1)
        action_probs, log_probs = self.policy(policy_input, temperature)
        
        # Sample or argmax
        if deterministic:
            actions = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
        
        # Log prob of selected actions
        log_probs_taken = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Entropy
        entropy = self.policy.get_entropy(action_probs, log_probs)
        
        # Store metrics
        self.last_entropy = entropy.mean().item()
        self.last_uncertainty = uncertainty.mean().item()
        
        return {
            'actions': actions,
            'action_probs': action_probs,
            'log_probs': log_probs_taken,
            'log_probs_all': log_probs,
            'beliefs': beliefs,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'messages': messages
        }
    
    def evaluate_actions(self,
                         obs: torch.Tensor,
                         actions: torch.Tensor,
                         aah_outputs: Dict[str, torch.Tensor],
                         beliefs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        
        if beliefs is None:
            hidden = self.belief_estimator.reset_hidden(B, self.n_agents, device)
            beliefs, _, _ = self.belief_estimator(obs, hidden)
        
        state = torch.cat([obs, beliefs], dim=-1)
        comm_weights = aah_outputs.get('comm_weights',
            torch.ones(B, self.n_agents, self.n_agents, device=device) / self.n_agents)
        messages = self.message_module(state, comm_weights)
        
        policy_input = torch.cat([obs, beliefs, messages], dim=-1)
        action_probs, log_probs = self.policy(policy_input, temperature=1.0)
        
        log_probs_taken = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = self.policy.get_entropy(action_probs, log_probs)
        
        return {
            'log_probs': log_probs_taken,
            'entropy': entropy,
            'action_probs': action_probs
        }
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'pcc_entropy': self.last_entropy or 0.0,
            'pcc_uncertainty': self.last_uncertainty or 0.0
        }


if __name__ == "__main__":
    print("Testing PCC v3.3 (MAPPO-style, no entropy floor)...")
    
    B, N, obs_dim, action_dim = 4, 3, 18, 5
    
    pcc = PCC(obs_dim=obs_dim, action_dim=action_dim, n_agents=N, hidden_dim=64)
    pcc.reset_beliefs(B)
    
    obs = torch.randn(B, N, obs_dim)
    aah_outputs = {
        'comm_weights': F.softmax(torch.randn(B, N, N), dim=-1),
    }
    
    # Test forward
    out = pcc(obs, aah_outputs, temperature=1.0)
    
    print(f"✓ actions: {out['actions'].shape}")
    print(f"✓ action_probs sum: {out['action_probs'].sum(dim=-1)}")
    print(f"✓ entropy@temp=1.0: {out['entropy'].mean().item():.4f} (max={np.log(action_dim):.4f})")
    
    # Test with low temperature
    out_cold = pcc(obs, aah_outputs, temperature=0.1)
    print(f"✓ entropy@temp=0.1: {out_cold['entropy'].mean().item():.4f}")
    
    # Verify entropy CAN go low (unlike v3.1 where it was floored)
    out_very_cold = pcc(obs, aah_outputs, temperature=0.01)
    print(f"✓ entropy@temp=0.01: {out_very_cold['entropy'].mean().item():.4f}")
    
    print(f"✓ Params: {sum(p.numel() for p in pcc.parameters()):,}")
    print("✓ PCC v3.3 ready!")