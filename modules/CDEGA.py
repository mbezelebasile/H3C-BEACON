"""
CDEGA.py - Coalition-Driven Emergent Goal Alignment 

STABILITY IMPROVEMENTS:
1. Reduced loss magnitude (scaling factors)
2. Detached intrinsic rewards (no interference with main learning)
3. Progressive loss activation (starts weak, increases over time)
4. Bounded goal representations
5. Simpler temporal tracking


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class GoalEncoder(nn.Module):
    """Goal encoder with bounded outputs."""
    
    def __init__(self, input_dim: int, goal_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, goal_dim)
        )
        
        # Small init
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        goals = self.encoder(x)
        # L2 normalize for stable similarity computation
        return F.normalize(goals, dim=-1, eps=1e-8)


class CDEGA(nn.Module):
    """
    Coalition-Driven Emergent Goal Alignment - STABILIZED
    
    Key stability features:
    - Reduced loss magnitudes
    - Progressive activation of auxiliary losses
    - Bounded representations
    - Detached intrinsic rewards
    """
    
    def __init__(self,
                 obs_dim: int,
                 n_agents: int,
                 goal_dim: int = 64,
                 hidden_dim: int = 128,
                 temperature: float = 0.1,
                 max_loss_weight: float = 0.05):  # Much smaller than before!
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.goal_dim = goal_dim
        self.temperature = temperature
        self.max_loss_weight = max_loss_weight
        
        # Goal encoder
        self.goal_encoder = GoalEncoder(obs_dim, goal_dim, hidden_dim)
        
        # Simple EMA for goal tracking (no parameters)
        self.register_buffer('goal_ema', None)
        self.ema_decay = 0.99
        
        # Progressive activation
        self.register_buffer('step_count', torch.tensor(0))
        self.warmup_steps = 1000  # Steps before losses are fully active
        
        # Intrinsic reward scale (very small)
        self.intrinsic_scale = 0.01
        
        # Metrics
        self.last_alignment_loss = None
        self.last_diversity = None
        self.last_goal_stability = None
    
    def get_loss_weight(self) -> float:
        """Progressive loss weight (starts small, increases)."""
        progress = min(self.step_count.item() / self.warmup_steps, 1.0)
        return self.max_loss_weight * progress
    
    def encode_goals(self, obs: torch.Tensor) -> torch.Tensor:
        return self.goal_encoder(obs)
    
    def update_ema(self, goals: torch.Tensor):
        """
        Update exponential moving average of goals.
        Uses mean across batch to handle variable batch sizes.
        
        Args:
            goals: [B, n_agents, goal_dim]
        """
        with torch.no_grad():
            # Compute mean goals per agent across batch: [n_agents, goal_dim]
            mean_goals = goals.mean(dim=0)
            
            if self.goal_ema is None:
                self.goal_ema = mean_goals.clone()
            else:
                # EMA update with mean goals (batch-size independent)
                self.goal_ema = self.ema_decay * self.goal_ema + (1 - self.ema_decay) * mean_goals
    
    def compute_alignment_loss(self,
                               goals: torch.Tensor,
                               coalition_probs: torch.Tensor) -> torch.Tensor:
        """
        Simplified alignment loss with bounded magnitude.
        """
        B, N, D = goals.shape
        
        # Similarity matrix (already normalized)
        similarity = torch.bmm(goals, goals.transpose(-2, -1))
        
        # Coalition mask
        coalition_mask = torch.bmm(coalition_probs, coalition_probs.transpose(-2, -1))
        
        # Remove diagonal
        eye = torch.eye(N, device=goals.device).unsqueeze(0)
        mask = coalition_mask * (1 - eye)
        
        # Simple loss: encourage similar goals within coalition
        # Bounded between 0 and 1
        alignment_loss = 1.0 - (similarity * mask).sum() / (mask.sum() + 1e-8)
        
        return torch.clamp(alignment_loss, 0, 1)
    
    def compute_diversity_loss(self, goals: torch.Tensor) -> torch.Tensor:
        """Prevent goal collapse - bounded loss."""
        B, N, D = goals.shape
        
        # Check if all goals are too similar
        similarity = torch.bmm(goals, goals.transpose(-2, -1))
        
        eye = torch.eye(N, device=goals.device).unsqueeze(0)
        off_diag = (similarity * (1 - eye)).abs()
        
        # Mean off-diagonal similarity (bounded 0-1)
        mean_sim = off_diag.sum() / (B * N * (N - 1) + 1e-8)
        
        # Penalize if too similar (soft penalty)
        diversity_loss = F.relu(mean_sim - 0.5)  # Only penalize if > 0.5 similarity
        
        return diversity_loss
    
    def forward(self,
                obs: torch.Tensor,
                coalition_probs: Optional[torch.Tensor] = None,
                compute_loss: bool = True) -> Dict[str, torch.Tensor]:
        
        self.step_count += 1
        
        # Encode goals
        goals = self.encode_goals(obs)
        
        # Update EMA
        self.update_ema(goals)
        
        outputs = {
            'goals': goals,
            # Expand goal_ema to batch dimension if needed: [n_agents, goal_dim] -> [B, n_agents, goal_dim]
            'goal_ema': self.goal_ema.unsqueeze(0).expand(goals.shape[0], -1, -1) if self.goal_ema is not None else goals
        }
        
        if compute_loss and coalition_probs is not None:
            loss_weight = self.get_loss_weight()
            
            alignment_loss = self.compute_alignment_loss(goals, coalition_probs)
            diversity_loss = self.compute_diversity_loss(goals)
            
            # Weighted losses (very small contribution)
            outputs['alignment_loss'] = loss_weight * alignment_loss
            outputs['diversity_loss'] = loss_weight * diversity_loss * 0.1
            
            # Metrics
            self.last_alignment_loss = alignment_loss.item()
            
            similarity = torch.bmm(goals, goals.transpose(-2, -1))
            eye = torch.eye(self.n_agents, device=goals.device).unsqueeze(0)
            self.last_diversity = 1.0 - (similarity * (1 - eye)).abs().mean().item()
            
            if self.goal_ema is not None:
                # goal_ema is [n_agents, goal_dim], need to compare with goals [B, n_agents, goal_dim]
                # Expand goal_ema to [B, n_agents, goal_dim] for comparison
                goal_ema_expanded = self.goal_ema.unsqueeze(0).expand(goals.shape[0], -1, -1)
                self.last_goal_stability = F.cosine_similarity(
                    goals.view(-1, goals.shape[-1]),
                    goal_ema_expanded.reshape(-1, goal_ema_expanded.shape[-1]),
                    dim=-1
                ).mean().item()
        
        return outputs
    
    def compute_intrinsic_rewards(self,
                                  obs: torch.Tensor,
                                  next_obs: torch.Tensor,
                                  coalition_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute intrinsic rewards - DETACHED to not interfere with main learning.
        """
        with torch.no_grad():
            goals = self.encode_goals(obs)
            next_goals = self.encode_goals(next_obs)
            
            # Goal progress (cosine similarity change)
            progress = F.cosine_similarity(next_goals, goals, dim=-1)
            
            # Coalition alignment bonus
            if coalition_probs is not None:
                coalition_mask = torch.bmm(coalition_probs, coalition_probs.transpose(-2, -1))
                similarity = torch.bmm(next_goals, next_goals.transpose(-2, -1))
                alignment = (similarity * coalition_mask).mean(dim=-1)
            else:
                alignment = torch.zeros_like(progress)
            
            # Small, bounded intrinsic reward
            intrinsic = self.intrinsic_scale * (progress + 0.5 * alignment)
            intrinsic = torch.clamp(intrinsic, -0.1, 0.1)
            
            return intrinsic
    
    def reset(self):
        self.goal_ema = None
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'cdega_alignment': self.last_alignment_loss or 0.0,
            'cdega_diversity': self.last_diversity or 0.0,
            'cdega_stability': self.last_goal_stability or 0.0,
            'cdega_loss_weight': self.get_loss_weight()
        }


if __name__ == "__main__":
    print("Testing Stabilized CDEGA Module...")
    
    B, N, obs_dim = 4, 3, 18
    
    cdega = CDEGA(obs_dim=obs_dim, n_agents=N, goal_dim=32, hidden_dim=64)
    
    obs = torch.randn(B, N, obs_dim)
    coalition_probs = F.softmax(torch.randn(B, N, 2), dim=-1)
    
    # Test forward
    out = cdega(obs, coalition_probs, compute_loss=True)
    
    print(f"✓ goals: {out['goals'].shape}")
    print(f"✓ alignment_loss: {out['alignment_loss'].item():.6f}")
    print(f"✓ diversity_loss: {out['diversity_loss'].item():.6f}")
    
    # Test loss progression
    for _ in range(500):
        cdega(obs, coalition_probs)
    out2 = cdega(obs, coalition_probs, compute_loss=True)
    print(f"✓ alignment_loss after 500 steps: {out2['alignment_loss'].item():.6f}")
    
    # Test intrinsic rewards
    next_obs = torch.randn(B, N, obs_dim)
    intrinsic = cdega.compute_intrinsic_rewards(obs, next_obs, coalition_probs)
    print(f"✓ intrinsic_rewards: {intrinsic.shape}, mean={intrinsic.mean().item():.4f}")
    
    # Gradient flow
    loss = out['alignment_loss'] + out['diversity_loss']
    loss.backward()
    print("✓ Gradient flow OK")
    print(f"✓ Params: {sum(p.numel() for p in cdega.parameters()):,}")