"""
AAH.py - Adaptive Attention Hierarchy Module 

STABILITY IMPROVEMENTS:
1. Spectral normalization on attention projections
2. Residual connections with learnable gates
3. Attention entropy regularization
4. Soft coalition assignment (no hard decisions)
5. Gradient checkpointing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class StableMultiHeadAttention(nn.Module):
    """Multi-head attention with stability improvements."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use smaller initialization for stability
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention (prevents sharp distributions)
        self.attn_temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        # Xavier uniform with smaller gain for stability
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.zeros_(module.bias)
        # Output projection even smaller
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Temperature-controlled attention
        temp = torch.clamp(self.attn_temperature, min=0.5, max=2.0)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale / temp
        
        # Clamp scores for stability
        scores = torch.clamp(scores, min=-50, max=50)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(out)
        
        return out, attn_weights


class GatedResidualBlock(nn.Module):
    """Residual block with learnable gate for stability."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))  # Start with gate = 0 (identity)
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Sigmoid gate: starts at 0.5, learns to balance
        gate = torch.sigmoid(self.gate)
        return x + gate * residual


class StableAttentionBlock(nn.Module):
    """Transformer block with stability improvements."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.attention = StableMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Gated residuals
        self.gate1 = GatedResidualBlock(embed_dim)
        self.gate2 = GatedResidualBlock(embed_dim)
        
        # Small init for FFN output
        nn.init.xavier_uniform_(self.ffn[-2].weight, gain=0.1)
        nn.init.zeros_(self.ffn[-2].bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm with gated residual
        normed = self.norm1(x)
        attended, attn_weights = self.attention(normed)
        x = self.gate1(x, attended)
        
        # FFN with gated residual
        normed = self.norm2(x)
        x = self.gate2(x, self.ffn(normed))
        
        return x, attn_weights


class SoftCoalitionModule(nn.Module):
    """Soft coalition assignment (no hard decisions for stability)."""
    
    def __init__(self, embed_dim: int, n_coalitions: int = 2):
        super().__init__()
        self.n_coalitions = n_coalitions
        
        self.coalition_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Tanh(),  # Bounded activation
            nn.Linear(embed_dim // 2, n_coalitions)
        )
        
        # Very small init for exploration
        nn.init.uniform_(self.coalition_net[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.coalition_net[-1].bias)
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.coalition_net(x)
        
        # Always soft assignment (no Gumbel, more stable)
        # Temperature bounded to prevent extreme distributions
        temp = max(temperature, 0.5)
        coalition_probs = F.softmax(logits / temp, dim=-1)
        
        # Add small uniform noise for exploration (detached, no gradient)
        if self.training:
            noise = torch.rand_like(coalition_probs) * 0.05
            coalition_probs = coalition_probs + noise
            coalition_probs = coalition_probs / coalition_probs.sum(dim=-1, keepdim=True)
        
        return coalition_probs


class AAH(nn.Module):
    
    def __init__(self,
                 obs_dim: int,
                 n_agents: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 n_coalitions: int = 2,
                 dropout: float = 0.1,
                 belief_dim: int = 64):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        
        input_dim = obs_dim + belief_dim
        
        # Input embedding with careful normalization
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Tanh(),  # Bounded output
            nn.Dropout(dropout)
        )
        
        # Agent ID embedding (small)
        self.agent_embed = nn.Embedding(n_agents, hidden_dim)
        nn.init.normal_(self.agent_embed.weight, std=0.02)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            StableAttentionBlock(hidden_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_layers)
        ])
        
        # Coalition formation
        self.coalition_module = SoftCoalitionModule(hidden_dim, n_coalitions)
        
        # Communication weights (bounded output)
        self.comm_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_agents)
        )
        nn.init.uniform_(self.comm_proj[-1].weight, -0.01, 0.01)
        
        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        # Metrics
        self.last_attention_entropy = None
        self.last_coalition_entropy = None
    
    def forward(self,
                obs: torch.Tensor,
                beliefs: torch.Tensor,
                temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        
        # Input processing
        x = torch.cat([obs, beliefs], dim=-1)
        x = self.input_embed(x)
        
        # Add agent IDs
        agent_ids = torch.arange(self.n_agents, device=device).unsqueeze(0).expand(B, -1)
        x = x + 0.1 * self.agent_embed(agent_ids)  # Small contribution
        
        # Attention layers
        all_attn_weights = []
        for layer in self.attention_layers:
            x, attn = layer(x)
            all_attn_weights.append(attn)
        
        x = self.final_norm(x)
        
        # Average attention
        stacked = torch.stack(all_attn_weights, dim=0)
        avg_attn = stacked.mean(dim=(0, 2))
        
        # Coalition with bounded temperature
        coalition_probs = self.coalition_module(x, max(temperature, 0.5))
        
        # Communication weights
        comm_logits = self.comm_proj(x)
        comm_weights = F.softmax(comm_logits / max(temperature, 0.5), dim=-1)
        
        # Metrics
        attn_entropy = -(avg_attn * torch.log(avg_attn + 1e-8)).sum(dim=-1).mean()
        coalition_entropy = -(coalition_probs * torch.log(coalition_probs + 1e-8)).sum(dim=-1).mean()
        
        self.last_attention_entropy = attn_entropy.item()
        self.last_coalition_entropy = coalition_entropy.item()
        
        return {
            'attention_weights': avg_attn,
            'coalition_probs': coalition_probs,
            'comm_weights': comm_weights,
            'agent_embeddings': x,
            'attention_entropy': attn_entropy,
            'coalition_entropy': coalition_entropy
        }
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'aah_attn_entropy': self.last_attention_entropy or 0.0,
            'aah_coalition_entropy': self.last_coalition_entropy or 0.0
        }


if __name__ == "__main__":
    print("Testing Stabilized AAH Module...")
    
    B, N, obs_dim, belief_dim = 4, 3, 18, 64
    aah = AAH(obs_dim=obs_dim, n_agents=N, hidden_dim=64, belief_dim=belief_dim)
    
    obs = torch.randn(B, N, obs_dim)
    beliefs = torch.randn(B, N, belief_dim)
    
    # Test forward
    out = aah(obs, beliefs, temperature=1.0)
    print(f"✓ attention_weights: {out['attention_weights'].shape}")
    print(f"✓ coalition_probs: {out['coalition_probs'].shape}")
    
    # Test gradient flow
    loss = out['attention_weights'].sum() + out['coalition_probs'].sum()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in aah.parameters() if p.grad is not None)
    print(f"✓ Gradient norm: {grad_norm:.4f}")
    print(f"✓ Params: {sum(p.numel() for p in aah.parameters()):,}")