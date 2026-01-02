"""
Simple Spread Environment (MPE)
================================
Cooperative navigation: N agents must cover N landmarks.

Standard Benchmark:
- 3 agents, 3 landmarks
- Agents rewarded for covering landmarks
- Collision penalty

Reference: 
    Lowe et al., 2017 - "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
    https://arxiv.org/abs/1706.02275

Author: H3C-BEACON Research Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_env import BaseEnv, PettingZooWrapper


class SimpleSpreadEnv(BaseEnv):
    """
    Built-in Simple Spread implementation.
    
    Used as fallback when PettingZoo is not available.
    Matches PettingZoo simple_spread_v3 behavior.
    """
    
    def __init__(self, n_agents: int = 3, max_steps: int = 25):
        # Observation: [vel(2), pos(2), landmark_rel(n*2), other_agents_rel((n-1)*2)]
        # For 3 agents: 2 + 2 + 6 + 4 = 14
        obs_dim = 2 + 2 + n_agents * 2 + (n_agents - 1) * 2
        act_dim = 5  # [no_action, left, right, down, up]
        
        super().__init__(n_agents, obs_dim, act_dim, max_steps)
        
        self.n_landmarks = n_agents
        
        # Physical parameters (matching PettingZoo)
        self.dt = 0.1
        self.damping = 0.25
        self.max_speed = 1.3
        self.agent_size = 0.15
        self.accel = 5.0
        
        # State
        self.agent_pos = None
        self.agent_vel = None
        self.landmark_pos = None
    
    def reset(self) -> List[np.ndarray]:
        self.current_step = 0
        
        # Random initial positions
        self.agent_pos = np.random.uniform(-1, 1, (self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2))
        self.landmark_pos = np.random.uniform(-1, 1, (self.n_landmarks, 2))
        
        return self._get_obs()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        self.current_step += 1
        
        # Apply actions
        for i, action in enumerate(actions):
            force = np.zeros(2)
            if action == 1:  # left
                force[0] = -self.accel
            elif action == 2:  # right
                force[0] = self.accel
            elif action == 3:  # down
                force[1] = -self.accel
            elif action == 4:  # up
                force[1] = self.accel
            
            # Update velocity with damping
            self.agent_vel[i] = self.agent_vel[i] * (1 - self.damping) + force * self.dt
            
            # Clip velocity
            speed = np.linalg.norm(self.agent_vel[i])
            if speed > self.max_speed:
                self.agent_vel[i] = self.agent_vel[i] / speed * self.max_speed
        
        # Update positions
        self.agent_pos += self.agent_vel * self.dt
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check done
        done = self.current_step >= self.max_steps
        dones = [done] * self.n_agents
        
        return self._get_obs(), rewards, dones, {'win': False}
    
    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for all agents."""
        obs = []
        
        for i in range(self.n_agents):
            agent_obs = []
            
            # Self velocity
            agent_obs.extend(self.agent_vel[i])
            
            # Self position
            agent_obs.extend(self.agent_pos[i])
            
            # Relative positions to all landmarks
            for landmark in self.landmark_pos:
                rel_pos = landmark - self.agent_pos[i]
                agent_obs.extend(rel_pos)
            
            # Relative positions to other agents
            for j in range(self.n_agents):
                if i != j:
                    rel_pos = self.agent_pos[j] - self.agent_pos[i]
                    agent_obs.extend(rel_pos)
            
            obs.append(np.array(agent_obs, dtype=np.float32))
        
        return obs
    
    def _compute_rewards(self) -> List[float]:
        """Compute cooperative reward based on landmark coverage."""
        total_reward = 0.0
        
        # Distance to landmarks
        for landmark in self.landmark_pos:
            min_dist = float('inf')
            for agent in self.agent_pos:
                dist = np.linalg.norm(landmark - agent)
                min_dist = min(min_dist, dist)
            total_reward -= min_dist
        
        # Collision penalty
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist < self.agent_size * 2:
                    total_reward -= 1.0
        
        # Shared reward
        return [total_reward] * self.n_agents
    
    def render(self, mode: str = 'human'):
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print("Agents:", self.agent_pos)
        print("Landmarks:", self.landmark_pos)


def make_simple_spread(n_agents: int = 3, max_steps: int = 25) -> BaseEnv:
    """
    Create Simple Spread environment.
    
    Uses PettingZoo if available, otherwise falls back to built-in.
    
    Args:
        n_agents: Number of agents (default: 3, standard benchmark)
        max_steps: Maximum steps per episode (default: 25)
    
    Returns:
        Environment instance
    """
    try:
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env(
            N=n_agents, 
            max_cycles=max_steps, 
            continuous_actions=False
        )
        print(f"  ✓ Using PettingZoo simple_spread_v3")
        return PettingZooWrapper(env)
    except ImportError:
        print(f"  ⚠️ PettingZoo not available, using built-in")
        return SimpleSpreadEnv(n_agents=n_agents, max_steps=max_steps)
    except Exception as e:
        print(f"  ⚠️ PettingZoo error: {e}, using built-in")
        return SimpleSpreadEnv(n_agents=n_agents, max_steps=max_steps)