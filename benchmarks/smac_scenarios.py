"""
StarCraft Multi-Agent Challenge (SMAC) - 27m_vs_30m
====================================================
Large-scale cooperative multi-agent scenario in StarCraft II.

Standard Benchmark:
- 27m_vs_30m: 27 Marines vs 30 enemy Marines

Reference:
    Samvelyan et al., 2019 - "The StarCraft Multi-Agent Challenge"
    https://arxiv.org/abs/1902.04043

Requires: pysc2, smac (pip install smac)

Author: H3C-BEACON Research Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_env import BaseEnv, SMACWrapper


class SMAC27mVs30mEnv(BaseEnv):
    """
    Built-in simulation of 27m_vs_30m scenario.
    
    27 Marines vs 30 enemy Marines.
    Large-scale scenario testing coordination and focus fire.
    """
    
    def __init__(self, max_steps: int = 120):
        self.n_allies = 27
        self.n_enemies = 30
        n_agents = self.n_allies
        
        # Observe 5 nearest allies and 5 nearest enemies
        self.n_observable = 5
        obs_dim = 4 + self.n_observable * 4 + self.n_observable * 4  # self + allies + enemies
        act_dim = 6 + self.n_observable  # no-op, stop, 4 moves, attack 5 enemies
        
        super().__init__(n_agents, obs_dim, act_dim, max_steps)
        
        # Marine stats
        self.marine_health_max = 45.0
        self.marine_damage = 6.0
        self.marine_range = 5.0
        self.marine_speed = 0.05
        
        # State
        self.ally_health = None
        self.ally_pos = None
        self.enemy_health = None
        self.enemy_pos = None
        
        self.win = False
    
    def reset(self) -> List[np.ndarray]:
        self.current_step = 0
        self.win = False
        
        # Initialize allies on left
        self.ally_health = np.full(self.n_allies, self.marine_health_max)
        self.ally_pos = np.zeros((self.n_allies, 2), dtype=np.float32)
        for i in range(self.n_allies):
            row = i // 9
            col = i % 9
            self.ally_pos[i] = [0.1 + col * 0.04, 0.2 + row * 0.1]
        
        # Initialize enemies on right
        self.enemy_health = np.full(self.n_enemies, self.marine_health_max)
        self.enemy_pos = np.zeros((self.n_enemies, 2), dtype=np.float32)
        for i in range(self.n_enemies):
            row = i // 10
            col = i % 10
            self.enemy_pos[i] = [0.6 + col * 0.04, 0.2 + row * 0.1]
        
        return self._get_obs()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        self.current_step += 1
        reward = 0.0
        
        # Process ally actions
        for i, action in enumerate(actions):
            if self.ally_health[i] <= 0:
                continue
            
            if action == 0:  # no-op
                pass
            elif action == 1:  # stop
                pass
            elif action == 2:  # move N
                self.ally_pos[i, 1] = min(1.0, self.ally_pos[i, 1] + self.marine_speed)
            elif action == 3:  # move S
                self.ally_pos[i, 1] = max(0.0, self.ally_pos[i, 1] - self.marine_speed)
            elif action == 4:  # move E
                self.ally_pos[i, 0] = min(1.0, self.ally_pos[i, 0] + self.marine_speed)
            elif action == 5:  # move W
                self.ally_pos[i, 0] = max(0.0, self.ally_pos[i, 0] - self.marine_speed)
            elif action >= 6:  # attack
                target_idx = action - 6
                nearest = self._get_nearest_enemies(i)
                if target_idx < len(nearest):
                    target = nearest[target_idx]
                    dist = np.linalg.norm(self.enemy_pos[target] - self.ally_pos[i])
                    if dist <= self.marine_range:
                        self.enemy_health[target] -= self.marine_damage
                        reward += self.marine_damage * 0.02
                        if self.enemy_health[target] <= 0:
                            reward += 0.5
        
        # Enemy behavior: attack nearest ally
        for i in range(self.n_enemies):
            if self.enemy_health[i] <= 0:
                continue
            
            alive_allies = [j for j in range(self.n_allies) if self.ally_health[j] > 0]
            if not alive_allies:
                continue
            
            distances = [np.linalg.norm(self.enemy_pos[i] - self.ally_pos[j]) for j in alive_allies]
            target = alive_allies[np.argmin(distances)]
            min_dist = min(distances)
            
            if min_dist <= self.marine_range:
                self.ally_health[target] -= self.marine_damage
                if self.ally_health[target] <= 0:
                    reward -= 0.5
            else:
                direction = self.ally_pos[target] - self.enemy_pos[i]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                self.enemy_pos[i] += direction * self.marine_speed
        
        # Check win/lose
        all_allies_dead = np.sum(self.ally_health > 0) == 0
        all_enemies_dead = np.sum(self.enemy_health > 0) == 0
        
        if all_enemies_dead:
            reward += 20.0
            self.win = True
        
        done = all_allies_dead or all_enemies_dead or self.current_step >= self.max_steps
        rewards = [reward / self.n_agents] * self.n_agents
        dones = [done] * self.n_agents
        
        return self._get_obs(), rewards, dones, {'win': self.win}
    
    def _get_nearest_enemies(self, ally_idx: int) -> List[int]:
        alive_mask = self.enemy_health > 0
        if not np.any(alive_mask):
            return []
        
        alive_indices = np.where(alive_mask)[0]
        distances = np.linalg.norm(self.enemy_pos[alive_indices] - self.ally_pos[ally_idx], axis=1)
        sorted_indices = alive_indices[np.argsort(distances)]
        return sorted_indices[:self.n_observable].tolist()
    
    def _get_obs(self) -> List[np.ndarray]:
        obs = []
        
        for i in range(self.n_allies):
            if self.ally_health[i] <= 0:
                obs.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue
            
            agent_obs = []
            
            # Self info
            agent_obs.append(self.ally_health[i] / self.marine_health_max)
            agent_obs.extend(self.ally_pos[i])
            agent_obs.append(1.0)  # Alive
            
            # Nearest allies
            alive_allies = [j for j in range(self.n_allies) if j != i and self.ally_health[j] > 0]
            if alive_allies:
                dists = [np.linalg.norm(self.ally_pos[i] - self.ally_pos[j]) for j in alive_allies]
                nearest_allies = [alive_allies[k] for k in np.argsort(dists)[:self.n_observable]]
            else:
                nearest_allies = []
            
            for k in range(self.n_observable):
                if k < len(nearest_allies):
                    j = nearest_allies[k]
                    agent_obs.append(self.ally_health[j] / self.marine_health_max)
                    agent_obs.extend(self.ally_pos[j] - self.ally_pos[i])
                    agent_obs.append(1.0)
                else:
                    agent_obs.extend([0, 0, 0, 0])
            
            # Nearest enemies
            nearest_enemies = self._get_nearest_enemies(i)
            for k in range(self.n_observable):
                if k < len(nearest_enemies):
                    j = nearest_enemies[k]
                    agent_obs.append(self.enemy_health[j] / self.marine_health_max)
                    agent_obs.extend(self.enemy_pos[j] - self.ally_pos[i])
                    agent_obs.append(1.0)
                else:
                    agent_obs.extend([0, 0, 0, 0])
            
            obs.append(np.array(agent_obs[:self.obs_dim], dtype=np.float32))
        
        return obs
    
    def render(self, mode: str = 'human'):
        alive_a = np.sum(self.ally_health > 0)
        alive_e = np.sum(self.enemy_health > 0)
        print(f"Step {self.current_step}: Allies: {alive_a}/{self.n_allies}, Enemies: {alive_e}/{self.n_enemies}")


def make_27m_vs_30m(max_steps: int = 120) -> BaseEnv:
    """
    Create 27m_vs_30m scenario.
    
    Uses real SMAC if available, otherwise built-in simulation.
    
    Args:
        max_steps: Maximum episode length (default: 120)
    
    Returns:
        Environment instance
    """
    try:
        from smac.env import StarCraft2Env
        env = StarCraft2Env(map_name="27m_vs_30m")
        print(f"  ✓ Using SMAC 27m_vs_30m")
        return SMACWrapper(env)
    except ImportError:
        print(f"  ⚠️ SMAC not available, using built-in simulation")
        return SMAC27mVs30mEnv(max_steps=max_steps)
    except Exception as e:
        print(f"  ⚠️ SMAC error: {e}, using built-in")
        return SMAC27mVs30mEnv(max_steps=max_steps)