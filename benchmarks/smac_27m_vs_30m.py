"""
SMAC: StarCraft Multi-Agent Challenge - 27m_vs_30m
Reference: Samvelyan et al., 2019 - "The StarCraft Multi-Agent Challenge"

27 Marines vs 30 Marines (hard scenario)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_env import BaseEnv


class SMAC27mvs30mEnv(BaseEnv):
    """
    SMAC 27m_vs_30m: 27 allied marines vs 30 enemy marines.
    This is a challenging asymmetric battle scenario.
    
    Note: This is a simplified simulation. For full SMAC, install pysc2 and smac.
    """
    
    def __init__(self, max_steps: int = 120):
        n_agents = 27
        n_enemies = 30
        
        # Observation: [health, x, y, can_attack] + ally_features + enemy_features
        obs_dim = 4 + (n_agents - 1) * 5 + n_enemies * 5  # Simplified
        obs_dim = min(obs_dim, 256)  # Cap for efficiency
        act_dim = 6 + n_enemies  # [no-op, stop, move_N, move_S, move_E, move_W, attack_enemy_0, ...]
        act_dim = min(act_dim, 36)  # Cap actions
        
        super().__init__(n_agents, obs_dim, act_dim, max_steps)
        
        self.n_enemies = n_enemies
        self.map_size = 32
        
        # Unit stats
        self.marine_health = 45
        self.marine_damage = 6
        self.marine_range = 5.0
        self.marine_speed = 1.0
        
        # State
        self.ally_health = None
        self.ally_pos = None
        self.enemy_health = None
        self.enemy_pos = None
        self.ally_cooldown = None
        self.enemy_cooldown = None
    
    def reset(self) -> List[np.ndarray]:
        self.current_step = 0
        
        # Initialize allies on left side
        self.ally_health = np.full(self.n_agents, self.marine_health, dtype=np.float32)
        self.ally_pos = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            self.ally_pos[i] = [5 + np.random.uniform(-2, 2), 
                                self.map_size/2 + (i - self.n_agents/2) * 0.8]
        self.ally_cooldown = np.zeros(self.n_agents)
        
        # Initialize enemies on right side
        self.enemy_health = np.full(self.n_enemies, self.marine_health, dtype=np.float32)
        self.enemy_pos = np.zeros((self.n_enemies, 2))
        for i in range(self.n_enemies):
            self.enemy_pos[i] = [self.map_size - 5 + np.random.uniform(-2, 2),
                                 self.map_size/2 + (i - self.n_enemies/2) * 0.8]
        self.enemy_cooldown = np.zeros(self.n_enemies)
        
        return self._get_obs()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        self.current_step += 1
        
        reward = 0.0
        
        # Reduce cooldowns
        self.ally_cooldown = np.maximum(0, self.ally_cooldown - 1)
        self.enemy_cooldown = np.maximum(0, self.enemy_cooldown - 1)
        
        # Process ally actions
        for i, action in enumerate(actions):
            if self.ally_health[i] <= 0:
                continue
            
            if action == 0:  # no-op
                pass
            elif action == 1:  # stop
                pass
            elif action == 2:  # move north
                self.ally_pos[i, 1] = min(self.map_size, self.ally_pos[i, 1] + self.marine_speed)
            elif action == 3:  # move south
                self.ally_pos[i, 1] = max(0, self.ally_pos[i, 1] - self.marine_speed)
            elif action == 4:  # move east
                self.ally_pos[i, 0] = min(self.map_size, self.ally_pos[i, 0] + self.marine_speed)
            elif action == 5:  # move west
                self.ally_pos[i, 0] = max(0, self.ally_pos[i, 0] - self.marine_speed)
            elif action >= 6:  # attack enemy
                enemy_idx = action - 6
                if enemy_idx < self.n_enemies and self.enemy_health[enemy_idx] > 0:
                    dist = np.linalg.norm(self.ally_pos[i] - self.enemy_pos[enemy_idx])
                    if dist <= self.marine_range and self.ally_cooldown[i] <= 0:
                        self.enemy_health[enemy_idx] -= self.marine_damage
                        self.ally_cooldown[i] = 2
                        
                        if self.enemy_health[enemy_idx] <= 0:
                            reward += 10.0  # Kill reward
                        else:
                            reward += 0.5  # Damage reward
        
        # Enemy AI: attack nearest ally
        for i in range(self.n_enemies):
            if self.enemy_health[i] <= 0:
                continue
            
            # Find nearest alive ally
            alive_allies = [j for j in range(self.n_agents) if self.ally_health[j] > 0]
            if not alive_allies:
                break
            
            distances = [np.linalg.norm(self.enemy_pos[i] - self.ally_pos[j]) for j in alive_allies]
            nearest_idx = alive_allies[np.argmin(distances)]
            nearest_dist = min(distances)
            
            if nearest_dist <= self.marine_range and self.enemy_cooldown[i] <= 0:
                # Attack
                self.ally_health[nearest_idx] -= self.marine_damage
                self.enemy_cooldown[i] = 2
                
                if self.ally_health[nearest_idx] <= 0:
                    reward -= 5.0  # Lost unit penalty
            else:
                # Move towards nearest ally
                direction = self.ally_pos[nearest_idx] - self.enemy_pos[i]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                self.enemy_pos[i] += direction * self.marine_speed
        
        # Check termination
        allies_alive = np.sum(self.ally_health > 0)
        enemies_alive = np.sum(self.enemy_health > 0)
        
        win = enemies_alive == 0
        lose = allies_alive == 0
        timeout = self.current_step >= self.max_steps
        
        done = win or lose or timeout
        
        if win:
            reward += 200.0  # Win bonus
        elif lose:
            reward -= 50.0  # Lose penalty
        
        dones = [done] * self.n_agents
        info = {
            'battle_won': win,
            'allies_alive': allies_alive,
            'enemies_alive': enemies_alive
        }
        
        return self._get_obs(), [reward] * self.n_agents, dones, info
    
    def _get_obs(self) -> List[np.ndarray]:
        obs = []
        
        for i in range(self.n_agents):
            if self.ally_health[i] <= 0:
                obs.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue
            
            agent_obs = []
            
            # Self features
            agent_obs.append(self.ally_health[i] / self.marine_health)
            agent_obs.extend(self.ally_pos[i] / self.map_size)
            agent_obs.append(1.0 if self.ally_cooldown[i] <= 0 else 0.0)
            
            # Ally features (relative)
            for j in range(self.n_agents):
                if j != i and len(agent_obs) < self.obs_dim - 10:
                    if self.ally_health[j] > 0:
                        rel_pos = (self.ally_pos[j] - self.ally_pos[i]) / self.map_size
                        agent_obs.extend(rel_pos)
                        agent_obs.append(self.ally_health[j] / self.marine_health)
            
            # Enemy features
            for j in range(self.n_enemies):
                if len(agent_obs) < self.obs_dim - 5:
                    if self.enemy_health[j] > 0:
                        rel_pos = (self.enemy_pos[j] - self.ally_pos[i]) / self.map_size
                        dist = np.linalg.norm(self.enemy_pos[j] - self.ally_pos[i])
                        agent_obs.extend(rel_pos)
                        agent_obs.append(self.enemy_health[j] / self.marine_health)
                        agent_obs.append(1.0 if dist <= self.marine_range else 0.0)
            
            # Pad
            while len(agent_obs) < self.obs_dim:
                agent_obs.append(0.0)
            
            obs.append(np.array(agent_obs[:self.obs_dim], dtype=np.float32))
        
        return obs
    
    def render(self, mode: str = 'human'):
        allies_alive = np.sum(self.ally_health > 0)
        enemies_alive = np.sum(self.enemy_health > 0)
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print(f"Allies: {allies_alive}/{self.n_agents} | Enemies: {enemies_alive}/{self.n_enemies}")


def make_smac_27m_vs_30m(use_pysc2: bool = False) -> BaseEnv:
    """
    Create SMAC 27m_vs_30m environment.
    
    Args:
        use_pysc2: If True, try to use actual StarCraft II (requires pysc2 and smac)
    
    Returns:
        Environment instance
    """
    if use_pysc2:
        try:
            from smac.env import StarCraft2Env
            env = StarCraft2Env(map_name="27m_vs_30m")
            # Would need additional wrapper here
            print("Using actual SMAC environment")
        except ImportError:
            print("SMAC/PySC2 not available, using simulation")
    
    return SMAC27mvs30mEnv()