"""
SMACLite - StarCraft Multi-Agent Challenge Lite
================================================

Simulateur léger pour SMAC sans dépendance à StarCraft II.
Permet de tester les algorithmes MARL sur des scénarios SMAC.

Scenarios supportés:
- 3m: 3 Marines vs 3 Marines
- 8m: 8 Marines vs 8 Marines  
- 2s3z: 2 Stalkers + 3 Zealots vs 2S+3Z
- 3s5z: 3 Stalkers + 5 Zealots vs 3S+5Z
- 27m_vs_30m: 27 Marines vs 30 Marines (super hard)

Authors: Basile BETE MBEZELE, Ghislain ALO'O ABESSOLO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces


# ============================================================
# UNIT TYPES
# ============================================================

class UnitType:
    """Unit type definitions matching SMAC."""
    MARINE = 'marine'
    ZEALOT = 'zealot'
    STALKER = 'stalker'
    
    # Unit stats
    STATS = {
        'marine': {
            'hp': 45,
            'damage': 6,
            'range': 5,
            'speed': 1.0,
            'cooldown': 15,
        },
        'zealot': {
            'hp': 100,
            'shield': 50,
            'damage': 8,
            'range': 1,  # Melee
            'speed': 0.8,
            'cooldown': 22,
        },
        'stalker': {
            'hp': 80,
            'shield': 80,
            'damage': 10,
            'range': 6,
            'speed': 1.2,
            'cooldown': 20,
        },
    }


# ============================================================
# UNIT CLASS
# ============================================================

class Unit:
    """Individual unit in the simulation."""
    
    def __init__(self, unit_id: int, unit_type: str, team: int, 
                 position: np.ndarray):
        self.id = unit_id
        self.type = unit_type
        self.team = team  # 0 = ally, 1 = enemy
        self.position = position.astype(np.float32)
        
        stats = UnitType.STATS[unit_type]
        self.max_hp = stats['hp']
        self.hp = self.max_hp
        self.shield = stats.get('shield', 0)
        self.max_shield = self.shield
        self.damage = stats['damage']
        self.range = stats['range']
        self.speed = stats['speed']
        self.cooldown_max = stats['cooldown']
        self.cooldown = 0
        
        self.alive = True
        self.last_action = 0
        
    def reset(self, position: np.ndarray):
        """Reset unit to initial state."""
        self.position = position.astype(np.float32)
        self.hp = self.max_hp
        self.shield = self.max_shield
        self.cooldown = 0
        self.alive = True
        self.last_action = 0
        
    def take_damage(self, damage: float) -> float:
        """Apply damage, return actual damage dealt."""
        if not self.alive:
            return 0
        
        actual_damage = 0
        
        # Shield absorbs first
        if self.shield > 0:
            shield_damage = min(self.shield, damage)
            self.shield -= shield_damage
            damage -= shield_damage
            actual_damage += shield_damage
        
        # Then HP
        if damage > 0:
            hp_damage = min(self.hp, damage)
            self.hp -= hp_damage
            actual_damage += hp_damage
            
            if self.hp <= 0:
                self.alive = False
                self.hp = 0
        
        return actual_damage
    
    def can_attack(self) -> bool:
        """Check if unit can attack."""
        return self.alive and self.cooldown == 0
    
    def attack(self, target: 'Unit') -> float:
        """Attack target, return damage dealt."""
        if not self.can_attack():
            return 0
        
        distance = np.linalg.norm(self.position - target.position)
        if distance > self.range:
            return 0
        
        damage = self.damage
        self.cooldown = self.cooldown_max
        
        return target.take_damage(damage)
    
    def move(self, direction: np.ndarray, map_size: Tuple[float, float]):
        """Move unit in direction."""
        if not self.alive:
            return
        
        new_pos = self.position + direction * self.speed
        
        # Clamp to map bounds
        new_pos[0] = np.clip(new_pos[0], 0, map_size[0])
        new_pos[1] = np.clip(new_pos[1], 0, map_size[1])
        
        self.position = new_pos
    
    def update(self):
        """Update unit state (cooldown tick)."""
        if self.cooldown > 0:
            self.cooldown -= 1


# ============================================================
# SMAC LITE ENVIRONMENT
# ============================================================

class SMACLiteEnv:
    """
    Lightweight SMAC environment simulator.
    
    Action space (per agent):
    - 0: No-op / Dead
    - 1: Stop
    - 2: Move North
    - 3: Move South
    - 4: Move East
    - 5: Move West
    - 6+: Attack enemy i (i = action - 6)
    """
    
    # Map configurations
    MAP_CONFIGS = {
        '3m': {
            'ally_units': [('marine', 3)],
            'enemy_units': [('marine', 3)],
            'map_size': (32, 32),
            'episode_limit': 60,
        },
        '8m': {
            'ally_units': [('marine', 8)],
            'enemy_units': [('marine', 8)],
            'map_size': (32, 32),
            'episode_limit': 120,
        },
        '2s3z': {
            'ally_units': [('stalker', 2), ('zealot', 3)],
            'enemy_units': [('stalker', 2), ('zealot', 3)],
            'map_size': (32, 32),
            'episode_limit': 120,
        },
        '3s5z': {
            'ally_units': [('stalker', 3), ('zealot', 5)],
            'enemy_units': [('stalker', 3), ('zealot', 5)],
            'map_size': (32, 32),
            'episode_limit': 150,
        },
        '27m_vs_30m': {
            'ally_units': [('marine', 27)],
            'enemy_units': [('marine', 30)],
            'map_size': (64, 64),
            'episode_limit': 180,
        },
        '5m_vs_6m': {
            'ally_units': [('marine', 5)],
            'enemy_units': [('marine', 6)],
            'map_size': (32, 32),
            'episode_limit': 70,
        },
    }
    
    def __init__(self, map_name: str = '3m', seed: int = None,
                 reward_sparse: bool = False,
                 reward_death_value: float = 10.0,
                 reward_win: float = 200.0):
        
        if map_name not in self.MAP_CONFIGS:
            raise ValueError(f"Unknown map: {map_name}. Available: {list(self.MAP_CONFIGS.keys())}")
        
        self.map_name = map_name
        self.config = self.MAP_CONFIGS[map_name]
        self.map_size = self.config['map_size']
        self.episode_limit = self.config['episode_limit']
        
        # Reward shaping
        self.reward_sparse = reward_sparse
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        
        # Random seed
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Create units
        self.allies = []
        self.enemies = []
        self._create_units()
        
        # Dimensions
        self.n_agents = len(self.allies)
        self.n_enemies = len(self.enemies)
        self.n_actions = 6 + self.n_enemies  # no-op, stop, 4 moves, n_enemies attacks
        
        # Observation dimension
        # Per ally: [alive, hp, shield, cooldown, rel_x, rel_y] + enemy features
        # Enemy features: [alive, hp, shield, rel_x, rel_y, distance, in_range]
        self.ally_feats_dim = 6
        self.enemy_feats_dim = 7
        self.obs_dim = self.ally_feats_dim + self.n_enemies * self.enemy_feats_dim
        
        # State dimension (global state)
        self.state_dim = (self.n_agents * self.ally_feats_dim + 
                         self.n_enemies * self.enemy_feats_dim)
        
        # Episode state
        self.steps = 0
        self.battles_won = 0
        self.battles_game = 0
        
    def _create_units(self):
        """Create ally and enemy units."""
        self.allies = []
        self.enemies = []
        
        unit_id = 0
        
        # Allies
        for unit_type, count in self.config['ally_units']:
            for i in range(count):
                pos = self._random_position(team=0)
                unit = Unit(unit_id, unit_type, team=0, position=pos)
                self.allies.append(unit)
                unit_id += 1
        
        # Enemies
        for unit_type, count in self.config['enemy_units']:
            for i in range(count):
                pos = self._random_position(team=1)
                unit = Unit(unit_id, unit_type, team=1, position=pos)
                self.enemies.append(unit)
                unit_id += 1
    
    def _random_position(self, team: int) -> np.ndarray:
        """Generate random starting position."""
        if team == 0:  # Allies start on left
            x = self.rng.uniform(2, self.map_size[0] * 0.3)
        else:  # Enemies start on right
            x = self.rng.uniform(self.map_size[0] * 0.7, self.map_size[0] - 2)
        
        y = self.rng.uniform(2, self.map_size[1] - 2)
        return np.array([x, y], dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.steps = 0
        
        # Reset unit positions and states
        for unit in self.allies:
            unit.reset(self._random_position(team=0))
        
        for unit in self.enemies:
            unit.reset(self._random_position(team=1))
        
        return self.get_obs()
    
    def get_obs(self) -> np.ndarray:
        """Get observations for all agents."""
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        
        for i, ally in enumerate(self.allies):
            obs[i] = self._get_agent_obs(ally)
        
        return obs
    
    def _get_agent_obs(self, agent: Unit) -> np.ndarray:
        """Get observation for single agent."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        if not agent.alive:
            return obs
        
        # Self features
        obs[0] = 1.0  # alive
        obs[1] = agent.hp / agent.max_hp
        obs[2] = agent.shield / max(agent.max_shield, 1)
        obs[3] = agent.cooldown / agent.cooldown_max
        obs[4] = agent.position[0] / self.map_size[0]
        obs[5] = agent.position[1] / self.map_size[1]
        
        # Enemy features
        idx = self.ally_feats_dim
        for enemy in self.enemies:
            if enemy.alive:
                obs[idx] = 1.0  # alive
                obs[idx + 1] = enemy.hp / enemy.max_hp
                obs[idx + 2] = enemy.shield / max(enemy.max_shield, 1)
                
                # Relative position
                rel_pos = enemy.position - agent.position
                obs[idx + 3] = rel_pos[0] / self.map_size[0]
                obs[idx + 4] = rel_pos[1] / self.map_size[1]
                
                # Distance and in-range
                distance = np.linalg.norm(rel_pos)
                obs[idx + 5] = distance / np.linalg.norm(self.map_size)
                obs[idx + 6] = 1.0 if distance <= agent.range else 0.0
            
            idx += self.enemy_feats_dim
        
        return obs
    
    def get_state(self) -> np.ndarray:
        """Get global state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        idx = 0
        
        # Ally states
        for ally in self.allies:
            if ally.alive:
                state[idx] = 1.0
                state[idx + 1] = ally.hp / ally.max_hp
                state[idx + 2] = ally.shield / max(ally.max_shield, 1)
                state[idx + 3] = ally.cooldown / ally.cooldown_max
                state[idx + 4] = ally.position[0] / self.map_size[0]
                state[idx + 5] = ally.position[1] / self.map_size[1]
            idx += self.ally_feats_dim
        
        # Enemy states
        for enemy in self.enemies:
            if enemy.alive:
                state[idx] = 1.0
                state[idx + 1] = enemy.hp / enemy.max_hp
                state[idx + 2] = enemy.shield / max(enemy.max_shield, 1)
                state[idx + 3] = 0  # No cooldown for enemies in state
                state[idx + 4] = enemy.position[0] / self.map_size[0]
                state[idx + 5] = enemy.position[1] / self.map_size[1]
            idx += self.enemy_feats_dim
        
        return state
    
    def get_avail_actions(self) -> np.ndarray:
        """Get available actions for all agents."""
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        
        for i, ally in enumerate(self.allies):
            avail[i] = self._get_agent_avail_actions(ally)
        
        return avail
    
    def _get_agent_avail_actions(self, agent: Unit) -> np.ndarray:
        """Get available actions for single agent."""
        avail = np.zeros(self.n_actions, dtype=np.float32)
        
        if not agent.alive:
            avail[0] = 1  # Only no-op
            return avail
        
        # Always available
        avail[0] = 1  # No-op
        avail[1] = 1  # Stop
        
        # Movement (check bounds)
        if agent.position[1] < self.map_size[1] - 1:
            avail[2] = 1  # North
        if agent.position[1] > 1:
            avail[3] = 1  # South
        if agent.position[0] < self.map_size[0] - 1:
            avail[4] = 1  # East
        if agent.position[0] > 1:
            avail[5] = 1  # West
        
        # Attack actions
        if agent.can_attack():
            for j, enemy in enumerate(self.enemies):
                if enemy.alive:
                    distance = np.linalg.norm(agent.position - enemy.position)
                    if distance <= agent.range:
                        avail[6 + j] = 1
        
        return avail
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute actions and return (obs, rewards, dones, info).
        """
        self.steps += 1
        
        total_damage_dealt = 0
        total_damage_received = 0
        enemies_killed = 0
        allies_killed = 0
        
        # Process ally actions
        for i, (ally, action) in enumerate(zip(self.allies, actions)):
            if not ally.alive:
                continue
            
            action = int(action)
            ally.last_action = action
            
            if action == 0:  # No-op
                pass
            elif action == 1:  # Stop
                pass
            elif action == 2:  # Move North
                ally.move(np.array([0, 1]), self.map_size)
            elif action == 3:  # Move South
                ally.move(np.array([0, -1]), self.map_size)
            elif action == 4:  # Move East
                ally.move(np.array([1, 0]), self.map_size)
            elif action == 5:  # Move West
                ally.move(np.array([-1, 0]), self.map_size)
            elif action >= 6:  # Attack
                target_idx = action - 6
                if target_idx < len(self.enemies):
                    target = self.enemies[target_idx]
                    if target.alive:
                        was_alive = target.alive
                        damage = ally.attack(target)
                        total_damage_dealt += damage
                        if was_alive and not target.alive:
                            enemies_killed += 1
        
        # Enemy AI (simple attack closest)
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            
            # Find closest alive ally
            closest_ally = None
            closest_dist = float('inf')
            
            for ally in self.allies:
                if ally.alive:
                    dist = np.linalg.norm(enemy.position - ally.position)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_ally = ally
            
            if closest_ally is not None:
                if closest_dist <= enemy.range and enemy.can_attack():
                    # Attack
                    was_alive = closest_ally.alive
                    damage = enemy.attack(closest_ally)
                    total_damage_received += damage
                    if was_alive and not closest_ally.alive:
                        allies_killed += 1
                else:
                    # Move towards
                    direction = closest_ally.position - enemy.position
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    enemy.move(direction, self.map_size)
        
        # Update cooldowns
        for unit in self.allies + self.enemies:
            unit.update()
        
        # Check terminal conditions
        all_allies_dead = all(not ally.alive for ally in self.allies)
        all_enemies_dead = all(not enemy.alive for enemy in self.enemies)
        timeout = self.steps >= self.episode_limit
        
        terminated = all_allies_dead or all_enemies_dead or timeout
        
        # Determine winner
        battle_won = all_enemies_dead and not all_allies_dead
        
        if terminated:
            self.battles_game += 1
            if battle_won:
                self.battles_won += 1
        
        # Compute rewards
        if self.reward_sparse:
            if battle_won:
                reward = self.reward_win
            elif all_allies_dead:
                reward = -self.reward_win
            else:
                reward = 0
        else:
            # Shaped reward
            reward = total_damage_dealt * 0.1  # Damage dealt
            reward -= total_damage_received * 0.05  # Damage received
            reward += enemies_killed * self.reward_death_value  # Kill bonus
            reward -= allies_killed * self.reward_death_value * 0.5  # Death penalty
            
            if battle_won:
                reward += self.reward_win
        
        # Per-agent rewards (shared)
        rewards = np.full(self.n_agents, reward / self.n_agents, dtype=np.float32)
        
        # Dones
        dones = np.full(self.n_agents, terminated, dtype=np.float32)
        
        # Info
        info = {
            'battle_won': battle_won,
            'enemies_killed': enemies_killed,
            'allies_killed': allies_killed,
            'damage_dealt': total_damage_dealt,
            'damage_received': total_damage_received,
            'steps': self.steps,
            'win_rate': self.battles_won / max(self.battles_game, 1),
        }
        
        return self.get_obs(), rewards, dones, info
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_env_info(self) -> Dict:
        """Get environment information."""
        return {
            'n_agents': self.n_agents,
            'n_enemies': self.n_enemies,
            'n_actions': self.n_actions,
            'obs_shape': self.obs_dim,
            'state_shape': self.state_dim,
            'episode_limit': self.episode_limit,
            'map_name': self.map_name,
        }


# ============================================================
# FACTORY FUNCTION
# ============================================================

def make_smac_env(map_name: str = '3m', seed: int = None, **kwargs) -> SMACLiteEnv:
    """
    Factory function to create SMAC environment.
    
    Args:
        map_name: Map name (3m, 8m, 2s3z, 3s5z, 27m_vs_30m)
        seed: Random seed
        **kwargs: Additional arguments
        
    Returns:
        SMACLiteEnv instance
    """
    # Handle prefixed names
    if map_name.startswith('smac_'):
        map_name = map_name[5:]
    
    return SMACLiteEnv(map_name=map_name, seed=seed, **kwargs)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing SMACLite Environment")
    print("=" * 50)
    
    for map_name in ['3m', '8m', '27m_vs_30m']:
        print(f"\nTesting map: {map_name}")
        
        env = make_smac_env(map_name, seed=42)
        info = env.get_env_info()
        print(f"  Agents: {info['n_agents']}, Enemies: {info['n_enemies']}")
        print(f"  Obs dim: {info['obs_shape']}, Actions: {info['n_actions']}")
        
        # Run episode
        obs = env.reset()
        total_reward = 0
        
        for step in range(100):
            avail = env.get_avail_actions()
            
            # Random valid actions
            actions = []
            for i in range(env.n_agents):
                valid_actions = np.where(avail[i] == 1)[0]
                actions.append(np.random.choice(valid_actions))
            
            obs, rewards, dones, info = env.step(np.array(actions))
            total_reward += rewards.sum()
            
            if dones.all():
                break
        
        print(f"  Episode finished: steps={info['steps']}, reward={total_reward:.2f}, won={info['battle_won']}")
        env.close()
    
    print("\n✓ All tests passed!")
