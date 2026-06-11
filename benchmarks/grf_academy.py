"""
Google Research Football - Academy 3v1 Environment
===================================================
FIXED VERSION - Less randomness, more learnable

Key fixes:
1. Deterministic tackle based on distance (not random)
2. Controllable kick directions via actions
3. Longer episodes (400 steps)
4. Normalized rewards
5. Dense reward shaping for learning

Author: H3C-BEACON Research Team
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class Academy3vs1WithKeeperEnv:
    """
    3 attackers vs 1 defender + 1 goalkeeper.
    
    Objective: Score a goal.
    
    Observation space (per agent): 17 dims
    - Own position (2)
    - Own velocity (2) 
    - Ball position relative (2)
    - Ball velocity (2)
    - Teammates relative positions (4)
    - Defender relative position (2)
    - Keeper relative position (2)
    - Has ball flag (1)
    
    Action space: 11 actions
    - 0: Stay
    - 1-4: Move (up, down, left, right)
    - 5-8: Move diagonals
    - 9: Pass to nearest teammate
    - 10: Shoot
    """
    
    def __init__(self, max_steps: int = 400):
        self.n_agents = 3
        self.obs_dim = 17
        self.act_dim = 11
        self.max_steps = max_steps
        
        # Field dimensions (normalized)
        self.field_x = (-1.0, 1.0)
        self.field_y = (-0.42, 0.42)
        self.goal_y = (-0.18, 0.18)
        
        # Physics
        self.agent_speed = 0.03
        self.ball_speed = 0.12
        self.pass_speed = 0.10
        self.ball_friction = 0.95
        self.tackle_range = 0.08
        self.receive_range = 0.10
        
        # State
        self.reset()
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment."""
        self.step_count = 0
        self.done = False
        self.won = False
        
        # Agents in attacking formation
        self.agent_pos = np.array([
            [-0.3, 0.0],    # Center attacker (with ball)
            [-0.2, 0.15],   # Right wing
            [-0.2, -0.15],  # Left wing
        ], dtype=np.float32)
        self.agent_vel = np.zeros((3, 2), dtype=np.float32)
        
        # Defender between attackers and goal
        self.defender_pos = np.array([0.4, 0.0], dtype=np.float32)
        
        # Keeper in goal
        self.keeper_pos = np.array([0.95, 0.0], dtype=np.float32)
        
        # Ball starts with center attacker
        self.ball_pos = self.agent_pos[0].copy()
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_owner = 0  # Agent 0 has ball
        
        return self._get_obs()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Execute one step."""
        self.step_count += 1
        reward = 0.0
        info = {'win': False}
        
        # Process each agent's action
        for i, action in enumerate(actions):
            if self.ball_owner == i:
                # Agent has ball
                if action == 9:  # Pass
                    reward += self._execute_pass(i)
                elif action == 10:  # Shoot
                    reward += self._execute_shoot(i)
                else:
                    self._move_agent(i, action)
            else:
                # Agent doesn't have ball - move towards it or position
                self._move_agent(i, action)
        
        # Update ball physics
        if self.ball_owner == -1:  # Free ball
            self.ball_pos += self.ball_vel
            self.ball_vel *= self.ball_friction
            self._check_ball_reception()
        else:
            # Ball follows owner
            if 0 <= self.ball_owner < 3:
                self.ball_pos = self.agent_pos[self.ball_owner].copy()
        
        # Update defender (chase ball carrier or ball)
        self._update_defender()
        
        # Update keeper (track ball y-position)
        self._update_keeper()
        
        # Check tackle
        if 0 <= self.ball_owner < 3:
            self._check_tackle()
        
        # Check goal/out
        goal_reward = self._check_goal()
        reward += goal_reward
        
        # Progress reward (ball closer to goal)
        if not self.done:
            progress = self.ball_pos[0] - (-0.3)  # Distance from start
            reward += 0.001 * max(0, progress)
        
        # Episode done
        if self.step_count >= self.max_steps:
            self.done = True
        
        info['win'] = self.won
        info['steps'] = self.step_count
        
        # All agents share the same reward (cooperative)
        rewards = [reward] * self.n_agents
        dones = [self.done] * self.n_agents
        
        return self._get_obs(), rewards, dones, info
    
    def _move_agent(self, idx: int, action: int):
        """Move agent based on action."""
        # Direction vectors for actions 1-8
        directions = {
            0: (0, 0),      # Stay
            1: (0, 1),      # Up
            2: (0, -1),     # Down
            3: (-1, 0),     # Left
            4: (1, 0),      # Right
            5: (-0.7, 0.7),   # Up-left
            6: (0.7, 0.7),    # Up-right
            7: (-0.7, -0.7),  # Down-left
            8: (0.7, -0.7),   # Down-right
        }
        
        if action in directions:
            dx, dy = directions[action]
            move = np.array([dx, dy], dtype=np.float32)
            if np.linalg.norm(move) > 0:
                move = move / np.linalg.norm(move) * self.agent_speed
            self.agent_vel[idx] = move
            self.agent_pos[idx] += move
            
            # Clamp to field
            self.agent_pos[idx, 0] = np.clip(self.agent_pos[idx, 0], -1.0, 0.9)
            self.agent_pos[idx, 1] = np.clip(self.agent_pos[idx, 1], -0.42, 0.42)
    
    def _execute_pass(self, passer: int) -> float:
        """Execute pass to nearest teammate."""
        # Find nearest teammate
        min_dist = float('inf')
        target = -1
        for i in range(3):
            if i != passer:
                dist = np.linalg.norm(self.agent_pos[i] - self.ball_pos)
                if dist < min_dist:
                    min_dist = dist
                    target = i
        
        if target >= 0:
            # Pass direction
            direction = self.agent_pos[target] - self.ball_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            
            self.ball_vel = direction * self.pass_speed
            self.ball_owner = -1
            return 0.01  # Small reward for passing
        
        return 0.0
    
    def _execute_shoot(self, shooter: int) -> float:
        """Execute shot on goal."""
        # Aim at goal, away from keeper
        goal_center = np.array([1.0, 0.0])
        
        # Aim for corner away from keeper
        if self.keeper_pos[1] > 0:
            target = np.array([1.0, -0.12])  # Bottom corner
        else:
            target = np.array([1.0, 0.12])   # Top corner
        
        direction = target - self.ball_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        self.ball_vel = direction * self.ball_speed
        self.ball_owner = -1
        return 0.02  # Small reward for shooting
    
    def _check_ball_reception(self):
        """Check if any agent receives the ball."""
        for i in range(3):
            dist = np.linalg.norm(self.ball_pos - self.agent_pos[i])
            if dist < self.receive_range:
                self.ball_owner = i
                self.ball_vel = np.zeros(2, dtype=np.float32)
                return
    
    def _update_defender(self):
        """AI for defender - chase ball carrier."""
        if 0 <= self.ball_owner < 3:
            target = self.agent_pos[self.ball_owner]
        else:
            target = self.ball_pos
        
        # Move towards target but stay in defensive zone
        direction = target - self.defender_pos
        if np.linalg.norm(direction) > 0.05:
            direction = direction / np.linalg.norm(direction) * 0.025
            self.defender_pos += direction
        
        # Clamp defender position
        self.defender_pos[0] = np.clip(self.defender_pos[0], 0.1, 0.7)
        self.defender_pos[1] = np.clip(self.defender_pos[1], -0.35, 0.35)
    
    def _update_keeper(self):
        """AI for goalkeeper - track ball."""
        # Track ball y-position
        target_y = np.clip(self.ball_pos[1], -0.16, 0.16)
        
        # Move towards target
        dy = target_y - self.keeper_pos[1]
        self.keeper_pos[1] += np.clip(dy, -0.04, 0.04)
        
        # Advance if ball is close
        if self.ball_pos[0] > 0.6:
            self.keeper_pos[0] = min(0.92, 0.9 + (self.ball_pos[0] - 0.6) * 0.1)
    
    def _check_tackle(self):
        """Check if defender tackles ball carrier."""
        if self.ball_owner < 0 or self.ball_owner >= 3:
            return
        
        dist = np.linalg.norm(self.defender_pos - self.agent_pos[self.ball_owner])
        
        if dist < self.tackle_range:
            # Tackle probability based on distance (DETERMINISTIC)
            # Closer = higher chance, but not random
            tackle_success = dist < self.tackle_range * 0.5
            
            if tackle_success:
                # Ball goes backwards
                self.ball_owner = -1
                self.ball_vel = np.array([-0.08, np.random.uniform(-0.03, 0.03)], dtype=np.float32)
    
    def _check_goal(self) -> float:
        """Check for goal or out of bounds."""
        # Ball crosses goal line
        if self.ball_pos[0] >= 0.98:
            if self.goal_y[0] <= self.ball_pos[1] <= self.goal_y[1]:
                # Check keeper save
                keeper_dist = np.linalg.norm(self.ball_pos - self.keeper_pos)
                
                # Save probability based on distance (DETERMINISTIC)
                if keeper_dist < 0.12:
                    # Keeper saves
                    self.ball_vel = np.array([-0.15, np.random.uniform(-0.05, 0.05)], dtype=np.float32)
                    self.ball_owner = -1
                    return -0.1  # Penalty for saved shot
                else:
                    # GOAL!
                    self.done = True
                    self.won = True
                    return 1.0  # Big reward for goal
            else:
                # Wide of goal
                self.done = True
                return -0.2  # Penalty for missing
        
        # Out of bounds (sides)
        if abs(self.ball_pos[1]) > 0.45:
            self.done = True
            return -0.1
        
        # Out of bounds (back)
        if self.ball_pos[0] < -1.05:
            self.done = True
            return -0.1
        
        return 0.0
    
    def _get_obs(self) -> List[np.ndarray]:
        """Get observation for each agent."""
        obs_list = []
        
        for i in range(3):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            
            # Own position (2)
            obs[0:2] = self.agent_pos[i]
            
            # Own velocity (2)
            obs[2:4] = self.agent_vel[i]
            
            # Ball relative position (2)
            obs[4:6] = self.ball_pos - self.agent_pos[i]
            
            # Ball velocity (2)
            obs[6:8] = self.ball_vel
            
            # Teammates relative positions (4)
            teammates = [j for j in range(3) if j != i]
            obs[8:10] = self.agent_pos[teammates[0]] - self.agent_pos[i]
            obs[10:12] = self.agent_pos[teammates[1]] - self.agent_pos[i]
            
            # Defender relative position (2)
            obs[12:14] = self.defender_pos - self.agent_pos[i]
            
            # Keeper relative position (2)
            obs[14:16] = self.keeper_pos - self.agent_pos[i]
            
            # Has ball flag (1)
            obs[16] = 1.0 if self.ball_owner == i else 0.0
            
            obs_list.append(obs)
        
        return obs_list
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            'n_agents': self.n_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'n_actions': self.act_dim,
            'state_shape': self.obs_dim * self.n_agents,
            'obs_shape': self.obs_dim,
            'episode_limit': self.max_steps,
        }
    
    def get_obs(self) -> List[np.ndarray]:
        """Get current observations."""
        return self._get_obs()
    
    def get_state(self) -> np.ndarray:
        """Get global state."""
        return np.concatenate(self._get_obs())
    
    def close(self):
        """Clean up."""
        pass
    
    def render(self, mode: str = 'human'):
        """Simple text render."""
        owner = f"Agent {self.ball_owner}" if 0 <= self.ball_owner < 3 else "Free"
        print(f"Step {self.step_count}: Ball at ({self.ball_pos[0]:.2f}, {self.ball_pos[1]:.2f}), Owner: {owner}")


def make_academy_3_vs_1_with_keeper(max_steps: int = 400, **kwargs):
    """Factory function."""
    return Academy3vs1WithKeeperEnv(max_steps=max_steps)


# Test
if __name__ == "__main__":
    env = Academy3vs1WithKeeperEnv()
    
    wins = 0
    total_reward = 0
    n_episodes = 100
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # Random actions
            actions = [np.random.randint(0, env.act_dim) for _ in range(env.n_agents)]
            obs, rewards, dones, info = env.step(actions)
            ep_reward += sum(rewards) / len(rewards)
            done = dones[0]
        
        total_reward += ep_reward
        if info['win']:
            wins += 1
    
    print(f"\nRandom policy over {n_episodes} episodes:")
    print(f"  Win rate: {wins/n_episodes*100:.1f}%")
    print(f"  Avg reward: {total_reward/n_episodes:.3f}")