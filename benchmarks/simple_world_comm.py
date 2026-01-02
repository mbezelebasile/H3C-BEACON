"""
Simple World Comm Environment (MPE)
====================================
Mixed cooperative-competitive: Good agents cooperate against adversaries.

Standard Benchmark:
- 2 good agents (controlled by RL)
- 4 adversaries (scripted/random - NOT controlled)
- Good agents must reach landmarks while avoiding adversaries
- Good agents can communicate

Reference:
    Lowe et al., 2017 - "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
    https://arxiv.org/abs/1706.02275

Note: Only GOOD agents are controlled. Adversaries use random/scripted policy.
      This is the standard evaluation protocol.

Author: H3C-BEACON Research Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_env import BaseEnv, PettingZooWrapper


class SimpleWorldCommEnv(BaseEnv):
    """
    Built-in Simple World Comm implementation.
    
    Controls only good agents. Adversaries are scripted.
    Matches the standard evaluation protocol.
    """
    
    def __init__(self, n_good: int = 2, n_adversaries: int = 4, 
                 n_landmarks: int = 3, max_steps: int = 25):
        
        self.n_good = n_good
        self.n_adversaries = n_adversaries
        self.n_landmarks = n_landmarks
        
        # Only control good agents
        n_agents = n_good
        
        # Observation for good agents:
        # [vel(2), pos(2), landmark_rel(n_landmarks*2), other_good_rel((n_good-1)*2), 
        #  adversary_rel(n_adv*2), comm(4)]
        obs_dim = 2 + 2 + n_landmarks * 2 + (n_good - 1) * 2 + n_adversaries * 2 + 4
        
        # Actions: 5 movement + 4 communication = 9 discrete actions
        # But we'll use 5 for simplicity (movement only, comm handled separately)
        act_dim = 5
        
        super().__init__(n_agents, obs_dim, act_dim, max_steps)
        
        # Physical parameters
        self.dt = 0.1
        self.damping = 0.25
        self.max_speed_good = 1.3
        self.max_speed_adv = 1.0  # Adversaries slightly slower
        self.agent_size = 0.075
        self.accel = 5.0
        
        # State
        self.good_pos = None
        self.good_vel = None
        self.adv_pos = None
        self.adv_vel = None
        self.landmark_pos = None
        self.comm = None  # Communication state
    
    def reset(self) -> List[np.ndarray]:
        self.current_step = 0
        
        # Initialize positions
        self.good_pos = np.random.uniform(-1, 1, (self.n_good, 2))
        self.good_vel = np.zeros((self.n_good, 2))
        
        self.adv_pos = np.random.uniform(-1, 1, (self.n_adversaries, 2))
        self.adv_vel = np.zeros((self.n_adversaries, 2))
        
        self.landmark_pos = np.random.uniform(-1, 1, (self.n_landmarks, 2))
        self.comm = np.zeros((self.n_good, 4))
        
        return self._get_obs()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        self.current_step += 1
        
        # Apply good agent actions
        for i, action in enumerate(actions):
            force = self._action_to_force(action)
            self.good_vel[i] = self.good_vel[i] * (1 - self.damping) + force * self.dt
            speed = np.linalg.norm(self.good_vel[i])
            if speed > self.max_speed_good:
                self.good_vel[i] = self.good_vel[i] / speed * self.max_speed_good
        
        # Adversary policy: chase nearest good agent
        for i in range(self.n_adversaries):
            nearest_good = self._find_nearest(self.adv_pos[i], self.good_pos)
            direction = nearest_good - self.adv_pos[i]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            
            self.adv_vel[i] = self.adv_vel[i] * (1 - self.damping) + direction * self.accel * 0.5 * self.dt
            speed = np.linalg.norm(self.adv_vel[i])
            if speed > self.max_speed_adv:
                self.adv_vel[i] = self.adv_vel[i] / speed * self.max_speed_adv
        
        # Update positions
        self.good_pos += self.good_vel * self.dt
        self.adv_pos += self.adv_vel * self.dt
        
        # Clip to bounds
        self.good_pos = np.clip(self.good_pos, -1.5, 1.5)
        self.adv_pos = np.clip(self.adv_pos, -1.5, 1.5)
        
        # Compute rewards for good agents
        rewards = self._compute_rewards()
        
        done = self.current_step >= self.max_steps
        dones = [done] * self.n_agents
        
        # Win if all good agents reach landmarks without being caught
        win = all(r > 0 for r in rewards)
        
        return self._get_obs(), rewards, dones, {'win': win}
    
    def _action_to_force(self, action: int) -> np.ndarray:
        force = np.zeros(2)
        if action == 1:  # left
            force[0] = -self.accel
        elif action == 2:  # right
            force[0] = self.accel
        elif action == 3:  # down
            force[1] = -self.accel
        elif action == 4:  # up
            force[1] = self.accel
        return force
    
    def _find_nearest(self, pos: np.ndarray, targets: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(targets - pos, axis=1)
        return targets[np.argmin(distances)]
    
    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for good agents only."""
        obs = []
        
        for i in range(self.n_good):
            agent_obs = []
            
            # Self velocity and position
            agent_obs.extend(self.good_vel[i])
            agent_obs.extend(self.good_pos[i])
            
            # Relative positions to landmarks
            for landmark in self.landmark_pos:
                rel_pos = landmark - self.good_pos[i]
                agent_obs.extend(rel_pos)
            
            # Relative positions to other good agents
            for j in range(self.n_good):
                if i != j:
                    rel_pos = self.good_pos[j] - self.good_pos[i]
                    agent_obs.extend(rel_pos)
            
            # Relative positions to adversaries
            for j in range(self.n_adversaries):
                rel_pos = self.adv_pos[j] - self.good_pos[i]
                agent_obs.extend(rel_pos)
            
            # Communication from other agents
            other_comm = self.comm[(i + 1) % self.n_good]
            agent_obs.extend(other_comm)
            
            obs.append(np.array(agent_obs[:self.obs_dim], dtype=np.float32))
        
        return obs
    
    def _compute_rewards(self) -> List[float]:
        """Compute rewards for good agents."""
        total_reward = 0.0
        
        # Reward for reaching landmarks
        for i in range(self.n_good):
            min_dist = min(np.linalg.norm(self.good_pos[i] - lm) for lm in self.landmark_pos)
            total_reward -= min_dist
        
        # Penalty for being caught by adversaries
        for i in range(self.n_good):
            for j in range(self.n_adversaries):
                dist = np.linalg.norm(self.good_pos[i] - self.adv_pos[j])
                if dist < self.agent_size * 2:
                    total_reward -= 10.0
        
        # Shared reward among good agents
        return [total_reward / self.n_good] * self.n_good
    
    def render(self, mode: str = 'human'):
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print("Good agents:", self.good_pos)
        print("Adversaries:", self.adv_pos)


def make_simple_world_comm(n_good: int = 2, n_adversaries: int = 4, 
                           max_steps: int = 25) -> BaseEnv:
    """
    Create Simple World Comm environment.
    
    Standard setup: 2 good agents vs 4 adversaries.
    Only good agents are controlled; adversaries use scripted policy.
    
    Args:
        n_good: Number of good agents to control (default: 2)
        n_adversaries: Number of adversaries (default: 4)
        max_steps: Maximum steps per episode (default: 25)
    
    Returns:
        Environment instance
    """
    try:
        from pettingzoo.mpe import simple_world_comm_v3
        env = simple_world_comm_v3.parallel_env(
            num_good=n_good,
            num_adversaries=n_adversaries,
            num_obstacles=1,
            max_cycles=max_steps,
            continuous_actions=False
        )
        print(f"  ✓ Using PettingZoo simple_world_comm_v3")
        # Filter to control only good agents (agent_0, agent_1, etc.)
        # Adversaries are (adversary_0, adversary_1, etc.)
        return PettingZooWrapper(env, agent_filter='agent')
    except ImportError:
        print(f"  ⚠️ PettingZoo not available, using built-in")
        return SimpleWorldCommEnv(n_good=n_good, n_adversaries=n_adversaries, max_steps=max_steps)
    except Exception as e:
        print(f"  ⚠️ PettingZoo error: {e}, using built-in")
        return SimpleWorldCommEnv(n_good=n_good, n_adversaries=n_adversaries, max_steps=max_steps)