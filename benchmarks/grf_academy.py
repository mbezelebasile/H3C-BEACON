"""
Google Research Football - Academy Scenarios
=============================================
Multi-agent football scenarios for cooperative learning.

Standard Benchmark:
- academy_3_vs_1_with_keeper: 3 attackers vs 1 defender + keeper

Reference:
    Kurach et al., 2020 - "Google Research Football: A Novel Reinforcement Learning Environment"
    https://arxiv.org/abs/1907.11180

Requires: gfootball (pip install gfootball)

Author: H3C-BEACON Research Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_env import BaseEnv, GRFWrapper


class Academy3vs1WithKeeperEnv(BaseEnv):
    """
    Built-in simulation of academy_3_vs_1_with_keeper.
    
    3 attackers try to score against 1 defender and 1 goalkeeper.
    Tests cooperative passing and shooting.
    """
    
    def __init__(self, max_steps: int = 100):
        n_agents = 3
        
        # Observation per agent:
        # [pos(2), vel(2), ball_pos_rel(2), ball_vel(2), 
        #  teammates_rel((n-1)*2), defender_rel(2), keeper_rel(2), has_ball(1)]
        obs_dim = 2 + 2 + 2 + 2 + (n_agents - 1) * 2 + 2 + 2 + 1
        
        # Actions: [idle, left, right, top, bottom, top_left, top_right, bottom_left, kick]
        act_dim = 9
        
        super().__init__(n_agents, obs_dim, act_dim, max_steps)
        
        # Field dimensions (normalized)
        self.field_width = 2.0  # -1 to 1
        self.field_height = 0.84  # -0.42 to 0.42
        
        # Agent parameters
        self.agent_speed = 0.05
        self.kick_power = 0.25
        self.ball_friction = 0.95
        self.tackle_range = 0.08
        self.receive_range = 0.1
        
        # State
        self.agent_pos = None
        self.agent_vel = None
        self.defender_pos = None
        self.keeper_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_owner = None  # -1 for free, 0-2 for agents, 3 for defender, 4 for keeper
        
        self.goal_scored = False
        self.win = False
    
    def reset(self) -> List[np.ndarray]:
        self.current_step = 0
        self.goal_scored = False
        self.win = False
        
        # Initialize attackers in formation
        self.agent_pos = np.array([
            [-0.5, 0.0],    # Center forward
            [-0.3, 0.2],    # Right wing
            [-0.3, -0.2],   # Left wing
        ], dtype=np.float32)
        self.agent_vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        
        # Defender starts in front of goal
        self.defender_pos = np.array([0.3, 0.0], dtype=np.float32)
        
        # Keeper in goal
        self.keeper_pos = np.array([0.9, 0.0], dtype=np.float32)
        
        # Ball starts with center forward
        self.ball_pos = self.agent_pos[0].copy()
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_owner = 0
        
        return self._get_obs()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        self.current_step += 1
        reward = 0.0
        
        # Movement directions
        directions = {
            0: [0, 0],      # idle
            1: [-1, 0],     # left
            2: [1, 0],      # right
            3: [0, 1],      # top
            4: [0, -1],     # bottom
            5: [-0.7, 0.7], # top_left
            6: [0.7, 0.7],  # top_right
            7: [-0.7, -0.7],# bottom_left
            8: [0, 0],      # kick (special)
        }
        
        # Process agent actions
        for i, action in enumerate(actions):
            if action == 8 and self.ball_owner == i:  # Kick
                # Aim towards goal with some randomness
                goal_y = np.random.uniform(-0.15, 0.15)
                goal_pos = np.array([1.0, goal_y])
                kick_dir = goal_pos - self.ball_pos
                kick_dir = kick_dir / (np.linalg.norm(kick_dir) + 1e-8)
                self.ball_vel = kick_dir * self.kick_power
                self.ball_owner = -1
                reward += 0.1  # Reward for shooting
            else:
                # Move
                move = np.array(directions.get(action, [0, 0]), dtype=np.float32)
                if np.linalg.norm(move) > 0:
                    move = move / np.linalg.norm(move) * self.agent_speed
                self.agent_vel[i] = move
                self.agent_pos[i] = np.clip(
                    self.agent_pos[i] + self.agent_vel[i],
                    [-1.0, -0.42], [1.0, 0.42]
                )
        
        # Update ball position if free
        if self.ball_owner == -1:
            self.ball_pos += self.ball_vel
            self.ball_vel *= self.ball_friction
            
            # Check if agent receives ball
            for i in range(self.n_agents):
                if np.linalg.norm(self.ball_pos - self.agent_pos[i]) < self.receive_range:
                    self.ball_owner = i
                    self.ball_vel = np.zeros(2)
                    reward += 0.05  # Reward for receiving
                    break
        else:
            # Ball follows owner
            if self.ball_owner >= 0 and self.ball_owner < self.n_agents:
                self.ball_pos = self.agent_pos[self.ball_owner].copy()
        
        # Defender AI: track ball or ball carrier
        target = self.ball_pos if self.ball_owner == -1 else (
            self.agent_pos[self.ball_owner] if self.ball_owner < self.n_agents else self.ball_pos
        )
        def_dir = target - self.defender_pos
        if np.linalg.norm(def_dir) > 0:
            def_dir = def_dir / np.linalg.norm(def_dir) * self.agent_speed * 0.7
        self.defender_pos = np.clip(self.defender_pos + def_dir, [-1.0, -0.42], [1.0, 0.42])
        
        # Defender tackle
        if self.ball_owner >= 0 and self.ball_owner < self.n_agents:
            if np.linalg.norm(self.defender_pos - self.agent_pos[self.ball_owner]) < self.tackle_range:
                if np.random.random() < 0.4:  # 40% tackle success
                    self.ball_owner = -1
                    self.ball_vel = np.array([-0.15, np.random.uniform(-0.1, 0.1)])
                    reward -= 0.2  # Penalty for losing ball
        
        # Keeper AI: guard goal
        if self.ball_pos[0] > 0.5:
            keeper_target_y = np.clip(self.ball_pos[1], -0.2, 0.2)
            keeper_move = np.clip(keeper_target_y - self.keeper_pos[1], -0.06, 0.06)
            self.keeper_pos[1] += keeper_move
        
        # Check goal
        if self.ball_pos[0] >= 0.95 and abs(self.ball_pos[1]) < 0.2:
            keeper_dist = np.linalg.norm(self.ball_pos - self.keeper_pos)
            if keeper_dist > 0.12:  # Goal scored!
                reward += 10.0
                self.goal_scored = True
                self.win = True
            else:  # Keeper save
                self.ball_vel = np.array([-0.2, np.random.uniform(-0.1, 0.1)])
                self.ball_owner = -1
                reward -= 0.1
        
        # Out of bounds
        if abs(self.ball_pos[1]) > 0.45 or self.ball_pos[0] < -1.05:
            reward -= 0.5
            self.goal_scored = True  # End episode
        
        # Progress reward
        if self.ball_owner >= 0 and self.ball_owner < self.n_agents:
            progress = (self.ball_pos[0] + 1) / 2
            reward += 0.01 * progress
        
        done = self.goal_scored or self.current_step >= self.max_steps
        rewards = [reward / self.n_agents] * self.n_agents
        dones = [done] * self.n_agents
        
        return self._get_obs(), rewards, dones, {'win': self.win}
    
    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for each agent."""
        obs = []
        
        for i in range(self.n_agents):
            agent_obs = []
            
            # Self position and velocity
            agent_obs.extend(self.agent_pos[i])
            agent_obs.extend(self.agent_vel[i])
            
            # Ball position and velocity (relative)
            agent_obs.extend(self.ball_pos - self.agent_pos[i])
            agent_obs.extend(self.ball_vel)
            
            # Teammates (relative positions)
            for j in range(self.n_agents):
                if i != j:
                    agent_obs.extend(self.agent_pos[j] - self.agent_pos[i])
            
            # Defender (relative)
            agent_obs.extend(self.defender_pos - self.agent_pos[i])
            
            # Keeper (relative)
            agent_obs.extend(self.keeper_pos - self.agent_pos[i])
            
            # Has ball
            agent_obs.append(1.0 if self.ball_owner == i else 0.0)
            
            obs.append(np.array(agent_obs, dtype=np.float32))
        
        return obs
    
    def render(self, mode: str = 'human'):
        owner_str = f"Agent {self.ball_owner}" if self.ball_owner >= 0 else "Free"
        print(f"Step {self.current_step}: Ball at {self.ball_pos}, Owner: {owner_str}")


def make_academy_3_vs_1_with_keeper(max_steps: int = 100) -> BaseEnv:
    """
    Create academy_3_vs_1_with_keeper scenario.
    
    Uses real GRF if available, otherwise built-in simulation.
    """
    try:
        import gfootball.env as football_env
        env = football_env.create_environment(
            env_name='academy_3_vs_1_with_keeper',
            representation='simple115v2',
            stacked=False,
            rewards='scoring,checkpoints',
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            number_of_left_players_agent_controls=3,
        )
        print(f"  ✓ Using GRF academy_3_vs_1_with_keeper")
        return GRFWrapper(env, n_agents=3)
    except ImportError:
        print(f"  ⚠️ gfootball not available, using built-in simulation")
        return Academy3vs1WithKeeperEnv(max_steps=max_steps)
    except Exception as e:
        print(f"  ⚠️ GRF error: {e}, using built-in")
        return Academy3vs1WithKeeperEnv(max_steps=max_steps)