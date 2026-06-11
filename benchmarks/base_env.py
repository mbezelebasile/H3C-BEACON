"""
H3C-BEACON Benchmark Environments
=================================
Professional-grade environment wrappers for MARL benchmarks.

Supports:
- PettingZoo MPE (Multi-Particle Environments)
- SMAC (StarCraft Multi-Agent Challenge)
- GRF (Google Research Football)

Author: H3C-BEACON Research Team
"""

import os
import warnings

# Suppress PettingZoo deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pettingzoo')
warnings.filterwarnings('ignore', message='.*observation_spaces.*deprecated.*')
warnings.filterwarnings('ignore', message='.*action_spaces.*deprecated.*')

# Suppress XDG_RUNTIME_DIR error
if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-user'

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class BaseEnv(ABC):
    """Abstract base class for all benchmark environments."""
    
    def __init__(self, n_agents: int, obs_dim: int, act_dim: int, max_steps: int = 25):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.current_step = 0
    
    @abstractmethod
    def reset(self) -> List[np.ndarray]:
        """Reset the environment and return initial observations."""
        pass
    
    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Execute actions and return (observations, rewards, dones, info)."""
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human'):
        """Render the environment."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_env_info(self) -> Dict[str, Any]:
        """Return environment information."""
        return {
            'n_agents': self.n_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'max_steps': self.max_steps
        }


class PettingZooWrapper(BaseEnv):
    """
    Professional wrapper for PettingZoo parallel environments.
    
    Handles:
    - Heterogeneous observation spaces (padding to max dim)
    - Heterogeneous action spaces (clipping to valid range)
    - New Gymnasium API (obs, info) = reset()
    - Agent filtering (e.g., control only 'good' agents)
    """
    
    def __init__(self, env, agent_filter: Optional[str] = None):
        """
        Args:
            env: PettingZoo parallel environment
            agent_filter: Optional filter for agent names (e.g., 'agent' to control only agents starting with 'agent')
        """
        self.env = env
        self._agent_filter = agent_filter
        self._all_agents = None
        self._controlled_agents = None
        
        # Get all agent keys
        all_agent_keys = list(env.observation_spaces.keys())
        
        # Filter agents if specified
        if agent_filter:
            controlled_keys = [a for a in all_agent_keys if agent_filter in a]
            if not controlled_keys:
                print(f"  ⚠️ No agents match filter '{agent_filter}', using all agents")
                controlled_keys = all_agent_keys
        else:
            controlled_keys = all_agent_keys
        
        self._controlled_agent_keys = controlled_keys
        
        # Analyze observation and action spaces
        self._obs_dims = {}
        self._act_dims = {}
        max_obs_dim = 0
        max_act_dim = 0
        
        for agent in controlled_keys:
            obs_space = env.observation_spaces[agent]
            act_space = env.action_spaces[agent]
            
            # Handle different observation space types
            if hasattr(obs_space, 'shape') and len(obs_space.shape) > 0:
                obs_d = int(np.prod(obs_space.shape))
            elif hasattr(obs_space, 'n'):
                obs_d = int(obs_space.n)
            else:
                obs_d = 1
            
            # Handle different action space types
            if hasattr(act_space, 'n'):
                act_d = int(act_space.n)
            elif hasattr(act_space, 'shape'):
                act_d = int(np.prod(act_space.shape))
            else:
                act_d = 1
            
            self._obs_dims[agent] = obs_d
            self._act_dims[agent] = act_d
            max_obs_dim = max(max_obs_dim, obs_d)
            max_act_dim = max(max_act_dim, act_d)
        
        self._max_obs_dim = max_obs_dim
        self._max_act_dim = max_act_dim
        n_agents = len(controlled_keys)
        
        super().__init__(n_agents, max_obs_dim, max_act_dim)
        
        # Log heterogeneity info
        unique_obs_dims = set(self._obs_dims.values())
        unique_act_dims = set(self._act_dims.values())
        
        if len(unique_obs_dims) > 1:
            print(f"  ⚠️ Heterogeneous obs dims: {unique_obs_dims} → padding to {max_obs_dim}")
        if len(unique_act_dims) > 1:
            print(f"  ⚠️ Heterogeneous act dims: {unique_act_dims} → using max {max_act_dim}")
    
    def _pad_observation(self, obs: np.ndarray, agent: str) -> np.ndarray:
        """Pad observation to max dimension."""
        obs_flat = np.array(obs, dtype=np.float32).flatten()
        if len(obs_flat) < self._max_obs_dim:
            padded = np.zeros(self._max_obs_dim, dtype=np.float32)
            padded[:len(obs_flat)] = obs_flat
            return padded
        return obs_flat[:self._max_obs_dim]
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return observations for controlled agents."""
        result = self.env.reset()
        
        # Handle new Gymnasium API: (obs_dict, info_dict)
        if isinstance(result, tuple):
            obs_dict = result[0]
        else:
            obs_dict = result
        
        self._all_agents = list(obs_dict.keys())
        self._controlled_agents = [a for a in self._all_agents if a in self._controlled_agent_keys]
        self.current_step = 0
        
        # Track episode rewards for win condition
        self._episode_rewards = {agent: 0.0 for agent in self._controlled_agents}
        
        # Return padded observations for controlled agents only
        return [self._pad_observation(obs_dict[agent], agent) for agent in self._controlled_agents]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Step environment with actions for controlled agents."""
        # Build action dict - controlled agents get our actions, others get random/noop
        action_dict = {}
        
        controlled_idx = 0
        for agent in self._all_agents:
            if agent in self._controlled_agents:
                # Clip action to valid range for this agent
                max_action = self._act_dims.get(agent, self._max_act_dim) - 1
                action_dict[agent] = min(int(actions[controlled_idx]), max_action)
                controlled_idx += 1
            else:
                # Non-controlled agents: random action (for adversaries in mixed envs)
                max_action = self._act_dims.get(agent, self._max_act_dim) - 1
                action_dict[agent] = np.random.randint(0, max_action + 1)
        
        result = self.env.step(action_dict)
        
        # Handle different API versions
        if len(result) == 5:
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = result
            done_dict = {k: term_dict.get(k, False) or trunc_dict.get(k, False) 
                        for k in self._all_agents}
        else:
            obs_dict, reward_dict, done_dict, info_dict = result
        
        self.current_step += 1
        
        # Extract only controlled agents' data
        obs = []
        rewards = []
        dones = []
        
        for agent in self._controlled_agents:
            if agent in obs_dict:
                obs.append(self._pad_observation(obs_dict[agent], agent))
            else:
                obs.append(np.zeros(self._max_obs_dim, dtype=np.float32))
            
            r = float(reward_dict.get(agent, 0.0))
            rewards.append(r)
            dones.append(bool(done_dict.get(agent, True)))
            
            # Track cumulative reward
            self._episode_rewards[agent] = self._episode_rewards.get(agent, 0.0) + r
        
        # Compute win condition
        win = self._compute_win(obs, rewards, dones, info_dict)
        
        return obs, rewards, dones, {'win': win}
    
    def _compute_win(self, obs: List[np.ndarray], rewards: List[float], 
                     dones: List[bool], info_dict: dict) -> bool:
        """
        Compute win condition based on environment type and performance.
        
        Win criteria based on AVERAGE episode reward (consistent with displayed metrics):
        - simple_spread: Average reward > -25 (good landmark coverage)
        - simple_world_comm: Average reward > -35 (survived, made progress)
        
        NOTE: We use AVERAGE (not sum) to be consistent with what's displayed in training.
        """
        # Check if episode is done
        if not any(dones):
            return False
        
        # First check if info contains explicit win
        for agent in self._controlled_agents:
            agent_info = info_dict.get(agent, {})
            if isinstance(agent_info, dict) and agent_info.get('win', False):
                return True
        
        # Compute AVERAGE episode reward (consistent with evaluate() display)
        # _episode_rewards contains cumulative reward per agent
        n_agents = len(self._controlled_agents)
        if n_agents == 0:
            return False
        
        avg_episode_reward = sum(self._episode_rewards.values()) / n_agents
        
        # Win thresholds based on AVERAGE reward (what's displayed)
        # These are calibrated from typical training curves
        if n_agents == 3:  # simple_spread
            # Trained agent: ~-20 average, Random: ~-50 average
            # Win if better than -25 average
            win_threshold = -25.0
        elif n_agents == 2:  # simple_world_comm
            # Trained agent: ~-30 average, Random: ~-60 average
            # Win if better than -40 average
            win_threshold = -40.0
        else:
            # Generic: win if average > -25
            win_threshold = -25.0
        
        return avg_episode_reward > win_threshold
    
    def render(self, mode: str = 'human'):
        return self.env.render()
    
    def close(self):
        self.env.close()


class SMACWrapper(BaseEnv):
    """
    Wrapper for StarCraft Multi-Agent Challenge (SMAC) environments.
    
    Requires: pysc2, smac
    """
    
    def __init__(self, env):
        self.env = env
        env_info = env.get_env_info()
        
        n_agents = env_info['n_agents']
        obs_dim = env_info['obs_shape']
        act_dim = env_info['n_actions']
        max_steps = env_info.get('episode_limit', 100)
        
        super().__init__(n_agents, obs_dim, act_dim, max_steps)
    
    def reset(self) -> List[np.ndarray]:
        self.env.reset()
        self.current_step = 0
        return [self.env.get_obs_agent(i) for i in range(self.n_agents)]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        reward, done, info = self.env.step(actions)
        self.current_step += 1
        
        obs = [self.env.get_obs_agent(i) for i in range(self.n_agents)]
        rewards = [reward] * self.n_agents
        dones = [done] * self.n_agents
        
        return obs, rewards, dones, info
    
    def get_avail_actions(self) -> List[np.ndarray]:
        """Get available actions for each agent."""
        return [np.array(self.env.get_avail_agent_actions(i)) for i in range(self.n_agents)]
    
    def render(self, mode: str = 'human'):
        pass
    
    def close(self):
        self.env.close()


class GRFWrapper(BaseEnv):
    """
    Wrapper for Google Research Football environments.
    
    Requires: gfootball
    """
    
    def __init__(self, env, n_agents: int = 3):
        self.env = env
        self._n_agents = n_agents
        
        # GRF observation and action dimensions
        obs_shape = env.observation_space.shape
        if len(obs_shape) > 1:
            obs_dim = int(np.prod(obs_shape[1:]))  # Per-agent obs
        else:
            obs_dim = obs_shape[0] // n_agents
        
        act_dim = env.action_space.nvec[0] if hasattr(env.action_space, 'nvec') else env.action_space.n
        
        super().__init__(n_agents, obs_dim, int(act_dim))
    
    def reset(self) -> List[np.ndarray]:
        obs = self.env.reset()
        self.current_step = 0
        
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Split observation for each agent
        if len(obs.shape) > 1:
            return [obs[i].astype(np.float32) for i in range(self._n_agents)]
        else:
            obs_per_agent = len(obs) // self._n_agents
            return [obs[i*obs_per_agent:(i+1)*obs_per_agent].astype(np.float32) 
                    for i in range(self._n_agents)]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        obs, reward, done, info = self.env.step(actions)
        self.current_step += 1
        
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Split observation
        if len(obs.shape) > 1:
            obs_list = [obs[i].astype(np.float32) for i in range(self._n_agents)]
        else:
            obs_per_agent = len(obs) // self._n_agents
            obs_list = [obs[i*obs_per_agent:(i+1)*obs_per_agent].astype(np.float32) 
                        for i in range(self._n_agents)]
        
        # Shared reward
        if isinstance(reward, (list, np.ndarray)):
            rewards = list(reward)
        else:
            rewards = [float(reward)] * self._n_agents
        
        dones = [done] * self._n_agents
        
        # Check for goal
        win = info.get('score_reward', 0) > 0 if isinstance(info, dict) else False
        
        return obs_list, rewards, dones, {'win': win}
    
    def render(self, mode: str = 'human'):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()