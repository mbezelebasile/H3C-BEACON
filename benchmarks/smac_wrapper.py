v"""
SMACLite Wrapper pour H3C-BEACON
"""
import numpy as np
from smaclite.env import StarCraft2Env

class SMACLiteWrapper:
    """Wrapper pour SMACLite compatible avec H3C-BEACON."""
    
    def __init__(self, map_name: str = '3m', seed: int = 42):
        self.map_name = map_name
        self.env = StarCraft2Env(map_name=map_name, seed=seed)
        
        # Récupérer les infos de l'environnement
        env_info = self.env.get_env_info()
        self.n_agents = env_info['n_agents']
        self.obs_dim = env_info['obs_shape']
        self.action_dim = env_info['n_actions']
        self.episode_limit = env_info['episode_limit']
        
    def reset(self):
        """Reset l'environnement."""
        self.env.reset()
        return self.get_obs()
    
    def get_obs(self):
        """Retourne les observations de tous les agents."""
        return np.array([self.env.get_obs_agent(i) for i in range(self.n_agents)])
    
    def get_state(self):
        """Retourne l'état global."""
        return self.env.get_state()
    
    def get_avail_actions(self):
        """Retourne les actions disponibles."""
        return np.array([self.env.get_avail_agent_actions(i) for i in range(self.n_agents)])
    
    def step(self, actions):
        """Exécute les actions."""
        reward, terminated, info = self.env.step(actions)
        
        obs = self.get_obs()
        rewards = np.full(self.n_agents, reward / self.n_agents, dtype=np.float32)
        dones = np.full(self.n_agents, terminated, dtype=np.float32)
        
        info['battle_won'] = info.get('battle_won', False)
        
        return obs, rewards, dones, info
    
    def close(self):
        self.env.close()