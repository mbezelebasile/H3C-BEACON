"""
H3C-BEACON: Base Trainer Class
All baseline algorithms inherit from this class.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import time
import os
import json
from datetime import datetime


class BaseTrainer(ABC):
    """Abstract base class for all MARL trainers."""
    
    def __init__(
        self,
        env,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.env = env
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = torch.device(device)
        
        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.best_reward = float('-inf')
        
        # Logging
        self.train_rewards = []
        self.eval_rewards = []
        self.losses = []
        self.metrics_history = []
        
        # Timing
        self.start_time = None
        
    @abstractmethod
    def select_actions(self, observations: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        """Select actions for all agents given observations."""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the policy/value networks."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model checkpoints."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model checkpoints."""
        pass
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name."""
        return self.__class__.__name__.replace("Trainer", "")
    
    def collect_episode(self, max_steps: int = 25) -> Tuple[List[Dict], float]:
        """Collect one episode of experience."""
        observations = self.env.reset()
        episode_data = []
        episode_reward = 0
        
        for step in range(max_steps):
            actions = self.select_actions(observations, explore=True)
            next_observations, rewards, dones, infos = self.env.step(actions)
            
            transition = {
                'obs': observations,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_observations,
                'dones': dones
            }
            episode_data.append(transition)
            
            if isinstance(rewards, (list, np.ndarray)):
                episode_reward += np.mean(rewards)
            else:
                episode_reward += rewards
            
            observations = next_observations
            self.total_steps += self.n_agents
            
            if all(dones) if isinstance(dones, (list, np.ndarray)) else dones:
                break
        
        self.episodes += 1
        return episode_data, episode_reward
    
    def evaluate(self, n_episodes: int = 10, max_steps: int = 25) -> Tuple[float, float, float]:
        """Evaluate current policy. Returns (mean_reward, std_reward, win_rate).
        
        Note: Rewards are AVERAGED across agents per step for comparability.
        """
        rewards = []
        wins = 0
        
        for _ in range(n_episodes):
            observations = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                actions = self.select_actions(observations, explore=False)
                observations, reward, dones, info = self.env.step(actions)
                
                # Average reward across agents (consistent metric)
                if isinstance(reward, (list, np.ndarray)):
                    episode_reward += np.mean(reward)
                else:
                    episode_reward += reward
                
                if all(dones) if isinstance(dones, (list, np.ndarray)) else dones:
                    break
            
            rewards.append(episode_reward)
            
            # Win detection: use info from environment (which now uses avg reward)
            if isinstance(info, dict) and info.get('win', False):
                wins += 1
        
        win_rate = wins / n_episodes * 100
        return np.mean(rewards), np.std(rewards), win_rate
    
    def train(
        self,
        total_steps: int,
        eval_interval: int = 160000,
        log_interval: int = 32000,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main training loop."""
        self.start_time = time.time()
        
        print(f"Training {self.get_algorithm_name()} on {self.env.__class__.__name__}")
        print(f"Agents: {self.n_agents} | Steps: {total_steps}")
        print("=" * 60)
        
        # Initial evaluation
        print("\nRunning initial evaluation...")
        init_mean, init_std, init_win = self.evaluate()
        print(f"Initial: {init_mean:.2f} ± {init_std:.2f}\n")
        self.eval_rewards.append({'step': 0, 'mean': init_mean, 'std': init_std, 'win_rate': init_win})
        self.best_reward = init_mean
        self.best_win_rate = init_win
        
        episode_rewards = []
        next_log_step = log_interval
        next_eval_step = eval_interval
        
        while self.total_steps < total_steps:
            episode_data, episode_reward = self.collect_episode()
            episode_rewards.append(episode_reward)
            
            # Only prepare batch if we have data (on-policy)
            # Off-policy algorithms (QMIX, VDN) return empty episode_data
            if episode_data:
                batch = self._prepare_batch(episode_data)
                metrics = self.update(batch)
            else:
                # Off-policy: update uses internal replay buffer
                metrics = self.update({})
            
            # Log at exact intervals
            if self.total_steps >= next_log_step:
                elapsed = (time.time() - self.start_time) / 60
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                self._log_progress(next_log_step, avg_reward, metrics, elapsed)
                next_log_step += log_interval
            
            # Eval at exact intervals
            if self.total_steps >= next_eval_step:
                eval_mean, eval_std, eval_win = self.evaluate()
                is_best = eval_mean > self.best_reward
                
                if is_best:
                    self.best_reward = eval_mean
                    self.best_win_rate = eval_win
                    if save_path:
                        self.save(os.path.join(save_path, "best_model.pt"))
                
                self.eval_rewards.append({
                    'step': next_eval_step,
                    'mean': eval_mean,
                    'std': eval_std,
                    'win_rate': eval_win
                })
                
                if is_best:
                    print(f"[EVAL] Step {next_eval_step:>6} | Reward: {eval_mean:.2f} ± {eval_std:.2f} | ★ NEW BEST: {eval_mean:.2f} | Win: {eval_win:.1f}%")
                else:
                    print(f"[EVAL] Step {next_eval_step:>6} | Reward: {eval_mean:.2f} ± {eval_std:.2f} | Best: {self.best_reward:.2f} | Win: {eval_win:.1f}%")
                
                next_eval_step += eval_interval
        
        final_mean, final_std, final_win = self.evaluate(n_episodes=10)
        results = self._compute_final_metrics(final_mean, final_std, final_win)
        
        # Calculate improvement for display
        initial_reward = self.eval_rewards[0]['mean'] if self.eval_rewards else 0
        improvement = self.best_reward - initial_reward
        
        print(f"\n" + "=" * 60)
        print(f"Training Complete!")
        print(f"Final: {final_mean:.2f} ± {final_std:.2f}")
        print(f"Best:  {self.best_reward:.2f}")
        print(f"Win Rate: {final_win:.1f}%")
        print("")
        print(f"📊 Stability Metrics:")
        print(f"   Variance: {results.get('variance', 0):.4f}")
        print(f"   CV: {results.get('cv', 0):.4f}")
        print(f"   Stability Score: {results.get('stability', 0):.4f}")
        print("")
        print(f"📈 Convergence Metrics:")
        print(f"   AUC: {results.get('auc', 0):.2f}")
        print(f"   Improvement: {improvement:+.2f} (from {initial_reward:.2f} to {self.best_reward:.2f})")
        steps_90 = results.get('steps_to_90')
        print(f"   Steps to 90%: {steps_90 if steps_90 else 'N/A'}")
        conv_speed = results.get('convergence_speed', 0)
        print(f"   Convergence Speed: {conv_speed:.2f} (reward/M steps)")
        print("=" * 60 + "\n")
        
        if save_path:
            self._save_results(save_path, results)
        
        return results
    
    def _prepare_batch(self, episode_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert episode data to tensors. Handles heterogeneous observations."""
        
        # Handle empty data
        if not episode_data:
            return {}
        
        # Helper to safely stack observations
        def safe_stack(data_list):
            if not data_list:
                return np.array([], dtype=np.float32)
            try:
                return np.stack([np.array(d, dtype=np.float32) for d in data_list])
            except ValueError:
                # Handle heterogeneous dimensions by padding
                max_len = max(max(len(o) for o in d) if isinstance(d, list) else len(d) for d in data_list)
                n_items = len(data_list)
                n_agents = len(data_list[0]) if isinstance(data_list[0], list) else 1
                result = np.zeros((n_items, n_agents, max_len), dtype=np.float32)
                for i, d in enumerate(data_list):
                    if isinstance(d, list):
                        for j, obs in enumerate(d):
                            result[i, j, :len(obs)] = obs
                    else:
                        result[i, 0, :len(d)] = d
                return result
        
        obs_list = [t['obs'] for t in episode_data]
        next_obs_list = [t['next_obs'] for t in episode_data]
        
        batch = {
            'obs': torch.FloatTensor(safe_stack(obs_list)).to(self.device),
            'actions': torch.LongTensor(np.array([t['actions'] for t in episode_data])).to(self.device),
            'rewards': torch.FloatTensor(np.array([t['rewards'] for t in episode_data])).to(self.device),
            'next_obs': torch.FloatTensor(safe_stack(next_obs_list)).to(self.device),
            'dones': torch.FloatTensor(np.array([t['dones'] for t in episode_data])).to(self.device)
        }
        return batch
    
    def _log_progress(self, step: int, avg_reward: float, metrics: Dict[str, float], elapsed: float):
        """Log training progress."""
        # Show entropy for policy-based, epsilon for value-based
        if 'entropy' in metrics and metrics['entropy'] > 0:
            explore_str = f"H: {metrics['entropy']:.2f}"
        elif 'epsilon' in metrics:
            explore_str = f"ε: {metrics['epsilon']:.2f}"
        else:
            explore_str = f"H: {metrics.get('entropy', 0):.2f}"
        
        log_str = f"Step {step:>7} | Train: {avg_reward:>8.2f} | Loss: {metrics.get('loss', 0):.4f} | {explore_str} | Time: {elapsed:.1f}min"
        print(log_str)
        self.metrics_history.append({'step': step, 'reward': avg_reward, **metrics})
    
    def _compute_final_metrics(self, final_mean: float, final_std: float, final_win: float = 0) -> Dict[str, Any]:
        """Compute final performance metrics."""
        eval_rewards = [e['mean'] for e in self.eval_rewards]
        
        # AUC (normalized by number of evaluations)
        auc = np.trapz(eval_rewards) / len(eval_rewards) if eval_rewards else 0
        
        # Variance and CV
        variance = np.var(eval_rewards) if len(eval_rewards) > 1 else 0
        mean_reward = np.mean(eval_rewards) if eval_rewards else 0
        cv = np.std(eval_rewards) / (np.abs(mean_reward) + 1e-8) if len(eval_rewards) > 1 else 0
        stability = 1 / (1 + cv)
        
        # ================================================================
        # FIXED: Steps to 90% of IMPROVEMENT (not 90% of best!)
        # ================================================================
        # For both positive and negative rewards:
        # target_90 = initial + 0.9 * (best - initial)
        # This means "90% of the way from initial to best"
        
        initial_reward = self.eval_rewards[0]['mean'] if self.eval_rewards else 0
        improvement = self.best_reward - initial_reward
        
        # Target is 90% of the improvement from initial
        target_90 = initial_reward + 0.9 * improvement
        
        steps_to_90 = None
        for e in self.eval_rewards:
            # For negative rewards: higher is better, so we check >=
            # For positive rewards: higher is also better, so >= works for both
            if improvement >= 0:  # We improved (or stayed same)
                if e['mean'] >= target_90:
                    steps_to_90 = e['step']
                    break
            else:  # We got worse (unlikely but handle it)
                if e['mean'] <= target_90:
                    steps_to_90 = e['step']
                    break
        
        # ================================================================
        # FIXED: Convergence speed calculation
        # ================================================================
        if steps_to_90 and steps_to_90 > 0:
            # Speed = how fast we reached 90% of improvement
            # Normalized by 1M steps for comparability
            convergence_speed = (0.9 * abs(improvement)) / steps_to_90 * 1e6
        else:
            # Fallback: use total improvement rate
            if len(self.eval_rewards) >= 2 and self.total_steps > 0:
                convergence_speed = abs(improvement) / self.total_steps * 1e6
            else:
                convergence_speed = 0
        
        return {
            'algorithm': self.get_algorithm_name(),
            'best': self.best_reward,
            'final_mean': final_mean,
            'final_std': final_std,
            'final_win_rate': final_win,
            'auc': auc,
            'variance': variance,
            'cv': cv,
            'stability': stability,
            'steps_to_90': steps_to_90,
            'convergence_speed': convergence_speed,
            'total_steps': self.total_steps,
            'eval_history': self.eval_rewards,
            'metrics_history': self.metrics_history
        }
    
    def _save_results(self, path: str, results: Dict[str, Any]):
        """Save results to disk."""
        os.makedirs(path, exist_ok=True)
        
        results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results.items() if k not in ['eval_history', 'metrics_history']}
        
        with open(os.path.join(path, "results.json"), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        np.savez(os.path.join(path, "history.npz"),
                eval_history=self.eval_rewards,
                metrics_history=self.metrics_history)
        
        print(f"Results saved to: {path}")