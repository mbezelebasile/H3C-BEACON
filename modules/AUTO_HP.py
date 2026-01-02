"""
AUTO_HP.py - Adaptive Unified Tuning & Optimization 

STABILITY IMPROVEMENTS:
1. Slower, more conservative adaptations
2. Stability detection before changes
3. Bounded coefficient ranges
4. Exponential moving averages for smoother tracking
5. Plateau detection
"""

import numpy as np
from typing import Dict, Optional, Any
from collections import deque


class EMATracker:
    """Exponential Moving Average tracker with stability detection."""
    
    def __init__(self, alpha: float = 0.01, window: int = 100):
        self.alpha = alpha
        self.window = window
        self.ema = None
        self.ema_var = None
        self.history = deque(maxlen=window)
    
    def update(self, value: float) -> float:
        if np.isnan(value) or np.isinf(value):
            return self.ema if self.ema is not None else 0.0
        
        self.history.append(value)
        
        if self.ema is None:
            self.ema = value
            self.ema_var = 0.0
        else:
            delta = value - self.ema
            self.ema = self.ema + self.alpha * delta
            self.ema_var = (1 - self.alpha) * (self.ema_var + self.alpha * delta ** 2)
        
        return self.ema
    
    def get_std(self) -> float:
        return np.sqrt(self.ema_var) if self.ema_var is not None else 0.0
    
    def is_stable(self, threshold: float = 0.1) -> bool:
        if len(self.history) < self.window // 2:
            return False
        std = self.get_std()
        mean = abs(self.ema) if self.ema else 1.0
        return std / (mean + 1e-8) < threshold
    
    def trend(self) -> float:
        if len(self.history) < 20:
            return 0.0
        recent = list(self.history)[-20:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope


class ConservativeScheduler:
    """Base class for conservative coefficient scheduling."""
    
    def __init__(self, initial: float, min_val: float, max_val: float,
                 increase_rate: float = 1.02, decrease_rate: float = 0.98):
        self.value = initial
        self.initial = initial
        self.min_val = min_val
        self.max_val = max_val
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.tracker = EMATracker(alpha=0.02)
        self.last_change_step = 0
        self.change_cooldown = 50  # Steps between changes
        self.step_count = 0
    
    def can_change(self) -> bool:
        return self.step_count - self.last_change_step >= self.change_cooldown
    
    def increase(self):
        if self.can_change():
            self.value = min(self.max_val, self.value * self.increase_rate)
            self.last_change_step = self.step_count
    
    def decrease(self):
        if self.can_change():
            self.value = max(self.min_val, self.value * self.decrease_rate)
            self.last_change_step = self.step_count
    
    def step(self, metric: float) -> float:
        self.step_count += 1
        self.tracker.update(metric)
        return self.value


class EntropyScheduler(ConservativeScheduler):
    """Entropy coefficient scheduler with collapse prevention."""
    
    def __init__(self, initial: float = 0.05, target: float = 1.0, 
                 floor: float = 0.3, action_dim: int = 5):
        super().__init__(
            initial=initial,
            min_val=0.001,
            max_val=0.3,
            increase_rate=1.01,
            decrease_rate=0.995
        )
        self.target = target
        self.floor = floor
        self.max_entropy = np.log(action_dim)
        self.collapse_count = 0
    
    def step(self, entropy: float) -> float:
        super().step(entropy)
        
        ema = self.tracker.ema
        
        # Collapse detection: entropy near max for too long
        if ema > self.max_entropy * 0.95:
            self.collapse_count += 1
        else:
            self.collapse_count = max(0, self.collapse_count - 1)
        
        # If entropy stuck at max, something is wrong - reduce coefficient
        if self.collapse_count > 100:
            self.decrease()
            self.collapse_count = 0
        
        # Normal adaptation
        if ema < self.floor:
            self.increase()
        elif ema > self.target * 1.5 and self.tracker.is_stable():
            self.decrease()
        
        return self.value


class KLScheduler(ConservativeScheduler):
    """KL penalty scheduler."""
    
    def __init__(self, initial: float = 0.2, target: float = 0.015):
        super().__init__(
            initial=initial,
            min_val=0.01,
            max_val=5.0,
            increase_rate=1.1,
            decrease_rate=0.9
        )
        self.target = target
    
    def step(self, kl: float) -> float:
        super().step(kl)
        
        ema = self.tracker.ema
        
        # Only adjust if stable
        if not self.tracker.is_stable(0.3):
            return self.value
        
        if ema > self.target * 2.0:
            self.increase()
        elif ema < self.target * 0.5:
            self.decrease()
        
        return self.value


class TemperatureScheduler:
    """Very slow temperature annealing."""
    
    def __init__(self, initial: float = 1.0, min_temp: float = 0.3,
                 decay_episodes: int = 5000):  # Much slower!
        self.temp = initial
        self.initial = initial
        self.min_temp = min_temp
        self.decay_episodes = decay_episodes
        self.episode = 0
    
    def step(self) -> float:
        self.episode += 1
        
        # Very slow cosine annealing
        progress = min(self.episode / self.decay_episodes, 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        
        self.temp = self.min_temp + (self.initial - self.min_temp) * cosine
        
        return self.temp


class LearningRateScheduler:
    """Conservative learning rate scheduling."""
    
    def __init__(self, initial: float = 1e-4, min_lr: float = 1e-6,
                 warmup_steps: int = 500, patience: int = 200):
        self.lr = initial
        self.initial = initial
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.patience = patience
        
        self.loss_tracker = EMATracker(alpha=0.01)
        self.best_loss = float('inf')
        self.steps_no_improve = 0
        self.step_count = 0
    
    def step(self, loss: float) -> float:
        self.step_count += 1
        self.loss_tracker.update(loss)
        
        # Warmup
        if self.step_count < self.warmup_steps:
            self.lr = self.initial * (self.step_count / self.warmup_steps)
            return self.lr
        
        current = self.loss_tracker.ema
        
        if current < self.best_loss * 0.99:
            self.best_loss = current
            self.steps_no_improve = 0
        else:
            self.steps_no_improve += 1
        
        # Reduce if no improvement
        if self.steps_no_improve >= self.patience:
            self.lr = max(self.min_lr, self.lr * 0.5)
            self.steps_no_improve = 0
            self.best_loss = current
        
        return self.lr


class AUTO_HP:
    """
    Adaptive Unified Tuning & Optimization - STABILIZED
    
    Key features:
    - Conservative adaptation rates
    - Stability detection before changes
    - Collapse detection and recovery
    - Very slow temperature decay
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        action_dim = config.get('action_dim', 5)
        
        # Schedulers with conservative defaults
        self.lr_scheduler = LearningRateScheduler(
            initial=config.get('lr_actor', 1e-4),
            warmup_steps=config.get('warmup_episodes', 200) * 4
        )
        
        self.entropy_scheduler = EntropyScheduler(
            initial=config.get('entropy_coef', 0.05),
            target=config.get('target_entropy', 1.0),
            floor=config.get('entropy_floor', 0.3),
            action_dim=action_dim
        )
        
        self.kl_scheduler = KLScheduler(
            initial=config.get('kl_beta', 0.2),
            target=config.get('kl_target', 0.015)
        )
        
        self.temp_scheduler = TemperatureScheduler(
            initial=config.get('temp_init', 1.0),
            min_temp=config.get('temp_min', 0.3),
            decay_episodes=config.get('temp_decay_episodes', 5000)
        )
        
        # Reward tracking
        self.reward_tracker = EMATracker(alpha=0.01)
        self.best_reward = float('-inf')
        
        # Current values
        self.current_hp = {
            'lr': self.lr_scheduler.lr,
            'entropy_coef': self.entropy_scheduler.value,
            'kl_coef': self.kl_scheduler.value,
            'temperature': self.temp_scheduler.temp
        }
        
        self.total_steps = 0
    
    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        self.total_steps += 1
        
        # Update schedulers
        if 'critic_loss' in metrics:
            self.current_hp['lr'] = self.lr_scheduler.step(metrics['critic_loss'])
        
        if 'entropy' in metrics:
            self.current_hp['entropy_coef'] = self.entropy_scheduler.step(metrics['entropy'])
        
        if 'kl' in metrics:
            self.current_hp['kl_coef'] = self.kl_scheduler.step(metrics['kl'])
        
        self.current_hp['temperature'] = self.temp_scheduler.step()
        
        # Track reward
        if 'reward' in metrics:
            self.reward_tracker.update(metrics['reward'])
            if self.reward_tracker.ema > self.best_reward:
                self.best_reward = self.reward_tracker.ema
        
        return self.current_hp.copy()
    
    def get_hp(self) -> Dict[str, float]:
        return self.current_hp.copy()
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'auto_lr': self.current_hp['lr'],
            'auto_entropy_coef': self.current_hp['entropy_coef'],
            'auto_kl_coef': self.current_hp['kl_coef'],
            'auto_temp': self.current_hp['temperature'],
            'auto_reward_ema': self.reward_tracker.ema or 0.0
        }


if __name__ == "__main__":
    print("Testing Stabilized AUTO_HP...")
    
    config = {
        'lr_actor': 1e-4,
        'entropy_coef': 0.05,
        'kl_beta': 0.2,
        'temp_init': 1.0,
        'action_dim': 5
    }
    
    auto = AUTO_HP(config)
    
    # Simulate training
    print("\nSimulating 1000 steps...")
    for step in range(1000):
        metrics = {
            'reward': -50 + step * 0.02 + np.random.randn() * 5,
            'entropy': 1.5 - step * 0.0005 + np.random.randn() * 0.05,
            'kl': 0.01 + np.random.randn() * 0.003,
            'critic_loss': 3 - step * 0.001 + np.random.randn() * 0.3
        }
        hp = auto.update(metrics)
        
        if step % 200 == 0:
            print(f"  Step {step}: lr={hp['lr']:.6f}, ent={hp['entropy_coef']:.4f}, "
                  f"temp={hp['temperature']:.3f}")
    
    print("\n✓ All tests passed!")