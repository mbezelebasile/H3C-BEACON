#!/usr/bin/env python3
"""
H3C-BEACON: Training Script
============================
Compatible with H3CTrainer v1.0 (DGAT-BC)

Usage:
    python train.py --algo H3C --env simple_spread --steps 1000000
    python train.py --algo all --env all --steps 1000000
"""

import argparse
import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, List


# Suppress PettingZoo deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pettingzoo')
warnings.filterwarnings('ignore', message='.*observation_spaces.*deprecated.*')
warnings.filterwarnings('ignore', message='.*action_spaces.*deprecated.*')

# Suppress XDG_RUNTIME_DIR error (common in Colab/Docker)
if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-user'

# Suppress other common warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame welcome message

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import baseline trainers
from baselines import (
    MAPPOTrainer, IPPOTrainer, QMIXTrainer, 
    VDNTrainer, COMATrainer, FACMACTrainer
)
from benchmarks import BENCHMARKS
from resultats.visualizer import ResultsVisualizer

# Try to import H3C - may have different module structure
try:
    from modules.H3CTrainer import H3CTrainer
    H3C_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import H3CTrainer: {e}")
    H3C_AVAILABLE = False



# Algorithm Registry
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

BASELINE_ALGORITHMS = {
    'MAPPO': MAPPOTrainer,
    'IPPO': IPPOTrainer,
    'QMIX': QMIXTrainer,
    'VDN': VDNTrainer,
    'COMA': COMATrainer,
    'FACMAC': FACMACTrainer
}



# Default Configurations
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

H3C_CONFIG = {
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'lr_actor': 3e-4,
    'lr_critic': 1e-3,
    'lr_min_ratio': 0.2,
    'max_grad_norm': 0.5,
    'clip_epsilon': 0.2,
    'ppo_epochs': 4,
    'mini_batch_size': 256,      # 64 → 256 (faster batching)
    'rollout_length': 512,       # 256 → 512 (less frequent updates)
    'value_loss_coef': 0.5,
    'sil_coef': 0.1,
    'kl_elite_coef': 0.05,
    'temp_init': 1.0,
    'hidden_dim': 128,
    'belief_dim': 64,
    'goal_dim': 64,
    'message_dim': 32,
    'n_coalitions': 2,
}

BASELINE_CONFIGS = {
    'MAPPO': {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4
    },
    'IPPO': {
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4
    },
    'QMIX': {
        'lr': 5e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 500000,
        'batch_size': 32,
        'target_update': 200
    },
    'VDN': {
        'lr': 5e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 500000,
        'batch_size': 32,
        'target_update': 200
    },
    'COMA': {
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'td_lambda': 0.8,
        'max_grad_norm': 10
    },
    'FACMAC': {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5
    }
}



# Environment Factory
# 

def create_env(env_name: str, **kwargs):
    """Create environment by name."""
    if env_name not in BENCHMARKS:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[env_name](**kwargs)


def get_env_info(env) -> Dict[str, int]:
    """Extract environment dimensions."""
    info = env.get_env_info()
    return {
        'n_agents': info['n_agents'],
        'obs_dim': info['obs_dim'],
        'act_dim': info['act_dim']
    }



# H3C Training Loop (uses H3CTrainer's own interface)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def train_h3c(
    env,
    env_info: Dict,
    total_steps: int,
    eval_interval: int,
    log_interval: int,
    device: str,
    save_path: str
) -> Dict[str, Any]:
    """Train H3C using its native interface - OPTIMIZED VERSION."""
    
    if not H3C_AVAILABLE:
        raise ImportError("H3CTrainer not available. Check modules folder.")
    
    # H3C config
    config = H3C_CONFIG.copy()
    config['n_steps'] = total_steps
    config['device'] = device
    
    # Create H3C trainer with ITS signature: (obs_dim, action_dim, n_agents, config)
    trainer = H3CTrainer(
        obs_dim=env_info['obs_dim'],
        action_dim=env_info['act_dim'],
        n_agents=env_info['n_agents'],
        config=config
    )
    
    print(f"Training H3C on {env.__class__.__name__}")
    print(f"Agents: {env_info['n_agents']} | Steps: {total_steps}")
    print(f"Device: {device} | Batch: {config['mini_batch_size']} | Rollout: {config['rollout_length']}")
    print("=" * 60)
    
    # Training state
    step = 0
    episode = 0
    best_reward = float('-inf')
    eval_rewards = []
    episode_rewards = []
    next_log_step = log_interval
    next_eval_step = eval_interval
    
    # Pre-allocate for speed
    n_agents = env_info['n_agents']
    
    start_time = time.time()
    
    # Initial evaluation (fewer episodes for speed)
    print("\nRunning initial evaluation...")
    init_mean, init_std, init_win = evaluate_h3c(trainer, env, env_info, n_episodes=5)
    print(f"Initial: {init_mean:.2f} ± {init_std:.2f}\n")
    eval_rewards.append({'step': 0, 'mean': init_mean, 'std': init_std, 'win_rate': init_win})
    best_reward = init_mean
    best_win_rate = init_win
    
    while step < total_steps:
        # Reset environment and episode
        obs_list = env.reset()
        obs = np.array(obs_list, dtype=np.float32)  # Explicit dtype for speed
        trainer.reset_episode()
        
        episode_reward = 0.0
        done = False
        
        while not done and step < total_steps:
            # Get actions from H3C
            actions, action_probs, log_probs, values = trainer.get_actions(obs, explore=True)
            
            # Step environment
            next_obs_list, rewards, dones, infos = env.step(actions.tolist() if isinstance(actions, np.ndarray) else actions)
            next_obs = np.array(next_obs_list, dtype=np.float32)
            
            # Convert rewards/dones efficiently
            if isinstance(rewards, list):
                rewards = np.array(rewards, dtype=np.float32)
            if isinstance(dones, list):
                dones_arr = np.array(dones, dtype=np.float32)
            else:
                dones_arr = np.full(n_agents, float(dones), dtype=np.float32)
            
            # Store transition
            trainer.store_transition(obs, actions, rewards, dones_arr, values, log_probs, action_probs)
            
            episode_reward += float(np.mean(rewards))
            step += 1
            
            # Check if should update
            if trainer.should_update():
                metrics = trainer.update(next_obs)
                
                # Log progress at exact intervals
                if step >= next_log_step:
                    elapsed = (time.time() - start_time) / 60
                    avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    steps_per_sec = step / (elapsed * 60) if elapsed > 0 else 0
                    print(f"Step {next_log_step:>7} | Train: {avg_reward:>8.2f} | Loss: {metrics['policy_loss']:.4f} | H: {metrics['entropy']:.2f} | Time: {elapsed:.1f}min ({steps_per_sec:.0f}/s)")
                    next_log_step += log_interval
            
            # Update observation
            obs = next_obs
            
            # Check done
            if isinstance(dones, (list, np.ndarray)):
                done = all(dones) if isinstance(dones, list) else bool(np.all(dones))
            else:
                done = bool(dones)
        
        # End episode
        trainer.end_episode(episode_reward)
        episode += 1
        episode_rewards.append(episode_reward)
        
        # Evaluation at exact intervals (fewer episodes for speed, same quality)
        if step >= next_eval_step:
            eval_mean, eval_std, eval_win = evaluate_h3c(trainer, env, env_info, n_episodes=10)
            
            # Check recovery
            trainer.check_recovery(eval_mean)
            
            is_best = eval_mean > best_reward
            if is_best:
                best_reward = eval_mean
                best_win_rate = eval_win
                if save_path:
                    trainer.save(os.path.join(save_path, "best_model.pt"))
            
            eval_rewards.append({'step': next_eval_step, 'mean': eval_mean, 'std': eval_std, 'win_rate': eval_win})
            
            if is_best:
                print(f"[EVAL] Step {next_eval_step:>6} | Reward: {eval_mean:.2f} ± {eval_std:.2f} | ★ NEW BEST: {eval_mean:.2f} | Win: {eval_win:.1f}%")
            else:
                print(f"[EVAL] Step {next_eval_step:>6} | Reward: {eval_mean:.2f} ± {eval_std:.2f} | Best: {best_reward:.2f} | Win: {eval_win:.1f}%")
            
            next_eval_step += eval_interval
    
    # Final evaluation
    final_mean, final_std, final_win = evaluate_h3c(trainer, env, env_info, n_episodes=10)
    
    # Compute metrics
    eval_values = [e['mean'] for e in eval_rewards]
    auc = np.trapz(eval_values) / len(eval_values) if eval_values else 0
    variance = np.var(eval_values) if len(eval_values) > 1 else 0
    mean_reward = np.mean(eval_values) if eval_values else 0
    cv = np.std(eval_values) / (np.abs(mean_reward) + 1e-8) if len(eval_values) > 1 else 0
    stability = 1 / (1 + cv)
    
    # Steps to 90% - FIXED for negative rewards
    if best_reward >= 0:
        target_90 = best_reward * 0.9
    else:
        target_90 = best_reward * 0.9  # For negative: e.g., -26.73 * 0.9 = -24.06
    
    steps_to_90 = None
    for e in eval_rewards:
        if e['mean'] >= target_90:
            steps_to_90 = e['step']
            break
    
    # Convergence speed - FIXED calculation
    if steps_to_90 and steps_to_90 > 0:
        if best_reward >= 0:
            convergence_speed = best_reward / steps_to_90
        else:
            initial_reward = eval_rewards[0]['mean'] if eval_rewards else 0
            improvement = best_reward - initial_reward
            convergence_speed = abs(improvement) / steps_to_90
    else:
        if len(eval_rewards) >= 2:
            initial_reward = eval_rewards[0]['mean']
            improvement = best_reward - initial_reward
            convergence_speed = abs(improvement) / step if step > 0 else 0
        else:
            convergence_speed = 0
    
    results = {
        'algorithm': 'H3C',
        'best': best_reward,
        'final_mean': final_mean,
        'final_std': final_std,
        'final_win_rate': final_win,
        'auc': auc,
        'variance': variance,
        'cv': cv,
        'stability': stability,
        'steps_to_90': steps_to_90,
        'convergence_speed': convergence_speed,
        'total_steps': step,
        'eval_history': eval_rewards
    }
    
    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "results.json"), 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'eval_history'}, f, indent=2)
        np.savez(os.path.join(save_path, "history.npz"), eval_history=eval_rewards)
    
    print(f"\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Final: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Best:  {best_reward:.2f}")
    print(f"Win Rate: {final_win:.1f}%")
    print("")
    print(f" Stability Metrics:")
    print(f"   Variance: {variance:.4f}")
    print(f"   CV: {cv:.4f}")
    print(f"   Stability Score: {stability:.4f}")
    print("")
    print(f" Convergence Metrics:")
    print(f"   AUC: {auc:.2f}")
    print(f"   Steps to 90%: {steps_to_90 if steps_to_90 else 'N/A'}")
    print(f"   Convergence Speed: {convergence_speed:.4f}")
    print("=" * 60 + "\n")
    
    return results


def evaluate_h3c(trainer, env, env_info, n_episodes: int = 10) -> tuple:
    """Evaluate H3C trainer. Returns (mean, std, win_rate). OPTIMIZED."""
    rewards = []
    wins = 0
    n_agents = env_info['n_agents']
    
    for _ in range(n_episodes):
        obs_list = env.reset()
        obs = np.array(obs_list, dtype=np.float32)
        trainer.reset_episode()
        
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 25
        
        while not done and steps < max_steps:
            actions, _, _, _ = trainer.get_actions(obs, explore=False)
            next_obs_list, reward, dones, info = env.step(actions.tolist() if isinstance(actions, np.ndarray) else actions)
            obs = np.array(next_obs_list, dtype=np.float32)
            
            episode_reward += float(np.mean(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward)
            steps += 1
            
            if isinstance(dones, (list, np.ndarray)):
                done = all(dones) if isinstance(dones, list) else bool(np.all(dones))
            else:
                done = bool(dones)
        
        rewards.append(episode_reward)
        # Count as win if positive reward
        if isinstance(info, dict) and info.get('win', False):
            wins += 1
        elif episode_reward > 0:
            wins += 1
    
    win_rate = wins / n_episodes * 100
    return float(np.mean(rewards)), float(np.std(rewards)), win_rate



# Baseline Training (uses BaseTrainer interface)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def train_baseline(
    algo_name: str,
    env,
    env_info: Dict,
    total_steps: int,
    eval_interval: int,
    log_interval: int,
    device: str,
    save_path: str
) -> Dict[str, Any]:
    """Train a baseline algorithm."""
    
    trainer_class = BASELINE_ALGORITHMS[algo_name]
    config = BASELINE_CONFIGS.get(algo_name, {}).copy()
    
    # Create trainer with BaseTrainer signature
    trainer = trainer_class(
        env=env,
        n_agents=env_info['n_agents'],
        obs_dim=env_info['obs_dim'],
        act_dim=env_info['act_dim'],
        config=config,
        device=device
    )
    
    # Use BaseTrainer's train method
    results = trainer.train(
        total_steps=total_steps,
        eval_interval=eval_interval,
        log_interval=log_interval,
        save_path=save_path
    )
    
    return results



# Main Training Function


def train_single(
    algo_name: str,
    env_name: str,
    total_steps: int,
    eval_interval: int,
    log_interval: int,
    device: str,
    seed: int,
    results_dir: str
) -> Dict[str, Any]:
    """Train a single algorithm on a single environment."""
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environment
    print(f"\n{'='*60}")
    print(f"Creating environment: {env_name}")
    print(f"{'='*60}")
    
    env = create_env(env_name)
    env_info = get_env_info(env)
    
    print(f"  Agents: {env_info['n_agents']}")
    print(f"  Obs dim: {env_info['obs_dim']}")
    print(f"  Act dim: {env_info['act_dim']}")
    
    # Setup save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, env_name, algo_name, timestamp)
    os.makedirs(save_path, exist_ok=True)
    
    # Train
    if algo_name == 'H3C':
        results = train_h3c(
            env=env,
            env_info=env_info,
            total_steps=total_steps,
            eval_interval=eval_interval,
            log_interval=log_interval,
            device=device,
            save_path=save_path
        )
    else:
        results = train_baseline(
            algo_name=algo_name,
            env=env,
            env_info=env_info,
            total_steps=total_steps,
            eval_interval=eval_interval,
            log_interval=log_interval,
            device=device,
            save_path=save_path
        )
    
    env.close()
    return results


def train_all_algorithms(
    env_name: str,
    algorithms: List[str],
    total_steps: int,
    eval_interval: int,
    log_interval: int,
    device: str,
    seed: int,
    results_dir: str
) -> Dict[str, Dict[str, Any]]:
    """Train all specified algorithms on an environment."""
    all_results = {}
    
    for algo_name in algorithms:
        print(f"\n{'#'*60}")
        print(f"# Training {algo_name} on {env_name}")
        print(f"{'#'*60}")
        
        # RESET SEED before each algorithm for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        try:
            results = train_single(
                algo_name=algo_name,
                env_name=env_name,
                total_steps=total_steps,
                eval_interval=eval_interval,
                log_interval=log_interval,
                device=device,
                seed=seed,
                results_dir=results_dir
            )
            all_results[algo_name] = results
            
        except Exception as e:
            print(f"ERROR training {algo_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results



# Results Summary


def print_summary(results: Dict[str, Dict[str, Any]], env_name: str):
    """Print summary table of results."""
    print(f"\n{'='*70}")
    print(f" RESULTS SUMMARY: {env_name}")
    print(f"{'='*70}")
    print(f"{'Algorithm':<12} {'Best':>12} {'Final':>12} {'AUC':>12} {'Stability':>10}")
    print(f"{'-'*70}")
    
    sorted_algos = sorted(results.keys(), 
                         key=lambda x: results[x].get('best', float('-inf')),
                         reverse=True)
    
    for i, algo in enumerate(sorted_algos):
        data = results[algo]
        prefix = "WIN " if i == 0 else "   "
        print(
            f"{prefix}{algo:<9} "
            f"{data.get('best', 0):>12.2f} "
            f"{data.get('final_mean', 0):>12.2f} "
            f"{data.get('auc', 0):>12.2f} "
            f"{data.get('stability', 0):>10.4f}"
        )
    
    print(f"{'='*70}")


# Main Entry Point

def parse_args():
    parser = argparse.ArgumentParser(description='H3C-BEACON Training Script')
    
    parser.add_argument('--algo', type=str, default='H3C',
                       help='Algorithm (H3C, MAPPO, IPPO, QMIX, VDN, COMA, FACMAC, or "all")')
    parser.add_argument('--env', type=str, default='simple_spread',
                       help='Environment (simple_spread, simple_world_comm, 27m_vs_30m, academy_3_vs_1_with_keeper, or "all")')
    parser.add_argument('--steps', type=int, default=1000000,
                       help='Total training steps')
    parser.add_argument('--eval-interval', type=int, default=160000,
                       help='Steps between evaluations')
    parser.add_argument('--log-interval', type=int, default=32000,
                       help='Steps between logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cpu, cuda, or auto')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--results-dir', type=str, default='resultats',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate plots after training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device selection with detailed info
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n🚀 GPU detected: {gpu_name}")
        else:
            device = 'cpu'
            print(f"\n⚠️  No GPU detected - using CPU (will be slower)")
    else:
        device = args.device
    
    # Enable optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # Faster convolutions
    
    print(f"\n{'='*60}")
    print(f" H3C-BEACON Training (Optimized)")
    print(f"{'='*60}")
    print(f" Algorithm(s): {args.algo}")
    print(f" Environment(s): {args.env}")
    print(f" Total Steps: {args.steps:,}")
    print(f" Device: {device}")
    print(f" Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    # Determine algorithms
    if args.algo.lower() == 'all':
        algorithms = ['H3C'] + list(BASELINE_ALGORITHMS.keys())
    else:
        algorithms = [args.algo]
    
    # Determine environments
    if args.env.lower() == 'all':
        environments = list(BENCHMARKS.keys())
    else:
        environments = [args.env]
    
    # Validate
    all_algos = {'H3C'} | set(BASELINE_ALGORITHMS.keys())
    for algo in algorithms:
        if algo not in all_algos:
            print(f"ERROR: Unknown algorithm '{algo}'")
            print(f"Available: {sorted(all_algos)}")
            sys.exit(1)
    
    for env in environments:
        if env not in BENCHMARKS:
            print(f"ERROR: Unknown environment '{env}'")
            print(f"Available: {list(BENCHMARKS.keys())}")
            sys.exit(1)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Training
    start_time = time.time()
    all_results = {}
    
    for env_name in environments:
        print(f"\n{'*'*60}")
        print(f"* Environment: {env_name}")
        print(f"{'*'*60}")
        
        all_results[env_name] = train_all_algorithms(
            env_name=env_name,
            algorithms=algorithms,
            total_steps=args.steps,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            device=device,
            seed=args.seed,
            results_dir=args.results_dir
        )
    
    elapsed = (time.time() - start_time) / 60
    
    # Print summaries
    print(f"\n{'#'*60}")
    print(f"# TRAINING COMPLETE")
    print(f"# Total time: {elapsed:.1f} minutes")
    print(f"{'#'*60}")
    
    for env_name, env_results in all_results.items():
        print_summary(env_results, env_name)
    
    # Visualization
    if args.visualize:
        print("\nGenerating visualizations...")
        visualizer = ResultsVisualizer(args.results_dir)
        for env_name in all_results.keys():
            visualizer.generate_all_plots(env_name)
    
    print("\nDone!")


if __name__ == '__main__':
    main()