#!/usr/bin/env python3
"""
H3C-BEACON Ablation Study Runner v1.0
Usage:
    # Étude complète (7 variantes × 3 seeds)
    python run_ablation.py --env simple_spread --steps 250000 --seeds 3
    
    # Variantes spécifiques
    python run_ablation.py --env simple_spread --variants full no_dgat no_bayesian
    
    # Test rapide (1 seed, 50K steps)
    python run_ablation.py --env simple_spread --steps 50000 --seeds 1 --quick

Author: H3C-BEECON Team
Version: 1.0
"""

import argparse
import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional
import copy

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import H3C modules
from modules.H3ctrainer import H3CTrainer, AblationConfig, ABLATION_VARIANTS



# ENVIRONMENT SETUP
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_simple_spread_env():
    """Create MPE Simple Spread environment."""
    try:
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
        env.reset()
        return env
    except ImportError:
        print("ERROR: pettingzoo not installed. Run: pip install pettingzoo[mpe]")
        sys.exit(1)


def create_simple_world_comm_env():
    """Create MPE Simple World Comm environment."""
    try:
        from pettingzoo.mpe import simple_world_comm_v3
        env = simple_world_comm_v3.parallel_env(num_good=2, num_adversaries=4, 
                                                  num_obstacles=1, max_cycles=25,
                                                  continuous_actions=False)
        env.reset()
        return env
    except ImportError:
        print("ERROR: pettingzoo not installed. Run: pip install pettingzoo[mpe]")
        sys.exit(1)


ENVIRONMENTS = {
    'simple_spread': {
        'create_fn': create_simple_spread_env,
        'n_agents': 3,
        'obs_dim': 18,
        'action_dim': 5,
        'win_threshold': -25,
    },
    'simple_world_comm': {
        'create_fn': create_simple_world_comm_env,
        'n_agents': 2,  # Good agents only for training
        'obs_dim': 34,
        'action_dim': 5,
        'win_threshold': -40,
    }
}



# PAPER ABLATION VARIANTS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PAPER_VARIANTS = [
    "full",           # H3C-BEACON (all components)
    "no_dgat",        # − DGAT
    "no_bayesian",    # − Bayesian Fusion
    "no_coalition",   # − Coalition Formation
    "no_dual_critic", # − Dual Critic
    "no_rtd_elite",   # − RTD++ Elite
    "no_entropy",     # − Entropy Annealing
]



# TRAINING LOOP


def train_ablation_variant(
    variant_name: str,
    env_name: str,
    total_steps: int,
    seed: int,
    config: Dict[str, Any],
    eval_interval: int = 10000,
    eval_episodes: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a single ablation variant.
    
    Returns:
        Dict with training results and metrics
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Get ablation config
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_name}")
    
    ablation = ABLATION_VARIANTS[variant_name]
    
    # Create environment
    env_config = ENVIRONMENTS[env_name]
    env = env_config['create_fn']()
    
    # Update config
    config = config.copy()
    config['n_steps'] = total_steps
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create trainer with ablation
    trainer = H3CTrainer(
        obs_dim=env_config['obs_dim'],
        action_dim=env_config['action_dim'],
        n_agents=env_config['n_agents'],
        config=config,
        ablation=ablation
    )
    
    # Training tracking
    episode_rewards = []
    eval_rewards = []
    best_reward = float('-inf')
    step = 0
    episode = 0
    
    start_time = time.time()
    
    while step < total_steps:
        # Reset environment
        obs_dict, _ = env.reset(seed=seed + episode)
        agents = list(obs_dict.keys())
        obs = np.array([obs_dict[a] for a in agents])
        
        trainer.reset_episode()
        episode_reward = 0
        done = False
        
        while not done:
            # Get actions
            actions, action_probs, log_probs, values = trainer.get_actions(obs, explore=True)
            
            # Create action dict
            action_dict = {agents[i]: actions[i] for i in range(len(agents))}
            
            # Step environment
            next_obs_dict, rewards_dict, terms_dict, truncs_dict, _ = env.step(action_dict)
            
            # Process results
            next_obs = np.array([next_obs_dict[a] for a in agents])
            rewards = np.array([rewards_dict[a] for a in agents])
            dones = np.array([terms_dict[a] or truncs_dict[a] for a in agents]).astype(float)
            
            # Store transition
            trainer.store_transition(obs, actions, rewards, dones, values, log_probs, action_probs)
            
            episode_reward += rewards.mean()
            obs = next_obs
            step += 1
            
            # Check if episode done
            done = any(terms_dict.values()) or any(truncs_dict.values())
            
            # Update if buffer full
            if trainer.should_update():
                update_info = trainer.update(obs)
        
        # End episode
        trainer.end_episode(episode_reward)
        episode_rewards.append(episode_reward)
        episode += 1
        
        # Evaluation
        if step % eval_interval < env_config.get('max_cycles', 25) or step >= total_steps:
            eval_reward = evaluate(trainer, env, env_config, eval_episodes)
            eval_rewards.append((step, eval_reward))
            
            # Check recovery
            trainer.check_recovery(eval_reward)
            
            if eval_reward > best_reward:
                best_reward = eval_reward
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"  Step {step:>7}/{total_steps} | Eval: {eval_reward:>7.2f} | "
                      f"Best: {best_reward:>7.2f} | Time: {elapsed/60:.1f}m")
    
    # Final evaluation
    final_reward = evaluate(trainer, env, env_config, eval_episodes * 2)
    
    env.close()
    
    # Compute metrics
    elapsed = time.time() - start_time
    
    return {
        'variant': variant_name,
        'seed': seed,
        'env': env_name,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'total_episodes': episode,
        'total_steps': step,
        'training_time': elapsed,
        'ablation_config': {
            'use_dgat': ablation.use_dgat,
            'use_bayesian': ablation.use_bayesian,
            'use_coalition': ablation.use_coalition,
            'use_dual_critic': ablation.use_dual_critic,
            'use_rtd_elite': ablation.use_rtd_elite,
            'use_entropy_annealing': ablation.use_entropy_annealing,
        }
    }


def evaluate(trainer, env, env_config, n_episodes: int = 10) -> float:
    """Evaluate trainer on environment."""
    total_rewards = []
    agents = None
    
    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        if agents is None:
            agents = list(obs_dict.keys())
        obs = np.array([obs_dict[a] for a in agents])
        
        episode_reward = 0
        done = False
        
        while not done:
            actions, _, _, _ = trainer.get_actions(obs, explore=False)
            action_dict = {agents[i]: actions[i] for i in range(len(agents))}
            
            next_obs_dict, rewards_dict, terms_dict, truncs_dict, _ = env.step(action_dict)
            
            next_obs = np.array([next_obs_dict[a] for a in agents])
            rewards = np.array([rewards_dict[a] for a in agents])
            
            episode_reward += rewards.mean()
            obs = next_obs
            done = any(terms_dict.values()) or any(truncs_dict.values())
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


# ABLATION STUDY RUNNER


def run_ablation_study(
    env_name: str,
    variants: List[str],
    total_steps: int,
    n_seeds: int,
    config: Dict[str, Any],
    results_dir: str,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run complete ablation study with multiple seeds.
    """
    all_results = {}
    start_time = time.time()
    
    total_runs = len(variants) * n_seeds
    current_run = 0
    
    print("\n" + "=" * 70)
    print(" H3C-BEACON ABLATION STUDY v5.3")
    print("=" * 70)
    print(f" Environment:    {env_name}")
    print(f" Variants:       {len(variants)}")
    print(f" Seeds:          {n_seeds}")
    print(f" Total runs:     {total_runs}")
    print(f" Steps per run:  {total_steps:,}")
    print(f" Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70 + "\n")
    
    for variant_name in variants:
        variant_results = []
        
        for seed in range(n_seeds):
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] {variant_name} (seed={seed})")
            print("-" * 50)
            
            try:
                result = train_ablation_variant(
                    variant_name=variant_name,
                    env_name=env_name,
                    total_steps=total_steps,
                    seed=seed,
                    config=config,
                    verbose=verbose
                )
                variant_results.append(result)
                
                print(f"✓ Completed: Best={result['best_reward']:.2f}, "
                      f"Final={result['final_reward']:.2f}")
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Aggregate results across seeds
        if variant_results:
            all_results[variant_name] = aggregate_results(variant_results)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "=" * 70)
    print(f" ABLATION STUDY COMPLETE")
    print(f" Total time: {elapsed:.1f} minutes")
    print("=" * 70 + "\n")
    
    return all_results


def aggregate_results(results_list: List[Dict]) -> Dict[str, Any]:
    """Aggregate results from multiple seeds."""
    if not results_list:
        return {}
    
    bests = [r['best_reward'] for r in results_list]
    finals = [r['final_reward'] for r in results_list]
    times = [r['training_time'] for r in results_list]
    
    return {
        'variant': results_list[0]['variant'],
        'env': results_list[0]['env'],
        'ablation_config': results_list[0]['ablation_config'],
        'n_seeds': len(results_list),
        
        # Aggregated metrics
        'best_mean': float(np.mean(bests)),
        'best_std': float(np.std(bests)),
        'best_max': float(np.max(bests)),
        'best_min': float(np.min(bests)),
        
        'final_mean': float(np.mean(finals)),
        'final_std': float(np.std(finals)),
        
        'time_mean': float(np.mean(times)),
        
        # Per-seed results
        'per_seed': results_list
    }



# ANALYSIS AND REPORTING


def analyze_results(results: Dict[str, Dict], baseline: str = "full") -> Dict[str, Any]:
    """Analyze ablation results and compute relative impacts."""
    if baseline not in results:
        print(f"Warning: Baseline '{baseline}' not in results")
        return {}
    
    baseline_best = results[baseline]['best_mean']
    
    analysis = {
        'baseline': baseline,
        'baseline_performance': baseline_best,
        'variants': {}
    }
    
    for variant_name, variant_data in results.items():
        variant_best = variant_data['best_mean']
        
        # Compute impact (for negative rewards: less negative = better)
        absolute_diff = variant_best - baseline_best
        
        if baseline_best != 0:
            # For negative rewards: positive diff means worse (more negative)
            relative_diff = (variant_best - baseline_best) / abs(baseline_best) * 100
        else:
            relative_diff = 0
        
        analysis['variants'][variant_name] = {
            'best_mean': variant_best,
            'best_std': variant_data['best_std'],
            'absolute_diff': absolute_diff,
            'relative_diff': relative_diff,
            'degradation': -relative_diff if relative_diff < 0 else 0,
        }
    
    # Rank by impact (worst degradation first)
    ranked = sorted(
        [(k, v) for k, v in analysis['variants'].items() if k != baseline],
        key=lambda x: x[1]['relative_diff']
    )
    analysis['ranking'] = [v[0] for v in ranked]
    
    return analysis


def print_results_table(results: Dict[str, Dict], env_name: str):
    """Print formatted ablation results table."""
    analysis = analyze_results(results)
    
    print("\n" + "=" * 85)
    print(f" ABLATION STUDY RESULTS: {env_name}")
    print("=" * 85)
    print(f"{'Variant':<20} {'Best (mean±std)':<22} {'Δ Absolute':<15} {'Δ Relative':<15}")
    print("-" * 85)
    
    # Sort by performance (best first)
    sorted_variants = sorted(
        results.items(),
        key=lambda x: x[1]['best_mean'],
        reverse=True  # Higher (less negative) is better
    )
    
    for i, (variant_name, variant_data) in enumerate(sorted_variants):
        best_mean = variant_data['best_mean']
        best_std = variant_data['best_std']
        
        var_analysis = analysis['variants'].get(variant_name, {})
        abs_diff = var_analysis.get('absolute_diff', 0)
        rel_diff = var_analysis.get('relative_diff', 0)
        
        prefix = "WIN " if i == 0 else "   "
        
        print(f"{prefix}{variant_name:<17} {best_mean:>8.2f} ± {best_std:<10.2f} "
              f"{abs_diff:>+10.2f}     {rel_diff:>+10.1f}%")
    
    print("=" * 85)
    
    # Component importance ranking
    print("\n Component Importance (by degradation when removed):")
    print("-" * 50)
    
    importance = []
    for variant in analysis.get('ranking', []):
        if variant != 'full':
            degradation = analysis['variants'][variant]['degradation']
            component = variant.replace('no_', '').upper()
            importance.append((component, degradation))
    
    # Sort by degradation (highest first = most important)
    importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (component, degradation) in enumerate(importance):
        bar = "---" * int(degradation / 5) if degradation > 0 else ""
        print(f"   {i+1}. {component:<12} {degradation:>6.1f}% {bar}")
    
    print()


def save_results(results: Dict, analysis: Dict, save_dir: str, env_name: str):
    """Save results to JSON files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare serializable results
    serializable = {}
    for variant, data in results.items():
        serializable[variant] = {
            k: v for k, v in data.items()
            if k != 'per_seed'  # Exclude large per-seed data
        }
    
    # Save results
    results_file = os.path.join(save_dir, f"ablation_results_{env_name}.json")
    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Save analysis
    analysis_file = os.path.join(save_dir, f"ablation_analysis_{env_name}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save LaTeX table
    latex_file = os.path.join(save_dir, f"ablation_table_{env_name}.tex")
    save_latex_table(results, analysis, latex_file)
    
    print(f"\n Results saved to: {save_dir}/")
    print(f"   - ablation_results_{env_name}.json")
    print(f"   - ablation_analysis_{env_name}.json")
    print(f"   - ablation_table_{env_name}.tex")


def save_latex_table(results: Dict, analysis: Dict, filepath: str):
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study Results}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & Best Reward & $\Delta$ Abs. & $\Delta$ Rel. \\",
        r"\midrule",
    ]
    
    # Sort by performance
    sorted_vars = sorted(results.items(), key=lambda x: x[1]['best_mean'], reverse=True)
    
    for variant, data in sorted_vars:
        var_analysis = analysis['variants'].get(variant, {})
        
        name = variant.replace('_', r'\_')
        best = f"{data['best_mean']:.2f} $\\pm$ {data['best_std']:.2f}"
        abs_diff = f"{var_analysis.get('absolute_diff', 0):+.2f}"
        rel_diff = f"{var_analysis.get('relative_diff', 0):+.1f}\\%"
        
        if variant == 'full':
            lines.append(f"\\textbf{{{name}}} & \\textbf{{{best}}} & -- & -- \\\\")
        else:
            lines.append(f"{name} & {best} & {abs_diff} & {rel_diff} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))



# MAIN
def get_default_config() -> Dict[str, Any]:
    """Default H3C configuration."""
    return {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'lr_min_ratio': 0.2,
        'max_grad_norm': 0.5,
        'clip_epsilon': 0.2,
        'ppo_epochs': 4,
        'mini_batch_size': 64,
        'rollout_length': 256,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='H3C-BEACON Ablation Study v5.3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full study (7 variants × 3 seeds)
  python run_ablation.py --env simple_spread --steps 250000 --seeds 3

  # Specific variants
  python run_ablation.py --variants full no_dgat no_bayesian --steps 100000

  # Quick test
  python run_ablation.py --quick
        """
    )
    
    parser.add_argument('--env', type=str, default='simple_spread',
                       choices=list(ENVIRONMENTS.keys()),
                       help='Environment name')
    
    parser.add_argument('--variants', nargs='+', default=PAPER_VARIANTS,
                       help='Ablation variants to test')
    
    parser.add_argument('--steps', type=int, default=250000,
                       help='Training steps per variant')
    
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds')
    
    parser.add_argument('--results-dir', type=str, default='resultats/ablation',
                       help='Results directory')
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (50K steps, 1 seed, 3 variants)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbosity')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.steps = 50000
        args.seeds = 1
        args.variants = ['full', 'no_dgat', 'no_bayesian']
        print(" Quick test mode: 50K steps, 1 seed, 3 variants")
    
    # Validate variants
    for v in args.variants:
        if v not in ABLATION_VARIANTS:
            print(f"ERROR: Unknown variant '{v}'")
            print(f"Available: {list(ABLATION_VARIANTS.keys())}")
            sys.exit(1)
    
    # Get config
    config = get_default_config()
    
    # Run ablation study
    results = run_ablation_study(
        env_name=args.env,
        variants=args.variants,
        total_steps=args.steps,
        n_seeds=args.seeds,
        config=config,
        results_dir=args.results_dir,
        verbose=not args.quiet
    )
    
    # Analyze and display
    print_results_table(results, args.env)
    
    # Save results
    analysis = analyze_results(results)
    save_results(results, analysis, args.results_dir, args.env)
    
    print("✅ Ablation study complete!")


if __name__ == '__main__':
    main()