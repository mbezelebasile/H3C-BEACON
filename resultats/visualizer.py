"""
H3C-BEACON: Results Visualization
Generate plots and tables for experimental results.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


# Color scheme
COLORS = {
    'H3C': '#E74C3C',      # Red (our method)
    'MAPPO': '#3498DB',    # Blue
    'IPPO': '#9B59B6',     # Purple
    'QMIX': '#2ECC71',     # Green
    'VDN': '#F39C12',      # Orange
    'COMA': '#1ABC9C',     # Teal
    'FACMAC': '#E67E22'    # Dark Orange
}

MARKERS = {
    'H3C': 'o', 'MAPPO': 's', 'IPPO': '^',
    'QMIX': 'D', 'VDN': 'v', 'COMA': 'p', 'FACMAC': 'h'
}


class ResultsVisualizer:
    """Visualize and compare experimental results."""
    
    def __init__(self, results_dir: str = "resultats"):
        self.results_dir = results_dir
    
    def load_results(self, env_name: str) -> Dict[str, Dict]:
        """Load results for all algorithms on an environment."""
        results = {}
        env_dir = os.path.join(self.results_dir, env_name)
        
        if not os.path.exists(env_dir):
            return results
        
        for algo_name in os.listdir(env_dir):
            algo_dir = os.path.join(env_dir, algo_name)
            if os.path.isdir(algo_dir):
                runs = sorted([d for d in os.listdir(algo_dir) if os.path.isdir(os.path.join(algo_dir, d))])
                if runs:
                    latest_run = os.path.join(algo_dir, runs[-1])
                    
                    results_file = os.path.join(latest_run, "results.json")
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            results[algo_name] = json.load(f)
                    
                    history_file = os.path.join(latest_run, "history.npz")
                    if os.path.exists(history_file):
                        history = np.load(history_file, allow_pickle=True)
                        if algo_name in results:
                            results[algo_name]['eval_history'] = history['eval_history'].tolist()
        
        return results
    
    def plot_learning_curves(self, results: Dict[str, Dict], env_name: str, 
                             save_path: Optional[str] = None):
        """Plot learning curves for all algorithms."""
        plt.figure(figsize=(12, 6))
        
        for algo_name, data in results.items():
            if 'eval_history' in data:
                steps = [e['step'] for e in data['eval_history']]
                rewards = [e['mean'] for e in data['eval_history']]
                stds = [e.get('std', 0) for e in data['eval_history']]
                
                color = COLORS.get(algo_name, '#666666')
                marker = MARKERS.get(algo_name, 'o')
                lw = 3.5 if algo_name == 'H3C' else 1.8
                alpha = 1.0 if algo_name == 'H3C' else 0.7
                label = f"🏆 {algo_name}" if algo_name == 'H3C' else algo_name
                
                plt.plot(np.array(steps)/1000, rewards, color=color, marker=marker,
                        markersize=6, linewidth=lw, alpha=alpha, label=label)
                plt.fill_between(np.array(steps)/1000,
                               np.array(rewards) - np.array(stds),
                               np.array(rewards) + np.array(stds),
                               color=color, alpha=0.1)
        
        plt.xlabel('Steps (×1000)', fontsize=12)
        plt.ylabel('Evaluation Reward', fontsize=12)
        plt.title(f'{env_name}: Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_performance_bars(self, results: Dict[str, Dict], env_name: str,
                              save_path: Optional[str] = None):
        """Plot bar comparison of metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algorithms = sorted(results.keys(), 
                          key=lambda x: results[x].get('best', float('-inf')), 
                          reverse=True)
        colors = [COLORS.get(a, '#666') for a in algorithms]
        
        # Best Reward
        ax = axes[0]
        values = [results[a].get('best', 0) for a in algorithms]
        bars = ax.bar(algorithms, values, color=colors, edgecolor='black')
        if algorithms and algorithms[0] == 'H3C':
            bars[0].set_edgecolor('#8B0000')
            bars[0].set_linewidth(3)
        ax.set_ylabel('Best Reward')
        ax.set_title('Best Reward')
        ax.tick_params(axis='x', rotation=45)
        
        # AUC
        ax = axes[1]
        values = [results[a].get('auc', 0) for a in algorithms]
        bars = ax.bar(algorithms, values, color=colors, edgecolor='black')
        if algorithms and algorithms[0] == 'H3C':
            bars[0].set_edgecolor('#8B0000')
            bars[0].set_linewidth(3)
        ax.set_ylabel('AUC')
        ax.set_title('Area Under Curve')
        ax.tick_params(axis='x', rotation=45)
        
        # Stability
        ax = axes[2]
        values = [results[a].get('stability', 0) for a in algorithms]
        bars = ax.bar(algorithms, values, color=colors, edgecolor='black')
        if algorithms and algorithms[0] == 'H3C':
            bars[0].set_edgecolor('#8B0000')
            bars[0].set_linewidth(3)
        ax.set_ylabel('Stability')
        ax.set_title('Stability Score')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1])
        
        plt.suptitle(f'{env_name}: Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_improvement(self, results: Dict[str, Dict], env_name: str,
                         baseline: str = 'H3C', save_path: Optional[str] = None):
        """Plot improvement of H3C over baselines."""
        if baseline not in results:
            print(f"Baseline {baseline} not found")
            return
        
        baseline_best = results[baseline].get('best', 0)
        
        plt.figure(figsize=(10, 6))
        
        algorithms = [a for a in results.keys() if a != baseline]
        improvements = []
        
        for algo in algorithms:
            algo_best = results[algo].get('best', 0)
            imp = (algo_best - baseline_best) / (abs(algo_best) + 1e-8) * 100
            improvements.append(imp)
        
        colors = [COLORS.get(a, '#666') for a in algorithms]
        
        bars = plt.barh(algorithms, improvements, color=colors, edgecolor='black', height=0.6)
        plt.xlabel('Improvement (%)')
        plt.title(f'{env_name}: {baseline} vs Baselines', fontsize=13, fontweight='bold')
        plt.axvline(x=0, color='black', linewidth=0.8)
        
        for bar, imp in zip(bars, improvements):
            x = imp + 1 if imp >= 0 else imp - 8
            plt.text(x, bar.get_y() + bar.get_height()/2, f'+{imp:.1f}%', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def generate_summary_table(self, results: Dict[str, Dict], env_name: str) -> str:
        """Generate text summary table."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f" Results: {env_name}")
        lines.append(f"{'='*80}")
        lines.append(f"{'Algorithm':<12} {'Best':>10} {'Final':>10} {'AUC':>10} {'Stability':>10}")
        lines.append(f"{'-'*80}")
        
        algorithms = sorted(results.keys(),
                          key=lambda x: results[x].get('best', float('-inf')),
                          reverse=True)
        
        for i, algo in enumerate(algorithms):
            data = results[algo]
            prefix = "🏆 " if i == 0 else "   "
            lines.append(
                f"{prefix}{algo:<9} "
                f"{data.get('best', 0):>10.2f} "
                f"{data.get('final_mean', 0):>10.2f} "
                f"{data.get('auc', 0):>10.2f} "
                f"{data.get('stability', 0):>10.4f}"
            )
        
        lines.append(f"{'='*80}\n")
        return '\n'.join(lines)
    
    def generate_all_plots(self, env_name: str):
        """Generate all visualization plots."""
        results = self.load_results(env_name)
        
        if not results:
            print(f"No results found for {env_name}")
            return
        
        output_dir = os.path.join(self.results_dir, env_name, "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_learning_curves(results, env_name,
                                  os.path.join(output_dir, "learning_curves.png"))
        self.plot_performance_bars(results, env_name,
                                   os.path.join(output_dir, "performance.png"))
        self.plot_improvement(results, env_name,
                             save_path=os.path.join(output_dir, "improvement.png"))
        
        print(self.generate_summary_table(results, env_name))
        print(f"Plots saved to: {output_dir}")


def create_dashboard(results_by_env: Dict[str, Dict[str, Dict]], 
                     save_path: Optional[str] = None):
    """Create comprehensive dashboard."""
    n_envs = len(results_by_env)
    fig, axes = plt.subplots(n_envs, 3, figsize=(16, 5*n_envs))
    
    if n_envs == 1:
        axes = axes.reshape(1, -1)
    
    for row, (env_name, results) in enumerate(results_by_env.items()):
        algorithms = sorted(results.keys(),
                          key=lambda x: results[x].get('best', float('-inf')),
                          reverse=True)
        
        # Learning curves
        ax = axes[row, 0]
        for algo in algorithms:
            if 'eval_history' in results[algo]:
                steps = [e['step']/1000 for e in results[algo]['eval_history']]
                rewards = [e['mean'] for e in results[algo]['eval_history']]
                lw = 3 if algo == 'H3C' else 1.5
                ax.plot(steps, rewards, color=COLORS.get(algo, '#666'),
                       linewidth=lw, label=algo)
        ax.set_xlabel('Steps (×1000)')
        ax.set_ylabel('Reward')
        ax.set_title(f'{env_name}: Learning Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Best rewards
        ax = axes[row, 1]
        colors = [COLORS.get(a, '#666') for a in algorithms]
        best = [results[a].get('best', 0) for a in algorithms]
        ax.bar(algorithms, best, color=colors, edgecolor='black')
        ax.set_ylabel('Best Reward')
        ax.set_title(f'{env_name}: Best Reward')
        ax.tick_params(axis='x', rotation=45)
        
        # Stability
        ax = axes[row, 2]
        stab = [results[a].get('stability', 0) for a in algorithms]
        ax.bar(algorithms, stab, color=colors, edgecolor='black')
        ax.set_ylabel('Stability')
        ax.set_title(f'{env_name}: Stability')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1])
    
    plt.suptitle('H3C-BEACON: Experimental Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()