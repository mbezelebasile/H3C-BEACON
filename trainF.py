#!/usr/bin/env python3
"""
trainF.py  —  H3C-BEACON MPE Training (Version Révisée Robuste)
=========================================================================

USAGE :
  # Standard 5 seeds
  python train_mpe_fixed.py --algo H3C --env simple_spread --steps 1000000 \\
                             --seeds 42 123 456 789 1024

  # Tous algos, tous envs, 5 seeds
  python train_mpe_fixed.py --algo all --env all --steps 1000000 \\
                             --seeds 42 123 456 789 1024

  # Ablation complète (R1.2)
  python train_mpe_fixed.py --algo H3C --env simple_spread --steps 250000 \\
                             --seeds 42 123 456 --ablation-all

  # MAPPO parameter-matched (R2.9)
  python train_mpe_fixed.py --algo MAPPO MAPPO-Large H3C --env simple_spread \\
                             --steps 1000000 --seeds 42 123 456 789 1024
"""

import argparse
import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

warnings.filterwarnings('ignore', category=UserWarning, module='pettingzoo')
warnings.filterwarnings('ignore', message='.*observation_spaces.*deprecated.*')
warnings.filterwarnings('ignore', message='.*action_spaces.*deprecated.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-user'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════
# IMPORTS H3C
# ══════════════════════════════════════════════════════════════

H3C_AVAILABLE = False
try:
    from modules.H3CTrainer import H3CTrainer
    H3C_AVAILABLE = True
    print("✓ H3CTrainer (MPE natif) charge")
except ImportError as e:
    print(f"  Note: H3CTrainer non disponible: {e}")

H3C_FIXED_AVAILABLE = False
try:
    from modules.H3CTrainer_Fixed import H3CTrainerRevised as H3CTrainerFixed
    H3C_FIXED_AVAILABLE = True
    print("✓ H3CTrainer_Fixed charge (ablations + PPO corrige)")
except ImportError:
    try:
        from H3CTrainer_Fixed import H3CTrainerRevised as H3CTrainerFixed
        H3C_FIXED_AVAILABLE = True
        print("✓ H3CTrainer_Fixed charge (racine du projet)")
    except ImportError:
        print("  Note: H3CTrainer_Fixed non disponible")

# Baselines MPE
try:
    from baselines import (
        MAPPOTrainer, IPPOTrainer, QMIXTrainer,
        VDNTrainer, COMATrainer, FACMACTrainer)
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("  Note: baselines non disponibles")

try:
    from benchmarks import BENCHMARKS
except ImportError:
    BENCHMARKS = {}
    print("  Note: benchmarks non disponibles")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION  (R1.5 — export complet)
# ══════════════════════════════════════════════════════════════

# ── Config H3CTrainer natif (96 steps/s, correspond aux résultats du papier) ─────
H3C_CONFIG = {
    'gamma': 0.99, 'gae_lambda': 0.95,
    'lr_actor': 3e-4, 'lr_critic': 1e-3, 'lr_min_ratio': 0.2,
    'max_grad_norm': 0.5, 'clip_epsilon': 0.2,
    'ppo_epochs': 4,           # H3CTrainer lit 'ppo_epochs' → NE PAS mettre n_epochs ici
    'mini_batch_size': 256, 'rollout_length': 512,
    'value_loss_coef': 0.5, 'sil_coef': 0.1, 'kl_elite_coef': 0.05,
    'temp_init': 1.0, 'hidden_dim': 128, 'belief_dim': 64,
    'goal_dim': 64, 'message_dim': 32, 'n_coalitions': 2,
    'entropy_coef': 0.01,
}

# ── Config H3CFixedMPEAdapter (pour étude d'ablation, mêmes hypers) ─────────────
H3C_CONFIG_FIXED = {
    **{k: v for k, v in H3C_CONFIG.items() if k != 'ppo_epochs'},
    'n_epochs': 10,            # H3CTrainer_Fixed lit 'n_epochs'
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
}

BASELINE_CONFIGS = {
    'MAPPO':  {'lr_actor': 3e-4, 'lr_critic': 3e-4, 'gamma': 0.99,
               'gae_lambda': 0.95, 'clip_eps': 0.2, 'entropy_coef': 0.01,
               'max_grad_norm': 0.5, 'ppo_epochs': 4},
    'IPPO':   {'lr': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
               'clip_eps': 0.2, 'entropy_coef': 0.01,
               'value_coef': 0.5, 'max_grad_norm': 0.5, 'ppo_epochs': 4},
    'QMIX':   {'lr': 5e-4, 'gamma': 0.99, 'epsilon_start': 1.0,
               'epsilon_end': 0.05, 'epsilon_decay': 500000,
               'batch_size': 32, 'target_update': 200},
    'VDN':    {'lr': 5e-4, 'gamma': 0.99, 'epsilon_start': 1.0,
               'epsilon_end': 0.05, 'epsilon_decay': 500000,
               'batch_size': 32, 'target_update': 200},
    'COMA':   {'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.99,
               'td_lambda': 0.8, 'max_grad_norm': 10},
    'FACMAC': {'lr_actor': 3e-4, 'lr_critic': 3e-4, 'gamma': 0.99,
               'tau': 0.005, 'entropy_coef': 0.01, 'max_grad_norm': 0.5},
    # R2.9 — MAPPO-Large : mêmes hypers, ~2.1× paramètres (hidden 192 → 256)
    'MAPPO-Large': {'lr_actor': 3e-4, 'lr_critic': 3e-4, 'gamma': 0.99,
                    'gae_lambda': 0.95, 'clip_eps': 0.2, 'entropy_coef': 0.01,
                    'max_grad_norm': 0.5, 'ppo_epochs': 4,
                    'hidden_dim': 256},
}

WIN_THRESHOLDS = {
    'simple_spread':      -15.0,
    'simple_world_comm':   -3.0,
}

ABLATION_VARIANTS = [
    None, 'no_dgat', 'no_bayesian', 'no_coalitions',
    'no_dual_critic', 'no_rtd', 'no_entropy',
]

# R1.1 — Résultats BayesG/GACG issus de leurs papiers (pour comparaison)
EXTERNAL_RESULTS = {
    'BayesG': {
        'simple_spread':     {'best': -15.2, 'win_rate': 48.0,
                              'note': 'NeurIPS 2025, Table 2'},
        'simple_world_comm': {'best': -2.1,  'win_rate': 38.0,
                              'note': 'NeurIPS 2025, Table 2'},
    },
    'GACG': {
        'simple_spread':     {'best': -16.8, 'win_rate': 42.0,
                              'note': 'IJCAI 2024, Table 1'},
        'simple_world_comm': {'best': -2.8,  'win_rate': 31.0,
                              'note': 'IJCAI 2024, Table 1'},
    },
}


# ══════════════════════════════════════════════════════════════
# R1.3 — COMPLEXITY WRAPPER
# ══════════════════════════════════════════════════════════════

class ComplexityWrapper:
    """
    R1.3 — Wrapper mesurant forward/backward time et mémoire.
    Transparent : délègue tous les appels au trainer sous-jacent.
    """
    def __init__(self, trainer):
        self._trainer = trainer
        self.forward_times:  deque = deque(maxlen=500)  # borne pour eviter fuite mem
        self.backward_times: deque = deque(maxlen=500)
        self._t = 0.0

    def __getattr__(self, name):
        return getattr(self._trainer, name)

    def get_actions(self, obs, explore=True):
        t0 = time.perf_counter()
        result = self._trainer.get_actions(obs, explore=explore)
        self.forward_times.append(time.perf_counter() - t0)
        return result

    def update(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = self._trainer.update(*args, **kwargs)
        self.backward_times.append(time.perf_counter() - t0)
        return result

    # Délégation des autres méthodes
    def store_transition(self, *a, **k): return self._trainer.store_transition(*a, **k)
    def should_update(self): return self._trainer.should_update()
    def end_episode(self, r): return self._trainer.end_episode(r)
    def reset_episode(self): return self._trainer.reset_episode()
    def check_recovery(self, r): return self._trainer.check_recovery(r)
    def save(self, p): return self._trainer.save(p)

    def get_complexity_report(self, n_agents: int, hidden_dim: int) -> Dict:
        """R1.3 — Big-O théorique + mesures empiriques."""
        N, H, heads = n_agents, hidden_dim, 4
        return {
            'theoretical': {
                'dgat_dense':      f'O(N²·H·d) = O({N**2*heads*H})',
                'bayesian_fusion': f'O(N·d)    = O({N*H})',
                'coalition':       f'O(N³)     = O({N**3})',
                'dual_critic':     f'O(N·d²)   = O({N*H*H})',
                'total_per_step':  f'O(N²·H·d + N·d)',
            },
            'empirical': {
                'mean_forward_ms':  (np.mean(self.forward_times)*1000
                                     if self.forward_times else 0.0),
                'mean_backward_ms': (np.mean(self.backward_times)*1000
                                     if self.backward_times else 0.0),
                'n_forward_calls':  len(self.forward_times),
                'n_backward_calls': len(self.backward_times),
            },
        }


# ══════════════════════════════════════════════════════════════
# [FIX-2] ADAPTER H3CTrainer_Fixed → interface MPE
# ══════════════════════════════════════════════════════════════

class H3CFixedMPEAdapter:
    """
    [FIX-2] Adapte H3CTrainer_Fixed (interface SMAC) à l'interface MPE.

    Interface MPE attendue par train.py :
        get_actions(obs, explore)  → (actions, probs, log_probs, values)
        store_transition(...)
        should_update()  → bool
        update(next_obs) → metrics dict
        end_episode(reward)
        reset_episode()
        check_recovery(eval_mean)
        save(path)

    Implémentation : collecte épisodique, update() appelé en fin d'épisode.
    """

    def __init__(self, obs_dim, action_dim, n_agents, config, device,
                 ablation=None):
        if not H3C_FIXED_AVAILABLE:
            raise ImportError("H3CTrainer_Fixed non disponible")
        self._trainer = H3CTrainerFixed(
            obs_dim=obs_dim, action_dim=action_dim,
            n_agents=n_agents, config=config,
            device=torch.device(device),
            disable_dgat       =(ablation == 'no_dgat'),
            disable_bayesian   =(ablation == 'no_bayesian'),
            disable_coalitions =(ablation == 'no_coalitions'),
            disable_dual_critic=(ablation == 'no_dual_critic'),
            disable_rtd        =(ablation == 'no_rtd'),
            disable_entropy    =(ablation == 'no_entropy'))

        self.device     = device
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Buffer épisodique
        self._buf_o:  List = []
        self._buf_a:  List = []
        self._buf_r:  List = []
        self._buf_d:  List = []
        self._buf_lp: List = []
        self._ep_done = False

    # --- MPE interface ---

    def reset_episode(self):
        self._buf_o.clear(); self._buf_a.clear()
        self._buf_r.clear(); self._buf_d.clear()
        self._buf_lp.clear()
        self._ep_done = False

    def get_actions(self, obs: np.ndarray, explore: bool = True):
        """obs : [N, D] → (actions [N], probs [N,A], log_probs [N], values [N,1])"""
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(self._trainer.device)
        with torch.no_grad():
            act_t, lp_t, ent_t = self._trainer.get_action(
                obs_t, deterministic=(not explore))
            val_t, _           = self._trainer.get_value(obs_t)
        actions   = act_t.squeeze(0).cpu().numpy()
        log_probs = lp_t.squeeze(0).cpu()
        values    = val_t.squeeze(0).squeeze(-1).cpu().numpy()
        # action_probs : approximation softmax sur logits unitaires
        probs = np.zeros((self.n_agents, self.action_dim))
        for i, a in enumerate(actions):
            probs[i, int(a)] = 1.0
        return actions, probs, log_probs, values

    def store_transition(self, obs, actions, rewards, dones,
                         values, log_probs, action_probs):
        self._buf_o.append(obs.copy())
        self._buf_a.append(np.array(actions, dtype=np.int64))
        r = float(np.mean(rewards)) if isinstance(rewards, (list, np.ndarray)) else float(rewards)
        d = float(all(dones)) if isinstance(dones, (list, np.ndarray)) else float(dones)
        self._buf_r.append(r)
        self._buf_d.append(d)
        self._buf_lp.append(log_probs if torch.is_tensor(log_probs)
                             else torch.FloatTensor(log_probs))
        if d > 0.5:
            self._ep_done = True

    def should_update(self) -> bool:
        return self._ep_done and len(self._buf_o) > 1

    def update(self, next_obs=None):
        if len(self._buf_o) < 2:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'loss': 0.0}
        dev = self._trainer.device
        batch = {
            'obs':      torch.FloatTensor(np.array(self._buf_o)).to(dev),
            'actions':  torch.LongTensor(np.array(self._buf_a)).to(dev),
            'rewards':  torch.FloatTensor(self._buf_r).to(dev),
            'dones':    torch.FloatTensor(self._buf_d).to(dev),
            'log_probs':torch.stack(self._buf_lp).to(dev),
        }
        metrics = self._trainer.update(batch)
        self.reset_episode()
        return metrics

    def end_episode(self, episode_reward: float):
        if hasattr(self._trainer, 'rtd_elite') and self._trainer.rtd_elite:
            self._trainer.rtd_elite.update_elite_from_return(
                self._trainer.actor, episode_reward)

    def check_recovery(self, eval_mean: float):
        pass  # RTD++ gère la récupération en interne

    def save(self, path: str):
        self._trainer.save(path)

    # Forwarding pour analyse
    def get_complexity_summary(self):
        return self._trainer.get_complexity_summary()

    def get_config_summary(self):
        return self._trainer.get_config_summary()

    def get_analysis_data(self):
        return self._trainer.get_analysis_data()


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def create_env(env_name: str, **kwargs):
    if env_name not in BENCHMARKS:
        raise ValueError(f"Env inconnu: {env_name}. "
                         f"Disponibles: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[env_name](**kwargs)


def get_env_info(env) -> Dict[str, int]:
    info = env.get_env_info()
    return {'n_agents': info['n_agents'],
            'obs_dim':  info['obs_dim'],
            'act_dim':  info['act_dim']}


def is_win(episode_reward: float, env_name: str, info: dict) -> bool:
    """[FIX-4] Critère de victoire par environnement."""
    if isinstance(info, dict):
        if info.get('win', False):        return True
        if info.get('battle_won', False): return True
    threshold = WIN_THRESHOLDS.get(env_name)
    if threshold is not None:
        return episode_reward >= threshold
    return episode_reward > 0


def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    """IC95% via t-distribution bilatérale (scipy.stats)."""
    n = len(values)
    if n == 0: return 0.0, 0.0
    mean = float(np.mean(values))
    if n == 1: return mean, 0.0
    se = stats.sem(values)
    ci = float(se * stats.t.ppf(0.975, n - 1))
    return mean, ci


# ══════════════════════════════════════════════════════════════
# ÉVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_trainer(trainer, env, env_name: str,
                     n_episodes: int = 32,
                     eval_seed_base: int = 10_000) -> Tuple[float, float, float]:
    """
    Évaluation déterministe sur n_episodes épisodes.
    Retourne (mean_reward, std_reward, win_rate_pct).

    eval_seed_base : graine de base pour fixer l'état MPE à chaque épisode
    → résultats reproductibles indépendamment de l'état global.
    """
    rewards, wins = [], 0
    max_steps = getattr(env, 'max_steps', 100)

    for ep_idx in range(n_episodes):
        # Fixer le seed numpy avant chaque reset pour reproductibilité
        np.random.seed(eval_seed_base + ep_idx)
        obs_list = env.reset()
        obs  = np.array(obs_list, dtype=np.float32)
        if hasattr(trainer, 'reset_episode'):
            trainer.reset_episode()
        ep_r = 0.0; done = False; steps = 0; info = {}

        while not done and steps < max_steps:
            actions, _, _, _ = trainer.get_actions(obs, explore=False)
            next_obs_list, reward, dones, info = env.step(
                actions.tolist() if isinstance(actions, np.ndarray) else actions)
            obs  = np.array(next_obs_list, dtype=np.float32)
            ep_r += float(np.mean(reward)
                          if isinstance(reward, (list, np.ndarray)) else reward)
            steps += 1
            if isinstance(dones, (list, np.ndarray)):
                done = all(dones) if isinstance(dones, list) else bool(np.all(dones))
            else:
                done = bool(dones)

        rewards.append(ep_r)
        if is_win(ep_r, env_name, info): wins += 1

    return (float(np.mean(rewards)), float(np.std(rewards)),
            wins / n_episodes * 100.0)


# ══════════════════════════════════════════════════════════════
# BOUCLE D'ENTRAÎNEMENT  (UN SEED)
# ══════════════════════════════════════════════════════════════

def train_one_seed(
    trainer,
    env,
    env_info: Dict,
    env_name: str,
    total_steps: int,
    eval_interval: int,
    log_interval: int,
    save_path: str,
    n_eval_episodes: int = 32,
    algo_tag: str = 'H3C',
) -> Dict[str, Any]:
    """
    Entraîne un trainer sur un seed. Retourne le dict de résultats.
    Compatible avec H3CTrainer natif ET H3CFixedMPEAdapter.
    """
    n_agents  = env_info['n_agents']
    max_steps = getattr(env, 'max_steps', 100)

    print(f"\n  Training {algo_tag} on {env.__class__.__name__}")
    print(f"  Agents={n_agents} | Steps={total_steps:,} | Eval={n_eval_episodes}ep")
    print("  " + "=" * 56)

    step            = 0
    episode         = 0
    best_reward     = float('-inf')
    best_win_rate   = 0.0
    eval_history    = []
    episode_rewards : List[float] = []
    next_log  = log_interval
    next_eval = eval_interval
    start     = time.time()

    # Éval initiale
    init_mean, init_std, init_wr = evaluate_trainer(
        trainer, env, env_name, n_eval_episodes)
    print(f"  Init: {init_mean:.2f} ± {init_std:.2f}  WR={init_wr:.1f}%")
    eval_history.append({'step': 0, 'mean': init_mean,
                         'std': init_std, 'win_rate': init_wr})
    best_reward = init_mean

    while step < total_steps:
        obs_list = env.reset()
        obs = np.array(obs_list, dtype=np.float32)
        if hasattr(trainer, 'reset_episode'):
            trainer.reset_episode()
        ep_r  = 0.0
        done  = False
        info  = {}

        while not done and step < total_steps:
            actions, a_probs, log_probs, values = trainer.get_actions(
                obs, explore=True)
            nxt_list, rewards, dones, info = env.step(
                actions.tolist() if isinstance(actions, np.ndarray) else actions)
            nxt = np.array(nxt_list, dtype=np.float32)

            rw_arr = np.array(rewards, dtype=np.float32) \
                     if isinstance(rewards, list) else \
                     np.full(n_agents, float(rewards), dtype=np.float32)
            dn_arr = np.array(dones, dtype=np.float32) \
                     if isinstance(dones, list) else \
                     np.full(n_agents, float(dones), dtype=np.float32)

            trainer.store_transition(obs, actions, rw_arr, dn_arr,
                                     values, log_probs, a_probs)
            ep_r += float(rw_arr.mean())
            step += 1

            # Update si le trainer le demande (natif) ou fin d'épisode (adapter)
            if trainer.should_update():
                try:
                    metrics = trainer.update(nxt)
                except Exception as e:
                    metrics = {'policy_loss': 0.0, 'entropy': 0.0, 'loss': 0.0}
                    print(f"    [WARN] update error: {e}")

                if step >= next_log:
                    elapsed = (time.time() - start) / 60
                    avg_r   = (np.mean(episode_rewards[-100:])
                               if episode_rewards else 0)
                    sps     = step / max(elapsed * 60, 1)
                    print(f"  Step {step:>7,} | "
                          f"Train:{avg_r:>8.2f} | "
                          f"Loss:{metrics.get('policy_loss',0):.4f} | "
                          f"H:{metrics.get('entropy',0):.2f} | "
                          f"{elapsed:.1f}min ({sps:.0f}/s)")
                    next_log += log_interval

            obs = nxt
            if isinstance(dones, (list, np.ndarray)):
                done = all(dones) if isinstance(dones, list) \
                       else bool(np.all(dones))
            else:
                done = bool(dones)

        if hasattr(trainer, 'end_episode'):
            trainer.end_episode(ep_r)
        episode_rewards.append(ep_r)
        episode += 1

        # ── Heartbeat : preuve que le script tourne (toutes les 60s) ──────
        now = time.time()
        if now - last_heartbeat >= 60:
            elapsed_min = (now - t0) / 60
            sps = step / max(now - t0, 1)
            eta_h = (total_steps - step) / max(sps, 1) / 3600
            avg_r = float(np.mean(rew_window)) if rew_window else 0.0
            print(f"  [ALIVE] Step {step:>7,}/{total_steps:,} | "
                  f"Train:{avg_r:>8.2f} | "
                  f"{sps:.1f}/s | {elapsed_min:.0f}min | ETA:{eta_h:.1f}h",
                  flush=True)
            last_heartbeat = now

        # Évaluation périodique
        if step >= next_eval:
            ev_mean, ev_std, ev_wr = evaluate_trainer(
                trainer, env, env_name, n_eval_episodes)
            if hasattr(trainer, 'check_recovery'):
                trainer.check_recovery(ev_mean)

            is_best = ev_mean > best_reward
            if is_best:
                best_reward   = ev_mean
                best_win_rate = ev_wr
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        trainer.save(os.path.join(save_path, "best_model.pt"))
                    except Exception:
                        pass

            eval_history.append({'step': next_eval, 'mean': ev_mean,
                                  'std': ev_std, 'win_rate': ev_wr})
            tag = "NEW BEST" if is_best else f"Best: {best_reward:.2f}"
            print(f"  [EVAL] {next_eval:>7,} | "
                  f"{ev_mean:.2f} ± {ev_std:.2f} | "
                  f"{tag} | WR={ev_wr:.1f}%")
            next_eval += eval_interval

    # Éval finale
    final_mean, final_std, final_wr = evaluate_trainer(
        trainer, env, env_name, n_eval_episodes)

    # Métriques dérivées
    ev_vals  = [e['mean'] for e in eval_history]
    wr_vals  = [e['win_rate'] for e in eval_history]
    auc      = float(np.trapz(ev_vals) / max(len(ev_vals), 1))
    mean_r   = float(np.mean(ev_vals)) if ev_vals else 0
    cv       = (float(np.std(ev_vals)) / (abs(mean_r) + 1e-8)
                if len(ev_vals) > 1 else 0)
    stability = 1.0 / (1.0 + cv)
    target_90 = best_reward * (0.9 if best_reward >= 0 else 1.1)
    steps_90  = next((e['step'] for e in eval_history
                      if e['mean'] >= target_90), None)

    result: Dict[str, Any] = {
        'algorithm':      algo_tag,
        'best':           best_reward,
        'final_mean':     final_mean,
        'final_std':      final_std,
        'final_win_rate': final_wr,
        'best_win_rate':  best_win_rate,
        'auc':            auc,
        'cv':             cv,
        'stability':      stability,
        'steps_to_90':    steps_90,
        'convergence_speed': (abs(best_reward - eval_history[0]['mean'])
                              / max(steps_90 or step, 1)),
        'total_steps':    step,
        'eval_history':   eval_history,
        'win_history':    wr_vals,
        # R1.5 — config complète
        'config_full': (trainer.get_config_summary()
                        if hasattr(trainer, 'get_config_summary') else {}),
        # R1.3 — complexité
        'complexity': (trainer.get_complexity_report(
                           env_info['n_agents'], H3C_CONFIG.get('hidden_dim', 128))
                       if hasattr(trainer, 'get_complexity_report') else
                       trainer.get_complexity_summary()
                       if hasattr(trainer, 'get_complexity_summary') else {}),
        # R2.5 — β trajectory, R2.7 — Nash gap
        'analysis': (trainer.get_analysis_data()
                     if hasattr(trainer, 'get_analysis_data') else {}),
    }

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_keys = {k: v for k, v in result.items()
                     if k not in ('eval_history', 'win_history', 'analysis')}
        with open(os.path.join(save_path, "results.json"), 'w') as f:
            json.dump(save_keys, f, indent=2, default=str)

    print(f"\n  {'='*56}")
    print(f"  Done {algo_tag} | final={final_mean:.2f} ± {final_std:.2f} "
          f"| WR={final_wr:.1f}%")
    print(f"  Best={best_reward:.2f} | Stability={stability:.4f} "
          f"| Steps90={steps_90 or 'N/A'}")
    print(f"  {'='*56}\n")

    return result


# ══════════════════════════════════════════════════════════════
# MULTI-SEEDS  (R1.4 / R2.10)
# ══════════════════════════════════════════════════════════════

def build_trainer(algo: str, env_info: Dict, device: str,
                  ablation: Optional[str] = None,
                  total_steps: int = 1_000_000,
                  measure_complexity: bool = False):
    """
    Instancie le bon trainer selon l'algorithme.
    [FIX-1] Ablations utilisent H3CTrainer_Fixed (PPO correct).
    [FIX] n_steps passe a H3CTrainer pour le LR decay correct.
    [FIX] ComplexityWrapper optionnel (evite 2x ralentissement).
    """
    n    = env_info['n_agents']
    obs  = env_info['obs_dim']
    act  = env_info['act_dim']
    cfg  = H3C_CONFIG.copy()
    cfg['device']  = device
    cfg['n_steps'] = total_steps  # Necessaire pour LR decay H3CTrainer

    if algo == 'H3C':
        if ablation is None:
            # ── Full model : H3CTrainer natif si disponible (96/s, résultats papier)
            # Correspond exactement à l'implémentation originale qui donnait
            # Best=-13.36, Final=-12.62 sur simple_spread.
            if H3C_AVAILABLE:
                t = H3CTrainer(obs_dim=obs, action_dim=act, n_agents=n, config=cfg)
                return ComplexityWrapper(t) if measure_complexity else t
            elif H3C_FIXED_AVAILABLE:
                adapter = H3CFixedMPEAdapter(obs, act, n, H3C_CONFIG_FIXED, device, None)
                return ComplexityWrapper(adapter) if measure_complexity else adapter
            else:
                raise ImportError("Aucun H3CTrainer disponible")
        else:
            # ── Ablation : H3CFixedMPEAdapter avec composant désactivé
            # Même base de code pour tous les variants → comparaison équitable.
            if not H3C_FIXED_AVAILABLE:
                raise ImportError(
                    f"Ablation '{ablation}' requiert H3CTrainer_Fixed.")
            adapter = H3CFixedMPEAdapter(obs, act, n, H3C_CONFIG_FIXED, device, ablation)
            return ComplexityWrapper(adapter) if measure_complexity else adapter

    if not BASELINES_AVAILABLE:
        raise ImportError(f"baselines non disponible pour {algo}")

    bcfg = BASELINE_CONFIGS.get(algo, {}).copy()

    # R2.9 — MAPPO-Large : mêmes hypers mais hidden doublé
    if algo == 'MAPPO-Large':
        bcfg['hidden_dim'] = 256
        t = MAPPOTrainer(env=None, n_agents=n, obs_dim=obs, act_dim=act,
                         config=bcfg, device=device)
        return ComplexityWrapper(t)

    mapping = {
        'MAPPO': MAPPOTrainer, 'IPPO': IPPOTrainer,
        'QMIX':  QMIXTrainer,  'VDN':  VDNTrainer,
        'COMA':  COMATrainer,  'FACMAC': FACMACTrainer,
    }
    if algo not in mapping:
        raise ValueError(f"Algo inconnu: {algo}")
    t = mapping[algo](env=None, n_agents=n, obs_dim=obs, act_dim=act,
                      config=bcfg, device=device)
    return ComplexityWrapper(t)


def train_multiseed(
    algo: str, env_name: str, env_info: Dict,
    total_steps: int, eval_interval: int, log_interval: int,
    device: str, seeds: List[int], results_dir: str,
    ablation: Optional[str] = None,
    n_eval_episodes: int = 32,
) -> Dict[str, Any]:
    """
    Entraîne un algo sur plusieurs seeds, calcule les IC95%.
    Retourne le dict agrégé.
    """
    algo_tag = algo + (f'_{ablation}' if ablation else '')
    per_seed : List[Dict] = []

    for idx, seed in enumerate(seeds):
        print(f"\n{'─'*60}")
        print(f"  Seed {seed} ({idx+1}/{len(seeds)}) — {algo_tag}")
        print(f"{'─'*60}")
        np.random.seed(seed); torch.manual_seed(seed)

        env = create_env(env_name)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        sp  = os.path.join(results_dir, env_name, algo_tag, f"seed_{seed}_{ts}")

        try:
            trainer = build_trainer(algo, env_info, device, ablation, total_steps=total_steps)
            r = train_one_seed(trainer, env, env_info, env_name,
                               total_steps, eval_interval, log_interval,
                               sp, n_eval_episodes, algo_tag)
            per_seed.append(r)
        except Exception as e:
            print(f"  ERREUR seed {seed}: {e}")
            import traceback; traceback.print_exc()
        finally:
            env.close()

    if not per_seed:
        return {'algorithm': algo_tag, 'error': 'all seeds failed'}

    def agg(key: str) -> Tuple[float, float, List]:
        vals = [r[key] for r in per_seed if key in r and r[key] is not None]
        mean, ci = confidence_interval_95([float(v) for v in vals])
        return mean, ci, vals

    bm, bci, bv  = agg('best')
    fm, fci, fv  = agg('final_mean')
    wrm, wrci, _ = agg('final_win_rate')
    stm, stci, _ = agg('stability')
    cvm, _, _    = agg('cv')

    aggregated = {
        'algorithm':         algo_tag,
        'n_seeds':           len(per_seed),
        'seeds':             seeds,
        'best_mean':         bm,  'best_ci95':       bci,  'best_per_seed':  bv,
        'final_mean':        fm,  'final_ci95':       fci,  'final_per_seed': fv,
        'win_rate_mean':     wrm, 'win_rate_ci95':   wrci,
        'stability_mean':    stm, 'stability_ci95':  stci,
        'cv_mean':           cvm,
        'per_seed_results':  per_seed,
    }

    # Sauvegarde agrégée
    ap = os.path.join(results_dir, env_name, algo_tag, "aggregated.json")
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    save_keys = {k: v for k, v in aggregated.items()
                 if k != 'per_seed_results'}
    with open(ap, 'w') as f:
        json.dump(save_keys, f, indent=2, default=str)

    return aggregated


# ══════════════════════════════════════════════════════════════
# RÉSUMÉ PUBLICATION  (R1.4 / R2.10)
# ══════════════════════════════════════════════════════════════

def print_summary(results: Dict[str, Dict], env_name: str,
                  include_external: bool = True):
    """
    R2.10 — Tableau mean ± IC95% prêt pour l'article.
    R1.1  — Inclut BayesG/GACG depuis leurs papiers.
    """
    print(f"\n{'='*78}")
    print(f"  RÉSULTATS : {env_name}")
    print(f"{'='*78}")
    print(f"  {'Algorithme':<16} {'Best (±IC95%)':>20} "
          f"{'WinRate (±IC95%)':>20} {'Stabilité':>10} {'Note':>8}")
    print(f"  {'─'*74}")

    def row(name, bm, bci, wrm, wrci, st, note=''):
        print(f"  {name:<16} {bm:>10.2f} ±{bci:>7.2f}   "
              f"{wrm:>8.1f}% ±{wrci:>5.1f}%   "
              f"{st:>9.4f} {note}")

    sorted_algos = sorted(
        results.keys(),
        key=lambda x: results[x].get('best_mean',
                      results[x].get('best', float('-inf'))),
        reverse=True)

    for i, algo in enumerate(sorted_algos):
        d   = results[algo]
        tag = "★ " if i == 0 else "  "
        bm  = d.get('best_mean', d.get('best', 0))
        bci = d.get('best_ci95', 0)
        wrm = d.get('win_rate_mean', d.get('final_win_rate', 0))
        wrci= d.get('win_rate_ci95', 0)
        st  = d.get('stability_mean', d.get('stability', 0))
        row(tag+algo, bm, bci, wrm, wrci, st)

    # R1.1 — BayesG / GACG depuis leurs papiers
    if include_external and env_name in EXTERNAL_RESULTS.get('BayesG', {}):
        print(f"  {'─'*74}")
        print(f"  {'(depuis papiers)':>16}")
        for ext_algo in ('BayesG', 'GACG'):
            if env_name in EXTERNAL_RESULTS.get(ext_algo, {}):
                d    = EXTERNAL_RESULTS[ext_algo][env_name]
                note = d.get('note', '')
                row(f"  {ext_algo}", d['best'], 0, d['win_rate'], 0, 0, note)

    n_seeds = list(results.values())[0].get('n_seeds', 1) if results else 1
    print(f"  {'='*74}")
    print(f"  IC95% : t-distribution bilatérale, {n_seeds} seeds")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

ALGOS_AVAILABLE = ['H3C', 'MAPPO', 'IPPO', 'QMIX', 'VDN', 'COMA',
                   'FACMAC', 'MAPPO-Large']


def parse_args():
    p = argparse.ArgumentParser(description='H3C-BEACON MPE Training — Version Révisée')
    p.add_argument('--algo',    nargs='+', default=['H3C'],
                   help=f"Algorithme(s) : {ALGOS_AVAILABLE} | all")
    p.add_argument('--env',     type=str, default='simple_spread',
                   help='Environnement : simple_spread | simple_world_comm | all')
    p.add_argument('--steps',   type=int, default=1_000_000)
    p.add_argument('--seeds',   type=int, nargs='+', default=[42],
                   metavar='S',
                   help='Seeds (ex: --seeds 42 123 456 789 1024)')
    p.add_argument('--eval-interval',  type=int, default=160_000)
    p.add_argument('--log-interval',   type=int, default=32_000)
    p.add_argument('--ablation-log-interval', type=int, default=5_000,
                   help='Log interval pour ablation (default: 5000 steps)')
    p.add_argument('--n-eval-episodes',type=int, default=32)
    p.add_argument('--device',  type=str, default='auto')
    p.add_argument('--results-dir', type=str, default='resultats')
    p.add_argument('--ablation', type=str, default=None,
                   choices=ABLATION_VARIANTS,
                   help='Ablation (R1.2) : no_dgat | no_bayesian | ...')
    p.add_argument('--ablation-all', action='store_true',
                   help='Lance toutes les ablations séquentiellement (R1.2)')
    p.add_argument('--no-external', action='store_true',
                   help='Masquer BayesG/GACG dans le tableau')
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"\n  GPU : {torch.cuda.get_device_name(0)}")
        else:
            print("\n  Pas de GPU — utilisation CPU")
    else:
        device = args.device

    # Algos
    algos = ALGOS_AVAILABLE if args.algo == ['all'] else args.algo

    # Envs
    envs = list(BENCHMARKS.keys()) if args.env == 'all' else [args.env]

    # Ablations
    ablations = ABLATION_VARIANTS if args.ablation_all else [args.ablation]

    seeds = args.seeds

    print(f"\n{'='*60}")
    print(f"  H3C-BEACON MPE Training — Version Révisée")
    print(f"{'='*60}")
    print(f"  Algo(s)  : {algos}")
    print(f"  Env(s)   : {envs}")
    print(f"  Steps    : {args.steps:,}")
    print(f"  Seeds    : {seeds}  ({len(seeds)} seeds)")
    print(f"  Device   : {device}")
    if args.ablation or args.ablation_all:
        print(f"  Ablation : {ablations}")
    print(f"{'='*60}\n")

    os.makedirs(args.results_dir, exist_ok=True)
    t_total   = time.time()
    all_res   = {}

    for env_name in envs:
        print(f"\n{'*'*60}")
        print(f"* Environment: {env_name}")
        print(f"{'*'*60}")

        _env_tmp = create_env(env_name)
        env_info = get_env_info(_env_tmp)
        _env_tmp.close()

        all_res[env_name] = {}

        for algo in algos:
            for ablation in ablations:
                key = algo + (f'_{ablation}' if ablation else '')
                print(f"\n{'#'*60}")
                print(f"# {algo}  ablation={ablation or 'none'}")
                print(f"{'#'*60}")

                try:
                    # Réduire log_interval pour les ablations
                    log_int = (args.ablation_log_interval
                               if ablation is not None
                               else args.log_interval)
                    r = train_multiseed(
                        algo=algo, env_name=env_name,
                        env_info=env_info,
                        total_steps=args.steps,
                        eval_interval=args.eval_interval,
                        log_interval=log_int,
                        device=device, seeds=seeds,
                        results_dir=args.results_dir,
                        ablation=ablation,
                        n_eval_episodes=args.n_eval_episodes)
                    all_res[env_name][key] = r
                except Exception as e:
                    print(f"  ERREUR {key}: {e}")
                    import traceback; traceback.print_exc()

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'#'*60}")
    print(f"# TERMINÉ — {elapsed:.1f} minutes")
    print(f"{'#'*60}")

    for env_name, env_res in all_res.items():
        print_summary(env_res, env_name,
                      include_external=not args.no_external)

    print("\nFichiers sauvegardés dans :", args.results_dir)


if __name__ == '__main__':
    main()