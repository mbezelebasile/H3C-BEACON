#!/usr/bin/env python3
"""
train_hanabi_full.py  _  H3C-BEACON sur Hanabi 
==============================================

USAGE :
  # Test rapide variant small (~20min CPU)
  python train_hanabi_full.py --algo H3C --variant small --steps 500000 --seed 42

  # Publication variant full, 5 seeds
  python train_hanabi_full.py --algo all --variant full --steps 5000000 ^
    --seeds 42 123 456 789 1024
"""

import os, sys, json, time, random, argparse, warnings, statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ.setdefault('XDG_RUNTIME_DIR', '/tmp/runtime-user')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
sys.stdout.reconfigure(line_buffering=True)  # force immediate output

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

H3C_AVAILABLE = False
try:
    from modules.H3CTrainer_Fixed import H3CTrainerRevised
    H3C_AVAILABLE = True
    print("✓ H3CTrainer_Fixed charge")
except ImportError:
    try:
        from H3CTrainer_Fixed import H3CTrainerRevised
        H3C_AVAILABLE = True
        print("✓ H3CTrainer_Fixed charge (racine)")
    except ImportError:
        print("  Note: H3CTrainer_Fixed non disponible")


# ══════════════════════════════════════════════════════════════
# 1. ENVIRONNEMENT HANABI SIMPLIFIÉ CORRECT
# ══════════════════════════════════════════════════════════════

class HanabiEnvSimple:

    CONFIGS = {
        'small': {'players': 2, 'colors': 3, 'ranks': 3,
                  'hand_size': 3, 'max_info': 4, 'max_life': 2},
        'full':  {'players': 2, 'colors': 5, 'ranks': 5,
                  'hand_size': 5, 'max_info': 8, 'max_life': 3},
        '3p':    {'players': 3, 'colors': 5, 'ranks': 5,
                  'hand_size': 4, 'max_info': 8, 'max_life': 3},
    }

    def __init__(self, variant: str = 'full', seed: int = 42):
        cfg = self.CONFIGS[variant]
        self.n_agents   = cfg['players']
        self.C          = cfg['colors']
        self.R          = cfg['ranks']
        self.hand_size  = cfg['hand_size']
        self.max_info   = cfg['max_info']
        self.max_life   = cfg['max_life']
        self.max_score  = self.C * self.R
        self.episode_limit = 100
        self.variant    = variant
        self._rng       = np.random.RandomState(seed)

        # Dimensions obs :
        # other hands : (N-1) * hand_size * (C*R) one-hot par carte
        # board       : C rangs (normalisés 0..1)
        # tokens      : 2
        # hint_own    : hand_size * (C + R) révélations
        other_hand_dim  = (self.n_agents - 1) * self.hand_size * (self.C * self.R)
        board_dim       = self.C
        tokens_dim      = 2
        hint_dim        = self.hand_size * (self.C + self.R)
        self.obs_dim    = other_hand_dim + board_dim + tokens_dim + hint_dim

        # Actions :
        # play_i    : hand_size
        # discard_i : hand_size
        # hint_color_c à agent j : (C) * (N-1)
        # hint_rank_r  à agent j : (R) * (N-1)
        self.n_actions = (self.hand_size * 2
                          + (self.C + self.R) * (self.n_agents - 1))

        print(f"  ✓ Hanabi-{variant} (implementation propre) | "
              f"N={self.n_agents} Obs={self.obs_dim} Act={self.n_actions} "
              f"MaxScore={self.max_score}")

        self._state = None

    # ── Gestion du jeu ──────────────────────────────────────────────────────

    def _build_deck(self):
      
        deck = [(c, r) for c in range(self.C) for r in range(self.R)]
        self._rng.shuffle(deck)
        return list(deck)

    def _draw(self, agent_idx, pos):
        if self._deck:
            card = self._deck.pop()
        else:
            # Deck vide : pioche aléatoire
            card = (self._rng.randint(self.C), self._rng.randint(self.R))
        self._hands[agent_idx][pos] = card
        # Réinitialise les indices connus pour cette position
        self._hints_color[agent_idx][pos] = -1
        self._hints_rank[agent_idx][pos]  = -1

    def reset(self) -> np.ndarray:
        self._deck       = self._build_deck()
        self._board      = [0] * self.C          # prochain rang attendu par couleur
        self._info       = self.max_info
        self._life       = self.max_life
        self._score      = 0
        self._step       = 0
        self._done       = False

        # Mains : list[N] de list[hand_size] de (color, rank)
        self._hands = [[None] * self.hand_size for _ in range(self.n_agents)]
        # Indices reçus : -1 = inconnu
        self._hints_color = [[-1] * self.hand_size for _ in range(self.n_agents)]
        self._hints_rank  = [[-1] * self.hand_size for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            for p in range(self.hand_size):
                self._draw(i, p)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

        for i in range(self.n_agents):
            idx = 0

            # Mains des autres agents (one-hot par carte)
            for j in range(self.n_agents):
                if j == i: continue
                for p in range(self.hand_size):
                    card = self._hands[j][p]
                    if card is not None:
                        c, r = card
                        card_id = c * self.R + r
                        obs[i, idx + card_id] = 1.0
                    idx += self.C * self.R

            # Board : rang atteint par couleur (normalisé)
            for c in range(self.C):
                obs[i, idx] = self._board[c] / self.R
                idx += 1

            # Jetons (normalisés)
            obs[i, idx]   = self._info / self.max_info
            obs[i, idx+1] = self._life / self.max_life
            idx += 2

            # Indices connus sur sa propre main
            for p in range(self.hand_size):
                hc = self._hints_color[i][p]
                hr = self._hints_rank[i][p]
                if hc >= 0:
                    obs[i, idx + hc] = 1.0
                idx += self.C
                if hr >= 0:
                    obs[i, idx + hr] = 1.0
                idx += self.R

        return obs

    def get_avail_actions(self) -> np.ndarray:
        
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        for i in range(self.n_agents):
            # Toujours jouer ou défausser
            for p in range(self.hand_size):
                avail[i, p]               = 1.0  # play
                avail[i, self.hand_size+p] = 1.0  # discard
            # Indices : seulement si info tokens disponibles
            if self._info > 0:
                base = self.hand_size * 2
                for j_offset in range(self.n_agents - 1):
                    for c in range(self.C):
                        avail[i, base + j_offset*(self.C+self.R) + c] = 1.0
                    for r in range(self.R):
                        avail[i, base + j_offset*(self.C+self.R) + self.C + r] = 1.0
        return avail

    def step(self, actions) -> Tuple[np.ndarray, float, bool, Dict]:
        if hasattr(actions, 'tolist'):
            actions = actions.tolist()
        if not isinstance(actions, list):
            actions = [int(actions)]

        self._step += 1
        total_reward = 0.0

        for i, a in enumerate(actions):
            if i >= self.n_agents: break
            a = int(a) % self.n_actions
            reward_i = self._apply_action(i, a)
            total_reward += reward_i

        # Terminaison
        self._done = (self._score >= self.max_score
                      or self._life <= 0
                      or self._step >= self.episode_limit
                      or (not self._deck and self._step > self.hand_size))

        info = {
            'score':      self._score,
            'max_score':  self.max_score,
            'battle_won': self._score >= self.max_score,
            'life':       self._life,
            'info':       self._info,
        }
        return self._get_obs(), total_reward, self._done, info

    def _apply_action(self, agent_idx: int, action: int) -> float:
        
        play_end    = self.hand_size
        discard_end = self.hand_size * 2

        if action < play_end:
            # Jouer la carte à la position `action`
            pos  = action
            card = self._hands[agent_idx][pos]
            if card is None:
                return 0.0
            c, r = card
            if self._board[c] == r:
                # Succès : carte dans l'ordre attendu
                self._board[c] += 1
                self._score    += 1
                reward = 1.0
                # Bonus si on complète une couleur
                if self._board[c] == self.R and self._info < self.max_info:
                    self._info += 1
            else:
                # Échec : mauvaise carte
                self._life -= 1
                reward = 0.0         # [FIX-2] pas de reward négatif
            self._draw(agent_idx, pos)
            return reward

        elif action < discard_end:
            # Défausser
            pos = action - self.hand_size
            self._draw(agent_idx, pos)
            if self._info < self.max_info:
                self._info += 1
            return 0.0

        else:
            # Indice
            if self._info <= 0:
                return 0.0
            self._info -= 1
            base   = action - discard_end
            others = [j for j in range(self.n_agents) if j != agent_idx]
            if not others: return 0.0
            # Décoder : quel agent, quelle couleur/rang
            per_other = self.C + self.R
            j_offset  = base // per_other
            hint_type = base % per_other
            if j_offset >= len(others): return 0.0
            target = others[j_offset]
            if hint_type < self.C:
                # Indice couleur
                c = hint_type
                for p in range(self.hand_size):
                    card = self._hands[target][p]
                    if card and card[0] == c:
                        self._hints_color[target][p] = c
            else:
                # Indice rang
                r = hint_type - self.C
                for p in range(self.hand_size):
                    card = self._hands[target][p]
                    if card and card[1] == r:
                        self._hints_rank[target][p] = r
            return 0.0

    def close(self): pass


# ══════════════════════════════════════════════════════════════
# 2. WRAPPER UNIFIÉ (HLE ou simplifié)
# ══════════════════════════════════════════════════════════════

class HanabiWrapper:

    VARIANTS = HanabiEnvSimple.CONFIGS

    def __init__(self, variant: str = 'full', seed: int = 42):
        self.variant_name = variant
        self._env         = None
        self._backend     = None

        # Tentative HLE
        try:
            import hanabi_learning_environment.rl_env as hle
            cfg = self.VARIANTS[variant]
            hle_cfg = {
                'colors': cfg['colors'], 'ranks': cfg['ranks'],
                'players': cfg['players'], 'hand_size': cfg['hand_size'],
                'max_information_tokens': cfg['max_info'],
                'max_life_tokens': cfg['max_life'],
                'observation_type': 1,
            }
            self._hle   = hle.HanabiEnv(hle_cfg)
            self._stk   = hle.ObservationStacker(self._hle, stack_size=1)
            self._backend = 'hle'
            self.n_agents  = cfg['players']
            self.obs_dim   = self._stk.observation_size()
            self.n_actions = self._hle.num_moves()
            self.max_score = cfg['colors'] * cfg['ranks']
            self.episode_limit = 200
            print(f"  ✓ hanabi-learning-environment | "
                  f"N={self.n_agents} Obs={self.obs_dim} Act={self.n_actions}")
        except Exception:
            # Fallback : implémentation propre
            self._env     = HanabiEnvSimple(variant, seed)
            self._backend = 'simple'
            self.n_agents   = self._env.n_agents
            self.obs_dim    = self._env.obs_dim
            self.n_actions  = self._env.n_actions
            self.max_score  = self._env.max_score
            self.episode_limit = self._env.episode_limit

        self.params = self.VARIANTS[variant]

    def reset(self) -> np.ndarray:
        if self._backend == 'hle':
            return self._reset_hle()
        return self._env.reset()

    def step(self, actions):
        if self._backend == 'hle':
            return self._step_hle(actions)
        return self._env.step(actions)

    def get_avail_actions(self) -> np.ndarray:
        if self._backend == 'hle':
            return self._avail_hle()
        return self._env.get_avail_actions()

    def close(self):
        if self._backend == 'hle' and hasattr(self, '_hle'):
            try: self._hle.close()
            except: pass
        elif self._env:
            self._env.close()

    # HLE specifics (unchanged from previous version)
    def _reset_hle(self):
        obs, _ = self._hle.reset()
        self._stk.reset_obs()
        return np.array([self._stk.get_observation_from_dict(obs, i)
                         for i in range(self.n_agents)], dtype=np.float32)

    def _avail_hle(self):
        legal = self._hle.state.legal_moves()
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        cur   = self._hle.state.cur_player()
        for m in legal:
            idx = self._hle.game.get_move_uid(m)
            if 0 <= idx < self.n_actions:
                avail[cur, idx] = 1.0
        for i in range(self.n_agents):
            if i != cur: avail[i, 0] = 1.0
        return avail

    def _step_hle(self, actions):
        if hasattr(actions, 'tolist'): actions = actions.tolist()
        cur   = self._hle.state.cur_player()
        act   = int(actions[cur]) if isinstance(actions, list) else int(actions)
        legal = [self._hle.game.get_move_uid(m)
                 for m in self._hle.state.legal_moves()]
        if act not in legal: act = legal[0]
        move = self._hle.game.get_move(act)
        obs_d, reward, done, info = self._hle.step(move)
        if not done: self._stk.add_observation(obs_d)
        obs = np.array([self._stk.get_observation_from_dict(obs_d, i)
                        for i in range(self.n_agents)], dtype=np.float32)
        sc = self._hle.state.score()
        info = {'score': sc, 'max_score': self.max_score,
                'battle_won': sc == self.max_score}
        return obs, float(reward), bool(done), info


# ══════════════════════════════════════════════════════════════
# 3. BASELINES CORRIGÉES
# ══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_d, h, out_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.ReLU(),
            nn.Linear(h, h),    nn.ReLU(),
            nn.Linear(h, out_d))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)


class HanabiBaselineTrainer:
    
    def __init__(self, obs_dim, n_actions, n_agents, algo, device,
                 hidden=128, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 n_epochs=10):
        self.obs_dim      = obs_dim
        self.n_actions    = n_actions
        self.n_agents     = n_agents
        self.algo         = algo
        self.device       = torch.device(device)
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.name         = algo
        self.eval_history: List[Dict] = []

        state_dim = obs_dim * n_agents
        self.actor  = MLP(obs_dim,   hidden, n_actions).to(self.device)
        self.critic = MLP(state_dim if algo == 'MAPPO' else obs_dim,
                          hidden, 1).to(self.device)
        self.optim  = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()), lr=lr)

    def get_action(self, obs, avail=None, deterministic=False):
        """obs : [N, D]"""
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).to(self.device)
            logits = self.actor(obs_t)
            if avail is not None:
                av_t   = torch.FloatTensor(avail).to(self.device)
                logits = logits + (1.0 - av_t) * (-1e9)
            if deterministic:
                actions = logits.argmax(-1)
            else:
                actions = Categorical(logits=logits).sample()
            log_probs = Categorical(logits=logits).log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu()

    def _gae(self, rewards, values, dones):
       
        T   = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv    = float(values[t + 1]) if t < T - 1 else 0.0
            delta = (rewards[t] + self.gamma * nv * (1 - dones[t])
                     - values[t])
            gae   = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            adv[t] = gae
        # [FIX-3] Normaliser
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, adv + values[:T]

    def update(self, batch) -> Dict:
        obs_b   = torch.FloatTensor(np.array(batch['obs'])).to(self.device)
        acts_b  = torch.LongTensor(np.array(batch['actions'])).to(self.device)
        rews_b  = np.array(batch['rewards'], dtype=np.float32)
        dones_b = np.array(batch['dones'],   dtype=np.float32)
        old_logp = torch.stack(batch['log_probs']).to(self.device)
        avail_b  = batch.get('avail')

        T, N, D = obs_b.shape
        state_b  = obs_b.reshape(T, N * D)

        # Valeurs pour GAE
        with torch.no_grad():
            if self.algo == 'MAPPO':
                # [FIX-1] .squeeze(-1) → [T], PAS de .mean(-1)
                vals = self.critic(state_b).squeeze(-1).cpu().numpy()
            else:
                vals = (self.critic(obs_b.reshape(T * N, D))
                        .view(T, N).mean(-1).cpu().numpy())

        adv_np, ret_np = self._gae(rews_b, vals, dones_b)
        adv_t = torch.FloatTensor(adv_np).to(self.device)
        ret_t = torch.FloatTensor(ret_np).to(self.device)

        losses = []
        ent_last = torch.tensor(0.0)

        for _ in range(self.n_epochs):
            logits = self.actor(obs_b.reshape(T * N, D)).view(T, N, -1)
            if avail_b is not None:
                av = torch.FloatTensor(np.array(avail_b)).to(self.device)
                logits = logits + (1.0 - av) * (-1e9)

            dist     = Categorical(logits=logits)
            new_logp = dist.log_prob(acts_b)      # OLD actions, NEW policy
            ent_last = dist.entropy().mean()

            if self.algo == 'MAPPO':
                v_new = (self.critic(state_b).squeeze(-1)
                         .unsqueeze(1).expand(T, N))
            else:
                v_new = self.critic(obs_b.reshape(T * N, D)).view(T, N)

            adv_e = adv_t.unsqueeze(1).expand(T, N)
            ret_e = ret_t.unsqueeze(1).expand(T, N)

            ratio = (new_logp - old_logp.detach()).exp()
            surr  = torch.min(
                ratio * adv_e,
                torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_e)

            # [FIX] vals is numpy → convert to tensor before operation
            vals_t    = torch.FloatTensor(vals).to(self.device)  # [T]
            vals_e    = vals_t.unsqueeze(1).expand(T, N)          # [T, N]
            v_clipped = vals_e + torch.clamp(v_new - vals_e, -0.5, 0.5)
            vf_loss1  = (v_new - ret_e.detach()).pow(2)
            vf_loss2  = (v_clipped - ret_e.detach()).pow(2)
            vf_loss   = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            loss = (-surr.mean()
                    + self.value_coef * vf_loss
                    - self.entropy_coef * ent_last)

            self.optim.zero_grad()
            loss.backward()
            # [FIX-4] Clipping gradient
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) +
                list(self.critic.parameters()), max_norm=0.5)
            self.optim.step()
            losses.append(loss.item())

        return {'loss': float(np.mean(losses)), 'entropy': float(ent_last.detach())}


# ══════════════════════════════════════════════════════════════
# 4. ÉVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(trainer, env: HanabiWrapper, device,
             n_ep: int = 32,
             use_h3c: bool = False) -> Tuple[float, float, float]:
    """Retourne (mean_score, std_score, perfect_rate_pct)."""
    scores, perfects = [], 0
    dev = torch.device(device)

    for _ in range(n_ep):
        obs  = env.reset(); done = False
        info = {'score': 0}
        for _ in range(env.episode_limit):
            if done: break
            avail = env.get_avail_actions()
            if use_h3c:
                obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                avail_t = torch.FloatTensor(avail).unsqueeze(0).to(dev)
                obs_t   = torch.nan_to_num(obs_t, nan=0.0, posinf=1.0, neginf=-1.0)
                try:
                    with torch.no_grad():
                        act_t, _, _ = trainer.get_action(
                            obs_t, avail_actions_t=avail_t, deterministic=True)
                    actions = act_t.squeeze(0).cpu().numpy()
                    if any(a != a for a in actions.flat):
                        raise ValueError("NaN")
                except Exception:
                    avail_np = avail
                    actions = np.array([
                        np.random.choice(np.where(avail_np[i] > 0)[0])
                        for i in range(avail_np.shape[0])], dtype=np.int64)
            else:
                actions, _ = trainer.get_action(obs, avail=avail,
                                                 deterministic=True)
            obs, _, done, info = env.step(actions)

        sc = info.get('score', 0)
        scores.append(sc)
        if info.get('battle_won', False): perfects += 1

    return (float(np.mean(scores)), float(np.std(scores)),
            perfects / n_ep * 100.0)


# ══════════════════════════════════════════════════════════════
# 5. BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════

def train_one(algo: str, variant: str, total_steps: int, seed: int,
              device: str, log_interval: int, eval_interval: int,
              save_dir: str, ablation: Optional[str] = None,
              n_eval_ep: int = 32) -> Dict:

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device)

    env = HanabiWrapper(variant=variant, seed=seed)
    print(f"\n{'*'*60}")
    print(f"* {algo} | Hanabi-{variant} | seed={seed} | {total_steps:,} steps")
    print(f"{'*'*60}")
    print(f"  Score max={env.max_score} | Backend={env._backend}")

    # Config commune (baselines MAPPO etc.)
    cfg = {'hidden_dim': 128, 'gamma': 0.99, 'gae_lambda': 0.95,
           'lr_actor': 3e-4, 'lr_critic': 1e-3, 'clip_epsilon': 0.2,
           'entropy_coef': 0.01, 'value_loss_coef': 0.5, 'n_epochs': 10,
           'max_grad_norm': 0.5}

    # Config H3C Hanabi : LR ultra-réduit + clipping très agressif
    # Série d'explosions NaN à step 410K-440K avec config précédente
    # → lr_actor 5e-5→1e-5, lr_critic 1e-4→3e-5, max_grad_norm 0.1→0.05
    # → clip_epsilon 0.1→0.05 (PPO ratio more conservative)
    # → n_epochs 4→2 (moins d'updates par batch = gradients plus petits)
    if algo == 'H3C':
        cfg = {**cfg,
               'lr_actor':      1e-4,   # 3e-5 too small for bad initialisations (seed=123)
               'lr_critic':     3e-4,   # lr_critic/lr_actor ratio = 3x (standard)
               'max_grad_norm': 0.05,   # maintained — anti-NaN (validated on seed=42)
               'clip_epsilon':  0.05,   # maintenu conservateur
               'entropy_coef':  0.08,   # increased 0.05→0.08 (bootstrap Hanabi exploration)
               'n_epochs':      2,      # 4    → 2     (÷2 updates/batch)
               'mini_batch_size': 64}   # unchanged

    use_h3c = (algo == 'H3C')

    if use_h3c:
        if not H3C_AVAILABLE:
            print("  ERREUR: H3CTrainer_Fixed "); return {}
        trainer = H3CTrainerRevised(
            obs_dim=env.obs_dim, action_dim=env.n_actions,
            n_agents=env.n_agents, config=cfg, device=dev,
            disable_dgat       =(ablation == 'no_dgat'),
            disable_bayesian   =(ablation == 'no_bayesian'),
            disable_coalitions =(ablation == 'no_coalitions'),
            disable_dual_critic=(ablation == 'no_dual_critic'),
            disable_rtd        =(ablation == 'no_rtd'),
            disable_entropy    =(ablation == 'no_entropy'))
        trainer.name = f"H3C{'_'+ablation if ablation else ''}"
    else:
        trainer = HanabiBaselineTrainer(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            n_agents=env.n_agents, algo=algo, device=device)
        trainer.name = algo

    # Métriques

    # ── Métriques ────────────────────────────────────────────────────────
    score_window = deque(maxlen=100)
    loss_window  = deque(maxlen=50)
    eval_records: List[Dict] = []
    best_score   = 0.0
    best_perfect = 0.0
    global_step  = 0; total_ep = 0
    last_log = 0; last_eval = 0
    last_heartbeat = 0
    t0 = time.time()

    # ── Recovery state ───────────────────────────────────────────────────
    best_state       = None   # best state_dict saved in RAM
    best_state_step  = 0
    nan_streak       = 0      # NaN consécutifs depuis dernier recovery
    recovery_count   = 0      # nombre total de recoveries effectués
    MAX_RECOVERY     = 5      # beyond this: permanent stop
    BASE_LR_ACTOR    = cfg.get('lr_actor', 1e-4)
    BASE_LR_CRITIC   = cfg.get('lr_critic', 3e-4)
    lr_factor        = 1.0    # multiplicateur courant du LR

    def get_lr(trainer):
        """Return current lr_actor."""
        if hasattr(trainer, 'actor_opt'):
            return trainer.actor_opt.param_groups[0]['lr']
        return None

    def set_lr(trainer, lr_a, lr_c):
        """Set lr_actor and lr_critic."""
        if hasattr(trainer, 'actor_opt'):
            for pg in trainer.actor_opt.param_groups:
                pg['lr'] = lr_a
        if hasattr(trainer, 'critic_opt'):
            for pg in trainer.critic_opt.param_groups:
                pg['lr'] = lr_c

    def save_best_state(trainer):
        """Save complete trainer state_dict in RAM."""
        import copy
        state = {}
        for attr in ['actor', 'critic', 'actor_opt', 'critic_opt',
                     'global_critic', 'coalition_net', 'belief_fusion',
                     'rtd_buffer', 'entropy_controller']:
            if hasattr(trainer, attr):
                obj = getattr(trainer, attr)
                if hasattr(obj, 'state_dict'):
                    state[attr] = copy.deepcopy(obj.state_dict())
        return state

    def restore_best_state(trainer, state):
        """Restaurer le state_dict sauvegardé."""
        for attr, sd in state.items():
            if hasattr(trainer, attr):
                obj = getattr(trainer, attr)
                if hasattr(obj, 'load_state_dict'):
                    try:
                        obj.load_state_dict(sd)
                    except Exception:
                        pass  # architecture mismatch → ignorer

    def do_recovery(trainer, step, reason):
        """Recovery : restaurer best checkpoint + réduire LR + reset momentum."""
        nonlocal recovery_count, lr_factor, nan_streak
        recovery_count += 1
        nan_streak = 0

        print(f"\n  [RECOVERY #{recovery_count}] step={step:,} | reason={reason}")

        # 1. Restaurer le meilleur checkpoint
        if best_state is not None:
            restore_best_state(trainer, best_state)
            print(f"    ← Restored checkpoint from step {best_state_step:,} "
                  f"(score={best_score:.2f})")
        else:
            print("    ← No checkpoint yet — continuing from current state")

        # 2. Réduire le LR progressivement
        lr_factor = max(0.05, lr_factor * 0.5)
        new_lr_a = BASE_LR_ACTOR  * lr_factor
        new_lr_c = BASE_LR_CRITIC * lr_factor
        set_lr(trainer, new_lr_a, new_lr_c)
        print(f"    LR: {BASE_LR_ACTOR:.1e} × {lr_factor:.3f} "
              f"= {new_lr_a:.2e} (actor)")

        # 3. Réinitialiser le momentum des optimiseurs (Adam state)
        for opt_name in ['actor_opt', 'critic_opt']:
            if hasattr(trainer, opt_name):
                opt = getattr(trainer, opt_name)
                opt.state.clear()
        print("    Optimizer momentum reset")

        # 4. Augmenter légèrement l'entropie pour ré-explorer
        if hasattr(trainer, 'entropy_coef'):
            trainer.entropy_coef = min(0.12,
                                       trainer.entropy_coef * 1.5)
            print(f"    entropy_coef → {trainer.entropy_coef:.3f}")

        return recovery_count < MAX_RECOVERY

    # ── Initial evaluation ─────────────────────────────────────────────
    sc0, std0, pf0 = evaluate(trainer, env, device, n_eval_ep, use_h3c)
    print(f"  Init: score={sc0:.2f}±{std0:.2f}/{env.max_score} "
          f"perfect={pf0:.1f}%")
    eval_records.append({'step': 0, 'score': sc0, 'perfect': pf0})

    # Savingr l'état initial comme baseline
    if use_h3c:
        best_state = save_best_state(trainer)

    # LR decay schedule : milestones à 500K et 750K
    LR_DECAY_STEPS = {500_000: 0.5, 750_000: 0.5}  # × factor cumulatif

    # ── Main training loop ────────────────────────────────────────────────
    while global_step < total_steps:
        obs = env.reset(); done = False; ep_r = 0.0
        buf_o, buf_a, buf_r, buf_d, buf_lp, buf_av = [], [], [], [], [], []
        ep_nan = False  # NaN detected in this episode

        for _ in range(env.episode_limit):
            if done:
                break
            avail = env.get_avail_actions()

            if use_h3c:
                obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                avail_t = torch.FloatTensor(avail).unsqueeze(0).to(dev)
                obs_t   = torch.nan_to_num(obs_t,
                                           nan=0.0, posinf=1.0, neginf=-1.0)
                try:
                    with torch.no_grad():
                        act_t, lp_t, _ = trainer.get_action(
                            obs_t, avail_actions_t=avail_t)
                    if torch.isnan(act_t).any() or torch.isnan(lp_t).any():
                        raise ValueError("NaN in output")
                    actions   = act_t.squeeze(0).cpu().numpy()
                    log_probs = lp_t.squeeze(0).cpu()
                    nan_streak = max(0, nan_streak - 1)  # décrémentation lente
                except Exception:
                    # Fallback: random legal action
                    avail_np = np.array(avail)
                    actions  = np.array([
                        np.random.choice(np.where(avail_np[i] > 0)[0])
                        for i in range(avail_np.shape[0])], dtype=np.int64)
                    log_probs = torch.zeros(len(actions))
                    ep_nan    = True
                    nan_streak += 1
            else:
                actions, log_probs = trainer.get_action(obs, avail=avail)

            next_obs, r, done, info = env.step(actions)
            ep_r += r
            global_step += 1

            # LR decay schedule: always active (unconditional)
            # Seed=42 has no explosion but needs decay after 500K steps
            # to prevent policy forgetting observed in practice
            if use_h3c and global_step in LR_DECAY_STEPS:
                factor = LR_DECAY_STEPS[global_step]
                lr_factor *= factor
                new_lra = BASE_LR_ACTOR  * lr_factor
                new_lrc = BASE_LR_CRITIC * lr_factor
                set_lr(trainer, new_lra, new_lrc)
                print(f'\n  [LR-DECAY] step={global_step:,} '
                      f'lr_actor={new_lra:.1e} lr_critic={new_lrc:.1e}')
                print(f"  [LR SCHEDULE] step={global_step:,} "
                      f"× {factor} → {BASE_LR_ACTOR*lr_factor:.2e}")

            if not ep_nan:  # ne pas mettre les transitions NaN dans le buffer
                buf_o.append(obs.copy())
                buf_a.append(np.array(actions, dtype=np.int64))
                buf_r.append(float(r))
                buf_d.append(float(done))
                buf_lp.append(log_probs if torch.is_tensor(log_probs)
                              else torch.FloatTensor(log_probs))
                buf_av.append(avail.copy())
            obs = next_obs

        # ── NaN streak → recovery ────────────────────────────────────────
        if nan_streak >= 3:
            keep_going = do_recovery(trainer, global_step, "NaN streak")
            if not keep_going:
                print(f"  [STOP] {MAX_RECOVERY} recoveries épuisées — arrêt.")
                break
            buf_o.clear(); buf_a.clear(); buf_r.clear()
            buf_d.clear(); buf_lp.clear(); buf_av.clear()
            continue  # nouvel épisode propre

        # ── Update PPO ────────────────────────────────────────────────────
        loss_val = 0.0
        if len(buf_o) > 1:
            if use_h3c:
                batch = {
                    'obs':      torch.FloatTensor(np.array(buf_o)).to(dev),
                    'actions':  torch.LongTensor(np.array(buf_a)).to(dev),
                    'rewards':  torch.FloatTensor(buf_r).to(dev),
                    'dones':    torch.FloatTensor(
                                np.array(buf_d, dtype=np.float32)).to(dev),
                    'log_probs':torch.stack(buf_lp).to(dev),
                    'avail_actions': torch.FloatTensor(
                                np.array(buf_av)).to(dev),
                }
                try:
                    metrics  = trainer.update(batch)
                    loss_val = float(metrics.get('loss', 0))

                    # Loss NaN/Inf → recovery
                    if not np.isfinite(loss_val) or abs(loss_val) > 1e4:
                        raise ValueError(f"loss={loss_val:.3g}")

                    loss_window.append(loss_val)

                    # Plateau de loss (politique figée) → nudge
                    if (len(loss_window) == 50
                            and np.std(list(loss_window)) < 5e-4
                            and recovery_count < MAX_RECOVERY):
                        do_recovery(trainer, global_step, "loss plateau")

                except Exception as e:
                    nan_streak += 1
                    print(f"  [UPDATE FAIL] step={global_step:,} {e}")
                    if nan_streak >= 3:
                        keep_going = do_recovery(
                            trainer, global_step, f"update error: {e}")
                        if not keep_going:
                            print("  [STOP] Max recoveries atteint.")
                            break

            else:
                # ── MAPPO / IPPO update ───────────────────────────────────
                batch = {
                    'obs':       buf_o,
                    'actions':   buf_a,
                    'rewards':   buf_r,
                    'dones':     buf_d,
                    'log_probs': buf_lp,
                    'avail':     buf_av,
                }
                m = trainer.update(batch)
                loss_val = float(m.get('loss', 0))
                loss_window.append(loss_val)

        # ── Episode bookkeeping ───────────────────────────────────────────
        total_ep      += 1
        score_window.append(info.get('score', 0))

        # Save best state (H3C only)
        if use_h3c and info.get('score', 0) >= best_score:
            best_score      = info.get('score', 0)
            best_state      = save_best_state(trainer)
            best_state_step = global_step

        # ── Heartbeat [ALIVE] every 600s ─────────────────────────────────
        now = time.time()
        if now - last_heartbeat >= 120:  # heartbeat every 2 minutes
            elapsed  = (now - t0) / 60
            remain   = (total_steps - global_step) / max(global_step, 1) * elapsed
            sc_mean  = float(np.mean(score_window)) if score_window else 0.0
            lv       = float(np.mean(loss_window))  if loss_window  else 0.0
            print(f"  [ALIVE] Step {global_step:>10,} | "
                  f"Score:{sc_mean:.2f}/25 | Loss:{lv:.4f} | "
                  f"{global_step/max(now-t0,1):.1f}/s | "
                  f"{elapsed:.0f}min | ETA:{remain:.1f}h")
            last_heartbeat = now

        # ── Periodic logging ─────────────────────────────────────────────
        if global_step - last_log >= log_interval:
            sc_mean = float(np.mean(score_window)) if score_window else 0.0
            lv      = float(np.mean(loss_window))  if loss_window  else 0.0
            elapsed = (time.time() - t0) / 60
            remain  = (total_steps - global_step) / max(global_step, 1) * elapsed
            spd     = global_step / max(time.time() - t0, 1)
            print(f"  Step {global_step:>10,}/{total_steps:,} | "
                  f"Score: {sc_mean:.2f}/25 | Loss:{lv:.4f} | "
                  f"{spd:>6.0f}/s | ETA:{remain:.1f}h")
            last_log = global_step

        # ── Periodic evaluation ───────────────────────────────────────────
        if global_step - last_eval >= eval_interval and global_step > 0:
            sc_e, std_e, pf_e = evaluate(trainer, env, device,
                                         n_eval_ep, use_h3c)
            print(f"\n  EVAL {global_step:,}: "
                  f"Score={sc_e:.2f}\u00b1{std_e:.2f}/25 "
                  f"Perfect={pf_e:.1f}%\n")
            eval_records.append({'step': global_step,
                                  'score': sc_e, 'perfect': pf_e})
            if sc_e > best_score:
                best_score   = sc_e
                best_perfect = pf_e
                if use_h3c:
                    best_state      = save_best_state(trainer)
                    best_state_step = global_step
            last_eval = global_step

    # ── Final evaluation ─────────────────────────────────────────────────
    sc_f, std_f, pf_f = evaluate(trainer, env, device, n_eval_ep, use_h3c)
    duration = (time.time() - t0) / 60

    print(f"\n{'='*60}")
    print(f"  {algo} | Hanabi-{variant} | seed={seed}")
    print(f"  Score final : {sc_f:.2f}\u00b1{std_f:.2f}/25")
    print(f"  Score best  : {best_score:.2f}")
    print(f"  Perfect     : {pf_f:.1f}%")
    print(f"  Dur\u00e9e       : {duration:.1f}min")
    print(f"{'='*60}")

    # ── Save JSON ────────────────────────────────────────────────────────
    import os
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{algo}_hanabifull_s{seed}_{ts}.json"
    result = {
        'algorithm':   algo,
        'variant':     variant,
        'seed':        seed,
        'final_score': sc_f,
        'final_std':   std_f,
        'best_score':  best_score,
        'perfect_rate':pf_f,
        'max_score':   env.max_score,
        'duration_min':duration,
        'eval_records':eval_records,
    }
    with open(os.path.join(save_dir, fname), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Sauvegarde : {save_dir}/{fname}")

    env.close()
    return result


def ci95(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0: return 0.0, 0.0
    m = statistics.mean(vals)
    if n == 1: return m, 0.0
    try:
        import scipy.stats as st
        return m, float(st.sem(vals) * st.t.ppf(0.975, n-1))
    except ImportError:
        s = statistics.stdev(vals)
        t = {1:12.71,2:4.30,3:3.18,4:2.78,5:2.57}.get(n-1, 2.0)
        return m, t * s / (n**0.5)


def print_summary(all_results: List[Dict], variant: str):
    max_sc = all_results[0]['max_score'] if all_results else 25
    groups: Dict = {}
    for r in all_results:
        groups.setdefault(r['algorithm'], []).append(r)

    print(f"\n{'='*72}")
    print(f"  RÉSULTATS Hanabi-{variant}  (max={max_sc})  — R2.8")
    print(f"{'='*72}")
    print(f"  {'Algo':<16} {'N':>3}  {'Score':>8}  {'±CI':>6}  "
          f"{'Best':>6}  {'Perfect%':>9}")
    print(f"  {'─'*58}")

    for i, algo in enumerate(sorted(groups,
            key=lambda a: statistics.mean(r['best_score'] for r in groups[a]),
            reverse=True)):
        runs  = groups[algo]
        sc    = [r['final_score'] for r in runs]
        bsc   = [r['best_score']  for r in runs]
        pf    = [r['perfect_rate'] for r in runs]
        m_sc, ci_sc  = ci95(sc)
        m_bsc, _     = ci95(bsc)
        m_pf, ci_pf  = ci95(pf)
        tag = "★ " if i == 0 else "  "
        print(f"  {tag}{algo:<14} {len(runs):>3}  "
              f"{m_sc:>7.2f}  ±{ci_sc:>5.2f}  "
              f"{m_bsc:>5.2f}  {m_pf:>7.1f}%±{ci_pf:.1f}%")
    print(f"{'='*72}")

    # Référence littérature
    print(f"\n  Références littérature (HLE, 2 joueurs) :")
    print(f"    Random policy   :  ~3-5 / {max_sc}")
    print(f"    VDN             :  ~18  / {max_sc}")
    print(f"    SAD (ICLR 2020) :  ~23  / {max_sc}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

ALGOS_ALL = ['H3C', 'MAPPO', 'IPPO', 'VDN']
ABLATIONS = ['no_dgat','no_bayesian','no_coalitions',
             'no_dual_critic','no_rtd','no_entropy']
STEPS_REC = {'small': 500_000, 'full': 5_000_000, '3p': 5_000_000}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo',    nargs='+', default=['H3C'],
                   help='H3C MAPPO IPPO VDN | all')
    p.add_argument('--variant', default='full',
                   choices=['full','small','3p'])
    p.add_argument('--steps',   type=int, default=None)
    p.add_argument('--seeds',   type=int, nargs='+', default=[42])
    p.add_argument('--ablation',default=None, choices=ABLATIONS)
    p.add_argument('--save-dir',  default='results/hanabi')
    p.add_argument('--log-interval',  type=int, default=10_000)
    p.add_argument('--h3c-log-interval', type=int, default=10_000,
                   help='Log interval specifique H3C (default 10000)')
    p.add_argument('--eval-interval', type=int, default=200_000)
    p.add_argument('--n-eval-ep', type=int, default=32)
    p.add_argument('--device',  default=None)
    args = p.parse_args()

    device = (args.device if args.device else
              'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu': print("  No GPU detected — using CPU")

    algos       = ALGOS_ALL if args.algo == ['all'] else args.algo
    total_steps = args.steps or STEPS_REC[args.variant]
    seeds       = args.seeds

    print(f"\n{'='*60}")
    print(f"  H3C-BEACON Hanabi — Version Corrigee (R2.8)")
    print(f"  Algos : {algos}  |  Variant : {args.variant}")
    print(f"  Steps : {total_steps:,}  |  Seeds : {seeds}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    all_results = []
    for algo in algos:
        for seed in seeds:
            eff_log = args.h3c_log_interval if algo == 'H3C' else args.log_interval
            r = train_one(algo, args.variant, total_steps, seed, device,
                          eff_log, args.eval_interval,
                          args.save_dir,
                          ablation=args.ablation if algo=='H3C' else None,
                          n_eval_ep=args.n_eval_ep)
            if r: all_results.append(r)

    if all_results:
        print_summary(all_results, args.variant)


if __name__ == '__main__':
    main()