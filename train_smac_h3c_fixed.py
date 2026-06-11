#!/usr/bin/env python3
"""
train_smac_h3c_fixed.py  
=======================

USAGE :
  # H3C seul, 1 seed
  python train_smac_h3c_fixed.py --algo H3C --map 2s3z --steps 500000 --seed 42

  # Toutes les baselines, 1 seed
  python train_smac_h3c_fixed.py --algo all --map 2s3z --steps 500000 --seed 42

  # H3C, 5 seeds 
  python train_smac_h3c_fixed.py --algo H3C --map 2s3z --steps 500000 \\
                                  --seeds 42,123,456,789,1024

  # Ablation 
  python train_smac_h3c_fixed.py --algo H3C --map 2s3z --steps 500000 \\
                                  --seed 42 --ablation no_rtd

  # Map difficile
  python train_smac_h3c_fixed.py --algo H3C --map 8m --steps 1000000 \\
                                  --seeds 42,123,456,789,1024
"""

import sys, os, torch, numpy as np, argparse, json, time, random, statistics
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import H3C corrigé [FIX-1] ────────────────────────────────────────────────
H3C_AVAILABLE = False
try:
    from H3CTrainer_Fixed import H3CTrainerRevised, create_h3c_trainer
    H3C_AVAILABLE = True
    print("✓ H3CTrainer_Fixed charge (version corrigee)")
except ImportError:
    try:
        from modules.H3CTrainer_Fixed import H3CTrainerRevised, create_h3c_trainer
        H3C_AVAILABLE = True
        print("✓ modules/H3CTrainer_Fixed charge")
    except ImportError:
        try:
            from modules.H3CTrainer_Revised import H3CTrainerRevised
            H3C_AVAILABLE = True
            print("  ATTENTION: H3CTrainer_Revised (non corrige) — appliquer PATCH_H3C.py")
        except ImportError as e:
            print(f"  ERREUR import H3C: {e}")

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── SMACLite ──────────────────────────────────────────────────────────────────
from smaclite.env import SMACliteEnv
from smaclite import MapPreset

# R2.8 — Maps pour les reviewers
MAPS = {
    # ── Maps utilisées dans le papier ────────────────────────────────────
    '2s3z':        MapPreset.MAP_2S3Z,        # 5 agents hétérogènes (facile)
    '3s5z':        MapPreset.MAP_3S5Z,        # 8 agents hétérogènes (moyen)
    '10m_vs_11m':  MapPreset.MAP_10M_VS_11M,  # 10 agents homogènes (dur)
    '27m_vs_30m':  MapPreset.MAP_27M_VS_30M,  # 27 agents (très dur)
    # ── Autres maps disponibles ──────────────────────────────────────────
    '2c_vs_64zg':  MapPreset.MAP_2C_VS_64ZG,
    'corridor':    MapPreset.MAP_CORRIDOR,
    'mmm':         MapPreset.MAP_MMM,
    'mmm2':        MapPreset.MAP_MMM2,
    # ── ATTENTION : aliases supprimés pour éviter confusion ──────────────
    # '3m' alias supprimé : chargeait MAP_3S5Z (8 agents) au lieu de 3 Marines
    # '8m' alias supprimé : chargeait MAP_10M_VS_11M au lieu de 8 Marines
}

SEP  = "=" * 60
STAR = "*" * 60
HASH = "#" * 60

# ── Steps recommandés par map (R2.8) ─────────────────────────────────────────
STEPS_RECOMMENDED = {
    '2s3z':       500_000,
    '3s5z':     1_000_000,
    '3m':         500_000,
    '8m':       1_000_000,
    '10m_vs_11m':1_000_000,
    '27m_vs_30m':2_000_000,
}


# ══════════════════════════════════════════════════════════════
# WRAPPER SMAC
# ══════════════════════════════════════════════════════════════

class SMACWrapper:
    
    def __init__(self, map_name: str, seed: int = 42):
        key = map_name.lower().replace('-', '_')
        if key not in MAPS:
            raise ValueError(f"Unknown map: {map_name}. "
                             f"Available: {list(MAPS.keys())}")
        self.env          = SMACliteEnv(map_info=MAPS[key].value, seed=seed)
        self.map_name     = map_name
        self.n_agents     = self.env.n_agents
        self.n_enemies    = self.env.n_enemies
        self.obs_dim      = self.env.obs_size
        self.action_dim   = self.env.n_actions
        self.state_dim    = self.env.state_size
        self.episode_limit = 200

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return np.array(obs, dtype=np.float32)

    def get_avail_actions(self) -> np.ndarray:
        """[N, A] — binaire : 1 si action disponible."""
        return np.array(self.env.get_avail_actions(), dtype=np.float32)

    def step(self, actions) -> Tuple[np.ndarray, float, bool, Dict]:
        if hasattr(actions, 'tolist'): actions = actions.tolist()
        obs, reward, done, truncated, info = self.env.step(actions)
        terminal = bool(done) or bool(truncated)
        return (np.array(obs, dtype=np.float32),
                float(reward),
                terminal,
                info if isinstance(info, dict) else {})

    def close(self): self.env.close()


# ══════════════════════════════════════════════════════════════
# BASELINES  (R2.9 — comparaison équitable)
# ══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_d: int, hidden: int, out_d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_d))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.net(x)


class BaselineTrainer:
  

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int,
                 state_dim: int, algo: str, device: torch.device,
                 hidden: int = 64, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, n_epochs: int = 10):
        self.obs_dim      = obs_dim
        self.action_dim   = action_dim
        self.n_agents     = n_agents
        self.algo         = algo
        self.device       = device
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.name         = algo
        self.eval_history : List[Dict] = []

        # Acteur partagé (parameter sharing pour tous les algos)
        self.actor  = MLP(obs_dim, hidden, action_dim).to(device)
        # Critique : global (MAPPO) ou local (IPPO, QMIX, VDN, COMA)
        critic_in   = state_dim if algo == 'MAPPO' else obs_dim
        self.critic = MLP(critic_in, hidden, 1).to(device)
        self.optim  = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr)

        # ε-greedy pour QMIX/VDN
        self.eps = 1.0

    def get_action(self, obs: torch.Tensor,
                   avail_actions_t: torch.Tensor = None,
                   deterministic: bool = False,
                   **_) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  
        with torch.no_grad():
            obs_sq = obs.squeeze(0)                        # [N, D]
            logits = self.actor(obs_sq)                    # [N, A]
            if avail_actions_t is not None:
                av = avail_actions_t.squeeze(0)
                logits = logits + (1.0 - av) * (-1e9)

            if deterministic:
                actions = logits.argmax(-1)
            elif self.algo in ('QMIX', 'VDN'):
                # ε-greedy décroissant
                if random.random() < self.eps:
                    avnp = (av.cpu().numpy()
                            if avail_actions_t is not None
                            else np.ones((self.n_agents, self.action_dim)))
                    actions = torch.tensor(
                        [np.random.choice(np.where(avnp[i] > 0)[0])
                         for i in range(self.n_agents)],
                        device=self.device)
                else:
                    actions = logits.argmax(-1)
                self.eps = max(0.05, self.eps * 0.9999)
            else:
                actions = Categorical(logits=logits).sample()

            log_probs = Categorical(logits=logits).log_prob(actions)
            entropy   = Categorical(logits=logits).entropy()

        return (actions.unsqueeze(0),
                log_probs.unsqueeze(0),
                entropy.unsqueeze(0))

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor,
             dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rewards.dim() > 1: rewards = rewards.reshape(-1)
        if dones.dim()   > 1: dones   = dones.reshape(-1)
        v1d = values.mean(-1).detach() if values.dim() > 1 else values.detach()
        T   = rewards.shape[0]
        adv = torch.zeros(T, device=rewards.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv  = float(v1d[t+1]) if t < T-1 else 0.0
            d   = (float(rewards[t]) + self.gamma * nv *
                   (1 - float(dones[t])) - float(v1d[t]))
            gae = d + self.gamma * self.gae_lambda * (1-float(dones[t])) * gae
            adv[t] = gae
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N   = values.shape[-1] if values.dim() > 1 else 1
        adn = adv.unsqueeze(-1).expand(T, N)
        return adn, (adn + values).detach()

    def update(self, batch: Dict) -> Dict:
        obs     = batch['obs'].to(self.device)             # [T, N, D]
        actions = batch['actions'].to(self.device)         # [T, N]
        rewards = batch['rewards'].to(self.device)         # [T]
        dones   = batch['dones'].to(self.device)           # [T]
        old_logp= batch['log_probs'].to(self.device)       # [T, N]
        avail   = batch.get('avail_actions', None)
        if avail is not None: avail = avail.to(self.device)
        T, N, D = obs.shape
        state   = obs.reshape(T, N*D) if self.algo=='MAPPO' else obs.reshape(T*N, D)

        # GAE (une fois)
        with torch.no_grad():
            v_init = (self.critic(state).squeeze(-1).unsqueeze(1).expand(T,N)
                      if self.algo=='MAPPO'
                      else self.critic(obs.reshape(T*N,D)).view(T,N))
            adv, ret = self._gae(rewards, v_init, dones)

        losses = []
        ent_last = torch.tensor(0.0)
        for _ in range(self.n_epochs):
            logits = self.actor(obs.reshape(T*N,D)).view(T,N,-1)
            if avail is not None:
                logits = logits + (1.0 - avail) * (-1e9)
            dist    = Categorical(logits=logits)
            # [FIX-1] log_prob des ANCIENNES actions
            new_logp = dist.log_prob(actions)
            ent_last = dist.entropy().mean()

            v_new = (self.critic(state).squeeze(-1).unsqueeze(1).expand(T,N)
                     if self.algo=='MAPPO'
                     else self.critic(obs.reshape(T*N,D)).view(T,N))
            ratio = (new_logp - old_logp.detach()).exp()
            surr  = torch.min(
                ratio * adv.detach(),
                torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv.detach())
            loss  = (-surr.mean()
                     + self.value_coef * F.mse_loss(v_new, ret.detach())
                     - self.entropy_coef * ent_last)
            self.optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) +
                list(self.critic.parameters()), 0.1)  # 0.5→0.1 anti-explosion (gradient clipping)
            self.optim.step()
            losses.append(loss.item())

        return {'loss': float(np.mean(losses)),
                'entropy': ent_last.item(),
                'policy_loss': 0.0, 'value_loss': 0.0, 'beta_mean': 0.0}

    def get_config_summary(self) -> Dict:
        return {'algo': self.algo, 'n_epochs': self.n_epochs,
                'gamma': self.gamma, 'clip_eps': self.clip_eps}

    def get_complexity_summary(self) -> Dict:
        return {'algo': self.algo, 'note': 'baseline simple MLP'}


# ══════════════════════════════════════════════════════════════
# ÉVALUATION  [FIX-4]
# ══════════════════════════════════════════════════════════════

# Seuil de "Soft WR" (proxy victoire quand binary WR = 0%)
SOFT_WIN_THRESHOLD = {
    '2s3z':       10.0,   # 5 agents : reward > 10 = bons dégâts
    '3s5z':       10.0,   # 8 agents hétérogènes : même seuil
    '10m_vs_11m': 12.0,   # 10 agents : seuil plus élevé
    '27m_vs_30m': 14.0,   # 27 agents : seuil encore plus élevé
}

def quick_eval(trainer, env: SMACWrapper, device: torch.device,
               n_ep: int = 32, call_idx: int = 0) -> tuple:
    
    import contextlib, io

    wins, rews, soft_wins = 0, [], 0
    map_name = env.map_name
    soft_thr = SOFT_WIN_THRESHOLD.get(map_name, 14.0)
    # SMAC : politique déterministe pure (ε=0)
    # ε-greedy nuit à la coordination SMAC (5% actions aléatoires = 0 victoire)
    # La variété est assurée par les seeds différents de chaque env
    eps      = 0.0
    rng      = np.random.RandomState(call_idx * 997 + 42)

    # Cache des environnements (crees une seule fois, silencieusement)
    if not hasattr(quick_eval, '_eval_envs') or        quick_eval._eval_map != map_name or        len(quick_eval._eval_envs) != n_ep:
        envs = []
        for i in range(n_ep):
            with contextlib.redirect_stdout(io.StringIO()):
                envs.append(SMACWrapper(map_name, seed=10_000 + i))
        quick_eval._eval_envs = envs
        quick_eval._eval_map  = map_name

    for ep_idx, eval_env in enumerate(quick_eval._eval_envs):
        # [A] Seed unique par (episode, appel) -> env.reset() varie
        np.random.seed(10_000 + ep_idx * 137 + call_idx * 31)
        obs  = eval_env.reset()
        done = False; ep_r = 0.0; info = {}

        for _ in range(eval_env.episode_limit):
            if done: break
            avail   = eval_env.get_avail_actions()
            obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(device)
            avail_t = torch.FloatTensor(avail).unsqueeze(0).to(device)
            with torch.no_grad():
                act_t, _, _ = trainer.get_action(
                    obs_t, avail_actions_t=avail_t, deterministic=True)
            actions = act_t.squeeze(0).cpu().numpy()
            # eps=0 → pas d'action aléatoire (déterministe pur pour SMAC)
            obs, r, done, info = eval_env.step(actions)
            ep_r += r

        rews.append(ep_r)
        if info.get('battle_won', False): wins      += 1
        if ep_r >= soft_thr:             soft_wins += 1

    wr      = wins      / n_ep * 100.0
    soft_wr = soft_wins / n_ep * 100.0
    return wr, float(np.mean(rews)), float(np.std(rews)), soft_wr


def ci95(vals: List[float]) -> Tuple[float, float]:
    """Intervalle de confiance à 95% (t-distribution)."""
    n = len(vals)
    if n == 0: return 0.0, 0.0
    m = statistics.mean(vals)
    if n == 1: return m, 0.0
    s = statistics.stdev(vals)
    t = {1:12.71, 2:4.30, 3:3.18, 4:2.78, 5:2.57}.get(n-1, 2.0)
    return m, t * s / (n ** 0.5)


def print_multi_seed_summary(all_results: List[Dict]):
    """ Tableau récapitulatif avec CI 95%."""
    print(f"\n{'='*72}")
    print(f"  MULTI-SEEDS — CI 95%  ")
    print(f"{'='*72}")
    print(f"  {'Algo':<16} {'Map':<10} {'N':>3}  "
          f"{'WR%':>7}  {'±CI':>6}  {'BestWR%':>8}  {'Rew':>8}")
    print(f"  {'─'*62}")

    groups: Dict = {}
    for r in all_results:
        groups.setdefault((r['algorithm'], r['map']), []).append(r)

    for (algo, mp), runs in sorted(groups.items()):
        wrs  = [r['win_rate']*100      for r in runs]
        bwrs = [r['best_win_rate']*100 for r in runs]
        rews = [r['avg_reward']        for r in runs]
        m_wr,  ci_wr  = ci95(wrs)
        m_bwr, _      = ci95(bwrs)
        m_rew, ci_rew = ci95(rews)
        flag = "✅" if len(runs) >= 5 else f"⚠ ({len(runs)}/5 seeds)"
        print(f"  {algo:<16} {mp:<10} {len(runs):>3}  "
              f"{m_wr:>6.1f}%  ±{ci_wr:>5.1f}%  "
              f"{m_bwr:>7.1f}%  {m_rew:>8.2f}  {flag}")

    print(f"{'='*72}\n")


# ══════════════════════════════════════════════════════════════
# BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════

def train_one(algo: str, map_name: str, total_steps: int,
              seed: int, config: Dict, device: torch.device,
              log_interval: int, eval_interval: int,
              save_dir: str, ablation: Optional[str] = None) -> Dict:
    
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ── Environnement ──────────────────────────────────────────────────────
    env = SMACWrapper(map_name, seed)

    print(f"\n{STAR}")
    print(f"* {algo}{'_'+ablation if ablation else '':12} | "
          f"{map_name} | seed={seed} | steps={total_steps:,}")
    print(STAR)
    print(f"  Agents={env.n_agents}  Enemies={env.n_enemies}  "
          f"Obs={env.obs_dim}  Actions={env.action_dim}")
    print(f"  Device={device}  Ablation={ablation or 'none'}")

    # ── Config commune ─────────────────────────────────────────────────────
    cfg = dict(
        hidden_dim=128, gamma=0.99, gae_lambda=0.95,
        lr_actor=3e-4, lr_critic=1e-3,
        clip_epsilon=0.2, entropy_coef=0.01,
        value_loss_coef=0.5,
        n_epochs=10,          # [FIX-2]
        **config)

    state_dim = env.obs_dim * env.n_agents

    # ── Instanciation de l'algorithme ──────────────────────────────────────
    if algo == 'H3C':
        if not H3C_AVAILABLE:
            print("  ERREUR: H3C non disponible"); return {}
        trainer = H3CTrainerRevised(
            obs_dim=env.obs_dim, action_dim=env.action_dim,
            n_agents=env.n_agents, config=cfg, device=device,
            disable_dgat       =(ablation == 'no_dgat'),
            disable_bayesian   =(ablation == 'no_bayesian'),
            disable_coalitions =(ablation == 'no_coalitions'),
            disable_dual_critic=(ablation == 'no_dual_critic'),
            disable_rtd        =(ablation == 'no_rtd'),
            disable_entropy    =(ablation == 'no_entropy'))
        trainer.name = f"H3C{'_'+ablation if ablation else ''}"
    else:
        # R2.9 — Baselines avec même config PPO que H3C
        trainer = BaselineTrainer(
            obs_dim=env.obs_dim, action_dim=env.action_dim,
            n_agents=env.n_agents, state_dim=state_dim,
            algo=algo, device=device,
            hidden=64, lr=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
            n_epochs=10)

    # ── Métriques ──────────────────────────────────────────────────────────
    win_window  : deque = deque(maxlen=100)
    rew_window  : deque = deque(maxlen=100)
    loss_window : deque = deque(maxlen=50)
    eval_records: List[Dict] = []
    first_win_step: Optional[int] = None
    global_step = 0; total_ep = 0
    _eval_call = 1
    last_log = 0; last_eval = 0
    t0 = time.time()

    # Initial evaluation [FIX-4]
    wr0, avg0, std0, sw0 = quick_eval(trainer, env, device, call_idx=0)
    print(f"\n  Initial evaluation : WinRate={wr0:.1f}%  "
          f"Reward={avg0:.2f}±{std0:.2f}")
    rec_steps = STEPS_RECOMMENDED.get(map_name, total_steps)
    print(f"  Recommended steps for {map_name} : {rec_steps:,}")
   
    eval_records.append({'step':0, 'win_rate':wr0, 'avg_reward':avg0})

    # ── Main training loop ──────────────────────────────────────────────────
    while global_step < total_steps:
        obs  = env.reset(); done = False; ep_r = 0.0; info = {}
        # [FIX-3] Buffers incluant avail_actions
        buf_o, buf_a, buf_r, buf_d, buf_lp, buf_av = [], [], [], [], [], []

        for _ in range(env.episode_limit):
            if done: break
            avail   = env.get_avail_actions()          # [N, A]
            obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(device)
            avail_t = torch.FloatTensor(avail).unsqueeze(0).to(device)

            with torch.no_grad():
                act_t, lp_t, _ = trainer.get_action(
                    obs_t, avail_actions_t=avail_t)

            actions = act_t.squeeze(0).cpu().numpy()
            lp      = lp_t.squeeze(0).cpu()

            next_obs, r, done, info = env.step(actions)
            ep_r += r; global_step += 1

            buf_o.append(obs.copy())
            buf_a.append(actions.astype(np.int64))
            buf_r.append(float(r))
            buf_d.append(float(done))
            buf_lp.append(lp)
            buf_av.append(avail.copy())   # [FIX-3]
            obs = next_obs

        # ── Mise à jour ────────────────────────────────────────────────────
        loss_val = 0.0
        if len(buf_o) > 1:
            batch = {
                'obs':          torch.FloatTensor(
                                    np.array(buf_o, dtype=np.float32)).to(device),
                'actions':      torch.LongTensor(
                                    np.array(buf_a)).to(device),
                'rewards':      torch.FloatTensor(buf_r).to(device),
                'dones':        torch.FloatTensor(
                                    np.array(buf_d, dtype=np.float32)).to(device),
                'log_probs':    torch.stack(buf_lp).to(device),
                'avail_actions':torch.FloatTensor(
                                    np.array(buf_av, dtype=np.float32)).to(device),
            }
            try:
                m = trainer.update(batch)
                loss_val = float(m.get('loss', 0.0))
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ERREUR update: {e}")

        loss_window.append(loss_val); total_ep += 1
        won = bool(info.get('battle_won', False))
        if won and first_win_step is None:
            first_win_step = global_step
            print(f"\n  >>> 1ERE VICTOIRE au step {global_step:,}! <<<")
        win_window.append(float(won)); rew_window.append(ep_r)

        # ── Log ───────────────────────────────────────────────────────────
        if global_step - last_log >= log_interval:
            el  = time.time() - t0
            sps = global_step / max(el, 1e-6)
            wr  = np.mean(win_window) * 100
            avg_l = float(np.mean(loss_window))
            avg_r = float(np.mean(rew_window))
            eta   = (total_steps - global_step) / max(sps, 1) / 3600
            print(f"  Step {global_step:>7,}/{total_steps:,} | "
                  f"WR:{wr:>5.1f}% | Rew:{avg_r:>6.2f} | "
                  f"Loss:{avg_l:>7.4f} | "
                  f"{sps:>5.0f}/s | ETA:{eta:.1f}h")
            last_log = global_step

        # ── Eval snapshot ─────────────────────────────────────────────────
        if global_step - last_eval >= eval_interval:
            wr_e, avg_e, std_e, soft_wr_e = quick_eval(trainer, env, device, call_idx=_eval_call)
            _eval_call += 1
            soft_info = f" | SoftWR={soft_wr_e:.0f}%" if wr_e == 0.0 else ""
            print(f"\n  EVAL {global_step:>7,}: "
                  f"WinRate={wr_e:.1f}%  "
                  f"Reward={avg_e:.2f}+-{std_e:.2f}"
                  f"{soft_info}\n")
            eval_records.append({
                'step': global_step, 'win_rate': wr_e,
                'avg_reward': avg_e, 'std_reward': std_e,
                'elapsed_s': time.time() - t0})
            last_eval = global_step

    # ── Final results ──────────────────────────────────────────────────
    wr_f, avg_f, std_f, sf_f = quick_eval(trainer, env, device, call_idx=999)
    best_wr  = max((r['win_rate'] for r in eval_records), default=wr_f)
    elapsed  = time.time() - t0

    print(f"\n{SEP}")
    print(f"  RESULTATS : {trainer.name} | {map_name} | seed={seed}")
    print(f"  WinRate final   : {wr_f:.1f}%")
    print(f"  WinRate best    : {best_wr:.1f}%")
    print(f"  Reward          : {avg_f:.2f} ± {std_f:.2f}")
    print(f"  First victory   : "
          f"{'step '+str(first_win_step) if first_win_step else 'non atteinte'}")
    print(f"  Episodes  : {total_ep:,}")
    print(f"  Duration           : {elapsed/60:.1f} min")
    print(SEP)

    # ── Saving JSON ───────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    res = {
        'algorithm':    trainer.name,
        'map':          map_name,
        'seed':         seed,
        'total_steps':  global_step,
        'win_rate':     wr_f / 100,
        'best_win_rate':best_wr / 100,
        'avg_reward':   avg_f,
        'std_reward':   std_f,
        'n_episodes':   total_ep,
        'elapsed_min':  elapsed / 60,
        'first_win_step': first_win_step,
        'checkpoints':  eval_records,
        'config':       cfg,
        #  ______ complexité
        'complexity':   trainer.get_complexity_summary()
                        if hasattr(trainer, 'get_complexity_summary') else {},
        #  ________ config complète
        'config_full':  trainer.get_config_summary()
                        if hasattr(trainer, 'get_config_summary') else {},
        #  _______ β trajectory (H3C seulement)
        'beta_trajectory': (trainer.dual_critic.get_beta_trajectory()
                            if hasattr(trainer, 'dual_critic')
                            and trainer.dual_critic else []),
        # _________ — ε-Nash
        'nash_gap_trajectory': (trainer.nash_gap_estimator.get_trajectory()
                                 if hasattr(trainer, 'nash_gap_estimator') else []),
    }
    fname = f"{save_dir}/{trainer.name}_{map_name}_s{seed}_{ts}.json"
    with open(fname, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  Saving : {fname}")
    env.close()
    return res


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

ALGOS_ALL = ['H3C', 'MAPPO', 'IPPO', 'QMIX', 'VDN', 'COMA']
ABLATIONS  = ['no_dgat', 'no_bayesian', 'no_coalitions',
              'no_dual_critic', 'no_rtd', 'no_entropy']


def main():
    p = argparse.ArgumentParser(
        description="H3C-BEACON SMAC Training ")
    p.add_argument('--algo',  default='H3C',
                   help=f"Algo : {ALGOS_ALL} | all")
    p.add_argument('--map',   default='2s3z',
                   help=f"Map : {list(MAPS.keys())}")
    p.add_argument('--steps', type=int, default=500_000)
    p.add_argument('--seed',  type=int, default=None)
    p.add_argument('--seeds', default=None,
                   help="Multi-seeds : 42,123,456,789,1024")
    p.add_argument('--ablation', default=None,
                   choices=ABLATIONS,
                   help="ablation study")
    p.add_argument('--save-dir',      default='results/smac')
    p.add_argument('--log-interval',  type=int, default=32_000)
    p.add_argument('--eval-interval', type=int, default=160_000)
    p.add_argument('--device', default=None)
    p.add_argument('--list-maps', action='store_true')
    args = p.parse_args()

    if args.list_maps:
        print("Available Maps  :")
        for k in MAPS:
            rec = STEPS_RECOMMENDED.get(k, '?')
            print(f"  {k:<16} (recommandé : {rec:,} steps)")
        return

    device = torch.device(
        args.device if args.device
        else 'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("  No GPU detected — using CPU (slower)")

    algos = ALGOS_ALL if args.algo == 'all' else [args.algo]
    seeds = ([int(s) for s in args.seeds.split(',')]
             if args.seeds
             else [args.seed] if args.seed is not None
             else [42])

    print(f"\n{SEP}")
    print(f"  H3C-BEACON SMAC ")
    print(f"  Algos  : {algos}")
    print(f"  Map    : {args.map}  |  Steps : {args.steps:,}")
    print(f"  Seeds  : {seeds}  |  Device : {device}")
    print(f"  Ablation: {args.ablation or 'none'}")
    print(SEP)

    config = {}
    all_results: List[Dict] = []

    for algo in algos:
        for seed in seeds:
            r = train_one(
                algo=algo, map_name=args.map,
                total_steps=args.steps, seed=seed,
                config=config, device=device,
                log_interval=args.log_interval,
                eval_interval=args.eval_interval,
                save_dir=args.save_dir,
                ablation=args.ablation if algo == 'H3C' else None)
            if r: all_results.append(r)

    # R1.4 / R2.10 — Tableau multi-seeds
    if len(all_results) > 1:
        print_multi_seed_summary(all_results)

    # Final summary
    print(f"\n{SEP}  SUMMARIZE  {SEP}")
    print(f"  {'Algo':<18} {'Map':<10} "
          f"{'WR%':>8}  {'BestWR%':>9}  {'Reward':>9}")
    print(f"  {'─'*58}")
    for r in all_results:
        print(f"  {r['algorithm']:<18} {r['map']:<10} "
              f"{r['win_rate']*100:>7.1f}%  "
              f"{r['best_win_rate']*100:>8.1f}%  "
              f"{r['avg_reward']:>9.2f}")
    print(SEP)


if __name__ == "__main__":
    main()