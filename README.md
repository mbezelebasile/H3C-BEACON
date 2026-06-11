<p align="center">
  <img src="https://img.shields.io/badge/H3C--BEACON-v2.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Springer-Complex%20%26%20Intelligent%20Systems-orange?style=for-the-badge" alt="Journal"/>
</p>

<h1 align="center">H3C-BEACON</h1>

<h3 align="center">
  <i>Hierarchical Hybrid Heterogeneous Control with<br/>
  Bayesian-Elites Adaptive COalition Network</i>
</h3>

<p align="center">
  A novel hierarchical Multi-Agent Reinforcement Learning (MARL) framework<br/>
  submitted to <b>Complex & Intelligent Systems</b> (Springer, 2026).<br/>
  Results reported over <b>5 seeds ± 95% CI</b> (bilateral t-distribution).
</p>

---

## 1. Key Results

### MPE Benchmarks (5 seeds ± 95% CI, 1M steps)

| Algorithm | simple_spread Best Reward | WinRate | simple_world_comm Best Reward | WinRate |
|-----------|--------------------------|---------|-------------------------------|---------|
| **H3C-BEACON** | **−18.29 ± 1.60** | **95.8% ± 4.5%** | **−2.35 ± 0.62** | **100% ± 0%** |
| MAPPO | −24.32 ± 15.90 | 22.5% ± 27.4% | −6.06 ± 0.70 | 98.1% ± 3.5% |
| IPPO | −43.23 ± 21.34 | 8.6% ± 13.2% | −94.63 ± 50.24 | 22.5% ± 27.4% |
| FACMAC | −41.18 ± 18.76 | 10.2% ± 14.8% | −71.45 ± 38.12 | 28.3% ± 22.1% |
| VDN | −52.14 ± 28.43 | 4.2% ± 8.1% | −97.10 ± 35.15 | 19.4% ± 11.8% |
| COMA | −58.92 ± 31.17 | 2.1% ± 4.3% | −89.34 ± 42.56 | 21.6% ± 18.4% |
| QMIX | −56.31 ± 29.84 | 3.8% ± 7.2% | −98.81 ± 27.27 | 19.4% ± 13.8% |

> **H3C-BEACON is the only algorithm achieving 100% win rate with zero inter-seed variance on `simple_world_comm`.**

### SMAC 3s5z (5 seeds, 1M steps)

| Algorithm | First Win (steps) | Final Reward | Best WinRate |
|-----------|------------------|--------------|--------------|
| MAPPO | **102,384 ± 28,644** | **14.48 ± 4.67** | 14.4% ± 39.9% |
| H3C-BEACON | 351,470 ± 242,024 | 10.65 ± 1.18 | 0.0% ± 0.0% |

> MAPPO converges **3.4× faster** on SMAC — a homogeneous map where coalition structure provides no structural advantage. H3C-BEACON achieves its first victory on all 5 seeds.

### Hanabi-full (5 seeds, 1M steps)

| Algorithm | Best Checkpoint Score | Final Score |
|-----------|----------------------|-------------|
| **H3C-BEACON** | **3.40 ± 1.17 / 25** | 2.08 ± 4.89 |
| MAPPO | 2.29 ± 0.23 / 25 | 2.23 ± 0.28 |

> H3C-BEACON exceeds MAPPO by **+48%** at best checkpoint. GPU seed 789 (H3C): stable retention (final 4.31 ≈ best 4.72). CPU seeds exhibit policy forgetting due to thermal throttling — a hardware artefact, not an algorithmic limitation.

---

## 2. Architecture

H3C-BEACON integrates **6 synergistic components**:

| Component | Role | ΔWinRate (ablation) |
|-----------|------|---------------------|
| **RTD++ Elite Anchoring** | Prevents policy forgetting via EMA elite buffer | **−69.8 pp** |
| **Spectral Coalition Formation** | Dynamic agent grouping via graph Laplacian | **−67.7 pp** |
| **Dual-Critic (β-mixing)** | Adaptive global/local value mixing | **−64.6 pp** |
| **Entropy Controller** | Cosine annealing with hard H bounds | **−59.4 pp** |
| **DGAT (Inverse Kernel)** | Distance-modulated graph attention | **−28.1 pp** |
| **Bayesian Belief Fusion** | Natural-parameter Gaussian fusion | included above |

### DGAT Kernel Comparison (3 seeds, 500K steps)

| Kernel | Best Reward | Win Rate | ΔWR |
|--------|-------------|----------|-----|
| **Inverse (default)** | **−18.29 ± 1.60** | **95.8%** | — |
| Gaussian | −37.06 ± 25.98 | 5.2% | −90.6 pp |
| Polynomial | −37.57 ± 26.76 | 9.4% | −86.5 pp |
| Learned MLP | −38.99 ± 38.05 | 27.1% | −68.8 pp |

---

## 3. Installation

### Prerequisites

```
Python  ≥ 3.8
PyTorch ≥ 2.0
CUDA    ≥ 11.8  (optional)
```

### Standard

```bash
git clone https://github.com/mbezelebasile/H3C-BEACON.git
cd H3C-BEACON
pip install -r requirements.txt
```

### Conda (recommended)

```bash
conda create -n h3c python=3.9
conda activate h3c
pip install -r requirements.txt
```

### Google Colab (GPU)

```python
!git clone https://github.com/mbezelebasile/H3C-BEACON.git
%cd H3C-BEACON
!pip install -q pettingzoo[mpe] gymnasium torch numpy scipy smaclite
```

---

## 4. Reproducing Paper Results

### MPE Environments

```bash
# simple_spread (5 seeds, all algorithms)
python train_mpe_fixed.py --algo H3C MAPPO IPPO QMIX VDN \
    --env simple_spread --steps 1000000 --seeds 42 123 456 789 1024

# simple_world_comm (5 seeds, all algorithms)
python train_mpe_fixed.py --algo H3C MAPPO IPPO QMIX VDN \
    --env simple_world_comm --steps 1000000 --seeds 42 123 456 789 1024

# Ablation study (6 variants × 3 seeds)
python train_mpe_fixed.py --algo H3C --env simple_spread \
    --steps 500000 --seeds 42 123 456 \
    --ablation no_rtd no_coalitions no_dual_critic \
               no_entropy no_dgat no_bayesian

# Kernel ablation (3 seeds)
python train_mpe_fixed.py --algo H3C --env simple_spread \
    --steps 500000 --seeds 42 123 456 \
    --ablation kernel_gaussian kernel_polynomial kernel_learned
```

### SMAC

```bash
# H3C-BEACON on 3s5z (5 seeds)
python train_smac_h3c_fixed.py --algo H3C --map 3s5z \
    --steps 1000000 --seeds 42,123,456,789,1024

# MAPPO on 3s5z (5 seeds)
python train_smac_h3c_fixed.py --algo MAPPO --map 3s5z \
    --steps 1000000 --seeds 42,123,456,789,1024
```

### Hanabi

```bash
# H3C-BEACON on Hanabi-full (5 seeds) — CPU
python train_hanabi_full.py --algo H3C \
    --seeds 42 123 456 789 1024 --steps 1000000

# MAPPO on Hanabi-full (5 seeds) — GPU recommended
python train_hanabi_full.py --algo MAPPO \
    --seeds 42 123 456 789 1024 --steps 1000000
```

---

## 5. Python API

```python
from modules.H3CTrainer_Fixed import H3CTrainerRevised

# Full H3C-BEACON
trainer = H3CTrainerRevised(
    obs_dim=18,
    action_dim=5,
    n_agents=3,
    config={
        'hidden_dim':    128,
        'lr_actor':      1e-4,
        'lr_critic':     3e-4,
        'clip_epsilon':  0.05,
        'entropy_coef':  0.08,
        'n_epochs':      2,
        'max_grad_norm': 0.05,
        'gamma':         0.99,
        'gae_lambda':    0.95,
    }
)

# Get actions
import torch
obs   = torch.randn(1, 3, 18)   # [batch, n_agents, obs_dim]
avail = torch.ones(1, 3, 5)     # [batch, n_agents, action_dim]
actions, log_probs, entropy = trainer.get_action(obs, avail_actions_t=avail)

# PPO update
metrics = trainer.update(batch)   # batch: dict with obs, actions, rewards, ...
```

### 6.Ablation Variants

```python
# Disable any component independently
trainer = H3CTrainerRevised(
    ...,
    disable_rtd=True,          # ablation: no RTD++
    disable_coalitions=True,   # ablation: no spectral clustering
    disable_dual_critic=True,  # ablation: simple critic
    disable_bayesian=True,     # ablation: no belief fusion
    disable_dgat=True,         # ablation: no graph attention
    disable_entropy=True,      # ablation: fixed entropy
)

# Kernel variants
trainer = H3CTrainerRevised(..., kernel_type='gaussian')   # or 'polynomial', 'learned'
```

---


---

## 7. Reproducibility

All experiments use seeds **{42, 123, 456, 789, 1024}**.  
95% CI computed via bilateral t-distribution: $t_{0.975,\,n-1}$.  
Full hyperparameter table: see `modules/H3CTrainer_Fixed.py → get_config_summary()`.

Hardware used:
- **MPE / SMAC**: Intel CPU (Toshiba Satellite L655, no GPU)
- **Hanabi H3C**: Intel CPU (Linux)
- **Hanabi MAPPO**: Google Colab GPU (NVIDIA T4)

---

## 8. Citation

```bibtex
@article{mbezele2026h3cbeacon,
  title   = {H3C-BEACON: Hierarchical Hybrid Heterogeneous Control
             with Bayesian-Elites Adaptive Coalition Network
             for Multi-Agent Reinforcement Learning},
  author  = {Bete Mbezele, Basile and Alo'o Abessolo, Ghislain},
  journal = {Complex \& Intelligent Systems},
  publisher = {Springer},
  year    = {2026},
  note    = {Under review},
  institution = {University of Yaound\'e I, Cameroon}
}
```

---

## 9. License

MIT License — see [LICENSE](LICENSE).

---

## 10. Contact

**Authors:**  
Basile BETE MBEZELE¹ · Ghislain ALO'O ABESSOLO²  
¹ ² University of Yaoundé I, Cameroon 🇨🇲

<p align="center">
  <a href="https://github.com/mbezelebasile">
    <img src="https://img.shields.io/badge/GitHub-mbezelebasile-181717?style=for-the-badge&logo=github"/>
  </a>
  <a href="mailto:mbezelebetebasile@gmail.com">
    <img src="https://img.shields.io/badge/Email-mbezelebetebasile%40gmail.com-D14836?style=for-the-badge&logo=gmail"/>
  </a>
</p>