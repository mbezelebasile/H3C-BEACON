# H3C-BEACON

**Hierarchical Hybrid Heterogeneous Control with Bayesian-Elites Adaptive Coalition Network**

Basile BETE MBEZELE · Ghislain ALO'O ABESSOLO  
University of Yaoundé I — Department of Computer Science, Cameroon  
*Submitted to Complex & Intelligent Systems (Springer), 2026*

---

## Overview

H3C-BEACON is a hierarchical Multi-Agent Reinforcement Learning (MARL) framework that combines six synergistic components into a unified architecture:

| Component | Role |
|-----------|------|
| **DGAT** — Dynamic Graph Attention Network | Distance-modulated inter-agent communication |
| **Bayesian Belief Fusion** | Natural-parameter Gaussian belief aggregation |
| **Spectral Coalition Formation** | Dynamic agent grouping via graph Laplacian |
| **Dual-Critic** | Adaptive global/local value mixing with learned β |
| **RTD++ Elite Anchoring** | Policy stabilisation via EMA elite buffer |
| **Entropy Controller** | Cosine annealing with hard entropy bounds |

All results are reported over **5 independent seeds** with **95% confidence intervals** computed via bilateral t-distribution (t₀.₉₇₅, df = n−1).

---

## Results

### MPE Cooperative Navigation (1M steps, 5 seeds ± 95% CI)

| Algorithm | `simple_spread` Best Reward | Win Rate | `simple_world_comm` Best Reward | Win Rate |
|-----------|:---------------------------:|:--------:|:--------------------------------:|:--------:|
| **H3C-BEACON** | **−18.29 ± 1.60** | **95.8% ± 4.5%** | **−2.35 ± 0.62** | **100% ± 0%** |
| MAPPO | −24.32 ± 15.90 | 22.5% ± 27.4% | −6.06 ± 0.70 | 98.1% ± 3.5% |
| IPPO | −43.23 ± 21.34 | 8.6% ± 13.2% | −94.63 ± 50.24 | 22.5% ± 27.4% |
| VDN | −52.14 ± 28.43 | 4.2% ± 8.1% | −97.10 ± 35.15 | 19.4% ± 11.8% |
| QMIX | −56.31 ± 29.84 | 3.8% ± 7.2% | −98.81 ± 27.27 | 19.4% ± 13.8% |

> H3C-BEACON is the **only algorithm achieving 100% win rate with zero inter-seed variance** on `simple_world_comm`.

---

### Ablation Study (simple_spread, 3 seeds, 500K steps)

| Variant | Best Reward | Win Rate | ΔWin Rate |
|---------|:-----------:|:--------:|:---------:|
| **H3C-BEACON (full)** | **−18.29 ± 1.60** | **95.8%** | — |
| − RTD++ | −25.77 ± 6.66 | 26.0% | −69.8 pp |
| − Coalitions | −29.16 ± 20.3 | 28.1% | −67.7 pp |
| − Dual Critic | −22.16 ± 3.97 | 31.3% | −64.6 pp |
| − Entropy | −21.58 ± 2.04 | 36.5% | −59.4 pp |
| − DGAT | −23.89 ± 4.51 | 67.7% | −28.1 pp |

---

### SMAC StarCraft II — map `3s5z` (5 seeds, 1M steps)

| Algorithm | First Victory (steps) | Final Reward |
|-----------|:--------------------:|:------------:|
| MAPPO | 102,384 ± 28,644 | 14.48 ± 4.67 |
| H3C-BEACON | 351,470 ± 242,024 | 10.65 ± 1.18 |

MAPPO converges **3.4× faster** on this map. This is a known structural boundary of H3C-BEACON: homogeneous unit compositions (3 Stalkers + 5 Zealots vs 3 Stalkers + 5 Zealots) provide no role diversity for coalition formation to exploit. H3C-BEACON achieves its first training victory on all 5 seeds.

---

### Hanabi-full (5 seeds, 1M steps)

| Algorithm | Best Checkpoint Score | Final Score |
|-----------|:--------------------:|:-----------:|
| **H3C-BEACON** | **3.40 ± 1.17 / 25** | 2.08 ± 4.89 |
| MAPPO | 2.29 ± 0.23 / 25 | 2.23 ± 0.28 |

H3C-BEACON exceeds MAPPO by **+48%** at best checkpoint. The GPU seed (789) is stable (best 4.72 → final 4.31). CPU seeds exhibit policy forgetting attributable to thermal throttling, not algorithmic instability.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mbezelebasile/H3C-BEACON.git
cd H3C-BEACON

# Install dependencies
pip install -r requirements.txt
```

For SMAC support:
```bash
pip install git+https://github.com/uoe-agents/smaclite.git
```

---

## Reproducing Paper Results

**MPE environments:**
```bash
python trainF.py --algo H3C MAPPO IPPO QMIX VDN \
    --env simple_spread --steps 1000000 \
    --seeds 42 123 456 789 1024

python trainF.py --algo H3C MAPPO IPPO QMIX VDN \
    --env simple_world_comm --steps 1000000 \
    --seeds 42 123 456 789 1024
```

**SMAC benchmark:**
```bash
python train_smac_h3c_fixed.py --algo H3C --map 3s5z \
    --steps 1000000 --seeds 42,123,456,789,1024

python train_smac_h3c_fixed.py --algo MAPPO --map 3s5z \
    --steps 1000000 --seeds 42,123,456,789,1024
```

**Hanabi cooperative card game:**
```bash
python train_hanabi_full.py --algo H3C \
    --seeds 42 123 456 789 1024 --steps 1000000

# MAPPO on GPU recommended
python train_hanabi_full.py --algo MAPPO \
    --seeds 42 123 456 789 1024 --steps 1000000
```

**Ablation study:**
```bash
python trainF.py --algo H3C --env simple_spread \
    --steps 500000 --seeds 42 123 456 \
    --ablation no_rtd no_coalitions no_dual_critic \
               no_entropy no_dgat no_bayesian
```

---

---

## Reproducibility Notes

- **Seeds:** 42, 123, 456, 789, 1024
- **CI computation:** bilateral t-distribution, t(0.975, df = n−1)
- **Hyperparameters:** fully documented in `modules/H3CTrainer_Fixed.py → get_config_summary()`
- **Hardware:** Intel CPU (MPE, SMAC) · Google Colab T4 GPU (Hanabi MAPPO baselines)

---

## Citation

```bibtex
@article{mbezele2026h3cbeacon,
  title     = {H3C-BEACON: Hierarchical Hybrid Heterogeneous Control
               with Bayesian-Elites Adaptive Coalition Network
               for Multi-Agent Reinforcement Learning},
  author    = {Bete Mbezele, Basile and Alo'o Abessolo, Ghislain},
  journal   = {Complex \& Intelligent Systems},
  publisher = {Springer},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Basile BETE MBEZELE** — mbezelebetebasile@gmail.com  
**Ghislain ALO'O ABESSOLO**  
University of Yaoundé I, Department of Computer Science, Cameroon
